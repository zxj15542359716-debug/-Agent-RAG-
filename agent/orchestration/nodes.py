import json

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.orchestration.events import REPORT, emit, node_span
from agent.orchestration.report_schema import AfterSalesReport, output_contract, render_report_text
from agent.orchestration.researchers import SPECS, get_spec, run_researcher
from agent.orchestration.router import ROUTE_REPORT, classify_intent
from model.fectory import chat_model
from utils.Prompt_loader import load_report_synthesize_prompt
from utils.usage_ledger import scope
from utils.config_handler import orchestration_config
from utils.logger_handler import logger


def trim_history(state: dict, runtime: Runtime) -> dict:
    """把会话历史裁剪到最近 N 轮（config/orchestration.yml: checkpoint.max_turns，默认 10）。"""
    messages = state.get("messages") or []
    keep = int(orchestration_config["checkpoint"]["max_turns"])
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idx) <= keep:
        return {}
    cut = human_idx[len(human_idx) - keep]
    dropped = [m for m in messages[:cut] if getattr(m, "id", None)]
    if not dropped:
        return {}
    logger.info(f"[trim_history]历史共 {len(messages)} 条，裁掉最早 {len(dropped)} 条，保留最近 {keep} 轮")
    #只在真的裁了才发事件：每轮都发一条"什么都没做"的进度会让前端时间线闪
    with node_span("trim_history", "整理历史") as span:
        span["summary"] = f"裁掉最早 {len(dropped)} 条"
    return {"messages": [RemoveMessage(id=m.id) for m in dropped]}


def classify(state: dict, runtime: Runtime) -> dict:
    """意图路由节点（纯规则、零 LLM 调用）：写 route，并清空上一轮的报告残留。 """
    query = state.get("query") or ""
    with node_span("classify", "意图识别") as span:
        route = classify_intent(query)
        span["summary"] = "报告场景（并行调研）" if route == ROUTE_REPORT else "日常问答（快路径）"
    return {
        "route": route,
        "fault_finding": None,
        "warranty_finding": None,
        "history_finding": None,
        "report": None,
    }


def make_normal_node(react_agent):
    """构造日常问答节点：命令式调用既有 ReactAgent，只把最终回答写回父图历史。"""

    def normal(state: dict, runtime: Runtime) -> dict:
        #事件跨度覆盖整段问答（含工具调用），前端时间线里显示为"智能问答 · 3.2s"
        with scope("normal"), node_span("normal", "智能问答") as span:
            result = react_agent.agent.invoke(
                {"messages": state["messages"]},
                #context 为空时给空字典兜底：内层中间件会直接 .get("report")，
                #传 None 会在"没有上下文的调用"（如自检/测试）里抛 AttributeError
                context=runtime.context or {},
            )
            answer = result["messages"][-1]
            span["summary"] = f"回答 {len(str(answer.content))} 字"
        return {"messages": [answer]}

    return normal


# ==================== 报告分支（第2步·2.3）====================

def _finding_summary(kind: str, data: dict, status: str) -> str:
    """给时间线准备一句话结论（前端节点条下面显示的那行小字）"""
    if status == "error":
        return "该路调研失败（报告将跳过这部分）"
    if status == "degraded":
        return "未取得结构化结论，已降级为文本兜底"
    if kind == "fault":
        return f"故障类型：{data.get('fault_type') or '暂无法判定'}；根因 {len(data.get('root_causes') or [])} 条"
    if kind == "warranty":
        return f"保修状态：{data.get('warranty_status') or '暂无相关信息'}"
    return f"查询月份 {len(data.get('months') or [])} 个；记录 {len(data.get('usage_records') or [])} 条"


def researcher(state: dict, runtime: Runtime) -> dict:
    """并行子研究者节点：由 Send 派发（state 即 ResearcherInput：kind + query）。"""
    kind = state["kind"]
    spec = get_spec(kind)
    with scope(kind), node_span("researcher", spec.label, kind=kind, depth=1) as span:
        data, status = run_researcher(kind, state.get("query", ""), runtime.context)
        span["status"] = status
        span["summary"] = _finding_summary(kind, data, status)
    return {spec.state_key: {"status": status, "data": data}}


def _truncate_finding(finding: dict | None, chars: int) -> dict | None:
    """按字符预算压缩单路结论：字符串截断、列表限长（截断策略的第一个档位）"""
    if not finding:
        return None
    data = finding.get("data") or {}
    out = {"status": finding.get("status", "")}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = value[:chars]
        elif isinstance(value, list):
            out[key] = [str(x)[:chars] for x in value][:8]
        else:
            out[key] = value
    return out


def _build_payload(query: str, findings: dict, sources: list,
                   finding_chars: int, citation_chars: int, drop: tuple) -> str:
    """拼装合成提示词的输入（JSON 文本）：三路结论 + 检索来源 + 用户原问题"""
    packed = {}
    for kind, spec in SPECS.items():
        if kind in drop:
            continue
        packed[kind] = _truncate_finding(findings.get(spec.state_key), finding_chars)
    packed_sources = [
        {"n": s.get("n"), "source": s.get("source"), "entry": s.get("entry"),
         "title": s.get("title"), "snippet": (s.get("snippet") or "")[:citation_chars]}
        for s in (sources or [])
    ]
    return json.dumps({"user_query": query, "findings": packed, "sources": packed_sources},
                      ensure_ascii=False)


def _count_tokens(messages: list) -> int:
    """估算输入 token：优先用模型自带分词器（tiktoken），失败则按"中文约 2 字符/token"粗估"""
    try:
        return int(chat_model.get_num_tokens_from_messages(messages))
    except Exception:
        return sum(len(str(m.content)) for m in messages) // 2


def synthesize_report(finding_map: dict, sources: list, query: str) -> tuple[AfterSalesReport, str]:
    """三路结论 → 结构化报告；输入超预算时按档位截断重试（第2步的"token 超限保护"）。

    截断档位（每往下一档，信息密度最低的一路先被丢弃）：
        ① 原样（单路结论按 finding_max_chars 截断，引用片段 200 字）
        ② 单路结论减半 + 丢弃"历史工单"
        ③ 单路结论再减半 + 再丢弃"保修政策"（只留故障诊断，它最贴近用户问题）
    仍超预算就用最后一档硬上——再退一步是"不生成报告"，那比超预算更糟。
    """
    cfg = orchestration_config["report"]
    budget = int(cfg.get("max_input_tokens", 6000))
    base = int(cfg.get("finding_max_chars", 1200))
    attempts = max(0, int(cfg.get("max_truncate_attempts", 2)))
    levels = [
        (base, 200, ()),
        (max(base // 2, 200), 120, ("history",)),
        (max(base // 4, 200), 80, ("history", "warranty")),
    ][: attempts + 1]

    system_prompt = load_report_synthesize_prompt()
    #输出契约（字段名/类型）由 Pydantic 模型自动生成后附加在输入末尾：
    #实测不写清楚键名时，模型会自创字段（basic_information / fault_diagnosis），导致校验失败
    contract = "\n\n" + output_contract(AfterSalesReport)
    messages, note = [], ""
    for i, (fchars, cchars, drop) in enumerate(levels):
        payload = _build_payload(query, finding_map, sources, fchars, cchars, drop) + contract
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=payload)]
        tokens = _count_tokens(messages)
        if tokens <= budget:
            note = f"输入 {tokens} tokens（预算 {budget}）" + (f"，截断 {i} 次后通过" if i else "")
            break
        note = f"输入 {tokens} tokens 超预算 {budget}，已截断 {i + 1} 次"
        logger.warning(f"[synthesize]{note}（本档丢弃路段：{list(drop) or '无'}）")

    try:
        #【实测结论】DeepSeek 不支持 json_schema 形式的 response_format
        #（"This response_format type is unavailable now"），但支持 json_mode；
        # 合成本就要求"只输出 JSON"，所以显式用 json_mode 拿到 Pydantic 校验过的对象。
        report = chat_model.with_structured_output(AfterSalesReport, method="json_mode").invoke(messages)
        if not isinstance(report, AfterSalesReport):
            report = AfterSalesReport(**dict(report))
        return report, note
    except Exception as e:
        #降级：结构化合成失败（接口不支持 / JSON 解析失败）→ 退回纯文本合成，报告仍可读
        logger.error(f"[synthesize]结构化合成失败，降级为纯文本：{e}", exc_info=True)
        try:
            text = chat_model.invoke(messages).content
        except Exception as e2:
            logger.error(f"[synthesize]纯文本合成同样失败：{e2}", exc_info=True)
            text = "报告生成失败，请稍后重试或联系官方售后。"
        return AfterSalesReport(degraded_text=str(text)), note + "｜已降级为纯文本"


def synthesize(state: dict, runtime: Runtime) -> dict:
    """报告合成节点：三路结论 + 引用来源 → 结构化报告 → 同时下发文本与结构化事件。"""
    finding_map = {spec.state_key: state.get(spec.state_key) for spec in SPECS.values()}
    sources = (runtime.context or {}).get("sources") or []
    query = state.get("query") or ""
    with scope("synthesize"), node_span("synthesize", "生成报告") as span:
        report, note = synthesize_report(finding_map, sources, query)
        text = render_report_text(report)
        span["summary"] = note
        span["status"] = "degraded" if report.degraded_text else "ok"
        #文本走既有 text 事件（浏览器里照常可读）；结构化数据走 report 事件（前端渲染卡片）
        emit({"type": "text", "content": text})
        emit({"type": REPORT, "data": report.model_dump()})
    return {"report": report.model_dump(), "messages": [AIMessage(content=text)]}


# ============================================================================================
# 【第 2 步 · 2.1 说明】编排图节点（agent/orchestration/nodes.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增 trim_history（历史裁剪）与 make_normal_node（日常问答节点工厂）。
# 为什么裁剪做成"图的第一个节点"而不是 middleware：两条分支（日常问答/报告）都要先裁剪，
# 放在图入口一次搞定；做成 middleware 只能覆盖 model 节点所在的子 Agent，报告分支会漏。
# 为什么 normal 节点要命令式 invoke 而不是挂子图：见 make_normal_node 文档字符串。
# 与其它文件的关系：graph.py 装配这些节点；state.py 定义它们读写的通道；
# app.py 通过父图 stream 拿到逐字 token（已实测：ns=('normal:...',) 且 node=='model'）。
#
# 【第 2 步 · 2.2 追加说明】新增 classify 节点（调用 router.classify_intent），并给
# trim_history / normal 套上 events.node_span 事件跨度。classify 顺带清空上一轮的报告残留
# （三个 finding 通道与 report 是普通通道，不清零会把上轮结论一直挂在 state 里）。
# 事件只描述"发生了什么"，不参与任何业务判断——emit 失败也不影响回答（见 events.py 说明）。
#
# ============================================================================================
# 【第 2 步 · 2.3 说明】报告分支节点（researcher / synthesize）
# --------------------------------------------------------------------------------------------
# 改动点：新增 researcher 节点（Send 派发的三路并行调研）与 synthesize 节点（三路结论 →
# 结构化报告 → 同时下发文本与 report 事件）；配套新增 _finding_summary/_truncate_finding/
# _build_payload/_count_tokens/synthesize_report 五个辅助函数。
# 为什么研究者写"各自的通道"而不是一个列表：见 state.py 说明（并行写同一通道会冲突/累积）。
# 为什么合成要分档截断：三路结论 + 引用片段拼起来可能超过模型输入预算，截断档位按
# "信息密度"排序丢（history → warranty），最后只留最贴近用户问题的故障诊断；每次截断都
# 记日志与时间线摘要，便于事后判断"这份报告是不是被截过"。
# 为什么结构化输出用 json_mode 而不是默认方式：DeepSeek 实测不支持 json_schema
# （"This response_format type is unavailable now"），json_mode 可用；两者都由
# Pydantic 模型校验，失败则整体降级为纯文本合成（报告仍可读，卡片显示降级模式）。
# 与其它文件的关系：researchers.py 提供 run_researcher；report_schema.py 提供模型与渲染；
# graph.py 用 Send 扇出、用条件边分流；app.py 把 report 事件转给前端卡片。
# ============================================================================================
# ============================================================================================
