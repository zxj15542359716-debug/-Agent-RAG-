#并行子研究者（第2步·2.3）
#【新增】三条并行的"调研线"，每条都是一个独立编译的 create_agent（各自的工具 + 提示词 + 输出结构）：
#   fault    故障诊断：rag_summarize（知识库检索 → 故障类型/根因/步骤/安全提醒）
#   warranty 保修政策：query_warranty + rag_summarize（真实保修记录 + 政策条款）
#   history  历史工单：fetch_external_data + get_current_month（使用与维修记录）
#三者共用约定：结构化输出（response_format）、不挂 checkpointer（研究者无独立会话状态）、
#模型调用次数上限（防结构化输出在模型节点里反复重试）。
import threading
import time
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from pydantic import BaseModel

from agent.orchestration.report_schema import (
    FaultFinding,
    HistoryFinding,
    WarrantyFinding,
    output_contract,
    parse_json_object,
)
from agent.tools.agent_tools import fetch_external_data, get_current_month, query_warranty, rag_summarize
from model.fectory import chat_model
from utils.Prompt_loader import load_researcher_prompts
from utils.config_handler import orchestration_config
from utils.logger_handler import logger


@dataclass(frozen=True)
class ResearcherSpec:
    kind: str            # fault | warranty | history
    label: str           # 中文名（时间线展示用）
    tools: tuple         # 该研究者可用的工具
    schema: type[BaseModel]   # 结构化输出模型
    state_key: str       # 结论写回父图的通道名


SPECS: dict[str, ResearcherSpec] = {
    "fault": ResearcherSpec("fault", "故障诊断", (rag_summarize,), FaultFinding, "fault_finding"),
    "warranty": ResearcherSpec("warranty", "保修政策", (query_warranty, rag_summarize), WarrantyFinding, "warranty_finding"),
    "history": ResearcherSpec("history", "历史工单", (fetch_external_data, get_current_month), HistoryFinding, "history_finding"),
}

_researchers: dict[str, Any] = {}
_build_lock = threading.Lock()
#并发闸门：config 里的 parallel_limit 限制同时打给模型的研究者数量（默认 3 = 三路齐发）
_semaphore = threading.BoundedSemaphore(max(1, int(orchestration_config["report"].get("parallel_limit", 3))))


def get_researcher(kind: str):
    """惰性构建并缓存某个研究者（双重检查锁，与 agent_tools._get_rag_service 同款）。

    为什么惰性：三份 create_agent 构建有成本，而报告场景是少数路径——日常问答不该为它买单。
    """
    agent = _researchers.get(kind)
    if agent is None:
        with _build_lock:
            agent = _researchers.get(kind)
            if agent is None:
                spec = SPECS[kind]
                agent = create_agent(
                    model=chat_model,
                    system_prompt=load_researcher_prompts(kind),
                    tools=list(spec.tools),
                    #【为什么不用 response_format=spec.schema】实测 DeepSeek 思考模式拒绝
                    # 强制 tool_choice（create_agent 的 ToolStrategy 正是强制调用指定函数）：
                    # "Thinking mode does not support this tool_choice"。
                    # 因此结构化输出改为"提示词约定 JSON + 解析校验"，见 _user_message 与
                    # report_schema.parse_json_object 的说明。
                    #模型调用上限：调研阶段可能在模型节点里反复重试，这里兜底掐断
                    middleware=[ModelCallLimitMiddleware(
                        run_limit=int(orchestration_config["report"].get("researcher_max_calls", 6)),
                        exit_behavior="end")],
                    checkpointer=False,
                )
                _researchers[kind] = agent
                logger.info(f"[researcher]构建完成：{spec.label}（工具 {[t.name for t in spec.tools]}）")
    return agent


def _user_message(spec: ResearcherSpec, query: str) -> str:
    """研究者的人话输入 + JSON 输出契约（字段清单由 Pydantic 模型自动生成）"""
    return (
        f"用户问题：{query}\n\n"
        "登录用户身份由系统自动注入（相关工具会使用当前登录用户的数据），无需向用户索要用户ID。\n"
        "请按你的职责完成调研（需要时调用工具检索/查询），工具调用不超过 2 次，"
        "拿到关键信息就尽快输出结论。\n\n"
        f"{output_contract(spec.schema)}"
    )


def _last_ai_text(result: dict) -> str:
    for msg in reversed(result.get("messages") or []):
        if getattr(msg, "content", "") and msg.__class__.__name__.startswith("AI"):
            return str(msg.content)
    return ""


def run_researcher(kind: str, query: str, context: dict | None = None) -> tuple[dict, str]:
    """执行一个研究者：返回 (结论数据, 状态)。status ∈ ok / degraded / error。

    降级链：结构化输出缺失 → 用最后一条 AI 消息文本兜底（degraded）；整段失败 → 空结论（error）。
    报告合成阶段对缺失/降级的一路会跳过它的字段，不会因此编造内容。
    """
    spec = SPECS[kind]
    agent = get_researcher(kind)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            #并发闸门：限制同时打给模型的研究者数（429 风险控制，见 config parallel_limit）
            with _semaphore:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": _user_message(spec, query)}]},
                    context=context or {},
                )
            text = _last_ai_text(result)
            try:
                #结构化结论：解析模型输出的 JSON 并用 Pydantic 校验（字段缺失/类型不符会被兜住）
                data = spec.schema.model_validate(parse_json_object(text)).model_dump()
                return data, "ok"
            except Exception as parse_err:
                logger.warning(f"[researcher]{spec.label} 结构化解析失败（{parse_err}），降级为文本兜底")
                return {"summary": text[:500]}, "degraded"
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(k in msg for k in ("rate limit", "429", "timeout", "timed out", "connection"))
            logger.warning(f"[researcher]{spec.label} 第 {attempt + 1} 次执行失败：{e}")
            if not retryable or attempt == 2:
                break
            time.sleep(1.5 * (attempt + 1))    #退避重试：并行打模型时更易触发限流
    logger.error(f"[researcher]{spec.label} 执行失败：{last_err}", exc_info=last_err is not None)
    return {}, "error"


def get_spec(kind: str) -> ResearcherSpec:
    return SPECS[kind]


# ============================================================================================
# 【第 2 步 · 2.3 说明】并行子研究者（agent/orchestration/researchers.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增三个研究者的构建（惰性 + 缓存 + 锁）与执行封装 run_researcher。
# 为什么每个研究者是"独立编译的 create_agent"而不是一个大 Agent 带全部工具：
# 工具面越窄越不容易跑偏（故障诊断不该去查保修记录），且三者的输出结构本就不同；
# create_agent 支持建多个互不影响的实例（工具/提示词/输出结构各自独立）。
# 为什么要并发闸门与重试：三路并行会同时打同一个模型接口，是限流（429）的高发点；
# parallel_limit 控制并发数、失败按指数退避重试（只对限流/超时/连接类错误重试，
# 结构化解析这类确定性错误重试没意义，直接降级）。
# 边界与兜底：结构化输出取不到 → 用最后一条 AI 文本降级（status=degraded）；
# 整段失败 → 空结论（status=error）。两种情况下报告仍会生成，只是那一路标缺失。
#
# 【实测记录】本子项最初用 create_agent(response_format=FaultFinding) 做结构化输出，
# 实测被 DeepSeek 拒绝："Thinking mode does not support this tool_choice"——
# 该参数走的是 ToolStrategy（强制调用指定函数），而思考模式不允许强制 tool_choice；
# 换 json_schema 形式的 response_format 同样不可用。因此改为：
#   提示词给出 JSON 字段契约（schema_hint 由 Pydantic 模型自动生成）
#   → 模型输出 JSON → report_schema.parse_json_object 容错解析 → Pydantic 校验。
# 这条路不依赖任何厂商特有的结构化输出能力，且解析失败会自动降级而不是报错。
# 与其它文件的关系：nodes.researcher 调用 run_researcher 并把结论写回父图通道；
# report_schema.py 提供结论模型；graph.py 用 Send 把三路并行派发出去。
# ============================================================================================
