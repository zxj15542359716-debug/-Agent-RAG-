#编排图节点（第2步·编排架构）
#【新增】父图的各个节点函数（报告分支的节点在 2.3 加入）：
#   trim_history —— 历史裁剪，保证"最近 N 轮"口径与改造前一致；
#   classify     —— 意图路由（纯规则），决定走日常问答还是报告分支；
#   normal       —— 日常问答分支，命令式调用既有 ReactAgent（零改写复用）。
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.runtime import Runtime

from agent.orchestration.events import node_span
from agent.orchestration.router import ROUTE_REPORT, classify_intent
from utils.config_handler import orchestration_config
from utils.logger_handler import logger


def trim_history(state: dict, runtime: Runtime) -> dict:
    """把会话历史裁剪到最近 N 轮（config/orchestration.yml: checkpoint.max_turns，默认 10）。

    【关键约束】只在"用户消息边界"下刀：找到倒数第 N 条 HumanMessage，从它之前截断。
    绝不能按条数硬切——切在对话中途会留下"孤儿 ToolMessage"（前面那条带 tool_calls 的
    AIMessage 被删了），下一次请求会被 DeepSeek 以 400 拒绝。这是本设计最容易踩的坑。

    无历史可裁时返回空字典（LangGraph 约定：不更新任何通道），避免每轮都产生一次无意义的
    checkpoint 写入。
    """
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
    """意图路由节点（纯规则、零 LLM 调用）：写 route，并清空上一轮的报告残留。

    清残留的必要性：三个 finding 通道与 report 是"普通通道"（每轮覆盖），若不清零，
    本轮若只走日常分支，上一轮报告的结构化结果会一直挂在 state 里（运行历史/调试时误导）。
    """
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
    """构造日常问答节点：命令式调用既有 ReactAgent，只把最终回答写回父图历史。

    为什么不直接把编译好的 Agent 用 add_node() 挂成子图：那样内层 Agent 的工具调用中间
    消息（AIMessage.tool_calls / ToolMessage）会一并合并进父图 messages，与改造前
    "只存 user + assistant 文本"的口径不一致——历史会迅速膨胀、10 轮裁剪也会提前触发。
    实测两种接法的流式表现一致（token 都逐字流出、langgraph_node 都是 model），
    因此选择可控性更好的命令式写法。

    context 必须显式转发：登录用户ID（越权防护）与来源收集容器都挂在 runtime.context 上，
    不转发会导致工具拿不到 current_user、引用溯源同时失效。
    """

    def normal(state: dict, runtime: Runtime) -> dict:
        #事件跨度覆盖整段问答（含工具调用），前端时间线里显示为"智能问答 · 3.2s"
        with node_span("normal", "智能问答") as span:
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
# ============================================================================================
