#编排图节点（第2步·编排架构）
#【新增】父图的各个节点函数（trim_history / normal，报告分支的节点在 2.2/2.3 加入）：
#   trim_history —— 历史裁剪，保证"最近 N 轮"口径与改造前一致；
#   normal       —— 日常问答分支，命令式调用既有 ReactAgent（零改写复用）。
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.runtime import Runtime

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
    return {"messages": [RemoveMessage(id=m.id) for m in dropped]}


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
        result = react_agent.agent.invoke(
            {"messages": state["messages"]},
            #context 为空时给空字典兜底：内层中间件会直接 .get("report")，
            #传 None 会在"没有上下文的调用"（如自检/测试）里抛 AttributeError
            context=runtime.context or {},
        )
        return {"messages": [result["messages"][-1]]}

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
# ============================================================================================
