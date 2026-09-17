#编排图装配（第2步·编排架构）
#【新增】把"单 ReAct"收成一条带分支的编排图：
#   START -> trim_history -> classify ─┬─(normal)─> normal ──────────────────────> END
#                                      └─(report)─> researcher ×3（并行）-> synthesize -> END
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.orchestration.checkpoint import get_checkpointer
from agent.orchestration.nodes import classify, make_normal_node, researcher, synthesize, trim_history
from agent.orchestration.router import ROUTE_REPORT
from agent.orchestration.state import RESEARCHER_KINDS, OrchestratorState, ResearcherInput
from agent.react_agent import ReactAgent
from utils.logger_handler import logger


def route_after_classify(state: dict):
    """分类后的分支选择：报告场景 → 三路 Send 并行；其余 → 日常问答快路径。

    Send("researcher", payload) 是一等公民的"扇出"：payload 即该节点的 state（配 input_schema），
    三路会跑在 LangGraph 的线程池里真并行，全部结束后才进入 synthesize。
    """
    if state.get("route") == ROUTE_REPORT:
        return [Send("researcher", {"kind": kind, "query": state.get("query", "")})
                for kind in RESEARCHER_KINDS]
    return "normal"


def build_orchestrator_graph(react_agent: ReactAgent, checkpointer=None):
    """装配编排图。checkpointer 传 None 时用全局 SqliteSaver（测试可传 InMemorySaver/临时库）。"""
    graph = StateGraph(OrchestratorState)
    graph.add_node("trim_history", trim_history)
    graph.add_node("classify", classify)
    graph.add_node("normal", make_normal_node(react_agent))
    #研究者的 state 只吃 {kind, query}（input_schema 收窄），不携带会话历史——
    #三路调研是独立的，历史只会稀释提示词
    graph.add_node("researcher", researcher, input_schema=ResearcherInput)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "trim_history")
    graph.add_edge("trim_history", "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["normal", "researcher"])
    graph.add_edge("normal", END)
    graph.add_edge("researcher", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else get_checkpointer())


class OrchestratorAgent:
    """编排图入口（对外与既有 ReactAgent 同形：app.py 用 self.graph.stream 驱动）。

    复用同一个既有 ReactAgent 实例作为日常问答分支——不改它的工具/提示词/中间件，
    保证已评测过的问答链路行为零变化。
    """

    def __init__(self, checkpointer=None):
        self.normal_agent = ReactAgent()
        self.graph = build_orchestrator_graph(self.normal_agent, checkpointer)
        logger.info("[orchestration]编排图构建完成（trim_history -> normal）")


if __name__ == "__main__":
    #自检（不联网、不落库）：打印图结构，确认节点与边装配正确
    #运行：cd 项目根 && .venv/Scripts/python.exe -m agent.orchestration.graph
    #（会有一条 sys.modules 的 RuntimeWarning：包 __init__ 已导入本模块，属 -m 运行方式所致，可忽略）
    from langgraph.checkpoint.memory import InMemorySaver

    agent = OrchestratorAgent(checkpointer=InMemorySaver())
    drawable = agent.graph.get_graph()
    print("=" * 62)
    print("编排图节点：")
    for node in drawable.nodes:
        print(f"  - {node}")
    print("编排图边：")
    for edge in drawable.edges:
        print(f"  - {edge.source} -> {edge.target}")
    print("=" * 62)


# ============================================================================================
# 【第 2 步 · 2.1 说明】编排图装配（agent/orchestration/graph.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增 build_orchestrator_graph() 与 OrchestratorAgent——一张图承载日常问答分支，
# 并挂上 SqliteSaver（会话持久化）。
# 为什么"一张图两条分支"而不是"日常一张图、报告一张图"：checkpointer 按
# (thread_id, checkpoint_ns) 存 channel 版本，同一会话若先后走两张拓扑不同的图，
# 会因 channel 对不上而报错或丢历史。一张图、两条分支是唯一不需要额外兼容层的形态。
# 为什么复用既有 ReactAgent 实例：它的工具、系统提示词、三个中间件都已过第 1 步评测，
# 任何重写都会让既有行为重新变成"未验证状态"；这里只把它的编译产物当函数调用。
# 与其它文件的关系：checkpoint.py 提供 saver，nodes.py 提供节点，state.py 提供通道；
# app.py 用 OrchestratorAgent().graph 替代原来的 ReactAgent().agent 作为流式入口。
#
# 【第 2 步 · 2.2 追加说明】图在 trim_history 之后插入 classify 节点（纯规则意图路由）。
# 本子项只把"路由决策"落地并记录（state.route + 时间线事件），报告分支的边在 2.3 接入——
# 之所以分开提交：路由是可独立测试的纯逻辑（tests/test_router.py 覆盖典型问法），
# 先把它锁死，2.3 接并行分支时只需改一条边，不必同时怀疑"是路由错了还是分支错了"。
#
# 【第 2 步 · 2.3 追加说明】报告分支接入：classify 之后由条件边按 state.route 分流，
# 报告场景用 Send 扇出三路 researcher（真并行），三路全部结束后进 synthesize，
# 最后 synthesize -> END。日常问答分支完全不变（normal 节点仍是同一个既有 Agent）。
# ============================================================================================
