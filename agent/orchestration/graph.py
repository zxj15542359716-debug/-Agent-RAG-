#编排图装配（第2步·编排架构）
#【新增】把"单 ReAct"收成一条带分支的编排图：
#   START -> trim_history -> classify -> normal -> END   （2.2：路由已就位，报告分支 2.3 接入）
from langgraph.graph import END, START, StateGraph

from agent.orchestration.checkpoint import get_checkpointer
from agent.orchestration.nodes import classify, make_normal_node, trim_history
from agent.orchestration.state import OrchestratorState
from agent.react_agent import ReactAgent
from utils.logger_handler import logger


def build_orchestrator_graph(react_agent: ReactAgent, checkpointer=None):
    """装配编排图。checkpointer 传 None 时用全局 SqliteSaver（测试可传 InMemorySaver/临时库）。"""
    graph = StateGraph(OrchestratorState)
    graph.add_node("trim_history", trim_history)
    graph.add_node("classify", classify)
    graph.add_node("normal", make_normal_node(react_agent))
    graph.add_edge(START, "trim_history")
    graph.add_edge("trim_history", "classify")
    #【2.2 临时直连】路由结果已写入 state.route（可观测、可测试），但报告分支尚未接入，
    #两条路由目前都先走日常问答；2.3 会用条件边把 route=="report" 接到并行子研究者
    graph.add_edge("classify", "normal")
    graph.add_edge("normal", END)
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
# ============================================================================================
