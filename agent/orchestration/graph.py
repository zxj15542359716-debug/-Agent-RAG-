from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.orchestration.checkpoint import get_checkpointer
from agent.orchestration.nodes import classify, make_normal_node, researcher, synthesize, trim_history
from agent.orchestration.router import ROUTE_REPORT
from agent.orchestration.state import RESEARCHER_KINDS, OrchestratorState, ResearcherInput
from agent.react_agent import ReactAgent
from utils.logger_handler import logger


def route_after_classify(state: dict):
    if state.get("route") == ROUTE_REPORT:
        return [Send("researcher", {"kind": kind, "query": state.get("query", "")})
                for kind in RESEARCHER_KINDS]
    return "normal"


def build_orchestrator_graph(react_agent: ReactAgent, checkpointer=None):
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

    def __init__(self, checkpointer=None):
        self.normal_agent = ReactAgent()
        self.graph = build_orchestrator_graph(self.normal_agent, checkpointer)
        logger.info("[orchestration]编排图构建完成（trim_history -> normal）")


if __name__ == "__main__":
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