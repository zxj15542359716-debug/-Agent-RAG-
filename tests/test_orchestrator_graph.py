#编排图装配测试（第2步·2.2）
#覆盖：日常问答分支确实复用既有 ReactAgent（工具/中间件原样在册，含报告兜底工具
#      fill_context_for_report）；编排图把路由结果写进 state.route；两条路由当前都先走
#      日常问答（报告分支 2.3 接入，这里锁住"未接入前不会误吞问题"）。
#离线：只构建 Agent（不发请求、不加载检索索引）。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver

from agent.orchestration.graph import OrchestratorAgent
from agent.react_agent import ReactAgent

EXPECTED_TOOLS = {
    "rag_summarize", "fetch_external_data", "query_warranty",
    "get_user_id", "get_current_month", "fill_context_for_report",
}


def _orchestrator():
    return OrchestratorAgent(checkpointer=InMemorySaver())


def _tool_names(agent):
    """取出既有 Agent 的工具注册表（编译后的节点被 PregelNode 包了一层，
    真正的 ToolNode 挂在 .bound 上——实测确认，别按直觉去取 .tools_by_name）"""
    node = agent.nodes["tools"]
    tool_node = getattr(node, "bound", node)
    return set(getattr(tool_node, "tools_by_name", {}).keys())


def test_normal_path_reuses_existing_react_agent():
    orch = _orchestrator()
    assert isinstance(orch.normal_agent, ReactAgent)
    node_names = set(orch.normal_agent.agent.get_graph().nodes)
    #既有 Agent 的节点结构原样保留（模型节点 + 工具节点 + 请求前中间件节点）
    assert {"model", "tools", "log_before_model.before_model"} <= node_names


def test_all_tools_still_registered():
    orch = _orchestrator()
    names = _tool_names(orch.normal_agent.agent)
    assert names == EXPECTED_TOOLS
    #报告兜底路径仍在：规则漏判的报告类问题靠这个工具切提示词（行为与改造前一致）
    assert "fill_context_for_report" in names


def test_orchestrator_graph_contains_route_node():
    orch = _orchestrator()
    node_names = set(orch.graph.get_graph().nodes)
    assert {"trim_history", "classify", "normal"} <= node_names


def test_route_is_recorded_in_state():
    """2.2 阶段：报告问法会被标记为 route=report（分支 2.3 才接入，当前仍由日常问答作答）"""
    fake_calls = []

    class _Inner:
        def invoke(self, payload, context=None):
            from uuid import uuid4
            from langchain_core.messages import AIMessage

            fake_calls.append(payload)
            return {"messages": [AIMessage(content="收到", id=f"t-{uuid4().hex}")]}

    class _Fake:
        agent = _Inner()

    from agent.orchestration.graph import build_orchestrator_graph

    graph = build_orchestrator_graph(_Fake(), checkpointer=InMemorySaver())
    graph.invoke({"messages": [{"role": "user", "content": "帮我生成一份售后报告"}], "query": "帮我生成一份售后报告"},
                 config={"configurable": {"thread_id": "u1:s1"}})
    state = graph.get_state({"configurable": {"thread_id": "u1:s1"}})
    assert state.values["route"] == "report"
    assert fake_calls, "报告问法在 2.2 阶段仍应由日常问答作答（分支未接入）"


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_orchestrator_graph.py
