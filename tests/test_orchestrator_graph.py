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


def test_normal_route_is_recorded_and_answered_by_normal_agent():
    """日常问法：route=normal，且确实由既有 Agent（替身）作答"""
    calls = []

    class _Inner:
        def invoke(self, payload, context=None):
            from uuid import uuid4

            from langchain_core.messages import AIMessage

            calls.append(payload)
            return {"messages": [AIMessage(content="收到", id=f"t-{uuid4().hex}")]}

    class _Fake:
        agent = _Inner()

    from agent.orchestration.graph import build_orchestrator_graph

    graph = build_orchestrator_graph(_Fake(), checkpointer=InMemorySaver())
    graph.invoke({"messages": [{"role": "user", "content": "耳机没声音怎么办"}], "query": "耳机没声音怎么办"},
                 config={"configurable": {"thread_id": "u1:s1"}})
    state = graph.get_state({"configurable": {"thread_id": "u1:s1"}})
    assert state.values["route"] == "normal"
    assert calls, "日常问法应由既有 Agent 作答"


def test_report_path_fans_out_three_researchers_in_parallel(monkeypatch):
    """报告问法 → 三路子研究者"真并行" → 三路结论都进合成器。

    并行性用 threading.Barrier(3) 验证：三路必须同时到达屏障，否则超时抛错——
    这条断言能挡住"Send 写成了顺序边"这类静默退化（顺序执行时报告耗时会变成三倍）。
    """
    import threading

    import agent.orchestration.nodes as nodes_mod
    from agent.orchestration.report_schema import AfterSalesReport

    barrier = threading.Barrier(3, timeout=5)
    kinds_seen = []

    def fake_run_researcher(kind, query, context=None):
        kinds_seen.append(kind)
        barrier.wait()          #三路必须同时在跑，否则这里抛 BrokenBarrierError
        return {"fault_type": f"{kind}-结论"}, "ok"

    synth_calls = {}

    def fake_synthesize_report(finding_map, sources, query):
        synth_calls["findings"] = finding_map
        synth_calls["query"] = query
        return AfterSalesReport(fault_type="按键失灵"), "输入 10 tokens（预算 6000）"

    monkeypatch.setattr(nodes_mod, "run_researcher", fake_run_researcher)
    monkeypatch.setattr(nodes_mod, "synthesize_report", fake_synthesize_report)

    class _Inner:
        def invoke(self, payload, context=None):
            raise AssertionError("报告分支不应调用日常问答 Agent")

    class _Fake:
        agent = _Inner()

    from agent.orchestration.graph import build_orchestrator_graph

    graph = build_orchestrator_graph(_Fake(), checkpointer=InMemorySaver())
    graph.invoke({"messages": [{"role": "user", "content": "帮我生成一份售后报告"}],
                  "query": "帮我生成一份售后报告"},
                 config={"configurable": {"thread_id": "u1:s1"}})

    assert sorted(kinds_seen) == ["fault", "history", "warranty"]
    assert set(synth_calls["findings"]) == {"fault_finding", "warranty_finding", "history_finding"}
    assert all(f["status"] == "ok" for f in synth_calls["findings"].values())
    assert synth_calls["query"] == "帮我生成一份售后报告"
    state = graph.get_state({"configurable": {"thread_id": "u1:s1"}})
    assert state.values["report"]["fault_type"] == "按键失灵"
    #报告正文写回会话历史（下一轮对话能看到上一份报告）
    assert "售后服务报告" in state.values["messages"][-1].content


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_orchestrator_graph.py
