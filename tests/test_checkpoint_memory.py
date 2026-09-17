#会话持久化测试（第2步·2.1）
#覆盖：历史裁剪只在用户消息边界下刀 / 无超限时不动历史 / checkpoint 跨实例读回（模拟重启）/
#      不同 thread 互不可见（隔离性）/ 日常问答节点只把最终回答写回父图历史。
#全部离线：既有 Agent 用假对象替身（只回一条带唯一 id 的 AI 消息），不加载索引、不调模型。
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.orchestration.checkpoint import make_thread_id
from agent.orchestration.graph import build_orchestrator_graph
from agent.orchestration.nodes import trim_history


class _FakeInner:
    """假的既有 Agent：记录收到的输入与上下文，回一条唯一 id 的回答。"""

    def __init__(self):
        self.n = 0
        self.last_input = None
        self.last_context = None

    def invoke(self, payload, context=None):
        self.n += 1
        self.last_input = payload
        self.last_context = context
        #id 必须全局唯一：add_messages 按 id 去重，两份假对象若都从 fake-ai-1 开始编号，
        #后一条回答会顶掉前一条（这正是改造中实测到的 add_messages 语义）
        return {"messages": [AIMessage(content=f"第{self.n}次回答", id=f"fake-ai-{uuid4().hex}")]}


class _FakeReactAgent:
    """替身：形状与 ReactAgent 一致（有 .agent.invoke）"""

    def __init__(self):
        self.agent = _FakeInner()


def _open_saver(db_path):
    """按 checkpoint.py 同款参数打开一个 SqliteSaver（测试用临时库）"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def _turns(n, start=0):
    """构造 n 轮（user+assistant）消息"""
    msgs = []
    for i in range(start, start + n):
        msgs.append(HumanMessage(content=f"问题{i}", id=f"h-{i}"))
        msgs.append(AIMessage(content=f"回答{i}", id=f"a-{i}"))
    return msgs


# ------------------------------ 裁剪逻辑 ------------------------------
def test_trim_history_keeps_recent_turns_at_user_boundary():
    state = {"messages": _turns(12)}
    out = trim_history(state, None)
    removed_ids = [m.id for m in out["messages"]]
    #12 轮保留最近 10 轮 → 裁掉最早 2 轮（4 条）
    assert removed_ids == ["h-0", "a-0", "h-1", "a-1"]
    #裁剪点必须落在用户消息上（保留的第一条应是 HumanMessage），否则会产生孤儿 ToolMessage
    kept = [m for m in state["messages"] if m.id not in removed_ids]
    assert isinstance(kept[0], HumanMessage)


def test_trim_history_noop_when_within_limit():
    assert trim_history({"messages": _turns(10)}, None) == {}
    assert trim_history({"messages": _turns(3)}, None) == {}
    assert trim_history({"messages": []}, None) == {}


# ------------------------------ 图与持久化 ------------------------------
def test_normal_node_writes_only_final_answer(tmp_path):
    fake = _FakeReactAgent()
    graph = build_orchestrator_graph(fake, checkpointer=_open_saver(tmp_path / "ck.db"))
    graph.invoke({"messages": [{"role": "user", "content": "键盘失灵"}], "query": "键盘失灵"},
                 config={"configurable": {"thread_id": "u1:s1"}},
                 context={"report": False, "current_user": "2483", "sources": []})
    state = graph.get_state({"configurable": {"thread_id": "u1:s1"}})
    types = [type(m).__name__ for m in state.values["messages"]]
    #父图只留 user + assistant（口径同改造前的会话记忆），工具消息不落历史
    assert types == ["HumanMessage", "AIMessage"]
    #运行时上下文原样转发给既有 Agent：登录用户ID（越权防护）与来源容器都挂在这，
    #不转发会让工具拿不到 current_user、引用溯源同时失效
    assert fake.agent.last_context["current_user"] == "2483"


def test_history_survives_new_saver_instance(tmp_path):
    db = tmp_path / "ck.db"
    fake = _FakeReactAgent()
    graph1 = build_orchestrator_graph(fake, checkpointer=_open_saver(db))
    for i in range(2):
        graph1.invoke({"messages": [{"role": "user", "content": f"第{i}问"}], "query": f"第{i}问"},
                      config={"configurable": {"thread_id": "u1:s1"}})

    #换一个 saver 实例读同一个库 = 模拟进程重启
    fake2 = _FakeReactAgent()
    graph2 = build_orchestrator_graph(fake2, checkpointer=_open_saver(db))
    graph2.invoke({"messages": [{"role": "user", "content": "那键盘呢"}], "query": "那键盘呢"},
                  config={"configurable": {"thread_id": "u1:s1"}})
    state = graph2.get_state({"configurable": {"thread_id": "u1:s1"}})
    #前两轮由 graph1/fake1 产生（回答计数 1、2），第三轮由 graph2/fake2 产生（计数从 1 重新开始）
    assert [m.content for m in state.values["messages"]] == [
        "第0问", "第1次回答", "第1问", "第2次回答", "那键盘呢", "第1次回答"]
    #重启后本轮提问时，本轮之前的历史确实被送进了既有 Agent（多轮衔接的前提）
    assert len(fake2.agent.last_input["messages"]) == 5  #历史 4 条 + 本轮 1 条


def test_threads_are_isolated(tmp_path):
    fake = _FakeReactAgent()
    graph = build_orchestrator_graph(fake, checkpointer=_open_saver(tmp_path / "ck.db"))
    for thread in ("2483:s1", "9999:s1"):   #不同用户、相同 session_id
        graph.invoke({"messages": [{"role": "user", "content": f"你好-{thread}"}], "query": "你好"},
                     config={"configurable": {"thread_id": thread}})
    for thread in ("2483:s1", "9999:s1"):
        state = graph.get_state({"configurable": {"thread_id": thread}})
        assert len(state.values["messages"]) == 2
        assert state.values["messages"][0].content == f"你好-{thread}"


def test_graph_trims_to_ten_turns(tmp_path):
    fake = _FakeReactAgent()
    graph = build_orchestrator_graph(fake, checkpointer=_open_saver(tmp_path / "ck.db"))
    for i in range(12):
        graph.invoke({"messages": [{"role": "user", "content": f"第{i}问"}], "query": f"第{i}问"},
                     config={"configurable": {"thread_id": "u1:s1"}})
    state = graph.get_state({"configurable": {"thread_id": "u1:s1"}})
    humans = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 10
    assert humans[0].content == "第2问"      #最早两轮被裁掉
    assert humans[-1].content == "第11问"


def test_make_thread_id_isolates_users():
    assert make_thread_id("2483", "s1") == "2483:s1"
    assert make_thread_id("9999", "s1") != make_thread_id("2483", "s1")
    assert make_thread_id("2483", "") == ""      #无会话ID → 空串（调用方退化为一次性线程）


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_checkpoint_memory.py
