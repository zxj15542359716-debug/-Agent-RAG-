#token 记账测试（第2步·2.4）
#覆盖：账本跨线程累计（并行研究者场景）/ 回调经图运行时入账并归属到"阶段" /
#      DashScope usage 取值兼容 dict 与对象 / 流水落库。
#全部离线：用自定义假模型（自带 usage_metadata）与临时库，不调任何真实接口。
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from service.database_service import DatabaseService
from utils.usage_ledger import (
    RunRecorder,
    UsageCallbackHandler,
    begin_run,
    current_recorder,
    end_run,
    persist_to_db,
    record_embedding,
    record_rerank,
    scope,
    usage_tokens,
)


class _UsageFakeChat(BaseChatModel):
    """每次调用返回带 usage_metadata 的回答（模拟真实接口返回用量）"""

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def _generate(self, messages, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        msg = AIMessage(content="假回答",
                        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


def test_recorder_accumulates_across_threads():
    """账本要能从并行的三个研究者线程同时入账（这正是报告场景的真实形态）"""
    rec = RunRecorder("r1", "test")
    barrier = threading.Barrier(3, timeout=5)

    def worker(n):
        barrier.wait()
        for _ in range(n):
            rec.add_llm("fault", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    threads = [threading.Thread(target=worker, args=(i,)) for i in (10, 20, 30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = rec.snapshot()
    assert snap["llm_calls"] == 60
    assert snap["total_tokens"] == 900
    assert snap["by_node"]["fault"] == {"calls": 60, "input": 600, "output": 300, "total": 900}


def test_callback_records_usage_through_graph_runtime():
    """回调处理器挂在图运行的 config 上时，能把用量记进账本并归属到阶段名"""
    model = _UsageFakeChat()

    def node(state, runtime: Runtime):
        with scope("probe_node"):
            model.invoke(state["messages"])
        return {}

    graph = StateGraph(_State)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    app = graph.compile()

    rec = begin_run("t-run", "test")
    try:
        app.invoke({"messages": [{"role": "user", "content": "hi"}]},
                   config={"callbacks": [UsageCallbackHandler()]})
    finally:
        end_run()
    snap = rec.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["total_tokens"] == 120
    assert snap["by_node"]["probe_node"]["total"] == 120


def test_ledger_scoped_helpers_record_dashscope_usage():
    rec = begin_run("t2", "test")
    try:
        record_embedding(37)
        record_rerank(1200, calls=3)
        record_rerank(0, calls=1)      #拿不到 token 也要记次数
    finally:
        end_run()
    snap = rec.snapshot()
    assert snap["embedding_tokens"] == 37
    assert snap["rerank_tokens"] == 1200
    assert snap["rerank_calls"] == 4
    assert current_recorder() is None   #end_run 之后账本要清干净，避免串到下一次请求


def test_usage_tokens_accepts_dict_and_object():
    class _Obj:
        total_tokens = 9

    assert usage_tokens({"total_tokens": 5}) == 5
    assert usage_tokens(_Obj()) == 9
    assert usage_tokens(None) == 0
    assert usage_tokens({}) == 0


def test_persist_to_db_writes_usage_events(tmp_path):
    db = DatabaseService(db_path=str(tmp_path / "usage.db"), import_seed=False)
    rec = RunRecorder("run-1", "chat")
    rec.add_llm("fault", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    rec.add_llm("synthesize", {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
    rec.add_embedding(20)
    rec.add_rerank(900, calls=2)
    persist_to_db(db, rec, chat_model_name="deepseek-v4-pro", embedding_model_name="text-embedding-v4")

    rows = db.usage_summary()
    by_model = {(r["provider"], r["model"], r["unit"]): r["amount"] for r in rows}
    assert by_model[("deepseek", "deepseek-v4-pro", "token")] == 155      #15 + 140
    assert by_model[("dashscope", "text-embedding-v4", "token")] == 20
    assert by_model[("dashscope", "gte-rerank-v2", "token")] == 900
    #流水按节点分开记，能回答"钱花在哪条研究线上"
    notes = [r for r in rows if r["provider"] == "deepseek" and r["events"]]
    assert sum(r["events"] for r in notes) == 2


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_usage_ledger.py
