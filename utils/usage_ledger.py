#token 成本记账（第2步·2.4）
#【新增】把"每次模型/嵌入/重排调用花了多少 token"记成账，一轮对话一个账本（RunRecorder）：
#   1. LLM 调用：经 UsageCallbackHandler 逐次入账——它能覆盖工具内部的 LLM 调用
#      （rag_summarize 的总结、评测判官）与 Send 并行的子研究者，不依赖调用方自觉；
#   2. DashScope 的 embedding / rerank：SDK 不经 LangChain 回调，由各自调用点显式入账；
#   3. 账本最后由 app.py 写进 runs / usage_events 两张表（见 service/database_service.py）。
#
#跨线程关联（本模块最关键的一处）：请求线程用 begin_run() 把账本放进 ContextVar，
#LangGraph 的每个任务都会 copy_context()，所以所有节点线程（含 Send 并行的子研究者）
#都能读到同一个账本。反过来——绝不在节点里 set()：节点内设置不会传给下一个节点，
#会让账目静默丢失。节点"当前阶段"用 threading.local 记录（并行分支天然各记各的）。
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.callbacks import BaseCallbackHandler

from utils.logger_handler import logger

_recorder: ContextVar["RunRecorder | None"] = ContextVar("usage_recorder", default=None)
_phase = threading.local()


class RunRecorder:
    """一次运行的用量账本。

    线程安全：Send 并行的三个子研究者会从不同线程同时入账，因此累计操作加锁；
    账本对象本身跨线程共享引用（放进 ContextVar 后不再替换），这是与
    "ContextVar 不跨节点传递"这一限制共存的唯一正确姿势。
    """

    def __init__(self, run_id: str = "", source: str = "chat") -> None:
        self.run_id = run_id
        self.source = source
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.embedding_tokens = 0
        self.rerank_tokens = 0
        self.rerank_calls = 0
        self.by_node: dict[str, dict] = {}
        self.notes: list[str] = []

    # ---- 入账 ----
    def add_llm(self, node: str, usage: dict) -> None:
        node = node or "llm"
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
            self.total_tokens += int(usage.get("total_tokens", 0))
            slot = self.by_node.setdefault(node, {"calls": 0, "input": 0, "output": 0, "total": 0})
            slot["calls"] += 1
            slot["input"] += int(usage.get("input_tokens", 0))
            slot["output"] += int(usage.get("output_tokens", 0))
            slot["total"] += int(usage.get("total_tokens", 0))

    def add_embedding(self, tokens: int) -> None:
        with self._lock:
            self.embedding_tokens += int(tokens or 0)

    def add_rerank(self, tokens: int, calls: int = 1) -> None:
        with self._lock:
            self.rerank_tokens += int(tokens or 0)
            self.rerank_calls += int(calls or 0)

    def note(self, text: str) -> None:
        with self._lock:
            if text and text not in self.notes:
                self.notes.append(text)

    # ---- 出账 ----
    def snapshot(self) -> dict:
        """导出账本（写库 / 下发 run 事件 / 打印用）"""
        with self._lock:
            return {
                "run_id": self.run_id,
                "source": self.source,
                "llm_calls": self.llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "embedding_tokens": self.embedding_tokens,
                "rerank_tokens": self.rerank_tokens,
                "rerank_calls": self.rerank_calls,
                "by_node": {k: dict(v) for k, v in self.by_node.items()},
                "notes": list(self.notes),
            }


# ---------------- 账本生命周期 ----------------

def current_recorder() -> "RunRecorder | None":
    return _recorder.get()


def begin_run(run_id: str, source: str = "chat") -> RunRecorder:
    """在请求线程（或离线脚本主线程）开启一个账本。

    ContextVar 只在这里 set 一次；LangGraph 的 BackgroundExecutor 对每个任务
    copy_context()（实测确认），因此节点线程与并行 Send worker 都能读到它。
    """
    recorder = RunRecorder(run_id, source)
    _recorder.set(recorder)
    return recorder


def end_run() -> None:
    _recorder.set(None)


@contextmanager
def scope(kind: str):
    """在节点内标记"当前阶段"（fault/warranty/history/synthesize/normal...）。

    只写线程本地变量：并行分支各在自己的线程里跑，天然互不干扰；
    回调入账时优先用这个阶段名，其次才是框架给的节点名——这样"钱花在哪条研究线上"看得清。
    """
    prev = getattr(_phase, "kind", "")
    _phase.kind = kind
    try:
        yield
    finally:
        _phase.kind = prev


def current_phase() -> str:
    return getattr(_phase, "kind", "")


# ---------------- DashScope 显式入账入口 ----------------

def usage_tokens(usage) -> int:
    """从 DashScope 返回的 usage 里取 token 总数。

    实测：embedding 的 resp.usage 是 dict（{'total_tokens': 5}），而 SDK 里另一些响应
    是带属性的对象，所以两种都要认；取不到就返回 0（不记账，不影响调用）。
    """
    if not usage:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0) or 0)
    return int(getattr(usage, "total_tokens", 0) or 0)


def record_embedding(tokens: int) -> None:
    rec = current_recorder()
    if rec is not None and tokens:
        rec.add_embedding(tokens)


def record_rerank(tokens: int, calls: int = 1) -> None:
    rec = current_recorder()
    if rec is not None and (tokens or calls):
        rec.add_rerank(tokens, calls)


# ---------------- 回调处理器 ----------------

def _extract_usage(response) -> dict:
    """从 LLMResult 取用量：优先消息上的 usage_metadata，退回 llm_output.token_usage"""
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage:
            return {"input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)}
    except Exception:
        pass
    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage:
        return {"input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0)}
    return {}


class UsageCallbackHandler(BaseCallbackHandler):
    """把每次 LLM 调用的用量写进当前账本。

    LangChain 的回调会贯穿整棵调用树（包括工具内部的 LLM 调用与并行子图），
    因此挂一个处理器就能覆盖全部模型调用——这正是"成本要算得全"的关键。
    拿不到 usage（接口不返回 / 流式未带）时静默跳过，不影响回答。
    """

    def __init__(self) -> None:
        self._meta: dict[str, str] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs):
        #记下这次调用属于哪个节点/命名空间（结束回调里用来归属）
        meta = metadata or {}
        self._meta[str(run_id)] = meta.get("langgraph_node") or ""

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        recorder = current_recorder()
        if recorder is None:
            return
        node = current_phase() or self._meta.pop(str(run_id), "") or "llm"
        usage = _extract_usage(response)
        if usage:
            recorder.add_llm(node, usage)


# ---------------- 落库 ----------------

def persist_to_db(db, recorder: RunRecorder, chat_model_name: str = "", embedding_model_name: str = "") -> None:
    """把账本写进 usage_events（按节点/按来源各记一条）。

    为什么单独记流水表而不是只存 runs 的汇总列：流水能回答"钱花在哪条研究线/哪次调用上"，
    汇总列只能回答"这轮一共多少"——第 2 步要的是前者。
    写库失败只记日志：记账再重要，也不能因为它失败影响回答。
    """
    try:
        snapshot = recorder.snapshot()
        for node, slot in snapshot["by_node"].items():
            db.add_usage_event(source=recorder.source, provider="deepseek", model=chat_model_name,
                               unit="token", amount=slot["total"], run_id=recorder.run_id,
                               note=f"节点 {node}（{slot['calls']} 次调用）")
        if snapshot["embedding_tokens"]:
            db.add_usage_event(source=recorder.source, provider="dashscope", model=embedding_model_name,
                               unit="token", amount=snapshot["embedding_tokens"],
                               run_id=recorder.run_id, note="向量化")
        if snapshot["rerank_tokens"] or snapshot["rerank_calls"]:
            db.add_usage_event(source=recorder.source, provider="dashscope", model="gte-rerank-v2",
                               unit="token", amount=snapshot["rerank_tokens"],
                               run_id=recorder.run_id, note=f"重排 {snapshot['rerank_calls']} 次")
    except Exception as e:
        logger.warning(f"[usage]用量流水写入失败（不影响回答）：{e}", exc_info=True)


# ============================================================================================
# 【第 2 步 · 2.4 说明】token 成本记账（utils/usage_ledger.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增 RunRecorder（线程安全账本）、UsageCallbackHandler（回调入账）、
# begin_run/end_run/scope（生命周期与阶段标记）、record_embedding/record_rerank（DashScope 入口）、
# persist_to_db（写流水表）。
# 为什么用回调而不是在每个调用点手写计数：模型调用散落在 Agent、工具内部、子研究者、
# 合成器四处，靠人记得加计数迟早漏；LangChain 回调贯穿整棵调用树，挂一次就全覆盖。
# 为什么全局只 set 一次 ContextVar：节点内 set 不会传给下游节点（实测），
# 那会让并行分支拿到 None、账目静默丢失；而请求线程 set 一次，copy_context 会带给所有节点。
# 为什么"阶段"用 threading.local：并行研究者跑在不同线程，线程本地天然隔离，
# 不需要锁也不会串号；账本累计才需要锁（多线程同时入账）。
# 拿不到 usage 时：跳过该次（不记 0 也不报错）；app.py 会在 run 备注里写 usage_unavailable，
# 前端据此不显示 token 行——功能可降级，记账失败绝不影响回答。
# 口径提醒：这里算的是"我们消耗了多少"，不是"账户还剩多少额度"——剩余额度只能去
# 阿里云百炼控制台看（scripts/usage_report.py 的输出里也会写明这句）。
# ============================================================================================
