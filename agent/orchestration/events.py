#节点级事件协议（第2步·2.4，2.2 起先落地骨架）
#【新增】父图节点向 SSE 流写"自定义事件"的统一出口（对应 LangGraph 的 stream_mode="custom"）。
#事件契约（消费侧见 app.py 的 stream_events 与前端时间线）：
#   node_start {"type","id","node","kind","label","depth","ts"}      节点开始
#   node_end   {"type","id","node","kind","status","duration_ms","summary"}  节点结束（含失败/降级）
#   report     {"type","data"}                                       结构化报告（合成完成后一次性下发）
#   run        {"type",...}                                          本轮汇总（流末尾，含 token 账）
#id 由 node_start 生成、node_end 复用，前端据此精确配对，不依赖事件到达顺序。
import time
from contextlib import contextmanager
from uuid import uuid4

NODE_START = "node_start"
NODE_END = "node_end"
REPORT = "report"
RUN = "run"


def emit(event: dict) -> None:
    """把事件写进 LangGraph 自定义流。

    非图环境（单测直接调用节点函数）或消费端未订阅 custom 流时静默跳过：
    事件是可观测性的"锦上添花"，绝不能因为它失败而影响回答主流程。
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:
        pass


@contextmanager
def node_span(node: str, label: str, kind: str = "", depth: int = 0):
    """节点执行跨度：进入发 node_start，退出发 node_end（带耗时与状态）。

    用法：
        with node_span("researcher", "故障诊断", kind="fault", depth=1) as span:
            ...
            span["status"] = "degraded"      # ok | degraded | error
            span["summary"] = "未取到结构化结论"
    异常会先记为 error 再原样抛出（节点失败要在时间线上看得见，但不能被这里吞掉）。
    """
    span_id = uuid4().hex
    started = time.time()
    emit({"type": NODE_START, "id": span_id, "node": node, "kind": kind,
          "label": label, "depth": depth, "ts": int(started * 1000)})
    info = {"status": "ok", "summary": ""}
    try:
        yield info
    except Exception as e:
        info["status"] = "error"
        info["summary"] = f"{type(e).__name__}: {e}"[:200]
        raise
    finally:
        #结束事件同样带 label/kind：前端按 id 配对后直接用结束事件渲染即可，
        #不必回查开始事件（实测漏带 label 时时间线只能显示内部节点名）
        emit({"type": NODE_END, "id": span_id, "node": node, "kind": kind, "label": label,
              "status": info["status"], "summary": info["summary"],
              "duration_ms": int((time.time() - started) * 1000)})


# ============================================================================================
# 【第 2 步 · 2.2/2.4 说明】节点级事件协议（agent/orchestration/events.py）
# --------------------------------------------------------------------------------------------
# 改动点：定义事件类型常量、emit()（写自定义流）与 node_span()（节点跨度上下文管理器）。
# 为什么用 LangGraph 自定义流而不是在节点里直接往 SSE 队列写：节点跑在 LangGraph 自己的
# 线程池里（并行研究者天然多线程），直接持有生成器/队列会引入跨线程写共享对象的复杂度；
# stream_mode="custom" 是框架自带的正规通道（图外的 app.py 统一消费）。
# 为什么 emit() 要吞异常：事件写失败的合理后果是"前端少一条进度"，绝不应该是"回答失败"。
# 与其它文件的关系：nodes.py 的各节点用 node_span 包住执行体；app.py 用
# stream_mode=["messages","custom"] 消费并把事件转成 SSE（2.4）；前端按 id 配对渲染时间线。
# ============================================================================================
