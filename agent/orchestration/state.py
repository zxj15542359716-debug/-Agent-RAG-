#编排图的状态定义（第2步·编排架构）
#【新增】父图（编排图）与子研究者共用的状态结构。
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    """编排图状态。

    messages 是唯一被 checkpointer 持久化的"会话历史"通道：
      - 日常问答：normal 节点只把最终回答写回（与改造前"只存 user + assistant"口径一致）；
      - 报告场景：报告正文由 synthesize 节点写回一条 AI 消息。
    其余字段是"本轮运行"的临时状态，由 classify 节点在每轮开头清空，避免串轮。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str              #本轮原始问题
    route: str              #路由结果："normal" | "report"
    run_id: str             #本轮运行号（= 运行历史表 runs.run_id，uuid4().hex）
    #三个研究者各写各的通道：并行写同一通道需要 reducer 且会跨轮累积，
    #分开写则天然"每轮覆盖"，无并发写冲突、无残留
    fault_finding: dict | None      #故障诊断结论
    warranty_finding: dict | None   #保修政策结论
    history_finding: dict | None    #历史工单结论
    report: dict | None             #合成后的结构化报告（Pydantic dump）


class ResearcherInput(TypedDict):
    """Send() 派发给研究者的入参（同时作为 researcher 节点的 input_schema）。"""

    kind: str    #fault | warranty | history
    query: str


#研究者种类 -> 中文标签（前端时间线用；与 app.py 的 _TOOL_LABELS 同款用途）
RESEARCHER_KINDS = ("fault", "warranty", "history")
RESEARCHER_LABELS = {
    "fault": "故障诊断",
    "warranty": "保修政策",
    "history": "历史工单",
}


# ============================================================================================
# 【第 2 步 · 2.1 说明】编排图状态（agent/orchestration/state.py）
# --------------------------------------------------------------------------------------------
# 为什么 messages 之外还要单列三个 finding 通道：LangGraph 的并行分支如果同时写一个
# 普通通道会抛 InvalidUpdateError，写成 Annotated[list, add] 又会跨轮累积（下一轮报告会把
# 上一轮结论再合成一遍）。三个独立通道是"每轮覆盖"的语义，既无冲突也无残留，
# 由 classify 节点每轮开头统一清零。
# 与其它文件的关系：graph.py 用它编译图；nodes.py 的节点函数读写这些键；
# app.py 只消费 messages（其它键不落前端）。
# ============================================================================================
