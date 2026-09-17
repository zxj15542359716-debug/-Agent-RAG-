from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    """编排图状态。"""

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