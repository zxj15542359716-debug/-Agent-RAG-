#编排包（第2步·编排架构）
#【新增】把"单 ReAct"升级为一张带分支的编排图：
#   START -> trim_history -> classify ─┬─(normal)─> 既有 ReactAgent（日常问答快路径）
#                                      └─(report)─> 三路并行子研究者 -> 结构化合成 -> END
#对外只暴露 OrchestratorAgent：app.py 拿它与既有 ReactAgent 同样的方式使用（.graph.stream）。
from agent.orchestration.graph import OrchestratorAgent, build_orchestrator_graph

__all__ = ["OrchestratorAgent", "build_orchestrator_graph"]
