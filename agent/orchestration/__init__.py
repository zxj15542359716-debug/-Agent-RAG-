#编排包（第2步·编排架构）
#【新增】把"单 ReAct"升级为一张带分支的编排图：
#   START -> trim_history -> classify ─┬─(normal)─> 既有 ReactAgent（日常问答快路径）
#                                      └─(report)─> 三路并行子研究者 -> 结构化合成 -> END
#对外只暴露 OrchestratorAgent：app.py 拿它与既有 ReactAgent 同样的方式使用（.graph.stream）。
from agent.orchestration.graph import OrchestratorAgent, build_orchestrator_graph

__all__ = ["OrchestratorAgent", "build_orchestrator_graph"]

# ============================================================================================
# 【第 2 步 · 2.1~2.4 说明】编排包的对外接口（本文件的定位）
# --------------------------------------------------------------------------------------------
# 本包是第 2 步的全部编排逻辑，按职责分文件：
#   state.py       状态通道定义        graph.py       图装配 + OrchestratorAgent（唯一入口）
#   router.py      意图路由（纯规则）  nodes.py       各节点实现（裁剪/路由/问答/研究者/合成）
#   researchers.py 三路并行子研究者    report_schema.py 结构化报告模型与解析/渲染
#   events.py      节点事件协议        checkpoint.py  会话持久化（SqliteSaver）
# 对外只暴露 OrchestratorAgent：app.py 以 self.graph.stream(...) 驱动，与改造前的
# ReactAgent 用法同形（`.agent` -> `.graph` 的唯一区别），因此 Web 层改动面很小。
# 验证：.venv/Scripts/python.exe -m agent.orchestration.graph（打印图结构与节点/边）
# ============================================================================================
