#意图路由（第2步·2.2）——纯规则判定，零 LLM 调用
#【新增】决定本轮问题走"日常问答快路径"还是"报告生成分支（并行子研究者）"。
from utils.config_handler import orchestration_config
from utils.logger_handler import logger

ROUTE_NORMAL = "normal"
ROUTE_REPORT = "report"


def classify_intent(query: str, cfg: dict | None = None) -> str:
    """判定意图：返回 ROUTE_REPORT（报告场景）或 ROUTE_NORMAL（日常问答）。

    规则 = 生成类动词 ∧ 报告类名词，两者同时命中才算报告场景（词表在 config/orchestration.yml）。
    为什么要"双条件"：只认名词会把"报告一下键盘的保养方法"当成生成报告（那是知识问答）；
    只认动词会把"帮我写个吐槽"之类也放进来。双条件把误判面压到最小。

    为什么用规则而不是让模型判定：日常问答是绝对主路径，多一次 LLM 调用等于给每次提问
    加 1~2 秒延迟和一份 token 成本；而规则的词表可配置、判定可单测、日志可解释，
    事后还能靠日志统计误判率（见 tests/test_router.py 的用例表）。

    漏判兜底：既有 Agent 里的 fill_context_for_report 工具与 report_prompt_switch 中间件
    保持原样——规则没认出来的报告类问题落回日常路径后，模型仍会调用该工具切到报告提示词，
    行为与改造前一致（只是没有并行子研究者与结构化卡片）。
    """
    cfg = (cfg or orchestration_config["router"]) or {}
    nouns = cfg.get("report_nouns") or []
    verbs = cfg.get("report_verbs") or []
    text = (query or "").strip()
    hit_nouns = [w for w in nouns if w in text]
    hit_verbs = [w for w in verbs if w in text]
    if hit_nouns and hit_verbs:
        logger.info(f"[router]报告场景（命中名词{hit_nouns} 动词{hit_verbs}）：{text[:40]}")
        return ROUTE_REPORT
    logger.info(f"[router]日常问答（名词{hit_nouns} 动词{hit_verbs}）：{text[:40]}")
    return ROUTE_NORMAL


if __name__ == "__main__":
    #自检（不联网）：跑一组典型问法，核对判定结果
    #运行：cd 项目根 && .venv/Scripts/python.exe -m agent.orchestration.router
    samples = [
        "帮我生成一份我的售后使用报告",
        "来一份这个月的使用报告",
        "导出我的维修报告",
        "报告一下键盘的保养方法",          #只有名词，应判日常
        "给我推荐一款游戏鼠标",            #只有动词，应判日常
        "耳机左耳没声音怎么办",
        "",
    ]
    print("=" * 62)
    for q in samples:
        print(f"  {classify_intent(q):>6}  <-  {q or '(空)'}")
    print("=" * 62)


# ============================================================================================
# 【第 2 步 · 2.2 说明】意图路由（agent/orchestration/router.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增 classify_intent()——报告场景 vs 日常问答的纯规则判定，词表在
# config/orchestration.yml 的 router 段。
# 为什么不做成"让模型判一下"：见函数文档字符串（延迟/token/可测性）。
# 为什么词表要可配置：与第 1 步的检索权重同理——判定规则是会被真实问法反复打磨的量，
# 放配置里改词表不用动代码，且能拿日志里的误判样例直接回归（改完词表重跑 tests/test_router.py）。
# 与其它文件的关系：nodes.py 的 classify 节点调用它并把结果写进 state.route；
# graph.py 依据 route 选择分支（2.3 起接上报告分支）；app.py 的 sources 事件与它无关。
# ============================================================================================
