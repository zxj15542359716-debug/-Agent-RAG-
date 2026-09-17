from utils.config_handler import orchestration_config
from utils.logger_handler import logger

ROUTE_NORMAL = "normal"
ROUTE_REPORT = "report"


def classify_intent(query: str, cfg: dict | None = None) -> str:
    """判定意图：返回 ROUTE_REPORT（报告场景）或 ROUTE_NORMAL（日常问答）。"""
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
