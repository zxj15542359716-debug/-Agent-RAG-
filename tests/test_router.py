#意图路由测试（第2步·2.2）
#覆盖：典型报告问法判为 report；只有名词或只有动词的问法判为 normal；空输入兜底；
#      classify 节点清空上一轮报告残留并写入 route。全部离线（纯规则，不加载模型/索引）。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestration.nodes import classify
from agent.orchestration.router import ROUTE_NORMAL, ROUTE_REPORT, classify_intent

#应判为报告场景：生成类动词 + 报告类名词同时出现
REPORT_CASES = [
    "帮我生成一份我的售后使用报告",
    "来一份这个月的使用报告",
    "导出我的维修报告",
    "帮我出一份工单报告",
    "整理一下我的年度报告",
    "给我生成过去三个月的使用报告",
]

#应判为日常问答
NORMAL_CASES = [
    "报告一下键盘的保养方法",      #只有名词、没有生成类动词 → 是知识问答而非生成报告
    "给我推荐一款游戏鼠标",        #只有动词、没有报告类名词
    "耳机左耳没声音怎么办",
    "机械键盘进水了还能用吗",
    "保修期是多久",
    "帮我写一段产品介绍",          #有动词无报告名词
]


def test_report_intents_are_detected():
    for q in REPORT_CASES:
        assert classify_intent(q) == ROUTE_REPORT, f"应判报告场景：{q}"


def test_normal_intents_are_not_hijacked():
    for q in NORMAL_CASES:
        assert classify_intent(q) == ROUTE_NORMAL, f"应判日常问答：{q}"


def test_empty_query_falls_back_to_normal():
    assert classify_intent("") == ROUTE_NORMAL
    assert classify_intent(None) == ROUTE_NORMAL


def test_classify_node_writes_route_and_clears_residue():
    stale = {
        "query": "帮我生成一份售后报告",
        "route": "normal",
        "fault_finding": {"summary": "上一轮的结论"},
        "warranty_finding": {"summary": "上一轮的结论"},
        "history_finding": {"summary": "上一轮的结论"},
        "report": {"故障类型": "上一轮的报告"},
    }
    out = classify(stale, None)
    assert out["route"] == ROUTE_REPORT
    #三个 finding 通道与 report 必须被清零，否则上轮结论会挂在本轮 state 里误导排查
    for key in ("fault_finding", "warranty_finding", "history_finding", "report"):
        assert out[key] is None


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_router.py
