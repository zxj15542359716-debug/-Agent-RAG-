#报告结构化与合成逻辑测试（第2步·2.3）
#覆盖：结构化报告渲染成分节文本 / 降级模式渲染 / 结论压缩（截断）/
#      合成输入在超预算时按档位丢弃信息密度最低的一路 / 结构化失败时降级为纯文本。
#全部离线：合成阶段用假模型替身（不联网），只验证"预算与降级"这类确定性逻辑。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage

from agent.orchestration import nodes
from agent.orchestration.nodes import _build_payload, _truncate_finding, synthesize_report
from agent.orchestration.report_schema import (
    AfterSalesReport,
    BasicInfo,
    Citation,
    FaultFinding,
    parse_json_object,
    render_report_text,
    schema_hint,
)

FINDINGS = {
    "fault_finding": {"status": "ok", "data": {"fault_type": "按键失灵", "root_causes": ["轴体进灰"] * 12,
                                               "steps": ["拔掉键盘", "清理轴体"], "summary": "S" * 3000}},
    "warranty_finding": {"status": "ok", "data": {"warranty_status": "保修期内", "terms": ["非人为损坏可保"],
                                                  "summary": "W" * 3000}},
    "history_finding": {"status": "ok", "data": {"months": ["2026-09"], "usage_records": ["无维修记录"],
                                                 "summary": "H" * 3000}},
}
SOURCES = [{"n": 1, "source": "故障排除.txt", "entry": "7", "title": "键盘按键失灵",
            "snippet": "S" * 500}]


# ------------------------------ 渲染 ------------------------------
def test_render_report_text_contains_sections():
    report = AfterSalesReport(
        basic_info=BasicInfo(consulted_at="2026-09", device_types="机械键盘",
                             issue_type="故障排除", warranty_status="保修期内"),
        fault_type="按键失灵", root_causes=["轴体进灰"], steps=["清理轴体"],
        warranty_terms=["非人为损坏可保"], risk_notes=["拆解前断电"], follow_up="建议送修",
        citations=[Citation(n=1, source="故障排除.txt", entry="7", title="键盘按键失灵")],
    )
    text = render_report_text(report)
    for section in ("一、基本信息", "二、故障诊断", "三、处理步骤", "四、适用保修条款",
                    "五、风险提醒与后续建议", "六、引用来源"):
        assert section in text
    assert "轴体进灰" in text and "故障排除.txt" in text


def test_render_degraded_mode_keeps_raw_text():
    text = render_report_text(AfterSalesReport(degraded_text="模型自由发挥的一段报告"))
    assert "降级模式" in text and "模型自由发挥的一段报告" in text


# ------------------------------ JSON 契约与解析 ------------------------------
def test_schema_hint_lists_fields_from_model():
    hint = schema_hint(FaultFinding)
    assert "fault_type" in hint and "root_causes" in hint and "steps" in hint
    #描述来自 Pydantic 模型，schema 改了提示词自动跟着变
    assert "故障类型" in hint


def test_parse_json_object_handles_fences_and_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('好的，结论如下：\n{"a": {"b": 2}}\n以上。') == {"a": {"b": 2}}
    #字符串里的花括号不应干扰配对
    assert parse_json_object('{"a": "含{花括号}的文本"}') == {"a": "含{花括号}的文本"}


def test_parse_json_object_rejects_garbage():
    import pytest as _pytest

    for bad in ("", "没有 JSON", '{"a": 1'):
        with _pytest.raises(ValueError):
            parse_json_object(bad)


def test_tolerant_model_handles_observed_deviations():
    """用真实跑出来的失败载荷做回归：模型会把"该给字符串的写成数组"、把条款写成一整段话、
    把引用来源写成裸字符串——这三种偏差都不该让整份报告降级。"""
    report = AfterSalesReport.model_validate({
        "basic_info": {"consulted_at": "2026-09", "device_types": ["耳机"],
                       "issue_type": "单侧无声", "warranty_status": "已过保修期"},
        "fault_type": "单侧无声",
        "root_causes": ["线材断裂", "单元损坏"],
        "steps": "先换设备测试，再检查插头接触。",          #该数组，给了字符串
        "warranty_terms": ["进水不在保修范围"],
        "citations": ["故障排除.txt · 7"],                  #该对象数组，给了裸字符串
        "follow_up": "建议送修。",
    })
    assert report.basic_info.device_types == "耳机"        #数组 → 顿号连接
    assert report.steps == ["先换设备测试，再检查插头接触。"]  #字符串 → 单元素数组
    assert report.citations[0].title == "故障排除.txt · 7"
    assert report.citations[0].n == 0                      #缺省字段用默认值，不报错


# ------------------------------ 压缩与预算 ------------------------------
def test_truncate_finding_caps_strings_and_list_length():
    out = _truncate_finding(FINDINGS["fault_finding"], chars=10)
    assert len(out["summary"]) == 10          #长字符串被截断
    assert len(out["root_causes"]) == 8       #列表限长 8 条
    assert all(len(x) <= 10 for x in out["root_causes"])
    assert out["status"] == "ok"
    assert _truncate_finding(None, 10) is None


def test_build_payload_drops_requested_kinds():
    payload = _build_payload("问题", FINDINGS, SOURCES, 100, 50, ("history",))
    assert "history" not in payload and "fault" in payload and "warranty" in payload
    assert "user_query" in payload


class _FakeStructured:
    def __init__(self, sink):
        self.sink = sink

    def invoke(self, messages):
        self.sink["messages"] = messages
        return AfterSalesReport(fault_type="按键失灵")


class _FakeChat:
    """假模型：token 计数由外部函数决定，用于驱动"超预算 → 截断"的分支"""

    def __init__(self, token_fn, structured_raises=False):
        self.token_fn = token_fn
        self.structured_raises = structured_raises
        self.sink = {}

    def get_num_tokens_from_messages(self, messages):
        return self.token_fn(messages)

    def with_structured_output(self, schema, method=None):
        #生产代码用 method="json_mode"（DeepSeek 实测支持的那一种），替身同样接受该参数
        self.sink["method"] = method
        if self.structured_raises:
            raise RuntimeError("接口不支持结构化输出")
        return _FakeStructured(self.sink)

    def invoke(self, messages):
        self.sink["fallback_messages"] = messages
        return AIMessage(content="降级后的纯文本报告")


def test_synthesize_truncates_until_within_budget(monkeypatch):
    """预算充足 → 三路都在；预算永远不够 → 用到最后一档（只留故障诊断）"""
    fake = _FakeChat(lambda msgs: 10)          #永远在预算内
    monkeypatch.setattr(nodes, "chat_model", fake)
    report, note = synthesize_report(FINDINGS, SOURCES, "帮我生成报告")
    payload = fake.sink["messages"][-1].content
    assert '"fault"' in payload and '"warranty"' in payload and '"history"' in payload
    assert "截断" not in note

    fake2 = _FakeChat(lambda msgs: 999999)     #永远超预算
    monkeypatch.setattr(nodes, "chat_model", fake2)
    report, note = synthesize_report(FINDINGS, SOURCES, "帮我生成报告")
    payload = fake2.sink["messages"][-1].content
    assert '"fault"' in payload
    assert '"history"' not in payload and '"warranty"' not in payload
    assert "超预算" in note


def test_synthesize_falls_back_to_plain_text(monkeypatch):
    fake = _FakeChat(lambda msgs: 10, structured_raises=True)
    monkeypatch.setattr(nodes, "chat_model", fake)
    report, note = synthesize_report(FINDINGS, SOURCES, "帮我生成报告")
    assert report.degraded_text == "降级后的纯文本报告"
    assert "降级" in note


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_report_schema.py
