import json
import re
from typing import get_args, get_origin

from pydantic import BaseModel, Field, field_validator


class _TolerantModel(BaseModel):
    """对模型输出的常见偏差做容错：该给数组的字段给了单个字符串时，自动包成单元素数组。"""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_common_deviations(cls, value, info):
        field = cls.model_fields.get(info.field_name)
        if field is None:
            return value
        annotation = field.annotation
        if get_origin(annotation) is list:
            #该给数组给了标量 → 包成单元素数组（实测：模型把条款写成一整段话）
            if not isinstance(value, list):
                if value is None:
                    return []
                if isinstance(value, str):
                    return [value] if value.strip() else []
                return [value]
            inner = (get_args(annotation) or (None,))[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                #数组元素该是对象却给了字符串（实测：引用来源写成 "故障排除.txt · 7"）→ 挂到 title
                return [item if isinstance(item, dict) else
                        ({"title": item} if isinstance(item, str) else {"title": str(item)})
                        for item in value]
            return value
        if annotation is str and isinstance(value, list):
            #该给字符串给了数组（实测：device_types 写成 ["耳机"]）→ 用顿号连成一句
            return "、".join(str(x) for x in value)
        return value


class Citation(_TolerantModel):
    """引用来源（与第 1 步 sources 事件的字段一致，便于报告与溯源对齐）"""

    n: int = Field(0, description="参考资料序号（与检索结果编号一致）")
    source: str = Field("", description="来源文件")
    entry: str = Field("", description="条目号")
    title: str = Field("", description="条目标题")
    snippet: str = Field("", description="片段摘录")


class FaultFinding(_TolerantModel):
    """故障诊断研究者的结论"""

    fault_type: str = Field("", description="故障类型，如 按键失灵/进水/电池续航衰减；无法判断写'暂无法判定'")
    root_causes: list[str] = Field(default_factory=list, description="可能根因，按可能性从高到低，每条一句话")
    steps: list[str] = Field(default_factory=list, description="排查与处理步骤，可操作、按顺序")
    safety_notes: list[str] = Field(default_factory=list, description="安全提醒（涉及通电/拆解/进水/电池时必须给出）")
    summary: str = Field("", description="一段话结论，供报告合成使用")


class WarrantyFinding(_TolerantModel):
    """保修政策研究者的结论"""

    warranty_status: str = Field("", description="保修状态：保修期内/已过保修期/暂无相关信息")
    terms: list[str] = Field(default_factory=list, description="适用保修条款与政策要点")
    purchases: list[str] = Field(default_factory=list, description="相关购买记录（外设类型/购买时间/月份）")
    summary: str = Field("", description="一段话结论，供报告合成使用")


class HistoryFinding(_TolerantModel):
    """历史工单研究者的结论"""

    months: list[str] = Field(default_factory=list, description="实际查询到的月份（YYYY-MM）")
    usage_records: list[str] = Field(default_factory=list, description="使用与维修记录要点")
    patterns: list[str] = Field(default_factory=list, description="从记录看出的规律（如重复报修、多次同型号故障）")
    summary: str = Field("", description="一段话结论，供报告合成使用")


class BasicInfo(_TolerantModel):
    """报告基本信息"""

    consulted_at: str = Field("", description="咨询时间（YYYY-MM 或具体日期；缺失写'暂无相关信息'）")
    device_types: str = Field("", description="涉及品类（耳机/机械键盘/磁轴键盘/麦克风/鼠标）")
    issue_type: str = Field("", description="问题类型：故障排除/维护保养/选购咨询/常见问答/保修查询")
    warranty_status: str = Field("", description="保修状态（含购买时间与保修截止年月；不涉及写'暂无相关信息'）")


class AfterSalesReport(_TolerantModel):
    """售后服务报告（结构化输出的顶层对象）"""

    basic_info: BasicInfo = Field(default_factory=BasicInfo, description="第一部分：基本信息")
    fault_type: str = Field("", description="故障类型")
    root_causes: list[str] = Field(default_factory=list, description="根因分析，按可能性排序")
    steps: list[str] = Field(default_factory=list, description="处理/排查步骤，可操作")
    warranty_terms: list[str] = Field(default_factory=list, description="适用保修条款")
    risk_notes: list[str] = Field(default_factory=list, description="风险提醒与后续建议")
    citations: list[Citation] = Field(default_factory=list, description="引用来源（必须来自检索结果，不得编造）")
    follow_up: str = Field("", description="后续建议（一段话）")
    degraded_text: str = Field("", description="结构化输出失败时的原始文本（降级模式；正常报告此字段为空）")


def _type_label(annotation) -> str:
    """把字段类型翻译成给模型看的中文说明（含嵌套对象/对象数组的键名展开）"""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        inner = args[0] if args else str
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return "对象数组（每项含 " + "、".join(inner.model_fields.keys()) + "）"
        return "字符串数组"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "对象（含 " + "、".join(annotation.model_fields.keys()) + "）"
    return "字符串"


def schema_hint(schema: type[BaseModel]) -> str:
    """把 Pydantic 模型转成给模型看的"字段清单"，用作提示词里的 JSON 契约。"""
    return "\n".join(
        f"- {name}（{_type_label(field.annotation)}）：{field.description or '（无说明）'}"
        for name, field in schema.model_fields.items()
    )


def output_contract(schema: type[BaseModel]) -> str:
    """拼出完整的"输出契约"文本块，供提示词/消息末尾直接追加"""
    return (
        "# 输出格式\n"
        "只输出一个 JSON 对象（不要代码块围栏、不要任何解释性文字），字段如下：\n"
        f"{schema_hint(schema)}\n"
        "注意：标注为“字符串数组”的字段必须写成 JSON 数组（即使只有一条也要写 [\"…\"]）；"
        "对象字段必须使用上面列出的键名，不要自创字段名。"
    )


def parse_json_object(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象（容错代码块围栏与前后缀说明）。"""
    if not text:
        raise ValueError("模型输出为空")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("输出里没有 JSON 对象")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("JSON 对象不完整")


def render_report_text(report: AfterSalesReport) -> str:
    """把结构化报告渲染成分节纯文本。"""
    def bullets(items: list[str]) -> str:
        return "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) if items else "暂无相关信息"

    info = report.basic_info
    if report.degraded_text:
        #降级模式：结构化合成失败时保留原始文本，至少让用户看到"模型说了什么"
        return "售后服务报告（降级模式：结构化合成失败，以下为原始生成文本）\n\n" + report.degraded_text
    lines = [
        "售后服务报告",
        "",
        "一、基本信息",
        f"咨询时间：{info.consulted_at or '暂无相关信息'}",
        f"涉及品类：{info.device_types or '暂无相关信息'}",
        f"问题类型：{info.issue_type or '暂无相关信息'}",
        f"保修状态：{info.warranty_status or '暂无相关信息'}",
        "",
        "二、故障诊断",
        f"故障类型：{report.fault_type or '暂无法判定'}",
        "可能根因：",
        bullets(report.root_causes),
        "",
        "三、处理步骤",
        bullets(report.steps),
        "",
        "四、适用保修条款",
        bullets(report.warranty_terms),
        "",
        "五、风险提醒与后续建议",
        bullets(report.risk_notes),
        report.follow_up or "",
    ]
    if report.citations:
        lines += ["", "六、引用来源"]
        for c in report.citations:
            lines.append(f"[{c.n}] {c.source} · {c.entry} {c.title}".rstrip())
    return "\n".join(line for line in lines if line is not None)
