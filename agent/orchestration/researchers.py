import threading
import time
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from pydantic import BaseModel

from agent.orchestration.report_schema import (
    FaultFinding,
    HistoryFinding,
    WarrantyFinding,
    output_contract,
    parse_json_object,
)
from agent.tools.agent_tools import fetch_external_data, get_current_month, query_warranty, rag_summarize
from model.fectory import chat_model
from utils.Prompt_loader import load_researcher_prompts
from utils.config_handler import orchestration_config
from utils.logger_handler import logger


@dataclass(frozen=True)
class ResearcherSpec:
    kind: str            # fault | warranty | history
    label: str           # 中文名（时间线展示用）
    tools: tuple         # 该研究者可用的工具
    schema: type[BaseModel]   # 结构化输出模型
    state_key: str       # 结论写回父图的通道名


SPECS: dict[str, ResearcherSpec] = {
    "fault": ResearcherSpec("fault", "故障诊断", (rag_summarize,), FaultFinding, "fault_finding"),
    "warranty": ResearcherSpec("warranty", "保修政策", (query_warranty, rag_summarize), WarrantyFinding, "warranty_finding"),
    "history": ResearcherSpec("history", "历史工单", (fetch_external_data, get_current_month), HistoryFinding, "history_finding"),
}

_researchers: dict[str, Any] = {}
_build_lock = threading.Lock()
#并发闸门：config 里的 parallel_limit 限制同时打给模型的研究者数量（默认 3 = 三路齐发）
_semaphore = threading.BoundedSemaphore(max(1, int(orchestration_config["report"].get("parallel_limit", 3))))


def get_researcher(kind: str):
    """惰性构建并缓存某个研究者（双重检查锁，与 agent_tools._get_rag_service 同款）。"""
    agent = _researchers.get(kind)
    if agent is None:
        with _build_lock:
            agent = _researchers.get(kind)
            if agent is None:
                spec = SPECS[kind]
                agent = create_agent(
                    model=chat_model,
                    system_prompt=load_researcher_prompts(kind),
                    tools=list(spec.tools),
                    middleware=[ModelCallLimitMiddleware(
                        run_limit=int(orchestration_config["report"].get("researcher_max_calls", 6)),
                        exit_behavior="end")],
                    checkpointer=False,
                )
                _researchers[kind] = agent
                logger.info(f"[researcher]构建完成：{spec.label}（工具 {[t.name for t in spec.tools]}）")
    return agent


def _user_message(spec: ResearcherSpec, query: str) -> str:
    """研究者的人话输入 + JSON 输出契约（字段清单由 Pydantic 模型自动生成）"""
    return (
        f"用户问题：{query}\n\n"
        "登录用户身份由系统自动注入（相关工具会使用当前登录用户的数据），无需向用户索要用户ID。\n"
        "请按你的职责完成调研（需要时调用工具检索/查询），工具调用不超过 2 次，"
        "拿到关键信息就尽快输出结论。\n\n"
        f"{output_contract(spec.schema)}"
    )


def _last_ai_text(result: dict) -> str:
    for msg in reversed(result.get("messages") or []):
        if getattr(msg, "content", "") and msg.__class__.__name__.startswith("AI"):
            return str(msg.content)
    return ""


def run_researcher(kind: str, query: str, context: dict | None = None) -> tuple[dict, str]:
    """执行一个研究者：返回 (结论数据, 状态)。status ∈ ok / degraded / error。"""
    spec = SPECS[kind]
    agent = get_researcher(kind)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            #并发闸门：限制同时打给模型的研究者数（429 风险控制，见 config parallel_limit）
            with _semaphore:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": _user_message(spec, query)}]},
                    context=context or {},
                )
            text = _last_ai_text(result)
            try:
                #结构化结论：解析模型输出的 JSON 并用 Pydantic 校验（字段缺失/类型不符会被兜住）
                data = spec.schema.model_validate(parse_json_object(text)).model_dump()
                return data, "ok"
            except Exception as parse_err:
                logger.warning(f"[researcher]{spec.label} 结构化解析失败（{parse_err}），降级为文本兜底")
                return {"summary": text[:500]}, "degraded"
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(k in msg for k in ("rate limit", "429", "timeout", "timed out", "connection"))
            logger.warning(f"[researcher]{spec.label} 第 {attempt + 1} 次执行失败：{e}")
            if not retryable or attempt == 2:
                break
            time.sleep(1.5 * (attempt + 1))    #退避重试：并行打模型时更易触发限流
    logger.error(f"[researcher]{spec.label} 执行失败：{last_err}", exc_info=last_err is not None)
    return {}, "error"


def get_spec(kind: str) -> ResearcherSpec:
    return SPECS[kind]