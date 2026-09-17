#生产工具集（Agent 实际注册的工具）
#【重构】原文件混杂教学 mock 工具与全局 external_data 字典，本次重构要点：
#1. 数据解析下沉到 service/external_record_service.py，本层只做参数校验与友好返回
#2. mock 工具（get_weather/get_user_city 等）迁出到同目录 mock_tools.py，生产 Agent 不注册
#3. RAG 服务改为惰性单例：导入本模块不再加载 FAISS 索引/embedding 模型
#4. 工具内异常不抛给模型（langgraph 会把异常文本装进 error 消息喂给模型），
#   堆栈与绝对路径只进日志，返回给模型的是简洁中文提示
import json
import re
import threading
from datetime import datetime

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from utils.logger_handler import logger
from service.external_record_service import get_external_record_service

# ---- RAG 惰性单例：首次调用工具时才初始化（加载 FAISS 索引），导入期零开销 ----
_rag_service = None
_rag_init_lock = threading.Lock()

def _get_rag_service():
    """惰性获取 RagSummarizeService 单例，延迟导入避免模块加载期重初始化。"""
    global _rag_service
    if _rag_service is None:
        with _rag_init_lock:
            if _rag_service is None:
                logger.info("[工具]首次初始化 RAG 总结服务（加载 FAISS 索引）")
                from rag.rag_service import RagSummarizeService
                _rag_service = RagSummarizeService()
    return _rag_service

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """RAG 总结工具：检索知识库并总结回答，异常兜底返回友好提示。"""
    try:
        answer, sources = _get_rag_service().rag_summarize(query)
        _publish_sources(sources)
        return answer
    except Exception as e:
        #堆栈只进日志，不让模型看到异常细节与文件路径
        logger.error(f"[rag_summarize]检索失败：{str(e)}", exc_info=True)
        return "知识库检索暂时不可用，请稍后再试或联系人工客服。"


def _publish_sources(sources: list[dict]) -> None:
    """【第1步·1.4】把检索来源写入运行时上下文（app.py 放入的共享列表，见 stream_events）。"""
    try:
        sink = (get_runtime().context or {}).get("sources")
    except Exception:
        return
    if not isinstance(sink, list):
        return
    seen = {s.get("id") for s in sink}
    for s in sources:
        if s.get("id") and s["id"] not in seen:
            sink.append(s)
            seen.add(s["id"])

#参数校验正则：用户ID为4位数字，月份为 YYYY-MM
_USER_ID_PATTERN = re.compile(r"^\d{4}$")
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#【新增】运行时上下文读取：app.py 在 context 中注入 "current_user"（登录用户ID），
#工具据此把数据查询默认绑定到登录用户——登录后即使用户不报ID也能查到自己的记录，
#且模型传了别的ID也会被覆盖，防止越权查询他人数据
def _current_user_id() -> str:
    """取当前登录用户ID；未登录或不在图运行环境内（如 __main__ 自检）返回空字符串"""
    try:
        return (get_runtime().context or {}).get("current_user", "")
    except Exception:
        return ""

@tool(description="从外部系统中获取用户使用与售后维修记录，返回字符串，没检测到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """查询用户在某月的外设使用与售后维修记录。 """
    #【新增】登录状态下强制使用登录用户的ID（数据默认查登录者本人，防止越权）
    current_user = _current_user_id()
    if current_user:
        user_id = current_user
    #参数校验：非法参数返回友好文案，避免把 KeyError/异常文本喂给模型
    if not _USER_ID_PATTERN.fullmatch(user_id or ""):
        logger.warning(f"[fetch_external_data]非法用户ID参数：{user_id!r}")
        return "查询失败：用户ID应为4位数字（如 2483），请确认后重试。"
    if not _MONTH_PATTERN.fullmatch(month or ""):
        logger.warning(f"[fetch_external_data]非法月份参数：{month!r}")
        return "查询失败：月份应为YYYY-MM格式（如 2025-04），请确认后重试。"

    service = get_external_record_service()
    data = service.get_records(user_id, month)

    if data is not None:
        logger.info(f"[fetch_external_data]命中 {user_id} 在 {month} 的使用记录")
        return json.dumps(data, ensure_ascii=False)

    #数据源故障（文件缺失等）与"业务上无记录"必须区分文案
    if service._load_error is not None:
        logger.error(f"[fetch_external_data]外部数据源异常：{service._load_error}")
        return "外部使用记录数据暂不可用，请稍后再试。"

    logger.warning(f"[fetch_external_data]未检索到 {user_id} 在 {month} 的使用记录")
    return ""

@tool(description="查询外设保修状态。外设保修期自购买时间起12个月（一年）。user_id为4位数字字符串；purchase_month为购买时间YYYY-MM格式（如2025-04），不填时返回该用户全部购买记录的保修状态")
def query_warranty(user_id: str, purchase_month: str = "") -> str:
    """保修查询工具：自购买时间起保修一年（agent.yml 的 warranty_months 可配置）。 """
    #【新增】登录状态下强制使用登录用户的ID（只能查自己的保修状态，防止越权）
    current_user = _current_user_id()
    if current_user:
        user_id = current_user
    if not _USER_ID_PATTERN.fullmatch(user_id or ""):
        logger.warning(f"[query_warranty]非法用户ID参数：{user_id!r}")
        return "查询失败：用户ID应为4位数字（如 2483），请确认后重试。"

    if purchase_month and not _MONTH_PATTERN.fullmatch(purchase_month):
        logger.warning(f"[query_warranty]非法购买时间参数：{purchase_month!r}")
        return "查询失败：购买时间应为YYYY-MM格式（如 2025-04），请确认后重试。"

    service = get_external_record_service()

    #不指定购买时间时，查询该用户全部购买记录；指定则只查该月
    months = [purchase_month] if purchase_month else service.get_user_months(user_id)
    if not months:
        if service._load_error is not None:
            logger.error(f"[query_warranty]外部数据源异常：{service._load_error}")
            return "保修数据暂不可用，请稍后再试。"
        logger.warning(f"[query_warranty]未查询到 {user_id} 的购买记录")
        return f"未查询到用户 {user_id} 的购买记录。"

    parts: list[str] = []
    for month in months:
        info = service.get_warranty_info(user_id, month)
        if info is None:
            parts.append(f"购买时间 {month}：未查询到对应购买记录")
            continue
        if info["状态"] == "保修期内":
            detail = f"剩余 {info['剩余月数']} 个月"
        else:
            detail = f"已过期 {info['已过期月数']} 个月"
        parts.append(
            f"购买时间 {info['购买时间']}：{info['状态']}（保修截止 {info['保修截止']}，{detail}）"
        )
        logger.info(f"[query_warranty]{user_id} {info['购买时间']} -> {info['状态']}")

    return "；".join(parts)

@tool(description="获取当前月份，返回 YYYY-MM 格式字符串（如 2026-09），用于报告生成的默认查询月份")
def get_current_month() -> str:
    """真实实现：返回系统当前年月，替代原随机数 mock"""
    return datetime.now().strftime("%Y-%m")

@tool(description="获取用户ID")
def get_user_id() -> str:
    """登录状态下直接返回当前登录用户的ID；未登录返回引导文案让 Agent 向用户询问"""
    current_user = _current_user_id()
    if current_user:
        return f"当前登录用户的ID为 {current_user}，查询使用记录或保修时直接使用该ID。"
    return "当前会话未接入登录系统，请向用户询问其4位数字用户ID后再查询使用记录。"

@tool(description="没有入参,返回值，调用后触发中间件自动为报告生成的场景注入上下文信息，为提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    #注意：工具名被 middleware.py 的 monitor_tool 硬编码匹配（触发报告提示词切换），
    #不可改名、不可加参数，返回语义保持原样
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m agent.tools.agent_tools
    print("存在记录:", fetch_external_data.invoke({"user_id": "2483", "month": "2025-04"}))
    print("无记录:", repr(fetch_external_data.invoke({"user_id": "2483", "month": "1999-01"})))
    print("非法ID:", fetch_external_data.invoke({"user_id": "abc", "month": "2025-04"}))
    print("非法月份:", fetch_external_data.invoke({"user_id": "2483", "month": "2025/04"}))
    print("保修-已过保:", query_warranty.invoke({"user_id": "2483", "purchase_month": "2025-04"}))
    print("保修-全部记录:", query_warranty.invoke({"user_id": "2483"}))
    print("当前月份:", get_current_month.invoke({}))
    print("用户ID引导:", get_user_id.invoke({}))
    print("上下文标记:", fill_context_for_report.invoke({}))