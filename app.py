#电脑外设智能售后 Web 前端
#【新增】FastAPI 单文件应用：内嵌聊天页面（HTML/CSS/JS）+ SSE 流式聊天接口
#运行方式：cd 项目根 && .venv/Scripts/python.exe app.py，浏览器访问 http://127.0.0.1:8618
#【修改】端口由 8000 改为 8618：8000 被系统进程占用且在 Windows 排除端口范围（winerror 10013）
#接口（【修改】除首页/登录/注册外，业务接口一律要求 Authorization: Bearer <token>，
#身份由服务端从 token 解析——原实现信任请求体 user_id，存在越权漏洞）：
#  GET  /            返回聊天页面
#  POST /api/login   登录成功返回 {"ok":true,"user_id":...,"token":...}（JWT，2小时有效）；
#                    同一 (用户ID, IP) 5 分钟内失败满 5 次锁定 15 分钟（防爆破）
#  POST /api/register 注册：系统分配4位用户ID（密码至少6位）
#  POST /api/report  售后上报（身份取自 token）
#  POST /api/chat    请求体 {"query": "...", "session_id": "..."}，SSE 流式返回事件：
#                  data: {"type":"tool","id":...,"content":...}  工具调用开始（显示为状态条）
#                  data: {"type":"tool_done","id":...}            工具执行完成（状态条变完成态）
#                  data: {"type":"text","content":...}  模型回答 token 分片（前端逐字渐现）
#                  data: {"type":"sources","items":[...]} 参考来源（第1步·引用溯源，回答后下发）
#                  data: {"type":"error","content":...} 出错提示
#                  data: {"type":"done"}                结束标记
import json
import re
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessageChunk, ToolMessage

#【第2步·2.1】对话入口由"单 ReAct"换成编排图（OrchestratorAgent）：
#编排图内部仍复用既有 ReactAgent 作为日常问答分支（行为零变化），并挂上 SqliteSaver 会话持久化
from agent.orchestration import OrchestratorAgent
from agent.orchestration.checkpoint import make_thread_id
from service.auth_service import create_token, get_current_user
from service.database_service import get_database_service
from service.login_guard import get_login_guard
from utils.logger_handler import logger

#会话ID白名单：前端为 crypto.randomUUID()（36 位十六进制+短横线）。
#【第2步·2.1】它会被拼进 checkpointer 的 thread_id，放任客户端传任意字符串
#等于把"会话键"的构造权交出去，因此非法值一律当"无会话"处理
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9\-]{8,64}")

def _safe_session_id(session_id: str) -> str:
    return session_id if session_id and _SESSION_ID_RE.fullmatch(session_id) else ""

app = FastAPI(title="电脑外设智能售后")

#工具名 -> 中文友好描述，前端以状态条展示，避免把原始 JSON 直接丢给用户看
_TOOL_LABELS = {
    "rag_summarize": "检索知识库并总结回答",
    "fetch_external_data": "查询用户外设使用记录",
    "query_warranty": "查询外设保修状态",
    "get_user_id": "获取用户ID",
    "get_current_month": "获取当前月份",
    "fill_context_for_report": "切换报告生成模式",
}

#售后上报允许的外设类型（与数据中的外设类型取值一致）
_ALLOWED_DEVICE_TYPES = ("键盘", "耳机", "鼠标")

#Agent 惰性单例：首次对话时才构建（构建图不发 API 请求，但避免重复构建）
_agent = None

def get_agent() -> OrchestratorAgent:
    global _agent
    if _agent is None:
        logger.info("[web]首次构建 OrchestratorAgent（编排图）")
        _agent = OrchestratorAgent()
    return _agent

class ChatRequest(BaseModel):
    query: str
    #会话ID：前端每个标签页生成一个（crypto.randomUUID），后端据此维护临时对话历史；
    #留空则不启用会话记忆（兼容直接发 {"query": "..."} 的旧前端）
    session_id: str = ""
    #【修改】原 user_id 字段已删除：用户身份改由 Authorization: Bearer <token> 提供
    #（见 get_current_user 依赖）——请求体自报身份正是越权漏洞的根源

def stream_events(query: str, session_id: str = "", user_id: str = ""):
    """以事件字典迭代 Agent 流式输出：区分工具调用状态与回答文本。

    【第2步·2.1 会话持久化】会话历史改由 LangGraph checkpointer 落 SQLite：
    按 thread_id="用户ID:会话ID" 读写（与原内存记忆同一个隔离键），本轮只发送新增的
    这一条用户消息，历史由 checkpointer 在服务端拼好——因此进程重启后同一会话仍能接上。
    session_id 为空或格式非法（旧前端）时用一次性线程，等价于改造前的"无记忆单轮问答"。
    已知行为差异：本轮用户消息在请求开始时就已入库，若中途出错，历史里会留下一条没有
    回答的提问（改造前是整轮都不写）；这是 checkpoint 机制的固有语义，重试不影响下一轮。

    【修改】stream_mode 由 values 改为 messages：模型输出按 token 逐块下发，
    前端实现打字机式流式显示；工具调用通过 tool_call_chunks 识别（首个带 name 的分片）。
    【修复流式失效】原实现把文本缓冲到消息边界才一次性下发，前端整段接收、
    没有打字机效果；改为逐 token 立即下发，浏览器能看到文字逐个出现。
    【修复重复回答】模型第一轮可能一边输出回答一边发起工具调用，工具返回后第二轮
    再输出最终回答，导致同一问题出现两遍答案。现在发现当前消息发起工具调用时，
    向客户端下发 clear 事件回滚该消息已流出的文字（前端清空气泡、重新显示思考圈），
    只保留工具状态条，最终回答照常流式展示。

    【新增】工具完成事件：工具执行完毕 ToolMessage 返回时下发 tool_done（带 tool_call_id），
    前端据此把对应状态条从"正在调用"改为"已调用"并停止呼吸动画。修复此前"结果已经
    出来但工具提示条仍在转"的显示问题（工具条创建后永不结束、done 事件前端未处理）。

    【第2步·2.1 子图流式】日常问答由编排图的 normal 节点委托给既有 ReAct Agent 执行，
    因此 stream 必须开 subgraphs=True，否则收不到子图的 token 分片（只剩一条聚合消息）。
    """
    agent = get_agent()
    #【第2步·2.1】会话线程键：与原 mem_key 同一个隔离键（用户ID:会话ID），
    #不同用户即使拿到同一 session_id 也读不到对方历史；无会话ID时用一次性线程保证"无记忆"语义
    thread_id = make_thread_id(user_id, _safe_session_id(session_id)) or f"ephemeral:{uuid4().hex}"
    #【第1步·1.4】溯源收集容器：挂在运行时上下文里，rag_summarize 工具检索后把来源追加进来，
    #流正常结束时统一下发（sources 事件）；工具在容器缺失时静默跳过
    sources_sink: list = []
    #【第2步·2.1】只发本轮新消息：历史由 checkpointer 从 SQLite 读回并与本轮合并
    input_dict = {"messages": [{"role": "user", "content": query}], "query": query}
    seen_tool_indices: set = set()   #记录已上报过的工具调用序号，避免一个调用报多次
    seen_tool_done_ids: set = set()  #记录已下发 tool_done 的 tool_call_id，避免工具返回分片时重复下发
    cur_msg_id = None                #当前消息 id（用于划分消息边界）
    cur_msg_has_tool = False         #当前消息是否发起过工具调用（是则其文本作废）

    def start_new_message() -> None:
        """消息边界：重置"本条消息是否发起过工具调用"标记
        （工具调用的 index 在每条消息内都从 0 计，跨消息不复位会把新调用误判为已上报）"""
        nonlocal cur_msg_has_tool
        cur_msg_has_tool = False

    try:
        #【第2步·2.3】双流模式：messages=模型分片（打字机/工具状态），custom=节点事件与报告
        #（node_start/node_end 来自 events.node_span，report/text 来自 synthesize）。
        #subgraphs=True 必须开（日常问答在子 Agent 里执行，不开收不到逐字分片）；
        #config 里的 thread_id 决定本次对话读写哪条会话（checkpointer 据此续接历史）
        #context 与 react_agent.execute_stream 保持一致（middleware 依赖 runtime.context["report"]）；
        #【新增】current_user 注入登录用户ID，工具据此默认查询登录者本人的数据
        for ns, mode, payload in agent.graph.stream(
            input_dict, stream_mode=["messages", "custom"], subgraphs=True,
            config={"configurable": {"thread_id": thread_id}},
            context={"report": False, "current_user": user_id, "sources": sources_sink}):
            #自定义事件直接透传：事件字典的形状本就是前端消费的形状
            #（node_start/node_end/report/text，见 agent/orchestration/events.py 的事件契约）
            if mode == "custom":
                if isinstance(payload, dict) and payload.get("type"):
                    yield payload
                continue

            chunk, meta = payload
            #【第2步·2.3】命名空间顶层段：'normal'=日常问答子图，'researcher'=报告分支的子研究者
            top = ns[0].split(":")[0] if ns else ""

            #工具执行完成（ToolMessage 返回）：下发 tool_done，前端把对应状态条标记为完成；
            #按 tool_call_id 去重（部分框架版本工具返回可能拆成多个分片）
            #【第2步·2.3】只处理日常问答的：研究者的工具调用不进聊天气泡（进度由时间线呈现）
            if isinstance(chunk, ToolMessage):
                if top != "normal":
                    continue
                tool_call_id = getattr(chunk, "tool_call_id", "")
                if tool_call_id and tool_call_id not in seen_tool_done_ids:
                    seen_tool_done_ids.add(tool_call_id)
                    yield {"type": "tool_done", "id": tool_call_id}
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue   #其余消息(HumanMessage等)不展示，原始数据在日志中可查
            #【修复重复回答的根因】rag_summarize 等工具内部会再次调用 LLM 生成总结，
            #其 token 会以 langgraph_node=tools 混入外层 messages 流，形成"工具内部回答"，
            #若不按节点过滤就会与模型节点的最终回答一起展示，造成两遍回答。只展示 model 节点。
            #【第2步·2.3 白名单修正】报告分支的三个子研究者，其 token 同样是 node=="model"
            #（只是命名空间是 researcher:xxx）——只按节点名过滤会把研究者的中间文本串进聊天气泡
            #（等于复活"重复回答"老 bug），因此必须同时限定"日常问答子图的 model 节点"。
            if not (meta.get("langgraph_node") == "model" and top == "normal"):
                continue

            #消息边界：结算上一条消息；工具调用的 index 在每条消息内都从0计，
            #跨消息会误判重复，此处清空去重集合，确保每条消息的工具调用都能上报
            if chunk.id and cur_msg_id is not None and chunk.id != cur_msg_id:
                start_new_message()
                seen_tool_indices.clear()
            if chunk.id:
                cur_msg_id = chunk.id

            #工具调用：分片流中 name 只出现在首个分片，按 (index) 去重后上报状态条；
            #本消息首次发现工具调用时回滚已流出的文字（模型"边答边调工具"的中间内容），
            #避免与工具返回后的最终回答重复展示两遍
            for tool_call_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                idx = tool_call_chunk.get("index")
                name = tool_call_chunk.get("name")
                if name and idx not in seen_tool_indices:
                    seen_tool_indices.add(idx)
                    if not cur_msg_has_tool:
                        cur_msg_has_tool = True   #标记本条消息带工具调用，后续文本不再下发
                        yield {"type": "clear"}   #通知前端清空当前气泡、重新显示思考圈
                    label = _TOOL_LABELS.get(name, name)
                    #带上 tool_call_id，前端据此把 tool_done 事件匹配到对应状态条
                    yield {"type": "tool", "id": tool_call_chunk.get("id", ""),
                           "content": f"正在调用工具：{label}"}

            #回答文本：逐 token 立即下发（打字机效果）；带工具调用的消息文本已作废，跳过
            content = chunk.content
            if isinstance(content, str) and content:
                if not cur_msg_has_tool:
                    yield {"type": "text", "content": content}
            elif isinstance(content, list):
                text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                if text and not cur_msg_has_tool:
                    yield {"type": "text", "content": text}

        #【第1步·1.4】引用溯源：把本次对话中检索到的来源统一下发，
        #前端在回答气泡下方渲染"参考来源"折叠列表（按 id 去重已在工具侧完成）
        if sources_sink:
            yield {"type": "sources", "items": sources_sink}
    except Exception as e:
        #异常只进日志，返回给前端的是友好提示
        logger.error(f"[web]对话流异常：{str(e)}", exc_info=True)
        yield {"type": "error", "content": "服务暂时不可用，请稍后再试。"}

class LoginRequest(BaseModel):
    user_id: str
    password: str

@app.post("/api/login")
def login(payload: LoginRequest, request: Request):
    """登录校验接口。

    用户ID必须是4位数字（与数据格式一致）；密码只校验是否与数据库中的密码匹配。
    【修改】① 校验通过签发 JWT（随响应返回 token），后续业务接口凭 token 识别身份；
    ② 接入登录护栏：同一 (用户ID, IP) 5 分钟内失败满 5 次锁定 15 分钟，防暴力枚举。
    """
    user_id = payload.user_id.strip()
    password = payload.password
    if not re.fullmatch(r"\d{4}", user_id):
        return {"ok": False, "message": "用户ID应为4位数字（如 2483）"}
    #密码不限制长度，仅要求非空
    if not password:
        return {"ok": False, "message": "密码不能为空"}
    #防爆破：先查锁定状态（未锁定返回 0）
    client_ip = request.client.host if request.client else "unknown"
    guard = get_login_guard()
    locked_seconds = guard.check(user_id, client_ip)
    if locked_seconds:
        logger.warning(f"[web]登录已锁定：{user_id}@{client_ip}，剩余 {locked_seconds}s")
        return {"ok": False, "message": f"尝试过于频繁，请 {locked_seconds} 秒后再试"}
    if get_database_service().verify_user(user_id, password):
        guard.reset(user_id, client_ip)
        return {"ok": True, "user_id": user_id, "token": create_token(user_id)}
    guard.record_failure(user_id, client_ip)
    return {"ok": False, "message": "用户不存在或密码错误"}

class RegisterRequest(BaseModel):
    password: str

@app.post("/api/register")
def register(payload: RegisterRequest):
    """注册接口：系统随机分配一个未被占用的4位用户ID，密码由用户设置。

    密码要求至少 6 位（【修改】原实现仅要求非空，1 位密码配合4位ID极易被穷举）；
    成功返回 {"ok": true, "user_id": ...}；失败返回中文提示。
    """
    password = payload.password
    if not password:
        return {"ok": False, "message": "密码不能为空"}
    #【新增】最小长度校验
    if len(password) < 6:
        return {"ok": False, "message": "密码长度至少 6 位"}
    try:
        user_id = get_database_service().create_user_auto(password)
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "message": str(e)}
    logger.info(f"[web]新用户注册，系统分配ID：{user_id}")
    return {"ok": True, "user_id": user_id}

class ReportRequest(BaseModel):
    #【修改】user_id 字段已删除：身份取自 Authorization: Bearer <token>
    device_type: str
    fault: str

@app.post("/api/report")
def report(payload: ReportRequest, user_id: str = Depends(get_current_user)):
    """售后上报接口：用户选择外设类型并填写故障描述，上报时间由服务器自动记录。

    身份来自登录 token（不再信任请求体）；保留外设类型白名单与故障描述非空校验；
    成功返回 {"ok": true, "report_id": ..., "time": ...}（time 为服务器记录的上报时间）。
    """
    device_type = payload.device_type.strip()
    fault = payload.fault.strip()
    if device_type not in _ALLOWED_DEVICE_TYPES:
        return {"ok": False, "message": "外设类型无效"}
    if not fault:
        return {"ok": False, "message": "请填写故障描述"}
    try:
        report_id = get_database_service().add_report(user_id, device_type, fault)
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    record = get_database_service().get_report(report_id)
    logger.info(f"[web]售后上报：用户{user_id} 上报{device_type}故障（单号#{report_id}）")
    return {"ok": True, "report_id": report_id, "time": record["上报时间"]}

@app.post("/api/chat")
def chat(payload: ChatRequest, user_id: str = Depends(get_current_user)):
    """SSE 流式聊天接口（身份取自登录 token）"""
    query = payload.query.strip()
    if not query:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    def gen():
        for event in stream_events(query, payload.session_id, user_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/", response_class=HTMLResponse)
def index():
    """聊天页面（HTML/CSS/JS 内嵌）"""
    return PAGE_HTML

# ---------------- 前端页面（小清新风格：薄荷绿渐变 + 毛玻璃卡片 + 动效） ----------------
PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>外设售后小助手</title>
<style>
  :root {
    --mint: #10b981; --mint-deep: #0d9d78; --sky: #38bdf8;
    --ink: #243b53; --sub: #7c8ea0; --line: #e3f0ec;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    height: 100vh; display: flex; justify-content: center; align-items: center;
    background: linear-gradient(160deg, #dff7ee 0%, #e9f6fb 45%, #eef3ff 100%);
    padding: 0;
  }
  #app {
    width: min(860px, 100vw); height: min(92vh, 100vh);
    display: flex; flex-direction: column;
    background: rgba(255,255,255,.72);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,.9);
    border-radius: 26px;
    box-shadow: 0 24px 70px rgba(13,157,120,.14);
    overflow: hidden;
    position: relative;   /* 登录浮层绝对定位的锚点 */
  }
  header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 24px 14px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 14px; font-size: 20px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #34d399, #38bdf8);
    box-shadow: 0 6px 16px rgba(16,185,129,.3);
  }
  .brand h1 { font-size: 18px; font-weight: 700; color: var(--ink); letter-spacing: .5px; }
  .brand p { font-size: 12px; color: var(--sub); margin-top: 2px; }
  .status {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; color: var(--mint-deep);
    background: rgba(16,185,129,.1); padding: 5px 12px; border-radius: 999px;
  }
  /* 登录浮层：覆盖整个应用，登录成功后才显示聊天界面 */
  #login-screen {
    position: absolute; inset: 0; z-index: 10;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,.86); backdrop-filter: blur(12px);
  }
  .login-card {
    width: min(340px, 86vw); padding: 34px 30px 28px;
    background: #fff; border: 1px solid var(--line); border-radius: 22px;
    box-shadow: 0 20px 50px rgba(13,157,120,.16); text-align: center;
  }
  .login-card .logo { margin: 0 auto 12px; }
  .login-card h2 { font-size: 17px; color: var(--ink); margin-bottom: 4px; }
  .login-card .tip { font-size: 12px; color: var(--sub); margin-bottom: 18px; }
  .login-card input {
    width: 100%; padding: 11px 15px; margin-bottom: 12px;
    border: 1.5px solid var(--line); border-radius: 12px; font-size: 14px;
    font-family: inherit; outline: none; transition: all .2s ease;
  }
  .login-card input:focus { border-color: var(--mint); box-shadow: 0 0 0 4px rgba(16,185,129,.12); }
  .login-card button {
    width: 100%; padding: 12px; margin-top: 4px; border: none; border-radius: 12px;
    background: linear-gradient(135deg, #34d399, #0ea5e9); color: #fff;
    font-size: 15px; font-family: inherit; font-weight: 600; cursor: pointer;
    box-shadow: 0 8px 18px rgba(16,185,129,.28); transition: all .18s ease;
  }
  .login-card button:hover:not(:disabled) { transform: translateY(-2px); }
  .login-card button:disabled { opacity: .6; cursor: not-allowed; }
  .login-err { min-height: 18px; font-size: 12.5px; color: #ef4444; margin-top: 10px; }
  /* 退出登录按钮：登录后才显示，鼠标悬停变红提示可点击 */
  #logout { display: none; cursor: pointer; }
  #logout:hover { background: rgba(239,68,68,.12); color: #ef4444; }
  /* 售后上报弹窗与上报成功提示页：与登录浮层同款毛玻璃，不上报时不打扰聊天 */
  #report-modal, #report-success-modal {
    position: absolute; inset: 0; z-index: 8;
    display: none; align-items: center; justify-content: center;
    background: rgba(255,255,255,.86); backdrop-filter: blur(12px);
  }
  .login-card select, .login-card textarea {
    width: 100%; padding: 11px 15px; margin-bottom: 12px;
    border: 1.5px solid var(--line); border-radius: 12px; font-size: 14px;
    font-family: inherit; outline: none; background: #fff; resize: none;
    transition: all .2s ease;
  }
  .login-card select:focus, .login-card textarea:focus { border-color: var(--mint); box-shadow: 0 0 0 4px rgba(16,185,129,.12); }
  .report-actions { display: flex; gap: 10px; }
  .report-actions button { flex: 1; }
  /* 弹窗里的次要按钮（取消） */
  .login-card button.ghost {
    background: #fff; color: var(--sub); border: 1.5px solid var(--line);
    box-shadow: none; font-weight: 500;
  }
  /* 注册成功/上报成功的绿色提示 */
  .login-ok { color: var(--mint-deep) !important; }
  /* 注册成功后展示用户ID的醒目提示框：登录前一直显示 */
  .reg-ok-box {
    margin: 0 0 12px; padding: 10px 14px;
    background: rgba(16,185,129,.08); border: 1px dashed var(--mint);
    border-radius: 12px; font-size: 13px; color: var(--mint-deep);
    text-align: center; line-height: 1.7;
  }
  .reg-ok-box b { font-size: 22px; letter-spacing: 3px; display: block; }
  .login-card a { color: var(--mint-deep); text-decoration: none; font-size: 12.5px; }
  .login-card a:hover { text-decoration: underline; }
  .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--mint);
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,.45); } 50% { box-shadow: 0 0 0 5px rgba(16,185,129,0); } }

  #chips { display: flex; gap: 8px; flex-wrap: wrap; padding: 4px 24px 14px; }
  .chip {
    font-size: 12.5px; padding: 7px 14px; border-radius: 999px; cursor: pointer;
    color: var(--mint-deep); background: rgba(255,255,255,.75);
    border: 1px solid var(--line); user-select: none; white-space: nowrap;
    transition: all .18s ease;
  }
  .chip:hover { transform: translateY(-2px); background: var(--mint); color: #fff; border-color: var(--mint); box-shadow: 0 6px 14px rgba(16,185,129,.25); }
  /* "新对话"按钮用虚线边框与其他快捷提问区分 */
  #new-chat { border-style: dashed; }

  #chat { flex: 1; overflow-y: auto; padding: 10px 24px 20px; scroll-behavior: smooth; }
  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-thumb { background: rgba(16,185,129,.25); border-radius: 3px; }

  .msg { display: flex; margin-bottom: 14px; animation: rise .35s ease both; }
  @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
  .bubble {
    max-width: 80%; padding: 11px 16px; border-radius: 18px;
    font-size: 14.5px; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
  }
  .msg.user { justify-content: flex-end; }
  .msg.user .bubble {
    background: linear-gradient(135deg, #34d399, #0ea5e9); color: #fff;
    border-bottom-right-radius: 6px; box-shadow: 0 8px 20px rgba(14,165,233,.22);
  }
  .msg.bot .bubble {
    background: #fff; color: var(--ink);
    border: 1px solid var(--line); border-bottom-left-radius: 6px;
    box-shadow: 0 4px 14px rgba(36,59,83,.05);
  }

  /* 流式文字逐字渐现 */
  .tok { animation: fadeIn .3s ease both; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  /* 流式输出时气泡末尾的闪烁光标 */
  .bubble.typing::after { content: "▍"; color: var(--mint); animation: blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* 思考中的旋转加载圈 */
  .thinking { display: flex; align-items: center; gap: 9px; color: var(--sub); font-size: 13.5px; }
  .spinner {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2.5px solid rgba(16,185,129,.2); border-top-color: var(--mint);
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .tool-line {
    display: flex; align-items: center; gap: 8px;
    color: var(--sub); font-size: 12px; margin: 2px 0 12px 4px;
    animation: rise .3s ease both;
  }
  .t-dot { width: 6px; height: 6px; border-radius: 50%; background: #93c5fd; animation: pulse 1.4s infinite; }
  /* 工具执行完成：圆点变绿并停止呼吸动画 */
  .tool-line.done .t-dot { background: var(--mint); animation: none; }
  .err { color: #ef4444; font-size: 13px; margin: 2px 0 12px 4px; }

  /* 【第1步·1.4】参考来源折叠列表：挂在回答消息下方，默认收起 */
  .sources-box { margin: -4px 0 14px 4px; max-width: 80%; }
  .sources-box details {
    background: rgba(16,185,129,.06); border: 1px dashed rgba(16,185,129,.45);
    border-radius: 10px; padding: 7px 12px; font-size: 12.5px; color: var(--ink);
  }
  .sources-box summary { cursor: pointer; color: var(--mint-deep); font-weight: 600; outline: none; }
  .source-item { margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--line); }
  .source-head { font-weight: 600; }
  .source-snippet { color: var(--sub); margin-top: 2px; line-height: 1.6; }

  #input-bar {
    display: flex; gap: 10px; align-items: center;
    padding: 14px 20px 18px;
  }
  #input {
    flex: 1; padding: 12px 18px; border-radius: 999px; font-size: 14.5px;
    border: 1.5px solid var(--line); background: rgba(255,255,255,.9);
    outline: none; font-family: inherit; color: var(--ink);
    transition: all .2s ease;
  }
  #input:focus { border-color: var(--mint); box-shadow: 0 0 0 4px rgba(16,185,129,.12); }
  #send {
    padding: 12px 26px; border: none; border-radius: 999px; cursor: pointer;
    background: linear-gradient(135deg, #34d399, #0ea5e9); color: #fff;
    font-size: 14.5px; font-family: inherit; font-weight: 600;
    box-shadow: 0 8px 18px rgba(16,185,129,.28);
    transition: all .18s ease;
  }
  #send:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 24px rgba(16,185,129,.34); }
  #send:active:not(:disabled) { transform: none; }
  #send:disabled { background: #c3d4d0; box-shadow: none; cursor: not-allowed; }

  @media (max-width: 640px) {
    #app { height: 100vh; border-radius: 0; }
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">
      <div class="logo">🎧</div>
      <div>
        <h1>外设售后小助手</h1>
        <p>故障排除 · 保养 · 选购 · 保修 · 报告</p>
      </div>
    </div>
    <span class="status"><span class="dot"></span>在线</span>
    <span class="status" id="logout">退出登录</span>
  </header>
  <div id="chips">
    <div class="chip">键盘按键失灵怎么办</div>
    <div class="chip">机械键盘怎么清洁保养</div>
    <div class="chip">打射击游戏怎么选耳机</div>
    <div class="chip">查一下我的保修状态</div>
    <div class="chip">生成我的使用报告</div>
    <div class="chip" id="new-chat">新对话</div>
    <div class="chip" id="report-btn">售后上报</div>
  </div>
  <div id="chat"></div>
  <div id="input-bar">
    <input id="input" type="text" placeholder="输入你的问题，回车发送…" autocomplete="off">
    <button id="send">发送</button>
  </div>
  <!-- 登录浮层：覆盖整个应用，校验通过前聊天界面不可交互 -->
  <div id="login-screen">
    <div class="login-card">
      <div class="logo">🎧</div>
      <h2>外设售后小助手</h2>
      <p class="tip">请先登录，新用户点击下方注册</p>
      <div id="login-form">
        <input id="login-id" type="text" placeholder="用户ID（4位数字，如 2483）" autocomplete="off">
        <input id="login-pwd" type="password" placeholder="密码" autocomplete="off">
        <button id="login-btn">登 录</button>
      </div>
      <div id="register-form" style="display:none">
        <input id="reg-pwd" type="password" placeholder="设置密码（至少6位）" autocomplete="off">
        <input id="reg-pwd2" type="password" placeholder="再次输入密码" autocomplete="off">
        <button id="register-btn">注 册</button>
      </div>
      <div class="login-err" id="login-err"></div>
      <!-- 注册成功后展示系统分配的用户ID：登录前一直显示，避免用户错过 -->
      <div class="reg-ok-box" id="reg-ok" style="display:none">
        注册成功！你的用户ID是 <b id="reg-ok-id"></b>
        <span>请牢记，已自动填入下方登录框，点击"登 录"即可进入</span>
      </div>
      <p class="tip" style="margin:0">
        <a href="javascript:void(0)" id="to-register">没有账号？注册新账号</a>
        <a href="javascript:void(0)" id="to-login" style="display:none">已有账号？返回登录</a>
      </p>
    </div>
  </div>
  <!-- 售后上报弹窗：选类型+填故障，上报时间由服务器自动记录 -->
  <div id="report-modal">
    <div class="login-card">
      <h2>售后上报</h2>
      <p class="tip">请选择外设类型并描述故障，系统将自动记录上报时间</p>
      <select id="report-type">
        <option value="键盘">键盘</option>
        <option value="耳机">耳机</option>
        <option value="鼠标">鼠标</option>
      </select>
      <textarea id="report-fault" rows="4" placeholder="请填写故障描述，如：键盘空格键回弹卡涩、鼠标左键单击变双击"></textarea>
      <div class="report-actions">
        <button id="report-cancel" class="ghost">取消</button>
        <button id="report-submit">提交上报</button>
      </div>
      <div class="login-err" id="report-msg"></div>
    </div>
  </div>
  <!-- 上报成功提示页：关闭后回到聊天界面，并自动把故障交给聊天模型获取解答 -->
  <div id="report-success-modal">
    <div class="login-card">
      <div class="logo">✅</div>
      <h2>上报成功</h2>
      <p class="tip" id="report-success-text"></p>
      <button id="report-success-close">获取解答</button>
    </div>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

//会话记忆：每个标签页持有一个随机会话ID，随每次请求发给后端，
//后端用它维护临时对话历史（最近10轮），让多轮问答上下文连贯
let sessionId = crypto.randomUUID();

//欢迎语（页面加载与点击"新对话"时复用）
const WELCOME_TEXT = '你好呀 👋 我是外设售后小助手，可以帮你解答耳机、键盘、鼠标、麦克风的故障排除、保养、选购、保修与使用报告问题。点击上方快捷提问或直接输入开始吧～';

// ---- 登录逻辑：校验通过前聊天界面被浮层遮挡，无法交互 ----
const loginScreen = document.getElementById('login-screen');
const loginId = document.getElementById('login-id');
const loginPwd = document.getElementById('login-pwd');
const loginBtn = document.getElementById('login-btn');
const loginErr = document.getElementById('login-err');
const logoutBtn = document.getElementById('logout');
let loggedUser = null;   //已登录的用户ID，未登录为 null
let authToken = null;    //【新增】登录签发的 JWT，仅存内存（刷新页面需重新登录）
//【新增】带身份凭证的请求头：业务接口（chat/report）一律携带 Bearer token
function authHeaders() {
  return { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authToken };
}

async function doLogin() {
  loginErr.className = 'login-err';
  loginErr.textContent = '';
  const user_id = loginId.value.trim();
  const password = loginPwd.value;
  loginBtn.disabled = true;
  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id, password }),
    });
    const data = await res.json();
    if (data.ok) {
      loggedUser = data.user_id;
      authToken = data.token || null;              //【新增】保存服务端签发的登录凭证
      loginScreen.style.display = 'none';          //登录成功：撤掉浮层，进入聊天界面
      logoutBtn.style.display = '';                //恢复 CSS 默认展示（.status 样式）
      logoutBtn.textContent = '已登录 ' + loggedUser + ' · 退出';
      loginPwd.value = '';
      input.focus();
    } else {
      loginErr.textContent = data.message || '登录失败';
    }
  } catch (e) {
    loginErr.textContent = '请求失败：' + e.message;
  } finally {
    loginBtn.disabled = false;
  }
}

loginBtn.addEventListener('click', doLogin);
//回车快捷操作：用户ID框回车跳到密码框，密码框回车直接登录
loginId.addEventListener('keydown', e => { if (e.key === 'Enter') loginPwd.focus(); });
loginPwd.addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });

//退出登录：回到登录浮层，换新会话ID并清空聊天记录
logoutBtn.addEventListener('click', () => {
  loggedUser = null;
  authToken = null;                              //【新增】清除登录凭证
  logoutBtn.style.display = 'none';
  loginScreen.style.display = 'flex';
  sessionId = crypto.randomUUID();
  chat.innerHTML = '';
  const fresh = addBotMsg(false);
  fresh.textContent = WELCOME_TEXT;
  loginId.focus();
});

// ---- 登录/注册表单切换 ----
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const toRegister = document.getElementById('to-register');
const toLogin = document.getElementById('to-login');
const regPwd = document.getElementById('reg-pwd');
const regPwd2 = document.getElementById('reg-pwd2');
const registerBtn = document.getElementById('register-btn');
const regOk = document.getElementById('reg-ok');   //注册成功后展示用户ID的提示框

function showLoginForm() {
  loginForm.style.display = ''; registerForm.style.display = 'none';
  toRegister.style.display = ''; toLogin.style.display = 'none';
  loginErr.className = 'login-err'; loginErr.textContent = '';
  loginId.focus();
}
function showRegisterForm() {
  loginForm.style.display = 'none'; registerForm.style.display = '';
  toRegister.style.display = 'none'; toLogin.style.display = '';
  loginErr.className = 'login-err'; loginErr.textContent = '';
  regOk.style.display = 'none';   //开始新的注册时收起上次的用户ID提示
  regPwd.focus();
}
toRegister.addEventListener('click', showRegisterForm);
toLogin.addEventListener('click', showLoginForm);

// ---- 注册：系统随机分配一个未被占用的4位用户ID，密码由用户设置（不限长度） ----
async function doRegister() {
  loginErr.className = 'login-err';
  loginErr.textContent = '';
  const password = regPwd.value;
  const password2 = regPwd2.value;
  if (!password) { loginErr.textContent = '密码不能为空'; return; }
  if (password.length < 6) { loginErr.textContent = '密码长度至少 6 位'; return; }
  if (password !== password2) { loginErr.textContent = '两次输入的密码不一致'; return; }
  registerBtn.disabled = true;
  try {
    const res = await fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    });
    const data = await res.json();
    if (data.ok) {
      //注册成功：把系统分配的ID醒目地展示在登录卡片上（登录前一直显示，不再一闪而过），
      //并把ID与密码自动填入登录表单，由用户确认后自行点击登录
      document.getElementById('reg-ok-id').textContent = data.user_id;
      regOk.style.display = '';
      loginId.value = data.user_id;
      loginPwd.value = password;
      regPwd.value = ''; regPwd2.value = '';
      showLoginForm();
      loginBtn.focus();
    } else {
      loginErr.textContent = data.message || '注册失败';
    }
  } catch (e) {
    loginErr.textContent = '请求失败：' + e.message;
  } finally {
    registerBtn.disabled = false;
  }
}
registerBtn.addEventListener('click', doRegister);
regPwd2.addEventListener('keydown', e => { if (e.key === 'Enter') doRegister(); });

// ---- 售后上报：弹窗选择外设类型+填写故障，上报时间由服务器自动记录 ----
const reportModal = document.getElementById('report-modal');
const reportType = document.getElementById('report-type');
const reportFault = document.getElementById('report-fault');
const reportMsg = document.getElementById('report-msg');
const reportSubmit = document.getElementById('report-submit');

document.getElementById('report-btn').addEventListener('click', () => {
  reportMsg.className = 'login-err';
  reportMsg.textContent = '';
  reportFault.value = '';
  reportModal.style.display = 'flex';   //弹窗展示，不跳转页面
  reportType.focus();
});
document.getElementById('report-cancel').addEventListener('click', () => {
  reportModal.style.display = 'none';
});
//最近一次成功上报的类型与故障描述（成功提示页关闭后用于自动咨询聊天模型）
let lastReportDevice = null;
let lastReportFault = null;
reportSubmit.addEventListener('click', async () => {
  const device_type = reportType.value;
  const fault = reportFault.value.trim();
  if (!fault) {
    reportMsg.className = 'login-err';
    reportMsg.textContent = '请填写故障描述';
    return;
  }
  reportSubmit.disabled = true;
  try {
    const res = await fetch('/api/report', {
      method: 'POST',
      headers: authHeaders(),
      //【修改】身份不再放请求体：user_id 由服务端从 Authorization 头解析
      body: JSON.stringify({ device_type, fault }),
    });
    if (res.status === 401) { handleAuthExpired('登录已过期，请重新登录'); return; }
    const data = await res.json();
    if (data.ok) {
      //上报成功：关闭上报弹窗，弹出独立的成功提示页（不在弹窗内提示）
      lastReportDevice = device_type;
      lastReportFault = fault;
      reportFault.value = '';
      reportModal.style.display = 'none';
      document.getElementById('report-success-text').textContent =
        '单号 #' + data.report_id + '，上报时间 ' + data.time + '，已为你记录。';
      document.getElementById('report-success-modal').style.display = 'flex';
    } else {
      reportMsg.className = 'login-err';
      reportMsg.textContent = data.message || '上报失败';
    }
  } catch (e) {
    reportMsg.className = 'login-err';
    reportMsg.textContent = '请求失败：' + e.message;
  } finally {
    reportSubmit.disabled = false;
  }
});

//成功提示页："获取解答"关闭提示、回到聊天页面，并把故障问题作为消息发给聊天模型
document.getElementById('report-success-close').addEventListener('click', () => {
  document.getElementById('report-success-modal').style.display = 'none';
  //构造问题交给聊天模型（rag_summarize 会检索知识库给出排查与解决建议）
  send('我上报了一个售后故障：' + lastReportDevice + '出现"' + lastReportFault +
       '"，请帮我分析可能的原因并给出排查解决建议。');
});

function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

function addUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = '<div class="bubble"></div>';
  div.querySelector('.bubble').textContent = text;
  chat.appendChild(div);
  scrollToBottom();
}

//新建客服消息气泡，thinking 为 true 时先展示旋转加载圈
function addBotMsg(thinking) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = '<div class="bubble"></div>';
  const bubble = div.querySelector('.bubble');
  if (thinking) bubble.innerHTML = '<div class="thinking"><span class="spinner"></span><span>正在思考…</span></div>';
  chat.appendChild(div);
  scrollToBottom();
  return bubble;
}

//逐 token 追加：每个分片一个 span，用 CSS 动画实现文字逐渐出现
function appendToken(bubble, text) {
  const span = document.createElement('span');
  span.className = 'tok';
  span.textContent = text;
  bubble.appendChild(span);
  scrollToBottom();
}

function addToolLine(text, toolId) {
  const div = document.createElement('div');
  div.className = 'tool-line';
  if (toolId) div.dataset.toolId = toolId;   //记录调用ID，工具完成事件据此匹配对应状态条
  div.innerHTML = '<span class="t-dot"></span>';
  const label = document.createElement('span');
  label.textContent = text;
  div.appendChild(label);
  chat.appendChild(div);
  scrollToBottom();
}

//把一条工具状态条标记为完成：绿点、停呼吸动画、"正在"改"已"
function markToolDone(line) {
  if (line.classList.contains('done')) return;
  line.classList.add('done');
  const label = line.lastElementChild;
  if (label) label.textContent = label.textContent.replace('正在调用工具：', '已调用工具：');
}
function finishToolLine(toolId) {
  chat.querySelectorAll('.tool-line[data-tool-id="' + toolId + '"]').forEach(markToolDone);
}
//流结束或出错兜底：所有未完成的工具条一律标记完成，避免呼吸动画一直转
function finishAllToolLines() {
  chat.querySelectorAll('.tool-line:not(.done)').forEach(markToolDone);
}

function addError(text) {
  const div = document.createElement('div');
  div.className = 'err';
  div.textContent = '⚠ ' + text;
  chat.appendChild(div);
  scrollToBottom();
}

//【新增】登录凭证失效（401）：退回登录浮层并提示重新登录
function handleAuthExpired(message) {
  loggedUser = null;
  authToken = null;
  logoutBtn.style.display = 'none';
  sessionId = crypto.randomUUID();
  chat.innerHTML = '';
  const fresh = addBotMsg(false);
  fresh.textContent = WELCOME_TEXT;
  loginScreen.style.display = 'flex';
  loginErr.className = 'login-err';
  loginErr.textContent = message || '登录已过期，请重新登录';
  loginId.focus();
}

//【第1步·1.4】渲染"参考来源"折叠列表：挂在对应回答消息下方，可展开查看来源与片段
function renderSources(bubble, items) {
  if (!items || !items.length) return;
  const msgEl = bubble.closest('.msg');
  if (!msgEl) return;
  const box = document.createElement('div');
  box.className = 'sources-box';
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = '参考来源（' + items.length + ' 条）';
  details.appendChild(summary);
  items.forEach((it) => {
    const row = document.createElement('div');
    row.className = 'source-item';
    const head = document.createElement('div');
    head.className = 'source-head';
    //来源内容全部走 textContent，不参与 HTML 拼接
    head.textContent = (it.source || '') + (it.entry ? ' · ' + it.entry : '') +
                       (it.title ? ' · ' + it.title : '');
    const snip = document.createElement('div');
    snip.className = 'source-snippet';
    snip.textContent = it.snippet || '';
    row.appendChild(head);
    row.appendChild(snip);
    details.appendChild(row);
  });
  box.appendChild(details);
  msgEl.insertAdjacentElement('afterend', box);
  scrollToBottom();
}

async function send(query) {
  if (!query.trim()) return;
  addUserMsg(query);
  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;

  const bubble = addBotMsg(true);   //先显示旋转加载圈
  let hasToken = false;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: authHeaders(),
      //【修改】身份不再放请求体：user_id 由服务端从 Authorization 头解析
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    if (res.status === 401) { handleAuthExpired('登录已过期，请重新登录'); return; }
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    //逐块读取 SSE 流，按 \\n\\n 分割事件
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === 'text') {
          if (!hasToken) {                     //首块文本：撤掉加载圈，进入流式输出
            bubble.querySelector('.thinking')?.remove();
            bubble.classList.add('typing');    //显示闪烁光标
            hasToken = true;
          }
          appendToken(bubble, ev.content);     //逐 token 渐现
        } else if (ev.type === 'tool') {
          addToolLine(ev.content, ev.id);      //工具调用状态条（记录调用ID，完成后按ID标记）
        } else if (ev.type === 'tool_done') {
          finishToolLine(ev.id);               //工具执行完成：对应状态条变完成态
        } else if (ev.type === 'clear') {
          //模型发起了工具调用：之前流出的文字是"边答边调工具"的中间内容，
          //清掉气泡里已显示的文字、重新显示思考圈，等最终回答流式写入
          bubble.querySelectorAll('.tok').forEach(s => s.remove());
          bubble.classList.remove('typing');
          hasToken = false;
          if (!bubble.querySelector('.thinking')) {
            bubble.innerHTML = '<div class="thinking"><span class="spinner"></span><span>正在思考…</span></div>';
          }
        } else if (ev.type === 'sources') {
          renderSources(bubble, ev.items);     //【第1步·1.4】渲染参考来源折叠列表
        } else if (ev.type === 'error') {
          addError(ev.content);
          finishAllToolLines();                //出错中断：剩余工具条不再等待
        }
        scrollToBottom();
      }
    }
    bubble.classList.remove('typing');
    finishAllToolLines();   //流正常结束：全部工具条标记完成（正常已被 tool_done 逐个标记，此处兜底）
    if (!hasToken) bubble.textContent = '（无回复内容）';
  } catch (e) {
    addError('请求失败：' + e.message);
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener('click', () => send(input.value));
input.addEventListener('keydown', e => { if (e.key === 'Enter') send(input.value); });
document.querySelectorAll('.chip').forEach(c =>
  c.addEventListener('click', () => send(c.textContent))
);

//欢迎语
const welcome = addBotMsg(false);
welcome.textContent = WELCOME_TEXT;

//"新对话"：换一个全新的会话ID（旧会话记忆留在后端内存，随 LRU 自动淘汰），
//清空页面气泡并重新显示欢迎语
document.getElementById('new-chat').addEventListener('click', () => {
  sessionId = crypto.randomUUID();
  chat.innerHTML = '';
  const fresh = addBotMsg(false);
  fresh.textContent = WELCOME_TEXT;
  input.focus();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import os

    import uvicorn
    #host 绑定 127.0.0.1 仅本机访问，如需局域网演示可改为 0.0.0.0
    #【新增】端口可用环境变量 APP_PORT 覆盖（默认 8618），便于与原目录的旧服务同时运行
    port = int(os.getenv("APP_PORT", "8618"))
    uvicorn.run(app, host="127.0.0.1", port=port)


# ============================================================================================
# 【第 1 步 · 1.4 说明】引用溯源的 Web 侧改造（本文件的改动说明）
# --------------------------------------------------------------------------------------------
# 改动点（共 4 处 + 前端样式/函数）：
#   1. stream_events 创建 sources_sink 列表并放进 runtime context（与 current_user 同款机制）；
#   2. rag_summarize 工具把检索来源追加进该列表（见 agent/tools/agent_tools.py 底部说明）；
#   3. 对话流正常结束后 yield {"type":"sources","items":[...]} —— 新增第 6 种 SSE 事件；
#   4. 前端新增 'sources' 分支 + renderSources()：在回答消息下方渲染"参考来源"折叠列表，
#      逐条展示 来源文件·条目号·标题·片段（全部 textContent 渲染，无 HTML 拼接）。
# 为什么在"流结束"而不是"工具完成"时下发：
#   一轮对话可能检索多次（报告模式会多轮调工具），结束时统一下发可按 id 去重、一次展示全量来源；
#   且来源天然属于"答案之后"的信息，跟在前端打字机效果之后展示体验更自然。
# 边界处理：
#   - 无检索（闲聊类问题）→ sources_sink 为空 → 不发 sources 事件，前端无变化；
#   - 异常中断 → 不走正常分支，不发来源（与"出错不写会话记忆"同款保守策略）；
#   - 来源内容来自自家知识库，仍统一用 textContent 渲染，杜绝注入类问题面。
# 与评测的关系：
#   sources 事件是 run_answer.py 做 groundedness 判分的间接依据（答案结论应能在来源中找到）——
#   这也是"可追溯"从展示需求升级为质量需求的地方。
# 验证方式：
#   启动服务后问一个知识库问题：气泡下方出现"参考来源（N 条）"，展开可见条目明细；
#   浏览器 DevTools 里可看到 data: {"type":"sources", ...} 事件。
# ============================================================================================

# ============================================================================================
# 【第 2 步 · 2.1 说明】会话持久化（本文件的改动说明）
# --------------------------------------------------------------------------------------------
# 改动点（共 5 处）：
#   1. 对话入口由 ReactAgent 换成 agent.orchestration.OrchestratorAgent（编排图）；编排图内部
#      仍复用既有 ReactAgent 跑日常问答，工具/提示词/中间件一字未动，问答行为零变化。
#   2. 会话历史不再走进程内 session_memory_service，改由 LangGraph checkpointer 落 SQLite：
#      thread_id = "用户ID:会话ID"（与原 mem_key 同一个隔离键），进程重启后同一会话可续聊。
#   3. 本轮只发送新增的那条用户消息（{messages:[{role:user,...}], query}），历史由
#      checkpointer 在服务端读回并合并——不再每轮重发全量历史。
#   4. 流式调用改为 agent.graph.stream(..., stream_mode="messages", subgraphs=True,
#      config={"configurable":{"thread_id":thread_id}})：元组形状变为 (命名空间, (分片, 元数据))；
#      subgraphs=True 是硬前提（日常问答在子 Agent 中执行，不开收不到逐字分片）。
#   5. 新增 _SESSION_ID_RE/_safe_session_id：会话ID 格式校验（前端为 crypto.randomUUID）。
# 为什么保留 sources/tool/tool_done/clear/text 事件语义不变：
#   第 1 步的引用溯源与前端打字机效果已评测/已验收，本次只换"历史从哪来"，不碰"回答怎么展示"。
# 边界与兜底：
#   - session_id 为空或非法（旧前端）→ 用一次性 ephemeral 线程，等价于改造前的无记忆单轮问答；
#     这些孤立线程不参与续聊，由 scripts/prune_checkpoints.py 定期清理。
#   - 编译期已知行为差异：用户消息在请求开始即入库，中途出错会在历史里留下一条没有回答的提问
#     （改造前是整轮不写）。这是 checkpoint 的固有语义，不影响下一轮对话。
# 与其它文件的关系：
#   agent/orchestration/{graph,nodes,checkpoint,state}.py 提供图与持久化；
#   service/session_memory_service.py 已停用（文件保留 + 顶部弃用说明 + 测试防回退）。
# 验证方式：
#   1) .venv/Scripts/python.exe -m pytest tests/test_checkpoint_memory.py tests/test_session_memory_deprecated.py
#   2) 起服务聊两句 → Ctrl+C 重启 → 同一标签页继续问"那键盘呢"，回答应能接上上文；
#      换账号用同一 session_id 提问，上下文应为空（隔离性）。
# ============================================================================================

