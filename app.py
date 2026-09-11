#电脑外设智能售后 Web 前端
#【新增】FastAPI 单文件应用：内嵌聊天页面（HTML/CSS/JS）+ SSE 流式聊天接口
#运行方式：cd 项目根 && .venv/Scripts/python.exe app.py，浏览器访问 http://127.0.0.1:8618
#【修改】端口由 8000 改为 8618：8000 被系统进程占用且在 Windows 排除端口范围（winerror 10013）
#接口：
#  GET  /          返回聊天页面
#  POST /api/chat  请求体 {"query": "..."}，SSE 流式返回事件：
#                  data: {"type":"tool","id":...,"content":...}  工具调用开始（显示为状态条）
#                  data: {"type":"tool_done","id":...}            工具执行完成（状态条变完成态）
#                  data: {"type":"text","content":...}  模型回答 token 分片（前端逐字渐现）
#                  data: {"type":"error","content":...} 出错提示
#                  data: {"type":"done"}                结束标记
import json
import re

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessageChunk, ToolMessage

from agent.react_agent import ReactAgent
from service.database_service import get_database_service
from service.session_memory_service import get_session_memory_service
from utils.logger_handler import logger

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

def get_agent() -> ReactAgent:
    global _agent
    if _agent is None:
        logger.info("[web]首次构建 ReactAgent")
        _agent = ReactAgent()
    return _agent

class ChatRequest(BaseModel):
    query: str
    #会话ID：前端每个标签页生成一个（crypto.randomUUID），后端据此维护临时对话历史；
    #留空则不启用会话记忆（兼容直接发 {"query": "..."} 的旧前端）
    session_id: str = ""
    #登录用户ID：注入运行时上下文，工具查询使用记录/保修时默认绑定该用户（防越权）；
    #留空表示未登录，数据查询退回"由模型索要ID"的旧行为
    user_id: str = ""

def stream_events(query: str, session_id: str = "", user_id: str = ""):
    """以事件字典迭代 Agent 流式输出：区分工具调用状态与回答文本。

    【新增】临时会话记忆：按 session_id 从内存会话存储取出最近 N 轮历史，
    与本次问题拼成完整 messages 一起送入模型，多轮对话因此更连贯（如"那键盘呢"）；
    流正常结束后把本轮问答写回存储，出错则不写，重试时不带残缺上下文。

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
    """
    agent = get_agent()
    memory = get_session_memory_service()
    #历史消息 + 本次问题拼成完整输入；session_id 为空（旧前端）则退化为无记忆单轮问答
    history = memory.get_history(session_id) if session_id else []
    input_dict = {"messages": [*history, {"role": "user", "content": query}]}
    seen_tool_indices: set = set()   #记录已上报过的工具调用序号，避免一个调用报多次
    seen_tool_done_ids: set = set()  #记录已下发 tool_done 的 tool_call_id，避免工具返回分片时重复下发
    cur_msg_id = None                #当前消息 id（用于划分消息边界）
    cur_msg_has_tool = False         #当前消息是否发起过工具调用（是则其文本作废）
    cur_msg_text: list[str] = []     #当前消息已下发的文本（消息结束时判断是否计入最终回答）
    answer_parts: list[str] = []     #收集最终回答文本（正常结束后写入会话记忆）

    def settle_message() -> None:
        """消息结束时结算：带工具调用的消息文本作废；否则计入最终回答"""
        nonlocal cur_msg_text, cur_msg_has_tool
        if cur_msg_text and not cur_msg_has_tool:
            answer_parts.append("".join(cur_msg_text))
        cur_msg_text = []
        cur_msg_has_tool = False

    try:
        #context 与 react_agent.execute_stream 保持一致（middleware 依赖 runtime.context["report"]）；
        #【新增】current_user 注入登录用户ID，工具据此默认查询登录者本人的数据
        for chunk, meta in agent.agent.stream(
            input_dict, stream_mode="messages",
            context={"report": False, "current_user": user_id}):
            #工具执行完成（ToolMessage 返回）：下发 tool_done，前端把对应状态条标记为完成；
            #按 tool_call_id 去重（部分框架版本工具返回可能拆成多个分片）
            if isinstance(chunk, ToolMessage):
                tool_call_id = getattr(chunk, "tool_call_id", "")
                if tool_call_id and tool_call_id not in seen_tool_done_ids:
                    seen_tool_done_ids.add(tool_call_id)
                    yield {"type": "tool_done", "id": tool_call_id}
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue   #其余消息(HumanMessage等)不展示，原始数据在日志中可查
            #【修复重复回答的根因】rag_summarize 等工具内部会再次调用 LLM 生成总结，
            #其 token 会以 langgraph_node=tools 混入外层 messages 流，形成"工具内部回答"，
            #若不按节点过滤就会与模型节点的最终回答一起展示，造成两遍回答。只展示 model 节点
            if meta.get("langgraph_node") != "model":
                continue

            #消息边界：结算上一条消息；工具调用的 index 在每条消息内都从0计，
            #跨消息会误判重复，此处清空去重集合，确保每条消息的工具调用都能上报
            if chunk.id and cur_msg_id is not None and chunk.id != cur_msg_id:
                settle_message()
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
                        cur_msg_text.clear()      #本消息已下发的文本作废
                        yield {"type": "clear"}   #通知前端清空当前气泡、重新显示思考圈
                    label = _TOOL_LABELS.get(name, name)
                    #带上 tool_call_id，前端据此把 tool_done 事件匹配到对应状态条
                    yield {"type": "tool", "id": tool_call_chunk.get("id", ""),
                           "content": f"正在调用工具：{label}"}

            #回答文本：逐 token 立即下发（打字机效果）；带工具调用的消息文本已作废，跳过
            content = chunk.content
            if isinstance(content, str) and content:
                if not cur_msg_has_tool:
                    cur_msg_text.append(content)
                    yield {"type": "text", "content": content}
            elif isinstance(content, list):
                text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                if text and not cur_msg_has_tool:
                    cur_msg_text.append(text)
                    yield {"type": "text", "content": text}

        #流结束：结算最后一条消息
        settle_message()
    except Exception as e:
        #异常只进日志，返回给前端的是友好提示
        logger.error(f"[web]对话流异常：{str(e)}", exc_info=True)
        yield {"type": "error", "content": "服务暂时不可用，请稍后再试。"}
    else:
        #【新增】整段对话正常结束才写入记忆：出错或前端中断（GeneratorExit）时不保存，
        #避免把没有回答的半截对话留给下一轮；无最终回答文本同样不保存
        if answer_parts and session_id:
            memory.append_message(session_id, "user", query)
            #多段文本直接拼接为一条助手消息（正常流程只有最终回答这一段）
            memory.append_message(session_id, "assistant", "".join(answer_parts))

class LoginRequest(BaseModel):
    user_id: str
    password: str

@app.post("/api/login")
def login(payload: LoginRequest):
    """登录校验接口。

    用户ID必须是4位数字（与数据格式一致）；密码不限长度、不做格式限制，
    只校验是否与数据库中的密码匹配（演示数据默认密码 1111）。
    校验通过返回 {"ok": true, "user_id": ...}；失败返回中文提示。
    """
    user_id = payload.user_id.strip()
    password = payload.password
    if not re.fullmatch(r"\d{4}", user_id):
        return {"ok": False, "message": "用户ID应为4位数字（如 2483）"}
    #密码不限制长度，仅要求非空
    if not password:
        return {"ok": False, "message": "密码不能为空"}
    if get_database_service().verify_user(user_id, password):
        return {"ok": True, "user_id": user_id}
    return {"ok": False, "message": "用户不存在或密码错误"}

class RegisterRequest(BaseModel):
    password: str

@app.post("/api/register")
def register(payload: RegisterRequest):
    """注册接口：系统随机分配一个未被占用的4位用户ID，密码由用户设置。

    密码不限长度、不做格式限制，仅要求非空；
    成功返回 {"ok": true, "user_id": ...}；失败返回中文提示。
    """
    password = payload.password
    if not password:
        return {"ok": False, "message": "密码不能为空"}
    try:
        user_id = get_database_service().create_user_auto(password)
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "message": str(e)}
    logger.info(f"[web]新用户注册，系统分配ID：{user_id}")
    return {"ok": True, "user_id": user_id}

class ReportRequest(BaseModel):
    user_id: str
    device_type: str
    fault: str

@app.post("/api/report")
def report(payload: ReportRequest):
    """售后上报接口：用户选择外设类型并填写故障描述，上报时间由服务器自动记录。

    校验：用户ID为4位数字且存在、外设类型在白名单内、故障描述非空；
    成功返回 {"ok": true, "report_id": ..., "time": ...}（time 为服务器记录的上报时间）。
    """
    user_id = payload.user_id.strip()
    device_type = payload.device_type.strip()
    fault = payload.fault.strip()
    if not re.fullmatch(r"\d{4}", user_id):
        return {"ok": False, "message": "用户ID应为4位数字"}
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
def chat(payload: ChatRequest):
    """SSE 流式聊天接口"""
    query = payload.query.strip()
    if not query:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    def gen():
        for event in stream_events(query, payload.session_id, payload.user_id):
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
      <p class="tip">请先登录（老用户默认密码 1111），新用户点击下方注册</p>
      <div id="login-form">
        <input id="login-id" type="text" placeholder="用户ID（4位数字，如 2483）" autocomplete="off">
        <input id="login-pwd" type="password" placeholder="密码" autocomplete="off">
        <button id="login-btn">登 录</button>
      </div>
      <div id="register-form" style="display:none">
        <input id="reg-pwd" type="password" placeholder="设置密码（不限长度）" autocomplete="off">
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: loggedUser, device_type, fault }),
    });
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId, user_id: loggedUser }),
    });
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
    import uvicorn
    #host 绑定 127.0.0.1 仅本机访问，如需局域网演示可改为 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8618)
