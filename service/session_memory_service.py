#临时会话记忆服务
#【新增】让同一浏览器会话内的多轮对话共享上下文：模型能"记得"本会话前面聊过什么，
#回答更连贯。记忆只保存在本进程内存中（临时），服务重启即清空；不写数据库、不跨会话共享。
#仿照 external_record_service.py 的模式：模块级惰性单例 + 容量限制，供 app.py 聊天接口使用。
#
#【第2步·2.1 已弃用】聊天链路已不再使用本模块：会话历史改由 LangGraph checkpointer
#落 SQLite（agent/orchestration/checkpoint.py + graph.py 的 trim_history 节点），
#隔离键仍是"用户ID:会话ID"，但获得了重启续聊、断点续跑能力（内存版做不到）。
#为什么文件保留而不删除：① 它是第 0 步"会话按用户隔离"那次安全改造的载体与证据，
#删掉会让那段说明失去落点；② 本模块的 __main__ 自检仍可独立运行，便于对照理解；
#③ tests/test_session_memory_deprecated.py 会断言 app.py 不再引用它（防止悄悄回退）。
#何时可以删：确认不再需要上述三用途时（例如课程交付定稿后）。
import threading
from collections import OrderedDict, deque

#每个会话最多保留的对话轮数（1轮 = 用户1条 + 助手1条，即最多 2*MAX_TURNS 条消息）
MAX_TURNS = 10
#全局最多缓存的会话数，超出后按 LRU 淘汰最久未使用的会话
MAX_SESSIONS = 100
#单条消息内容的最大字符数，超长截断，避免极端输入撑爆内存
MAX_CONTENT_CHARS = 2000


class SessionMemoryService:
    """进程内临时会话记忆：session_id -> 消息列表（按时间顺序）

    - get_history 返回该会话最近 N 轮消息，由调用方拼进模型的 messages 输入
    - append_message 追加一条消息；deque 的 maxlen 自动丢弃最早的旧消息
    - OrderedDict 维护"最近使用"顺序，会话数超过 MAX_SESSIONS 时淘汰最久未用的
    """

    def __init__(self, max_turns: int = MAX_TURNS,
                 max_sessions: int = MAX_SESSIONS,
                 max_content_chars: int = MAX_CONTENT_CHARS) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._max_content_chars = max_content_chars
        #deque(maxlen=2*max_turns) 自带滑窗：装满后自动从左侧弹出最旧消息，
        #历史长度天然被限制在最近 max_turns 轮，控制模型输入 token 消耗
        self._sessions: OrderedDict[str, deque[dict]] = OrderedDict()
        #【修复并发隐患】FastAPI 多线程下多个会话同时读写，加锁保证
        #"查找-追加-LRU淘汰"这类多步操作不被其他线程穿插（GIL 只保证单步原子）
        self._lock = threading.Lock()

    def _touch(self, session_id: str) -> None:
        """标记会话为"最近使用"，LRU 淘汰时从最久未用的开始丢"""
        self._sessions.move_to_end(session_id)

    def _trim_content(self, content: str) -> str:
        """消息内容超长截断，防止极端输入占用过多内存"""
        if len(content) > self._max_content_chars:
            return content[:self._max_content_chars]
        return content

    def get_history(self, session_id: str) -> list[dict]:
        """返回该会话的对话历史（浅拷贝，调用方可安全拼接新消息）；会话不存在返回空列表"""
        with self._lock:
            if session_id in self._sessions:
                self._touch(session_id)
                return list(self._sessions[session_id])
            return []

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """向会话追加一条消息（role 为 user/assistant），并顺带完成 LRU 淘汰"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=2 * self._max_turns)
            self._sessions[session_id].append({"role": role, "content": self._trim_content(content)})
            self._touch(session_id)
            #会话数超限：丢最久未使用的（OrderedDict 最左侧）
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        """清空指定会话的记忆。前端"新对话"目前用换新ID实现，此处留作接口备用"""
        with self._lock:
            self._sessions.pop(session_id, None)


#模块级惰性单例：与 external_record_service.py 的用法一致
_instance: SessionMemoryService | None = None


def get_session_memory_service() -> SessionMemoryService:
    """获取全局唯一的会话记忆服务实例（惰性创建）"""
    global _instance
    if _instance is None:
        _instance = SessionMemoryService()
    return _instance


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m service.session_memory_service
    svc = get_session_memory_service()
    svc.append_message("s1", "user", "我的机械键盘按键失灵了")
    svc.append_message("s1", "assistant", "请先检查轴体是否有异物，再尝试拔插轴体……")
    print("会话历史:", svc.get_history("s1"))
    print("不存在的会话:", svc.get_history("s2"))
    svc.clear("s1")
    print("清空后:", svc.get_history("s1"))


# ============================================================================================
# 【第 2 步 · 2.1 说明】本文件在第 2 步的状态（已弃用，保留不删）
# --------------------------------------------------------------------------------------------
# 状态：聊天链路已不再引用本模块（app.py 的 import 已移除）。会话历史改由
# agent/orchestration/checkpoint.py 的 SqliteSaver 持久化，由 graph.py 的 trim_history
# 节点执行"保留最近 10 轮"（max_turns 与这里原 MAX_TURNS=10 对齐，换实现不换行为口径）。
# 为什么保留：① 第 0 步"会话按用户隔离"安全改造的落点与证据；② __main__ 自检可独立运行，
# 便于对照内存版与持久化版的差异；③ tests/test_session_memory_deprecated.py 断言
# app.py 不再引用它——文件在、但被测试盯住，防止有人悄悄改回内存态而无人察觉。
# 边界：本模块的 LRU 淘汰、内容截断等保护仍然有效，只是不再位于对话链路上。
# ============================================================================================
