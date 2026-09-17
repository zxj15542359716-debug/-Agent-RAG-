#临时会话记忆服务
import threading
from collections import OrderedDict, deque

#每个会话最多保留的对话轮数（1轮 = 用户1条 + 助手1条，即最多 2*MAX_TURNS 条消息）
MAX_TURNS = 10
#全局最多缓存的会话数，超出后按 LRU 淘汰最久未使用的会话
MAX_SESSIONS = 100
#单条消息内容的最大字符数，超长截断，避免极端输入撑爆内存
MAX_CONTENT_CHARS = 2000


class SessionMemoryService:
    """进程内临时会话记忆：session_id -> 消息列表（按时间顺序"""

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