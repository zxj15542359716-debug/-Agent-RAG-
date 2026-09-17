#登录防爆破护栏
import threading
import time
from collections import defaultdict, deque

_WINDOW_SECONDS = 300      #失败统计的滑动窗口
_MAX_FAILURES = 5          #窗口内允许的失败次数上限
_LOCK_SECONDS = 900        #达到阈值后的锁定时长
_MAX_TRACKED_KEYS = 10000  #防御性上限：异常流量下限制内存占用


class LoginGuard:
    """滑动窗口失败计数 + 锁定（线程安全）"""

    def __init__(self, window: int = _WINDOW_SECONDS,
                 max_failures: int = _MAX_FAILURES,
                 lock_seconds: int = _LOCK_SECONDS,
                 max_keys: int = _MAX_TRACKED_KEYS) -> None:
        self._window = window
        self._max_failures = max_failures
        self._lock_seconds = lock_seconds
        self._max_keys = max_keys
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(user_id: str, client_ip: str) -> str:
        return f"{user_id}|{client_ip}"

    def check(self, user_id: str, client_ip: str) -> int:
        """检查是否处于锁定；返回剩余锁定秒数（未锁定返回 0）"""
        key = self._key(user_id, client_ip)
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            if until > now:
                return int(until - now) + 1
            if until:   #锁定已过期，清理
                del self._locked_until[key]
            return 0

    def record_failure(self, user_id: str, client_ip: str) -> None:
        """记录一次登录失败；窗口内失败数达到阈值则触发锁定"""
        key = self._key(user_id, client_ip)
        now = time.monotonic()
        with self._lock:
            q = self._failures[key]
            q.append(now)
            #滑出窗口的旧失败不计入
            while q and now - q[0] > self._window:
                q.popleft()
            if len(q) >= self._max_failures:
                self._locked_until[key] = now + self._lock_seconds
                q.clear()
            #防御性上限：键数量异常增长时丢弃最早插入的键
            if len(self._failures) > self._max_keys:
                self._failures.pop(next(iter(self._failures)))

    def reset(self, user_id: str, client_ip: str) -> None:
        """登录成功：清零该键的失败记录与锁定"""
        key = self._key(user_id, client_ip)
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


#模块级惰性单例：与 session_memory_service.py 的用法一致
_instance: LoginGuard | None = None


def get_login_guard() -> LoginGuard:
    """获取全局唯一的登录护栏实例（惰性创建）"""
    global _instance
    if _instance is None:
        _instance = LoginGuard()
    return _instance
