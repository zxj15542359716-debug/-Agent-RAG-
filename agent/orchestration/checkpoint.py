#会话持久化（第2步·2.1）——LangGraph 官方 SqliteSaver
#【新增】把会话状态从"进程内内存"改为"SQLite 持久化"：
#   1. 进程重启不丢会话（断点续跑）；
#   2. 同一 thread_id 的历史由 checkpointer 维护，app.py 不再需要每轮重发全量历史；
#   3. 为后续能力（人工中断、失败重放）留好地基。
#落库位置与业务库分离（config/orchestration.yml 的 checkpoint.db_path）：
#checkpoints/writes 等表的建表与结构由 Saver 自己管（setup()），与业务表混在一个库里，
#会让"业务数据迁移"和"Saver 升级"互相绑死，排查问题时也难分清是谁在写。
import os
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver

from utils.config_handler import orchestration_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

_lock = threading.Lock()
_saver: SqliteSaver | None = None


def _db_path() -> str:
    return get_abs_path(orchestration_config["checkpoint"]["db_path"])


def get_checkpointer() -> SqliteSaver:
    """checkpointer 惰性单例（双重检查锁）。

    与 agent_tools._get_rag_service() 同款写法：FastAPI 线程池下多个请求可能同时首跑，
    不加锁会各自建连接、各自 setup()。连接用 check_same_thread=False（官方示例即如此），
    并发安全由 SqliteSaver 内部的 threading.Lock + 下面的 WAL/busy_timeout 共同兜底。
    """
    global _saver
    if _saver is None:
        with _lock:
            if _saver is None:
                path = _db_path()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                conn = sqlite3.connect(path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")     #多线程读写不互斥
                conn.execute("PRAGMA busy_timeout=10000")   #与 DatabaseService 同款兜底
                saver = SqliteSaver(conn)
                saver.setup()                                #建表，幂等
                _saver = saver
                logger.info(f"[checkpoint]会话持久化已启用：{path}")
    return _saver


def make_thread_id(user_id: str, session_id: str) -> str:
    """会话线程键：`用户ID:会话ID`。

    与改造前 app.py 的 mem_key 完全同一个隔离键——一个用户即使拿到别人的 session_id，
    thread_id 前缀也不同，读不到对方历史（安全属性原样保留）。
    session_id 为空时返回空串，调用方据此退化为"无会话记忆"的单轮问答。
    """
    return f"{user_id}:{session_id}" if user_id and session_id else ""


# ============================================================================================
# 【第 2 步 · 2.1 说明】checkpoint 落库（agent/orchestration/checkpoint.py）
# --------------------------------------------------------------------------------------------
# 改动点：新增 SqliteSaver 惰性单例与 thread_id 生成函数，供 agent/orchestration/graph.py
# 编译父图时挂载、app.py 发起对话时取用。
# 为什么用官方 SqliteSaver 而不是自己写状态落库：checkpointer 不只是"存 messages"，
# 它要按 superstep 存 channel 版本、pending writes（人工中断）、待重放任务——自己实现
# 等于重写一遍 LangGraph 的持久化协议；用官方的还能白送断点续跑与后续的 interrupt 能力。
# 为什么独立库文件：见文件头部说明（表结构归 Saver 管，不与业务表耦合）。
# 边界与兜底：连接失败/建库失败会直接抛错（fail-fast，与第 0 步配置校验口径一致）——
# 会话持久化是第 2 步的核心能力，静默降级成内存态会让"重启续聊"静默失效，反而更难查。
# 与其它文件的关系：graph.py 调用 get_checkpointer()；app.py 调用 make_thread_id()；
# 维护脚本 scripts/prune_checkpoints.py 直接打开同一个库做清理。
# ============================================================================================
