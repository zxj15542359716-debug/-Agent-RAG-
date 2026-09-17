#会话持久化库维护（第2步·2.5）
#用途：控制 data/database/checkpoints.db 的体积——LangGraph 每个 superstep 都会写一条
#checkpoint（含完整 channel 值），长会话积少成多，需要定期只保留最近若干条。
#
#用法（cd 项目根）：
#  .venv/Scripts/python.exe -m scripts.prune_checkpoints --dry-run          # 只统计，不删除
#  .venv/Scripts/python.exe -m scripts.prune_checkpoints --keep 50          # 每个会话保留最近 50 条
#  .venv/Scripts/python.exe -m scripts.prune_checkpoints --drop-ephemeral   # 另清一次性的无会话线程
#
#说明：
#  - checkpoint 是"会话可续聊"的底座，删得过多会影响回看历史（但不会影响当前对话继续）；
#    默认保留 50 条 ≈ 十几轮对话的完整快照，对演示与课程场景足够。
#  - 一次性线程（thread_id 以 ephemeral: 开头）来自"无 session_id 的旧前端"，
#    用完即弃，可安全清掉。
import argparse
import os
import sqlite3
import sys

from utils.config_handler import orchestration_config
from utils.path_tool import get_abs_path

EPHEMERAL_PREFIX = "ephemeral:"


def _connect() -> sqlite3.Connection:
    path = get_abs_path(orchestration_config["checkpoint"]["db_path"])
    if not os.path.exists(path):
        raise RuntimeError(f"checkpoint 库不存在（还没跑过对话？）：{path}")
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def prune(conn: sqlite3.Connection, keep: int, drop_ephemeral: bool, dry_run: bool) -> dict:
    """按 thread 保留最近 keep 条；可选清掉 ephemeral 线程。返回统计信息。"""
    stats = {"threads": 0, "checkpoints": 0, "writes": 0, "ephemeral_threads": 0}

    threads = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    for thread_id in threads:
        if drop_ephemeral and thread_id.startswith(EPHEMERAL_PREFIX):
            stats["ephemeral_threads"] += 1
            stale_ids = [r[0] for r in conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id=?", (thread_id,))]
        else:
            #按 rowid 倒序 = 写入时间倒序（checkpoint_id 虽然也是时间序，但 rowid 更直白可靠）
            ids = [r[0] for r in conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC",
                (thread_id,))]
            stale_ids = ids[keep:]
        if not stale_ids:
            continue
        stats["threads"] += 1
        for cid in stale_ids:
            stats["writes"] += conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id=? AND checkpoint_id=?",
                (thread_id, cid)).fetchone()[0]
            if not dry_run:
                conn.execute("DELETE FROM writes WHERE thread_id=? AND checkpoint_id=?", (thread_id, cid))
                conn.execute("DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_id=?", (thread_id, cid))
            stats["checkpoints"] += 1
    if not dry_run:
        conn.commit()
        conn.execute("VACUUM")     #回收磁盘空间（写少读多的库值得压一次）
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="会话持久化库维护（保留最近 N 条 checkpoint）")
    parser.add_argument("--keep", type=int, default=50, help="每个会话保留最近 N 条（默认 50）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    parser.add_argument("--drop-ephemeral", action="store_true",
                        help="额外清掉 thread_id 以 ephemeral: 开头的一次性线程（无会话ID的旧前端产生）")
    args = parser.parse_args()

    try:
        conn = _connect()
    except RuntimeError as e:
        print(f"[跳过] {e}")
        return 0      #还没有会话库不算错误：维护脚本对"没数据"应当友好退出

    before = os.path.getsize(get_abs_path(orchestration_config["checkpoint"]["db_path"]))
    try:
        stats = prune(conn, max(1, args.keep), args.drop_ephemeral, args.dry_run)
    finally:
        conn.close()
    after = os.path.getsize(get_abs_path(orchestration_config["checkpoint"]["db_path"]))

    print("=" * 62)
    print(f"{'（演练，未删除）' if args.dry_run else '（已执行删除）'}")
    print(f"涉及会话数    : {stats['threads']}")
    print(f"清理 checkpoint: {stats['checkpoints']}")
    print(f"清理 writes    : {stats['writes']}")
    print(f"一次性线程     : {stats['ephemeral_threads']}")
    print(f"库体积         : {before / 1024:.1f} KB -> {after / 1024:.1f} KB")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================================
# 【第 2 步 · 2.5 说明】会话库维护脚本（scripts/prune_checkpoints.py）
# --------------------------------------------------------------------------------------------
# 为什么需要：checkpointer 每个 superstep 写一条完整快照（含 messages 等 channel 值），
# 长会话/多次演示后库会变大；保留策略按"每个会话最近 N 条"最直观，也不影响当前对话继续。
# 为什么放行 "--dry-run"：删数据是不可逆操作，默认先让人看清会删多少（本项目的运维惯例）。
# 为什么单独处理 ephemeral 线程：无 session_id 的旧前端会为每次请求新建一次性线程，
# 这些线程永远不会被续聊，属于纯垃圾；但默认不删（要显式加 --drop-ephemeral）。
# 与其它文件的关系：库路径取自 config/orchestration.yml 的 checkpoint.db_path，
# 与 agent/orchestration/checkpoint.py 使用同一个库；删的是 LangGraph 自己管理的表。
# ============================================================================================
