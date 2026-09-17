#会话持久化库维护
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