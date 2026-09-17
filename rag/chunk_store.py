
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

from utils.config_handler import faiss_config
from utils.path_tool import get_abs_path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_chunks (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    source          TEXT NOT NULL,
    entry_no        TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    chunk_idx       INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT NOT NULL DEFAULT '',
    content_md5     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_chunks_md5 ON kb_chunks(content_md5);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_source ON kb_chunks(source);
"""


class ChunkStore:


    def __init__(self, db_path: str | None = None) -> None:
        #路径默认取配置（faiss.yml 的 chunk_store_path），测试可传临时路径
        self._db_path = db_path or get_abs_path(
            faiss_config.get("chunk_store_path", "faiss.db/chunks.db"))
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    @contextmanager
    def _tx(self):
        conn = self._conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def upsert_many(self, chunks: list, embedding_model: str = "") -> int:
        added = 0
        with self._lock, self._tx() as conn:
            for c in chunks:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO kb_chunks"
                    " (id, text, source, entry_no, category, title, chunk_idx, embedding_model, content_md5)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (c.id, c.text, c.source, c.entry_no, c.category, c.title,
                     c.chunk_idx, embedding_model, c.content_md5))
                added += cur.rowcount
        return added

    def has_content_md5(self, content_md5: str) -> bool:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT 1 FROM kb_chunks WHERE content_md5 = ? LIMIT 1",
                (content_md5,)).fetchone()
        return row is not None

    def all_chunks(self) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, text, source, entry_no, category, title FROM kb_chunks"
                " ORDER BY source, chunk_idx").fetchall()
        return [dict(r) for r in rows]

    def get_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT * FROM kb_chunks WHERE id IN ({placeholders})", ids).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def count(self) -> int:
        with self._tx() as conn:
            return conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]

    def stats_by_source(self) -> dict[str, int]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM kb_chunks GROUP BY source ORDER BY source").fetchall()
        return {r["source"]: r["n"] for r in rows}

    def clear(self) -> None:
        with self._lock, self._tx() as conn:
            conn.execute("DELETE FROM kb_chunks")


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.chunk_store
    store = ChunkStore()
    print("chunk 总数:", store.count())
    print("各来源:", store.stats_by_source())