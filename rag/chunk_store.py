#chunk 账本（SQLite）——第1步·检索纵深
#【新增】记录每个入库 chunk 的原文与元数据，一份数据供四处使用：
#1. BM25 关键词检索的数据源（混合检索的另一路召回，jieba 分词后建 BM25Okapi 索引）；
#2. 引用溯源（SSE sources 事件）的取数来源（来源文件/条目号/标题/片段）；
#3. 检索评测的对照表（按 来源#条目号 判命中，见 eval/run_retrieval.py）；
#4. 内容级去重的全局账本（content_md5 唯一索引，跨文件、跨批次生效）。
#存放位置与向量索引同目录（faiss.db/chunks.db），随索引一起重建、随 faiss.db/ 一起被 gitignore。
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

from utils.config_handler import faiss_config
from utils.path_tool import get_abs_path

#表结构说明：
#- id：主键，`来源#条目号`（与 chunker.Chunk.id 一致，跨重建稳定）
#- content_md5：唯一索引——同内容 chunk 全库只存一条，插入用 INSERT OR IGNORE
#- embedding_model：记录入库时使用的向量模型名，供索引版本检测与问题排查
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
    """chunk 账本：入库写入、检索读取、去重判重。

    与 database_service 同款连接策略：每次操作独立开连接、事务结束即关闭
    （sqlite3 连接非线程安全，FastAPI 多线程下按操作开合最稳妥）。
    """

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
        """批量写入 chunk（INSERT OR IGNORE：内容重复的自动跳过），返回实际写入条数"""
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
        """该内容是否已在账本中（跨文件内容级去重用）"""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT 1 FROM kb_chunks WHERE content_md5 = ? LIMIT 1",
                (content_md5,)).fetchone()
        return row is not None

    def all_chunks(self) -> list[dict[str, Any]]:
        """全量读出（BM25 索引构建用；条目 ~1500 条，内存开销可忽略）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, text, source, entry_no, category, title FROM kb_chunks"
                " ORDER BY source, chunk_idx").fetchall()
        return [dict(r) for r in rows]

    def get_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """按 id 批量取（溯源展示用）"""
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
        """各来源文件的 chunk 数（重建后核对用）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM kb_chunks GROUP BY source ORDER BY source").fetchall()
        return {r["source"]: r["n"] for r in rows}

    def clear(self) -> None:
        """清空账本（全量重建索引时调用）"""
        with self._lock, self._tx() as conn:
            conn.execute("DELETE FROM kb_chunks")


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.chunk_store
    store = ChunkStore()
    print("chunk 总数:", store.count())
    print("各来源:", store.stats_by_source())


# ============================================================================================
# 【第 1 步 · 1.1 说明】chunk 账本（本文件的作用与设计）
# --------------------------------------------------------------------------------------------
# 为什么需要它：
#   原来入库的 chunk 只存在于 FAISS 索引内部（向量 + 文本混在 index.pkl 里），
#   想用它们做别的事（关键词检索、引用溯源、评测对照、内容去重）都没有稳定的数据出口。
#   把 chunk 同步存一份到 SQLite 账本后，索引与账本各司其职：
#     FAISS 管"向量召回"，账本管"元数据/关键词/溯源/去重"。
# 为什么用 SQLite 而不是新起服务：
#   与业务库同一技术选型（零部署、单文件、标准库自带），且天然支持唯一索引做去重；
#   数据量与索引同生命周期，放在 faiss.db/ 目录随索引一起重建、一起忽略提交。
# 表结构：
#   kb_chunks(id 主键=来源#条目号, text 原文, source 来源文件, entry_no 条目号,
#             category 品类, title 标题, chunk_idx 文件内序, embedding_model 入库模型, content_md5 内容指纹)
#   - UNIQUE(content_md5)：跨文件、跨批次的内容级去重（PDF/TXT 重复在此被拦下）；
#   - INDEX(source)：按来源统计与排查。
# 与其它组件的关系：
#   chunker.py（切分产出）→ 本账本（持久化）→ vector_store.py（向量入库）
#                                          ↘ hybrid_retriever.py（BM25 路 + 溯源取数）
#                                          ↘ eval/（判命中对照）
# 验证方式：
#   scripts/reindex.py --rebuild 后 stats_by_source() 各文件条数应与切分器自检一致；
#   重复执行 reindex（增量）时 count() 不再增长（说明去重与增量生效）。
# ============================================================================================
