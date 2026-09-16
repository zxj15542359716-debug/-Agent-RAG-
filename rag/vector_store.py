#向量库服务（FAISS）——第1步·检索纵深（改造版）
#【第1步·1.1 改造】相对原版的四点变化：
#1. 切分由"固定 200 字/20 重叠"改为"按条目切分"（rag/chunker.py），chunk 带元数据；
#2. 新增 chunk 账本（rag/chunk_store.py）：内容级去重 + BM25 数据源 + 引用溯源 + 评测对照；
#3. 新增索引版本检测（faiss.db/index_meta.json）：记录 embedding 模型与切分配置，
#   加载时校验——换模型后旧索引不再被静默使用（原实现会返回垃圾结果且无任何提示）；
#4. 新增全量重建入口（rebuild()，配合 scripts/reindex.py 使用）。
import json
import os
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from model.fectory import embed_model
from rag.chunker import dedupe, split_text
from rag.chunk_store import ChunkStore
from utils.config_handler import faiss_config, rag_config
from utils.file_handler import get_file_md5_H, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

#索引结构版本：切分策略或索引组织方式变化时递增，强制重建（见 _check_meta）
_SCHEMA_VERSION = 1


class VectorStoreService:
    def __init__(self):
        # FAISS 的持久化目录与文件（本工程所有向量序列化后存进这一个文件）
        self.index_dir = get_abs_path(faiss_config["persist_directory"])
        self.index_file = os.path.join(self.index_dir, "index.pkl")
        self.meta_file = os.path.join(self.index_dir, "index_meta.json")
        # chunk 账本（与索引同目录，随索引一起重建）
        self.chunk_store = ChunkStore()

        # 本地已有索引则加载；版本检测不通过时【不炸进程】——记录错误置空，
        # 检索时报出带修复指引的明确错误，reindex --rebuild 可修复
        self.load_error: str | None = None
        if os.path.exists(self.index_file):
            try:
                self._load_index()
            except RuntimeError as e:
                logger.error(f"[向量库]{e}")
                self.load_error = str(e)
                self.vector_store = None
        else:
            self.vector_store = None

    # ---------------- 索引元数据与版本检测 ----------------

    def _current_meta(self) -> dict:
        """本次构建的索引元数据（写入 index_meta.json，加载时逐项校验）"""
        return {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": rag_config["embedding_model_name"],
            "chunk_strategy": "entry-based",
            "max_entry_chars": faiss_config.get("max_entry_chars", 500),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _check_meta(self) -> None:
        """加载前版本检测：元数据缺失（旧版索引）或 embedding 模型变更 → 拒绝加载。

        【修复静默故障】原实现换 embedding 模型后旧 index.pkl 照常加载，向量空间不一致
        时不报错、只是静默返回错误结果；现在直接给出可执行的修复指引（--rebuild）。
        """
        if not os.path.exists(self.meta_file):
            raise RuntimeError(
                "索引缺少版本信息（index_meta.json），多为改造前构建的旧索引。"
                "请运行 python -m scripts.reindex --rebuild 重建后再使用。")
        with open(self.meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError(
                f"索引结构版本不匹配（索引 {meta.get('schema_version')} / 当前 {_SCHEMA_VERSION}），"
                "请运行 python -m scripts.reindex --rebuild 重建。")
        if meta.get("embedding_model") != rag_config["embedding_model_name"]:
            raise RuntimeError(
                f"embedding 模型已变更（索引 {meta.get('embedding_model')} → 当前 "
                f"{rag_config['embedding_model_name']}），新旧向量空间不兼容。"
                "请运行 python -m scripts.reindex --rebuild 重建索引。")

    # ---------------- 索引读写 ----------------

    def _save_index(self):
        """保存索引：FAISS 自带 save_local 走 C++ 文件读写、不支持中文路径，
        故用 Python 原生 open 写序列化字节（沿用原实现）；同时写入版本元数据"""
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.index_file, "wb") as f:
            f.write(self.vector_store.serialize_to_bytes())
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self._current_meta(), f, ensure_ascii=False, indent=2)

    def _load_index(self):
        """加载索引（含版本检测）；allow_dangerous_deserialization 为加载本机 pickle 所必需"""
        self._check_meta()
        with open(self.index_file, "rb") as f:
            data = f.read()
        self.vector_store = FAISS.deserialize_from_bytes(
            data, embed_model, allow_dangerous_deserialization=True)

    def ensure_loaded(self) -> None:
        """确保索引可用；不可用时抛出带修复指引的 RuntimeError（供检索链路调用）"""
        if self.vector_store is not None:
            return
        if os.path.exists(self.index_file):
            self._load_index()   # 版本不符会在此抛出明确错误
            self.load_error = None
            return
        raise RuntimeError(
            "向量索引不存在。首次使用请运行 python -m scripts.reindex 构建知识库索引。")

    def index_size(self) -> int:
        """索引内向量条数（重建核对用）"""
        return int(self.vector_store.index.ntotal) if self.vector_store is not None else 0

    # ---------------- 检索接口 ----------------

    def get_retriver(self):
        """兼容旧调用的纯向量检索器；索引不可用时返回 None（调用方有兜底文案）"""
        try:
            self.ensure_loaded()
        except RuntimeError as e:
            logger.warning(f"[获取检索器]{e}")
            return None
        return self.vector_store.as_retriever(search_kwargs={"k": faiss_config["k"]})

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        """向量召回（带分数）——混合检索的向量路入口。

        注意：FAISS 返回的是 L2 距离（越小越相似），距离→相似度的换算
        统一在 rag/hybrid_retriever.py 中处理，避免各处重复踩坑。
        """
        self.ensure_loaded()
        return self.vector_store.similarity_search_with_score(query, k=k)

    # ---------------- 入库与重建 ----------------

    def _check_md5_hex(self, md5_for_check: str) -> bool:
        """文件级去重：md5 记录在 md5.text（一行一个），命中说明该文件已入库"""
        md5_path = get_abs_path(faiss_config["md5_hex_store"])
        if not os.path.exists(md5_path):
            open(md5_path, "w", encoding="utf_8").close()
            return False
        with open(md5_path, "r", encoding="utf-8") as f:
            return any(line.strip() == md5_for_check for line in f)

    def _save_md5_hex(self, md5_for_check: str) -> None:
        with open(get_abs_path(faiss_config["md5_hex_store"]), "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def _read_file_text(self, path: str) -> str:
        """读取知识库文件为纯文本（PDF 逐页抽取后拼接；页边界对条目切分无影响）"""
        if path.lower().endswith(".txt"):
            docs = txt_loader(path)
        elif path.lower().endswith(".pdf"):
            docs = pdf_loader(path)
        else:
            return ""
        return "\n".join(d.page_content for d in docs)

    def load_document(self) -> dict:
        """增量入库：只处理新文件/内容变化的文件（文件级 md5 判定）。

        流程：读文件 → 条目切分 → 内容级去重（批内 + 账本）→ 向量入库 → 写账本 → 记 md5。
        返回统计 dict 供 scripts/reindex.py 汇报。
        """
        # 防御：索引文件在但加载失败（版本不符）时禁止增量——否则会以"只含新文件的部分索引"
        # 覆盖原索引，造成知识丢失；此时必须走 rebuild
        if self.vector_store is None and os.path.exists(self.index_file):
            raise RuntimeError(
                "现有索引无法加载（版本信息缺失或模型变更），增量入库会覆盖出残缺索引——"
                "请改用 python -m scripts.reindex --rebuild 全量重建。")

        stats = {"files": 0, "chunks": 0, "dropped_dup": 0, "skipped_files": 0}
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(faiss_config["data_path"]),
            tuple(faiss_config["allowed_knowledge_file_type"]),
        )
        #txt 先于 pdf 处理：同内容重复时保留质量更高的 TXT 版本（见 chunker.dedupe 说明）
        ordered = sorted(allowed_files_path,
                         key=lambda p: (not p.lower().endswith(".txt"), p))

        for path in ordered:
            md5_H = get_file_md5_H(path)
            if not md5_H:
                continue
            if self._check_md5_hex(md5_H):
                logger.info(f"[加载知识库]{path}已经存在知识库，已经跳过")
                stats["skipped_files"] += 1
                continue

            try:
                text = self._read_file_text(path)
                if not text.strip():
                    logger.warning(f"[加载知识库]{path}内没有有效内容，已经跳过")
                    continue

                source_name = os.path.basename(path)
                chunks = split_text(text, source_name)
                if not chunks:
                    logger.warning(f"[加载知识库]{path}切分后没有有效内容，已经跳过")
                    continue

                # 内容级去重①：本轮文件内部重复（如 PDF 内重复标题）
                chunks, dropped_inner = dedupe(chunks)
                # 内容级去重②：与账本中既有内容重复（跨文件/跨批次，PDF 与 TXT 重叠在此消除）
                fresh = [c for c in chunks if not self.chunk_store.has_content_md5(c.content_md5)]
                stats["dropped_dup"] += len(dropped_inner) + (len(chunks) - len(fresh))

                if not fresh:
                    logger.info(f"[加载知识库]{source_name} 内容与既有知识全部重复，整文件跳过")
                    self._save_md5_hex(md5_H)
                    continue

                docs = [Document(page_content=c.text, metadata={
                    "chunk_id": c.id, "source": c.source, "entry_no": c.entry_no,
                    "category": c.category, "title": c.title}) for c in fresh]

                # 首次入库用 from_documents 创建索引；已有索引时 add_documents 增量追加
                # （FAISS 支持对已加载索引追加写入，Chroma 正是在此处报 hnsw 错误）
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(docs, embed_model)
                else:
                    self.vector_store.add_documents(docs)

                self.chunk_store.upsert_many(fresh, rag_config["embedding_model_name"])
                self._save_index()
                self._save_md5_hex(md5_H)

                stats["files"] += 1
                stats["chunks"] += len(fresh)
                logger.info(f"[加载知识库]{source_name} 入库 {len(fresh)} 条"
                            f"（内容去重丢弃 {len(dropped_inner) + (len(chunks) - len(fresh))}）")
            except Exception as e:
                # exc_info=True 记录详细堆栈；单个文件失败不影响其余文件
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)

        return stats

    def rebuild(self) -> dict:
        """全量重建：清空索引/元数据/账本/文件去重记录后重新导入。

        换 embedding 模型、增减切分策略后必须执行；增量模式（load_document）无法
        感知这类变化，只有重建才能保证索引与配置一致。
        """
        self.vector_store = None
        self.load_error = None
        for f in (self.index_file, self.meta_file):
            if os.path.exists(f):
                os.remove(f)
        self.chunk_store.clear()
        md5_path = get_abs_path(faiss_config["md5_hex_store"])
        if os.path.exists(md5_path):
            os.remove(md5_path)
        logger.info("[重建索引]已清空旧索引/元数据/账本/去重记录，开始全量导入")
        return self.load_document()


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.vector_store
    #演示：加载索引做一次向量检索，并打印每个结果的结构化元数据
    vs = VectorStoreService()
    vs.ensure_loaded()
    for doc, score in vs.similarity_search_with_score("键盘按键失灵", k=3):
        meta = doc.metadata
        print(f"[{meta.get('chunk_id')}] 距离={score:.3f} 条目={meta.get('title')}")
        print(doc.page_content[:80].replace("\n", " "))
        print("_" * 40)


# ============================================================================================
# 【第 1 步 · 1.1 说明】向量库改造（本文件的改动点与原因）
# --------------------------------------------------------------------------------------------
# 改动 1：切分接入 rag/chunker.py（条目级）
#   load_document 不再用 split_documents 机械切分，改为 split_text 按条目切；
#   每个 Document 携带结构化元数据（chunk_id/source/entry_no/category/title），
#   这些元数据随向量一起存进索引，检索能直接取回，是引用溯源的原材料。
# 改动 2：内容级去重（原文件级 md5 防不住"文件名不同、内容相同"）
#   三层防线：① 文件级 md5（md5.text，跳过未变化文件）；② 批内内容 md5（chunker.dedupe）；
#   ③ 账本全局内容 md5（chunk_store 唯一索引 + has_content_md5 预查）。
#   处理顺序"txt 先于 pdf"保证重复内容保留 TXT 版本（PDF 抽取质量低于 TXT）。
# 改动 3：索引版本检测（index_meta.json）
#   记录 schema_version / embedding_model / 切分配置 / 构建时间；加载时逐项校验。
#   原实现换 embedding 模型后旧索引静默可用、结果悄悄变垃圾；现在拒绝加载并给出
#   明确的修复命令（--rebuild）。加载失败不炸进程：置 load_error 留给检索链路报错。
# 改动 4：重建入口 rebuild()
#   清空 index.pkl / index_meta.json / chunks.db / md5.text 后全量导入；
#   另加防御：索引存在但加载失败时禁止"增量"（否则会覆盖出残缺索引）。
# 已知取舍：
#   FAISS 入库与账本写入、md5 记录之间没有事务——极端崩溃场景下（入库后、账本前）
#   重复执行可能产生重复向量。演示级取舍，若需强一致可把"账本写入"作为唯一事实源、
#   加一个对账步骤（比对 index_size 与账本 count，不一致则自动重建）。
# 验证方式：
#   python -m scripts.reindex --rebuild → 核对账本各来源条数、index_size；
#   重复执行 python -m scripts.reindex（增量）→ 应显示全部跳过、总量不增。
# ============================================================================================
