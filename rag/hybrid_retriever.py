#混合检索器（第1步·检索纵深）
#【新增】"向量 + BM25 两路召回 → 加权融合 → (可选) cross-encoder 重排"完整流水线，
#替代原实现"纯向量 top-3"的单一召回。四种模式供服务运行与消融评测共用：
#  vector_k3 / vector_k40 —— 纯向量（k 可指定；k3 即改造前行为，用于对照）
#  hybrid                —— 向量 + BM25 融合后取前 N
#  hybrid_rerank         —— 融合候选再经重排模型精排（生产默认，见 retrieval.yml）
import jieba
from dataclasses import dataclass, replace

from rank_bm25 import BM25Okapi

from rag.chunk_store import ChunkStore
from rag.vector_store import VectorStoreService
from utils.config_handler import retrieval_config
from utils.logger_handler import logger


def _tokenize(text: str) -> list[str]:
    """BM25 分词：jieba 精确模式，过滤纯空白 token"""
    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class RetrievedChunk:
    """一条召回结果（两路召回与融合、重排统一用这一种结构流转）"""
    id: str
    text: str
    source: str
    entry_no: str
    category: str
    title: str
    score: float       #排序分：向量路=相似度(0~1)；BM25 路=原始分（融合前归一化）；重排路=relevance_score
    stage: str         #来源阶段：vector / keyword / fuse / rerank（调试与消融分析用）

    def to_source_item(self, index: int) -> dict:
        """转成提示词编号与 SSE 溯源事件共用的来源条目（index 从 1 起）"""
        return {
            "n": index,
            "id": self.id,
            "source": self.source,
            "entry": self.entry_no,
            "title": self.title,
            "snippet": self.text[:80].replace("\n", " "),
            "score": round(self.score, 4),
        }


class HybridRetriever:
    def __init__(self, vector_store: VectorStoreService | None = None, config: dict | None = None):
        #共享 VectorStoreService 实例（它已持有 chunk 账本），避免重复加载 FAISS 索引
        self.vector_store = vector_store or VectorStoreService()
        self.chunk_store: ChunkStore = self.vector_store.chunk_store
        self.cfg = dict(retrieval_config)
        if config:
            self.cfg.update(config)
        self._bm25 = None          #BM25Okapi 实例（惰性构建）
        self._bm25_rows: list[dict] = []

    # ---------------- 召回路 1：向量 ----------------

    def _vector_recall(self, query: str, k: int) -> list[RetrievedChunk]:
        """FAISS 向量召回。注意 FAISS 返回的是 L2 距离（越小越相似），
        这里统一换算成相似度 1/(1+d) —— 单调递减映射，保证"分数越大越相关"，
        后续融合与展示都不必再关心距离方向（这是最容易踩反的坑）。"""
        hits: list[RetrievedChunk] = []
        for doc, dist in self.vector_store.similarity_search_with_score(query, k=k):
            m = doc.metadata
            hits.append(RetrievedChunk(
                id=m.get("chunk_id", ""), text=doc.page_content,
                source=m.get("source", ""), entry_no=m.get("entry_no", ""),
                category=m.get("category", ""), title=m.get("title", ""),
                score=1.0 / (1.0 + float(dist)), stage="vector"))
        return hits

    # ---------------- 召回路 2：BM25 关键词 ----------------

    def _ensure_bm25(self) -> None:
        """惰性构建 BM25 索引（全量 chunk 分词一次，约 1 秒；构建后进程内复用）"""
        if self._bm25 is not None:
            return
        self._bm25_rows = self.chunk_store.all_chunks()
        corpus = [_tokenize(r["text"]) for r in self._bm25_rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info(f"[BM25]关键词索引构建完成：{len(self._bm25_rows)} 条")

    def _keyword_recall(self, query: str, k: int) -> list[RetrievedChunk]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:      #无任何关键词重叠的候选不注入（避免零分噪声挤占融合池）
                continue
            r = self._bm25_rows[i]
            hits.append(RetrievedChunk(
                id=r["id"], text=r["text"], source=r["source"], entry_no=r["entry_no"],
                category=r["category"], title=r["title"],
                score=float(scores[i]), stage="keyword"))
        return hits

    # ---------------- 融合 ----------------

    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        """min-max 归一化（两路分数量纲不同：向量路 0~1，BM25 无上界）"""
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    def _fuse(self, vector_hits: list[RetrievedChunk],
              keyword_hits: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """加权融合：各路分数先归一化，再按 fuse_weights 加权求和（对齐 RAGFlow 默认
        0.7 全文 / 0.3 向量）；两路都命中的条目分数叠加，自然排到前面。"""
        w_k = float(self.cfg["fuse_weights"].get("keyword", 0.7))
        w_v = float(self.cfg["fuse_weights"].get("vector", 0.3))

        scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}
        if vector_hits:
            for h, s in zip(vector_hits, self._minmax([h.score for h in vector_hits])):
                scores[h.id] = scores.get(h.id, 0.0) + w_v * s
                best.setdefault(h.id, h)
        if keyword_hits:
            for h, s in zip(keyword_hits, self._minmax([h.score for h in keyword_hits])):
                scores[h.id] = scores.get(h.id, 0.0) + w_k * s
                best.setdefault(h.id, h)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for cid, s in ranked:
            #复制后再改写分数：不修改两路召回传入的原始对象——
            #否则同一批对象被再次融合（测试/复用场景）时分数已污染，排序计算全错
            hit = replace(best[cid])
            hit.score = s
            hit.stage = "fuse"
            out.append(hit)
        return out

    # ---------------- 重排 ----------------

    def _rerank(self, query: str, candidates: list[RetrievedChunk],
                top_n: int) -> list[RetrievedChunk]:
        """按配置选择重排档位；任何重排失败都退化为"融合序"而不是让整条链路报错"""
        provider = self.cfg.get("rerank_provider", "none")
        if not candidates or provider == "none":
            return candidates[:top_n]
        try:
            if provider == "dashscope":
                return self._rerank_dashscope(query, candidates, top_n)
            if provider == "local":
                return self._rerank_local(query, candidates, top_n)
            logger.warning(f"[重排]未知 provider={provider}，跳过重排返回融合序")
        except Exception as e:
            logger.error(f"[重排]{provider} 重排失败，退化为融合序：{str(e)}", exc_info=True)
        return candidates[:top_n]

    def _rerank_dashscope(self, query: str, candidates: list[RetrievedChunk],
                          top_n: int) -> list[RetrievedChunk]:
        """DashScope 云重排（gte-rerank-v2）：复用已有 DASHSCOPE_API_KEY，零新增部署"""
        import dashscope
        docs = [c.text[:1024] for c in candidates]   #重排接口对单条文本长度有上限
        resp = dashscope.TextReRank.call(
            model=self.cfg.get("rerank_model", "gte-rerank-v2"),
            query=query, documents=docs, top_n=top_n, return_documents=False)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope 重排接口返回 {resp.status_code}："
                               f"{getattr(resp, 'code', '')} {getattr(resp, 'message', '')}")
        out: list[RetrievedChunk] = []
        for r in resp.output.results:
            c = candidates[r.index]
            c.score = float(r.relevance_score)
            c.stage = "rerank"
            out.append(c)
        return out

    def _rerank_local(self, query: str, candidates: list[RetrievedChunk],
                      top_n: int) -> list[RetrievedChunk]:
        """本地 cross-encoder 重排（可选档位）：需自行安装 sentence-transformers，
        首次运行会下载 BAAI/bge-reranker-base（约 1GB）；不装则报错并自动退化为融合序"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "未安装本地重排依赖。安装：.venv/Scripts/python.exe -m pip install "
                "-i https://pypi.org/simple/ sentence-transformers；或改用 "
                "rerank_provider=dashscope") from e
        if getattr(self, "_local_reranker", None) is None:
            self._local_reranker = CrossEncoder(self.cfg.get("rerank_local_model",
                                                             "BAAI/bge-reranker-base"))
        pairs = [(query, c.text) for c in candidates]
        scores = self._local_reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
            c.stage = "rerank"
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]

    # ---------------- 统一入口 ----------------

    def search(self, query: str, mode: str | None = None,
               top_n: int | None = None) -> list[RetrievedChunk]:
        """检索入口。

        mode：vector_k3 / vector_k40（纯向量，k 为召回条数）/ hybrid（融合）/
              hybrid_rerank（融合+重排）；缺省用 retrieval.yml 的 mode。
        top_n：返回条数；缺省用 retrieval.yml 的 rerank_top_n（生产=5）。
        """
        mode = mode or self.cfg.get("mode", "hybrid")
        top_n = top_n or int(self.cfg["rerank_top_n"])

        if mode.startswith("vector"):
            k = int(mode.split("_k")[1]) if "_k" in mode else int(self.cfg["vector_top_k"])
            hits = self._vector_recall(query, k)
            return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]

        vec_hits = self._vector_recall(query, int(self.cfg["vector_top_k"]))
        kw_hits = self._keyword_recall(query, int(self.cfg["bm25_top_k"]))
        fused = self._fuse(vec_hits, kw_hits, int(self.cfg["fuse_top_k"]))
        if mode == "hybrid":
            return fused[:top_n]
        return self._rerank(query, fused, top_n)


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.hybrid_retriever
    #演示：同一问题在四种模式下的 top-5 对比（重排会调用 DashScope 接口）
    r = HybridRetriever()
    q = "鼠标回报率8000Hz值得开吗"
    for mode in ("vector_k3", "vector_k40", "hybrid", "hybrid_rerank"):
        print(f"\n===== {mode} =====")
        for i, hit in enumerate(r.search(q, mode=mode), 1):
            print(f"  {i}. [{hit.score:.4f}][{hit.stage}] {hit.source}#{hit.entry_no} {hit.title}")


# ============================================================================================
# 【第 1 步 · 1.3 说明】混合检索与重排（本文件的设计说明）
# --------------------------------------------------------------------------------------------
# 为什么这样改（对应原实现的三个短板）：
#   原实现 = 纯向量 k=3 + 无重排。问题：① 召回上限被 k=3 锁死；② 纯向量对精确词
#   （型号/数字/专有名词，如"8000Hz""48kHz""RT"）不敏感，语义相近但答不对题的条目会
#   排在前面；③ 向量相似度只是粗排，"最像"不等于"最能回答问题"。
# 流水线（两路召回 → 融合 → 重排）：
#   ① 向量路：FAISS similarity_search_with_score 取 vector_top_k=40 条；
#      注意 L2 距离→相似度换算 1/(1+d) 在 _vector_recall 内统一完成（方向踩反是常见 bug）；
#   ② 关键词路：jieba 分词 + BM25Okapi（rank_bm25）取 bm25_top_k=40 条，零分候选不注入；
#   ③ 融合：两路分数各自 min-max 归一化后按 0.7(关键词)/0.3(向量) 加权求和，
#      两路都命中的条目分数叠加自然靠前；权重与候选池都可在 retrieval.yml 调整；
#   ④ 重排：候选池交 cross-encoder 精排取 rerank_top_n=5。
#      dashscope=云端 gte-rerank-v2（默认，零部署）；local=bge-reranker-base（需自行安装）；
#      none=跳过（消融对照组）。重排任何失败都退化为融合序，不拖垮整条链路。
# 四种模式的用途：
#   vector_k3   = 改造前行为（基线对照）；vector_k40 = 单看"扩大召回池"的收益；
#   hybrid      = 单看"混合召回"的收益；hybrid_rerank = 完整流水线（生产默认）。
#   eval/run_retrieval.py 一次跑齐四种模式，输出 recall@5 / MRR@10 对比表。
# 设计取舍：
#   - BM25 索引走内存构建（781 条约 1 秒），进程内复用；知识库更新（reindex）后
#     需重启服务或调用 reload 才会生效——演示级取舍，生产可换 Elasticsearch/OpenSearch；
#   - 融合用"归一化加权和"而非 RRF：与 RAGFlow 默认口径一致、权重可解释；
#     若要更稳的排序可分制融合（RRF），换 _fuse 一个函数即可，评测脚本能量化差异。
# 验证方式：python -m rag.hybrid_retriever 对比四模式 top-5；
#           python -m eval.run_retrieval --modes vector_k3,vector_k40,hybrid,hybrid_rerank
# ============================================================================================
