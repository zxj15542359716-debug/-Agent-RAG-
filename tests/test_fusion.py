#融合逻辑测试（第1步·1.3）
#用合成分数验证 min-max 归一化与加权融合的数学行为，不依赖 embedding / 外部 API。
from rag.hybrid_retriever import HybridRetriever, RetrievedChunk


def _hit(cid: str, score: float, stage: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text="t", source="s.txt", entry_no="1",
                          category="", title="t", score=score, stage=stage)


def test_minmax_normalization():
    assert HybridRetriever._minmax([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    assert HybridRetriever._minmax([2.0, 2.0]) == [1.0, 1.0]   # 全等时统一给 1，避免除零


def test_fuse_respects_weights():
    """同一批候选：两路权重变化应改变排序——这正是 retrieval.yml 里 fuse_weights 的语义"""
    r = HybridRetriever.__new__(HybridRetriever)   # 绕过 __init__（避免加载真实索引）

    vec_hits = [_hit("A", 1.0, "vector"), _hit("B", 0.5, "vector")]
    kw_hits = [_hit("C", 10.0, "keyword"), _hit("A", 5.0, "keyword")]

    # 纯向量权重：A（两路都命中，向量第一）应在首位
    r.cfg = {"fuse_weights": {"keyword": 0.0, "vector": 1.0}}
    out = r._fuse(vec_hits, kw_hits, top_k=3)
    assert out[0].id == "A"
    assert {h.id for h in out} == {"A", "B", "C"}   # 融合池包含两路候选

    # 纯关键词权重：C（仅关键词路第一）应升到首位
    r.cfg = {"fuse_weights": {"keyword": 1.0, "vector": 0.0}}
    out2 = r._fuse(vec_hits, kw_hits, top_k=3)
    assert out2[0].id == "C"


def test_source_item_shape():
    """溯源条目结构：n/id/source/entry/title/snippet/score 齐备（SSE 事件与前端依赖）"""
    item = _hit("s.txt#12", 0.42, "fuse").to_source_item(3)
    assert item["n"] == 3
    assert item["id"] == "s.txt#12"
    assert item["source"] == "s.txt"
    assert item["entry"] == "1"
    assert item["score"] == 0.42
    assert isinstance(item["snippet"], str)


# ============================================================================================
# 【第 1 步 · 1.3 说明】本测试文件锁住的行为（为什么测这些）
# --------------------------------------------------------------------------------------------
# 1. min-max 归一化：两路分数量纲完全不同（向量路 0~1、BM25 无上界），归一化错了权重就失真；
#    全等输入（单候选）必须给 1 而不是除零。
# 2. 融合权重语义：权重变化必须真实改变排序——用"纯向量权重"和"纯关键词权重"两个极端
#    端到端验证 _fuse，而不是只看实现细节；retrieval.yml 的调参（0.4/0.6）依赖该语义。
# 3. 溯源条目结构：to_source_item 的输出字段被 SSE 事件与前端 renderSources 直接消费，
#    字段缺失会在浏览器端表现为"参考来源显示不全"。
# 运行：.venv/Scripts/python.exe -m pytest tests/test_fusion.py
# ============================================================================================
