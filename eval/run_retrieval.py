#检索评测脚本（第1步·1.2）
#【新增】对 golden.yaml 的 52 条标注逐题跑检索，输出 recall@5 / MRR@10，
#支持一次跑多种模式（vector_k3 / vector_k40 / hybrid / hybrid_rerank）做消融对比。
#用法（cd 项目根）：
#  .venv/Scripts/python.exe -m eval.run_retrieval                                   # 默认跑全部四种模式
#  .venv/Scripts/python.exe -m eval.run_retrieval --modes hybrid --limit 10         # 只跑一种、前10题
#结果：eval/results/retrieval_<mode>.md（逐题命中明细）+ eval/results/comparison.md（汇总对比表）
import argparse
import sys
from datetime import datetime

import yaml

from rag.hybrid_retriever import HybridRetriever
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

_RESULTS_DIR = "eval/results"
_GOLDEN_PATH = "eval/golden.yaml"


def load_golden(limit: int | None = None) -> list[dict]:
    with open(get_abs_path(_GOLDEN_PATH), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data["items"]
    return items[:limit] if limit else items


def evaluate(retriever: HybridRetriever, items: list[dict], mode: str,
             top_k: int = 10) -> dict:
    """单模式评测：逐题检索 → 判断首次命中位次 → 汇总指标。

    指标定义：
      recall@5——前 5 条内至少命中一个标注条目的题目占比（本库每题一般 1~2 个相关条目，
              等价于常用口径的 hit@5；标注多条目时任一命中即算）
      MRR@10 ——前 10 条内首个命中位次 rank 的 1/rank 的平均（未命中计 0）
    """
    hit_at_5 = 0
    mrr_sum = 0.0
    rows: list[dict] = []

    for it in items:
        results = retriever.search(it["question"], mode=mode, top_n=top_k)
        relevant = {(src, entry) for src, entry in it["relevant"]}
        rank = next((i + 1 for i, r in enumerate(results) if (r.source, r.entry_no) in relevant), None)
        if rank is not None and rank <= 5:
            hit_at_5 += 1
        if rank is not None:
            mrr_sum += 1.0 / rank
        rows.append({
            "id": it["id"], "question": it["question"], "rank": rank,
            "top1": f"{results[0].source}#{results[0].entry_no}" if results else "",
            "expect": "、".join(f"{s}#{e}" for s, e in it["relevant"]),
        })
        logger.info(f"[评测][{mode}] {it['id']} rank={rank} 期望={rows[-1]['expect']} 实际top1={rows[-1]['top1']}")

    n = len(items)
    return {"mode": mode, "n": n, "recall@5": hit_at_5 / n, "mrr@10": mrr_sum / n, "rows": rows}


def write_report(result: dict, timestamp: str) -> str:
    """写单模式明细报告，返回文件路径"""
    mode = result["mode"]
    path = get_abs_path(f"{_RESULTS_DIR}/retrieval_{mode}.md")
    lines = [
        f"# 检索评测明细 · {mode}",
        "",
        f"- 题目数：{result['n']}　**recall@5 = {result['recall@5']:.3f}**　**MRR@10 = {result['mrr@10']:.3f}**",
        f"- 运行时间：{timestamp}",
        "",
        "| 题号 | 首次命中位次 | 期望条目 | 实际 top1 | 问题 |",
        "|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        rank = r["rank"] if r["rank"] else "未命中"
        lines.append(f"| {r['id']} | {rank} | {r['expect']} | {r['top1']} | {r['question']} |")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_comparison(all_results: list[dict], timestamp: str) -> str:
    """写多模式汇总对比表（README 引用同一份数据）"""
    path = get_abs_path(f"{_RESULTS_DIR}/comparison.md")
    lines = [
        "# 检索质量对比（第1步·RAG 纵深）",
        "",
        f"- 评测集：eval/golden.yaml（52 题，标注到条目级）　运行时间：{timestamp}",
        "- 指标：recall@5 = 前 5 条命中标注条目的题目占比；MRR@10 = 首个命中位次倒数的均值",
        "",
        "| 检索模式 | recall@5 | MRR@10 | 说明 |",
        "|---|---|---|---|",
    ]
    notes = {
        "vector_k3": "改造前行为（纯向量 3 条）",
        "vector_k40": "仅扩大召回池",
        "hybrid": "向量+BM25 融合（0.4/0.6，经评测调参）",
        "hybrid_rerank": "融合 + gte-rerank-v2 精排（生产默认）",
    }
    for r in all_results:
        lines.append(f"| {r['mode']} | {r['recall@5']:.3f} | {r['mrr@10']:.3f} | {notes.get(r['mode'], '')} |")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评测（golden set）")
    parser.add_argument("--modes", default="vector_k3,vector_k40,hybrid,hybrid_rerank",
                        help="逗号分隔的模式列表")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 题（调试用）")
    parser.add_argument("--top-k", type=int, default=10, help="评测时检索返回条数（默认10）")
    args = parser.parse_args()

    items = load_golden(args.limit)
    retriever = HybridRetriever()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        logger.info(f"[评测]开始模式 {mode}（{len(items)} 题）")
        result = evaluate(retriever, items, mode, top_k=args.top_k)
        path = write_report(result, timestamp)
        all_results.append(result)
        print(f"{mode:>15}: recall@5 = {result['recall@5']:.3f}   MRR@10 = {result['mrr@10']:.3f}"
              f"   → {path}")

    if len(all_results) > 1:
        path = write_comparison(all_results, timestamp)
        print(f"\n对比表已写入：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================================
# 【第 1 步 · 1.2 说明】检索评测（本文件的指标口径与使用方式）
# --------------------------------------------------------------------------------------------
# 为什么先建评测再改检索：
#   改造顺序是"先测基线，再动刀"——没有基线的优化只是感觉；先跑出 vector_k3（改造前行为）
#   的数字，之后每加一层（召回池/混合/重排）都能回答"这一层带来了多少收益"。
# 指标口径：
#   recall@5：前 5 条内命中标注条目的题目占比。标注到"条目级"（来源文件+条目号），
#             比"命中某文件就算对"严格得多，能真实区分切分与召回质量；
#   MRR@10 ：首个命中位次倒数的均值，反映"正确的排得有多靠前"（重排的主要受益指标）。
# 四种模式对应四层收益（消融实验）：
#   vector_k3 → vector_k40 ：只扩大召回池的收益（原 k=3 是致命瓶颈）
#   vector_k40 → hybrid    ：关键词路补充精确词召回（型号/数字类问题的主要受益者）
#   hybrid → hybrid_rerank ：重排把"最像的"换成"最该回答的"（top-1 质量提升）
# 运行成本：检索评测不调用生成模型——只有 embedding（向量化查询）与重排调用，
#   52 题 × 4 模式约几十秒、费用可忽略；可随时复跑做回归。
# 与 run_answer.py 的分工：
#   本文件评"检索得对不对"；run_answer.py 评"最终答案好不好"（LLM-as-judge）。
#   检索指标先坏，答案质量必然坏——所以排障顺序永远是先看本文件。
# 结果文件：eval/results/retrieval_<mode>.md（逐题明细，失败题一眼可见）
#           eval/results/comparison.md（汇总表，会被 README 引用）
# ============================================================================================
