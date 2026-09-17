#答案质量评测脚本（LLM-as-judge）
#结果：eval/results/answer_quality.md（逐题得分与总均分）
import argparse
import json
import re
import sys
from datetime import datetime

import yaml
from langchain_core.prompts import ChatPromptTemplate

from model.fectory import chat_model
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

_GOLDEN_PATH = "eval/golden.yaml"
_RESULT_PATH = "eval/results/answer_quality.md"

#裁判提示词：只输出 JSON，便于程序解析（数值要求整数 1~5）
_JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "你是严格的 RAG 答案评测员。请根据【问题】【答案要点】【检索来源】【模型回答】打分（1-5 整数）：\n"
    "- relevance：回答是否切题、直接回应用户问题\n"
    "- groundedness：关键结论是否都能在检索来源中找到依据（编造来源外的事实=低分）\n"
    "- completeness：是否覆盖了【答案要点】中的信息\n"
    '只输出 JSON（不要多余文字）：{{"relevance": n, "groundedness": n, "completeness": n, "reason": "一句话理由"}}\n\n'
    "【问题】{question}\n"
    "【答案要点】{key_points}\n"
    "【检索来源】{sources}\n"
    "【模型回答】{answer}"
)


def _parse_scores(content: str) -> dict:
    """从裁判输出中提取 JSON（容忍模型偶尔包裹的说明文字/代码块）"""
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError(f"裁判输出无法解析：{content[:120]!r}")
    data = json.loads(match.group(0))
    return {k: int(data.get(k, 0)) for k in ("relevance", "groundedness", "completeness")} | {
        "reason": str(data.get("reason", ""))}


def main() -> int:
    parser = argparse.ArgumentParser(description="答案质量评测（LLM-as-judge）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 题（控制调用成本/时长）")
    args = parser.parse_args()

    with open(get_abs_path(_GOLDEN_PATH), encoding="utf-8") as f:
        items = yaml.safe_load(f)["items"]
    if args.limit:
        items = items[:args.limit]

    rag = RagSummarizeService()
    # 判官需要看到来源"全文"才能公平评 groundedness——sources 里只有 80 字片段，
    # 这里按 id 回账本取全文（截断到 400 字/条，控制判官输入长度）
    from rag.chunk_store import ChunkStore
    store = ChunkStore()

    rows: list[dict] = []
    for it in items:
        answer, sources = rag.rag_summarize(it["question"])
        full = store.get_by_ids([s["id"] for s in sources])
        src_text = "\n".join(
            f"{s['source']}#{s['entry']} {s['title']}："
            f"{full.get(s['id'], {}).get('text', s['snippet'])[:400]}"
            for s in sources) or "（无检索来源）"
        try:
            resp = chat_model.invoke(_JUDGE_PROMPT.format_messages(
                question=it["question"],
                key_points="、".join(it.get("key_points", [])),
                sources=src_text,
                answer=answer))
            scores = _parse_scores(resp.content if isinstance(resp.content, str) else str(resp.content))
        except Exception as e:
            logger.error(f"[答案评测]{it['id']} 判分失败：{e}")
            scores = {"relevance": 0, "groundedness": 0, "completeness": 0, "reason": f"判分失败：{e}"}
        rows.append({"id": it["id"], **scores})
        logger.info(f"[答案评测]{it['id']} 得分 {scores}")

    n = len(rows)
    avg = {k: sum(r[k] for r in rows) / n for k in ("relevance", "groundedness", "completeness")}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 答案质量评测（LLM-as-judge）",
        "",
        f"- 题目数：{n}（golden.yaml{' 前 N 题' if args.limit else ' 全量'}）　裁判：DeepSeek　时间：{timestamp}",
        f"- 均分（1~5）：**relevance {avg['relevance']:.2f} · groundedness {avg['groundedness']:.2f}"
        f" · completeness {avg['completeness']:.2f}**",
        "",
        "| 题号 | 切题 | 有据 | 完整 | 评语 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {r['relevance']} | {r['groundedness']} | {r['completeness']} | {r['reason']} |")
    import os
    os.makedirs(os.path.dirname(get_abs_path(_RESULT_PATH)), exist_ok=True)
    with open(get_abs_path(_RESULT_PATH), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"答案质量评测完成（{n} 题）：relevance={avg['relevance']:.2f} "
          f"groundedness={avg['groundedness']:.2f} completeness={avg['completeness']:.2f}")
    print(f"明细已写入：{_RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())