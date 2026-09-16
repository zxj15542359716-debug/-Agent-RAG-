#结构化切分器（第1步·检索纵深）
#【新增】把知识库文件按"条目"切分为 chunk，替代原先固定 200 字/20 重叠的机械切分：
#1. 本知识库本身就是结构化文本（编号条目 / 问答对），按条目边界切分能保持语义完整——
#   原来 200 字会把"现象/原因/排查/解决"一条完整故障说明劈碎，检索到碎片也答不好；
#2. 每个 chunk 携带元数据（来源文件/条目号/品类/标题），这是引用溯源（SSE sources 事件）
#   与检索评测（按条目判命中）的数据基础；
#3. 只有条目超长（> max_entry_chars）时才退回 RecursiveCharacterTextSplitter 兜底切分。
import hashlib
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import faiss_config
from utils.logger_handler import logger

#条目超过该长度才二次切分（本库条目多在 60~250 字，通常不会触发）
_MAX_ENTRY_CHARS = int(faiss_config.get("max_entry_chars", 500))

#三种条目形态的识别正则（设计说明见文件底部注释块）
_NUM_ENTRY_RE = re.compile(r"^(\d{1,4})\s*[.．、]\s*【(.+?)】")
_QA_RE = re.compile(r"^问\s*(\d{1,4})\s*[：:]")
_SECTION_RE = re.compile(r"^【(.+?)篇】")
#PDF 抽取消噪：页眉/页码模式（如 "电脑外设100问 · 第 3 页"）是排版噪声不是内容差异；
#去噪后 PDF 与 TXT 的同内容条目才能正确判重（只允许吃掉中文/字母/数字/空白，不会误伤正文标点）
_PAGE_NOISE_RE = re.compile(r"[一-鿿A-Za-z0-9\s]{0,16}[·•]\s*第\s*\d+\s*页")

#兜底切分器（参数沿用 faiss.yml 的历史配置，仅在超长条目时使用）
_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=faiss_config["chunk_size"],
            chunk_overlap=faiss_config["chunk_overlap"],
            separators=faiss_config["separators"],
            length_function=len,
        )
    return _splitter


@dataclass
class Chunk:
    """一条切分结果。

    - id 稳定：`来源#条目号`（跨索引重建不变），是评测标注与溯源的主键；
    - content_md5：去掉全部空白后的内容指纹，用于跨文件内容级去重。
    """
    id: str
    text: str
    source: str        #来源文件名（如 故障排除.txt）
    entry_no: str      #条目编号："1"（编号条目）/ "问12"（问答）/ ""（兜底切分）
    category: str      #品类（耳机/机械键盘/…），取自条目【】或篇章标题
    title: str         #条目标题（首行冒号前部分 / 问题行），截断到 40 字
    chunk_idx: int     #同一来源文件内的顺序号
    content_md5: str   #内容指纹（去空白后 md5）


def _norm_for_md5(text: str) -> str:
    """去重指纹：先去掉页眉/页码噪声，再去除所有空白字符。

    PDF 抽取会插入页眉（"… · 第 N 页"）与多余空格/换行；归一化后可与 TXT 中
    同内容条目判为重复，从源头消除双份入库（实测：PDF 151 条 → 仅 1 条封面残留）。
    """
    text = _PAGE_NOISE_RE.sub("", text)
    return re.sub(r"\s+", "", text)


def _make_chunk(chunks: list[Chunk], source: str, entry_no: str, category: str,
                title: str, text: str, uid: str | None = None) -> None:
    """组装一个 Chunk 并追加（空文本跳过；uid 用于超长条目二次切分的子片编号）"""
    text = text.strip()
    if not text:
        return
    cid = uid or (f"{source}#{entry_no}" if entry_no else f"{source}#auto{len(chunks)}")
    chunks.append(Chunk(
        id=cid,
        text=text,
        source=source,
        entry_no=entry_no,
        category=category,
        title=title.strip()[:40],
        chunk_idx=len(chunks),
        content_md5=hashlib.md5(_norm_for_md5(text).encode("utf-8")).hexdigest(),
    ))


def _emit(chunks: list[Chunk], source: str, entry_no: str, category: str,
          title: str, text: str) -> None:
    """结算一条条目：正常长度直接成块；超长则兜底二次切分（子片沿用同一 entry_no，
    id 加 `~N` 后缀，保证"按条目判命中"的评测与溯源依然对齐）"""
    text = text.strip()
    if not text:
        return
    if len(text) <= _MAX_ENTRY_CHARS:
        _make_chunk(chunks, source, entry_no, category, title, text)
        return
    parts = _get_splitter().split_text(text)
    for i, part in enumerate(parts):
        uid = f"{source}#{entry_no}~{i + 1}" if entry_no else None
        _make_chunk(chunks, source, entry_no, category, title, part, uid=uid)


def _split_numbered(lines: list[str], source: str) -> list[Chunk]:
    """编号条目切分：`N.【品类】标题` 为块起点，块延伸到下一个编号/篇章标题/文末。

    自动兼容两种排版：故障排除（块内多行：现象/原因/排查/解决）与
    维护保养/选购指南（一行即一条）。
    """
    chunks: list[Chunk] = []
    cur_no = cur_cat = cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_lines
        _emit(chunks, source, cur_no, cur_cat, cur_title, "\n".join(cur_lines))
        cur_lines = []

    for ln in lines:
        m = _NUM_ENTRY_RE.match(ln)
        sec = _SECTION_RE.match(ln)
        if m:
            flush()
            cur_no = m.group(1)
            cur_cat = m.group(2)
            rest = ln[m.end():].strip()
            #标题 = 首个冒号前的内容（一行一条的排版）；无冒号则整行即标题（多行条目的排版）
            cur_title = rest.split("：", 1)[0].split(":", 1)[0]
            cur_lines = [ln]
        elif sec:
            flush()
            cur_no = ""            #篇章标题自身不入库，仅作为后续条目的兜底品类
            cur_cat = sec.group(1)
        elif ln.strip():
            cur_lines.append(ln)
    flush()
    return chunks


def _split_qa(lines: list[str], source: str) -> list[Chunk]:
    """问答对切分：`问N：…` 与其下的 `答：…`（可能多行）合成一条 chunk"""
    chunks: list[Chunk] = []
    cur_no = cur_cat = cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_lines
        _emit(chunks, source, cur_no, cur_cat, cur_title, "\n".join(cur_lines))
        cur_lines = []

    for ln in lines:
        m = _QA_RE.match(ln)
        sec = _SECTION_RE.match(ln)
        if m:
            flush()
            cur_no = f"问{m.group(1)}"
            cur_title = ln[m.end():]
            cur_lines = [ln]
        elif sec:
            flush()
            cur_no = ""
            cur_cat = sec.group(1)
        elif ln.strip():
            cur_lines.append(ln)
    flush()
    return chunks


def _split_fallback(text: str, source: str) -> list[Chunk]:
    """识别不出条目结构（如 PDF 抽取的变体文本）：退回字符切分兜底，无条目元数据"""
    chunks: list[Chunk] = []
    for part in _get_splitter().split_text(text):
        _make_chunk(chunks, source, "", "", "", part)
    return chunks


def split_text(text: str, source: str) -> list[Chunk]:
    """入口：按内容自动识别条目形态并切分。

    - 含 `问N：` 行 → 问答对模式
    - 含 `N.【…】` 行 → 编号条目模式
    - 都不匹配 → 字符切分兜底
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    #过滤数据文件头部的 # 注释行（如"# 覆盖品类：…共160条"）
    lines = [ln for ln in lines if not ln.lstrip().startswith("#")]

    if any(_QA_RE.match(ln) for ln in lines):
        return _split_qa(lines, source)
    if any(_NUM_ENTRY_RE.match(ln) for ln in lines):
        return _split_numbered(lines, source)
    logger.info(f"[切分器]{source} 未识别条目结构，退回字符切分")
    return _split_fallback(text, source)


def dedupe(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """内容级去重：同内容（去空白后 md5 相同）只保留最先出现的一条，跨文件生效。

    PDF 与同名 TXT 的重叠内容在这里被消除——调用方需保证 .txt 先于 .pdf 处理，
    这样保留下来的永远是质量更高的 TXT 版本。
    返回 (保留, 丢弃) 两个列表，丢弃名单供日志统计。
    """
    seen: set[str] = set()
    kept: list[Chunk] = []
    dropped: list[Chunk] = []
    for c in chunks:
        if c.content_md5 in seen:
            dropped.append(c)
            continue
        seen.add(c.content_md5)
        kept.append(c)
    return kept, dropped


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.chunker
    #自检：对 data/ 下各知识库文件跑一遍切分，打印统计与样例
    import os

    from utils.path_tool import get_abs_path

    data_dir = get_abs_path(faiss_config["data_path"])
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, encoding="utf-8") as f:
            cs = split_text(f.read(), name)
        cs, dropped = dedupe(cs)
        print(f"{name}: {len(cs)} 条（去重丢弃 {len(dropped)}）")
        if cs:
            sample = cs[0]
            print(f"   样例 [{sample.id}] {sample.category} | {sample.title} | {len(sample.text)}字")


# ============================================================================================
# 【第 1 步 · 1.1 说明】结构化切分（本文件的作用与设计）
# --------------------------------------------------------------------------------------------
# 为什么改：
#   原实现 RecursiveCharacterTextSplitter(chunk_size=200, overlap=20) 按字符机械切分——
#   - 一条"故障排除"条目（现象/原因/排查/解决，约 150~250 字）被劈成 2~3 个碎片，
#     检索命中碎片后模型拼不出完整答案，这是检索质量的下限来源；
#   - 分片不带任何元数据，引用溯源（哪条资料支撑了结论）无从做起；
#   - 评测也无法"按条目判命中"（只知道检索到某文件，不知道对不对）。
# 怎么切（split_text 自动识别三种形态）：
#   1) 问答对（电脑外设100问.txt / 2.txt）：`问N：…` 与紧随的 `答：…` 合成一条；
#   2) 编号条目（故障排除.txt / 维护保养.txt / 选购指南.txt）：`N.【品类】标题` 为块起点，
#      块延伸到下一个编号或篇章标题——同一套逻辑自动兼容"一行一条"与"多行一条"两种排版；
#   3) 结构识别失败（如 PDF 抽取的变体文本）：退回原字符切分兜底，不元数据、不报错。
# 关键设计：
#   - chunk.id = `来源#条目号`（如 故障排除.txt#12、电脑外设100问.txt#问3），跨索引重建稳定，
#     是评测标注（golden.yaml）与溯源展示的主键；
#   - content_md5 = "去页眉噪声 + 去全部空白"后的 md5：PDF 与 TXT 的重叠内容在此判重，
#     配合"txt 文件先处理"的顺序，双份入库问题从源头消除（原文件级 MD5 防不住这个）；
#     实测：PDF 151 条中 150 条判重成功，仅剩 1 条封面页残留 chunk（可接受的噪声）；
#     彻底治理同源文档可在 faiss.yml 的 allowed_knowledge_file_type 里移除 "pdf"；
#   - 超长条目（>max_entry_chars=500）才二次切分，子片沿用同一 entry_no（id 加 ~N 后缀），
#     判命中与溯源仍按条目对齐。
# 怎么验证：
#   1) python -m rag.chunker 自检各文件切分统计与样例；
#   2) scripts/reindex.py --rebuild 重建后核对 chunks 账本各来源条数；
#   3) eval/run_retrieval.py 的 recall@5 / MRR 直接反映切分质量。
# ============================================================================================
