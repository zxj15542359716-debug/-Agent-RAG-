#结构化切分器
import hashlib
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import faiss_config
from utils.logger_handler import logger

#条目超过该长度才二次切分
_MAX_ENTRY_CHARS = int(faiss_config.get("max_entry_chars", 500))

#三种条目形态的识别正则
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
    """一条切分结果"""
    id: str
    text: str
    source: str        #来源文件名（如 故障排除.txt）
    entry_no: str      #条目编号："1"（编号条目）/ "问12"（问答）/ ""（兜底切分）
    category: str      #品类（耳机/机械键盘/…），取自条目【】或篇章标题
    title: str         #条目标题（首行冒号前部分 / 问题行），截断到 40 字
    chunk_idx: int     #同一来源文件内的顺序号
    content_md5: str   #内容指纹（去空白后 md5）


def _norm_for_md5(text: str) -> str:
    """去重指纹：先去掉页眉/页码噪声，再去除所有空白字符。"""
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
    """编号条目切分：`N.【品类】标题` 为块起点，块延伸到下一个编号/篇章标题/文末。"""
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
    """入口：按内容自动识别条目形态并切分"""
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
    """内容级去重：同内容（去空白后 md5 相同）只保留最先出现的一条，跨文件生效。"""
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