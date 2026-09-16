#切分器与内容去重测试（第1步·1.1）
#锁住条目级切分的核心行为：三种形态识别、标题/编号提取、页眉去噪判重、超长条目二次切分。
#不依赖外部服务，纯本地运行。
import hashlib

from rag.chunker import Chunk, _norm_for_md5, dedupe, split_text


def test_split_numbered_multiline():
    """多行条目（故障排除形态）：一条 = 一个 chunk，编号/品类/标题正确，注释行被过滤"""
    text = (
        "# 故障排除参考数据（注释行应被过滤）\n"
        "1.【耳机】单边无声\n"
        "现象：左耳完全无声，右耳正常。\n"
        "原因：线材内部断裂。\n"
        "解决：送修。\n"
        "\n"
        "2.【鼠标】丢帧跳帧\n"
        "现象：指针一卡一卡。\n"
    )
    chunks = split_text(text, "t.txt")
    assert len(chunks) == 2
    assert chunks[0].id == "t.txt#1"
    assert chunks[0].category == "耳机"
    assert chunks[0].title == "单边无声"
    assert "原因：线材内部断裂。" in chunks[0].text
    assert chunks[1].id == "t.txt#2"


def test_split_numbered_oneline():
    """一行一条（维护保养形态）：标题取冒号前部分"""
    text = "1.【耳机】耳罩清洁：每月用微湿软布轻拭。\n2.【耳机】头梁清洁：每月擦拭。\n"
    chunks = split_text(text, "t.txt")
    assert [c.title for c in chunks] == ["耳罩清洁", "头梁清洁"]
    assert chunks[0].entry_no == "1"


def test_split_qa():
    """问答对：问+答合成一条；篇章标题提供品类"""
    text = "【耳机篇】\n\n问1：连不上手机怎么办？\n答：先进入配对模式。\n\n问2：怎么配对？\n答：长按电源键。\n"
    chunks = split_text(text, "q.txt")
    assert [c.id for c in chunks] == ["q.txt#问1", "q.txt#问2"]
    assert chunks[0].category == "耳机"
    assert "答：先进入配对模式。" in chunks[0].text


def _mk_chunk(cid: str, source: str, text: str) -> Chunk:
    return Chunk(id=cid, text=text, source=source, entry_no="问8", category="",
                 title="q", chunk_idx=0,
                 content_md5=hashlib.md5(_norm_for_md5(text).encode("utf-8")).hexdigest())


def test_dedupe_ignores_pdf_page_noise():
    """内容级去重：页眉噪声不参与指纹——PDF 版与 TXT 版判为重复，保留先出现的（TXT）"""
    c_txt = _mk_chunk("a.txt#问8", "a.txt", "问8：降噪怎么开？\n答：长按降噪键。")
    c_pdf = _mk_chunk("b.pdf#问8", "b.pdf", "问8：降噪怎么开？ 某文档 · 第 3 页\n答：长按降噪键。")
    kept, dropped = dedupe([c_txt, c_pdf])
    assert len(kept) == 1 and kept[0].source == "a.txt"
    assert len(dropped) == 1


def test_long_entry_split_keeps_entry_no():
    """超长条目二次切分：子片沿用同一 entry_no（评测判命中与溯源不受影响）"""
    text = "1.【耳机】大条目：" + ("很长的内容。" * 120)   # 超过 max_entry_chars=500
    chunks = split_text(text, "t.txt")
    assert len(chunks) > 1
    assert all(c.entry_no == "1" for c in chunks)
    assert chunks[0].id.endswith("~1")


# ============================================================================================
# 【第 1 步 · 1.1 说明】本测试文件锁住的行为（为什么测这些）
# --------------------------------------------------------------------------------------------
# 1. 三种条目形态的识别（多行编号/单行编号/问答对）——切分错了，检索、溯源、评测全部失真，
#    是最上游的回归点；用例直接用了数据文件的真实排版片段。
# 2. 页眉噪声判重——这是"PDF 与 TXT 双份入库"的关键修复，用合成样例锁住（不依赖真实 PDF）。
# 3. 超长条目二次切分后 entry_no 保持不变——保证"按条目判命中"的评测与前端溯源在
#    切分兜底时依然对齐。
# 运行：.venv/Scripts/python.exe -m pytest tests/test_chunking.py
# ============================================================================================
