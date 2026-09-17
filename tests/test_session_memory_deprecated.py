#会话记忆弃用断言（第2步·2.1）
#目的：锁死"聊天链路不再使用进程内会话记忆"这一事实——文件保留是为了留档与自检，
#但代码里不许再出现引用，否则"重启续聊/持久化"会在无人察觉的情况下失效（测试兜住）。
#同时单测 SessionMemoryService 本身仍可用（保留其历史价值与自检能力）。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.session_memory_service import SessionMemoryService


def test_app_no_longer_references_session_memory():
    """app.py 的代码行（非注释、非字符串说明）里不得再出现 session_memory_service"""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    code_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
    offenders = [line for line in code_lines if "session_memory_service" in line]
    assert offenders == [], f"app.py 仍在代码里引用已弃用的会话记忆：{offenders}"


def test_inprocess_memory_still_functional():
    """弃用 ≠ 报废：模块自身仍可独立使用（__main__ 自检依赖它）"""
    svc = SessionMemoryService()
    svc.append_message("s1", "user", "我的机械键盘按键失灵了")
    svc.append_message("s1", "assistant", "先检查轴体是否有异物")
    history = svc.get_history("s1")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"].startswith("我的机械键盘")
    #滑窗仍然生效：超过 max_turns 轮的旧消息会被丢弃
    for i in range(20):
        svc.append_message("s1", "user", f"第{i}问")
    assert len(svc.get_history("s1")) <= 2 * svc._max_turns


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_session_memory_deprecated.py
