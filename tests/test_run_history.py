#运行历史表测试（第2步·2.4）
#覆盖：一轮运行的开始/结束/节点明细落库 / 越权防护（别人的运行查不到）/
#      列表按用户过滤与时间倒序 / 用量汇总的用户口径 / counts 覆盖新表。
#离线：只碰临时库（tmp_path），不调模型、不碰项目 data/ 目录。
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.database_service import DatabaseService


def _db(tmp_path):
    return DatabaseService(db_path=str(tmp_path / "runs.db"), import_seed=False)


def test_run_lifecycle_and_nodes(tmp_path):
    db = _db(tmp_path)
    db.start_run("r1", "2483", "s1", "帮我生成售后报告", route="report")
    db.add_run_node("r1", node="classify", seq=1, status="ok", started_at="2026-09-17 10:00:00",
                    duration_ms=0, label="意图识别", detail="报告场景（并行调研）")
    db.add_run_node("r1", node="researcher", seq=2, status="ok", started_at="2026-09-17 10:00:00",
                    duration_ms=3200, kind="fault", label="故障诊断", detail="故障类型：单边无声")
    db.finish_run("r1", status="ok", duration_ms=73000,
                  usage={"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600},
                  report_json='{"fault_type": "单边无声"}', detail="")

    run = db.get_run("r1", "2483")
    assert run["状态"] == "ok"
    assert run["路由"] == "report"
    assert run["总tokens"] == 1600
    assert run["报告JSON"] == '{"fault_type": "单边无声"}'
    assert [n["节点"] for n in run["nodes"]] == ["classify", "researcher"]
    assert run["nodes"][1]["类型"] == "fault"


def test_cannot_read_other_users_run(tmp_path):
    db = _db(tmp_path)
    db.start_run("r2", "2483", "s1", "问题")
    assert db.get_run("r2", "9999") is None      #换个人来查 → 查不到（越权防护）


def test_list_runs_filters_by_user_and_orders_desc(tmp_path):
    db = _db(tmp_path)
    for i, (run_id, user) in enumerate([("a", "2483"), ("b", "2483"), ("c", "9999")]):
        db.start_run(run_id, user, "s1", f"问题{i}")
    items = db.list_runs("2483")
    assert {i["run_id"] for i in items} == {"a", "b"}
    assert all(i["用户ID"] == "2483" for i in items)


def test_usage_summary_scoped_by_user(tmp_path):
    db = _db(tmp_path)
    db.start_run("r3", "2483", "s1", "问题")
    db.add_usage_event(source="chat", provider="deepseek", model="deepseek-v4-pro",
                       unit="token", amount=500, run_id="r3", note="节点 model")
    db.add_usage_event(source="reindex", provider="dashscope", model="text-embedding-v4",
                       unit="token", amount=50000, note="离线索引重建")   #无 run_id → 不属于任何用户

    mine = db.usage_summary(user_id="2483")
    assert [(r["provider"], r["amount"]) for r in mine] == [("deepseek", 500)]

    everything = db.usage_summary()
    assert sum(r["amount"] for r in everything) == 50500


def test_counts_includes_run_tables(tmp_path):
    db = _db(tmp_path)
    db.start_run("r4", "2483", "s1", "问题")
    db.add_run_node("r4", node="normal", seq=1, status="ok", started_at="2026-09-17 10:00:00")
    counts = db.counts()
    assert counts["runs"] == 1
    assert counts["run_nodes"] == 1
    assert counts["usage_events"] == 0


def test_finish_run_updates_existing_row(tmp_path):
    """收尾必须更新同一行（不是插新行），否则一次运行会出现两条记录。

    路由要在收尾时补写：它是 classify 节点跑完才知道的，start_run 时还拿不到。
    """
    db = _db(tmp_path)
    db.start_run("r5", "2483", "s1", "问题")
    db.finish_run("r5", status="ok", duration_ms=1234, route="normal")
    rows = db.list_runs("2483")
    assert len(rows) == 1
    assert rows[0]["状态"] == "ok"
    assert rows[0]["路由"] == "normal"


# 运行：cd 项目根 && .venv/Scripts/python.exe -m pytest tests/test_run_history.py
