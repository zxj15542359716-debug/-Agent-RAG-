#用量与运行报表（第2步·2.5）
#用途：在终端回答"token 花在哪了"——先按 来源/提供方/模型 汇总，再列最近的运行明细。
#运行：cd 项目根 && .venv/Scripts/python.exe -m scripts.usage_report [--days 7] [--user 2483] [--runs 5]
#
#口径提醒（脚本每次都会打印）：本报表统计的是"我们消耗了多少"，
#不回答"账户还剩多少免费额度"——剩余额度只能去阿里云百炼控制台查看。
import argparse
import sys
from datetime import datetime, timedelta

from service.database_service import get_database_service


def main() -> int:
    parser = argparse.ArgumentParser(description="用量与运行报表（第2步·2.4 的账本）")
    parser.add_argument("--days", type=int, default=7, help="统计最近 N 天（默认 7）")
    parser.add_argument("--user", default="", help="只看某个用户ID（不填=全部）")
    parser.add_argument("--runs", type=int, default=5, help="附带最近 N 条运行明细（默认 5，0=不列）")
    args = parser.parse_args()

    db = get_database_service()
    since = (datetime.now() - timedelta(days=max(1, args.days))).strftime("%Y-%m-%d %H:%M:%S")
    rows = db.usage_summary(user_id=args.user or None, since=since)

    scope = f"用户 {args.user}" if args.user else "全部用户"
    print("=" * 78)
    print(f"用量报表（最近 {args.days} 天 · {scope} · 起始 {since}）")
    print("-" * 78)
    if not rows:
        print("（这段时间没有用量记录：要么还没跑过对话，要么超过记录范围）")
    else:
        print(f"{'来源':<8}{'提供方':<11}{'模型':<22}{'单位':<7}{'数量':>10}{'条数':>7}")
        for r in rows:
            print(f"{r['source']:<8}{r['provider']:<11}{r['model']:<22}{r['unit']:<7}"
                  f"{r['amount']:>10}{r['events']:>7}")
        total_tokens = sum(r["amount"] for r in rows if r["unit"] == "token")
        print("-" * 78)
        print(f"合计（token 单位）：{total_tokens}")

    if args.runs and args.runs > 0:
        print("-" * 78)
        print("最近运行：")
        print(f"{'时间':<20}{'用户':<7}{'路由':<8}{'状态':<7}{'耗时':>8}{'tokens':>9}  问题")
        for run in db.list_runs(args.user or None, args.runs):
            ms = int(run.get("耗时毫秒") or 0)
            print(f"{run.get('开始时间', ''):<20}{run.get('用户ID', ''):<7}{run.get('路由', ''):<8}"
                  f"{run.get('状态', ''):<7}{(ms / 1000):>7.1f}s{int(run.get('总tokens') or 0):>9}"
                  f"  {(run.get('问题') or '')[:28]}")

    print("=" * 78)
    print("口径说明：以上是【我们消耗了多少】；【账户还剩多少免费额度】请到阿里云百炼控制台")
    print("（模型广场 → 对应模型 → 免费额度）查看——本地没有该数据，任何脚本都算不出来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================================
# 【第 2 步 · 2.5 说明】用量报表脚本（scripts/usage_report.py）
# --------------------------------------------------------------------------------------------
# 用途：把 runs / usage_events 两张表变成"人一眼能看懂"的报表——按来源与模型汇总消耗，
# 再列最近几次运行的路由/状态/耗时/token，用于演示与排障（例：报告为什么这么贵）。
# 为什么在终端打印而不是做成接口：它是运维/开发视角的工具（看全量、看历史）；
# 面向用户的视图走 GET /api/usage（只看得到自己的运行，见 app.py）。
# 为什么要显式打印"口径说明"：本地账本只能回答"花了多少"，回答不了"还剩多少额度"，
# 这句写在输出里可以避免"以为跑个脚本就知道剩余额度"的误解。
# 与其它文件的关系：数据来自 service/database_service.py 的三张表；
# 由 utils/usage_ledger.py 在每次对话结束时写入。
# ============================================================================================
