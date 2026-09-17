#数据库服务（SQLite）
#【新增】把外部使用记录从 CSV 升级为关系数据库：用户/购买记录/维修记录三张表，
#提供完整的增删改查；程序首次启动时若库为空，自动从 records.csv 导入初始数据。
#选型说明：SQLite 零配置、单文件、Python 标准库自带 sqlite3，适合本地单进程应用；
#表结构按 1个用户:N笔购买、1笔购买:N条维修 的一对多关系设计，外键开启级联删除，
#删除用户时其购买与维修记录一并清理。
import bcrypt
import csv
import hashlib
import os
import random
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

#【修改】密码哈希由"固定盐 + sha256"升级为 bcrypt：
#原方案是单轮快哈希（GPU 每秒可尝试数十亿次），且盐硬编码在源码中；
#bcrypt 自带随机盐、工作因子可调，专为抗离线暴力破解设计。
#_PASSWORD_SALT 仅保留用于校验历史遗留的 sha256 哈希（登录成功后自动升级为 bcrypt）。
_PASSWORD_SALT = "aftersales-salt"

#建表语句：三张表，外键声明 ON DELETE CASCADE 配合 PRAGMA foreign_keys 实现级联删除
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    用户ID   TEXT PRIMARY KEY,
    密码     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS purchases (
    购买ID   INTEGER PRIMARY KEY AUTOINCREMENT,
    用户ID   TEXT NOT NULL REFERENCES users(用户ID) ON DELETE CASCADE,
    特征     TEXT NOT NULL,
    外设类型 TEXT NOT NULL,
    购买时间 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS repairs (
    维修ID   INTEGER PRIMARY KEY AUTOINCREMENT,
    购买ID   INTEGER NOT NULL REFERENCES purchases(购买ID) ON DELETE CASCADE,
    维修日期 TEXT NOT NULL,
    损坏原因 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    上报ID   INTEGER PRIMARY KEY AUTOINCREMENT,
    用户ID   TEXT NOT NULL REFERENCES users(用户ID) ON DELETE CASCADE,
    外设类型 TEXT NOT NULL,
    故障描述 TEXT NOT NULL,
    上报时间 TEXT NOT NULL
);
-- 【第2步·2.4】运行历史三表：runs（每轮对话一行）/ run_nodes（节点级明细）/ usage_events（计量流水）
-- 说明：runs.user_id 故意不建外键——运行历史是审计账，用户被删也不该让账目消失（与业务表口径不同）
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    用户ID      TEXT NOT NULL,
    会话ID      TEXT NOT NULL DEFAULT '',
    问题        TEXT NOT NULL DEFAULT '',
    路由        TEXT NOT NULL DEFAULT '',
    状态        TEXT NOT NULL DEFAULT 'running',
    开始时间    TEXT NOT NULL,
    结束时间    TEXT,
    耗时毫秒    INTEGER DEFAULT 0,
    输入tokens  INTEGER DEFAULT 0,
    输出tokens  INTEGER DEFAULT 0,
    总tokens    INTEGER DEFAULT 0,
    嵌入tokens  INTEGER DEFAULT 0,
    重排tokens  INTEGER DEFAULT 0,
    重排次数    INTEGER DEFAULT 0,
    报告JSON    TEXT,
    备注        TEXT
);
CREATE TABLE IF NOT EXISTS run_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    序号        INTEGER NOT NULL DEFAULT 0,
    节点        TEXT NOT NULL,
    类型        TEXT DEFAULT '',
    标签        TEXT DEFAULT '',
    状态        TEXT NOT NULL DEFAULT 'ok',
    开始时间    TEXT NOT NULL,
    耗时毫秒    INTEGER DEFAULT 0,
    输入tokens  INTEGER DEFAULT 0,
    输出tokens  INTEGER DEFAULT 0,
    总tokens    INTEGER DEFAULT 0,
    命名空间    TEXT DEFAULT '',
    摘要        TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    时间        TEXT NOT NULL,
    来源        TEXT NOT NULL,
    提供方      TEXT NOT NULL,
    模型        TEXT NOT NULL,
    单位        TEXT NOT NULL,
    数量        INTEGER NOT NULL,
    run_id      TEXT,
    备注        TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_user_time ON runs(用户ID, 开始时间 DESC);
CREATE INDEX IF NOT EXISTS idx_run_nodes_run ON run_nodes(run_id, 序号);
CREATE INDEX IF NOT EXISTS idx_usage_events_ts ON usage_events(时间 DESC);
"""


def _hash_password(password: str) -> str:
    """密码哈希：bcrypt（自带随机盐）；按 72 字节截断（bcrypt 算法输入上限）"""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def _now() -> str:
    """统一时间戳格式（与既有上报时间口径一致：YYYY-MM-DD HH:MM:SS）"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _verify_password(password: str, stored: str) -> bool:
    """校验密码：兼容 bcrypt（$2 前缀）与历史 sha256 十六进制两种存储格式"""
    if stored.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8")[:72], stored.encode("utf-8"))
        except ValueError:
            return False
    return stored == hashlib.sha256((_PASSWORD_SALT + password).encode("utf-8")).hexdigest()


class DatabaseService:
    """SQLite 数据库服务：封装用户/购买/维修三张表的增删改查与登录校验。

    - 首次初始化时自动建表；三张表为空则从 CSV 导入初始数据（只执行一次）
    - 每次操作独立开连接：sqlite3 连接非线程安全，FastAPI 多线程下按操作开合最稳妥
    - 初始化失败时置 self.error，供 ExternalRecordService 判断"数据源故障"
    """

    def __init__(self, db_path: str | None = None, import_seed: bool = True) -> None:
        #数据库文件路径默认取配置文件，也允许测试时传入其他路径；
        #import_seed=False 跳过 CSV 初始导入（测试用：避免每个用例重复 bcrypt 哈希 170+ 个用户）
        self._db_path = db_path or get_abs_path(agent_config["database_path"])
        self.error: str | None = None   #初始化失败时置错误信息；None 表示数据源正常
        try:
            #确保数据库文件所在目录存在，再建表、按需导入 CSV
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._init_schema()
            if import_seed:
                self._import_from_csv_if_empty()
        except sqlite3.Error as e:
            logger.error(f"[数据库]初始化失败：{str(e)}")
            self.error = f"数据库初始化失败：{str(e)}"

    # ---------------- 基础设施 ----------------

    def _conn(self) -> sqlite3.Connection:
        """新建一个连接（Row 工厂便于按列名取值），并开启外键约束（级联删除生效）。

        【修复并发隐患】busy_timeout 设 10 秒：FastAPI 多线程下两个请求同时写库时，
        后者最多等待 10 秒而不是立刻抛"database is locked"。
        """
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    @contextmanager
    def _tx(self):
        """事务上下文：结束自动提交，并确保关闭连接。

        【修复连接泄漏】sqlite3 连接自带 with 只提交/回滚事务、不关闭连接，
        原写法每次操作泄漏一个连接与文件句柄，长期运行会耗尽句柄或残留锁；
        统一改用本上下文：事务结束（提交）后 finally 关闭连接。
        """
        conn = self._conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """建表（IF NOT EXISTS，重复执行无副作用）"""
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    def _import_from_csv_if_empty(self) -> None:
        """首次运行迁移：users 表为空时从 records.csv 导入初始数据。

        CSV 表头：用户ID,密码,特征,外设类型,售后记录,购买时间；
        售后记录内多条以 | 分隔，每条为"维修日期:损坏原因"，导入时拆进 repairs 表。
        """
        with self._tx() as conn:
            has_users = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
            if has_users:
                return   #库中已有数据，跳过导入（避免重复迁移）

        csv_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(csv_path):
            logger.warning(f"[数据库]初始 CSV 不存在：{csv_path}，跳过导入")
            return

        valid, skipped = 0, 0
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            #按表头名定位列，兼容列顺序变化
            col = {name: header.index(name)
                   for name in ("用户ID", "密码", "特征", "外设类型", "售后记录", "购买时间")}

            with self._tx() as conn:
                for line_no, row in enumerate(reader, start=2):
                    #字段数不足视为坏行，跳过并告警
                    if len(row) < len(col):
                        skipped += 1
                        logger.warning(f"[数据库]CSV第{line_no}行字段数不足，已跳过")
                        continue
                    user_id = row[col["用户ID"]].strip()
                    password = row[col["密码"]].strip() or "1111"   #空密码兜底为默认 1111
                    feature = row[col["特征"]].strip()
                    device_type = row[col["外设类型"]].strip()
                    repair_raw = row[col["售后记录"]].strip()
                    month = row[col["购买时间"]].strip()
                    if not user_id or not month:
                        skipped += 1
                        logger.warning(f"[数据库]CSV第{line_no}行用户ID或购买时间为空，已跳过")
                        continue

                    #同一用户多笔购买共享一条用户记录：不存在才插入，密码冲突时保留首次
                    exists = conn.execute("SELECT 1 FROM users WHERE 用户ID = ?", (user_id,)).fetchone()
                    if exists is None:
                        conn.execute("INSERT INTO users (用户ID, 密码) VALUES (?, ?)",
                                     (user_id, _hash_password(password)))
                    cur = conn.execute(
                        "INSERT INTO purchases (用户ID, 特征, 外设类型, 购买时间) VALUES (?, ?, ?, ?)",
                        (user_id, feature, device_type, month))
                    purchase_id = cur.lastrowid

                    #售后记录拆成逐条维修记录入库
                    for entry in repair_raw.split("|"):
                        entry = entry.strip()
                        if not entry:
                            continue
                        repair_date, sep, cause = entry.partition(":")
                        if not sep or not repair_date or not cause:
                            skipped += 1
                            logger.warning(f"[数据库]CSV第{line_no}行售后记录格式异常：{entry!r}，已跳过")
                            continue
                        conn.execute("INSERT INTO repairs (购买ID, 维修日期, 损坏原因) VALUES (?, ?, ?)",
                                     (purchase_id, repair_date, cause))
                    valid += 1

        logger.info(f"[数据库]CSV导入完成：有效 {valid} 条，跳过坏行 {skipped} 条")
        logger.info(f"[数据库]当前数据量：{self.counts()}")

    def counts(self) -> dict[str, int]:
        """各表的行数统计（日志与演示用）"""
        with self._tx() as conn:
            return {
                "users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "purchases": conn.execute("SELECT COUNT(*) FROM purchases").fetchone()[0],
                "repairs": conn.execute("SELECT COUNT(*) FROM repairs").fetchone()[0],
                "reports": conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0],
                #【第2步·2.4】运行历史三表（自检与演示时一眼能看到"账有没有在记"）
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "run_nodes": conn.execute("SELECT COUNT(*) FROM run_nodes").fetchone()[0],
                "usage_events": conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0],
            }

    # ---------------- 运行历史 / 用量账（第2步·2.4） ----------------
    # 设计口径：
    #   1. 一轮对话 = 一条 runs；每个节点事件 = 一条 run_nodes；每次计量 = 一条 usage_events。
    #   2. 写账失败绝不能影响回答：调用方（app.py / usage_ledger）统一捕获异常只记日志。
    #   3. 所有查询都按 用户ID 过滤（在 SQL 里强制），不接受前端传 user_id——与第 0 步口径一致。

    def start_run(self, run_id: str, user_id: str, session_id: str, query: str, route: str = "") -> None:
        """开始一轮运行：先落一条 running 记录，结束时用 finish_run 补全"""
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, 用户ID, 会话ID, 问题, 路由, 状态, 开始时间) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (run_id, user_id, session_id, query, route, _now()))

    def finish_run(self, run_id: str, status: str, duration_ms: int, usage: dict | None = None,
                   report_json: str | None = None, detail: str = "", route: str = "") -> None:
        """结束一轮运行：写状态、耗时与 token 汇总（异常路径也要调用，保证不漏账）。

        路由在这里一并补写：路由是 classify 节点跑完才知道的（start_run 时还没算出来）。
        """
        usage = usage or {}
        with self._tx() as conn:
            #路由用 NULLIF 兜一下：传空串时保留 start_run 已写入的值，避免"收尾把已知路由抹掉"
            conn.execute(
                "UPDATE runs SET 状态=?, 结束时间=?, 耗时毫秒=?, 输入tokens=?, 输出tokens=?, "
                "总tokens=?, 嵌入tokens=?, 重排tokens=?, 重排次数=?, 报告JSON=?, 备注=?, "
                "路由=COALESCE(NULLIF(?, ''), 路由) WHERE run_id=?",
                (status, _now(), int(duration_ms),
                 int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                 int(usage.get("total_tokens", 0)), int(usage.get("embedding_tokens", 0)),
                 int(usage.get("rerank_tokens", 0)), int(usage.get("rerank_calls", 0)),
                 report_json, detail, route, run_id))

    def add_run_node(self, run_id: str, node: str, seq: int, status: str, started_at: str,
                     duration_ms: int = 0, kind: str = "", label: str = "", ns: str = "",
                     usage: dict | None = None, detail: str = "") -> None:
        """记录一个节点的执行明细（时间线落库版）"""
        usage = usage or {}
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO run_nodes (run_id, 序号, 节点, 类型, 标签, 状态, 开始时间, 耗时毫秒, "
                "输入tokens, 输出tokens, 总tokens, 命名空间, 摘要) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, seq, node, kind, label, status, started_at, int(duration_ms),
                 int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                 int(usage.get("total_tokens", 0)), ns, detail))

    def get_run(self, run_id: str, user_id: str) -> dict | None:
        """按运行号取一条运行 + 其节点明细；非本人运行返回 None（越权防护）"""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=? AND 用户ID=?", (run_id, user_id)).fetchone()
            if row is None:
                return None
            run = dict(row)
            nodes = conn.execute(
                "SELECT * FROM run_nodes WHERE run_id=? ORDER BY 序号, id", (run_id,)).fetchall()
            run["nodes"] = [dict(n) for n in nodes]
            return run

    def list_runs(self, user_id: str, limit: int = 20) -> list[dict]:
        """列出某用户最近的运行记录（不含节点明细）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE 用户ID=? ORDER BY 开始时间 DESC LIMIT ?",
                (user_id, max(1, min(int(limit), 200)))).fetchall()
            return [dict(r) for r in rows]

    def add_usage_event(self, source: str, provider: str, model: str, unit: str, amount: int,
                        run_id: str | None = None, note: str = "") -> None:
        """记一条用量流水（chat / reindex / eval 等来源；单位是 token 或 call）"""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO usage_events (时间, 来源, 提供方, 模型, 单位, 数量, run_id, 备注) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), source, provider, model, unit, int(amount), run_id, note))

    def usage_summary(self, user_id: str | None = None, since: str | None = None) -> list[dict]:
        """按（来源/提供方/模型/单位）汇总用量；传 user_id 时只统计该用户运行产生的用量"""
        sql = ("SELECT 来源 AS source, 提供方 AS provider, 模型 AS model, 单位 AS unit, "
               "SUM(数量) AS amount, COUNT(*) AS events FROM usage_events")
        where, params = [], []
        if since:
            where.append("时间 >= ?")
            params.append(since)
        if user_id:
            where.append("run_id IN (SELECT run_id FROM runs WHERE 用户ID = ?)")
            params.append(user_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY 来源, 提供方, 模型, 单位 ORDER BY amount DESC"
        with self._tx() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ---------------- 用户表 CRUD ----------------

    def verify_user(self, user_id: str, password: str) -> bool:
        """登录校验：用户存在且密码匹配返回 True，其余情况一律 False。

        【新增】惰性迁移：历史用户存的是 sha256 哈希，校验通过后顺手改写为 bcrypt——
        存量账号无需重置密码、无需停机迁移，谁登录谁升级。
        """
        with self._tx() as conn:
            row = conn.execute("SELECT 密码 FROM users WHERE 用户ID = ?", (user_id,)).fetchone()
        if row is None or not _verify_password(password, row["密码"]):
            return False
        if not row["密码"].startswith("$2"):   #旧格式哈希：登录成功即升级为 bcrypt
            self.update_password(user_id, password)
        return True

    def create_user(self, user_id: str, password: str) -> None:
        """新增用户（密码不限长度、不做格式限制）；用户已存在抛 ValueError"""
        if not user_id or not password:
            raise ValueError("用户ID和密码不能为空")
        with self._tx() as conn:
            try:
                conn.execute("INSERT INTO users (用户ID, 密码) VALUES (?, ?)",
                             (user_id, _hash_password(password)))
            except sqlite3.IntegrityError:
                raise ValueError(f"用户 {user_id} 已存在") from None

    def update_password(self, user_id: str, new_password: str) -> bool:
        """修改用户密码；用户不存在返回 False"""
        if not new_password:
            return False
        with self._tx() as conn:
            cur = conn.execute("UPDATE users SET 密码 = ? WHERE 用户ID = ?",
                               (_hash_password(new_password), user_id))
        return cur.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """删除用户（外键级联：其购买记录与维修记录一并删除）；不存在返回 False"""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM users WHERE 用户ID = ?", (user_id,))
        return cur.rowcount > 0

    def list_users(self) -> list[dict[str, Any]]:
        """列出全部用户（不含密码哈希）"""
        with self._tx() as conn:
            return [dict(row) for row in conn.execute("SELECT 用户ID FROM users ORDER BY 用户ID")]

    def generate_free_user_id(self) -> str:
        """随机生成一个未被占用的4位用户ID（1000-9999）。

        连试 100 次都撞号视为可用ID耗尽（9000 个候选，正常使用不会发生）。
        """
        with self._tx() as conn:
            used = {r[0] for r in conn.execute("SELECT 用户ID FROM users")}
        for _ in range(100):
            uid = f"{random.randint(1000, 9999)}"
            if uid not in used:
                return uid
        raise RuntimeError("可用用户ID已耗尽，请联系管理员")

    def create_user_auto(self, password: str) -> str:
        """注册用：随机分配一个未被占用的4位用户ID并创建用户，返回分配的ID。

        密码不限长度、不做格式限制（由用户自行设置）；可用ID耗尽时抛 RuntimeError。
        """
        if not password:
            raise ValueError("密码不能为空")
        user_id = self.generate_free_user_id()
        with self._tx() as conn:
            conn.execute("INSERT INTO users (用户ID, 密码) VALUES (?, ?)",
                         (user_id, _hash_password(password)))
        return user_id

    # ---------------- 购买记录表 CRUD ----------------

    def add_purchase(self, user_id: str, feature: str, device_type: str, month: str) -> int:
        """新增一笔购买记录，返回自增的购买ID；用户不存在抛 ValueError"""
        with self._tx() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO purchases (用户ID, 特征, 外设类型, 购买时间) VALUES (?, ?, ?, ?)",
                    (user_id, feature, device_type, month))
            except sqlite3.IntegrityError:
                raise ValueError(f"用户 {user_id} 不存在") from None
        return cur.lastrowid

    def get_purchase(self, user_id: str, month: str) -> dict[str, Any] | None:
        """按用户+购买时间查询一笔购买记录；无记录返回 None"""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM purchases WHERE 用户ID = ? AND 购买时间 = ? LIMIT 1",
                (user_id, month)).fetchone()
        return dict(row) if row else None

    def get_purchases_by_user(self, user_id: str) -> list[dict[str, Any]]:
        """查询某用户的全部购买记录（按购买时间升序）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM purchases WHERE 用户ID = ? ORDER BY 购买时间", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def update_purchase(self, purchase_id: int, feature: str | None = None,
                        device_type: str | None = None, month: str | None = None) -> bool:
        """修改购买记录的可变字段（只更新传入的非空项）；记录不存在返回 False"""
        fields, values = [], []
        for name, value in (("特征", feature), ("外设类型", device_type), ("购买时间", month)):
            if value:
                fields.append(f"{name} = ?")
                values.append(value)
        if not fields:
            return False
        values.append(purchase_id)
        with self._tx() as conn:
            cur = conn.execute(f"UPDATE purchases SET {', '.join(fields)} WHERE 购买ID = ?", values)
        return cur.rowcount > 0

    def delete_purchase(self, purchase_id: int) -> bool:
        """删除一笔购买记录（级联删除其维修记录）；不存在返回 False"""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM purchases WHERE 购买ID = ?", (purchase_id,))
        return cur.rowcount > 0

    # ---------------- 维修记录表 CRUD ----------------

    def add_repair(self, purchase_id: int, repair_date: str, cause: str) -> int:
        """给一笔购买记录新增一条维修记录，返回自增的维修ID；购买记录不存在抛 ValueError"""
        with self._tx() as conn:
            try:
                cur = conn.execute("INSERT INTO repairs (购买ID, 维修日期, 损坏原因) VALUES (?, ?, ?)",
                                   (purchase_id, repair_date, cause))
            except sqlite3.IntegrityError:
                raise ValueError(f"购买记录 {purchase_id} 不存在") from None
        return cur.lastrowid

    def get_repairs(self, purchase_id: int) -> list[dict[str, Any]]:
        """查询一笔购买记录的全部维修记录（按维修日期升序）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM repairs WHERE 购买ID = ? ORDER BY 维修日期", (purchase_id,)).fetchall()
        return [dict(r) for r in rows]

    def update_repair(self, repair_id: int, repair_date: str | None = None,
                      cause: str | None = None) -> bool:
        """修改维修记录（只更新传入的非空项）；记录不存在返回 False"""
        fields, values = [], []
        for name, value in (("维修日期", repair_date), ("损坏原因", cause)):
            if value:
                fields.append(f"{name} = ?")
                values.append(value)
        if not fields:
            return False
        values.append(repair_id)
        with self._tx() as conn:
            cur = conn.execute(f"UPDATE repairs SET {', '.join(fields)} WHERE 维修ID = ?", values)
        return cur.rowcount > 0

    def delete_repair(self, repair_id: int) -> bool:
        """删除一条维修记录；不存在返回 False"""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM repairs WHERE 维修ID = ?", (repair_id,))
        return cur.rowcount > 0

    # ---------------- 售后上报表 CRUD ----------------

    def add_report(self, user_id: str, device_type: str, fault: str) -> int:
        """新增一条售后上报记录，上报时间由服务器自动记录（不信任前端时间）；
        用户不存在抛 ValueError，返回自增的上报ID。"""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._tx() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO reports (用户ID, 外设类型, 故障描述, 上报时间) VALUES (?, ?, ?, ?)",
                    (user_id, device_type, fault, report_time))
            except sqlite3.IntegrityError:
                raise ValueError(f"用户 {user_id} 不存在") from None
        return cur.lastrowid

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        """按上报ID查询一条上报记录；不存在返回 None"""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM reports WHERE 上报ID = ?", (report_id,)).fetchone()
        return dict(row) if row else None

    def get_reports_by_user(self, user_id: str) -> list[dict[str, Any]]:
        """查询某用户的全部上报记录（按上报ID升序）"""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM reports WHERE 用户ID = ? ORDER BY 上报ID", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def list_reports(self) -> list[dict[str, Any]]:
        """列出全部上报记录（按上报ID升序，演示与排查用）"""
        with self._tx() as conn:
            rows = conn.execute("SELECT * FROM reports ORDER BY 上报ID").fetchall()
        return [dict(r) for r in rows]

    def delete_report(self, report_id: int) -> bool:
        """删除一条上报记录；不存在返回 False"""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM reports WHERE 上报ID = ?", (report_id,))
        return cur.rowcount > 0


#模块级惰性单例：与 ExternalRecordService 的用法一致
_instance: DatabaseService | None = None
_instance_lock = threading.Lock()   #【修复并发竞态】见 get_database_service 注释


def get_database_service() -> DatabaseService:
    """获取全局唯一的数据库服务实例（惰性创建，双重检查加锁）。

    【修复并发竞态】FastAPI 多线程下两个并发首请求可能同时走到 None 分支，
    各自执行一次 DatabaseService()：除重复建表（IF NOT EXISTS 无害）外，
    更重要的是会各自检查"users 表为空"并同时执行 CSV 导入——购买记录会被重复导入两遍。
    加锁后保证初始化与导入只发生一次。
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DatabaseService()
    return _instance


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m service.database_service
    svc = get_database_service()
    print("三表行数:", svc.counts())
    print("登录校验(2483/1111):", svc.verify_user("2483", "1111"))
    print("登录校验(2483/错误密码):", svc.verify_user("2483", "0000"))
    print("登录校验(不存在的用户):", svc.verify_user("0000", "1111"))
    # 增：新用户（密码不限长度）+ 一笔购买 + 两条维修
    svc.create_user("9999", "abcdef123456")
    pid = svc.add_purchase("9999", "游戏|办公", "键盘", "2026-09")
    svc.add_repair(pid, "2026-09-05", "测试损坏原因A")
    rid = svc.add_repair(pid, "2026-09-08", "测试损坏原因B")
    print("新增后 9999 的购买:", svc.get_purchases_by_user("9999"))
    print("购买记录", pid, "的维修:", svc.get_repairs(pid))
    # 改：改密码、改购买类型、改维修原因
    print("改密码:", svc.update_password("9999", "新密码666"))
    print("新密码登录:", svc.verify_user("9999", "新密码666"))
    print("改购买类型:", svc.update_purchase(pid, device_type="鼠标"))
    print("改维修原因:", svc.update_repair(rid, cause="测试损坏原因B-已修复"))
    print("修改后维修:", svc.get_repairs(pid))
    # 删：删除用户，级联删除其购买与维修
    print("删除用户:", svc.delete_user("9999"))
    print("删除后 9999 的购买:", svc.get_purchases_by_user("9999"))
    print("删除后维修:", svc.get_repairs(pid))
    # 注册与售后上报演示：随机分配用户ID + 服务器自动记录上报时间
    new_id = svc.create_user_auto("demo密码123456")
    print("注册新用户，系统分配的ID:", new_id)
    print("新用户登录:", svc.verify_user(new_id, "demo密码123456"))
    rep_id = svc.add_report(new_id, "耳机", "右耳无声，疑似线材断裂")
    print("上报记录:", svc.get_report(rep_id))
    print("该用户全部上报:", svc.get_reports_by_user(new_id))
    print("删除上报:", svc.delete_report(rep_id))
    print("删除注册用户:", svc.delete_user(new_id))
    print("各表行数:", svc.counts())


# ============================================================================================
# 【第 2 步 · 2.4 说明】运行历史与用量账（本文件的改动说明）
# --------------------------------------------------------------------------------------------
# 改动点：_SCHEMA 新增三张表（runs / run_nodes / usage_events + 三个索引），counts() 补三表行数，
# 并新增对应 CRUD：start_run / finish_run / add_run_node / get_run / list_runs /
# add_usage_event / usage_summary。
# 为什么放业务库而不是单独的库：这三张表是"业务运行账"，与用户/购买/维修同源同生命周期，
# 放一起便于 SQL 联查（如"某用户的某次运行花了多少 token"），也不需要额外的连接管理。
# 与 checkpointer 的库分开（data/database/checkpoints.db）的原因见 checkpoint.py：那张库的表
# 结构归 LangGraph 管，版本随库升级而变，不跟业务表混。
# 为什么 runs.user_id 不建外键：运行历史是审计账，用户被注销不该让账目连带消失——
# 与业务表（级联删除）是两种口径，故意区别对待。
# 安全口径：所有查询方法都要求"用户ID"参数并在 SQL 里强制过滤，get_run 命中不了就返回 None；
# 接口层（app.py）不允许前端传 user_id，身份一律取登录 token（与第 0 步一致）。
# 失败口径：写账/查账异常由调用方捕获后只记日志——记账再重要也不能影响回答。
# 验证：.venv/Scripts/python.exe -m service.database_service（自检，含三张新表）
#       .venv/Scripts/python.exe -m pytest tests/test_run_history.py
# ============================================================================================
