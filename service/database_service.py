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
"""


def _hash_password(password: str) -> str:
    """密码哈希：bcrypt（自带随机盐）；按 72 字节截断（bcrypt 算法输入上限）"""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


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
            }

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
