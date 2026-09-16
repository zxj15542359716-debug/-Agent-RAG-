#密码哈希迁移测试：历史 sha256 哈希可正常登录，且登录成功后自动升级为 bcrypt
import hashlib

from service import database_service as dbs


def _stored_hash(db, user_id: str) -> str:
    with db._tx() as conn:   #测试内直接用内部事务上下文读取原始哈希（非公开 API）
        return conn.execute("SELECT 密码 FROM users WHERE 用户ID = ?", (user_id,)).fetchone()["密码"]


def test_old_sha256_hash_upgrades_to_bcrypt(tmp_path):
    """老格式（sha256+固定盐）密码：先可登录，登录后自动升级，旧密码继续有效"""
    db = dbs.DatabaseService(db_path=str(tmp_path / "m.db"), import_seed=False)
    old_hash = hashlib.sha256((dbs._PASSWORD_SALT + "pw123456").encode("utf-8")).hexdigest()
    with db._tx() as conn:
        conn.execute("INSERT INTO users (用户ID, 密码) VALUES (?, ?)", ("1000", old_hash))

    assert _stored_hash(db, "1000") == old_hash          #升级前：旧格式
    assert db.verify_user("1000", "pw123456") is True    #旧格式校验通过
    assert _stored_hash(db, "1000").startswith("$2")     #已惰性升级为 bcrypt
    assert db.verify_user("1000", "pw123456") is True    #升级后旧密码依然可登录
    assert db.verify_user("1000", "wrong-password") is False


def test_new_user_hash_is_bcrypt(tmp_path):
    """新注册用户直接使用 bcrypt 存储"""
    db = dbs.DatabaseService(db_path=str(tmp_path / "n.db"), import_seed=False)
    db.create_user("1001", "pw123456")
    assert _stored_hash(db, "1001").startswith("$2")
    assert db.verify_user("1001", "pw123456") is True
