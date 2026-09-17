#鉴权回归测试：锁住"登录签发 token、业务接口从 token 取身份、防爆破"的新行为，
#防止后续改动重新引入越权（请求体自报 user_id）等安全问题。
from datetime import datetime, timedelta, timezone

import jwt
import pytest

_TEST_SECRET = "test-secret-not-for-production-0123456789abcdef"


def login(client, user_id, password):
    return client.post("/api/login", json={"user_id": user_id, "password": password})


@pytest.fixture()
def auth_client(client_and_db):
    """带测试账号的客户端：2483 / pw123456"""
    client, db = client_and_db
    db.create_user("2483", "pw123456")
    return client, db


def test_login_returns_token(auth_client):
    """登录成功签发 JWT，sub 为登录用户"""
    client, _ = auth_client
    data = login(client, "2483", "pw123456").json()
    assert data["ok"] is True
    payload = jwt.decode(data["token"], _TEST_SECRET, algorithms=["HS256"])
    assert payload["sub"] == "2483"


def test_chat_requires_token(auth_client):
    """无凭证访问 /api/chat：401（原实现会直接信任请求体 user_id）"""
    client, _ = auth_client
    resp = client.post("/api/chat", json={"query": "你好", "user_id": "2483"})
    assert resp.status_code == 401


def test_chat_rejects_garbage_token(auth_client):
    """伪造的 token：401"""
    client, _ = auth_client
    resp = client.post("/api/chat", json={"query": "你好"},
                       headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_chat_rejects_expired_token(auth_client):
    """过期 token：401"""
    client, _ = auth_client
    now = datetime.now(timezone.utc)
    expired = jwt.encode(
        {"sub": "2483", "iat": now - timedelta(hours=3), "exp": now - timedelta(hours=1)},
        _TEST_SECRET, algorithm="HS256")
    resp = client.post("/api/chat", json={"query": "你好"},
                       headers={"Authorization": f"Bearer {expired}"})
    assert resp.status_code == 401


def test_report_uses_token_identity(auth_client):
    """带 token 上报：身份取自 token，记录归属登录用户本人"""
    client, db = auth_client
    token = login(client, "2483", "pw123456").json()["token"]
    resp = client.post("/api/report",
                       json={"device_type": "键盘", "fault": "空格键回弹卡涩"},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    reports = db.get_reports_by_user("2483")
    assert len(reports) == 1
    assert reports[0]["故障描述"] == "空格键回弹卡涩"


def test_report_requires_token(client_and_db):
    """无凭证上报：401"""
    client, _ = client_and_db
    resp = client.post("/api/report", json={"device_type": "键盘", "fault": "x"})
    assert resp.status_code == 401


def test_login_locked_after_failures(client_and_db):
    """防爆破：连续 5 次失败后锁定，正确密码也被拒绝。

    使用独立用户ID（9990）避免影响其他用例；TestClient 的客户端 IP 固定为 testclient。
    """
    client, _ = client_and_db
    for _ in range(5):
        assert login(client, "9990", "wrong-password").json()["ok"] is False
    data = login(client, "9990", "wrong-password").json()
    assert data["ok"] is False
    assert "尝试过于频繁" in data["message"]


def test_register_password_min_length(client_and_db):
    """注册密码最小长度 6 位"""
    client, _ = client_and_db
    assert client.post("/api/register", json={"password": "12345"}).json()["ok"] is False
    data = client.post("/api/register", json={"password": "123456"}).json()
    assert data["ok"] is True
    assert len(data["user_id"]) == 4


# ---------- 第2步·2.4：运行历史与用量接口（同样是"身份取自 token"口径） ----------

def test_runs_and_usage_require_token(auth_client):
    """无凭证访问运行历史 / 用量汇总：401"""
    client, _ = auth_client
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/usage").status_code == 401


def test_runs_only_returns_own_records(auth_client):
    """运行历史只返回本人记录：即使用别人的 run_id 也查不到（SQL 强制按登录用户过滤）"""
    client, db = auth_client
    db.start_run("run-of-2483", "2483", "s1", "我的问题")
    db.start_run("run-of-9999", "9999", "s1", "别人的问题")
    token = login(client, "2483", "pw123456").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    items = client.get("/api/runs", headers=headers).json()["items"]
    assert [r["run_id"] for r in items] == ["run-of-2483"]
    #用量汇总同样只看得到本人的流水
    db.add_usage_event(source="chat", provider="deepseek", model="deepseek-v4-pro",
                       unit="token", amount=321, run_id="run-of-2483")
    db.add_usage_event(source="chat", provider="deepseek", model="deepseek-v4-pro",
                       unit="token", amount=999, run_id="run-of-9999")
    usage = client.get("/api/usage", headers=headers).json()["items"]
    assert sum(u["amount"] for u in usage) == 321
