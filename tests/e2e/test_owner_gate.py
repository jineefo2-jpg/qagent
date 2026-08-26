"""单所有者锁定（auth/owner_gate.py）的行为契约。

守的是「从输入框到后台」链条的**后台**那一半：无论前端怎么被绕过，
中间件对每个受保护请求都要求「固定密钥一致 ∧ 会话用户是所有者」。
用 /docs 当受保护路径的探针 —— 它被锁定覆盖、又不牵扯任何业务依赖，
探针 200 说明的是「闸放行了」，与业务路由自身的失败区分开。
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

KEY = "test-agent-key-123"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("WARMUP", "0")                 # 别在测试里预热嵌入模型
    monkeypatch.setenv("AGENT_ACCESS_KEY", KEY)
    monkeypatch.delenv("REDIS_URL", raising=False)    # cache 走进程内 dict
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


def _login(client, email: str) -> None:
    """直接种会话 cookie —— 测的是中间件，不是登录流程本身。"""
    from auth.users import upsert_user
    from auth.sessions import create_web_session
    user = upsert_user(provider="email", provider_sub=email, email=email,
                       name=email.split("@")[0], avatar_url="")
    client.cookies.set("qa_auth", create_web_session(user.user_id))


def test_no_key_is_rejected(client):
    r = client.get("/docs")
    assert r.status_code == 403 and r.json()["owner_lock"] is True


def test_wrong_key_is_rejected(client):
    r = client.get("/docs", headers={"X-Agent-Key": "wrong"})
    assert r.status_code == 403


def test_key_without_login_is_401(client):
    r = client.get("/docs", headers={"X-Agent-Key": KEY})
    assert r.status_code == 401


def test_key_with_non_owner_session_is_403(client):
    _login(client, "intruder@example.com")
    r = client.get("/docs", headers={"X-Agent-Key": KEY})
    assert r.status_code == 403 and "仅限所有者" in r.json()["error"]


def test_key_with_owner_session_passes(client):
    from auth.owner_gate import owner_email
    _login(client, owner_email())
    r = client.get("/docs", headers={"X-Agent-Key": KEY})
    assert r.status_code == 200


def test_key_via_cookie_passes_for_eventsource(client):
    """EventSource 发不了自定义头，key 走 cookie 必须同样有效。"""
    from auth.owner_gate import owner_email
    _login(client, owner_email())
    client.cookies.set("agent_key", KEY)
    r = client.get("/docs")
    assert r.status_code == 200


def test_missing_server_key_fails_closed(client, monkeypatch):
    monkeypatch.delenv("AGENT_ACCESS_KEY")
    r = client.get("/docs", headers={"X-Agent-Key": KEY})
    assert r.status_code == 503 and "AGENT_ACCESS_KEY" in r.json()["error"]


def test_open_paths_stay_reachable(client):
    assert client.get("/api/auth/providers").json()["owner_lock"] is True
    assert client.get("/api/me").status_code == 401          # 自身的未登录 401，不是闸的
    assert client.get("/").status_code == 200                # 登录页壳必须能加载


def test_key_login_with_owner_email_creates_session(client):
    """访问密钥直登（2026-08-26 用户裁决）：key + 所有者邮箱 = 建会话，跳过验证码。"""
    from auth.owner_gate import owner_email
    r = client.post("/auth/key_login", json={"email": owner_email()},
                    headers={"X-Agent-Key": KEY})
    assert r.json()["success"] is True
    assert client.get("/docs", headers={"X-Agent-Key": KEY}).status_code == 200


def test_key_login_refuses_non_owner_email(client):
    r = client.post("/auth/key_login", json={"email": "intruder@example.com"},
                    headers={"X-Agent-Key": KEY})
    assert r.json()["success"] is False


def test_key_login_refuses_wrong_or_missing_key(client):
    from auth.owner_gate import owner_email
    assert client.post("/auth/key_login", json={"email": owner_email()},
                       headers={"X-Agent-Key": "wrong"}).json()["success"] is False
    assert client.post("/auth/key_login",
                       json={"email": owner_email()}).json()["success"] is False


def test_key_login_fails_closed_without_server_key(client, monkeypatch):
    from auth.owner_gate import owner_email
    monkeypatch.delenv("AGENT_ACCESS_KEY")
    assert client.post("/auth/key_login", json={"email": owner_email()},
                       headers={"X-Agent-Key": KEY}).json()["success"] is False


def test_send_code_refuses_non_owner_email(client):
    r = client.post("/auth/email/send_code", json={"email": "intruder@example.com"})
    body = r.json()
    assert body["success"] is False and "未被授权" in body["error"]
