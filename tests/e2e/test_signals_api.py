"""P3 Task 5：信号看板 + 持仓回写端点。写路径过 owner-lock + require_user 双闸；
CSV 方言在前端解析（V8），server 只认规范化行；'signal_assumed' 不许经 API 写。
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("duckdb")

KEY = "test-agent-key-123"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("WARMUP", "0")
    monkeypatch.setenv("AGENT_ACCESS_KEY", KEY)
    monkeypatch.delenv("REDIS_URL", raising=False)
    from ashare.data import _ledger
    monkeypatch.setattr(_ledger, "DEFAULT_LEDGER_PATH", str(tmp_path / "ledger.duckdb"))
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app, headers={"X-Agent-Key": KEY}) as c:
        yield c


def _login_owner(client):
    from auth.owner_gate import owner_email
    from auth.users import upsert_user
    from auth.sessions import create_web_session
    email = owner_email()
    user = upsert_user(provider="email", provider_sub=email, email=email,
                       name="o", avatar_url="")
    client.cookies.set("qa_auth", create_web_session(user.user_id))


def test_reads_are_empty_safe(client):
    _login_owner(client)
    assert client.get("/api/signals/latest").json() == {"plan": None}
    assert client.get("/api/signals").json() == {"plans": []}
    r = client.get("/api/portfolio/positions").json()
    assert r == {"as_of": None, "positions": []}
    assert client.get("/api/portfolio/confirms", params={"as_of": "2026-08-31"}).json()["confirms"] == []


def test_write_needs_login_even_with_key(client):
    r = client.post("/api/portfolio/confirm",
                    json={"as_of": "2026-08-31", "confirms": []})
    assert r.status_code == 401                     # key 过了中间件，写路径还要 require_user


def test_reconcile_roundtrip_and_source_rules(client):
    _login_owner(client)
    bad = client.post("/api/portfolio/reconcile",
                      json={"as_of": "2026-08-31", "source": "signal_assumed", "rows": []})
    assert bad.status_code == 422                   # 系统推演口径不许经 API 伪装成人工来源
    ok = client.post("/api/portfolio/reconcile",
                     json={"as_of": "2026-08-31", "source": "reconcile_csv",
                           "rows": [{"ts_code": "600000.SH", "shares": 200, "avg_cost": 10.5}]})
    assert ok.json()["success"] is True
    got = client.get("/api/portfolio/positions").json()
    assert got["as_of"] == "2026-08-31"
    assert got["positions"][0]["ts_code"] == "600000.SH" and got["positions"][0]["source"] == "reconcile_csv"


def test_confirm_state_whitelist_and_roundtrip(client):
    _login_owner(client)
    bad = client.post("/api/portfolio/confirm",
                      json={"as_of": "2026-08-31",
                            "confirms": [{"ts_code": "600000.SH", "state": "done"}]})
    assert bad.status_code == 422
    ok = client.post("/api/portfolio/confirm",
                     json={"as_of": "2026-08-31",
                           "confirms": [{"ts_code": "600000.SH", "state": "partial",
                                         "filled_shares": 100, "note": "半成"}]})
    assert ok.json()["success"] is True
    got = client.get("/api/portfolio/confirms", params={"as_of": "2026-08-31"}).json()["confirms"]
    assert got[0]["state"] == "partial" and got[0]["filled_shares"] == 100
