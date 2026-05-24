"""
brokers/credentials_store.py — encrypted per-user broker credential CRUD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def store_env(monkeypatch, tmp_path):
    """
    Isolated environment per test:
      - a fresh KEK only (any pre-existing BROKER_KEK_v* is cleared)
      - a fresh SQLite db at tmp_path
      - the BrokerRegistry adapter cache is cleared
      - a freshly-constructed CredentialsStore bound to the tmp connection
    Yields (store_instance, conn).
    """
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    conn = _db.init(db_path=tmp_path / "creds.db")

    from brokers.credentials_store import CredentialsStore
    from brokers.registry import _registry
    _registry.clear()

    s = CredentialsStore(conn=conn)
    yield s, conn

    conn.close()
    _registry.clear()


# ─────────────────────────────────────────────────────────────────────────────
# bind
# ─────────────────────────────────────────────────────────────────────────────

def test_bind_persists_encrypted_row(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    creds = AlpacaCredentials(api_key="PKtest", api_secret="abc123")
    binding_id = s.bind(
        user_id="u_42", broker_type="alpaca", label="main",
        creds=creds, actor="user",
    )
    assert binding_id > 0

    # Ciphertext must not contain the plaintext key.
    row = conn.execute(
        "SELECT encrypted_credential FROM broker_bindings WHERE id = ?",
        (binding_id,),
    ).fetchone()
    assert b"PKtest" not in row[0]
    assert b"abc123" not in row[0]


def test_bind_then_load_returns_same_credentials(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials

    original = AlpacaCredentials(
        api_key="PKtest", api_secret="abc123",
        base_url="https://paper-api.alpaca.markets",
    )
    s.bind("u_42", "alpaca", "main", creds=original, actor="user")
    loaded = s.load("u_42", "alpaca", "main", actor="system")
    assert loaded == original


def test_bind_writes_audit_row(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    s.bind("u_42", "alpaca", "main",
           AlpacaCredentials(api_key="k", api_secret="s"),
           actor="user")
    row = conn.execute(
        "SELECT actor, action, user_id, success FROM broker_audit_log WHERE action='bind'"
    ).fetchone()
    assert row == ("user", "bind", "u_42", 1)


def test_bind_duplicate_label_raises_and_audits_failure(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import CredentialsStoreError

    creds = AlpacaCredentials(api_key="k", api_secret="s")
    s.bind("u_42", "alpaca", "main", creds, actor="user")
    with pytest.raises(CredentialsStoreError, match="already exists"):
        s.bind("u_42", "alpaca", "main", creds, actor="user")

    fail_rows = conn.execute(
        "SELECT COUNT(*) FROM broker_audit_log WHERE action='bind' AND success=0"
    ).fetchone()
    assert fail_rows[0] == 1


def test_bind_mismatched_broker_type_raises(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import CredentialsStoreError

    with pytest.raises(CredentialsStoreError, match="mismatches"):
        s.bind("u_42", "tiger", "main",
               AlpacaCredentials(api_key="k", api_secret="s"),
               actor="user")


def test_bind_rejects_live_env_currently(store_env):
    """X3 c2: env can only be 'paper' or 'live'. live is allowed at the
    schema level but the trading-safety rules forbid using it in code paths.
    The store itself doesn't enforce paper-only (that's a higher layer);
    here we just verify the enum is honored."""
    s, _ = store_env
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import CredentialsStoreError

    creds = AlpacaCredentials(api_key="k", api_secret="s")
    with pytest.raises(CredentialsStoreError, match="env must be"):
        s.bind("u_42", "alpaca", "main", creds, actor="user", env="mainnet")


# ─────────────────────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────────────────────

def test_load_unknown_returns_none(store_env):
    s, _ = store_env
    assert s.load("u_42", "alpaca", "main", actor="system") is None
    assert s.load("u_42", "alpaca", None, actor="system") is None


def test_load_label_none_picks_oldest(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials
    import time

    s.bind("u_42", "alpaca", "first",
           AlpacaCredentials(api_key="k1", api_secret="s1"),
           actor="user")
    time.sleep(1.01)  # advance created_at by ≥ 1 s (column is integer seconds)
    s.bind("u_42", "alpaca", "second",
           AlpacaCredentials(api_key="k2", api_secret="s2"),
           actor="user")

    loaded = s.load("u_42", "alpaca", label=None, actor="system")
    assert loaded.api_key == "k1"


def test_load_updates_last_used_at(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    binding_id = s.bind("u_42", "alpaca", "main",
                        AlpacaCredentials(api_key="k", api_secret="s"),
                        actor="user")
    assert conn.execute(
        "SELECT last_used_at FROM broker_bindings WHERE id = ?", (binding_id,)
    ).fetchone()[0] is None

    s.load("u_42", "alpaca", "main", actor="llm")
    assert conn.execute(
        "SELECT last_used_at FROM broker_bindings WHERE id = ?", (binding_id,)
    ).fetchone()[0] is not None


def test_load_emits_use_audit(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    s.bind("u_42", "alpaca", "main",
           AlpacaCredentials(api_key="k", api_secret="s"),
           actor="user")
    s.load("u_42", "alpaca", "main", actor="llm")

    row = conn.execute(
        "SELECT actor, action, success FROM broker_audit_log WHERE action='use'"
    ).fetchone()
    assert row == ("llm", "use", 1)


def test_load_corrupted_ciphertext_audits_fail(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials
    from brokers.crypto import CryptoError

    binding_id = s.bind("u_42", "alpaca", "main",
                        AlpacaCredentials(api_key="k", api_secret="s"),
                        actor="user")
    # Tamper the ciphertext on disk.
    conn.execute(
        "UPDATE broker_bindings SET encrypted_credential = ? WHERE id = ?",
        (b"\x00" * 32, binding_id),
    )
    with pytest.raises(CryptoError):
        s.load("u_42", "alpaca", "main", actor="llm")
    # Failure was audited.
    row = conn.execute(
        "SELECT actor, action, success FROM broker_audit_log "
        "WHERE action='fail' AND user_id='u_42'"
    ).fetchone()
    assert row == ("llm", "fail", 0)


# ─────────────────────────────────────────────────────────────────────────────
# list_user_bindings
# ─────────────────────────────────────────────────────────────────────────────

def test_list_returns_metadata_no_ciphertext(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import BindingSummary

    s.bind("u_42", "alpaca", "main",
           AlpacaCredentials(api_key="k", api_secret="s"),
           actor="user")
    [summary] = s.list_user_bindings("u_42")
    assert isinstance(summary, BindingSummary)
    assert summary.user_id == "u_42"
    assert summary.broker_type == "alpaca"
    assert summary.label == "main"
    assert summary.env == "paper"
    # BindingSummary has no ciphertext field at all (verify via dir())
    assert "encrypted_credential" not in dir(summary)
    assert "dek_wrapped" not in dir(summary)


def test_list_filters_by_user(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials

    s.bind("u_alice", "alpaca", "main",
           AlpacaCredentials(api_key="ak", api_secret="as"),
           actor="user")
    s.bind("u_bob", "alpaca", "main",
           AlpacaCredentials(api_key="bk", api_secret="bs"),
           actor="user")

    assert len(s.list_user_bindings("u_alice")) == 1
    assert len(s.list_user_bindings("u_bob")) == 1
    assert len(s.list_user_bindings("u_carol")) == 0


# ─────────────────────────────────────────────────────────────────────────────
# unbind
# ─────────────────────────────────────────────────────────────────────────────

def test_unbind_removes_row(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    binding_id = s.bind("u_42", "alpaca", "main",
                        AlpacaCredentials(api_key="k", api_secret="s"),
                        actor="user")
    assert s.unbind(binding_id, "u_42", actor="user") is True
    count = conn.execute(
        "SELECT COUNT(*) FROM broker_bindings WHERE id = ?", (binding_id,)
    ).fetchone()[0]
    assert count == 0


def test_unbind_wrong_user_returns_false_and_keeps_row(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    binding_id = s.bind("u_alice", "alpaca", "main",
                        AlpacaCredentials(api_key="k", api_secret="s"),
                        actor="user")
    assert s.unbind(binding_id, "u_bob", actor="user") is False
    count = conn.execute(
        "SELECT COUNT(*) FROM broker_bindings WHERE id = ?", (binding_id,)
    ).fetchone()[0]
    assert count == 1


def test_unbind_emits_audit(store_env):
    s, conn = store_env
    from brokers.base import AlpacaCredentials

    binding_id = s.bind("u_42", "alpaca", "main",
                        AlpacaCredentials(api_key="k", api_secret="s"),
                        actor="user")
    s.unbind(binding_id, "u_42", actor="user")
    row = conn.execute(
        "SELECT actor, action, user_id, success FROM broker_audit_log "
        "WHERE action='unbind'"
    ).fetchone()
    assert row == ("user", "unbind", "u_42", 1)


# ─────────────────────────────────────────────────────────────────────────────
# Tiger credentials round-trip (X4)
# ─────────────────────────────────────────────────────────────────────────────

def test_tiger_bind_then_load_preserves_all_fields(store_env):
    s, _ = store_env
    from brokers.base import TigerCredentials

    original = TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nFAKE-RSA-PEM-BLOB\n-----END PRIVATE KEY-----",
        account="U99999999",
        license="TBNZ",
    )
    s.bind("u_42", "tiger", "main", creds=original, actor="user")
    loaded = s.load("u_42", "tiger", "main", actor="system")
    assert loaded == original


def test_tiger_private_key_not_in_ciphertext(store_env):
    s, conn = store_env
    from brokers.base import TigerCredentials

    secret = "SECRET-MARKER-XYZ"
    pem = f"-----BEGIN PRIVATE KEY-----\n{secret}\n-----END PRIVATE KEY-----"
    s.bind("u_42", "tiger", "main",
           TigerCredentials(tiger_id="x", private_key=pem, account="U999"),
           actor="user")
    row = conn.execute(
        "SELECT encrypted_credential FROM broker_bindings WHERE broker_type='tiger'"
    ).fetchone()
    assert secret.encode() not in row[0]


# ─────────────────────────────────────────────────────────────────────────────
# get_default_label
# ─────────────────────────────────────────────────────────────────────────────

def test_get_default_label_returns_first_bound(store_env):
    s, _ = store_env
    from brokers.base import AlpacaCredentials
    import time

    s.bind("u_42", "alpaca", "first",
           AlpacaCredentials(api_key="k1", api_secret="s1"),
           actor="user")
    time.sleep(1.01)
    s.bind("u_42", "alpaca", "second",
           AlpacaCredentials(api_key="k2", api_secret="s2"),
           actor="user")
    assert s.get_default_label("u_42", "alpaca") == "first"


def test_get_default_label_none_when_no_binding(store_env):
    s, _ = store_env
    assert s.get_default_label("u_42", "alpaca") is None


# ─────────────────────────────────────────────────────────────────────────────
# build_credentials — JSON payload → Credentials subclass (X5 c1)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_credentials_alpaca():
    from brokers.credentials_store import build_credentials
    from brokers.base import AlpacaCredentials

    creds = build_credentials("alpaca", {
        "api_key": "PKtest", "api_secret": "ssstest",
    })
    assert isinstance(creds, AlpacaCredentials)
    assert creds.api_key == "PKtest"
    assert creds.api_secret == "ssstest"
    assert "paper-api.alpaca.markets" in creds.base_url


def test_build_credentials_tiger():
    from brokers.credentials_store import build_credentials
    from brokers.base import TigerCredentials

    creds = build_credentials("tiger", {
        "tiger_id": "20151024",
        "private_key": "-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        "account": "U99999999",
        "license": "TBSG",
    })
    assert isinstance(creds, TigerCredentials)
    assert creds.tiger_id == "20151024"
    assert creds.account == "U99999999"
    assert creds.license == "TBSG"


def test_build_credentials_tiger_defaults_license_to_tbnz():
    from brokers.credentials_store import build_credentials
    creds = build_credentials("tiger", {
        "tiger_id": "1", "private_key": "x", "account": "U1",
    })
    assert creds.license == "TBNZ"


def test_build_credentials_mock_with_initial_cash():
    from brokers.credentials_store import build_credentials
    from brokers.base import MockCredentials

    creds = build_credentials("mock", {"initial_cash": 50000})
    assert isinstance(creds, MockCredentials)
    assert creds.initial_cash == 50000.0


def test_build_credentials_unknown_broker_raises():
    from brokers.credentials_store import build_credentials
    import pytest

    with pytest.raises(ValueError, match="Unsupported broker_type"):
        build_credentials("ftx", {})


def test_build_credentials_non_dict_payload_raises():
    from brokers.credentials_store import build_credentials
    import pytest

    with pytest.raises(ValueError, match="must be an object"):
        build_credentials("alpaca", "not a dict")
