"""
ADR-0002 multi-layer safety contract — store + server-helper level tests.

Pins down every Layer-1 / Layer-2 / Layer-5 invariant. UI layers (3, 4)
are exercised manually because they're in static HTML.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Module-level import so load_dotenv (in server.py) doesn't race the
# per-test monkeypatch.setenv (same trick as test_stream_lifecycle.py).
import server  # noqa: E402, F401


@pytest.fixture
def store_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "live.db")
    yield
    _db.close_default()


def _bind_tiger(user_id, label, env, store):
    from brokers.base import TigerCredentials
    return store.bind(
        user_id=user_id, broker_type="tiger", label=label,
        creds=TigerCredentials(
            tiger_id="20151024",
            private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
            account="U999",
            license="TBNZ",
        ),
        actor="user", env=env,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: env='live' is now allowed in store.bind
# ─────────────────────────────────────────────────────────────────────────────

def test_bind_accepts_env_live(store_env):
    """ADR-0002: env='live' is no longer rejected at the store layer."""
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "live-main", "live", store)
    assert binding_id > 0

    [summary] = store.list_user_bindings("u:alice")
    assert summary.env == "live"
    # Default: view-only
    assert summary.live_orders_enabled is False


def test_bind_default_env_is_still_paper(store_env):
    """Existing behaviour preserved — paper is still the default."""
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "main", "paper", store)
    [summary] = store.list_user_bindings("u:alice")
    assert summary.env == "paper"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: per-binding live_orders_enabled flag — CRUD invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_is_live_orders_enabled_defaults_false(store_env):
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "live-main", "live", store)
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_is_live_orders_enabled_false_for_paper_bindings(store_env):
    """Even if we somehow set the flag on a paper binding, paper is paper."""
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "main", "paper", store)
    # Directly forge a 1 in the DB
    from brokers import _db
    conn = _db.init()
    conn.execute(
        "UPDATE broker_bindings SET live_orders_enabled = 1 WHERE id = ?",
        (binding_id,),
    )
    # Helper still says false because env != 'live'
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_set_live_orders_enabled_happy_path(store_env):
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "live-main", "live", store)

    ok = store.set_live_orders_enabled(
        binding_id, "u:alice", True, actor="user",
        ack="我确认开启下单",
    )
    assert ok is True
    assert store.is_live_orders_enabled(binding_id, "u:alice") is True

    # Flip back off works too
    store.set_live_orders_enabled(binding_id, "u:alice", False, actor="user")
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_set_live_orders_enabled_refuses_paper_binding(store_env):
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "main", "paper", store)
    ok = store.set_live_orders_enabled(binding_id, "u:alice", True, actor="user")
    assert ok is False
    # Flag remains 0 on the row (paper)
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_set_live_orders_enabled_refuses_wrong_user(store_env):
    """Bob can't toggle Alice's binding even with her id."""
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "live-main", "live", store)

    ok = store.set_live_orders_enabled(binding_id, "u:bob", True, actor="user")
    assert ok is False
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_set_live_orders_enabled_REFUSES_llm_actor(store_env):
    """
    Critical safety invariant: the LLM tool path MUST NOT be able to
    enable live orders. Defended at the store layer as a fail-safe.
    """
    from brokers.credentials_store import store, CredentialsStoreError

    binding_id = _bind_tiger("u:alice", "live-main", "live", store)
    with pytest.raises(CredentialsStoreError, match="MUST NOT be flipped by the LLM"):
        store.set_live_orders_enabled(
            binding_id, "u:alice", True, actor="llm",
            ack="我确认开启下单",
        )
    # Flag stays 0
    assert store.is_live_orders_enabled(binding_id, "u:alice") is False


def test_set_live_orders_writes_audit_with_ack(store_env):
    from brokers.credentials_store import store
    from brokers import _db

    binding_id = _bind_tiger("u:alice", "live-main", "live", store)
    store.set_live_orders_enabled(
        binding_id, "u:alice", True, actor="user",
        ack="我确认开启下单",
    )

    conn = _db.init()
    row = conn.execute(
        "SELECT actor, action, user_id, detail FROM broker_audit_log "
        "WHERE binding_id = ? AND action = 'use' "
        "ORDER BY id DESC LIMIT 1",
        (binding_id,),
    ).fetchone()
    assert row is not None
    actor, action, user_id, detail = row
    assert actor == "user"
    assert "live_orders_enabled = 1" in detail
    # ack phrase is recorded (truncated to 80) so forensic logs prove
    # the user explicitly opted in
    assert "我确认开启下单" in detail


def test_set_live_orders_failure_audits(store_env):
    """Failed flip (wrong user, paper binding) writes a 'fail' audit row."""
    from brokers.credentials_store import store
    from brokers import _db

    binding_id = _bind_tiger("u:alice", "main", "paper", store)
    store.set_live_orders_enabled(binding_id, "u:alice", True, actor="user")

    conn = _db.init()
    row = conn.execute(
        "SELECT actor, action, success FROM broker_audit_log "
        "WHERE binding_id = ? AND action = 'fail' "
        "ORDER BY id DESC LIMIT 1",
        (binding_id,),
    ).fetchone()
    assert row == ("user", "fail", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5: server's _live_order_blocked() helper
# ─────────────────────────────────────────────────────────────────────────────

def test_live_block_returns_none_for_mock_broker(store_env):
    """Mock broker has no concept of live; never blocked."""
    broker = MagicMock()
    broker.name = "mock-paper"
    assert server._live_order_blocked("u:alice", broker) is None


def test_live_block_returns_none_when_no_binding(store_env):
    """If user has no Tiger binding, the registry will fail elsewhere —
    _live_order_blocked itself must not return a misleading 'live' error."""
    broker = MagicMock()
    broker.name = "tiger-paper"
    assert server._live_order_blocked("u:never_bound", broker) is None


def test_live_block_returns_none_for_paper_binding(store_env):
    from brokers.credentials_store import store
    _bind_tiger("u:alice", "main", "paper", store)

    broker = MagicMock()
    broker.name = "tiger-paper"
    assert server._live_order_blocked("u:alice", broker) is None


def test_live_block_BLOCKS_unenabled_live_binding(store_env):
    """The whole point: env='live' + enabled=0 → block with clear message."""
    from brokers.credentials_store import store
    _bind_tiger("u:alice", "live-main", "live", store)

    broker = MagicMock()
    broker.name = "tiger-paper"  # broker name doesn't reflect env
    msg = server._live_order_blocked("u:alice", broker)
    assert msg is not None
    assert "实盘" in msg
    assert "live_orders_enabled" in msg


def test_live_block_allows_enabled_live_binding(store_env):
    from brokers.credentials_store import store
    binding_id = _bind_tiger("u:alice", "live-main", "live", store)
    store.set_live_orders_enabled(
        binding_id, "u:alice", True, actor="user",
        ack="我确认开启下单",
    )

    broker = MagicMock()
    broker.name = "tiger-paper"
    assert server._live_order_blocked("u:alice", broker) is None


def test_live_block_respects_per_user_isolation(store_env):
    """Bob's live binding doesn't affect Alice's order flow."""
    from brokers.credentials_store import store
    _bind_tiger("u:bob", "live-main", "live", store)
    _bind_tiger("u:alice", "main", "paper", store)

    broker = MagicMock()
    broker.name = "tiger-paper"
    # Alice has only paper → never blocked
    assert server._live_order_blocked("u:alice", broker) is None
    # Bob has live (unenabled) → blocked
    assert server._live_order_blocked("u:bob", broker) is not None
