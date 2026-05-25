"""
SSE broker stream — lifecycle helper functions (X6 c3).

Tests target the helpers around the /api/broker/stream endpoint:
  _ensure_tiger_push_started
  _populate_initial_state
  _schedule_push_stop_if_idle

The endpoint itself is exercised via manual smoke (curl) — TestClient for
SSE is fiddly and the helpers carry the real logic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: import server at module-collection time so its load_dotenv
# runs ONCE here, BEFORE any test's monkeypatch.setenv. Otherwise the
# first test to import server gets its KEK clobbered by .env.
import server  # noqa: E402, F401


@pytest.fixture(autouse=True)
def clean_singletons():
    from brokers.realtime_state import state
    from brokers.tiger_push import hub
    state.clear()
    hub.clear()
    yield
    state.clear()
    hub.clear()


def _patch_push_sdk(monkeypatch):
    """Stub out tigeropen so hub.start doesn't touch the network."""
    from brokers import tiger_push

    fake_config_cls = MagicMock()
    fake_config = MagicMock()
    fake_config.socket_host_port = ("ssl", "fake-host", 9883)
    fake_config_cls.return_value = fake_config
    fake_pc_cls = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(tiger_push, "_import_push_sdk", lambda: {
        "TigerOpenClientConfig": fake_config_cls,
        "PushClient": fake_pc_cls,
    })


@pytest.fixture
def bound_user(monkeypatch, tmp_path):
    """Set up a user with a Tiger binding in an isolated SQLite + KEK."""
    import os
    from cryptography.fernet import Fernet
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "stream.db")

    from brokers.base import TigerCredentials
    from brokers.credentials_store import store
    store.bind(
        user_id="u:alice",
        broker_type="tiger",
        label="main",
        creds=TigerCredentials(
            tiger_id="20151024",
            private_key=(
                "-----BEGIN PRIVATE KEY-----\n"
                "fakebase64payload\n"
                "-----END PRIVATE KEY-----"
            ),
            account="U99999999",
            license="TBNZ",
        ),
        actor="user",
    )
    yield "u:alice"
    _db.close_default()


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_tiger_push_started
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_push_starts_when_binding_exists(monkeypatch, bound_user):
    _patch_push_sdk(monkeypatch)
    import server
    from brokers.tiger_push import hub

    started = server._ensure_tiger_push_started(bound_user)
    assert started is True
    assert hub.get(bound_user) is not None


def test_ensure_push_idempotent(monkeypatch, bound_user):
    _patch_push_sdk(monkeypatch)
    import server
    from brokers.tiger_push import hub

    server._ensure_tiger_push_started(bound_user)
    w1 = hub.get(bound_user)
    server._ensure_tiger_push_started(bound_user)
    w2 = hub.get(bound_user)
    assert w1 is w2  # not re-created


def test_ensure_push_returns_false_when_no_binding(monkeypatch, tmp_path):
    """User logged in but no Tiger binding → no push started, return False."""
    import os
    from cryptography.fernet import Fernet
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "empty.db")

    _patch_push_sdk(monkeypatch)
    import server
    from brokers.tiger_push import hub

    started = server._ensure_tiger_push_started("u:nobody")
    assert started is False
    assert hub.get("u:nobody") is None
    _db.close_default()


def test_ensure_push_swallows_start_failure(monkeypatch, bound_user):
    """If hub.start raises (network down), function returns False, no exception."""
    from brokers import tiger_push
    monkeypatch.setattr(tiger_push, "_import_push_sdk", lambda: (_ for _ in ()).throw(
        RuntimeError("Tiger gateway unreachable")
    ))
    import server

    started = server._ensure_tiger_push_started(bound_user)
    assert started is False


# ─────────────────────────────────────────────────────────────────────────────
# _schedule_push_stop_if_idle
# ─────────────────────────────────────────────────────────────────────────────

def test_schedule_stop_actually_stops_when_idle(monkeypatch, bound_user):
    """No subscribers after timer fires → hub.stop is called."""
    _patch_push_sdk(monkeypatch)
    import server
    from brokers.tiger_push import hub
    from brokers.realtime_state import state

    server._ensure_tiger_push_started(bound_user)
    assert hub.get(bound_user) is not None

    # Override grace period to ~0 for the test
    monkeypatch.setattr(server, "_PUSH_GRACE_PERIOD_SEC", 0.05)
    # Make sure there are no subscribers
    assert state.subscriber_count(bound_user) == 0

    server._schedule_push_stop_if_idle(bound_user)
    time.sleep(0.2)  # let timer fire

    assert hub.get(bound_user) is None


def test_schedule_stop_does_not_stop_if_reconnected(monkeypatch, bound_user):
    """Subscriber reconnects during grace → push stays running."""
    _patch_push_sdk(monkeypatch)
    import server
    from brokers.tiger_push import hub
    from brokers.realtime_state import state

    server._ensure_tiger_push_started(bound_user)

    monkeypatch.setattr(server, "_PUSH_GRACE_PERIOD_SEC", 0.05)
    server._schedule_push_stop_if_idle(bound_user)

    # Re-subscribe before the timer fires
    q = state.subscribe(bound_user)
    time.sleep(0.2)

    # Hub still running because we had a subscriber when timer fired
    assert hub.get(bound_user) is not None
    state.unsubscribe(bound_user, q)


# ─────────────────────────────────────────────────────────────────────────────
# _populate_initial_state
# ─────────────────────────────────────────────────────────────────────────────

def test_populate_initial_state_pulls_from_adapter(monkeypatch, bound_user):
    """Initial REST snapshot should land in RealtimeStateStore."""
    _patch_push_sdk(monkeypatch)
    import server
    from brokers.realtime_state import state

    # Fake adapter returned by the registry
    fake_adapter = MagicMock()
    fake_adapter.get_account.return_value = NS(
        to_dict=lambda: {
            "cash": 5000.0, "buying_power": 5000.0, "equity": 5000.0,
            "currency": "USD",
        }
    )
    fake_adapter.list_positions.return_value = [
        NS(to_dict=lambda: {
            "symbol": "AAPL", "qty": 10.0,
            "avg_entry_price": 150.0, "market_value": 1600.0,
            "unrealized_pl": 100.0, "unrealized_pl_pct": 6.7,
            "current_price": 160.0,
        }),
    ]
    fake_adapter.list_orders.return_value = []

    from brokers import registry as broker_registry
    monkeypatch.setattr(
        broker_registry._registry, "get",
        lambda user_id, broker_type=None, label=None: fake_adapter,
    )

    server._populate_initial_state(bound_user)

    snap = state.snapshot(bound_user)
    assert snap["account"]["cash"] == 5000.0
    assert len(snap["positions"]) == 1
    assert snap["positions"][0]["symbol"] == "AAPL"


def test_populate_initial_state_swallows_errors(monkeypatch, bound_user):
    """If adapter raises, function returns silently (push will fill in)."""
    import server

    from brokers import registry as broker_registry
    monkeypatch.setattr(
        broker_registry._registry, "get",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no binding")),
    )

    server._populate_initial_state(bound_user)  # should not raise
