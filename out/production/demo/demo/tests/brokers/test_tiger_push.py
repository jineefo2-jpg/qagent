"""
brokers/tiger_push.py — PushClient wrapper + hub.

Tests mock the SDK entirely so they never touch the network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clean_state():
    """Each test gets a fresh state + empty hub (singletons would leak otherwise)."""
    from brokers.realtime_state import state
    from brokers.tiger_push import hub
    state.clear()
    hub.clear()
    yield
    state.clear()
    hub.clear()


def _make_creds(**overrides):
    from brokers.base import TigerCredentials
    defaults = dict(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nxxx\n-----END PRIVATE KEY-----",
        account="U99999999",
        license="TBNZ",
    )
    defaults.update(overrides)
    return TigerCredentials(**defaults)


def _make_fake_sdk(push_client_factory=None):
    """Build a fake _import_push_sdk() return value."""
    fake_config_cls = MagicMock()
    fake_config = MagicMock()
    fake_config.socket_host_port = ("ssl", "openapi.tigerfintech.com", 9883)
    fake_config_cls.return_value = fake_config

    if push_client_factory is None:
        push_client_factory = lambda *a, **kw: MagicMock()  # noqa: E731
    fake_pc_cls = MagicMock(side_effect=push_client_factory)

    return {
        "TigerOpenClientConfig": fake_config_cls,
        "PushClient": fake_pc_cls,
    }


def _patch_sdk(monkeypatch, sdk=None):
    if sdk is None:
        sdk = _make_fake_sdk()
    from brokers import tiger_push
    monkeypatch.setattr(tiger_push, "_import_push_sdk", lambda: sdk)
    return sdk


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def test_start_constructs_pushclient_and_connects(monkeypatch):
    captured = {}
    def factory(host, port, use_ssl):
        client = MagicMock()
        captured["host"] = host
        captured["port"] = port
        captured["use_ssl"] = use_ssl
        captured["client"] = client
        return client
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    assert captured["host"] == "openapi.tigerfintech.com"
    assert captured["port"] == 9883
    assert captured["use_ssl"] is True
    captured["client"].connect.assert_called_once_with("20151024", "xxx")


def test_start_hooks_all_callbacks(monkeypatch):
    client_holder = {}
    def factory(*a, **kw):
        c = MagicMock()
        client_holder["c"] = c
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    c = client_holder["c"]

    # Every event callback must be a bound method on the wrapper
    for attr in (
        "connect_callback", "disconnect_callback", "error_callback",
        "asset_changed", "position_changed", "order_changed", "quote_changed",
    ):
        assigned = getattr(c, attr)
        assert callable(assigned), f"{attr} not set"


def test_start_does_NOT_subscribe_before_handshake(monkeypatch):
    """Pre-connect subscribes get silently dropped by tigeropen — must wait."""
    client_holder = {}
    def factory(*a, **kw):
        c = MagicMock()
        client_holder["c"] = c
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    c = client_holder["c"]

    c.subscribe_asset.assert_not_called()
    c.subscribe_position.assert_not_called()
    c.subscribe_order.assert_not_called()


def test_subscribe_happens_inside_connect_callback(monkeypatch):
    """The wrapper queues subscriptions for AFTER the WebSocket handshake."""
    client_holder = {}
    def factory(*a, **kw):
        c = MagicMock()
        client_holder["c"] = c
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    c = client_holder["c"]

    # SDK invokes connect_callback once the WebSocket opens
    c.connect_callback(frame=None)

    c.subscribe_asset.assert_called_once_with(account="U99999999")
    c.subscribe_position.assert_called_once_with(account="U99999999")
    c.subscribe_order.assert_called_once_with(account="U99999999")
    assert w.is_connected()


def test_stop_unsubscribes_and_disconnects(monkeypatch):
    client_holder = {}
    def factory(*a, **kw):
        c = MagicMock()
        client_holder["c"] = c
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    c = client_holder["c"]
    c.connect_callback(None)  # finish handshake
    assert w.is_connected()

    w.stop()
    c.unsubscribe_asset.assert_called_once()
    c.unsubscribe_position.assert_called_once()
    c.unsubscribe_order.assert_called_once()
    c.disconnect.assert_called_once()
    assert not w.is_connected()


def test_stop_is_idempotent(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    w.stop()
    w.stop()  # should not raise


def test_stop_swallows_errors(monkeypatch):
    """Cleanup never raises — even if the SDK throws during unsubscribe."""
    def factory(*a, **kw):
        c = MagicMock()
        c.unsubscribe_asset.side_effect = RuntimeError("network gone")
        c.disconnect.side_effect = RuntimeError("network gone")
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import TigerPushClientWrapper
    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()
    w.stop()  # MUST NOT raise
    assert not w.is_connected()


# ─────────────────────────────────────────────────────────────────────────────
# Callback → realtime_state normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_asset_changed_writes_to_realtime_state(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper
    from brokers.realtime_state import state

    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    frame = NS(cash=12000, buying_power=15000, gross_position_value=3000,
                currency="USD", account="U99999999")
    w._on_asset_changed(frame)

    snap = state.snapshot("u:alice")
    assert snap["account"]["cash"] == 12000
    assert snap["account"]["equity"] == 12000 + 3000


def test_position_changed_writes_to_realtime_state(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper
    from brokers.realtime_state import state

    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    frame = NS(
        contract=NS(symbol="AAPL"),
        quantity=10, average_cost=150.0, market_value=1600.0,
        unrealized_pnl=100.0, market_price=160.0,
    )
    w._on_position_changed(frame)

    snap = state.snapshot("u:alice")
    [pos] = snap["positions"]
    assert pos["symbol"] == "AAPL"
    assert pos["qty"] == 10
    assert pos["market_value"] == 1600.0


def test_order_changed_writes_to_realtime_state(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper
    from brokers.realtime_state import state

    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    frame = NS(
        id=12345, contract=NS(symbol="AAPL"),
        action="BUY", quantity=10, filled=0,
        limit_price=150.0, status="NEW", order_time="t",
    )
    w._on_order_changed(frame)

    snap = state.snapshot("u:alice")
    [order] = snap["orders"]
    assert order["broker_order_id"] == "12345"
    assert order["status"] == "NEW"


def test_quote_changed_writes_to_realtime_state(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper
    from brokers.realtime_state import state

    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    frame = NS(symbol="AAPL", latest_price=160.5,
                change=1.5, change_pct=0.94, volume=12345, timestamp=1700000000)
    w._on_quote_changed(frame)

    snap = state.snapshot("u:alice")
    assert snap["quote_count"] == 1


def test_callback_exception_does_not_crash_connection(monkeypatch):
    """A bad payload from Tiger must not kill the push connection."""
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import TigerPushClientWrapper

    w = TigerPushClientWrapper("u:alice", _make_creds())
    w.start()

    # Frame missing every field we read
    bad = NS()
    # Should swallow internally (no exception escapes)
    w._on_asset_changed(bad)
    w._on_position_changed(bad)
    w._on_order_changed(bad)
    w._on_quote_changed(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Hub multi-user
# ─────────────────────────────────────────────────────────────────────────────

def test_hub_starts_only_once_per_user(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import hub

    w1 = hub.start("u:alice", _make_creds(tiger_id="A"))
    w2 = hub.start("u:alice", _make_creds(tiger_id="A"))
    assert w1 is w2


def test_hub_two_users_get_independent_wrappers(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import hub

    wa = hub.start("u:alice", _make_creds(tiger_id="A", account="UA"))
    wb = hub.start("u:bob",   _make_creds(tiger_id="B", account="UB"))
    assert wa is not wb
    assert wa.account == "UA"
    assert wb.account == "UB"


def test_hub_stop_removes_user(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import hub
    hub.start("u:alice", _make_creds())
    assert hub.get("u:alice") is not None
    hub.stop("u:alice")
    assert hub.get("u:alice") is None


def test_hub_callbacks_route_to_correct_user(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import hub
    from brokers.realtime_state import state

    wa = hub.start("u:alice", _make_creds(tiger_id="A", account="UA"))
    wb = hub.start("u:bob",   _make_creds(tiger_id="B", account="UB"))

    wa._on_position_changed(NS(contract=NS(symbol="AAPL"), quantity=10,
                                 average_cost=150, market_value=1600,
                                 unrealized_pnl=100, market_price=160))
    wb._on_position_changed(NS(contract=NS(symbol="00700"), quantity=100,
                                 average_cost=380, market_value=40000,
                                 unrealized_pnl=2000, market_price=400))

    alice_snap = state.snapshot("u:alice")
    bob_snap = state.snapshot("u:bob")
    assert alice_snap["positions"][0]["symbol"] == "AAPL"
    assert bob_snap["positions"][0]["symbol"] == "00700"
    # Cross-contamination check
    assert all(p["symbol"] != "00700" for p in alice_snap["positions"])
    assert all(p["symbol"] != "AAPL"  for p in bob_snap["positions"])


def test_hub_start_rollback_on_failure(monkeypatch):
    """If wrapper.start() raises, the user_scope must NOT remain in the hub
    (otherwise retries would short-circuit on the dead entry)."""
    def bad_factory(*a, **kw):
        raise RuntimeError("connect failed")
    _patch_sdk(monkeypatch, _make_fake_sdk(bad_factory))

    from brokers.tiger_push import hub
    with pytest.raises(RuntimeError):
        hub.start("u:alice", _make_creds())
    assert hub.get("u:alice") is None


def test_hub_subscribe_quotes_forwards_to_client(monkeypatch):
    client_holder = {}
    def factory(*a, **kw):
        c = MagicMock()
        client_holder["c"] = c
        return c
    _patch_sdk(monkeypatch, _make_fake_sdk(factory))

    from brokers.tiger_push import hub
    hub.start("u:alice", _make_creds())
    hub.subscribe_quotes("u:alice", ["AAPL", "GOOG"])

    client_holder["c"].subscribe_quote.assert_called_once_with(["AAPL", "GOOG"])


def test_hub_subscribe_quotes_unknown_user_noop(monkeypatch):
    _patch_sdk(monkeypatch)
    from brokers.tiger_push import hub
    hub.subscribe_quotes("u:never_existed", ["AAPL"])  # should not raise
