"""
BrokerRegistry / get_current_broker 行为契约.

X3 c2 baseline:
  - Mock is the only broker that gets credentials from env (it has none).
  - Every other broker (alpaca, tiger, ...) requires an explicit per-user
    binding in `credentials_store`. Calling `get(...)` without a binding
    raises `BrokerError("no_broker_binding: ...")`.
"""

import os
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clean_registry():
    """每个用例独立的 registry 状态,避免相互污染。"""
    from brokers.registry import _registry
    _registry.clear()
    yield
    _registry.clear()


@pytest.fixture
def store_env(monkeypatch, tmp_path):
    """
    A real SQLite-backed credential store wired into the production singletons,
    plus a fresh KEK. Used by tests that exercise the bind → get round-trip.
    """
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    # Redirect singleton path to a tmp file by patching the module constant.
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "creds.db")
    yield
    _db.close_default()


# ─────────────────────────────────────────────────────────────────────────────
# Registry · 基础工厂行为
# ─────────────────────────────────────────────────────────────────────────────

def test_get_mock_returns_mock_adapter(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry
    from brokers.mock_adapter import MockAdapter

    adapter = _registry.get(user_id="u_42", broker_type="mock")
    assert isinstance(adapter, MockAdapter)


def test_get_alpaca_unbound_raises_no_broker_binding(store_env):
    """X3 c2: env-based Alpaca fallback is gone. Unbound user → clear error."""
    from brokers.registry import _registry
    from brokers.base import BrokerError

    with pytest.raises(BrokerError, match="no_broker_binding"):
        _registry.get(user_id="u_unbound", broker_type="alpaca")


def test_get_alpaca_after_bind_returns_alpaca_adapter(store_env):
    """Full bind → get round-trip. Adapter is constructed from stored creds."""
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers.alpaca_adapter import AlpacaAdapter
    from brokers.registry import _registry

    store.bind(
        user_id="u_42", broker_type="alpaca", label="main",
        creds=AlpacaCredentials(api_key="PKstored", api_secret="ssstored"),
        actor="user",
    )

    adapter = _registry.get(user_id="u_42", broker_type="alpaca")
    assert isinstance(adapter, AlpacaAdapter)
    assert adapter.api_key == "PKstored"
    assert adapter.api_secret == "ssstored"


def test_get_alpaca_resolves_label_none_to_oldest_binding(store_env):
    """label=None → store.get_default_label → oldest binding wins."""
    import time
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers.registry import _registry

    store.bind("u_42", "alpaca", "first",
               AlpacaCredentials(api_key="k1", api_secret="s1"),
               actor="user")
    time.sleep(1.01)
    store.bind("u_42", "alpaca", "second",
               AlpacaCredentials(api_key="k2", api_secret="s2"),
               actor="user")

    adapter = _registry.get(user_id="u_42", broker_type="alpaca", label=None)
    assert adapter.api_key == "k1"


def test_get_tiger_unbound_raises_no_broker_binding(store_env):
    """X4: Tiger is per-user too — unbound must raise (not env-fallback)."""
    from brokers.registry import _registry
    from brokers.base import BrokerError

    with pytest.raises(BrokerError, match="no_broker_binding"):
        _registry.get(user_id="u_unbound", broker_type="tiger")


def test_get_tiger_after_bind_returns_tiger_adapter(store_env):
    from brokers.base import TigerCredentials
    from brokers.credentials_store import store
    from brokers.tiger_adapter import TigerAdapter
    from brokers.registry import _registry

    store.bind(
        user_id="u_42", broker_type="tiger", label="main",
        creds=TigerCredentials(
            tiger_id="20151024",
            private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
            account="U99999999",
        ),
        actor="user",
    )
    adapter = _registry.get(user_id="u_42", broker_type="tiger")
    assert isinstance(adapter, TigerAdapter)
    assert adapter.account == "U99999999"


def test_bind_invalidates_registry_cache(store_env):
    """After bind/unbind, the registry's stale adapter must not be returned."""
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers.registry import _registry

    store.bind("u_42", "alpaca", "main",
               AlpacaCredentials(api_key="old", api_secret="s"),
               actor="user")
    a1 = _registry.get(user_id="u_42", broker_type="alpaca", label="main")
    assert a1.api_key == "old"

    # Unbind + re-bind with new creds; registry must reflect new state.
    [summary] = store.list_user_bindings("u_42")
    store.unbind(summary.id, "u_42", actor="user")
    store.bind("u_42", "alpaca", "main",
               AlpacaCredentials(api_key="new", api_secret="s"),
               actor="user")

    a2 = _registry.get(user_id="u_42", broker_type="alpaca", label="main")
    assert a2.api_key == "new"
    assert a1 is not a2


def test_get_unknown_broker_raises():
    from brokers.registry import _registry
    from brokers.base import BrokerError

    with pytest.raises(BrokerError):
        _registry.get(user_id="u_42", broker_type="not_a_broker")


def test_default_broker_type_falls_back_to_mock(monkeypatch):
    """BROKER_MODE unset → default 'mock'."""
    monkeypatch.delenv("BROKER_MODE", raising=False)
    from brokers.registry import _registry
    from brokers.mock_adapter import MockAdapter

    adapter = _registry.get(user_id="u_42")
    assert isinstance(adapter, MockAdapter)


# ─────────────────────────────────────────────────────────────────────────────
# Registry · 缓存语义(mock-based,不依赖 store)
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_hit_returns_same_instance(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry

    a1 = _registry.get(user_id="u_42", broker_type="mock")
    a2 = _registry.get(user_id="u_42", broker_type="mock")
    assert a1 is a2  # identity, not just equality


def test_different_users_get_different_instances(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry

    a_alice = _registry.get(user_id="u_alice", broker_type="mock")
    a_bob = _registry.get(user_id="u_bob", broker_type="mock")
    assert a_alice is not a_bob


def test_different_labels_get_different_instances(monkeypatch):
    """1:N — same user, same broker_type, different labels = different bindings."""
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry

    main = _registry.get(user_id="u_42", broker_type="mock", label="main")
    secondary = _registry.get(user_id="u_42", broker_type="mock", label="secondary")
    assert main is not secondary


def test_invalidate_drops_cache_entry(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry

    a1 = _registry.get(user_id="u_42", broker_type="mock")
    _registry.invalidate(user_id="u_42", broker_type="mock")
    a2 = _registry.get(user_id="u_42", broker_type="mock")
    assert a1 is not a2  # new instance after invalidation


# ─────────────────────────────────────────────────────────────────────────────
# get_current_broker · thread-local 解析
# ─────────────────────────────────────────────────────────────────────────────

def test_get_current_broker_uses_thread_local(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    import quant_agent
    from brokers.registry import get_current_broker

    quant_agent._request_ctx.device_id = "u_thread_test"
    adapter = get_current_broker()
    assert get_current_broker() is adapter

    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass


def test_get_current_broker_falls_back_to_default(monkeypatch):
    """No thread-local context → "default" user_id (script / test usage)."""
    monkeypatch.setenv("BROKER_MODE", "mock")
    import quant_agent
    from brokers.registry import get_current_broker, _registry

    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass

    adapter = get_current_broker()
    cached = _registry._cache.get(("default", "mock", ""))
    assert cached is adapter


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility · brokers.get_broker shim
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_get_broker_emits_deprecation(monkeypatch):
    """X2 commit 2 onwards: deprecation warning is emitted; adapter still works."""
    monkeypatch.setenv("BROKER_MODE", "mock")
    import quant_agent
    quant_agent._request_ctx.device_id = "u_legacy"

    from brokers import get_broker
    from brokers.mock_adapter import MockAdapter

    with pytest.warns(DeprecationWarning, match="get_current_broker"):
        adapter = get_broker()
    assert isinstance(adapter, MockAdapter)

    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass
