"""
BrokerRegistry / get_current_broker 行为契约。

This covers X2 commit 1's deliverable: a per-(user, broker, label) factory
with caching. Credentials are still env-sourced (transitional); the same tests
should keep passing after X3 swaps in credentials_store.
"""

import os
import sys
from pathlib import Path

import pytest

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


# ─────────────────────────────────────────────────────────────────────────────
# Registry · 基础工厂行为
# ─────────────────────────────────────────────────────────────────────────────

def test_get_mock_returns_mock_adapter(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mock")
    from brokers.registry import _registry
    from brokers.mock_adapter import MockAdapter

    adapter = _registry.get(user_id="u_42", broker_type="mock")
    assert isinstance(adapter, MockAdapter)


def test_get_alpaca_returns_alpaca_adapter(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PKtest")
    monkeypatch.setenv("ALPACA_API_SECRET", "secrettest")
    from brokers.registry import _registry
    from brokers.alpaca_adapter import AlpacaAdapter

    adapter = _registry.get(user_id="u_42", broker_type="alpaca")
    assert isinstance(adapter, AlpacaAdapter)
    # Constructor must have stored the creds without hitting the network.
    assert adapter.api_key == "PKtest"
    assert adapter.api_secret == "secrettest"
    assert "paper-api.alpaca.markets" in adapter.base_url


def test_get_unknown_broker_raises(monkeypatch):
    from brokers.registry import _registry
    from brokers.base import BrokerError

    with pytest.raises(BrokerError, match="Unknown broker_type"):
        _registry.get(user_id="u_42", broker_type="not_a_broker")


def test_default_broker_type_from_env(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "PKtest")
    monkeypatch.setenv("ALPACA_API_SECRET", "secrettest")
    from brokers.registry import _registry
    from brokers.alpaca_adapter import AlpacaAdapter

    # broker_type omitted → use BROKER_MODE env
    adapter = _registry.get(user_id="u_42")
    assert isinstance(adapter, AlpacaAdapter)


# ─────────────────────────────────────────────────────────────────────────────
# Registry · 缓存语义
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
    import quant_agent  # touches thread-local
    from brokers.registry import get_current_broker, _registry

    # Simulate the request context that server.py would set up
    quant_agent._request_ctx.device_id = "u_thread_test"

    adapter = get_current_broker()
    # Same user_id should reuse cache
    assert get_current_broker() is adapter

    # Cleanup
    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass


def test_get_current_broker_falls_back_to_default(monkeypatch):
    """No thread-local context → "default" user_id (script / test usage)."""
    monkeypatch.setenv("BROKER_MODE", "mock")
    import quant_agent
    from brokers.registry import get_current_broker, _registry

    # Ensure no device_id set
    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass

    adapter = get_current_broker()
    # "default" should be cached under key ("default", "mock", "")
    cached = _registry._cache.get(("default", "mock", ""))
    assert cached is adapter


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility · brokers.get_broker shim
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_get_broker_still_works(monkeypatch):
    """X2 commit 1: shim must be silent and behavioral equivalent."""
    monkeypatch.setenv("BROKER_MODE", "mock")
    import quant_agent
    quant_agent._request_ctx.device_id = "u_legacy"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        from brokers import get_broker
        from brokers.mock_adapter import MockAdapter
        adapter = get_broker()
        assert isinstance(adapter, MockAdapter)

    try:
        del quant_agent._request_ctx.device_id
    except AttributeError:
        pass
