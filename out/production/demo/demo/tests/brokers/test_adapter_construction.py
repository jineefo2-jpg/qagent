"""
Adapter constructors must take typed Credentials and MUST NOT read env on their own
after X2. (Env reading is now the registry's transitional responsibility.)
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# MockAdapter
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_adapter_construct_with_credentials():
    from brokers.mock_adapter import MockAdapter
    from brokers.base import MockCredentials

    adapter = MockAdapter(MockCredentials(initial_cash=50000.0))
    assert adapter._initial_cash == 50000.0
    assert adapter.is_configured() is True


def test_mock_adapter_construct_without_credentials():
    """Back-compat: MockAdapter still accepts None for transitional callers."""
    from brokers.mock_adapter import MockAdapter, INITIAL_CASH

    adapter = MockAdapter()
    assert adapter._initial_cash == INITIAL_CASH


def test_mock_adapter_rejects_wrong_credentials_type():
    from brokers.mock_adapter import MockAdapter
    from brokers.base import AlpacaCredentials, BrokerError

    with pytest.raises(BrokerError, match="MockCredentials"):
        MockAdapter(AlpacaCredentials(api_key="x", api_secret="y"))


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaAdapter
# ─────────────────────────────────────────────────────────────────────────────

def test_alpaca_adapter_construct_with_credentials(monkeypatch):
    """Adapter MUST take creds from constructor, NOT from env."""
    # Set env to obviously-wrong values to prove constructor wins.
    monkeypatch.setenv("ALPACA_API_KEY", "env_value_should_be_ignored")
    monkeypatch.setenv("ALPACA_API_SECRET", "env_value_should_be_ignored")

    from brokers.alpaca_adapter import AlpacaAdapter
    from brokers.base import AlpacaCredentials

    adapter = AlpacaAdapter(AlpacaCredentials(
        api_key="ctor_key",
        api_secret="ctor_secret",
        base_url="https://paper-api.alpaca.markets",
    ))
    assert adapter.api_key == "ctor_key"
    assert adapter.api_secret == "ctor_secret"


def test_alpaca_adapter_rejects_wrong_credentials_type():
    from brokers.alpaca_adapter import AlpacaAdapter
    from brokers.base import MockCredentials, BrokerError

    with pytest.raises(BrokerError, match="AlpacaCredentials"):
        AlpacaAdapter(MockCredentials())


def test_alpaca_adapter_constructor_makes_no_network_call(monkeypatch):
    """Constructor must be cheap. Lazy client creation is in _ensure_client."""
    from brokers.alpaca_adapter import AlpacaAdapter
    from brokers.base import AlpacaCredentials

    # If the constructor tried to import or call alpaca, this would explode
    # because alpaca-py is optional. Constructor should not touch it.
    adapter = AlpacaAdapter(AlpacaCredentials(api_key="x", api_secret="y"))
    assert adapter._client is None
    assert adapter._sdk is None


# ─────────────────────────────────────────────────────────────────────────────
# Credentials dataclasses
# ─────────────────────────────────────────────────────────────────────────────

def test_credentials_are_immutable():
    """Frozen dataclass — credentials cannot be mutated at runtime."""
    from brokers.base import AlpacaCredentials
    creds = AlpacaCredentials(api_key="x", api_secret="y")
    with pytest.raises(Exception):  # FrozenInstanceError, but exact type varies
        creds.api_key = "rotated"  # type: ignore[misc]


def test_alpaca_credentials_default_to_paper():
    """CLAUDE.md mandates paper as default base_url."""
    from brokers.base import AlpacaCredentials
    creds = AlpacaCredentials(api_key="x", api_secret="y")
    assert "paper-api.alpaca.markets" in creds.base_url
