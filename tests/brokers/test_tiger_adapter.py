"""
brokers/tiger_adapter.py — construction + lazy SDK loading.

These tests never touch the network and don't require `tigeropen` to be
installed. Adapter construction must be cheap; the SDK is imported only
on first `_ensure_client()` call, which we exercise via monkeypatching.
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


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def test_tiger_adapter_construct_with_credentials():
    from brokers.base import TigerCredentials
    from brokers.tiger_adapter import TigerAdapter

    creds = TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----",
        account="U99999999",
        license="TBNZ",
    )
    adapter = TigerAdapter(creds)
    assert adapter.tiger_id == "20151024"
    assert adapter.account == "U99999999"
    assert adapter.license == "TBNZ"
    assert adapter.is_configured() is True


def test_tiger_adapter_rejects_wrong_credentials_type():
    from brokers.alpaca_adapter import AlpacaAdapter  # noqa
    from brokers.base import AlpacaCredentials, BrokerError
    from brokers.tiger_adapter import TigerAdapter

    with pytest.raises(BrokerError, match="TigerCredentials"):
        TigerAdapter(AlpacaCredentials(api_key="x", api_secret="y"))


def test_tiger_adapter_is_not_configured_when_fields_missing():
    from brokers.base import TigerCredentials
    from brokers.tiger_adapter import TigerAdapter

    # Missing private_key
    adapter = TigerAdapter(TigerCredentials(
        tiger_id="20151024", private_key="", account="U99999999",
    ))
    assert adapter.is_configured() is False

    # Missing account
    adapter = TigerAdapter(TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        account="",
    ))
    assert adapter.is_configured() is False


def test_tiger_adapter_constructor_makes_no_network_call():
    """Even without tigeropen installed, constructor must succeed (lazy load)."""
    from brokers.base import TigerCredentials
    from brokers.tiger_adapter import TigerAdapter

    adapter = TigerAdapter(TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        account="U99999999",
    ))
    assert adapter._client is None
    assert adapter._sdk is None


def test_tiger_adapter_repr_never_leaks_private_key():
    """CLAUDE.md mandate: credentials MUST NEVER appear in any log/exception/repr."""
    from brokers.base import TigerCredentials
    from brokers.tiger_adapter import TigerAdapter

    secret_key = "-----BEGIN PRIVATE KEY-----\nSUPER-SECRET-VALUE-12345\n-----END-----"
    adapter = TigerAdapter(TigerCredentials(
        tiger_id="20151024", private_key=secret_key, account="U99999999",
    ))
    rep = repr(adapter)
    assert "SUPER-SECRET-VALUE-12345" not in rep
    assert "PRIVATE KEY" not in rep
    # But identifying metadata is OK
    assert "20151024" in rep
    assert "U99999999" in rep


# ─────────────────────────────────────────────────────────────────────────────
# Lazy SDK loading
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_client_raises_friendly_when_sdk_missing(monkeypatch):
    """If tigeropen isn't installed, the error must point to the install command."""
    from brokers.base import TigerCredentials, BrokerError
    from brokers import tiger_adapter

    # Simulate ImportError by replacing _import_tiger
    def fake_import():
        raise ImportError("simulated missing tigeropen")
    # We need _import_tiger() to raise BrokerError with install hint
    monkeypatch.setattr(tiger_adapter, "_import_tiger", lambda: (_ for _ in ()).throw(
        BrokerError("缺少 tigeropen 依赖,请运行: pip install tigeropen>=3.2.0,<4.0.0")
    ))

    adapter = tiger_adapter.TigerAdapter(TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        account="U99999999",
    ))
    with pytest.raises(BrokerError, match="pip install tigeropen"):
        adapter._ensure_client()


def test_ensure_client_unconfigured_raises_auth_error():
    from brokers.base import TigerCredentials, BrokerAuthError
    from brokers.tiger_adapter import TigerAdapter

    adapter = TigerAdapter(TigerCredentials())  # all fields blank
    with pytest.raises(BrokerAuthError, match="未配置"):
        adapter._ensure_client()


# ─────────────────────────────────────────────────────────────────────────────
# Mocked happy path — make sure get_account() reads SDK fields correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_get_account_with_mocked_sdk(monkeypatch):
    """Mock the lazy import + TradeClient so we never need a real Tiger account."""
    from brokers.base import TigerCredentials
    from brokers import tiger_adapter
    from brokers.tiger_adapter import TigerAdapter

    fake_client = MagicMock()
    fake_summary = NS(
        cash=12345.67, buying_power=15000.0,
        gross_position_value=5000.0, currency="USD", status="ACTIVE",
    )
    fake_account = NS(summary=fake_summary)
    fake_client.get_assets.return_value = [fake_account]

    fake_config_cls = MagicMock()  # returns a config object
    fake_trade_cls = MagicMock(return_value=fake_client)
    fake_lang = NS(zh_CN="zh_CN")

    monkeypatch.setattr(tiger_adapter, "_import_tiger", lambda: {
        "TigerOpenClientConfig": fake_config_cls,
        "TradeClient": fake_trade_cls,
        "Language": fake_lang,
        "Order": MagicMock(),
    })

    adapter = TigerAdapter(TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        account="U99999999",
    ))
    acct = adapter.get_account()

    assert acct.cash == 12345.67
    assert acct.buying_power == 15000.0
    assert acct.equity == 12345.67 + 5000.0
    assert acct.currency == "USD"
    assert acct.account_id == "U99999999"
    fake_client.get_assets.assert_called_once_with(account="U99999999")
