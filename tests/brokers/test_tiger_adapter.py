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

def _fake_sdk(client):
    """Build a fake _import_tiger() return value bound to the given client mock."""
    fake_config_cls = MagicMock()
    fake_trade_cls = MagicMock(return_value=client)
    return {
        "TigerOpenClientConfig": fake_config_cls,
        "TradeClient": fake_trade_cls,
        "Language": NS(zh_CN="zh_CN"),
        "Order": MagicMock(),
    }


def _build_adapter(monkeypatch, client):
    """Construct a TigerAdapter wired to a mocked SDK."""
    from brokers.base import TigerCredentials
    from brokers import tiger_adapter
    monkeypatch.setattr(tiger_adapter, "_import_tiger", lambda: _fake_sdk(client))
    return tiger_adapter.TigerAdapter(TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        account="U99999999",
    ))


def test_get_account_with_mocked_sdk(monkeypatch):
    """Mock the lazy import + TradeClient so we never need a real Tiger account."""
    fake_client = MagicMock()
    fake_summary = NS(
        cash=12345.67, buying_power=15000.0,
        gross_position_value=5000.0, currency="USD", status="ACTIVE",
    )
    fake_account = NS(summary=fake_summary)
    fake_client.get_assets.return_value = [fake_account]

    adapter = _build_adapter(monkeypatch, fake_client)
    acct = adapter.get_account()

    assert acct.cash == 12345.67
    assert acct.buying_power == 15000.0
    assert acct.equity == 12345.67 + 5000.0
    assert acct.currency == "USD"
    assert acct.account_id == "U99999999"
    fake_client.get_assets.assert_called_once_with(account="U99999999")


def test_place_order_uses_returned_int_as_broker_order_id(monkeypatch):
    """
    Tiger's place_order returns Optional[int] (the new broker order id),
    NOT an Order object. Regression test for the bug found while the
    user installed the real SDK.
    """
    from brokers.base import OrderIntent, OrderSide, OrderType
    fake_client = MagicMock()
    fake_client.get_contract.return_value = NS(symbol="AAPL")
    # create_order returns a local Order object (no id yet, no fill)
    fake_order = NS(
        filled=0, status="New", avg_fill_price=0, order_time="2026-05-25",
    )
    fake_client.create_order.return_value = fake_order
    # place_order returns the broker_order_id as an int
    fake_client.place_order.return_value = 7788

    adapter = _build_adapter(monkeypatch, fake_client)
    intent = OrderIntent.new(
        symbol="AAPL", side="buy", qty=10, order_type="limit", limit_price=150.0,
    )
    result = adapter.place_order(intent)

    assert result.broker_order_id == "7788"
    assert result.symbol == "AAPL"
    assert result.qty == 10.0
    assert result.filled_qty == 0
    assert result.limit_price == 150.0


def test_place_order_none_return_raises_network(monkeypatch):
    """If Tiger returns None (network failure mid-submit) → BrokerNetworkError."""
    from brokers.base import OrderIntent, BrokerNetworkError
    fake_client = MagicMock()
    fake_client.get_contract.return_value = NS(symbol="AAPL")
    fake_client.create_order.return_value = NS()
    fake_client.place_order.return_value = None

    adapter = _build_adapter(monkeypatch, fake_client)
    intent = OrderIntent.new(
        symbol="AAPL", side="buy", qty=10, order_type="limit", limit_price=150.0,
    )
    with pytest.raises(BrokerNetworkError, match="返回 None"):
        adapter.place_order(intent)


def test_place_order_market_is_rejected(monkeypatch):
    """Market orders are blocked at the adapter level per CLAUDE.md trading safety."""
    from brokers.base import OrderIntent, BrokerRejectedError
    adapter = _build_adapter(monkeypatch, MagicMock())
    intent = OrderIntent.new(symbol="AAPL", side="buy", qty=10, order_type="market")
    with pytest.raises(BrokerRejectedError, match="仅支持限价单"):
        adapter.place_order(intent)


def test_list_orders_reads_filled_field_not_filled_quantity(monkeypatch):
    """Regression: Tiger's Order field is `filled`, not `filled_quantity`."""
    fake_client = MagicMock()
    fake_order = NS(
        id=12345, contract=NS(symbol="AAPL"),
        action="BUY", order_type="LMT", quantity=100,
        filled=37,                     # tigeropen's actual field name
        filled_quantity=999,           # what my old code wrongly read — must NOT be used
        limit_price=150.0, status="PartiallyFilled",
        avg_fill_price=149.5, order_time="2026-05-25",
    )
    fake_client.get_orders.return_value = [fake_order]

    adapter = _build_adapter(monkeypatch, fake_client)
    [result] = adapter.list_orders(limit=10)

    assert result.broker_order_id == "12345"
    assert result.filled_qty == 37.0   # confirms we read `filled`
    assert result.qty == 100.0
