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
        "SecurityType": NS(STK="STK", OPT="OPT", FUT="FUT", WAR="WAR"),
        "Market": NS(US="US", HK="HK", CN="CN", ALL="ALL"),
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


def test_wrap_error_preserves_message():
    """X4 follow-up: _wrap_error no longer swallows the SDK message."""
    from brokers.base import TigerCredentials, BrokerAuthError
    from brokers.tiger_adapter import TigerAdapter

    a = TigerAdapter(TigerCredentials(
        tiger_id="x", private_key="-----BEGIN PRIVATE KEY-----\nbody\n-----END PRIVATE KEY-----",
        account="U1",
    ))
    err = a._wrap_error(RuntimeError(
        "[uuid-here]request sign failed. Unable to load PEM file. MismatchedTags(...)"
    ))
    s = str(err)
    assert isinstance(err, BrokerAuthError)         # "sign" → auth bucket
    assert "RuntimeError" in s                       # exception class is named
    assert "MismatchedTags" in s                     # original detail preserved
    assert "request sign failed" in s


def test_wrap_error_classifies_buckets():
    from brokers.base import (
        TigerCredentials, BrokerAuthError, BrokerNetworkError, BrokerRejectedError,
        BrokerError,
    )
    from brokers.tiger_adapter import TigerAdapter

    a = TigerAdapter(TigerCredentials(
        tiger_id="x", private_key="-----BEGIN PRIVATE KEY-----\nb\n-----END PRIVATE KEY-----",
        account="U1",
    ))
    assert isinstance(a._wrap_error(RuntimeError("sign verification failed")), BrokerAuthError)
    assert isinstance(a._wrap_error(RuntimeError("connection timeout")), BrokerNetworkError)
    assert isinstance(a._wrap_error(RuntimeError("insufficient buying power")), BrokerRejectedError)
    # No keyword match → generic BrokerError
    err = a._wrap_error(RuntimeError("something weird"))
    assert isinstance(err, BrokerError)
    # but NOT one of the subclasses
    assert type(err) is BrokerError


def test_wrap_error_redacts_pem_in_message():
    """Defence in depth: even if the SDK echoes a key, it gets redacted."""
    from brokers.base import TigerCredentials
    from brokers.tiger_adapter import TigerAdapter

    a = TigerAdapter(TigerCredentials(
        tiger_id="x", private_key="-----BEGIN PRIVATE KEY-----\nb\n-----END PRIVATE KEY-----",
        account="U1",
    ))
    leak = (
        "sign with key -----BEGIN RSA PRIVATE KEY-----\n"
        "SUPER_SECRET_KEY_BYTES_THAT_SHOULDNT_LEAK\n"
        "-----END RSA PRIVATE KEY-----"
    )
    err = a._wrap_error(RuntimeError(leak))
    s = str(err)
    assert "SUPER_SECRET_KEY_BYTES_THAT_SHOULDNT_LEAK" not in s
    assert "[REDACTED-PEM]" in s


def test_map_status_handles_real_tiger_formats():
    """
    Tiger SDK 实际返回 `OrderStatus.FILLED` (大写枚举值)。
    早期映射表用 CamelCase ("Filled") 导致所有状态都 fallback 到 NEW —
    用户报告"已成交订单还显示待成交"。这一组用例钉死真实格式都映射对。
    """
    from brokers.tiger_adapter import _map_status
    from brokers.base import OrderStatus

    cases = [
        # (Tiger 返回, 我们的映射)
        ("OrderStatus.FILLED",            OrderStatus.FILLED),
        ("FILLED",                         OrderStatus.FILLED),
        ("Filled",                         OrderStatus.FILLED),    # CamelCase 容错
        ("OrderStatus.PARTIALLY_FILLED",  OrderStatus.PARTIALLY_FILLED),
        ("PARTIALLY_FILLED",               OrderStatus.PARTIALLY_FILLED),
        ("PartiallyFilled",                OrderStatus.PARTIALLY_FILLED),
        ("OrderStatus.NEW",                OrderStatus.NEW),
        ("NEW",                            OrderStatus.NEW),
        ("OrderStatus.HELD",               OrderStatus.NEW),
        ("OrderStatus.CANCELLED",          OrderStatus.CANCELED),
        ("CANCELLED",                      OrderStatus.CANCELED),
        ("CANCELED",                       OrderStatus.CANCELED),   # 美式拼法也兼容
        ("OrderStatus.REJECTED",           OrderStatus.REJECTED),
        ("OrderStatus.EXPIRED",            OrderStatus.EXPIRED),
        ("OrderStatus.INACTIVE",           OrderStatus.CANCELED),
        ("OrderStatus.INITIAL",            OrderStatus.NEW),
        ("OrderStatus.PENDING_CANCEL",     OrderStatus.NEW),
    ]
    for raw, expected in cases:
        actual = _map_status(raw)
        assert actual == expected, f"{raw!r} → {actual} (expected {expected})"

    # 未知值兜底 NEW(不抛)
    assert _map_status("OrderStatus.SOMETHING_NEW_TIGER_INVENTED") == OrderStatus.NEW
    assert _map_status(None) == OrderStatus.NEW


def test_strip_pem_markers_returns_base64_only():
    """Pure-function test for the PEM → raw base64 helper."""
    from brokers.tiger_adapter import _strip_pem_markers

    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICXAIBAAKBgQDIyVcAAQUq7Q\n"
        "TWxqMDe44XOlS/yiTbcN/eZqAA==\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    body = _strip_pem_markers(pem)
    assert "BEGIN" not in body
    assert "END" not in body
    assert "\n" not in body
    assert body == "MIICXAIBAAKBgQDIyVcAAQUq7QTWxqMDe44XOlS/yiTbcN/eZqAA=="


def test_strip_pem_markers_handles_pkcs8_label():
    """Same helper must also handle 'BEGIN PRIVATE KEY' (no RSA, PKCS#8 label)."""
    from brokers.tiger_adapter import _strip_pem_markers

    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "ABCDEFG\n"
        "-----END PRIVATE KEY-----"
    )
    assert _strip_pem_markers(pem) == "ABCDEFG"


def test_ensure_client_assigns_raw_base64_not_full_pem(monkeypatch):
    """
    Regression: tigeropen.common.util.signature_utils.load_private_key calls
    base64.b64decode(private_key) directly. We MUST strip PEM markers before
    assigning to config.private_key, or signing fails with 'Incorrect padding'.
    """
    from brokers.base import TigerCredentials
    from brokers import tiger_adapter

    fake_config = MagicMock()
    fake_config_cls = MagicMock(return_value=fake_config)
    fake_client = MagicMock()
    fake_trade_cls = MagicMock(return_value=fake_client)

    monkeypatch.setattr(tiger_adapter, "_import_tiger", lambda: {
        "TigerOpenClientConfig": fake_config_cls,
        "TradeClient": fake_trade_cls,
        "Language": NS(zh_CN="zh_CN"),
        "SecurityType": NS(STK="STK", OPT="OPT", FUT="FUT", WAR="WAR"),
        "Market": NS(US="US", HK="HK", CN="CN", ALL="ALL"),
        "Order": MagicMock(),
    })

    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICXAIBAAKBgQDIyVcAAQUq7Q\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    adapter = tiger_adapter.TigerAdapter(TigerCredentials(
        tiger_id="20151024", private_key=pem, account="U99999999",
    ))
    adapter._ensure_client()

    # What we set on the fake config — the value tigeropen would consume.
    assigned = fake_config.private_key
    assert "BEGIN" not in assigned
    assert "END" not in assigned
    assert "\n" not in assigned
    assert assigned == "MIICXAIBAAKBgQDIyVcAAQUq7Q"


def test_list_positions_merges_across_sec_type_and_market(monkeypatch):
    """
    Tiger 的 get_positions 默认只返回一种 sec_type;我们要把 US 股 + HK 股 + 美股期权
    全部合并到一份持仓列表里返回(用户的真实需求)。
    """
    fake_client = MagicMock()
    call_log = []

    def get_positions(account, sec_type, market):
        call_log.append((str(sec_type), str(market)))
        # 模拟 Tiger 不同 (sec_type, market) 组合返回不同的持仓
        if sec_type == "STK" and market == "US":
            return [NS(contract=NS(symbol="AAPL", sec_type="STK"),
                       quantity=10, average_cost=150.0,
                       market_value=1600.0, unrealized_pnl=100.0, market_price=160.0)]
        if sec_type == "STK" and market == "HK":
            return [NS(contract=NS(symbol="00700", sec_type="STK"),
                       quantity=100, average_cost=380.0,
                       market_value=40000.0, unrealized_pnl=2000.0, market_price=400.0)]
        if sec_type == "OPT" and market == "US":
            return [NS(contract=NS(symbol="AAPL", sec_type="OPT",
                                    expiry="2025-06-20", strike=200.0, put_call="CALL"),
                       quantity=1, average_cost=5.0,
                       market_value=600.0, unrealized_pnl=100.0, market_price=6.0)]
        return []
    fake_client.get_positions = get_positions

    adapter = _build_adapter(monkeypatch, fake_client)
    positions = adapter.list_positions()

    symbols = [p.symbol for p in positions]
    assert "AAPL" in symbols, f"expected US stock AAPL, got {symbols}"
    assert "00700" in symbols, f"expected HK stock 00700, got {symbols}"
    # 期权展开了 expiry + strike + Call/Put
    assert any("AAPL 2025-06-20 C200" in s for s in symbols), f"expected option, got {symbols}"
    assert len(positions) == 3

    # 至少跑过 12 次组合(4 sec_type × 3 market)
    assert len(call_log) >= 12


def test_list_positions_deduplicates_when_combos_overlap(monkeypatch):
    """如果不同 (sec_type, market) 查询返回了相同 contract,只保留一份。"""
    fake_client = MagicMock()
    same_pos = NS(contract=NS(symbol="AAPL", sec_type="STK"),
                   quantity=10, average_cost=150.0,
                   market_value=1600.0, unrealized_pnl=100.0, market_price=160.0)
    fake_client.get_positions = MagicMock(return_value=[same_pos])

    adapter = _build_adapter(monkeypatch, fake_client)
    positions = adapter.list_positions()

    # 12 个组合每个都返回同一条 → 去重后只有 1
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


def test_list_positions_silently_skips_failing_combos(monkeypatch):
    """权限不足等错误应当被吞掉,只要至少一个组合成功就返回。"""
    fake_client = MagicMock()

    def get_positions(account, sec_type, market):
        if sec_type == "STK" and market == "US":
            return [NS(contract=NS(symbol="AAPL", sec_type="STK"),
                       quantity=1, average_cost=100.0,
                       market_value=110.0, unrealized_pnl=10.0, market_price=110.0)]
        raise RuntimeError(f"permission denied for {sec_type}/{market}")
    fake_client.get_positions = get_positions

    adapter = _build_adapter(monkeypatch, fake_client)
    positions = adapter.list_positions()

    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


def test_list_positions_raises_if_all_combos_fail(monkeypatch):
    """全军覆没时把最后一个错误抛出来,不要静默返回空数组。"""
    from brokers.base import BrokerError
    fake_client = MagicMock()
    fake_client.get_positions = MagicMock(
        side_effect=RuntimeError("network down")
    )

    adapter = _build_adapter(monkeypatch, fake_client)
    with pytest.raises(BrokerError, match="network down"):
        adapter.list_positions()


def test_list_orders_merges_across_combos(monkeypatch):
    """订单也跨 sec_type/market 合并,按 id 去重。"""
    fake_client = MagicMock()

    def get_orders(account, sec_type, market, limit):
        if sec_type == "STK" and market == "US":
            return [NS(id=111, contract=NS(symbol="AAPL"), action="BUY",
                       order_type="LMT", quantity=10, filled=0,
                       limit_price=150.0, status="New", avg_fill_price=0,
                       order_time="2026-05-25 09:00")]
        if sec_type == "OPT" and market == "US":
            return [NS(id=222, contract=NS(symbol="AAPL 2025-06-20 C200"),
                       action="BUY", order_type="LMT", quantity=1, filled=0,
                       limit_price=5.0, status="New", avg_fill_price=0,
                       order_time="2026-05-25 09:30")]
        return []
    fake_client.get_orders = get_orders

    adapter = _build_adapter(monkeypatch, fake_client)
    orders = adapter.list_orders(limit=50)

    ids = [o.broker_order_id for o in orders]
    assert "111" in ids
    assert "222" in ids
    assert len(orders) == 2


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
