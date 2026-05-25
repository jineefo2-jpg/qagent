"""
brokers/tiger_quote.py — search + chain shaping smoke tests.

These tests don't hit the network. The tigeropen SDK is monkeypatched
out so we can exercise the fuzzy-match / shape-normalization logic
without real credentials.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def creds():
    from brokers.base import TigerCredentials
    return TigerCredentials(
        tiger_id="20151024",
        private_key="-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----",
        account="U99999999",
        license="TBNZ",
    )


@pytest.fixture
def fake_quote_client(monkeypatch, creds):
    """Build a TigerQuoteClient whose underlying SDK call is mocked."""
    from brokers.tiger_quote import TigerQuoteClient

    qc = TigerQuoteClient(creds)

    fake_sdk_client = NS()
    qc._client = fake_sdk_client  # bypass _ensure_client
    qc._sdk = {
        "Market": NS(US="US_ENUM", HK="HK_ENUM", CN="CN_ENUM", SG="SG_ENUM"),
    }
    return qc, fake_sdk_client


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def test_construct_with_credentials(creds):
    from brokers.tiger_quote import TigerQuoteClient

    qc = TigerQuoteClient(creds)
    assert qc.tiger_id == "20151024"
    assert qc.account == "U99999999"


def test_construct_rejects_wrong_credentials_type():
    from brokers.base import AlpacaCredentials, BrokerError
    from brokers.tiger_quote import TigerQuoteClient

    with pytest.raises(BrokerError, match="TigerCredentials"):
        TigerQuoteClient(AlpacaCredentials(api_key="x", api_secret="y"))


def test_repr_does_not_leak_private_key(creds):
    from brokers.tiger_quote import TigerQuoteClient

    qc = TigerQuoteClient(creds)
    r = repr(qc)
    assert "FAKE" not in r
    assert "BEGIN" not in r
    assert "20151024" in r  # tiger_id is OK to show


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy symbol search — the main logic we wrote
# ─────────────────────────────────────────────────────────────────────────────

def test_search_symbols_empty_query_returns_empty(fake_quote_client):
    qc, _ = fake_quote_client
    assert qc.search_symbols("") == []
    assert qc.search_symbols("   ") == []


def test_search_symbols_prefix_beats_contains(fake_quote_client):
    qc, client = fake_quote_client
    # Return three rows. A "APP" query should prefer AAPL (prefix on A)?
    # We're searching for "APP" so prefix-on-symbol = AAPL no, AAPL doesn't start with APP.
    # Use query "AA" — AAPL starts with AA, BAAPL contains AA. AAPL should come first.
    client.get_symbol_names = lambda market: [
        ("BAAPL", "Beta Apple"),    # contains AA
        ("AAPL",  "Apple Inc."),     # starts with AA
        ("ZZZ",   "Zinc"),           # no match
    ]
    results = qc.search_symbols("AA", market="US")
    assert [r["symbol"] for r in results] == ["AAPL", "BAAPL"]


def test_search_symbols_falls_back_to_name(fake_quote_client):
    qc, client = fake_quote_client
    client.get_symbol_names = lambda market: [
        ("NVDA", "NVIDIA Corp"),
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft"),
    ]
    # "Apple" doesn't match any ticker but matches AAPL's name
    results = qc.search_symbols("Apple", market="US")
    assert len(results) >= 1
    assert results[0]["symbol"] == "AAPL"


def test_search_symbols_limit_respected(fake_quote_client):
    qc, client = fake_quote_client
    client.get_symbol_names = lambda market: [
        (f"AAA{i}", f"Comp {i}") for i in range(50)
    ]
    results = qc.search_symbols("AAA", market="US", limit=5)
    assert len(results) == 5


def test_search_symbols_handles_dict_rows(fake_quote_client):
    """SDK 偶尔会返回 dict-shaped 行,也能解析。"""
    qc, client = fake_quote_client
    client.get_symbol_names = lambda market: [
        {"symbol": "TSLA", "name": "Tesla, Inc."},
        ("AAPL", "Apple Inc."),
    ]
    results = qc.search_symbols("TSLA", market="US")
    assert any(r["symbol"] == "TSLA" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Brief / option chain shape normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_get_brief_normalizes_fields(fake_quote_client):
    qc, client = fake_quote_client
    client.get_briefs = lambda symbols, include_hour_trading: [
        NS(
            latest_price="213.45",
            prev_close="210.00",
            open="211.0",
            high="214.5",
            low="210.5",
            volume="1234567",
            change="3.45",
            change_percent="1.64",
            latest_time="1716200000000",
        )
    ]
    brief = qc.get_brief("AAPL")
    assert brief["symbol"] == "AAPL"
    assert brief["available"] is True
    assert brief["latest_price"] == 213.45
    assert brief["change_percent"] == 1.64
    assert brief["volume"] == 1234567


def test_get_option_chain_merges_call_put_by_strike(fake_quote_client, monkeypatch):
    qc, client = fake_quote_client
    import pandas as pd

    df = pd.DataFrame([
        {"strike": 220.0, "put_call": "CALL", "identifier": "AAPL  240620C220",
         "latest_price": 3.5, "bid_price": 3.4, "ask_price": 3.6,
         "volume": 1000, "open_interest": 5000, "implied_vol": 0.25},
        {"strike": 220.0, "put_call": "PUT", "identifier": "AAPL  240620P220",
         "latest_price": 2.1, "bid_price": 2.0, "ask_price": 2.2,
         "volume": 800, "open_interest": 4000, "implied_vol": 0.28},
        {"strike": 215.0, "put_call": "CALL", "identifier": "AAPL  240620C215",
         "latest_price": 6.0, "bid_price": 5.9, "ask_price": 6.1,
         "volume": 500, "open_interest": 2000, "implied_vol": 0.22},
    ])
    client.get_option_chain = lambda symbol, expiry: df

    chain = qc.get_option_chain("AAPL", "2024-06-20")
    # Two strikes, sorted ascending
    assert [r["strike"] for r in chain] == [215.0, 220.0]
    row220 = next(r for r in chain if r["strike"] == 220.0)
    assert row220["call"]["latest_price"] == 3.5
    assert row220["put"]["latest_price"] == 2.1
    row215 = next(r for r in chain if r["strike"] == 215.0)
    assert row215["call"]["latest_price"] == 6.0
    assert row215["put"] is None  # only call provided


def test_safe_float_handles_nan_and_strings():
    from brokers.tiger_quote import _safe_float, _safe_int
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("3.14") == 3.14
    assert _safe_float(float("nan")) is None
    assert _safe_int("12") == 12
    assert _safe_int(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Delay-tier fallback inside get_brief (X8 b)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_brief_falls_back_to_delay_on_permission_error(fake_quote_client):
    """
    实时 get_briefs 因为权限不足抛 BrokerAuthError → 自动降级
    get_stock_delay_briefs。返回里 source='tiger_delay'。
    """
    import pandas as pd
    qc, client = fake_quote_client

    def _raise_perm(symbols, include_hour_trading):
        raise RuntimeError("Tiger ApiException: code=4 msg=4000:permission denied")
    client.get_briefs = _raise_perm

    # Delay endpoint returns a DataFrame (col: pre_close/open/high/low/close/volume)
    client.get_stock_delay_briefs = lambda symbols: pd.DataFrame([{
        "symbol": "AAPL", "pre_close": 210.0, "open": 211.0, "high": 214.5,
        "low": 210.0, "close": 213.5, "volume": 1234567, "time": 1716200000000,
    }])

    out = qc.get_brief("AAPL")
    assert out["available"] is True
    assert out["source"] == "tiger_delay"
    assert out["delay_minutes"] == 15
    assert out["latest_price"] == 213.5
    assert out["change"] == pytest.approx(3.5)
    assert out["change_percent"] == pytest.approx(3.5 / 210.0 * 100)


def test_get_brief_realtime_path_marks_source_realtime(fake_quote_client):
    qc, client = fake_quote_client
    client.get_briefs = lambda symbols, include_hour_trading: [
        NS(latest_price="213.5", prev_close="210.0", change="3.5",
           change_percent="1.67", volume="100", open="211", high="214", low="210",
           latest_time="1716200000000"),
    ]
    out = qc.get_brief("AAPL")
    assert out["source"] == "tiger_realtime"
    assert out["latest_price"] == 213.5


def test_get_brief_non_permission_error_still_raises(fake_quote_client):
    """网络错误 / 其它非权限错误不应该误降级。"""
    from brokers.base import BrokerError
    qc, client = fake_quote_client

    def _raise_net(symbols, include_hour_trading):
        raise RuntimeError("Network unreachable")
    client.get_briefs = _raise_net
    # 延迟接口正常,但我们应该抛而不是降级
    client.get_stock_delay_briefs = lambda symbols: pytest.fail("不应该走到这里")

    # 必须缓存里没值,否则上一个测试的结果会被复用
    qc._brief_cache.pop("AAPL", None)
    with pytest.raises(BrokerError, match="Network unreachable"):
        qc.get_brief("AAPL")
