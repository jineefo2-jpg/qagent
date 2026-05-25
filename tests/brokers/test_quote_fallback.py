"""
brokers/quote_fallback.py — yfinance fallback smoke tests.

Net access (Yahoo) is monkeypatched out; we only exercise the shape
normalization / merge logic that we wrote ourselves.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_yf_get_expiries_uses_yf_options(monkeypatch):
    import brokers.quote_fallback as qf

    monkeypatch.setattr(qf, "_YF_OK", True)

    class _FakeTicker:
        options = ("2026-06-19", "2026-07-17")

    class _FakeYf:
        @staticmethod
        def Ticker(_sym):
            return _FakeTicker()

    monkeypatch.setattr(qf, "_yf", _FakeYf)
    out = qf.yf_get_expiries("AAPL")
    assert [e["date"] for e in out] == ["2026-06-19", "2026-07-17"]


def test_yf_get_chain_merges_calls_and_puts(monkeypatch):
    import brokers.quote_fallback as qf
    import pandas as pd

    monkeypatch.setattr(qf, "_YF_OK", True)

    calls_df = pd.DataFrame([
        {"strike": 220.0, "lastPrice": 3.5, "bid": 3.4, "ask": 3.6,
         "volume": 100, "openInterest": 1000, "impliedVolatility": 0.25,
         "contractSymbol": "AAPL260619C00220000"},
        {"strike": 215.0, "lastPrice": 6.0, "bid": 5.9, "ask": 6.1,
         "volume": 50, "openInterest": 500, "impliedVolatility": 0.22,
         "contractSymbol": "AAPL260619C00215000"},
    ])
    puts_df = pd.DataFrame([
        {"strike": 220.0, "lastPrice": 2.1, "bid": 2.0, "ask": 2.2,
         "volume": 80, "openInterest": 400, "impliedVolatility": 0.28,
         "contractSymbol": "AAPL260619P00220000"},
        # 注意 215 没有 put — 测稀疏合并
    ])

    class _FakeTicker:
        def option_chain(self, expiry):
            return NS(calls=calls_df, puts=puts_df)

    class _FakeYf:
        @staticmethod
        def Ticker(_sym):
            return _FakeTicker()

    monkeypatch.setattr(qf, "_yf", _FakeYf)
    chain = qf.yf_get_chain("AAPL", "2026-06-19")
    strikes = [r["strike"] for r in chain]
    assert strikes == [215.0, 220.0]
    row220 = next(r for r in chain if r["strike"] == 220.0)
    assert row220["call"]["latest_price"] == 3.5
    assert row220["put"]["latest_price"] == 2.1
    assert row220["call"]["identifier"] == "AAPL260619C00220000"
    row215 = next(r for r in chain if r["strike"] == 215.0)
    assert row215["call"]["latest_price"] == 6.0
    assert row215["put"] is None


def test_yf_get_brief_computes_change(monkeypatch):
    import brokers.quote_fallback as qf

    monkeypatch.setattr(qf, "_YF_OK", True)

    class _FakeInfo:
        last_price = 213.5
        previous_close = 210.0
        open = 211.0
        day_high = 214.5
        day_low = 210.0
        last_volume = 1234567

    class _FakeTicker:
        fast_info = _FakeInfo()

    class _FakeYf:
        @staticmethod
        def Ticker(_sym):
            return _FakeTicker()

    monkeypatch.setattr(qf, "_yf", _FakeYf)
    b = qf.yf_get_brief("AAPL")
    assert b["symbol"] == "AAPL"
    assert b["available"] is True
    assert b["latest_price"] == 213.5
    assert b["change"] == pytest.approx(3.5)
    assert b["change_percent"] == pytest.approx(3.5 / 210.0 * 100)
    assert b["source"] == "yahoo_delay"
    assert b["delay_minutes"] == 15


def test_yf_returns_empty_when_yf_missing(monkeypatch):
    import brokers.quote_fallback as qf
    monkeypatch.setattr(qf, "_YF_OK", False)
    assert qf.yf_get_expiries("AAPL") == []
    assert qf.yf_get_chain("AAPL", "2026-06-19") == []
    b = qf.yf_get_brief("AAPL")
    assert b["available"] is False
    assert b["source"] == "yahoo_delay"


def test_yf_brief_swallows_exceptions(monkeypatch):
    import brokers.quote_fallback as qf

    monkeypatch.setattr(qf, "_YF_OK", True)

    class _FakeYf:
        @staticmethod
        def Ticker(_sym):
            raise RuntimeError("yahoo down")

    monkeypatch.setattr(qf, "_yf", _FakeYf)
    b = qf.yf_get_brief("AAPL")
    assert b["available"] is False
    assert b["source"] == "yahoo_delay"
