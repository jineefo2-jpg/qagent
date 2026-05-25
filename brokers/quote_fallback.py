"""
brokers/quote_fallback.py — 免费延迟行情兜底 (X8 fallback)

当用户没有付费 Tiger 实时行情包时,用 Yahoo Finance (yfinance) 提供
延迟约 15min 的股票/期权数据。返回形状跟 TigerQuoteClient 完全一致,
server endpoints 可以无脑替换。

yfinance 是项目里已有的依赖 (requirements.txt:yfinance>=0.2.0),
quant_agent.py 也已在用 _quote_yfinance 作美股报价。

数据时效:
  - 股票快照:Yahoo 15min 延迟 (普通免费用户)
  - 期权链:Yahoo 同样延迟,字段包含 last/bid/ask/volume/OI/IV
  - 期权到期日:不需要付费,任何用户都能拿
"""
from __future__ import annotations

import logging
from typing import List, Optional


log = logging.getLogger(__name__)

try:
    import yfinance as _yf  # type: ignore
    _YF_OK = True
except Exception:
    _yf = None
    _YF_OK = False


def yf_available() -> bool:
    return _YF_OK


# ════════════════════════════════════════════════════════════
# 股票延迟报价
# ════════════════════════════════════════════════════════════

def yf_get_brief(symbol: str) -> dict:
    """
    返回跟 TigerQuoteClient.get_brief 一样的形状,多一个 source 字段。
    yfinance 失败时返回 available=False。
    """
    if not _YF_OK or not symbol:
        return {"symbol": symbol, "available": False, "source": "yahoo_delay"}
    sym = symbol.strip().upper()
    try:
        t = _yf.Ticker(sym)
        # fast_info 不发额外请求,比 .info 快
        info = t.fast_info
        last = _safe_float(getattr(info, "last_price", None))
        prev = _safe_float(getattr(info, "previous_close", None))
        change = (last - prev) if (last is not None and prev is not None) else None
        change_pct = (change / prev * 100) if (change is not None and prev) else None
        return {
            "symbol": sym,
            "available": last is not None,
            "latest_price": last,
            "prev_close": prev,
            "open": _safe_float(getattr(info, "open", None)),
            "high": _safe_float(getattr(info, "day_high", None)),
            "low": _safe_float(getattr(info, "day_low", None)),
            "volume": _safe_int(getattr(info, "last_volume", None)),
            "change": change,
            "change_percent": change_pct,
            "latest_time": 0,
            "source": "yahoo_delay",
            "delay_minutes": 15,
        }
    except Exception as e:
        log.warning("yf_get_brief(%s) failed: %s", sym, e)
        return {"symbol": sym, "available": False, "source": "yahoo_delay"}


# ════════════════════════════════════════════════════════════
# 期权到期日 / 期权链
# ════════════════════════════════════════════════════════════

def yf_get_expiries(symbol: str) -> List[dict]:
    """返回 [{date: 'YYYY-MM-DD', timestamp: 0}, ...]"""
    if not _YF_OK or not symbol:
        return []
    sym = symbol.strip().upper()
    try:
        opts = _yf.Ticker(sym).options or ()
    except Exception as e:
        log.warning("yf_get_expiries(%s) failed: %s", sym, e)
        return []
    return [{"date": str(d), "timestamp": 0} for d in opts]


def yf_get_chain(symbol: str, expiry: str) -> List[dict]:
    """
    返回 [{strike, call: {...}, put: {...}}, ...] 跟 TigerQuoteClient 一致。
    expiry 必须是 'YYYY-MM-DD'。
    """
    if not _YF_OK or not symbol or not expiry:
        return []
    sym = symbol.strip().upper()
    try:
        chain = _yf.Ticker(sym).option_chain(str(expiry))
    except Exception as e:
        log.warning("yf_get_chain(%s, %s) failed: %s", sym, expiry, e)
        return []

    by_strike: dict = {}
    for _, row in chain.calls.iterrows():
        strike = _safe_float(row.get("strike"))
        if strike is None:
            continue
        slot = by_strike.setdefault(strike, {"strike": strike, "call": None, "put": None})
        slot["call"] = _yf_leg(row)
    for _, row in chain.puts.iterrows():
        strike = _safe_float(row.get("strike"))
        if strike is None:
            continue
        slot = by_strike.setdefault(strike, {"strike": strike, "call": None, "put": None})
        slot["put"] = _yf_leg(row)

    return sorted(by_strike.values(), key=lambda x: x["strike"])


def _yf_leg(row) -> dict:
    return {
        "identifier": str(row.get("contractSymbol") or ""),
        "latest_price": _safe_float(row.get("lastPrice")),
        "bid": _safe_float(row.get("bid")),
        "ask": _safe_float(row.get("ask")),
        "volume": _safe_int(row.get("volume")),
        "open_interest": _safe_int(row.get("openInterest")),
        "implied_vol": _safe_float(row.get("impliedVolatility")),
    }


# ════════════════════════════════════════════════════════════
# 小工具
# ════════════════════════════════════════════════════════════

def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
