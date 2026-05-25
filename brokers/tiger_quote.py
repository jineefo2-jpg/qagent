"""
brokers/tiger_quote.py — Tiger QuoteClient 封装 (X8 a)

只读行情/期权链查询,跟 TradeClient (tiger_adapter.py) 完全独立。
设计:
  * 凭证由 BrokerRegistry / 调用方注入 TigerCredentials,不读 env
  * symbol_names 本地缓存 24h,做模糊搜索完全离线
  * 所有方法包成 plain dict / list,不暴露 Tiger DataFrame
  * 失败包装到 BrokerError 子类,跟 TradeClient 错误体系一致

不会:
  * 不做下单/撤单 (那是 tiger_adapter 的事)
  * 不订阅推送 (那是 tiger_push 的事)
  * 永远不会把 private_key 写日志
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from .base import (
    BrokerAuthError,
    BrokerError,
    BrokerNetworkError,
    TigerCredentials,
    redact_credentials,
)


log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# SDK lazy import
# ════════════════════════════════════════════════════════════

def _import_quote_sdk():
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.common.consts import Language, Market
        return {
            "TigerOpenClientConfig": TigerOpenClientConfig,
            "QuoteClient": QuoteClient,
            "Language": Language,
            "Market": Market,
        }
    except ImportError as e:
        raise BrokerError(
            "缺少 tigeropen 依赖,请运行: pip install tigeropen>=3.2.0,<4.0.0"
        ) from e


def _strip_pem_markers(pem: str) -> str:
    body = [
        ln for ln in pem.strip().splitlines()
        if "BEGIN" not in ln and "END" not in ln
    ]
    return "".join(body).strip()


# ════════════════════════════════════════════════════════════
# TigerQuoteClient
# ════════════════════════════════════════════════════════════

# Tiger get_symbol_names 单次响应里 US 股票超过 1 万条 — 全量拉一次,
# 缓存 24h 就够 (新上市/退市频率远低于一天)。
_SYMBOL_NAMES_TTL_SEC = 24 * 3600
# get_briefs / option_briefs 短缓存,避免前端高频轮询触发限流
_BRIEF_TTL_SEC = 2


class TigerQuoteClient:
    """Tiger QuoteClient 单用户封装。线程安全。"""

    def __init__(self, credentials: TigerCredentials):
        if not isinstance(credentials, TigerCredentials):
            raise BrokerError(
                f"TigerQuoteClient requires TigerCredentials, got {type(credentials).__name__}"
            )
        self.tiger_id = credentials.tiger_id
        self._private_key = credentials.private_key
        self.account = credentials.account
        self.license = credentials.license
        self._client = None
        self._sdk = None
        self._lock = threading.RLock()

        # symbol-names cache: market(str) → (ts, [(symbol, name), ...])
        self._names_cache: Dict[str, Tuple[float, List[Tuple[str, str]]]] = {}
        # brief cache: symbol → (ts, dict)
        self._brief_cache: Dict[str, Tuple[float, dict]] = {}

    def __repr__(self) -> str:
        return f"<TigerQuoteClient tiger_id={self.tiger_id!r}>"

    # ── client construction (lazy) ──────────────────────────

    def _ensure_client(self):
        with self._lock:
            if self._client is not None:
                return self._client
            if not (self.tiger_id and self._private_key):
                raise BrokerAuthError("Tiger 凭证未配置完整")
            self._sdk = _import_quote_sdk()
            try:
                config = self._sdk["TigerOpenClientConfig"]()
                config.private_key = _strip_pem_markers(self._private_key)
                config.tiger_id = self.tiger_id
                # QuoteClient 不需要 account,但传上无害
                if self.account:
                    config.account = self.account
                if self.license:
                    config.license = self.license
                config.language = self._sdk["Language"].zh_CN
                self._client = self._sdk["QuoteClient"](config)
            except Exception as e:
                raise BrokerAuthError(
                    f"Tiger QuoteClient 初始化失败: {type(e).__name__}"
                ) from e
            return self._client

    # ── symbol search (模糊匹配) ────────────────────────────

    def _load_symbol_names(self, market: str) -> List[Tuple[str, str]]:
        """拉全市场 symbol-name 列表,24h 缓存。"""
        market = (market or "US").upper()
        cached = self._names_cache.get(market)
        if cached and (time.monotonic() - cached[0]) < _SYMBOL_NAMES_TTL_SEC:
            return cached[1]

        client = self._ensure_client()
        Market = self._sdk["Market"]
        market_enum = getattr(Market, market, Market.US)
        try:
            rows = client.get_symbol_names(market=market_enum)
        except Exception as e:
            raise self._wrap_error(e) from e

        # SDK 返回 list[tuple(symbol, name)] 或 list[list[symbol, name]]
        normalized: List[Tuple[str, str]] = []
        for r in rows or []:
            try:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    normalized.append((str(r[0]).strip(), str(r[1]).strip()))
                elif isinstance(r, dict):
                    sym = str(r.get("symbol") or r.get("contractId") or "").strip()
                    name = str(r.get("name") or "").strip()
                    if sym:
                        normalized.append((sym, name))
            except Exception:
                continue

        self._names_cache[market] = (time.monotonic(), normalized)
        return normalized

    def search_symbols(self, query: str, market: str = "US", limit: int = 20) -> List[dict]:
        """
        模糊搜索 symbol。匹配规则:symbol 前缀 > symbol 包含 > name 包含 (不区分大小写)。
        返回: [{"symbol": "AAPL", "name": "Apple Inc.", "market": "US"}, ...]
        """
        q = (query or "").strip().upper()
        if not q:
            return []
        all_names = self._load_symbol_names(market)

        sym_prefix: List[Tuple[str, str]] = []
        sym_contain: List[Tuple[str, str]] = []
        name_contain: List[Tuple[str, str]] = []
        for sym, name in all_names:
            su = sym.upper()
            if su.startswith(q):
                sym_prefix.append((sym, name))
            elif q in su:
                sym_contain.append((sym, name))
            elif q in name.upper():
                name_contain.append((sym, name))
            if len(sym_prefix) >= limit:
                break

        merged = (sym_prefix + sym_contain + name_contain)[:limit]
        return [
            {"symbol": s, "name": n, "market": market.upper()}
            for s, n in merged
        ]

    # ── 实时行情 ────────────────────────────────────────────

    def get_brief(self, symbol: str) -> dict:
        """
        当前股价快照。优先实时 (get_briefs),无权限自动降级到 Tiger 延迟
        (get_stock_delay_briefs,~15min)。返回里 source 标识数据档位。

        字段: latest_price / change / change_percent / volume / prev_close
              + source: 'tiger_realtime' | 'tiger_delay'
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return {}

        cached = self._brief_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < _BRIEF_TTL_SEC:
            return cached[1]

        client = self._ensure_client()
        # 1) 先试实时
        used_delay = False
        try:
            rows = client.get_briefs(symbols=[symbol], include_hour_trading=True)
            out = self._parse_realtime_brief(symbol, rows)
        except Exception as e:
            wrapped = self._wrap_error(e)
            # 实时权限不足 → 降级延迟
            if isinstance(wrapped, BrokerAuthError):
                out = self._fetch_delay_brief(client, symbol)
                used_delay = True
            else:
                raise wrapped from e

        if used_delay and out.get("available"):
            out["source"] = "tiger_delay"
            out["delay_minutes"] = 15
        elif out.get("available"):
            out["source"] = "tiger_realtime"

        self._brief_cache[symbol] = (time.monotonic(), out)
        return out

    @staticmethod
    def _parse_realtime_brief(symbol: str, rows) -> dict:
        if not rows:
            return {"symbol": symbol, "available": False}
        r = rows[0]
        return {
            "symbol": symbol,
            "available": True,
            "latest_price": _safe_float(getattr(r, "latest_price", None)),
            "prev_close": _safe_float(getattr(r, "prev_close", None)),
            "open": _safe_float(getattr(r, "open", None)),
            "high": _safe_float(getattr(r, "high", None)),
            "low": _safe_float(getattr(r, "low", None)),
            "volume": _safe_int(getattr(r, "volume", None)),
            "change": _safe_float(getattr(r, "change", None)),
            "change_percent": (
                _safe_float(getattr(r, "change_percent", None))
                or _safe_float(getattr(r, "change_pct", None))
            ),
            "latest_time": _safe_int(getattr(r, "latest_time", None)),
        }

    def _fetch_delay_brief(self, client, symbol: str) -> dict:
        """
        Tiger 延迟行情 (get_stock_delay_briefs)。返回 DataFrame,列:
          symbol / pre_close / time / volume / open / high / low / close / halted
        没有独立的 change 字段,要自己算 close - pre_close。
        """
        try:
            df = client.get_stock_delay_briefs(symbols=[symbol])
        except Exception as e:
            # 连延迟权限都没 (极少见) → 抛原错误,server 层会兜底
            raise self._wrap_error(e) from e
        if df is None or len(df) == 0:
            return {"symbol": symbol, "available": False}
        row = df.iloc[0]
        close = _safe_float(row.get("close"))
        prev = _safe_float(row.get("pre_close"))
        change = (close - prev) if (close is not None and prev is not None) else None
        pct = (change / prev * 100) if (change is not None and prev) else None
        return {
            "symbol": symbol,
            "available": close is not None,
            "latest_price": close,
            "prev_close": prev,
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "volume": _safe_int(row.get("volume")),
            "change": change,
            "change_percent": pct,
            "latest_time": _safe_int(row.get("time")),
        }

    # ── 期权链 ─────────────────────────────────────────────

    def get_option_expirations(self, symbol: str) -> List[dict]:
        """
        返回 [{"date": "2026-06-19", "timestamp": 1750291200000}, ...]
        Tiger SDK 返回 DataFrame, 我们 to_dict('records') 再 normalize。
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return []
        client = self._ensure_client()
        try:
            df = client.get_option_expirations(symbols=symbol)
        except Exception as e:
            raise self._wrap_error(e) from e
        if df is None or df.empty:
            return []
        out: List[dict] = []
        for rec in df.to_dict("records"):
            date_str = str(rec.get("date") or rec.get("expiry") or "").strip()
            ts = rec.get("timestamp") or rec.get("expiry") or 0
            try:
                ts_int = int(ts)
            except Exception:
                ts_int = 0
            if not date_str and ts_int:
                # Tiger 返回的 timestamp 单位是毫秒
                date_str = time.strftime("%Y-%m-%d", time.gmtime(ts_int / 1000))
            if date_str:
                out.append({"date": date_str, "timestamp": ts_int})
        # 按日期升序
        out.sort(key=lambda d: d["date"])
        return out

    def get_option_chain(self, symbol: str, expiry) -> List[dict]:
        """
        返回 [{"strike": 220, "call": {...}, "put": {...}}, ...]

        expiry: 既支持 "YYYY-MM-DD" 字符串,也支持毫秒时间戳。
        """
        symbol = (symbol or "").strip().upper()
        if not symbol or expiry in (None, ""):
            return []
        client = self._ensure_client()
        try:
            df = client.get_option_chain(symbol=symbol, expiry=expiry)
        except Exception as e:
            raise self._wrap_error(e) from e
        if df is None or df.empty:
            return []

        rows = df.to_dict("records")
        # Tiger 每行是单边 (CALL 或 PUT) — 我们按 (strike, expiry) 合并
        by_strike: Dict[float, dict] = {}
        for r in rows:
            strike = _safe_float(r.get("strike"))
            if strike is None:
                continue
            put_call = str(r.get("put_call") or r.get("right") or "").upper()
            leg = {
                "identifier": str(r.get("identifier") or r.get("symbol") or ""),
                "latest_price": _safe_float(r.get("latest_price")),
                "bid": _safe_float(r.get("bid_price") or r.get("bid")),
                "ask": _safe_float(r.get("ask_price") or r.get("ask")),
                "volume": _safe_int(r.get("volume")),
                "open_interest": _safe_int(r.get("open_interest")),
                "implied_vol": _safe_float(r.get("implied_vol") or r.get("implied_volatility")),
            }
            slot = by_strike.setdefault(strike, {"strike": strike, "call": None, "put": None})
            if put_call.startswith("C"):
                slot["call"] = leg
            elif put_call.startswith("P"):
                slot["put"] = leg

        out = sorted(by_strike.values(), key=lambda d: d["strike"])
        return out

    def get_option_briefs(self, identifiers: List[str]) -> List[dict]:
        """实时期权报价 (LLM 工具用)。Tiger 返回 DataFrame。"""
        identifiers = [i for i in (identifiers or []) if i]
        if not identifiers:
            return []
        client = self._ensure_client()
        try:
            df = client.get_option_briefs(identifiers=identifiers)
        except Exception as e:
            raise self._wrap_error(e) from e
        if df is None or df.empty:
            return []
        out = []
        for r in df.to_dict("records"):
            out.append({
                "identifier": str(r.get("identifier") or r.get("symbol") or ""),
                "latest_price": _safe_float(r.get("latest_price")),
                "bid": _safe_float(r.get("bid_price") or r.get("bid")),
                "ask": _safe_float(r.get("ask_price") or r.get("ask")),
                "volume": _safe_int(r.get("volume")),
                "open_interest": _safe_int(r.get("open_interest")),
            })
        return out

    # ── helpers ─────────────────────────────────────────────

    def _wrap_error(self, exc: Exception) -> BrokerError:
        name = type(exc).__name__
        msg = redact_credentials(str(exc))
        low = msg.lower()
        if ("sign" in low or "auth" in low or "permission" in low
                or "private key" in low or "invalid token" in low):
            return BrokerAuthError(f"Tiger {name}: {msg}")
        if ("timeout" in low or "network" in low or "connect" in low
                or "name resolution" in low or "unreachable" in low):
            return BrokerNetworkError(f"Tiger {name}: {msg}")
        return BrokerError(f"Tiger {name}: {msg}")


# ════════════════════════════════════════════════════════════
# 小工具
# ════════════════════════════════════════════════════════════

def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        # pandas NaN 会变成 float('nan')
        if f != f:
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


# ════════════════════════════════════════════════════════════
# 进程内缓存 (跨请求复用同一个用户的 QuoteClient)
# ════════════════════════════════════════════════════════════

_CACHE: Dict[str, TigerQuoteClient] = {}
_CACHE_LOCK = threading.Lock()


def get_quote_client(user_scope: str, credentials: TigerCredentials) -> TigerQuoteClient:
    """
    按 user_scope 缓存 TigerQuoteClient 实例。credentials 来源由调用方决定
    (通常通过 credentials_store.store.load 解密拿到)。
    """
    with _CACHE_LOCK:
        existing = _CACHE.get(user_scope)
        if existing is not None and existing.tiger_id == credentials.tiger_id:
            return existing
        client = TigerQuoteClient(credentials)
        _CACHE[user_scope] = client
        return client


def drop_quote_client(user_scope: str) -> None:
    """解绑 / 凭证轮换后调用,扔掉旧实例。"""
    with _CACHE_LOCK:
        _CACHE.pop(user_scope, None)
