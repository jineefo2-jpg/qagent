"""老工具的 A 股本地路由（2026-08-26 工具审计）。

守三件事：A 股历史价/日历改走本地真值（后复权/真日历），factor_score 的 A 股
个股被引导去本地因子库，而 **ETF / 概念词 / 非 A 股路径一根手指都不碰**。
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("duckdb")
from ashare.agent_tools import local_calendar, local_history, to_ts_code

_MARKET = pathlib.Path("data/ashare_market.duckdb")
_need_db = pytest.mark.skipif(not _MARKET.exists(), reason="真实 market 库不存在")


@pytest.fixture(autouse=True)
def _fresh_query_state():
    """全套联跑时，前面的测试可能把 query 钉在/停在自己的临时库上（close_db 有意保留
    `_market_path`，见其文档字符串）。真库测试必须显式重开真库路径，前后各清一次。"""
    from ashare.data import query
    query.close_db()
    if _MARKET.exists():
        query.open_db(str(_MARKET))
    yield
    query.close_db()


def test_to_ts_code_conversions():
    assert to_ts_code("600519") == "600519.SH"
    assert to_ts_code("000858") == "000858.SZ"
    assert to_ts_code("300750") == "300750.SZ"
    assert to_ts_code("830799") == "830799.BJ"
    assert to_ts_code("sh600519") == "600519.SH"
    assert to_ts_code("600519.SS") == "600519.SH"
    assert to_ts_code("000858.SZ") == "000858.SZ"
    for bad in ("AAPL", "0700", "510300X", "", None):
        assert to_ts_code(bad) is None, bad


@_need_db
def test_local_history_shape_matches_web_contract():
    r = local_history("600519", 30)
    assert r["success"] is True and r["market"] == "A股"
    # 与 _hist_akshare 成功体同构：historical_prices 的统一后处理（returns/缓存）直接复用
    for k in ("symbol", "days_returned", "dates", "open", "close", "high", "low",
              "volume", "data_source"):
        assert k in r, k
    n = r["days_returned"]
    assert 0 < n <= 30 and n == len(r["dates"]) == len(r["close"]) == len(r["volume"])
    assert "后复权" in r["data_source"]
    assert all(c > 0 for c in r["close"])


@_need_db
def test_local_history_long_delisted_stock_falls_back():
    """近窗查早已退市的票（乐视网 300104，2020 退市）本地正确返回失败 → 调用方回退网络源。"""
    r = local_history("300104", 20)
    assert r["success"] is False


def test_local_history_rejects_non_ashare():
    assert local_history("AAPL", 30)["success"] is False


@_need_db
def test_local_calendar_between_matches_query():
    from ashare.data import query
    r = local_calendar("trading_days_between", "2019-01-01", "2019-12-31")
    assert r is not None and r["success"] is True
    query.open_db()
    assert r["trading_days"] == len(query.get_trade_dates("2019-12-31", start="2019-01-01"))
    assert "本地库" in r["note"]


def test_local_calendar_unknown_action_falls_back():
    assert local_calendar("parse", "2024-01-01", None) is None


# ══════════════ quant_agent 侧的路由行为 ══════════════
qa = pytest.importorskip("quant_agent")


def test_factor_score_redirects_ashare_stock_to_local_factors():
    r = qa.factor_score("600519")
    assert r["success"] is False and r["error_type"] == "use_local_factors"
    assert "get_factor_exposure" in r["hint"]


def test_factor_score_does_not_intercept_etf_or_concept():
    """拦截必须排在 ETF/概念词短路【之后】—— 510300 是 ETF，要拿到 ETF 的专属提示。"""
    assert qa.factor_score("510300")["error_type"] == "not_applicable_to_etf"
    assert qa.factor_score("芯片ETF")["error_type"] == "not_a_ticker"


@_need_db
def test_historical_prices_serves_ashare_from_local():
    from cache import cache
    cache.delete("quant:price:600519:63")            # 自清：跨进程 Redis 缓存可能存着旧的回退结果
    r = qa.historical_prices("600519", days=63)
    assert r["success"] is True
    assert "本地" in r["data_source"], f"走了 {r['data_source']}，未命中本地路由"
    assert r["sources_tried"][0]["source"] == "本地数仓"
    assert len(r["returns"]) == len(r["close"]) - 1  # 统一后处理照常工作


def test_trading_calendar_still_works_either_path():
    r = qa.trading_calendar("today")
    assert r["success"] is True and "date" in r
