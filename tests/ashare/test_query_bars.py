"""Task 11：get_bars / get_price_panel / get_daily_basic / get_index_bars（D8 后复权 / D9 停牌行）。

market_db：A 在 2024-01-15~17 停牌（占位行 OHLC=前收 vol=0）；close = base + i×0.1；adj_factor = 1 + i×0.01。
"""
from __future__ import annotations
import datetime as dt
import math
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import query
from ashare.data.query import AsOfDateError, UnknownFieldError

D = dt.date


@pytest.fixture
def q(market_db):
    query.open_db(market_db)
    yield query
    query.close_db()


# ══════════════ get_bars ══════════════
def test_lookback_counts_trading_days_not_records(q):
    """A 中间停牌 3 天：lookback=10 必须返回 10 行（含 3 行占位），不是 13 行。"""
    df = q.get_bars("2024-01-19", ["A00001.SZ"], lookback=10)
    assert len(df) == 10
    assert df.index.get_level_values("trade_date").min() == D(2024, 1, 8)
    assert int(df["is_suspended"].sum()) == 3


def test_suspended_rows_have_nan_ohlc_on_output(q):
    """存储层按 D9 存前收；query 输出层把停牌日 OHLC 置 NaN，由因子自己决定是否填充（架构 §4.1）。"""
    df = q.get_bars("2024-01-19", ["A00001.SZ"], lookback=10)
    sus = df[df["is_suspended"]]
    assert sus[["open", "high", "low", "close"]].isna().all().all()
    assert (sus["vol"] == 0).all()


def test_hfq_prices_equal_raw_times_adj_factor(q, market_db):
    df = q.get_bars("2024-01-05", ["B00002.SZ"], lookback=3, adjust="hfq")
    raw = q.get_bars("2024-01-05", ["B00002.SZ"], lookback=3, adjust="none")
    # 直接读库核对
    w = duckdb.connect(market_db, read_only=True)
    rows = w.execute("SELECT trade_date, close, adj_factor FROM daily_bar WHERE ts_code='B00002.SZ' "
                     "AND trade_date BETWEEN DATE '2024-01-03' AND DATE '2024-01-05' ORDER BY trade_date").fetchall()
    w.close()
    for (d, close, adj), (idx, r) in zip(rows, df.iterrows()):
        assert idx[1] == d and math.isclose(r["close"], close * adj)
    assert math.isclose(raw.iloc[-1]["close"], rows[-1][1])


def test_get_bars_never_returns_limit_columns(q):
    df = q.get_bars("2024-01-05", ["A00001.SZ"], lookback=2)
    assert "limit_up" not in df.columns and "limit_down" not in df.columns
    assert "is_suspended" in df.columns


def test_get_bars_upper_bound_is_as_of_date_inclusive(q):
    df = q.get_bars("2024-01-05", ["A00001.SZ", "B00002.SZ"], lookback=2)
    assert df.index.get_level_values("trade_date").max() == D(2024, 1, 5)
    assert set(df.index.get_level_values("ts_code")) == {"A00001.SZ", "B00002.SZ"}


def test_get_bars_start_and_fields(q):
    df = q.get_bars("2024-01-05", ["A00001.SZ"], start="2024-01-03", fields=("close",))
    assert list(df.columns) == ["close", "is_suspended"]
    assert len(df) == 3


def test_get_bars_beyond_calendar_raises(q):
    with pytest.raises(AsOfDateError):
        q.get_bars("2030-01-01", ["A00001.SZ"], lookback=1)


def test_get_bars_unknown_field_raises(q):
    with pytest.raises(UnknownFieldError):
        q.get_bars("2024-01-05", ["A00001.SZ"], lookback=1, fields=("limit_up",))


def test_get_bars_empty_is_typed_empty(q):
    df = q.get_bars("2024-01-05", ["ZZZZZZ.SZ"], lookback=3)
    assert df.empty and list(df.columns) == ["open", "high", "low", "close", "vol", "amount", "is_suspended"]


def test_delisted_stock_has_no_rows_after_delist(q):
    df = q.get_bars("2024-01-31", ["C00003.SH"], lookback=10)
    assert df.index.get_level_values("trade_date").max() == D(2024, 1, 24)


# ══════════════ get_price_panel ══════════════
def test_price_panel_wide_shape_and_nan_on_suspension(q):
    p = q.get_price_panel("2024-01-19", ["A00001.SZ", "B00002.SZ"], field="close", lookback=10)
    assert p.shape == (10, 2)
    assert list(p.columns) == ["A00001.SZ", "B00002.SZ"]
    assert p.loc[D(2024, 1, 16), "A00001.SZ"] != p.loc[D(2024, 1, 16), "A00001.SZ"]     # NaN，不 ffill
    assert not pd.isna(p.loc[D(2024, 1, 16), "B00002.SZ"])


# ══════════════ get_daily_basic ══════════════
def test_daily_basic_single_day_and_lookback(q):
    one = q.get_daily_basic("2024-01-05", ["A00001.SZ", "B00002.SZ"])
    assert list(one.index) == ["A00001.SZ", "B00002.SZ"] and "total_mv" in one.columns
    many = q.get_daily_basic("2024-01-05", ["A00001.SZ"], lookback=3)
    assert isinstance(many.index, pd.MultiIndex) and len(many) == 3


# ══════════════ get_index_bars ══════════════
def test_index_bars(market_db):
    from ashare.data import _db
    w = _db.connect_write(market_db)
    w.executemany("INSERT INTO index_daily (ts_code, trade_date, open, high, low, close, vol, amount, pe_ttm) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  [("000985.CSI", D(2024, 1, 4), 1, 1, 1, 100.0, 1, 1, 12.0),
                   ("000985.CSI", D(2024, 1, 5), 1, 1, 1, 101.0, 1, 1, 12.1)])
    w.close()
    query.open_db(market_db)
    try:
        df = query.get_index_bars("2024-01-05", "000985.CSI", lookback=2)
        assert list(df.index) == [D(2024, 1, 4), D(2024, 1, 5)]
        assert list(df.columns) == ["close", "pe_ttm"]
    finally:
        query.close_db()


# ══════════════ preload 命中 ══════════════
def test_preload_is_used_by_get_bars(q):
    q.preload("2024-01-02", "2024-01-12", tables=("daily_bar",))
    df = q.get_bars("2024-01-05", ["A00001.SZ"], lookback=2)
    assert len(df) == 2
    q.clear_preload()


def test_empty_fields_rejected(q):
    with pytest.raises(UnknownFieldError):
        q.get_bars("2024-01-05", ["A00001.SZ"], lookback=1, fields=())
    with pytest.raises(UnknownFieldError):
        q.get_daily_basic("2024-01-05", ["A00001.SZ"], fields=())


def test_daily_basic_uses_preload(q):
    q.preload("2024-01-02", "2024-01-12", tables=("daily_basic",))
    one = q.get_daily_basic("2024-01-05", ["A00001.SZ", "B00002.SZ"])
    assert list(one.index) == ["A00001.SZ", "B00002.SZ"] and one.loc["A00001.SZ", "total_mv"] == 1e6
    q.clear_preload()


def test_preload_invalidates_the_wide_panel_cache(q):
    """同一进程连跑两次回测（`gate1` 的 IS→OOS 就是）：第二次 `preload` 换了窗，
    宽面板缓存必须跟着失效 —— 否则第二次回测读到的是**第一次那个窗**的价格面板，
    新窗里的日期整段缺失、新窗才上市的票整列不存在，价格静默变 NaN。
    2026-08-27 样本外仪式实测：burn 死在这里（prev_weights 含 NaN，全是 2020 后上市的票）。"""
    days = q.get_trade_dates("2024-01-12", start="2024-01-02")
    early, late = days[:2], days[-2:]
    q.preload(early[0], early[-1], tables=("daily_bar",))
    first = q.get_price_panel(early[-1], ["A00001.SZ"], "close", lookback=2)
    assert first.notna().to_numpy().any()          # 第一个窗：正常建面板

    q.preload(late[0], late[-1], tables=("daily_bar",))
    second = q.get_price_panel(late[-1], ["A00001.SZ"], "close", lookback=2)
    q.clear_preload()
    assert list(second.index) and second.notna().to_numpy().any(), (
        "第二个 preload 窗读回空/NaN —— _PRELOAD_WIDE 仍是上一个窗的面板")
    assert second.index[-1] == late[-1]
