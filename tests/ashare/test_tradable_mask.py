"""Task 11：get_tradable_mask —— D6 的证据链。唯一首参非 as_of_date 的函数。"""
from __future__ import annotations
import datetime as dt
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, query

D = dt.date
COLS = ["can_buy", "can_sell", "reason", "open_hfq", "close_hfq", "amount", "amplitude"]


@pytest.fixture
def q(market_db):
    query.open_db(market_db)
    yield query
    query.close_db()


def _set_bar(path, code, d, **cols):
    query.close_db()
    w = _db.connect_write(path)
    sets = ", ".join(f"{k} = ?" for k in cols)
    w.execute(f"UPDATE daily_bar SET {sets} WHERE ts_code = ? AND trade_date = ?", [*cols.values(), code, d])
    w.close()
    query.open_db(path)


def test_columns_and_normal_day(q):
    m = q.get_tradable_mask("2024-01-05", ["A00001.SZ", "B00002.SZ"])
    assert list(m.columns) == COLS
    assert m.loc["A00001.SZ", "can_buy"] and m.loc["A00001.SZ", "can_sell"] and m.loc["A00001.SZ", "reason"] == ""
    assert m.loc["A00001.SZ", "amount"] > 0 and m.loc["A00001.SZ", "amplitude"] > 0


def test_open_hfq_is_raw_open_times_adj(q, market_db):
    m = q.get_tradable_mask("2024-01-05", ["B00002.SZ"])
    w = duckdb.connect(market_db, read_only=True)
    o, c, adj = w.execute("SELECT open, close, adj_factor FROM daily_bar WHERE ts_code='B00002.SZ' "
                          "AND trade_date=DATE '2024-01-05'").fetchone()
    w.close()
    assert abs(m.loc["B00002.SZ", "open_hfq"] - o * adj) < 1e-9
    assert abs(m.loc["B00002.SZ", "close_hfq"] - c * adj) < 1e-9


def test_suspended_blocks_both_sides(q):
    m = q.get_tradable_mask("2024-01-16", ["A00001.SZ"])
    assert (not m.loc["A00001.SZ", "can_buy"]) and (not m.loc["A00001.SZ", "can_sell"])
    assert m.loc["A00001.SZ", "reason"] == "suspended"


def test_limit_up_seal_blocks_buy_only(q, market_db):
    """一字涨停：open == limit_up 且 high == low → 买不进，卖得出。"""
    _set_bar(market_db, "B00002.SZ", D(2024, 1, 5), open=22.0, high=22.0, low=22.0, close=22.0, limit_up=22.0, limit_down=18.0)
    m = query.get_tradable_mask("2024-01-05", ["B00002.SZ"])
    assert (not m.loc["B00002.SZ", "can_buy"]) and m.loc["B00002.SZ", "can_sell"]
    assert m.loc["B00002.SZ", "reason"] == "limit_up_seal"


def test_limit_down_seal_blocks_sell_only(q, market_db):
    _set_bar(market_db, "B00002.SZ", D(2024, 1, 5), open=18.0, high=18.0, low=18.0, close=18.0, limit_up=22.0, limit_down=18.0)
    m = query.get_tradable_mask("2024-01-05", ["B00002.SZ"])
    assert m.loc["B00002.SZ", "can_buy"] and (not m.loc["B00002.SZ", "can_sell"])
    assert m.loc["B00002.SZ", "reason"] == "limit_down_seal"


def test_touch_limit_but_not_sealed_is_tradable(q, market_db):
    """摸板不封板（open==limit_up 但 high != low）→ 当天可成交。"""
    _set_bar(market_db, "B00002.SZ", D(2024, 1, 5), open=22.0, high=22.0, low=21.0, close=21.5, limit_up=22.0, limit_down=18.0)
    m = query.get_tradable_mask("2024-01-05", ["B00002.SZ"])
    assert m.loc["B00002.SZ", "can_buy"] and m.loc["B00002.SZ", "can_sell"]


def test_limit_unknown_blocks_both_sides_conservatively(q, market_db):
    """limit_source='unknown'（涨跌停算不出）→ 两侧皆不可交易：宁可少成交，不可凭空造成交。"""
    _set_bar(market_db, "B00002.SZ", D(2024, 1, 5), limit_up=None, limit_down=None, limit_source="unknown")
    m = query.get_tradable_mask("2024-01-05", ["B00002.SZ"])
    assert (not m.loc["B00002.SZ", "can_buy"]) and (not m.loc["B00002.SZ", "can_sell"])
    assert m.loc["B00002.SZ", "reason"] == "limit_unknown"


def test_no_quote_row(q):
    m = q.get_tradable_mask("2024-01-05", ["ZZZZZZ.SZ"])
    assert list(m.index) == ["ZZZZZZ.SZ"]
    assert (not m.loc["ZZZZZZ.SZ", "can_buy"]) and m.loc["ZZZZZZ.SZ", "reason"] == "no_quote"


def test_delisted_forces_sell_only(q):
    """delist_date <= exec_date：只能卖（强制清仓路径），不能买。C 于 2024-01-24 退市。"""
    m = q.get_tradable_mask("2024-01-24", ["C00003.SH"])
    assert (not m.loc["C00003.SH", "can_buy"]) and m.loc["C00003.SH", "can_sell"]
    assert m.loc["C00003.SH", "reason"] == "delisted"


def test_raw_limit_prices_do_not_leak(q):
    m = q.get_tradable_mask("2024-01-05", ["A00001.SZ"])
    assert "limit_up" not in m.columns and "limit_down" not in m.columns and "open" not in m.columns


def test_exec_date_must_be_a_trading_day(q):
    with pytest.raises(query.AsOfDateError):
        q.get_tradable_mask("2024-01-06", ["A00001.SZ"])          # 周六


def test_delisted_after_last_bar_still_sellable_at_last_close(q):
    """退市后 daily_bar 不再有行（C 最后一根 K 线是 01-24）。01-25 及以后仍须给强平路径：
    can_sell=True、reason=delisted、清仓价 = 最后一根非停牌 K 线的后复权收盘价（引擎再按 B8 打折）。"""
    m = q.get_tradable_mask("2024-01-26", ["C00003.SH"])
    r = m.loc["C00003.SH"]
    assert (not r.can_buy) and r.can_sell and r.reason == "delisted"
    last = q.get_bars("2024-01-24", ["C00003.SH"], lookback=1)["close"].iloc[0]
    assert abs(r.close_hfq - last) < 1e-9 and abs(r.open_hfq - last) < 1e-9


def test_unknown_code_is_still_no_quote(q):
    m = q.get_tradable_mask("2024-01-26", ["ZZZZZZ.SZ"])
    assert m.loc["ZZZZZZ.SZ", "reason"] == "no_quote"
