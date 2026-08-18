"""Task 9：query.py 骨架 —— 只读连接 / snapshot_id / 日历 / preload / 异常。"""
from __future__ import annotations
import datetime as dt
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import query
from ashare.data.query import AsOfDateError, QueryError

D = dt.date


@pytest.fixture
def q(market_db):
    query.open_db(market_db)
    yield query
    query.close_db()


# ══════════════ 连接与 D1 ══════════════
def test_connection_is_read_only(q):
    with pytest.raises(duckdb.InvalidInputException, match="read-only"):
        q._conn().execute("INSERT INTO calendar VALUES (DATE '2030-01-01', TRUE, NULL)")


def test_open_db_is_idempotent(q, market_db):
    c1 = q._conn()
    q.open_db(market_db)
    assert q._conn() is c1


def test_open_db_missing_file_raises_query_error(tmp_path):
    with pytest.raises(QueryError):
        query.open_db(str(tmp_path / "nope.duckdb"))


# ══════════════ snapshot_id ══════════════
def test_snapshot_id_is_stable_without_writes(q):
    a, b = q.snapshot_id(), q.snapshot_id()
    assert a == b and len(a) == 16


def test_snapshot_id_changes_after_write(market_db):
    from ashare.data import _db
    query.open_db(market_db)
    before = query.snapshot_id()
    query.close_db()                                  # 同进程不可同时持读写连接
    w = _db.connect_write(market_db)
    w.execute("INSERT INTO daily_basic (ts_code, trade_date, total_mv) VALUES ('Z00009.SZ', DATE '2024-01-02', 1.0)")
    w.close()
    query.open_db(market_db)
    after = query.snapshot_id()
    query.close_db()
    assert before != after


# ══════════════ 日历 ══════════════
def test_is_trade_date(q):
    assert q.is_trade_date("2024-01-02") is True
    assert q.is_trade_date("2024-01-01") is False       # 元旦
    assert q.is_trade_date(D(2024, 1, 6)) is False      # 周六


def test_prev_next_trade_date(q):
    assert q.prev_trade_date("2024-01-02") == D(2023, 12, 29)
    assert q.prev_trade_date("2024-01-08", n=2) == D(2024, 1, 4)
    assert q.next_trade_date("2024-01-05") == D(2024, 1, 8)
    assert q.next_trade_date("2024-01-06") == D(2024, 1, 8)     # 非交易日 → 下一交易日
    assert q.next_trade_date("2024-02-02") is None              # 日历尽头 → None，不抛


def test_get_trade_dates_daily_and_weekly_and_monthly(q):
    daily = q.get_trade_dates("2024-01-12", start="2024-01-02")
    assert daily == [D(2024,1,2), D(2024,1,3), D(2024,1,4), D(2024,1,5),
                     D(2024,1,8), D(2024,1,9), D(2024,1,10), D(2024,1,11), D(2024,1,12)]
    weekly = q.get_trade_dates("2024-01-31", start="2024-01-01", freq="W")
    assert weekly == [D(2024,1,5), D(2024,1,12), D(2024,1,19), D(2024,1,26), D(2024,1,31)]
    monthly = q.get_trade_dates("2024-02-02", start="2023-12-01", freq="M")
    assert monthly == [D(2023,12,29), D(2024,1,31), D(2024,2,2)]


def test_weekly_uses_last_trading_day_not_friday(q):
    """2024-01-31 是周三，但它是当周（也是本日历末段）最后一个交易日 → 计入。"""
    assert D(2024,1,31) in q.get_trade_dates("2024-01-31", start="2024-01-29", freq="W")


# ══════════════ 越界与格式 ══════════════
def test_as_of_beyond_calendar_raises(q):
    with pytest.raises(AsOfDateError):
        q.get_trade_dates("2030-01-01")
    with pytest.raises(AsOfDateError):
        q.is_trade_date("2000-01-01")


def test_bad_date_format_raises(q):
    with pytest.raises(AsOfDateError):
        q.is_trade_date("not-a-date")


def test_yyyymmdd_string_accepted(q):
    assert q.is_trade_date("20240102") is True


# ══════════════ preload 骨架 ══════════════
def test_preload_and_clear(q):
    q.preload("2024-01-02", "2024-01-05", tables=("daily_bar",))
    assert q._PRELOAD.get("daily_bar") is not None
    assert len(q._PRELOAD["daily_bar"]) > 0
    q.clear_preload()
    assert q._PRELOAD == {}
