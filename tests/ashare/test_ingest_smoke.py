"""DB 级冒烟：假数据源产出【Tushare 原始形态】(YYYYMMDD 字符串) 的 DataFrame，
经真实 adapter 的 _to_date 归一化，再走真实 DuckDB 写入路径。

这是唯一一层能抓住"单测两边都喂 date 所以看不见"的类型问题：
  - pretrade_date 未转日期 → VARCHAR 写进 DATE 列被 DuckDB 拒绝
  - fetchdf() 的 Timestamp 与 adapter 的 date 混排 → sort TypeError
"""
from __future__ import annotations
import datetime as dt
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, ingest
from ashare.data.sources.tushare import _to_date

D = dt.date


class FakeSource:
    """方法签名与返回列名同 TushareSource；数值形态同 Tushare 原始返回。"""

    def trade_cal(self, start, end, exchange="SSE"):
        return _to_date(pd.DataFrame({
            "exchange": ["SSE"] * 4,
            "cal_date": ["20240102", "20240103", "20240104", "20240105"],
            "is_open": [1, 1, 0, 1],
            "pretrade_date": ["20231229", "20240102", "20240103", "20240103"],
        }))

    def stock_basic(self):
        return _to_date(pd.DataFrame({
            "ts_code": ["600519.SH", "000001.SZ", "600401.SH"],
            "symbol": ["600519", "000001", "600401"],
            "name": ["贵州茅台", "平安银行", "退市海润"],
            "area": [None] * 3, "industry": [None] * 3,
            "market": ["主板"] * 3,
            "list_date": ["20010827", "19910403", "20000101"],
            "delist_date": [None, None, "20190702"],
            "is_hs": ["S", "S", "N"],
        }))

    def namechange(self, ts_code=None):
        # 只有 000001 有改名记录；600519 无记录（触发 basic 兜底路径 → Timestamp 混型）
        return _to_date(pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ", "600401.SH"],
            "name": ["ST深发展", "平安银行", "退市海润"],
            "start_date": ["20120101", "20130701", "20190601"],
            "end_date": ["20130630", None, "20190701"],
            "change_reason": ["", "", ""],
        }))

    # ── 日线三件套：600519 在 1/3 停牌（无行）──
    def daily(self, ts_code=None, trade_date=None, start=None, end=None):
        return _to_date(pd.DataFrame({
            "ts_code": [ts_code] * 2, "trade_date": ["20240102", "20240105"],
            "open": [100.0, 105.0], "high": [101.0, 106.0], "low": [99.0, 104.0],
            "close": [100.5, 105.5], "pre_close": [99.5, 100.5],
            "vol": [1000.0, 1200.0], "amount": [1e5, 1.2e5],
        }))

    def adj_factor(self, ts_code=None, trade_date=None, start=None, end=None):
        return _to_date(pd.DataFrame({
            "ts_code": [ts_code] * 2, "trade_date": ["20240102", "20240105"],
            "adj_factor": [10.0, 10.0],
        }))

    stk_limit_error: Exception | None = None

    def stk_limit(self, ts_code=None, trade_date=None, start=None, end=None):
        if self.stk_limit_error is not None:
            raise self.stk_limit_error
        return _to_date(pd.DataFrame({
            "ts_code": [ts_code], "trade_date": ["20240102"],
            "up_limit": [110.55], "down_limit": [90.45],
        }))


@pytest.fixture
def conn(tmp_db):
    c = _db.connect_write(tmp_db)
    _db.init_schema(c)
    yield c
    c.close()


def test_calendar_ingest_writes_dates_not_strings(conn):
    n = ingest.ingest_calendar(conn, FakeSource(), "20240101", "20240131")
    assert n == 4
    row = conn.execute("SELECT trade_date, is_open, pre_trade_date FROM calendar "
                       "WHERE trade_date = DATE '2024-01-03'").fetchone()
    assert row == (D(2024, 1, 3), True, D(2024, 1, 2))
    assert ingest.job_state(conn, "calendar:all") == "DONE"


def test_stock_basic_and_status_ingest_end_to_end(conn):
    src = FakeSource()
    assert ingest.ingest_stock_basic(conn, src) == 3
    n = ingest.ingest_stock_status(conn, src)
    assert n == 4                                   # 000001×2 + 600401×1 + 600519 兜底×1

    st = conn.execute("SELECT status FROM stock_status WHERE ts_code='000001.SZ' "
                      "ORDER BY start_date").fetchall()
    assert [r[0] for r in st] == ["ST", "NORMAL"]

    mt = conn.execute("SELECT start_date, end_date, status FROM stock_status "
                      "WHERE ts_code='600519.SH'").fetchone()
    assert mt == (D(2001, 8, 27), None, "NORMAL"), "无改名记录的股票补一条覆盖全生命周期的 NORMAL"

    hr = conn.execute("SELECT status FROM stock_status WHERE ts_code='600401.SH'").fetchone()
    assert hr[0] == "DELIST_PERIOD"


def test_reingest_is_idempotent_and_refreshes_ingested_at(conn):
    src = FakeSource()
    ingest.ingest_stock_basic(conn, src)
    t1 = conn.execute("SELECT _ingested_at FROM stock_basic WHERE ts_code='600519.SH'").fetchone()[0]
    ingest.ingest_stock_basic(conn, src)
    cnt = conn.execute("SELECT count(*) FROM stock_basic").fetchone()[0]
    t2 = conn.execute("SELECT _ingested_at FROM stock_basic WHERE ts_code='600519.SH'").fetchone()[0]
    assert cnt == 3, "按主键覆盖，重跑不翻倍"
    assert t2 >= t1, "_ingested_at 语义是最后写入时间（snapshot_id 依赖它）"


# ══════════════ ingest_daily_bar：真实 DuckDB 端到端 ══════════════
def _prime(conn, src):
    ingest.ingest_calendar(conn, src, "20240101", "20240131")
    ingest.ingest_stock_basic(conn, src)
    ingest.ingest_stock_status(conn, src)


def test_ingest_daily_bar_end_to_end_with_placeholder_row(conn):
    """曾在此处崩溃：fetchdf() 的 Timestamp/NaT 与日历的 date 比较 → TypeError，任务落 RETRY、0 行。"""
    src = FakeSource()
    _prime(conn, src)
    n = ingest.ingest_daily_bar(conn, src, "600519.SH", "20240102", "20240105")
    assert n == 3                                   # 1/2 实际, 1/3 占位, 1/5 实际（1/4 日历休市）
    rows = conn.execute("SELECT trade_date, close, vol, is_suspended, limit_up, limit_source "
                        "FROM daily_bar WHERE ts_code='600519.SH' ORDER BY trade_date").fetchall()
    assert rows[0] == (D(2024,1,2), 100.5, 1000.0, False, 110.55, "api")
    # 停牌占位：OHLC=前收 100.5、vol=0；API 只给了 1/2 → 1/3 走规则 100.5×1.1
    assert rows[1] == (D(2024,1,3), 100.5, 0.0, True, 110.55, "rule")
    assert rows[2][0] == D(2024,1,5) and rows[2][3] is False
    assert ingest.job_state(conn, "daily_bar:600519.SH:2024-01-02") == "DONE"


def test_ingest_daily_bar_second_batch_seeds_from_first(conn):
    """跨批次：第二批首日停牌 → 用第一批最后一个非停牌日的 close/adj 前推。"""
    src = FakeSource()
    _prime(conn, src)
    ingest.ingest_daily_bar(conn, src, "600519.SH", "20240102", "20240105")
    # 第二批 1/8 起，FakeSource.daily 只返回 1/2 与 1/5 → 1/8 无行 = 停牌
    conn.execute("INSERT INTO calendar VALUES (DATE '2024-01-08', TRUE, DATE '2024-01-05')")
    ingest.ingest_daily_bar(conn, src, "600519.SH", "20240108", "20240108")
    r = conn.execute("SELECT close, adj_factor, is_suspended FROM daily_bar "
                     "WHERE ts_code='600519.SH' AND trade_date=DATE '2024-01-08'").fetchone()
    assert r == (105.5, 10.0, True)


def test_stk_limit_permission_error_falls_back_to_rule_and_is_remembered(conn):
    src = FakeSource()
    src.stk_limit_error = RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108")
    _prime(conn, src)
    ingest.ingest_daily_bar(conn, src, "600519.SH", "20240102", "20240105")
    r = conn.execute("SELECT limit_up, limit_source FROM daily_bar "
                     "WHERE ts_code='600519.SH' AND trade_date=DATE '2024-01-02'").fetchone()
    assert r == (109.45, "rule")                    # pre_close 99.5 × 1.1 = 109.45
    assert getattr(src, "_stk_limit_denied", False) is True, "无权限要记住，后续股票不再浪费调用"


def test_stk_limit_transient_error_is_not_swallowed(conn):
    """限频 / 网络错误必须抛出进 RETRY，不能被当成"无权限"静默降级。"""
    src = FakeSource()
    src.stk_limit_error = RuntimeError("抱歉，您每分钟最多访问该接口500次")
    _prime(conn, src)
    with pytest.raises(RuntimeError, match="每分钟"):
        ingest.ingest_daily_bar(conn, src, "600519.SH", "20240102", "20240105")
    assert ingest.job_state(conn, "daily_bar:600519.SH:2024-01-02") == "RETRY"
    assert conn.execute("SELECT count(*) FROM daily_bar").fetchone()[0] == 0
