"""Task 14：promote（影子文件原子替换）+ pipeline（分批 / clamp / 全量 / 增量）。"""
from __future__ import annotations
import datetime as dt
import os
import pathlib
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, ingest, pipeline, promote, query
from ashare.data.sources.tushare import _to_date

D = dt.date


# ══════════════ promote ══════════════
def _mk_db(path, marker, *, with_calendar_row=False):
    c = _db.connect_write(str(path)); _db.init_schema(c)
    c.execute("INSERT INTO _meta VALUES ('marker', ?)", [marker])
    if with_calendar_row:                              # snapshot_id 是【数据】指纹，_meta 不算数据
        c.execute("INSERT INTO calendar VALUES (DATE '2024-01-02', TRUE, NULL)")
    c.close()


def test_promote_atomically_replaces_and_keeps_backups(tmp_path):
    market = tmp_path / "market.duckdb"
    _mk_db(market, "v1")
    for i in range(2, 6):
        staging = tmp_path / "staging.duckdb"
        _mk_db(staging, f"v{i}")
        promote.promote(str(staging), str(market), keep=3)
        assert not staging.exists()
        c = duckdb.connect(str(market), read_only=True)
        assert c.execute("SELECT value FROM _meta WHERE key='marker'").fetchone()[0] == f"v{i}"
        c.close()
    baks = sorted(p.name for p in tmp_path.glob("market.duckdb.bak.*"))
    assert len(baks) == 3, baks                        # 只留 3 份
    assert not list(tmp_path.glob("*.wal")), "promote 前必须 CHECKPOINT，不得残留 WAL"


def test_promote_refuses_when_staging_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote.promote(str(tmp_path / "nope.duckdb"), str(tmp_path / "market.duckdb"))


def test_query_reconnects_after_promote(tmp_path):
    market = tmp_path / "market.duckdb"
    _mk_db(market, "v1")
    query.open_db(str(market))
    s1 = query.snapshot_id()
    staging = tmp_path / "staging.duckdb"
    _mk_db(staging, "v2", with_calendar_row=True)
    promote.promote(str(staging), str(market))         # 读连接仍开着：os.replace 换 inode，旧连接读旧文件
    try:
        assert query.snapshot_id() != s1               # _conn() 每次 stat 路径 → 发现 inode 变 → 自动重连
    finally:
        query.close_db()


# ══════════════ pipeline helpers ══════════════
def test_yearly_batches():
    b = pipeline.yearly_batches(D(2010, 1, 1), D(2012, 3, 15))
    assert b == [(D(2010,1,1), D(2010,12,31)), (D(2011,1,1), D(2011,12,31)), (D(2012,1,1), D(2012,3,15))]


def test_clamp_end_to_last_open_day_not_after_today(market_db):
    c = _db.connect_read(market_db)
    try:
        assert pipeline.clamp_end(c, today=D(2024, 1, 6)) == D(2024, 1, 5)     # 周六 → 周五
        assert pipeline.clamp_end(c, today=D(2024, 1, 5)) == D(2024, 1, 5)
        assert pipeline.clamp_end(c, today=D(2030, 1, 1)) == D(2024, 2, 2)     # 不超过日历尽头
    finally:
        c.close()


def test_last_bar_date(market_db):
    c = _db.connect_read(market_db)
    try:
        assert pipeline.last_bar_date(c) == D(2024, 2, 2)
    finally:
        c.close()


# ══════════════ run_full / run_daily 端到端（假数据源）══════════════
class FakeSrc:
    """最小可跑全流程的假源：2 只股票、2024-01-02~01-05、一天停牌。"""
    def trade_cal(self, start, end, exchange="SSE"):
        # 按 [start, end] 生成"工作日开市"日历（元旦 01-01 休市），足够覆盖 pipeline 要求的前后余量
        s = start if hasattr(start, "year") else dt.date.fromisoformat(str(start))
        e = end if hasattr(end, "year") else dt.date.fromisoformat(str(end))
        days, d = [], s
        while d <= e:
            days.append(d); d += dt.timedelta(days=1)
        is_open = [int(x.weekday() < 5 and not (x.month == 1 and x.day == 1)) for x in days]
        pre, pres = None, []
        for x, o in zip(days, is_open):
            pres.append(pre.strftime("%Y%m%d") if pre else None)
            if o:
                pre = x
        return _to_date(pd.DataFrame({"exchange": ["SSE"] * len(days),
                                      "cal_date": [x.strftime("%Y%m%d") for x in days],
                                      "is_open": is_open, "pretrade_date": pres}))
    def stock_basic(self):
        return _to_date(pd.DataFrame({"ts_code": ["600519.SH", "000001.SZ"], "symbol": ["600519", "000001"],
                                      "name": ["贵州茅台", "平安银行"], "area": [None] * 2, "industry": ["白酒", "银行"],
                                      "market": ["主板"] * 2, "list_date": ["20010827", "19910403"],
                                      "delist_date": [None, None], "is_hs": ["S", "S"]}))
    def namechange(self, ts_code=None):
        return _to_date(pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date", "change_reason"]))
    def sw_members(self):
        raise RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    def daily(self, ts_code=None, trade_date=None, start=None, end=None):
        days = ["20240102", "20240103", "20240104", "20240105", "20240108"]
        if ts_code == "600519.SH":
            days = [d for d in days if d != "20240104"]        # 停牌一天
        s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        days = [d for d in days if s <= d <= e]
        return _to_date(pd.DataFrame({"ts_code": [ts_code] * len(days), "trade_date": days,
                                      "open": [10.0] * len(days), "high": [10.5] * len(days), "low": [9.5] * len(days),
                                      "close": [10.2] * len(days), "pre_close": [10.0] * len(days),
                                      "vol": [100.0] * len(days), "amount": [1000.0] * len(days)}))
    def adj_factor(self, ts_code=None, trade_date=None, start=None, end=None):
        d = self.daily(ts_code, start=start, end=end)
        return d[["ts_code", "trade_date"]].assign(adj_factor=1.0)
    def stk_limit(self, ts_code=None, trade_date=None, start=None, end=None):
        raise RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    def daily_basic(self, ts_code=None, trade_date=None, start=None, end=None):
        d = trade_date.strftime("%Y%m%d")
        return _to_date(pd.DataFrame({"ts_code": ["600519.SH", "000001.SZ"], "trade_date": [d] * 2,
                                      "pe_ttm": [30.0, 5.0], "total_mv": [2e8, 2e7], "turnover_rate_f": [0.2, 0.6]}))
    def index_daily(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame({"ts_code": [ts_code], "trade_date": ["20240102"], "open": [1.0], "high": [1.0],
                                      "low": [1.0], "close": [1.0], "vol": [1.0], "amount": [1.0]}))
    def index_dailybasic(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame({"ts_code": [ts_code], "trade_date": ["20240102"], "pe_ttm": [12.0]}))
    def income(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame({"ts_code": [ts_code], "ann_date": ["20230330"], "f_ann_date": ["20230330"],
                                      "end_date": ["20221231"], "report_type": ["1"], "update_flag": [0],
                                      "revenue": [1000.0], "n_income_attr_p": [100.0], "basic_eps": [1.0]}))
    def balancesheet(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame(columns=["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "update_flag", "total_assets"]))
    def cashflow(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame(columns=["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "update_flag", "n_cashflow_act"]))
    def fina_indicator(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "update_flag", "roe", "bps"]))
    def hk_hold(self, trade_date):
        return _to_date(pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": [trade_date.strftime("%Y%m%d")], "ratio": [7.0]}))
    def cn_m(self, start_m, end_m):
        return pd.DataFrame({"month": ["202312"], "m1_yoy": [1.3], "m2_yoy": [9.7]})
    def cn_cpi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202312"], "nt_yoy": [-0.3]})
    def cn_ppi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202312"], "ppi_yoy": [-2.7]})
    def cn_pmi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202312"], "pmi010000": [49.0]})
    def sf_month(self, start_m, end_m):
        return pd.DataFrame({"month": ["202212", "202312"], "stk_endval": [344.0, 378.0]})
    def shibor(self, start=None, end=None):
        return _to_date(pd.DataFrame({"date": ["20240102"], "3m": [2.4]}))
    def cn10y(self, start=None, end=None):
        return pd.DataFrame({"period": [D(2024, 1, 2)], "value": [2.56]})


def test_run_full_then_daily_end_to_end(tmp_path):
    staging = tmp_path / "staging.duckdb"
    market = tmp_path / "market.duckdb"
    src = FakeSrc()
    summary = pipeline.run_full(str(staging), src, start=D(2024, 1, 2), end=D(2024, 1, 5),
                                indices=("000985.CSI",))
    assert summary["daily_bar"] == 8                   # 2 只 × 4 交易日（含 1 占位）
    assert summary["validation"] == "passed"
    promote.promote(str(staging), str(market))

    query.open_db(str(market))
    try:
        bars = query.get_bars("2024-01-05", ["600519.SH"], lookback=4)
        assert len(bars) == 4 and int(bars["is_suspended"].sum()) == 1
        assert query.get_universe("2024-01-05", liquidity_drop_pct=0.0) == ["000001.SZ", "600519.SH"]
        assert query.get_macro("2024-01-25", ["m2_yoy"]).loc[D(2023, 12, 31), "m2_yoy"] == 9.7
        assert query.get_financial("2024-01-05", ["600519.SH"], ["revenue"]).loc["600519.SH", "revenue"] == 1000.0
    finally:
        query.close_db()

    # 增量：today=2024-01-08（周一）→ 补 01-08 一天；end clamp 到最后交易日
    staging2 = tmp_path / "staging2.duckdb"
    summary2 = pipeline.run_daily(str(market), str(staging2), src, today=D(2024, 1, 8), indices=("000985.CSI",))
    assert summary2["start"] == D(2024, 1, 8) and summary2["end"] == D(2024, 1, 8)
    assert summary2["daily_bar"] == 2
    promote.promote(str(staging2), str(market))
    query.open_db(str(market))
    try:
        assert query.get_bars("2024-01-08", ["600519.SH"], lookback=1).index[0][1] == D(2024, 1, 8)
    finally:
        query.close_db()


def test_run_daily_noop_when_up_to_date(tmp_path):
    staging = tmp_path / "staging.duckdb"; market = tmp_path / "market.duckdb"
    src = FakeSrc()
    pipeline.run_full(str(staging), src, start=D(2024, 1, 2), end=D(2024, 1, 5), indices=())
    promote.promote(str(staging), str(market))
    s = pipeline.run_daily(str(market), str(tmp_path / "s2.duckdb"), src, today=D(2024, 1, 5), indices=())
    assert s["daily_bar"] == 0 and s["skipped"] is True
