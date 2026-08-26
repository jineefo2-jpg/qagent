"""Task 7：财报 PIT / daily_basic / index_daily / 行业成分历史 的入库语义。"""
from __future__ import annotations
import datetime as dt
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, ingest
from ashare.data.sources.tushare import _to_date

D = dt.date


# ══════════════ merge_financial_frames（纯函数）══════════════
def _inc(rows):
    return _to_date(pd.DataFrame(rows, columns=["ts_code", "ann_date", "f_ann_date", "end_date",
                                                "report_type", "update_flag", "revenue", "n_income_attr_p", "basic_eps"]))

def _bal(rows):
    return _to_date(pd.DataFrame(rows, columns=["ts_code", "ann_date", "f_ann_date", "end_date",
                                                "report_type", "update_flag", "total_assets", "total_hldr_eqy_exc_min_int"]))

def _cf(rows):
    return _to_date(pd.DataFrame(rows, columns=["ts_code", "ann_date", "f_ann_date", "end_date",
                                                "report_type", "update_flag", "n_cashflow_act"]))

def _fina(rows):
    return _to_date(pd.DataFrame(rows, columns=["ts_code", "ann_date", "end_date", "update_flag", "roe", "bps"]))


def test_merge_joins_four_sources_into_one_dense_row():
    out, dropped = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0)]),
        _bal([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 5000.0, 4000.0)]),
        _cf([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 450.0)]),
        _fina([("600519.SH", "20210330", "20201231", 0, 0.28, 120.0)]))
    assert dropped == 0 and len(out) == 1
    r = out.iloc[0]
    assert (r.ts_code, r.ann_date, r.end_date, r.report_type, r.update_flag) == ("600519.SH", D(2021,3,30), D(2020,12,31), "1", 0)
    assert (r.revenue, r.total_assets, r.n_cashflow_act, r.roe, r.bps, r.basic_eps) == (1000.0, 5000.0, 450.0, 0.28, 120.0, 30.0)


def test_f_ann_date_overrides_ann_date():
    """Tushare 的 ann_date 有时是预约披露日；f_ann_date 才是实际公告日，PIT 键必须用它。"""
    out, _ = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210325", "20210330", "20201231", "1", 0, 1.0, 1.0, 1.0)]),
        _bal([]), _cf([]), _fina([]))
    assert out.iloc[0].ann_date == D(2021,3,30)


def test_ann_date_is_max_across_sources_conservative():
    """三张表公告日不一致时取最晚——数据要等全部披露才算可见（PIT 保守方向）。"""
    out, _ = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 1.0, 1.0, 1.0)]),
        _bal([("600519.SH", "20210330", "20210402", "20201231", "1", 0, 1.0, 1.0)]),
        _cf([]), _fina([]))
    assert len(out) == 1 and out.iloc[0].ann_date == D(2021,4,2)


def test_rows_without_ann_date_are_dropped_and_counted():
    out, dropped = ingest.merge_financial_frames(
        _inc([("600519.SH", None, None, "20201231", "1", 0, 1.0, 1.0, 1.0),
              ("600519.SH", "20210330", "20210330", "20201231", "1", 0, 2.0, 2.0, 2.0)]),
        _bal([]), _cf([]), _fina([]))
    assert dropped == 1 and len(out) == 1 and out.iloc[0].revenue == 2.0


def test_restatement_is_separate_row_not_overwrite():
    """update_flag=1（重述）与原始披露并存，各有各的 ann_date（D3）。"""
    out, _ = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0),
              ("600519.SH", "20220330", "20220330", "20201231", "1", 1, 1010.0, 405.0, 30.5)]),
        _bal([]), _cf([]), _fina([]))
    assert len(out) == 2
    assert sorted(out.update_flag) == [0, 1]
    assert out[out.update_flag == 1].iloc[0].ann_date == D(2022,3,30)


def test_same_period_two_announcements_both_kept():
    """同一报告期同一 update_flag 两次公告（如更正公告）→ 两行共存，PIT 由 query 层按 as_of 取。"""
    out, _ = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0),
              ("600519.SH", "20210415", "20210415", "20201231", "1", 0, 1001.0, 400.0, 30.0)]),
        _bal([]), _cf([]), _fina([]))
    assert len(out) == 2 and sorted(out.ann_date) == [D(2021,3,30), D(2021,4,15)]


def test_multi_announcement_across_sources_does_not_cross_product():
    """inc 与 bal 各有两次公告 → 不是 2×2=4 行；每个 ann_date 一行，且每源取该时点的最新版本。"""
    out, _ = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0),
              ("600519.SH", "20210415", "20210415", "20201231", "1", 0, 1001.0, 400.0, 30.0)]),
        _bal([("600519.SH", "20210330", "20210330", "20201231", "1", 0, 5000.0, 4000.0),
              ("600519.SH", "20210415", "20210415", "20201231", "1", 0, 5001.0, 4000.0)]),
        _cf([]), _fina([]))
    assert len(out) == 2
    r = out.set_index("ann_date")
    assert (r.loc[D(2021,3,30), "revenue"], r.loc[D(2021,3,30), "total_assets"]) == (1000.0, 5000.0)
    assert (r.loc[D(2021,4,15), "revenue"], r.loc[D(2021,4,15), "total_assets"]) == (1001.0, 5001.0)


def test_rows_without_end_date_are_dropped_and_counted():
    out, dropped = ingest.merge_financial_frames(
        _inc([("600519.SH", "20210330", "20210330", None, "1", 0, 1.0, 1.0, 1.0)]),
        _bal([]), _cf([]), _fina([]))
    assert dropped == 1 and len(out) == 0


# ══════════════ 入库（真实 DuckDB）══════════════
class FakeSrc:
    perm_error: Exception | None = None

    def income(self, ts_code, start=None, end=None):
        return _inc([(ts_code, "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0),
                     (ts_code, "20220330", "20220330", "20201231", "1", 1, 1010.0, 405.0, 30.5)])
    def balancesheet(self, ts_code, start=None, end=None):
        return _bal([(ts_code, "20210330", "20210330", "20201231", "1", 0, 5000.0, 4000.0)])
    def cashflow(self, ts_code, start=None, end=None):
        return _cf([(ts_code, "20210330", "20210330", "20201231", "1", 0, 450.0)])
    def fina_indicator(self, ts_code, start=None, end=None):
        return _fina([(ts_code, "20210330", "20201231", 0, 0.28, 120.0)])

    def daily_basic(self, ts_code=None, trade_date=None, start=None, end=None):
        return _to_date(pd.DataFrame({
            "ts_code": ["600519.SH", "000001.SZ"], "trade_date": ["20240102"] * 2,
            "turnover_rate": [0.1, 0.5], "turnover_rate_f": [0.2, 0.6], "volume_ratio": [1.0, 1.1],
            "pe": [30.0, 5.0], "pe_ttm": [31.0, 5.1], "pb": [8.0, 0.6], "ps": [10.0, 1.0], "ps_ttm": [10.1, 1.1],
            "dv_ratio": [1.5, 4.0], "dv_ttm": [1.6, 4.1],
            "total_share": [125.6, 1940.0], "float_share": [125.6, 1940.0], "free_share": [80.0, 1900.0],
            "total_mv": [2.1e8, 2.0e7], "circ_mv": [2.1e8, 2.0e7]}))

    def index_daily(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame({"ts_code": [ts_code] * 2, "trade_date": ["20240102", "20240103"],
                                      "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
                                      "close": [1.5, 2.5], "vol": [1.0, 1.0], "amount": [1.0, 1.0]}))
    def index_dailybasic(self, ts_code, start=None, end=None):
        return _to_date(pd.DataFrame({"ts_code": [ts_code], "trade_date": ["20240102"], "pe_ttm": [12.3]}))

    def stock_basic(self):
        return _to_date(pd.DataFrame({"ts_code": ["600519.SH", "000001.SZ"], "symbol": ["600519", "000001"],
                                      "name": ["贵州茅台", "平安银行"], "area": [None] * 2,
                                      "industry": ["白酒", "银行"], "market": ["主板"] * 2,
                                      "list_date": ["20010827", "19910403"], "delist_date": [None, None],
                                      "is_hs": ["S", "S"]}))
    def sw_members(self):
        if self.perm_error is not None:
            raise self.perm_error
        return _to_date(pd.DataFrame({"ts_code": ["600519.SH", "600519.SH"],
                                      "sw_l1": ["食品饮料", "食品饮料"], "sw_l2": ["白酒Ⅱ", "饮料乳品"], "sw_l3": ["白酒Ⅲ", "软饮料"],
                                      "in_date": ["20140101", "20210701"], "out_date": ["20210630", None]}))


@pytest.fixture
def conn(tmp_db):
    c = _db.connect_write(tmp_db)
    _db.init_schema(c)
    yield c
    c.close()


def test_schema_v2_has_industry_member_and_version_guard(conn):
    tabs = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "industry_member" in tabs
    assert conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()[0] == str(_db.SCHEMA_VERSION)
    # 旧代码打开新库必须拒绝
    conn.execute("UPDATE _meta SET value = ? WHERE key='schema_version'", [str(_db.SCHEMA_VERSION + 1)])
    with pytest.raises(RuntimeError, match="schema_version"):
        _db.init_schema(conn)


def test_ingest_financial_end_to_end(conn):
    n = ingest.ingest_financial(conn, FakeSrc(), "600519.SH", "20200101", "20221231")
    assert n == 2
    rows = conn.execute("SELECT ann_date, update_flag, revenue, total_assets, roe FROM financial_pit "
                        "WHERE ts_code='600519.SH' ORDER BY ann_date").fetchall()
    assert rows[0] == (D(2021,3,30), 0, 1000.0, 5000.0, 0.28)
    assert rows[1][:3] == (D(2022,3,30), 1, 1010.0) and rows[1][3] is None      # 重述行只有利润表字段
    assert ingest.job_state(conn, "financial_pit:600519.SH:2020-01-01") == "DONE"
    err = conn.execute("SELECT last_error FROM ingest_log WHERE job_id='financial_pit:600519.SH:2020-01-01'").fetchone()[0]
    assert err == ""                                 # 无丢弃行时不写 dropped 备注


def test_future_ann_date_rows_are_dropped_at_ingest(conn):
    """预约披露守卫：拉取时 ann_date 在未来的行是【尚未公布】的报表（2026-08-25 实测：
    12 只中报提前一天带出 ann_date=次日），入库即前视（D3），validate 的
    financial_ann_date 阻断级会拦。ingest 当场丢弃并在任务备注计数。"""
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d")

    class Src(FakeSrc):
        def income(self, ts_code, start=None, end=None):
            return _inc([(ts_code, "20210330", "20210330", "20201231", "1", 0, 1000.0, 400.0, 30.0),
                         (ts_code, tomorrow, tomorrow, "20260630", "1", 0, 2000.0, 800.0, 60.0)])

    ingest.ingest_financial(conn, Src(), "600519.SH", "20200101", "20991231")
    ends = {r[0] for r in conn.execute(
        "SELECT end_date FROM financial_pit WHERE ts_code='600519.SH'").fetchall()}
    assert D(2026, 6, 30) not in ends                # 未来公告的那行没进库
    assert D(2020, 12, 31) in ends                   # 正常行不受牵连
    err = conn.execute("SELECT last_error FROM ingest_log "
                       "WHERE job_id='financial_pit:600519.SH:2020-01-01'").fetchone()[0]
    assert "dropped_future_ann_date=1" in err


def test_ingest_daily_basic_is_per_day_and_skips_done(conn):
    """DONE 的日期第二次调用必须【跳过】而不是重拉 —— 与 ingest_daily_bar_by_date 同一契约。
    2026-08-25 实测教训：没有这道守卫，full 续跑会把 2010 年以来每一天重新调一遍 API，
    3000+ 次「无声重拉」在终端上表现为卡死，还平白多烧一遍限频配额。"""
    assert ingest.ingest_daily_basic(conn, FakeSrc(), "20240102") == 2
    assert ingest.ingest_daily_basic(conn, FakeSrc(), "20240102") == 0   # DONE → 跳过，不触 API
    assert conn.execute("SELECT count(*) FROM daily_basic").fetchone()[0] == 2
    r = conn.execute("SELECT pe_ttm, total_mv FROM daily_basic WHERE ts_code='600519.SH'").fetchone()
    assert r == (31.0, 2.1e8)


def test_ingest_index_daily_merges_pe_ttm(conn):
    assert ingest.ingest_index_daily(conn, FakeSrc(), "000985.CSI", "20240101", "20240131") == 2
    rows = conn.execute("SELECT trade_date, close, pe_ttm FROM index_daily ORDER BY trade_date").fetchall()
    assert rows == [(D(2024,1,2), 1.5, 12.3), (D(2024,1,3), 2.5, None)]


def test_ingest_industry_member_pit_rows(conn):
    src = FakeSrc()
    ingest.ingest_stock_basic(conn, src)
    assert ingest.ingest_industry_member(conn, src) == 2
    rows = conn.execute("SELECT sw_l2, in_date, out_date FROM industry_member WHERE ts_code='600519.SH' "
                        "ORDER BY in_date").fetchall()
    assert rows == [("白酒Ⅱ", D(2014,1,1), D(2021,6,30)), ("饮料乳品", D(2021,7,1), None)]
    assert conn.execute("SELECT value FROM _meta WHERE key='industry_source'").fetchone()[0] == "sw"


def test_ingest_industry_member_degrades_explicitly_without_permission(conn):
    """无申万成分权限 → 用 stock_basic.industry 降级，且必须显式记录，不得静默。"""
    src = FakeSrc()
    src.perm_error = RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    ingest.ingest_stock_basic(conn, src)
    n = ingest.ingest_industry_member(conn, src)
    assert n == 2
    rows = conn.execute("SELECT ts_code, sw_l1, in_date, out_date FROM industry_member ORDER BY ts_code").fetchall()
    assert rows == [("000001.SZ", "银行", D(1991,4,3), None), ("600519.SH", "白酒", D(2001,8,27), None)]
    assert conn.execute("SELECT value FROM _meta WHERE key='industry_source'").fetchone()[0] == "tushare_static"


def test_ingest_industry_member_requires_stock_basic(conn):
    src = FakeSrc()
    src.perm_error = RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    with pytest.raises(RuntimeError, match="stock_basic"):
        ingest.ingest_industry_member(conn, src)


def test_ingest_industry_member_is_full_refresh(conn):
    """先降级（static）再拿到 sw 权限重跑 → 表里只剩 sw 行，不混。"""
    src = FakeSrc()
    ingest.ingest_stock_basic(conn, src)
    src.perm_error = RuntimeError("抱歉，您没有访问该接口的权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    ingest.ingest_industry_member(conn, src)
    src.perm_error = None
    ingest.ingest_industry_member(conn, src)
    rows = conn.execute("SELECT count(*), max(sw_l2 IS NOT NULL) FROM industry_member").fetchone()
    assert rows == (2, True)
    assert conn.execute("SELECT value FROM _meta WHERE key='industry_source'").fetchone()[0] == "sw"


def test_ingest_industry_member_transient_error_propagates(conn):
    src = FakeSrc()
    src.perm_error = RuntimeError("抱歉，您每分钟最多访问该接口500次，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    ingest.ingest_stock_basic(conn, src)
    with pytest.raises(RuntimeError, match="每分钟"):
        ingest.ingest_industry_member(conn, src)


def test_industry_downgrade_refuses_to_overwrite_real_sw_history(conn):
    """已有真实申万历史时，积分不足/无权限不得降级 —— 降级会用今天的行业回填到上市日。"""
    src = FakeSrc()
    ingest.ingest_stock_basic(conn, src)
    ingest.ingest_industry_member(conn, src)                     # source='sw'
    src.perm_error = RuntimeError("抱歉，您积分不足，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。")
    with pytest.raises(RuntimeError, match="前视污染"):
        ingest.ingest_industry_member(conn, src)
    assert conn.execute("SELECT value FROM _meta WHERE key='industry_source'").fetchone()[0] == "sw"
    assert conn.execute("SELECT count(*) FROM industry_member").fetchone()[0] == 2   # 历史没被删
