"""Task 8：宏观 PIT（D4）与北向持股入库。"""
from __future__ import annotations
import datetime as dt
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, ingest
from ashare.data.ingest import rule_publish_date, month_end
from ashare.data.sources.tushare import _to_date

D = dt.date


# ══════════════ 纯函数 ══════════════
def test_month_end():
    assert month_end("202407") == D(2024,7,31)
    assert month_end("202402") == D(2024,2,29)
    assert month_end("202312") == D(2023,12,31)


def test_rule_publish_dates_are_conservative_late():
    """历史 publish_date 只能按规则回填，规则一律取保守晚值：宁可晚几天可见，绝不提前（D4）。"""
    p = D(2024,7,31)
    assert rule_publish_date("m2_yoy", p) == D(2024,8,15)
    assert rule_publish_date("m1_yoy", p) == D(2024,8,15)
    assert rule_publish_date("tsf_stock_yoy", p) == D(2024,8,15)
    assert rule_publish_date("cpi_yoy", p) == D(2024,8,10)
    assert rule_publish_date("ppi_yoy", p) == D(2024,8,10)
    assert rule_publish_date("pmi_mfg", p) == D(2024,8,1)
    assert rule_publish_date("shibor_3m", D(2024,7,15)) == D(2024,7,15)
    assert rule_publish_date("cn10y", D(2024,7,15)) == D(2024,7,15)


def test_rule_publish_date_december_rolls_year():
    assert rule_publish_date("m2_yoy", D(2023,12,31)) == D(2024,1,15)
    assert rule_publish_date("pmi_mfg", D(2023,12,31)) == D(2024,1,1)


def test_unknown_indicator_raises():
    with pytest.raises(KeyError):
        rule_publish_date("gdp_yoy", D(2024,7,31))


# ══════════════ 入库（真实 DuckDB）══════════════
class FakeSrc:
    def cn_m(self, start_m, end_m):
        return pd.DataFrame({"month": ["202406", "202407"], "m1_yoy": [-5.0, -6.6], "m2_yoy": [6.2, 6.3]})
    def cn_cpi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202406", "202407"], "nt_yoy": [0.2, 0.5]})
    def cn_ppi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202407"], "ppi_yoy": [-0.8]})
    def cn_pmi(self, start_m, end_m):
        return pd.DataFrame({"month": ["202407"], "pmi010000": [49.4]})
    def sf_month(self, start_m, end_m):
        # 13 个月存量：2023-07 → 2024-07，用于算 yoy
        months = [f"2023{m:02d}" for m in range(7, 13)] + [f"2024{m:02d}" for m in range(1, 8)]
        stk = [365.0 + i for i in range(13)]        # 2023-07=365, 2024-07=377
        return pd.DataFrame({"month": months, "stk_endval": stk})
    def shibor(self, start=None, end=None):
        return _to_date(pd.DataFrame({"date": ["20240715", "20240716"], "3m": [1.85, 1.86]}))
    def cn10y(self, start=None, end=None):
        return pd.DataFrame({"period": [D(2024,7,15)], "value": [2.26]})
    def hk_hold(self, trade_date):
        return _to_date(pd.DataFrame({"code": ["600519", "000001"], "trade_date": ["20240102"] * 2,
                                      "ts_code": ["600519.SH", "000001.SZ"], "name": ["贵州茅台", "平安银行"],
                                      "vol": [1e6, 2e6], "ratio": [7.12, 8.5], "exchange": ["SH", "SZ"]}))


@pytest.fixture
def conn(tmp_db):
    c = _db.connect_write(tmp_db)
    _db.init_schema(c)
    yield c
    c.close()


def test_ingest_m2_writes_pit_rows(conn):
    n = ingest.ingest_macro(conn, FakeSrc(), "m2_yoy", "20240601", "20240731")
    assert n == 2
    rows = conn.execute("SELECT period, publish_date, value, publish_date_source FROM macro_indicator "
                        "WHERE indicator='m2_yoy' ORDER BY period").fetchall()
    assert rows == [(D(2024,6,30), D(2024,7,15), 6.2, "rule"), (D(2024,7,31), D(2024,8,15), 6.3, "rule")]


def test_every_macro_row_publishes_no_earlier_than_period(conn):
    src = FakeSrc()
    for ind in ("m1_yoy", "m2_yoy", "cpi_yoy", "ppi_yoy", "pmi_mfg", "tsf_stock_yoy", "shibor_3m", "cn10y"):
        ingest.ingest_macro(conn, src, ind, "20240601", "20240731")
    bad = conn.execute("SELECT count(*) FROM macro_indicator WHERE publish_date < period").fetchone()[0]
    assert bad == 0
    nulls = conn.execute("SELECT count(*) FROM macro_indicator WHERE publish_date IS NULL "
                         "OR publish_date_source IS NULL").fetchone()[0]
    assert nulls == 0


def test_tsf_yoy_computed_from_13_month_stock(conn):
    n = ingest.ingest_macro(conn, FakeSrc(), "tsf_stock_yoy", "20240701", "20240731")
    assert n == 1                                    # 只写 [start, end] 内的月份
    r = conn.execute("SELECT period, value FROM macro_indicator WHERE indicator='tsf_stock_yoy'").fetchone()
    assert r[0] == D(2024,7,31) and abs(r[1] - (377.0 / 365.0 - 1) * 100) < 1e-9


def test_pmi_and_daily_indicators(conn):
    src = FakeSrc()
    ingest.ingest_macro(conn, src, "pmi_mfg", "20240701", "20240731")
    ingest.ingest_macro(conn, src, "shibor_3m", "20240715", "20240716")
    ingest.ingest_macro(conn, src, "cn10y", "20240715", "20240715")
    rows = dict(conn.execute("SELECT indicator, count(*) FROM macro_indicator GROUP BY indicator").fetchall())
    assert rows == {"pmi_mfg": 1, "shibor_3m": 2, "cn10y": 1}
    r = conn.execute("SELECT period, publish_date FROM macro_indicator WHERE indicator='shibor_3m' "
                     "ORDER BY period").fetchall()
    assert r == [(D(2024,7,15), D(2024,7,15)), (D(2024,7,16), D(2024,7,16))]


def test_observed_publish_date_coexists_with_rule(conn):
    """每日增量：某 (indicator, period) 在 observed_on 当天已能拿到、但库里尚无 publish_date <= observed_on 的行
    → 写一条 (observed_on, 'observed')。它与历史 rule 行并存（PIT：不覆盖）；已可见的 period 不重复写。"""
    ingest.ingest_macro(conn, FakeSrc(), "m2_yoy", "20240601", "20240731")
    n = ingest.ingest_macro(conn, FakeSrc(), "m2_yoy", "20240601", "20240731",
                            observed_on=D(2024,8,12))
    assert n == 1                                    # 2024-06 的 rule 行 07-15 已 <= 08-12 → 跳过；2024-07 的 rule 是 08-15 → 写 observed
    rows = conn.execute("SELECT period, publish_date, publish_date_source FROM macro_indicator "
                        "WHERE indicator='m2_yoy' AND period = DATE '2024-07-31' ORDER BY publish_date").fetchall()
    assert rows == [(D(2024,7,31), D(2024,8,12), "observed"), (D(2024,7,31), D(2024,8,15), "rule")]


def test_ingest_hk_hold(conn):
    n = ingest.ingest_hk_hold(conn, FakeSrc(), "20240102")
    assert n == 2
    r = conn.execute("SELECT hk_hold_ratio FROM money_flow WHERE ts_code='600519.SH'").fetchone()
    assert r == (7.12,)
    assert ingest.job_state(conn, "money_flow:2024-01-02") == "DONE"
