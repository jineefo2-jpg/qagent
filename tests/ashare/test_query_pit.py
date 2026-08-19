"""Task 12：get_financial / get_financial_ttm / get_macro / get_money_flow（D3 / D4 / B5）。

财报数据（A00001.SZ，累计口径，单位随意）：
  FY2022  end 2022-12-31 ann 2023-03-30  revenue 1000  n_income_attr_p 100  total_assets 5000
  Q1 2023 end 2023-03-31 ann 2023-04-25  revenue  300  n_income_attr_p  30  total_assets 5100
  H1 2023 end 2023-06-30 ann 2023-08-20  revenue  620  n_income_attr_p  62  total_assets 5200
  Q3 2023 end 2023-09-30 ann 2023-10-25  revenue  950  n_income_attr_p  95  total_assets 5300
  FY2023  end 2023-12-31 ann 2024-01-20  revenue 1300  n_income_attr_p 130  total_assets 5400   （首次公告）
  FY2023  end 2023-12-31 ann 2024-01-26  revenue 1310  n_income_attr_p 131  total_assets 5410   （更正公告，update_flag=0）
  FY2022  end 2022-12-31 ann 2024-01-22  revenue 1005  ... update_flag=1（重述）
  Q1 2022 end 2022-03-31 ann 2022-04-25  revenue  280  n_income_attr_p  28  total_assets 4900
"""
from __future__ import annotations
import datetime as dt
import math
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, query
from ashare.data.query import UnknownFieldError

D = dt.date


def _extend_calendar(w):
    """财报/宏观测试要用到 2023-05 / 2023-11 / 2024-02-20 等 as_of；把日历按工作日开市延到 2022-01-01~2024-02-29。
    与 market_db 既有的真实片段（2023-12-25~2024-02-02）不重叠。"""
    have = {r[0] for r in w.execute("SELECT trade_date FROM calendar").fetchall()}
    rows = []
    d = dt.date(2022, 1, 1)
    while d <= dt.date(2024, 2, 29):
        if d not in have:
            rows.append((d, d.weekday() < 5, None))
        d += dt.timedelta(days=1)
    w.executemany("INSERT INTO calendar VALUES (?, ?, ?)", rows)


@pytest.fixture
def q(market_db):
    w = _db.connect_write(market_db)
    _extend_calendar(w)
    fin = [
        # ts_code, ann_date, end_date, report_type, update_flag, revenue, n_income_attr_p, total_assets, roe
        ("A00001.SZ", D(2022,4,25), D(2022,3,31), "1", 0, 280.0, 28.0, 4900.0, 0.05),
        ("A00001.SZ", D(2023,3,30), D(2022,12,31), "1", 0, 1000.0, 100.0, 5000.0, 0.10),
        ("A00001.SZ", D(2023,4,25), D(2023,3,31), "1", 0, 300.0, 30.0, 5100.0, 0.06),
        ("A00001.SZ", D(2023,8,20), D(2023,6,30), "1", 0, 620.0, 62.0, 5200.0, 0.07),
        ("A00001.SZ", D(2023,10,25), D(2023,9,30), "1", 0, 950.0, 95.0, 5300.0, 0.08),
        ("A00001.SZ", D(2024,1,20), D(2023,12,31), "1", 0, 1300.0, 130.0, 5400.0, 0.11),
        ("A00001.SZ", D(2024,1,26), D(2023,12,31), "1", 0, 1310.0, 131.0, 5410.0, 0.11),
        ("A00001.SZ", D(2024,1,22), D(2022,12,31), "1", 1, 1005.0, 100.5, 5005.0, 0.10),
        # B 只有一期
        ("B00002.SZ", D(2023,3,30), D(2022,12,31), "1", 0, 50.0, 5.0, 200.0, 0.02),
    ]
    w.executemany("INSERT INTO financial_pit (ts_code, ann_date, end_date, report_type, update_flag, "
                  "revenue, n_income_attr_p, total_assets, roe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", fin)
    macro = [
        ("m2_yoy", D(2023,11,30), D(2023,12,20), 10.0, "rule"),
        ("m2_yoy", D(2023,12,31), D(2024,1,20), 9.7, "rule"),
        ("m2_yoy", D(2023,12,31), D(2024,1,12), 9.7, "observed"),
        ("m2_yoy", D(2024,1,31), D(2024,2,20), 8.7, "rule"),
        ("cpi_yoy", D(2023,12,31), D(2024,1,16), -0.3, "rule"),
    ]
    w.executemany("INSERT INTO macro_indicator (indicator, period, publish_date, value, publish_date_source) "
                  "VALUES (?, ?, ?, ?, ?)", macro)
    w.executemany("INSERT INTO money_flow (ts_code, trade_date, hk_hold_ratio) VALUES (?, ?, ?)",
                  [("A00001.SZ", D(2024,1,4), 5.0), ("A00001.SZ", D(2024,1,5), 5.2)])
    w.close()
    query.open_db(market_db)
    yield query
    query.close_db()


# ══════════════ get_financial ══════════════
def test_spec_assertion_pit_takes_latest_annouced_period(q):
    """规格 §11：as_of=2021-04-01 型断言 —— 这里 as_of 2024-01-21：FY2023 已于 01-20 公告 → end_date=2023-12-31。"""
    f = q.get_financial("2024-01-21", ["A00001.SZ"], ["revenue"])
    assert f.loc["A00001.SZ", "end_date"] == D(2023,12,31)
    assert f.loc["A00001.SZ", "ann_date"] == D(2024,1,20) <= D(2024,1,21)
    assert f.loc["A00001.SZ", "revenue"] == 1300.0
    assert f.loc["A00001.SZ", "lag_days"] == 1


def test_before_announcement_uses_previous_period(q):
    f = q.get_financial("2024-01-19", ["A00001.SZ"], ["revenue"])
    assert f.loc["A00001.SZ", "end_date"] == D(2023,9,30) and f.loc["A00001.SZ", "revenue"] == 950.0


def test_two_announcements_same_period_pick_by_as_of(q):
    """同一报告期两次公告：as_of 在两次之间取第一次值，之后取第二次。"""
    assert q.get_financial("2024-01-25", ["A00001.SZ"], ["revenue"]).loc["A00001.SZ", "revenue"] == 1300.0
    assert q.get_financial("2024-01-26", ["A00001.SZ"], ["revenue"]).loc["A00001.SZ", "revenue"] == 1310.0


def test_restated_rows_invisible_by_default(q):
    """update_flag=1 重述行默认不可见（D3）；include_restated=True 时对该报告期可见。"""
    f = q.get_financial("2024-01-31", ["A00001.SZ"], ["revenue"], n_periods=5)
    fy22 = f.xs(D(2022,12,31), level="end_date").loc["A00001.SZ"]
    assert fy22["revenue"] == 1000.0
    f2 = q.get_financial("2024-01-31", ["A00001.SZ"], ["revenue"], n_periods=5, include_restated=True)
    assert f2.xs(D(2022,12,31), level="end_date").loc["A00001.SZ", "revenue"] == 1005.0


def test_n_periods_multiindex_desc(q):
    f = q.get_financial("2024-01-31", ["A00001.SZ"], ["revenue"], n_periods=3)
    ends = list(f.loc["A00001.SZ"].index)
    assert ends == [D(2023,12,31), D(2023,9,30), D(2023,6,30)]


def test_missing_stock_is_absent_not_error(q):
    f = q.get_financial("2024-01-31", ["A00001.SZ", "ZZZZZZ.SZ"], ["revenue"])
    assert list(f.index) == ["A00001.SZ"]


def test_unknown_field_raises(q):
    with pytest.raises(UnknownFieldError):
        q.get_financial("2024-01-31", ["A00001.SZ"], ["nonexistent"])


# ══════════════ get_financial_ttm ══════════════
def test_ttm_at_year_end_equals_fy(q):
    """最新是年报 → TTM = 年报值。as_of 01-25 见首次公告 1300。"""
    s = q.get_financial_ttm("2024-01-25", ["A00001.SZ"], "revenue")
    assert s["A00001.SZ"] == 1300.0


def test_ttm_at_q3_uses_prev_fy_minus_prev_q3(q):
    """as_of 2023-11-01：最新 Q3 2023(950) + FY2022(1000) − Q3 2022(缺) → NaN，不外推。"""
    s = q.get_financial_ttm("2023-11-01", ["A00001.SZ"], "revenue")
    assert math.isnan(s["A00001.SZ"])


def test_ttm_at_q1_cross_year_reset(q):
    """as_of 2023-05-01：最新 Q1 2023(300) + FY2022(1000) − Q1 2022(280) = 1020。跨年重置。"""
    s = q.get_financial_ttm("2023-05-01", ["A00001.SZ"], "revenue")
    assert s["A00001.SZ"] == 1020.0


def test_ttm_stock_field_uses_average(q):
    """存量科目：期初期末均值 =（最新 + 去年同期）/2。as_of 2023-05-01：(5100 + 4900)/2 = 5000。"""
    s = q.get_financial_ttm("2023-05-01", ["A00001.SZ"], "total_assets")
    assert s["A00001.SZ"] == 5000.0


def test_ttm_ratio_field_rejected(q):
    with pytest.raises(UnknownFieldError):
        q.get_financial_ttm("2024-01-25", ["A00001.SZ"], "roe")


def test_ttm_missing_stock_is_nan(q):
    s = q.get_financial_ttm("2024-01-25", ["A00001.SZ", "ZZZZZZ.SZ"], "revenue")
    assert list(s.index) == ["A00001.SZ", "ZZZZZZ.SZ"] and math.isnan(s["ZZZZZZ.SZ"])


# ══════════════ get_macro（D4）══════════════
def test_macro_not_visible_before_publish_date(q):
    m = q.get_macro("2024-02-01", ["m2_yoy"])
    assert D(2024,1,31) not in m.index                 # 2024-01 的 publish 02-20
    assert m.loc[D(2023,12,31), "m2_yoy"] == 9.7


def test_macro_visible_on_publish_date(q):
    m = q.get_macro("2024-02-20", ["m2_yoy"])
    assert m.loc[D(2024,1,31), "m2_yoy"] == 8.7


def test_macro_observed_row_makes_period_visible_earlier(q):
    """2023-12 的 rule 是 01-20，observed 是 01-12 → 01-13 已可见（取 publish_date<=as_of 的最新一行）。"""
    m = q.get_macro("2024-01-13", ["m2_yoy"])
    assert m.loc[D(2023,12,31), "m2_yoy"] == 9.7
    assert m.loc[D(2023,12,31), "m2_yoy__publish_date"] == D(2024,1,12)
    m2 = q.get_macro("2024-01-25", ["m2_yoy"])
    assert m2.loc[D(2023,12,31), "m2_yoy__publish_date"] == D(2024,1,20)


def test_macro_multi_indicator_and_lookback(q):
    m = q.get_macro("2024-02-20", ["m2_yoy", "cpi_yoy"], lookback_periods=2)
    assert list(m.columns) == ["m2_yoy", "m2_yoy__publish_date", "cpi_yoy", "cpi_yoy__publish_date"]
    assert list(m.index) == [D(2023,12,31), D(2024,1,31)]
    assert math.isnan(m.loc[D(2024,1,31), "cpi_yoy"])


# ══════════════ get_money_flow（B5）══════════════
def test_money_flow_nan_before_data_not_zero(q):
    mf = q.get_money_flow("2024-01-05", ["A00001.SZ"], lookback=3)
    assert len(mf) == 3
    assert math.isnan(mf.loc[("A00001.SZ", D(2024,1,3)), "hk_hold_ratio"])
    assert mf.loc[("A00001.SZ", D(2024,1,5)), "hk_hold_ratio"] == 5.2


def test_financial_and_macro_reject_out_of_calendar_as_of(q):
    """Q2：唯一出口行为一致 —— 财报/宏观查询对越界 as_of 同样抛 AsOfDateError，不静默返回最新数据。"""
    from ashare.data.query import AsOfDateError
    with pytest.raises(AsOfDateError):
        q.get_financial("2030-01-01", ["A00001.SZ"], ["revenue"])
    with pytest.raises(AsOfDateError):
        q.get_financial_ttm("2030-01-01", ["A00001.SZ"], "revenue")
    with pytest.raises(AsOfDateError):
        q.get_macro("2030-01-01", ["m2_yoy"])


def test_visible_restated_prev_fy_stays_out_of_ttm(q):
    """重述行（FY2022 update_flag=1，ann 2024-01-22）在 as_of 2024-01-31 已"可见"，但 TTM 必须仍用原始披露 1000。
    as_of 2024-01-31 最新为 FY2023（年报）→ TTM = 1310（更正公告值）；此处用 n_periods 视图核对 FY2022 值仍为 1000。"""
    f = q.get_financial("2024-01-31", ["A00001.SZ"], ["revenue"], n_periods=5)
    assert f.xs(D(2022,12,31), level="end_date").loc["A00001.SZ", "revenue"] == 1000.0
    assert q.get_financial_ttm("2024-01-31", ["A00001.SZ"], "revenue")["A00001.SZ"] == 1310.0


def test_get_financial_multiindex_keeps_end_date_column(q):
    f = q.get_financial("2024-01-31", ["A00001.SZ"], ["revenue"], n_periods=2)
    assert list(f.index.names) == ["ts_code", "end_date"] and "end_date" in f.columns
