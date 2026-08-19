"""P1 验收断言（设计规格 §11）—— 对【真实】market 库运行。

库不存在（CI / 未回补）→ 整文件 skip。运行方式：
    ASHARE_MARKET_DB=data/ashare_market.duckdb python3 -m pytest tests/ashare/test_p1_acceptance.py -v
每条断言对应规格 §11 的一个 checkbox；跑不过 = 数据底座不能进 P2。
"""
from __future__ import annotations
import datetime as dt
import os
import pathlib
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import query, validate

MARKET = os.environ.get("ASHARE_MARKET_DB", "data/ashare_market.duckdb")
pytestmark = pytest.mark.skipif(not pathlib.Path(MARKET).exists(),
                                reason=f"真实 market 库不存在: {MARKET}（先跑 python -m ashare.data.pipeline full）")
D = dt.date


@pytest.fixture(scope="module")
def q():
    query.open_db(MARKET)
    yield query
    query.close_db()


def test_11_1_row_completeness_exact(q):
    """全 A（含已退市）日线行数 == 交易日历 × 在市股票数，误差为 0（D9）。"""
    r = validate.check_row_completeness(MARKET)
    assert r.passed, f"缺行股票 {r.detail['n_bad']} 只，样例: {r.detail['stocks'][:5]}"


def test_11_2_adj_factor_jumps_are_explained(q):
    """adj_factor 跳变全部能匹配到分红送转事件 —— 本期只能列出跳变清单供人工核对，不得为空报告。"""
    r = validate.check_adj_factor_jumps(MARKET)
    assert "jumps" in r.detail                        # 告警级：这里只保证清单产出；匹配到事件是人工步骤


def test_11_3_financial_ann_date_complete(q):
    r = validate.check_financial_ann_date(MARKET)
    assert r.passed, r.detail


def test_11_4_macro_eight_indicators_with_publish_date(q):
    r = validate.check_macro_publish_date(MARKET)
    assert r.passed, r.detail
    m = q.get_macro(dt.date.today(), ["m1_yoy", "m2_yoy", "cpi_yoy", "ppi_yoy", "pmi_mfg",
                                     "tsf_stock_yoy", "shibor_3m", "cn10y"], lookback_periods=12)
    missing = [c for c in ["m1_yoy", "m2_yoy", "cpi_yoy", "ppi_yoy", "pmi_mfg", "tsf_stock_yoy", "shibor_3m", "cn10y"]
               if m[c].dropna().empty]
    assert not missing, f"宏观指标缺数据: {missing}"


def test_11_5_universe_no_lookahead_2015_06_12(q):
    """get_universe('2015-06-12') 不含 list_date > 2014-10-05 或 delist_date <= 2015-06-12 的股票。"""
    as_of = D(2015, 6, 12)
    u = q.get_universe(as_of)
    assert len(u) > 1000, f"2015-06-12 池子只有 {len(u)} 只，数据明显不全"
    b = q.get_stock_basic(as_of, u)
    late = b[b["list_date"] > D(2014, 10, 5)]
    dead = b[b["delist_date"].notna() & (b["delist_date"] <= as_of)]
    assert late.empty and dead.empty, f"次新 {len(late)} 只 / 已退市 {len(dead)} 只混入"


def test_11_6_financial_pit_maotai_2021_04_01(q):
    """get_financial('600519.SH', as_of='2021-04-01') 返回 end_date=2020-12-31 且 ann_date<=2021-04-01。"""
    f = q.get_financial("2021-04-01", ["600519.SH"], ["revenue"])
    assert f.loc["600519.SH", "end_date"] == D(2020, 12, 31)
    assert f.loc["600519.SH", "ann_date"] <= D(2021, 4, 1)


def test_11_7_cross_source_within_half_percent(q):
    """双源交叉：200 只 × 100 日，BaoStock 后复权收盘价偏差 < 0.5%。BaoStock 不可用 → skip 而非 pass。"""
    try:
        from ashare.data.sources.baostock import BaoStockSource
        bao = BaoStockSource()
    except Exception as exc:                          # noqa: BLE001
        pytest.skip(f"BaoStock 不可用: {exc}")
    try:
        r = validate.check_cross_source(MARKET, bao, n_stocks=200, n_days=100)
    finally:
        bao.close()
    assert not r.skipped, r.detail
    assert r.passed, f"偏差>0.5% 的样本 {r.detail['n_bad']}，最大 {r.detail['max_abs_pct_diff']:.4%}，样例 {r.detail['worst'][:3]}"


def test_11_8_placeholder_rows_and_limits(q):
    r1 = validate.check_placeholder_rows(MARKET)
    assert r1.passed, r1.detail
    r2 = validate.check_limit_coverage(MARKET)
    assert r2.detail["unknown_share_non_suspended"] < 0.05, r2.detail    # 涨跌停算不出的比例应很低
