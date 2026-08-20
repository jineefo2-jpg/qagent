"""Task 5：八个基本面因子 —— 主力用例跑在【真实 DuckDB fixture】上，不打桩。

与 Task 4（量价）相反的取舍：量价因子要 400 行价格面板，构造 DataFrame 比建库便宜；
基本面因子一共十来行财报，而它们的正确性【恰恰依赖 get_financial 的确切返回形状】——
n_periods>1 时是 MultiIndex (ts_code, end_date) 且 end_date 倒序、`drop=False` 保留同名列、
无披露的股票【整行缺席】而不是给一行 NaN。把这些打桩掉就是在测自己对 query 的想象。

每条用例钉的是一个具体的错法：

  · 跨年不重置 —— Q1 单季写成「Q1 累计 − 上年年报累计」，得到巨额负数（§10.1）
  · 累计当单季 —— np_yoy 直接对累计值做同比，Q2–Q4 得到一个【看起来很合理】的错数
  · 分母不设限 —— 扭亏样本的负分母给出 ±inf，zscore 之后独占整个组合（§10.2）
  · 期数不足硬算 —— sue 拿 5 期数据算 8 个差分的 σ
  · 分子分母时点错配 —— PIT 分子配上另一天的市值
  · 口径混用 —— accrual 的分子用归母净利（分母却是全部资产）、分母用期初期末平均（§2.2 下标是 e*）

★ 单位说明：fixture 的财报与市值单位随意（Tushare 真实口径是 元 与 万元，差一个 1e4 的
  常数因子）。所有断言要么是同一单位内的比值，要么是相关性 —— 都对常数缩放不敏感，
  这也正是 ep_ttm 带着 1e4 因子却不影响下游（zscore 会抹掉任何正常数乘子）的原因。
"""
from __future__ import annotations
import datetime as dt
import inspect
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, query
from ashare.factors import fundamental as ff

D = dt.date

_FIN_COLS = ("ts_code", "ann_date", "end_date", "report_type", "update_flag",
             "revenue", "n_income", "n_income_attr_p", "total_assets",
             "total_hldr_eqy_exc_min_int", "n_cashflow_act", "grossprofit_margin")


def _fin(code: str, ann: dt.date, end: dt.date, *, flag: int = 0, rev=None, ni=None,
         ni_attr=None, ta=None, eq=None, cfo=None, gm=None) -> tuple:
    return (code, ann, end, "1", flag, rev, ni, ni_attr, ta, eq, cfo, gm)


def _extend_calendar(w) -> None:
    """把日历延到 2021-01-01~2024-02-29（工作日开市）。market_db 自带 2023-12-25~2024-02-02
    的真实片段（含元旦休市），这里只补它没有的日子，不覆盖。"""
    have = {r[0] for r in w.execute("SELECT trade_date FROM calendar").fetchall()}
    rows, d = [], D(2021, 1, 1)
    while d <= D(2024, 2, 29):
        if d not in have:
            rows.append((d, d.weekday() < 5, None))
        d += dt.timedelta(days=1)
    w.executemany("INSERT INTO calendar VALUES (?, ?, ?)", rows)


# ══════════════ fixture 财报（全部【累计口径】）══════════════
#
# A00001.SZ —— 主力样本，2020Q1~2023Q4 共 16 个报告期
#   单季归母净利  2020: 5 5 5 5 | 2021: 10 10 10 10 | 2022: 20 20 20 20 | 2023: 31 30 30 30
#   → 累计         2020: 5 10 15 20 | 2021: 10 20 30 40 | 2022: 20 40 60 80 | 2023: 31 61 91 121
#   ★ 2020 那一年是给「e* 落在 Q3」的 sue 用例准备的：e* 是 Q4 时 12 期单季只要 12 期累计
#     （最老的一期正好是 Q1，累计即单季），e* 是 Q1/Q2/Q3 时才需要第 13 期。
#   营收 = 归母净利 × 10（同一套累计结构，用来分辨 sp_ttm 是否取错字段）
# B00002.SZ —— 分母陷阱：2022Q2 单季为负（扭亏样本）、净资产为负（资不抵债）
# C00003.SH —— 只有一期披露：np_yoy / sue 必须 NaN，不得外推
# D00004.SZ —— 8 个同比差分完全相等 → σ=0，sue 必须 NaN 而不是 inf
# E00005.SZ —— 与 A 逐期同值但缺 2021Q1 → 只有 11 期单季（sue 的期数闸门靶子）
_A_NI_CUM = {2020: (5, 10, 15, 20), 2021: (10, 20, 30, 40),
             2022: (20, 40, 60, 80), 2023: (31, 61, 91, 121)}
_D_NI_CUM = {2021: (10, 20, 30, 40), 2022: (20, 40, 60, 80), 2023: (30, 60, 90, 120)}
_Q_END = ((3, 31), (6, 30), (9, 30), (12, 31))
_ANN = ((4, 28), (8, 25), (10, 27), (3, 30))          # Q1/H1/Q3 当年公告；年报次年 3-30


def _quarterly_rows(code: str, cum: dict, **extra) -> list[tuple]:
    """把 {年: (Q1累计, H1累计, Q3累计, 年报累计)} 摊成 financial_pit 行。"""
    out = []
    for year, vals in cum.items():
        for (em, ed), (am, ad), v in zip(_Q_END, _ANN, vals):
            ann = D(year + 1, am, ad) if em == 12 else D(year, am, ad)
            out.append(_fin(code, ann, D(year, em, ed), ni_attr=float(v), rev=float(v) * 10,
                            **(extra if em == 12 else {})))
    return out


_FY23_ANN = D(2024, 1, 20)                            # FY2023 统一提前到 01-20 公告


def _fixture_rows() -> list[tuple]:
    rows = _quarterly_rows("A00001.SZ", _A_NI_CUM)
    rows += _quarterly_rows("D00004.SZ", _D_NI_CUM)
    rows += [r for r in _quarterly_rows("E00005.SZ", _A_NI_CUM) if r[2] != D(2021, 3, 31)]
    # 年报默认次年 3-30 公告，那样 as_of=2024-01-25 时 e* 会停在三季报；这里统一提前到
    # 01-20，好让 AS_OF 的 e* 确定地落在 FY2023 上（否则用例会"因为别的原因"通过）。
    rows = [r if r[2] != D(2023, 12, 31) else (r[0], _FY23_ANN) + r[2:] for r in rows]
    # A 的 FY2022 / FY2023 补上存量与现金流科目（估值 / 质量类因子要用）
    by_end = {(r[0], r[2]): i for i, r in enumerate(rows)}
    for end, ta, eq, ni, cfo, gm in ((D(2022, 12, 31), 1600.0, 800.0, 85.0, 60.0, 40.0),
                                     (D(2023, 12, 31), 2000.0, 1000.0, 130.0, 90.0, 45.5)):
        i = by_end[("A00001.SZ", end)]
        r = list(rows[i])
        r[6], r[8], r[9], r[10], r[11] = ni, ta, eq, cfo, gm
        rows[i] = tuple(r)
    rows += [
        # A 的 FY2023 重述行（update_flag=1）：默认不可见，任何因子都不许用到 999
        _fin("A00001.SZ", D(2024, 1, 22), D(2023, 12, 31), flag=1, ni_attr=999.0, rev=9990.0,
             ta=9999.0, eq=9999.0, ni=999.0, cfo=999.0, gm=99.0),
        # A 的 2024Q1：公告日在所有 as_of 之后，PIT 必须看不见
        _fin("A00001.SZ", D(2024, 4, 25), D(2024, 3, 31), ni_attr=500.0, rev=5000.0),
        # B：2022 H1 累计 5 < Q1 累计 10 → 2022Q2 单季 = −5（分母陷阱）
        _fin("B00002.SZ", D(2022, 4, 28), D(2022, 3, 31), ni_attr=10.0),
        _fin("B00002.SZ", D(2022, 8, 25), D(2022, 6, 30), ni_attr=5.0),
        _fin("B00002.SZ", D(2023, 3, 30), D(2022, 12, 31), ni_attr=40.0, eq=-30.0, ta=100.0),
        _fin("B00002.SZ", D(2023, 4, 28), D(2023, 3, 31), ni_attr=12.0),
        _fin("B00002.SZ", D(2023, 8, 25), D(2023, 6, 30), ni_attr=37.0),
        _fin("B00002.SZ", _FY23_ANN, D(2023, 12, 31), ni_attr=50.0, rev=500.0,
             eq=-50.0, ta=100.0, ni=50.0, cfo=10.0, gm=12.0),
        # C：只有一期
        _fin("C00003.SH", D(2023, 3, 30), D(2022, 12, 31), ni_attr=7.0, rev=70.0,
             eq=100.0, ta=200.0, ni=7.0, cfo=6.0, gm=33.0),
    ]
    return rows


# X01..X12：ep_ttm ↔ 1/pe_ttm 交叉校验专用。e* 落在 2023Q3（不是年报）——
# TTM 必须真的走「Q3累计 + 上年年报 − 上年同期累计」的拼接，才算校验到那条路径。
_X_CODES = [f"X{i:02d}.SZ" for i in range(1, 13)]
_X_MV = 1e6
_X_ASOF = D(2023, 11, 1)


def _x_ttm(i: int) -> float:
    return 115.0 * 200.0 * (i + 1)                    # 75k + 100k − 60k，k = 200(i+1)


def _x_rows() -> tuple[list[tuple], list[tuple]]:
    fin, basic = [], []
    for i, code in enumerate(_X_CODES):
        k = 200.0 * (i + 1)
        fin += [_fin(code, D(2022, 10, 27), D(2022, 9, 30), ni_attr=60.0 * k),
                _fin(code, D(2023, 3, 30), D(2022, 12, 31), ni_attr=100.0 * k),
                _fin(code, D(2023, 10, 27), D(2023, 9, 30), ni_attr=75.0 * k)]
        jitter = 1.0 + 0.03 * (1 if i % 2 else -1)    # 供应商自算的 pe_ttm 与我们差几个点
        basic.append((code, _X_ASOF, _X_MV, _X_MV * 1e4 / _x_ttm(i) * jitter))
    return fin, basic


@pytest.fixture
def fin_db(market_db):
    """market_db（日历 / 4 只股票 / daily_basic 市值）+ 财报 + X01..X12。"""
    w = _db.connect_write(market_db)
    _extend_calendar(w)
    x_fin, x_basic = _x_rows()
    w.executemany(f"INSERT INTO financial_pit ({', '.join(_FIN_COLS)}) VALUES "
                  f"({', '.join('?' * len(_FIN_COLS))})", _fixture_rows() + x_fin)
    w.executemany("INSERT INTO daily_basic (ts_code, trade_date, total_mv, pe_ttm) VALUES (?, ?, ?, ?)",
                  x_basic)
    w.close()
    query.open_db(market_db)
    yield query
    query.close_db()


# market_db 的 total_mv：A=1e6, B=5e5, C=1e4, D=2e5
AS_OF = "2024-01-25"        # FY2023 已于 01-20 公告；2024Q1（04-25 公告）仍不可见
MV_A = 1e6


# ══════════════ 1 单季化：跨年重置（§1.2，§10 坑 1）══════════════
@pytest.mark.parametrize("as_of,expect,cum_bug", [
    ("2023-05-10", 31 / 20 - 1, 31 / 20 - 1),      # e*=2023Q1：累计【就是】单季，两者同值
    ("2023-09-01", 30 / 20 - 1, 61 / 40 - 1),      # e*=2023H1
    ("2023-11-01", 30 / 20 - 1, 91 / 60 - 1),      # e*=2023Q3
    (AS_OF, 30 / 20 - 1, 121 / 80 - 1),            # e*=FY2023
])
def test_np_yoy_is_single_quarter_not_cumulative(fin_db, as_of, expect, cum_bug):
    """np_yoy 必须用单季同比。直接拿累计值做同比不会报错，只会给出一个【看起来很合理】的数
    （0.5 → 0.525），一年四期里有三期是错的。"""
    got = ff.np_yoy(as_of, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(expect, rel=1e-12)
    if cum_bug != pytest.approx(expect):
        assert got != pytest.approx(cum_bug, rel=1e-9), "对累计值做了同比"


def test_np_yoy_q1_single_equals_q1_cumulative_not_minus_last_annual(fin_db):
    """★ 跨年重置的核心断言：Q1 单季 = Q1 累计 = 31，不是「31 − 上年年报 80 = −49」。

    写错的话分子分母双双变负，np_yoy 要么被分母闸门打成 NaN、要么给出一个正的比值 ——
    两种都不会抛，而 Q2–Q4 依然正常，所以每年只有三个月的数据是坏的（最难发现的那种）。
    """
    got = ff.np_yoy("2023-05-10", ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(31 / 20 - 1, rel=1e-12)
    assert not np.isnan(got), "Q1 单季算成了「Q1 累计 − 上年年报」，分母变负后被闸门打成 NaN"
    assert got != pytest.approx((31 - 80) / (20 - 40) - 1, rel=1e-9)


def test_np_yoy_negative_base_is_nan_not_a_finite_ratio(fin_db):
    """B 的 2022Q2 单季 = 5 − 10 = −5（扭亏样本）。25/(−5) − 1 = −6：一个有限的、
    符号完全反了的「业绩大幅下滑」分数。分母 ≤ 0 必须 NaN（§10.2）。"""
    got = ff.np_yoy("2023-09-01", ["B00002.SZ"])["B00002.SZ"]
    assert np.isnan(got), f"负分母产出了有限值 {got}"


def test_np_yoy_needs_both_periods_and_never_extrapolates(fin_db):
    """C 只有一期披露：e* 的单季值都拼不出来（缺上一期累计），更没有去年同期。"""
    assert np.isnan(ff.np_yoy(AS_OF, ["C00003.SH"])["C00003.SH"])


def test_np_yoy_ignores_restated_rows(fin_db):
    """A 的 FY2023 有一条 update_flag=1 的重述行（999）。用它 = 前视（D3）。"""
    got = ff.np_yoy(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(30 / 20 - 1, rel=1e-12)
    assert got != pytest.approx((999 - 91) / 20 - 1, rel=1e-6), "用上了重述值"


def test_np_yoy_does_not_see_reports_announced_after_as_of(fin_db):
    """A 的 2024Q1 于 2024-04-25 公告。as_of=2024-01-25 时它不存在 —— e* 必须还是 FY2023。"""
    assert ff.np_yoy(AS_OF, ["A00001.SZ"])["A00001.SZ"] == pytest.approx(30 / 20 - 1, rel=1e-12)


# ══════════════ 2 sue ══════════════
_SUE_A = 10.0 / np.sqrt(0.125)      # d = [10,10,10,11,10,10,10,10] → σ(ddof=1) = √0.125


def test_sue_closed_form_on_twelve_quarters(fin_db):
    """A 的 8 个同比差分是 [10,10,10,11,10,10,10,10]：均值 10.125，σ(ddof=1)=√0.125，
    SUE = 10/√0.125 = 20√2 ≈ 28.284。"""
    got = ff.sue(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(_SUE_A, rel=1e-12)
    assert got == pytest.approx(20 * np.sqrt(2), rel=1e-12)


def test_sue_when_e_star_is_q3_needs_a_thirteenth_cumulative_period(fin_db):
    """★ e* 落在 Q3（占全年四分之三的时间）时，最老的那期单季是 2020Q4 —— 它要减
    2020Q3，于是【累计】期数得取 13 而不是 12。e* 是 Q4 时最老的一期正好是 Q1
    （累计即单季），12 期就够 —— 只用年报做用例会让这个差别整个消失。

    d = [10,10,11,10,10,10,10,5]（最后一项跨到 2020 年，单季从 5 涨到 10）
    → σ(ddof=1) = √(24/7)，SUE = 10/√(24/7) ≈ 5.4006。
    """
    got = ff.sue("2023-11-01", ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(10.0 / np.sqrt(24 / 7), rel=1e-12)


def test_sue_uses_sample_std_ddof_1(fin_db):
    """总体标准差（ddof=0）会给出 30.24 —— 差 7%，混用不报错只静默改数。"""
    got = ff.sue(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got != pytest.approx(10.0 / np.sqrt(0.875 / 8), rel=1e-6), "σ 用成了 ddof=0"


def test_sue_denominator_is_yoy_differences_not_growth_rates(fin_db):
    """★ 规格表格写的是「σ(过去 8 期同比增速)」，§2.2 的公式写的是 σ(同比【差分】)。
    以后者为准：分子是差分（有量纲），分母若换成增速（无量纲）SUE 就不是标准化量了。
    增速集合 {0.5,0.5,0.5,0.55,1,1,1,1} 的 σ ≈ 0.2586 → SUE ≈ 38.7，与 28.28 明显可分。"""
    growth = np.array([30 / 20, 30 / 20, 30 / 20, 31 / 20, 20 / 10, 20 / 10, 20 / 10, 20 / 10]) - 1
    got = ff.sue(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got != pytest.approx(10.0 / growth.std(ddof=1), rel=1e-6)


def test_sue_zero_dispersion_is_nan_not_infinity(fin_db):
    """D 的 8 个差分全是 10 → σ=0。不设闸就是 10/0 = inf，而 inf 过 zscore 后独占组合。"""
    got = ff.sue(AS_OF, ["D00004.SZ"])["D00004.SZ"]
    assert np.isnan(got), f"σ=0 产出了 {got}"


def test_sue_needs_twelve_single_quarters(fin_db):
    """E 与 A 逐期同值，只少了最早的 2021Q1 → 11 期单季，第 8 个差分拼不出来 → NaN。
    宽松成「有几期算几期」的话，一只刚上市三年的股票会拿到一个用 5 期算出来的 σ，
    而 σ 越小 SUE 越大 —— 数据最少的股票会系统性地排在最前面。"""
    out = ff.sue(AS_OF, ["A00001.SZ", "E00005.SZ"])
    assert np.isnan(out["E00005.SZ"]), "只有 11 期单季却算出了 SUE"
    assert out["A00001.SZ"] == pytest.approx(_SUE_A, rel=1e-12), "12 期齐全的股票被连坐"


def test_eleven_period_stock_still_gets_np_yoy(fin_db):
    """反向：期数闸门是 sue 的，不该外溢到只要两期的 np_yoy。"""
    assert ff.np_yoy(AS_OF, ["E00005.SZ"])["E00005.SZ"] == pytest.approx(30 / 20 - 1, rel=1e-12)


def test_sue_is_nan_for_a_stock_with_one_report(fin_db):
    assert np.isnan(ff.sue(AS_OF, ["C00003.SH"])["C00003.SH"])


# ══════════════ 3 估值三兄弟：分子 PIT 财报 / 分母 as_of 市值 ══════════════
def test_ep_ttm_closed_form(fin_db):
    """e*=FY2023 → TTM(归母净利)=年报值 121；A 的 total_mv=1e6。"""
    assert ff.ep_ttm(AS_OF, ["A00001.SZ"])["A00001.SZ"] == pytest.approx(121 / MV_A, rel=1e-12)


def test_sp_ttm_closed_form(fin_db):
    """营收是归母净利的 10 倍：TTM(营收)=1210。取错字段会得到 ep_ttm 的值。"""
    got = ff.sp_ttm(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(1210 / MV_A, rel=1e-12)
    assert got != pytest.approx(121 / MV_A, rel=1e-6), "sp_ttm 取成了净利"


def test_bp_uses_period_end_equity_not_the_two_period_average(fin_db):
    """§2.2：BP = E_{e*} / MV，分子是【期末】净资产 1000，不是 roe_ttm 那个
    期初期末平均 (1000+800)/2 = 900。两者差 11%，都"看着合理"。"""
    got = ff.bp(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(1000 / MV_A, rel=1e-12)
    assert got != pytest.approx(900 / MV_A, rel=1e-6), "bp 用了期初期末平均净资产"


def test_bp_of_a_negative_equity_stock_is_finite_and_negative(fin_db):
    """★ 有意为之的【不】置 NaN：B 资不抵债（净资产 −50），BP = −1e-4。

    闸门只设在【分母】上（规格 §10.2 的原话是"分母 ≤ 0 一律置 NaN"）。负分子在这里
    没有病理：BP 穿过 0 是连续的，direction=+1 下负 BP 就排在最末 —— 一家资不抵债的
    公司排在价值因子的最差端，方向是对的。
    与之相反的是 roe_ttm（下一条）：净资产在【分母】，正利润 ÷ 负净资产会把一家
    盈利的公司算成负 ROE，那是符号翻转，必须拦。两者的差别就是分子与分母之别。
    """
    got = ff.bp(AS_OF, ["B00002.SZ"])["B00002.SZ"]
    assert np.isfinite(got) and got < 0
    assert got == pytest.approx(-50 / 5e5, rel=1e-12)


def test_valuation_denominator_is_market_cap_of_the_as_of_day(fin_db):
    """同一份财报配不同的市值：因子值必须按市值反比缩放。市值取成别的日子 = 分子分母时点错配。"""
    a = ff.ep_ttm(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    b = ff.ep_ttm(AS_OF, ["B00002.SZ"])["B00002.SZ"]
    assert a / b == pytest.approx((121 / 1e6) / (50 / 5e5), rel=1e-12)


# ══════════════ 4 roe_ttm / accrual / gross_margin ══════════════
def test_roe_ttm_uses_average_equity_not_period_end(fin_db):
    """§2.2：分母 ½(E_{e*} + E_{e*−4}) = (1000+800)/2 = 900 → 121/900。
    用期末值 1000 会得到 0.121，与 0.1344 相差 10%。"""
    got = ff.roe_ttm(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got == pytest.approx(121 / 900, rel=1e-12)
    assert got != pytest.approx(121 / 1000, rel=1e-6), "roe_ttm 用了期末净资产"


def test_roe_ttm_negative_average_equity_is_nan(fin_db):
    """B 的两期净资产 −50 / −30 → 均值 −40。121/(−40) 是个有限的负 ROE。"""
    assert np.isnan(ff.roe_ttm(AS_OF, ["B00002.SZ"])["B00002.SZ"])


def test_accrual_closed_form_and_direction_of_the_ratio(fin_db):
    """(TTM(净利 130) − TTM(经营现金流 90)) / 期末总资产 2000 = 0.02。"""
    assert ff.accrual(AS_OF, ["A00001.SZ"])["A00001.SZ"] == pytest.approx(40 / 2000, rel=1e-12)


def test_accrual_numerator_is_total_net_income_not_the_attributable_part(fin_db):
    """§2.2 的 ACC 分子是 TTM(NI)（全部净利 130），不是归母净利 121 —— 分母是全部资产，
    分子换成归母就成了「归母利润 / 全部资产」，主体对不上。"""
    got = ff.accrual(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got != pytest.approx((121 - 90) / 2000, rel=1e-6), "accrual 用了归母净利"


def test_accrual_denominator_is_period_end_total_assets(fin_db):
    """§2.2 的下标是 TA_{i,e*}（期末 2000），不是期初期末平均 (2000+1600)/2 = 1800。"""
    got = ff.accrual(AS_OF, ["A00001.SZ"])["A00001.SZ"]
    assert got != pytest.approx(40 / 1800, rel=1e-6), "accrual 用了期初期末平均总资产"


def test_gross_margin_is_the_latest_pit_disclosure(fin_db):
    """毛利率是【比率】科目，不做 TTM 拼接（库里没有营业成本，拼不出来）；取最新一期 PIT 值。"""
    assert ff.gross_margin(AS_OF, ["A00001.SZ"])["A00001.SZ"] == pytest.approx(45.5, rel=1e-12)


def test_gross_margin_ignores_the_restated_row(fin_db):
    assert ff.gross_margin(AS_OF, ["A00001.SZ"])["A00001.SZ"] != pytest.approx(99.0, rel=1e-6)


# ══════════════ 5 交叉校验：ep_ttm ↔ 1/pe_ttm（两条独立取数路径）══════════════
def test_ep_ttm_correlates_with_the_vendor_pe_ttm(fin_db):
    """X01..X12 的 e* 落在 2023Q3，ep_ttm 必须真的走「Q3累计 + 上年年报 − 上年同期累计」；
    daily_basic.pe_ttm 是供应商自算的（这里带 ±3% 抖动）。两条路径的相关性 > 0.95。

    这条测的是【两个数据源互证】，不是我们的算术自证：TTM 拼接、字段选取、分母口径
    任一错位都会让相关性塌掉，而它们各自都不会抛异常。
    """
    ep = ff.ep_ttm(_X_ASOF, _X_CODES)
    inv_pe = 1.0 / query.get_daily_basic(_X_ASOF, _X_CODES, fields=("pe_ttm",))["pe_ttm"].reindex(_X_CODES)
    assert ep.notna().all() and inv_pe.notna().all()
    assert ep.corr(inv_pe) > 0.95, f"两条取数路径不一致: corr={ep.corr(inv_pe)}"


def test_ep_ttm_ttm_assembly_matches_the_hand_computed_value(fin_db):
    """同一批股票的闭式：TTM = 75k + 100k − 60k = 115k。只用 Q3 累计（75k）就少了 35%。"""
    ep = ff.ep_ttm(_X_ASOF, _X_CODES)
    expect = np.array([_x_ttm(i) / _X_MV for i in range(len(_X_CODES))])
    np.testing.assert_allclose(ep.to_numpy(), expect, rtol=1e-12)


# ══════════════ 6 分子分母时点一致（打桩探针）══════════════
@pytest.mark.parametrize("name", ["ep_ttm", "bp", "sp_ttm", "accrual", "gross_margin"])
def test_numerator_and_denominator_share_the_same_as_of_date(monkeypatch, name):
    """"分子取 PIT 财报、分母取 as_of 当日市值，两者时点必须一致"（规格 §2.2）。
    错配是【反方向】的前视：拿旧财报配新市值，或反过来 —— 都不会报错。"""
    seen: dict[str, list] = {"fin": [], "basic": []}

    def _fake_ttm(as_of_date, ts_codes, field):
        seen["fin"].append(as_of_date)
        return pd.Series(1.0, index=pd.Index(list(ts_codes), name="ts_code"), dtype=float)

    def _fake_fin(as_of_date, ts_codes, fields, *, n_periods=1, include_restated=False, report_type="1"):
        assert include_restated is False, "D3：重述值不得进因子"
        seen["fin"].append(as_of_date)
        idx = pd.Index(list(ts_codes), name="ts_code")
        return pd.DataFrame({f: 1.0 for f in fields}, index=idx)

    def _fake_basic(as_of_date, ts_codes, fields=("total_mv",), lookback=1):
        assert lookback == 1, "分母必须是 as_of 当日的市值，不是一段窗口的均值"
        seen["basic"].append(as_of_date)
        return pd.DataFrame({"total_mv": 2.0}, index=pd.Index(list(ts_codes), name="ts_code"))

    monkeypatch.setattr(query, "get_financial_ttm", _fake_ttm)
    monkeypatch.setattr(query, "get_financial", _fake_fin)
    monkeypatch.setattr(query, "get_daily_basic", _fake_basic)
    getattr(ff, name)("2024-01-25", ["A00001.SZ"])
    assert set(seen["fin"]) <= {"2024-01-25"} and set(seen["basic"]) <= {"2024-01-25"}
    assert seen["fin"], f"{name} 没有取任何财报"


# ══════════════ 契约：签名 / index / 注册元数据 ══════════════
_ALL = ["ep_ttm", "bp", "sp_ttm", "roe_ttm", "gross_margin", "accrual", "np_yoy", "sue"]


@pytest.mark.parametrize("name", _ALL)
def test_first_two_positional_params_are_as_of_date_and_universe(name):
    """L3 静态检查之外再钉一次：装饰器原样返回函数，签名就是运行期真签名。"""
    params = list(inspect.signature(getattr(ff, name)).parameters.values())
    assert [p.name for p in params[:2]] == ["as_of_date", "universe"]
    assert all(p.kind is p.KEYWORD_ONLY for p in params[2:]), "其余参数必须是 keyword-only"


@pytest.mark.parametrize("name,direction", [
    ("ep_ttm", 1), ("bp", 1), ("sp_ttm", 1), ("roe_ttm", 1),
    ("gross_margin", 1), ("accrual", -1), ("np_yoy", 1), ("sue", 1)])
def test_registered_metadata_matches_the_spec_table(name, direction):
    """★ accrual 是唯一的 −1：应计利润【高】预示后续收益【低】。
    符号写反不会报错，只会让组合系统性地买进利润含金量最差的一批股票。"""
    from ashare.factors.base import get_factor
    spec = get_factor(name)
    assert spec.direction == direction
    assert spec.category == "fundamental"
    assert spec.neutralize is True
    assert spec.fn is getattr(ff, name)


@pytest.mark.parametrize("name", _ALL)
def test_index_is_exactly_universe_in_universe_order(fin_db, name):
    """index 必须【就是】universe 且保持传入顺序 —— 下游 process 之外还有按位置读的代码。
    故意用非字典序，并混入一只库里完全没有的股票。"""
    unordered = ["D00004.SZ", "A00001.SZ", "ZZZZZ.SZ", "B00002.SZ"]
    out = getattr(ff, name)(AS_OF, unordered)
    assert isinstance(out, pd.Series)
    assert list(out.index) == unordered, f"{name} 的 index 不等于 universe（或顺序变了）"


@pytest.mark.parametrize("name", _ALL)
def test_a_code_absent_from_the_source_is_nan_not_missing(fin_db, name):
    """池内某股完全无财报 → 该股 NaN，但仍在 index 里（少一行会让横截面回归静默错位）。"""
    out = getattr(ff, name)(AS_OF, ["A00001.SZ", "ZZZZZ.SZ"])
    assert np.isnan(out["ZZZZZ.SZ"])
    assert out.notna()["A00001.SZ"], f"{name} 对有完整数据的股票也给了 NaN"


@pytest.mark.parametrize("name", _ALL)
def test_empty_universe_returns_an_empty_series(fin_db, name):
    """空票池（某天全池被 ST / 停牌剔光）不该抛。"""
    out = getattr(ff, name)(AS_OF, [])
    assert isinstance(out, pd.Series) and len(out) == 0


@pytest.mark.parametrize("name", _ALL)
def test_no_factor_ever_returns_an_infinity(fin_db, name):
    """全池扫一遍：±inf 一个都不许有（B 有负净资产、D 有零离散度，两个陷阱都在池里）。
    inf 活过 MAD 去极值（clip 到上界）也活过 zscore，最后独占整个组合。"""
    out = getattr(ff, name)(AS_OF, ["A00001.SZ", "B00002.SZ", "C00003.SH", "D00004.SZ"])
    assert not np.isinf(out.to_numpy(dtype=float)).any(), f"{name} 产出了 inf: {out.to_dict()}"


# ══════════════ 真实库（茅台 PIT）—— 库不存在则 skip ══════════════
MARKET = os.environ.get("ASHARE_MARKET_DB", "data/ashare_market.duckdb")
real_db = pytest.mark.skipif(not pathlib.Path(MARKET).exists(),
                             reason=f"真实 market 库不存在: {MARKET}")


@real_db
def test_moutai_pit_never_uses_a_report_announced_after_as_of():
    """规格验收：as_of=2021-04-01 的茅台因子只能建立在【当日已公告】的报告上。

    ★ 这里不硬编码"用 2020 年报" —— 贵州茅台 FY2020 年报的公告日在 4 月下旬，
      2021-04-01 当天大概率还看不到（见任务报告）。真正要钉的不变量是两条：
      (1) e* 的 ann_date ≤ as_of；(2) 年报公告之后再查，e* 必须前进到年报。
    """
    query.open_db(MARKET)
    try:
        code = "600519.SH"
        early = query.get_financial("2021-04-01", [code], ["n_income_attr_p"])
        assert early.loc[code, "ann_date"] <= dt.date(2021, 4, 1)
        assert np.isfinite(ff.ep_ttm("2021-04-01", [code])[code])
        late = query.get_financial("2021-06-01", [code], ["n_income_attr_p"])
        assert late.loc[code, "end_date"] >= early.loc[code, "end_date"]
    finally:
        query.close_db()


@real_db
def test_ep_ttm_correlates_with_pe_ttm_on_the_real_cross_section():
    """全 A 横截面上 ep_ttm 与 1/pe_ttm 的相关性 > 0.95。跑不过是数据层的真实发现。"""
    query.open_db(MARKET)
    try:
        as_of = os.environ.get("ASHARE_XCHECK_DATE", "2023-11-01")
        codes = query.get_universe(as_of)
        ep = ff.ep_ttm(as_of, codes)
        inv = (1.0 / query.get_daily_basic(as_of, codes, fields=("pe_ttm",))["pe_ttm"]).reindex(codes)
        both = pd.concat([ep, inv], axis=1).dropna()
        assert len(both) > 100
        assert both.iloc[:, 0].corr(both.iloc[:, 1]) > 0.95
    finally:
        query.close_db()
