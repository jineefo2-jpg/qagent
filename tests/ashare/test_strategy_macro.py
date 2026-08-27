"""P3 Task 1：宏观择时层。

计划验收断言的原话：**不许**写「2015-06 分高于 2018-12」这类拿已知行情反推指标的测试 ——
正确的反测是逐指标喂构造数据，验证分位、切点、方向、月历对齐与窗不足语义。
真库只做一条冒烟（形状/取值域/北向窗不足旗）。
"""
from __future__ import annotations
import datetime as dt
import pathlib

import pandas as pd
import pytest

pytest.importorskip("duckdb")
from ashare.data import query
from ashare.strategy import macro

D = dt.date


# ══════════════ 纯函数边界 ══════════════
def test_box_cutpoints_are_30_70_inclusive_middle():
    assert macro._box(0.29) == 0.0
    assert macro._box(0.30) == 0.5          # V2：端点归中档
    assert macro._box(0.70) == 0.5
    assert macro._box(0.71) == 1.0


def test_pct_window_needs_full_60_months_and_includes_current():
    s = pd.Series(range(59), index=[D(2015, 1, 1)] * 59)
    assert macro._pct_in_window(s) is None                        # 59 < 60：窗不足
    s = pd.Series(range(60), dtype=float)
    assert macro._pct_in_window(s) == 1.0                         # 当期是窗内最大值（含当期秩）
    s2 = pd.Series([5.0] * 59 + [0.0])
    assert macro._pct_in_window(s2) == pytest.approx(1 / 60)      # 当期最小 → 1/60，不是 0


def test_grid_shift_is_calendar_not_positional():
    """缺一个月时位置 shift 会整体错位 —— 月历取值必须给 NaN。"""
    idx = [D(2024, 1, 31), D(2024, 2, 29), D(2024, 4, 30)]       # 缺 3 月
    s = pd.Series([1.0, 2.0, 4.0], index=idx)
    got = macro._grid_shift(s, 3)
    assert got.loc[D(2024, 4, 30)] == 1.0                        # 4 月的 3 个月前 = 1 月 ✓
    assert pd.isna(got.loc[D(2024, 2, 29)])                      # 2023-11 不存在 → NaN 不错位


def test_to_month_last_takes_last_obs_and_labels_month_end():
    s = pd.Series([1.0, 2.0, 9.0],
                  index=[D(2024, 1, 3), D(2024, 1, 31), D(2024, 2, 5)])
    got = macro._to_month_last(s)
    assert got.loc[D(2024, 1, 31)] == 2.0 and got.loc[D(2024, 2, 29)] == 9.0


# ══════════════ 合成世界全链路 ══════════════
@pytest.fixture()
def synthetic_world(monkeypatch):
    """两只票、月频宏观 2013-01 起、日频指数 2012 起 —— 足够把 60 个月窗填满。"""
    months = pd.date_range("2013-01-31", "2019-06-30", freq="ME").date
    days = [d.date() for d in pd.bdate_range("2012-01-02", "2019-06-28")]

    def fake_get_macro(as_of, inds, lookback_periods=60):
        as_of = query.norm_date(as_of)
        out = {}
        for i in inds:
            if i == "cn10y":
                out[i] = pd.Series(3.0, index=[d for d in days if d <= as_of])
            elif i == "m1_yoy":
                out[i] = pd.Series([5.0 + k % 7 for k in range(len(months))], index=months)
            elif i == "m2_yoy":
                out[i] = pd.Series(8.0, index=months)
            elif i == "tsf_stock_yoy":
                out[i] = pd.Series([10.0 + 0.1 * k for k in range(len(months))], index=months)
        df = pd.DataFrame(out)
        return df[df.index <= as_of]

    def fake_index_bars(as_of, code, lookback=250, fields=()):
        as_of = query.norm_date(as_of)
        idx = [d for d in days if d <= as_of]
        n = len(idx)
        return pd.DataFrame({"close": [100.0 + 0.01 * k for k in range(n)],   # 缓慢上行 → trend > 1
                             "pe_ttm": [20.0] * n}, index=pd.Index(idx, name="trade_date"))

    def fake_trade_dates(as_of, *, start=None, freq="D"):
        as_of = query.norm_date(as_of)
        return [d for d in days if d <= as_of]

    def fake_stock_basic(as_of, ts_codes=None):
        return pd.DataFrame(index=pd.Index(["A.SH", "B.SZ"], name="ts_code"))

    def fake_money_flow(d, codes, fields, lookback=1):
        k = days.index(query.norm_date(d))
        mi = pd.MultiIndex.from_product([codes, [d]], names=["ts_code", "trade_date"])
        return pd.DataFrame({"hk_hold_ratio": [min(50.0, 0.01 * k)] * len(codes)}, index=mi)

    def fake_daily_basic(d, codes, fields, lookback=1):
        return pd.DataFrame({"circ_mv": [1000.0] * len(codes)},
                            index=pd.Index(codes, name="ts_code"))

    for name, fn in (("get_macro", fake_get_macro), ("get_index_bars", fake_index_bars),
                     ("get_trade_dates", fake_trade_dates), ("get_stock_basic", fake_stock_basic),
                     ("get_money_flow", fake_money_flow), ("get_daily_basic", fake_daily_basic)):
        monkeypatch.setattr(query, name, fn)
    return {"days": days, "months": list(months)}


def test_indicator_math_on_synthetic_world(synthetic_world):
    df = macro.macro_indicators("2019-06-28")
    assert list(df.columns) == list(macro.INDICATORS)
    last = df.dropna(how="all").index.max()
    assert df.loc[last, "erp"] == pytest.approx(100.0 / 20.0 - 3.0)          # 单位：百分点
    assert df.loc[D(2019, 5, 31), "m1_m2_gap"] == pytest.approx(
        (5.0 + (synthetic_world["months"].index(D(2019, 5, 31)) % 7)) - 8.0)
    # tsf 是 0.1/月的匀速爬升 → 3 月变化恒 0.3
    assert df["tsf_yoy_chg"].dropna().iloc[-1] == pytest.approx(0.3)
    assert df.loc[last, "trend_ma200"] > 1.0                                  # 缓慢上行世界


def test_north_telescope_matches_hand_computation(synthetic_world):
    days = synthetic_world["days"]
    df = macro.macro_indicators("2019-06-28")
    s = df["north_flow_60"].dropna()
    label = s.index[-1]
    tds = [d for d in days if d <= D(2019, 6, 28)]
    ends = [d for d in tds if d.month != (tds[tds.index(d) + 1].month if tds.index(d) + 1 < len(tds) else 0)]
    t = max(d for d in tds if macro._month_end(d) == label)
    k_t, k_p = days.index(t), days.index(tds[tds.index(t) - 60])
    # N(d) = 2 只票 × ratio%/100 × 1000；C = 2000 → 望远镜差 / C
    n = lambda k: 2 * (min(50.0, 0.01 * k) / 100.0) * 1000.0
    assert s.loc[label] == pytest.approx((n(k_t) - n(k_p)) / 2000.0)


def test_score_boxes_and_position_formula(synthetic_world):
    got = macro.macro_score("2019-06-28")
    assert set(got["scores"]) == set(macro.INDICATORS)
    assert all(v in (0.0, 0.5, 1.0) for v in got["scores"].values())
    # 合成世界里 erp 恒定 → 窗内全相等 → pct = 1.0（≤ 当期的占比）→ 1 分
    assert got["scores"]["erp"] == 1.0
    # trend 单调爬升但增速衰减 → close/MA200 单调下行 → 当期是窗内最小 → 0 分
    assert got["scores"]["trend_ma200"] == 0.0
    assert got["position"] == pytest.approx(0.2 + 0.8 * got["score"])
    assert 0.2 <= got["position"] <= 1.0


def test_short_window_is_neutral_and_flagged(synthetic_world):
    got = macro.macro_score("2015-06-30")       # 宏观序列 2013 起 → 只有 ~30 个月
    assert got["window_short"], "窗不足必须打旗"
    for name in got["window_short"]:
        assert got["scores"][name] == 0.5


# ══════════════ 真库冒烟 ══════════════
_MARKET = pathlib.Path("data/ashare_market.duckdb")


@pytest.mark.skipif(not _MARKET.exists(), reason="真实 market 库不存在")
def test_smoke_on_real_db():
    query.close_db(); query.open_db(str(_MARKET))
    got = macro.macro_score("2019-06-28")
    assert set(got["scores"]) == set(macro.INDICATORS)
    assert all(v in (0.0, 0.5, 1.0) for v in got["scores"].values())
    assert 0.2 <= got["position"] <= 1.0
    # 北向 2016-12 起，2019-06 只有 ~31 个月 → 必然在窗不足名单里
    assert "north_flow_60" in got["window_short"]
