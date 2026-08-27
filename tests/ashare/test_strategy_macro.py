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
    """两只票、月频宏观 2013-01 起（次月 15 日可见）、日频指数 2012 起。缓存逐测清空。"""
    macro._CACHE.clear(); macro._SCORE_MEMO.clear()
    months = pd.date_range("2013-01-31", "2019-06-30", freq="ME").date
    days = [d.date() for d in pd.bdate_range("2012-01-02", "2019-06-28")]
    monkeypatch.setattr(query, "snapshot_id", lambda *, pin=False: "snap-syn")

    def fake_get_macro(as_of, inds, lookback_periods=60):
        as_of = query.norm_date(as_of)
        out = {}
        for i in inds:
            if i == "cn10y":
                idx = [d for d in days if d <= as_of]
                out[i] = pd.Series(3.0, index=idx)
                out[f"{i}__publish_date"] = pd.Series(idx, index=idx)      # 日频当日可见
            else:
                vals = {"m1_yoy": [5.0 + k % 7 for k in range(len(months))],
                        "m2_yoy": [8.0] * len(months),
                        "tsf_stock_yoy": [10.0 + 0.1 * k for k in range(len(months))]}[i]
                out[i] = pd.Series(vals, index=months, dtype=float)
                out[f"{i}__publish_date"] = pd.Series(
                    [m + dt.timedelta(days=15) for m in months], index=months)   # 次月中旬可见
        return pd.DataFrame(out)      # 可见性由缓存层按 __publish_date 切，这里给全量

    def fake_index_bars(as_of, code, lookback=250, fields=()):
        as_of = query.norm_date(as_of)
        idx = [d for d in days if d <= as_of]
        n = len(idx)
        return pd.DataFrame({"close": [100.0 + 0.01 * k for k in range(n)],
                             "pe_ttm": [20.0] * n}, index=pd.Index(idx, name="trade_date"))

    def fake_north(as_of):
        as_of = query.norm_date(as_of)
        idx = [d for d in days if d <= as_of]
        return pd.DataFrame({"north_mv": [0.2 * k for k in range(len(idx))],   # 2 票 × 0.01k%/100 × 1000
                             "circ_mv_total": [2000.0] * len(idx)},
                            index=pd.Index(idx, name="trade_date"))

    for name, fn in (("get_macro", fake_get_macro), ("get_index_bars", fake_index_bars),
                     ("get_north_aggregate", fake_north)):
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
    """N 匀速爬升 0.2/日 → 任何月末的 (N(t)−N(t−60))/C 恒为 0.2×60/2000。"""
    s = macro.macro_indicators("2019-06-28")["north_flow_60"].dropna()
    assert len(s) > 12
    assert s.iloc[-1] == pytest.approx(0.2 * 60 / 2000.0)
    assert s.std() == pytest.approx(0.0, abs=1e-12)


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


def test_monthly_value_invisible_before_publish_date(synthetic_world):
    """PIT：2019-05 的月频值 6 月 15 日才可见 —— 6 月 10 日查最新可见必须是 2019-04。"""
    df = macro.macro_indicators("2019-06-10")
    assert df["m1_m2_gap"].dropna().index.max() == D(2019, 4, 30)
    df2 = macro.macro_indicators("2019-06-16")
    assert df2["m1_m2_gap"].dropna().index.max() == D(2019, 5, 31)


def test_warm_cache_slice_equals_cold_rebuild(synthetic_world):
    """增量缓存的唯一契约：暖缓存切片 == 冷重建，逐位相同（任何 as_of）。"""
    macro.macro_indicators("2019-06-28")            # 以 2019-06-28 为终点建缓存
    warm = macro.macro_indicators("2015-06-30")     # 暖：切片
    macro._CACHE.clear(); macro._SCORE_MEMO.clear()
    cold = macro.macro_indicators("2015-06-30")     # 冷：以 2015-06-30 为终点重建
    pd.testing.assert_frame_equal(warm, cold)


@pytest.mark.skipif(not _MARKET.exists(), reason="真实 market 库不存在")
def test_perf_sequential_scores_are_millisecond_scale():
    """性能钉（2026-08-27 用户裁决：不许逐调用全量重建）：预热后连打 20 个交易日的
    macro_score 合计 < 1.5s —— 回测 511 期的宏观开销因此进秒级。"""
    import time
    query.close_db(); query.open_db(str(_MARKET))
    try:
        macro._CACHE.clear(); macro._SCORE_MEMO.clear()
        macro.macro_score("2019-06-28")                       # 预热（建缓存）
        dates = query.get_trade_dates("2019-06-28", freq="W")[-20:]
        t0 = time.monotonic()
        for d in dates:
            macro.macro_score(d)
        el = time.monotonic() - t0
        assert el < 1.5, f"20 次逐期评分耗时 {el:.2f}s —— 缓存没生效"
    finally:
        query.close_db()
