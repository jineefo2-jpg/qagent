"""Task 6：资金因子 + 三个风险因子 + 注册表装配。

本任务的三个失败模式全部是【静默】的 —— 不抛异常，不打日志，只产出一条画得出来的净值曲线：

  · 静默的 0     —— 北向持股 2016-12-05 才有数据。此前必须是 NaN：0 在本因子里是
                    合法取值（"持股比例没变"），填 0 会让 combine 把它当成一个可用因子
                    计入分母、而分子恒等于同一个常数 —— 2010–2016 六年被静默降权。
  · 静默的自我中性化 —— log_mv / industry / beta_250 【就是】pipeline.neutralize 的回归元。
                    neutralize=True 会让它们对自己回归，残差恒 0，zscore 之后是一列
                    放大了 1e16 倍的舍入误差，看起来和真因子没有任何区别。
  · 静默的假零收益 —— D9 的停牌占位行（vol=0、OHLC=前收）在 query 层被掩成 NaN。
                    beta 一旦 ffill 就把它还原成一个【假的 0 收益】，停牌越多的股票
                    beta 越低（系统性偏向 0），中性化于是少扣了它们的市场暴露。

★ 装配用例跑在【子进程】里。本进程内 test_factors_price / test_factors_fundamental 已经
  `from ashare.factors import price`，注册表早就满了 —— 在本进程断言 len == 18 的话，
  把 __init__.py 的四行 import 全删了照样绿。整个平台最便宜的一道保险，必须在干净解释器里验。
"""
from __future__ import annotations
import datetime as dt
import inspect
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, query
from ashare.factors import flow as ffl
from ashare.factors import pipeline, risk as frk

D = dt.date
AS_OF = "2024-01-05"
CODES = ["S00001.SZ", "S00002.SZ"]
MKT = "000985.CSI"                      # 中证全指


# ══════════════ 构造面板 / 打桩 query 的脚手架 ══════════════
def _dates(n: int) -> list[dt.date]:
    """n 个"交易日"（连续自然日即可：因子按标签对齐，只要两侧用同一套日期）。"""
    return [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def _panel(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=_dates(n)).rename_axis("trade_date")


def _mkt_path(n: int, seed: int = 7) -> tuple[np.ndarray, pd.Series]:
    """返回 (指数日对数收益, 指数收盘价 Series)。

    用【随机】游走而不是等比路径：等比路径每天收益相同，按位置错位对齐多少天
    都能算出同一个 beta，那条错法就测不出来了。
    """
    r = np.random.default_rng(seed).normal(0.0, 0.012, n)
    r[0] = 0.0
    return r, pd.Series(3000.0 * np.exp(np.cumsum(r)), index=_dates(n)).rename_axis("trade_date")


def _stock(r: np.ndarray, beta: float, base: float = 100.0) -> np.ndarray:
    """日对数收益恰为 beta × 指数收益的价格路径 → 真 beta 就是 beta（无残差，闭式精确）。"""
    return base * np.exp(beta * np.cumsum(r))


def _patch_prices(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    """替换 get_price_panel。★ 停牌日给 NaN 且【不 ffill】—— 真实 get_bars 就是这样
    （is_suspended 的行价格被掩成 NaN），要不要填由因子自己决定，这正是被测行为。"""
    def _fake(as_of_date, ts_codes, field="close", lookback=250, adjust="hfq"):
        assert field == "close", f"beta 只该取收盘价，实际取了 {field!r}"
        assert adjust == "hfq", "D8：后复权是唯一真值"
        if seen is not None:
            seen["px_lookback"] = lookback
        return panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
    monkeypatch.setattr(query, "get_price_panel", _fake)


def _patch_index(monkeypatch, closes: pd.Series, seen: dict | None = None) -> None:
    def _fake(as_of_date, index_code, lookback=250, fields=("close", "pe_ttm")):
        assert "close" in fields
        if seen is not None:
            seen["index_code"] = index_code
            seen["idx_lookback"] = lookback
        return closes.iloc[-lookback:].to_frame("close")
    monkeypatch.setattr(query, "get_index_bars", _fake)


def _patch_money(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    """替换 get_money_flow。★ 保留 NaN 行：真实实现按交易日历 reindex，缺的日子
    是一行 NaN 而不是"整行缺席"（与 get_bars 相反）。"""
    def _fake(as_of_date, ts_codes, fields=("hk_hold_ratio",), lookback=20):
        assert "hk_hold_ratio" in fields, f"北向因子应取 hk_hold_ratio，实际 {fields}"
        if seen is not None:
            seen["lookback"] = lookback
        sub = panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
        out = sub.stack(future_stack=True).rename("hk_hold_ratio").to_frame()
        out.index = out.index.set_names(["trade_date", "ts_code"])
        return out.reorder_levels(["ts_code", "trade_date"]).sort_index()
    monkeypatch.setattr(query, "get_money_flow", _fake)


def _patch_basic(monkeypatch, mv: dict[str, float]) -> None:
    """替换 get_daily_basic（lookback=1 → index=ts_code）。mv 里没有的代码【整行缺席】。"""
    def _fake(as_of_date, ts_codes, fields=("total_mv",), lookback=1):
        assert "total_mv" in fields and lookback == 1
        codes = [c for c in ts_codes if c in mv]
        return pd.DataFrame({"total_mv": [mv[c] for c in codes]},
                            index=pd.Index(codes, name="ts_code"))
    monkeypatch.setattr(query, "get_daily_basic", _fake)


def _patch_industry(monkeypatch, ind: dict[str, str]) -> None:
    def _fake(as_of_date, ts_codes=None, level="l1", *, min_members=5):
        codes = [c for c in (ts_codes if ts_codes is not None else ind) if c in ind]
        return pd.Series([ind[c] for c in codes],
                         index=pd.Index(codes, name="ts_code"), name="sw_l1")
    monkeypatch.setattr(query, "get_industry", _fake)


# ══════════════ 1 north_hold_chg_20 ══════════════
def test_north_hold_chg_20_is_the_difference_of_the_two_endpoints(monkeypatch):
    """每日 +0.01 个百分点，20 日变化 = 0.20（取【差】不取【比】：持股比例基数极小，
    0.01% → 0.02% 是"翻倍"但毫无意义，用比值会把最不被北向关注的票排在最前）。"""
    ramp = np.arange(60) * 0.01
    _patch_money(monkeypatch, _panel({c: ramp.copy() for c in CODES}))
    out = ffl.north_hold_chg_20(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 0.20, rtol=1e-12)


def test_north_hold_chg_20_window_is_exactly_20_not_19_or_21(monkeypatch):
    """只有 ratio_t 与 ratio_{t−20} 进公式：改 t−21 不动结果，改 t−20 一定动。"""
    ramp = np.arange(60) * 0.01
    _patch_money(monkeypatch, _panel({c: ramp.copy() for c in CODES}))
    base = ffl.north_hold_chg_20(AS_OF, CODES).iloc[0]

    far = ramp.copy(); far[-22] += 5.0                          # t−21，窗口外
    _patch_money(monkeypatch, _panel({c: far for c in CODES}))
    assert ffl.north_hold_chg_20(AS_OF, CODES).iloc[0] == pytest.approx(base)

    edge = ramp.copy(); edge[-21] += 5.0                        # t−20，窗口左端点
    _patch_money(monkeypatch, _panel({c: edge for c in CODES}))
    assert ffl.north_hold_chg_20(AS_OF, CODES).iloc[0] == pytest.approx(base - 5.0)


def test_north_hold_chg_20_no_data_is_nan_never_zero(monkeypatch):
    """★ 本任务最重要的一条：完全没有北向数据的股票必须是 NaN，不是 0.0。

    0 是这个因子的【合法取值】（持股比例没变），所以填 0 不会被任何形状/类型检查逮到。
    而 combine 会把"有值"的因子算进分母 —— 分母算了它、分子恒等于同一个常数，
    整段没有北向数据的历史（2010–2016 是全部 A 股，2016 之后是所有非通道标的）
    就被静默降权，净值曲线照画不误。
    """
    dead = np.full(60, np.nan)
    _patch_money(monkeypatch, _panel({"S00001.SZ": np.arange(60) * 0.01, "S00002.SZ": dead}))
    out = ffl.north_hold_chg_20(AS_OF, CODES)
    assert np.isnan(out["S00002.SZ"]), f"无数据被填成了 {out['S00002.SZ']!r}"
    assert out["S00002.SZ"] != 0.0 or np.isnan(out["S00002.SZ"])
    assert out.notna()["S00001.SZ"], "同一横截面里有数据的股票不应被连坐"


def test_north_hold_chg_20_ffills_a_gap_instead_of_returning_nan(monkeypatch):
    """窗口内偶发缺行（停牌 / 数据洞）：持股比例在停牌期间本来就不变，前值就是真值。"""
    v = np.arange(60) * 0.01
    holed = v.copy()
    holed[-1] = np.nan                                          # t 日缺
    holed[-21] = np.nan                                         # t−20 日也缺
    _patch_money(monkeypatch, _panel({"S00001.SZ": holed, "S00002.SZ": v}))
    out = ffl.north_hold_chg_20(AS_OF, CODES)
    assert out.notna().all(), "窗口内非空天数充足，偶发缺行应被 ffill 补上"
    np.testing.assert_allclose(out["S00001.SZ"], 0.20, rtol=1e-12)   # 两端各前移一天，仍是 20 步


@pytest.mark.parametrize("n_real,expect_nan", [(13, False), (12, True)])
def test_north_hold_chg_20_needs_60pct_real_observations(monkeypatch, n_real, expect_nan):
    """窗口 21 天 × 60% = 12.6 → 13 天真实数据才算数（与 price.py 同一口径）。
    12 天靠 ffill 硬撑出来的"20 日变化"是噪声。"""
    v = np.arange(60) * 0.01
    thin = v.copy()
    thin[-21:] = np.nan
    keep = np.linspace(-21, -1, n_real).astype(int)
    thin[keep] = v[keep]
    _patch_money(monkeypatch, _panel({"S00001.SZ": thin, "S00002.SZ": v}))
    out = ffl.north_hold_chg_20(AS_OF, CODES)
    assert bool(np.isnan(out["S00001.SZ"])) is expect_nan
    assert out.notna()["S00002.SZ"]


def test_north_hold_chg_20_history_shorter_than_the_window_is_nan(monkeypatch):
    """回测头几天取不满一个窗口 —— 全 NaN，不是 IndexError，也不是拿 5 天硬算。"""
    _patch_money(monkeypatch, _panel({c: np.arange(5) * 0.01 for c in CODES}))
    out = ffl.north_hold_chg_20(AS_OF, CODES)
    assert list(out.index) == CODES and out.isna().all()


# ══════════════ 2 log_mv ══════════════
def test_log_mv_is_the_natural_log_of_total_mv(monkeypatch):
    _patch_basic(monkeypatch, {"S00001.SZ": 1e6, "S00002.SZ": 2.5e5})
    out = frk.log_mv(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), [math.log(1e6), math.log(2.5e5)], rtol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_log_mv_non_positive_market_cap_is_nan_not_minus_inf(monkeypatch, bad):
    """ln(0) = −inf。一个 −inf 进 neutralize 的设计矩阵，lstsq 整列返回 nan ——
    那一天【全池】的因子作废，而报出来的样子是"这天没有信号"。"""
    _patch_basic(monkeypatch, {"S00001.SZ": bad, "S00002.SZ": 1e6})
    out = frk.log_mv(AS_OF, CODES)
    assert np.isnan(out["S00001.SZ"]) and np.isfinite(out["S00002.SZ"])


def test_log_mv_missing_row_stays_in_the_index_as_nan(monkeypatch):
    """池内某股当日无 daily_basic 行（停牌 / 新股）→ NaN，但仍在 index 里。"""
    _patch_basic(monkeypatch, {"S00001.SZ": 1e6})
    out = frk.log_mv(AS_OF, CODES)
    assert list(out.index) == CODES and np.isnan(out["S00002.SZ"])


# ══════════════ 3 industry ══════════════
def test_industry_returns_the_sw_l1_label_per_stock(monkeypatch):
    _patch_industry(monkeypatch, {"S00001.SZ": "银行", "S00002.SZ": "食品饮料"})
    out = frk.industry(AS_OF, CODES)
    assert list(out) == ["银行", "食品饮料"]
    assert list(out.index) == CODES


def test_industry_is_categorical_not_numeric_and_blows_up_in_the_alpha_chain(monkeypatch):
    """★ industry 不是数值因子、不进 combine。注册表里没有 is_alpha 字段，
    所以这里再上一道结构性保险：返回 category dtype，pipeline 的第一步 winsorize_mad
    要算 median()，在 category 上【当场 TypeError】。
    误用会炸，而不是产出一列"看起来是分数"的东西。
    """
    _patch_industry(monkeypatch, {"S00001.SZ": "银行", "S00002.SZ": "食品饮料"})
    out = frk.industry(AS_OF, CODES)
    assert isinstance(out.dtype, pd.CategoricalDtype), "必须是 category dtype"
    assert not pd.api.types.is_numeric_dtype(out)
    with pytest.raises(TypeError):
        pipeline.winsorize_mad(out)


def test_industry_unknown_code_is_nan_not_dropped(monkeypatch):
    """池内某股在 stock_basic 里根本没有行 → NaN 但保留 index：
    neutralize 按 universe 做横截面回归，少一个标签会静默错位。"""
    _patch_industry(monkeypatch, {"S00001.SZ": "银行"})
    out = frk.industry(AS_OF, CODES)
    assert list(out.index) == CODES and pd.isna(out["S00002.SZ"])


# ══════════════ 4 beta_250 ══════════════
@pytest.mark.parametrize("true_beta", [1.0, 2.0, 0.5])
def test_beta_250_recovers_the_true_beta_exactly(monkeypatch, true_beta):
    """个股日对数收益 = β × 指数日对数收益（无残差）→ Cov/Var 恰为 β。

    β=2 与 β=0.5 一起测：分子分母写反会把 2 变成 0.5，两个用例互为对方的反例。
    """
    r, idx = _mkt_path(300)
    _patch_prices(monkeypatch, _panel({c: _stock(r, true_beta) for c in CODES}))
    _patch_index(monkeypatch, idx)
    out = frk.beta_250(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), true_beta, rtol=1e-9)


def test_beta_250_uses_the_csi_all_share_index(monkeypatch):
    seen: dict = {}
    r, idx = _mkt_path(300)
    _patch_prices(monkeypatch, _panel({c: _stock(r, 1.0) for c in CODES}))
    _patch_index(monkeypatch, idx, seen)
    frk.beta_250(AS_OF, CODES)
    assert seen["index_code"] == MKT, "对标指数必须是中证全指（000985.CSI）"


def test_beta_250_suspended_days_are_dropped_not_treated_as_zero_returns(monkeypatch):
    """★ D9：停牌占位行在 query 层已被掩成 NaN。ffill 会把它还原成一个【假的 0 收益】——
    "大盘在动、这只票没动"从来不是一个观测，它根本没有观测。

    个股在有效日上严格是 2 倍波动，停牌日为 NaN。正确实现（丢掉停牌日 + 复牌当日）
    仍得到恰好 2.0；一旦 ffill，停牌日贡献 (0, r_mkt)、复牌日贡献 (跨期累计, 单日 r_mkt)，
    beta 被拉离 2 —— 方向是偏向 0，于是停牌越多的股票中性化时被少扣了市场暴露。
    """
    r, idx = _mkt_path(300)
    px = _stock(r, 2.0)
    halted = px.copy()
    halted[-60:-40] = np.nan                                    # 窗口内 20 天停牌
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    _patch_index(monkeypatch, idx)
    out = frk.beta_250(AS_OF, CODES)
    np.testing.assert_allclose(out["S00001.SZ"], 2.0, rtol=1e-9)
    np.testing.assert_allclose(out["S00002.SZ"], 2.0, rtol=1e-9)


def test_beta_250_market_variance_is_measured_on_the_same_valid_subset(monkeypatch):
    """分母的 Var(r_mkt) 必须在【每只股票自己的】有效子集上算。
    用全窗口方差配一个只剩 160 天的协方差，就是两个不同样本的比值 —— 停牌多的股票 beta 偏。

    构造：指数在个股停牌的那段特别剧烈（方差集中在被丢掉的日子里）。全窗口方差
    会明显大于有效子集方差，β 因此被压小；只有同一子集才还得回 2.0。
    """
    r, _ = _mkt_path(300)
    r[-60:-40] *= 8.0                                           # 被丢掉的那 20 天大幅放大
    idx = pd.Series(3000.0 * np.exp(np.cumsum(r)), index=_dates(300)).rename_axis("trade_date")
    px = _stock(r, 2.0)
    halted = px.copy()
    halted[-60:-40] = np.nan
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    _patch_index(monkeypatch, idx)
    np.testing.assert_allclose(frk.beta_250(AS_OF, CODES)["S00001.SZ"], 2.0, rtol=1e-9)


@pytest.mark.parametrize("n_gap,expect_nan", [(99, False), (100, True)])
def test_beta_250_needs_150_valid_observations(monkeypatch, n_gap, expect_nan):
    """251 根收盘价 → 250 个日收益；窗口内连续 n_gap 天停牌会毁掉 n_gap+1 个收益
    （停牌当日 + 复牌当日的前收是 NaN）。
    250 − 100 = 150 → 算；250 − 101 = 149 → NaN。ceil(0.60 × 250) = 150。"""
    r, idx = _mkt_path(300)
    px = _stock(r, 1.5)
    halted = px.copy()
    halted[-200:-200 + n_gap] = np.nan
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    _patch_index(monkeypatch, idx)
    out = frk.beta_250(AS_OF, CODES)
    assert bool(np.isnan(out["S00001.SZ"])) is expect_nan
    if not expect_nan:
        np.testing.assert_allclose(out["S00001.SZ"], 1.5, rtol=1e-9)
    assert out.notna()["S00002.SZ"], "同一横截面里数据完整的股票不应被连坐"


def test_beta_250_flat_market_is_nan_not_infinite(monkeypatch):
    """Var(r_mkt) = 0（指数一动不动）→ NaN。±inf 连 MAD 去极值都拦不住。"""
    flat = pd.Series(np.full(300, 3000.0), index=_dates(300)).rename_axis("trade_date")
    _patch_prices(monkeypatch, _panel({c: _stock(_mkt_path(300)[0], 1.0) for c in CODES}))
    _patch_index(monkeypatch, flat)
    out = frk.beta_250(AS_OF, CODES)
    assert out.isna().all(), f"零方差市场给出了 {out.to_dict()}"


def test_beta_250_aligns_on_dates_not_on_positions(monkeypatch):
    """★ 指数与个股的日期集合【不一样】时必须按标签对齐。

    真实里这一定会发生：`index_daily` 与 `daily_bar` 是两次入库，覆盖区间和数据洞
    互不保证一致，两个 DataFrame 行数相近而日期不同。按位置 zip（典型写法：两边
    `.to_numpy()` 之后直接相乘）不会抛，只会把个股 t 日的收益配到指数 t−5 日的收益上，
    随机路径下 beta 立刻塌成噪声 —— 而"beta 都接近 0"看起来只像"这批股票很防御"。

    构造：指数返回 [44, 294]、个股面板返回 [49, 299]，两边都是 251 行但错开 5 天。
    """
    r, idx = _mkt_path(300)
    _patch_prices(monkeypatch, _panel({c: _stock(r, 2.0) for c in CODES}))
    shifted = idx.iloc[44:295]                                  # 251 行，但比个股早 5 天
    monkeypatch.setattr(query, "get_index_bars",
                        lambda *a, **k: shifted.to_frame("close"))
    out = frk.beta_250(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 2.0, rtol=1e-9)


def test_beta_250_requests_251_closes_for_250_returns(monkeypatch):
    """250 个日收益需要 251 根收盘价。少取一根就只有 249 个收益，覆盖率闸门跟着漂。"""
    seen: dict = {}
    r, idx = _mkt_path(300)
    _patch_prices(monkeypatch, _panel({c: _stock(r, 1.0) for c in CODES}), seen)
    _patch_index(monkeypatch, idx, seen)
    frk.beta_250(AS_OF, CODES)
    assert seen["px_lookback"] == 251 and seen["idx_lookback"] == 251


# ══════════════ 契约：签名 / index / 空池 / 注册元数据 ══════════════
_FNS = {"north_hold_chg_20": ffl.north_hold_chg_20, "log_mv": frk.log_mv,
        "industry": frk.industry, "beta_250": frk.beta_250}


def _patch_everything(monkeypatch, codes: list[str], n: int = 300) -> None:
    r, idx = _mkt_path(n)
    _patch_prices(monkeypatch, _panel({c: _stock(r, 1.0) for c in codes}))
    _patch_index(monkeypatch, idx)
    _patch_money(monkeypatch, _panel({c: np.arange(n) * 0.01 for c in codes}))
    _patch_basic(monkeypatch, {c: 1e6 for c in codes})
    _patch_industry(monkeypatch, {c: "银行" for c in codes})


@pytest.mark.parametrize("name", sorted(_FNS))
def test_first_two_positional_params_are_as_of_date_and_universe(name):
    """L3 静态检查之外再钉一次：装饰器原样返回函数，签名就是运行期真签名。"""
    params = list(inspect.signature(_FNS[name]).parameters.values())
    assert [p.name for p in params[:2]] == ["as_of_date", "universe"]
    assert all(p.kind is p.KEYWORD_ONLY for p in params[2:]), "窗口参数必须是 keyword-only"


@pytest.mark.parametrize("name", sorted(_FNS))
def test_index_is_exactly_universe_in_universe_order(monkeypatch, name):
    """故意用非字典序的 universe：unstack / groupby 会悄悄把 index 排成字典序，
    下游按标签对齐所以不会炸，但任何按位置读这个 Series 的代码都会静默错配。"""
    unordered = ["S00002.SZ", "S00001.SZ"]
    _patch_everything(monkeypatch, unordered)
    out = _FNS[name](AS_OF, unordered)
    assert isinstance(out, pd.Series)
    assert list(out.index) == unordered, f"{name} 的 index 不等于 universe（或顺序变了）"


@pytest.mark.parametrize("name", sorted(_FNS))
def test_empty_universe_returns_an_empty_series(monkeypatch, name):
    """某天全池被停牌 / ST 剔光 —— 不该抛。"""
    _patch_everything(monkeypatch, CODES)
    out = _FNS[name](AS_OF, [])
    assert isinstance(out, pd.Series) and len(out) == 0


@pytest.mark.parametrize("name,direction,category,lookback,neutralize,available_from", [
    ("north_hold_chg_20", 1, "flow", 31, True, D(2016, 12, 5)),
    ("log_mv", -1, "risk", 1, False, None),
    ("industry", -1, "risk", 1, False, None),
    ("beta_250", -1, "risk", 260, False, None),
])
def test_registered_metadata_matches_the_spec_table(name, direction, category, lookback,
                                                    neutralize, available_from):
    from ashare.factors.base import get_factor
    spec = get_factor(name)
    assert (spec.direction, spec.category, spec.lookback_days) == (direction, category, lookback)
    assert spec.neutralize is neutralize
    assert spec.available_from == available_from
    assert spec.fn is _FNS[name]


@pytest.mark.parametrize("name", ["log_mv", "industry", "beta_250"])
def test_risk_factors_are_never_neutralized(name):
    """★ 这三个【就是】neutralize 的回归元。对自己回归的残差恒等于 0（数值上是浮点噪声），
    zscore 再把它放大 1e16 倍 —— 不抛、不告警，产出一列纯舍入误差冒充的因子。"""
    from ashare.factors.base import get_factor
    assert get_factor(name).neutralize is False
    assert get_factor(name).category == "risk"


def test_north_hold_chg_20_is_alpha_and_is_neutralized():
    """反向锚：资金因子【是】alpha，必须中性化 —— 免得有人把 neutralize=False 抄串行。"""
    from ashare.factors.base import get_factor
    assert get_factor("north_hold_chg_20").neutralize is True


# ══════════════ ★ 收口：注册表装配（子进程，干净解释器）══════════════
_EXPECTED_COUNTS = {"total": 18, "price": 6, "fundamental": 8, "flow": 1, "risk": 3}


def test_importing_ashare_factors_alone_registers_every_factor():
    """★ `import ashare.factors` 之后注册表必须是满的。

    @factor 只在模块【被导入时】注册。__init__.py 少 import 一个模块 → 那一类因子
    在注册表里根本不存在，而空/缺因子是【静默失败】：combine 拿不到值 → 合成分数全 NaN
    → build_targets 返回空 → 净值一条直线。读起来像"策略这段时间没信号"，
    而不是"代码没装配"。

    必须开子进程：本进程里别的测试模块早就直接 import 过 price / fundamental 了，
    在这里断言等于把最需要保护的那件事测掉。
    """
    script = ("import json, ashare.factors as f;"
              "print(json.dumps({'total': len(f.FACTOR_REGISTRY), "
              "**{c: len(f.list_factors(c)) for c in ('price','fundamental','flow','risk')}}))")
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(root),
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"干净解释器里 import ashare.factors 就失败了：\n{proc.stderr}"
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == _EXPECTED_COUNTS


def test_factor_names_are_unique_across_modules():
    """重名在 base.factor 里会 raise —— 但只有四个模块【都】被导入过才触发得到。
    上一条保证了导入，这条保证 18 个名字真的是 18 个不同的因子。"""
    import ashare.factors as f
    assert len(set(f.FACTOR_REGISTRY)) == _EXPECTED_COUNTS["total"]
    assert {"north_hold_chg_20", "log_mv", "industry", "beta_250"} <= set(f.FACTOR_REGISTRY)


# ══════════════ 真实取数链路（不打桩，直接读 fixture 库）══════════════
def _extend_calendar(w, start: dt.date, end: dt.date) -> None:
    """把日历补到 [start, end]（工作日开市）。market_db 自带 2023-12-25~2024-02-02
    的真实片段（含元旦休市），这里只补它没有的日子，不覆盖。"""
    have = {r[0] for r in w.execute("SELECT trade_date FROM calendar").fetchall()}
    rows, d = [], start
    while d <= end:
        if d not in have:
            rows.append((d, d.weekday() < 5, None))
        d += dt.timedelta(days=1)
    w.executemany("INSERT INTO calendar VALUES (?, ?, ?)", rows)


def _open_days(w, start: dt.date, end: dt.date) -> list[dt.date]:
    return [r[0] for r in w.execute(
        "SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", [start, end]).fetchall()]


@pytest.fixture
def flow_db(market_db):
    """market_db + 日历回补到 2015 + 【只有 2017 年】的 hk_hold_ratio。

    2016-12-05 之前一行都没有 —— 真实库就是这个样子（沪深港通个股持股披露的起点）。
    A00001.SZ 有数据；B00002.SZ 从来不是通道标的，一行都没有。
    """
    w = _db.connect_write(market_db)
    _extend_calendar(w, D(2015, 1, 1), D(2017, 12, 31))
    days = _open_days(w, D(2017, 1, 1), D(2017, 12, 31))
    w.executemany("INSERT INTO money_flow (ts_code, trade_date, hk_hold_ratio) VALUES (?, ?, ?)",
                  [("A00001.SZ", d, 1.0 + 0.01 * i) for i, d in enumerate(days)])
    w.close()
    query.open_db(market_db)
    yield query
    query.close_db()


_FLOW_CODES = ["A00001.SZ", "B00002.SZ"]


def test_north_hold_chg_20_before_2016_12_05_is_all_nan_end_to_end(flow_db):
    """★ available_from 那条决策的【端到端】验证：2015 年真实地跑一遍取数链路，
    结果必须是全 NaN 且不抛 —— 而不是 0.0、不是空 Series、不是 AsOfDateError。

    Task 7 的 compute_factor 会在 available_from 之前直接短路返回全 NaN（省一次取数）；
    这里验的是【短路之外】的那条路也是对的 —— 否则 store.build / 任何直接调 fn 的地方
    都会拿到 0，而 0 在本因子里是合法值，没人看得出来。
    """
    out = ffl.north_hold_chg_20("2015-06-12", _FLOW_CODES)
    assert list(out.index) == _FLOW_CODES
    assert out.isna().all(), f"北向数据存在之前算出了值：{out.to_dict()}"
    assert out.dtype == float, "全 NaN 也必须是 float dtype，object 会在下游静默变味"
    assert not (out == 0).any()


def test_north_hold_chg_20_on_real_market_db_after_2017(flow_db):
    """反向锚：同一条链路在数据可得之后必须算得出数 —— 否则上一条"全 NaN"是空证。
    A 的持股比例每交易日 +0.01 → 20 日变化 0.20；B 从来不是通道标的 → NaN，不是 0。"""
    out = ffl.north_hold_chg_20("2017-06-30", _FLOW_CODES)
    np.testing.assert_allclose(out["A00001.SZ"], 0.20, rtol=1e-9)
    assert np.isnan(out["B00002.SZ"]), "非通道标的被填成了 0/有限值"


def _fixture_hfq(base: float, i: int) -> float:
    """market_db 第 i 个交易日的后复权价：原始价 base+0.1i，复权因子 1+0.01i。"""
    return (base + 0.1 * i) * (1.0 + 0.01 * i)


@pytest.fixture
def beta_db(market_db):
    """market_db + 000985.CSI 指数日线，收盘价 == B00002.SZ 的【后复权】价。

    B 因此与指数逐日同步 → beta 恰为 1.0。任何一处错（按位置对齐、用了未复权价、
    收益口径写错）都会让这个 1.0 明显偏掉。A 在 01-15~01-17 停牌（真实的 vol=0 占位行）。
    """
    w = _db.connect_write(market_db)
    days = _open_days(w, D(2023, 1, 1), D(2024, 12, 31))
    w.executemany("INSERT INTO index_daily (ts_code, trade_date, close) VALUES (?, ?, ?)",
                  [(MKT, d, _fixture_hfq(20.0, i)) for i, d in enumerate(days)])
    w.close()
    query.open_db(market_db)
    yield query
    query.close_db()


def test_beta_250_on_real_market_db_tracks_the_index_exactly(beta_db):
    """fixture 只有 29 个交易日，用 window=20 跑完整链路（闸门 = ceil(0.6×20) = 12）。
    B 的后复权价【就是】指数收盘价 → beta 恰为 1。"""
    out = frk.beta_250("2024-02-02", ["A00001.SZ", "B00002.SZ"], window=20)
    np.testing.assert_allclose(out["B00002.SZ"], 1.0, rtol=1e-9)


def test_beta_250_on_real_market_db_excludes_the_suspended_days(beta_db):
    """★ D9 端到端：A 在 01-15~01-17 有三条真实的停牌占位行（vol=0、OHLC=前收）。
    query 层把它们的价格掩成 NaN，本因子不 ffill → 那 3 天 + 复牌当日共 4 个收益作废。

    预期值用 numpy 在【有效子集】上独立算一遍（与实现是两条不同的路径）；
    同时算出 ffill 版本的值做反例 —— 两者必须可区分，否则这条用例什么都没钉住。
    """
    i = np.arange(8, 29)                                        # window=20 → 21 根收盘价
    a = np.array([_fixture_hfq(10.0, k) for k in i])
    m = np.array([_fixture_hfq(20.0, k) for k in i])
    a_nan = a.copy()
    a_nan[[6, 7, 8]] = np.nan                                   # i = 14/15/16 → 位置 6/7/8
    ra, rm = np.diff(np.log(a_nan)), np.diff(np.log(m))
    ok = ~np.isnan(ra)
    assert ok.sum() == 16
    expect = np.cov(ra[ok], rm[ok], ddof=1)[0, 1] / np.var(rm[ok], ddof=1)

    ra_ffill = np.diff(np.log(pd.Series(a_nan).ffill().to_numpy()))
    wrong = np.cov(ra_ffill, rm, ddof=1)[0, 1] / np.var(rm, ddof=1)
    assert expect != pytest.approx(wrong, rel=1e-6), "fixture 分辨不出 ffill —— 用例失效"

    got = frk.beta_250("2024-02-02", ["A00001.SZ", "B00002.SZ"], window=20)["A00001.SZ"]
    np.testing.assert_allclose(got, expect, rtol=1e-9)
    assert got != pytest.approx(wrong, rel=1e-6), "停牌日被 ffill 成了假的 0 收益"


def test_industry_on_real_market_db(beta_db):
    """真实链路：industry_member 的 PIT 区间 + get_industry 的小行业合并（min_members=5，
    fixture 里每个行业都不足 5 家 → 全部并入 __OTHER__，这正是它该有的行为）。"""
    out = frk.industry("2024-01-25", ["A00001.SZ", "B00002.SZ"])
    assert list(out.index) == ["A00001.SZ", "B00002.SZ"]
    assert isinstance(out.dtype, pd.CategoricalDtype)
    assert set(out) == {"__OTHER__"}


def test_log_mv_on_real_market_db(beta_db):
    """market_db 的 total_mv：A=1e6, B=5e5。"""
    out = frk.log_mv("2024-01-25", ["A00001.SZ", "B00002.SZ"])
    np.testing.assert_allclose(out.to_numpy(), [math.log(1e6), math.log(5e5)], rtol=1e-9)
