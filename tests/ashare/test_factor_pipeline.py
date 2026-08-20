"""Task 3：因子处理链 —— MAD 去极值 → 行业+市值 WLS 中性化 → zscore → fillna(0)。

四步顺序不可调换（算法说明书 §3）。本文件的每个用例都钉住一条"为什么是这样"，
而不只是"跑通了"：
  · MAD vs 3σ      —— 同一份数据下 3σ 截不干净
  · WLS vs OLS     —— 大盘残差在 OLS 下系统性偏移，在 WLS 下接近 0
  · industry_source —— 降级的行业标签做中性化 = 前视污染，必须抛
  · 秩亏 / 样本不足 —— 返回原值 + warning，不静默返回 NaN
  · fillna 在 zscore 之后 —— 提前填 0 会拉动均值、污染其他所有股票的分数

中性化的数值检验用【构造的横截面】（monkeypatch query 的三个取数函数）：
WLS 的判别需要 100 只股票的市值分层，market_db 只有 4 只。真实取数链路由
test_neutralize_on_real_market_db 这一条覆盖（不打桩，直接读 fixture 库）。
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from ashare.data import query
from ashare.factors.base import FactorSpec
from ashare.factors import pipeline

D = dt.date
AS_OF = "2024-01-05"


# ══════════════ 构造横截面的脚手架 ══════════════
def _codes(n: int) -> list[str]:
    return [f"S{i:05d}.SZ" for i in range(n)]


def _patch_cross_section(monkeypatch, mv: pd.Series, ind: pd.Series | None = None,
                         source: str = "sw") -> None:
    """把 query 的三个取数函数换成给定的横截面。pipeline 在【调用时】才解析 query.xxx，
    所以打在 query 模块上即可，pipeline 的 import 路径保持真实。"""
    def _daily_basic(as_of_date, ts_codes, fields=("total_mv",), lookback=1):
        return pd.DataFrame({"total_mv": mv.reindex(list(ts_codes))})
    def _industry(as_of_date, ts_codes=None, level="l1", *, min_members=5):
        assert ind is not None, "本用例没有准备行业数据，说明被测路径不该取行业"
        return ind.reindex(list(ts_codes))
    monkeypatch.setattr(query, "get_daily_basic", _daily_basic)
    monkeypatch.setattr(query, "get_industry", _industry)
    monkeypatch.setattr(query, "industry_source", lambda: source)


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """加权相关系数。WLS 的正交性是【加权】意义下的：Σ wᵢεᵢxᵢ = 0，
    普通相关系数对 WLS 残差本来就不为 0（见 test_neutralize_uses_wls_not_ols 的解释）。"""
    ca, cb = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(ca * cb, weights=w) /
                 np.sqrt(np.average(ca * ca, weights=w) * np.average(cb * cb, weights=w)))


# ══════════════ 3.1 MAD 去极值 ══════════════
def test_mad_clips_the_outlier_exactly_to_the_bound():
    s = pd.Series(1.0 + 0.01 * np.arange(50), index=_codes(50))
    s.iloc[0] = 1000.0
    out = pipeline.winsorize_mad(s)

    m, mad = s.median(), (s - s.median()).abs().median()
    hi = m + 3 * 1.4826 * mad
    assert out.iloc[0] == pytest.approx(hi)                 # 极值恰好落在上界
    assert out.iloc[1:].equals(s.iloc[1:])                  # 正常值一个都没动


def test_three_sigma_does_not_clip_what_mad_clips():
    """钉住"为什么不用 3σ"：均值和标准差本身已被极值污染，3σ 的上界仍在天上。"""
    s = pd.Series(1.0 + 0.01 * np.arange(50), index=_codes(50))
    s.iloc[0] = 1000.0
    body_max = s.iloc[1:].max()                             # 正常样本的最大值 ≈ 1.49

    mad_max = pipeline.winsorize_mad(s).max()
    mu, sd = s.mean(), s.std()
    sigma_max = s.clip(mu - 3 * sd, mu + 3 * sd).max()

    assert mad_max < 2 * body_max                           # MAD：截到了正常量级
    assert sigma_max > 100 * body_max                       # 3σ：还差两个数量级（实测 ≈ 445）
    assert sigma_max > 100 * mad_max


def test_mad_leaves_nan_as_nan():
    """NaN 必须活着穿过第一步 —— fillna 只允许发生在链条末端。"""
    s = pd.Series([1.0, 2.0, np.nan, 3.0, 100.0], index=_codes(5))
    out = pipeline.winsorize_mad(s)
    assert np.isnan(out.iloc[2])
    assert out.notna().sum() == 4


def test_mad_with_zero_dispersion_returns_input_unchanged():
    """过半数取值相同 → MAD=0，上下界collapse 到中位数；照着截会把整个因子拍平。"""
    s = pd.Series([5.0] * 30 + [1.0, 9.0], index=_codes(32))
    assert pipeline.winsorize_mad(s).equals(s)


def test_mad_n_parameter_is_honoured():
    s = pd.Series(np.arange(41.0), index=_codes(41))
    assert pipeline.winsorize_mad(s, n=1.0).max() < pipeline.winsorize_mad(s, n=3.0).max()


# ══════════════ 3.3 标准化 ══════════════
def test_zscore_is_mean_zero_std_one():
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0], index=_codes(5))
    z = pipeline.zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0)


def test_zscore_of_a_constant_cross_section_is_nan_not_inf():
    """std=0 → 0/0 = NaN（不是 inf），末端 fillna(0) 正好把它变成"中性"。"""
    z = pipeline.zscore(pd.Series([3.0] * 10, index=_codes(10)))
    assert z.isna().all()


# ══════════════ 3.2 中性化 ══════════════
@pytest.fixture
def cross_section():
    """120 只股票：市值跨 5e8 ~ 1e12，6 个行业，因子值同时含市值暴露与行业暴露。"""
    rng = np.random.default_rng(11)
    n = 120
    codes = _codes(n)
    mv = pd.Series(np.exp(rng.uniform(np.log(5e8), np.log(1e12), n)), index=codes)
    ind = pd.Series([f"IND{i % 6}" for i in range(n)], index=codes)
    s = pd.Series(rng.normal(size=n) + 0.3 * np.log(mv.to_numpy())
                  + np.array([int(k[-1]) for k in ind]) * 0.4, index=codes)
    return s, mv, ind, codes


def test_neutralized_residual_carries_no_size_or_industry_exposure(monkeypatch, cross_section):
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind)

    resid, warns = pipeline.neutralize(s, AS_OF, codes)
    assert warns == []

    w = np.sqrt(mv.to_numpy())
    r, x = resid.to_numpy(), np.log(mv.to_numpy())
    assert abs(_wcorr(r, x, w)) < 1e-10                     # 残差不再含市值暴露
    for k in sorted(set(ind)):                              # 每个行业（含被丢掉的基准行业）
        m = (ind == k).to_numpy()
        assert abs(np.average(r[m], weights=w[m])) < 1e-9


def test_neutralize_uses_wls_not_ols(monkeypatch):
    """WLS ≠ OLS 的钉子。

    90 只小盘 + 10 只大盘，因子值是 log 市值的【单调但凹】函数（市值效应在头部饱和）——
    一条直线拟合不了两端。OLS 被 90 只小盘主导，大盘组整体掉在回归线一侧；
    sqrt(MV) 加权后大盘占了约 68% 的权重，WLS 的线穿过大盘组，其残差均值回到 0 附近。

    注：若因子值与 log 市值【严格线性】，OLS 与 WLS 的残差都恒为 0，这个用例就退化成
    永远通过 —— 判别 WLS 必须要有直线拟合不了的成分。
    """
    n_s, n_l = 90, 10
    codes = _codes(n_s + n_l)
    mv = pd.Series(np.concatenate([
        np.exp(np.linspace(np.log(1e9), np.log(1e10), n_s)),
        np.exp(np.linspace(np.log(8e11), np.log(2e12), n_l))]), index=codes)
    x = np.log(mv.to_numpy())
    s = pd.Series(np.log(x - 20.0), index=codes)
    big = np.arange(n_s + n_l) >= n_s
    _patch_cross_section(monkeypatch, mv)

    resid, warns = pipeline.neutralize(s, AS_OF, codes, by=("log_mv",))
    assert warns == []

    X = np.column_stack([np.ones_like(x), x])               # 同一份数据的 OLS 反事实
    beta_ols, *_ = np.linalg.lstsq(X, s.to_numpy(), rcond=None)
    ols = s.to_numpy() - X @ beta_ols

    scale = s.std()
    assert abs(ols[big].mean()) > 0.25 * scale              # OLS：大盘组系统性偏移（实测 0.34σ）
    assert abs(resid.to_numpy()[big].mean()) < 0.05 * scale  # WLS：回到 0 附近（实测 0.007σ）
    assert abs(resid.to_numpy()[big].mean()) < 0.1 * abs(ols[big].mean())


def test_neutralize_rejects_backfilled_industry_source(monkeypatch, cross_section):
    """industry_source != 'sw' → 行业是今天的值回填到上市日，中性化即前视污染。抛，不降级。"""
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind, source="tushare_static")
    with pytest.raises(RuntimeError, match="industry_source"):
        pipeline.neutralize(s, AS_OF, codes)


def test_neutralize_by_log_mv_only_works_on_degraded_industry(monkeypatch, cross_section):
    """不做行业中性化就不碰行业标签 —— 降级库上市值中性化仍然可用。"""
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind, source="tushare_static")
    resid, warns = pipeline.neutralize(s, AS_OF, codes, by=("log_mv",))
    assert warns == []
    assert abs(_wcorr(resid.to_numpy(), np.log(mv.to_numpy()), np.sqrt(mv.to_numpy()))) < 1e-10


def test_neutralize_with_too_few_observations_returns_input_plus_warning(monkeypatch):
    codes = _codes(29)
    mv = pd.Series(np.linspace(1e9, 1e10, 29), index=codes)
    s = pd.Series(np.arange(29.0), index=codes)
    _patch_cross_section(monkeypatch, mv)

    out, warns = pipeline.neutralize(s, AS_OF, codes, by=("log_mv",))
    assert out.equals(s)                                    # 原值，不是 NaN
    assert warns and any("29" in w for w in warns)


def test_neutralize_rank_deficient_returns_input_plus_warning(monkeypatch):
    """全池同一个市值 → log_mv 列与截距完全共线。lstsq 不会抛，它会返回一个
    最小范数解 —— 静默给出一份没做过中性化的"残差"。必须靠 rank 自己识别。"""
    codes = _codes(40)
    mv = pd.Series([2e9] * 40, index=codes)
    s = pd.Series(np.arange(40.0), index=codes)
    _patch_cross_section(monkeypatch, mv)

    out, warns = pipeline.neutralize(s, AS_OF, codes, by=("log_mv",))
    assert out.equals(s)
    assert warns and any("秩亏" in w for w in warns)


def test_neutralize_rejects_unknown_by_term(monkeypatch, cross_section):
    """by 拼错必须抛：静默跳过等于这一天的因子根本没中性化，而输出看起来完全正常。"""
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind)
    with pytest.raises(ValueError, match="by"):
        pipeline.neutralize(s, AS_OF, codes, by=("log_mv", "indsutry"))


def test_neutralize_on_real_market_db(market_db):
    """不打桩：真实读 fixture 库的 total_mv / 行业 / _meta.industry_source。
    market_db 只有 4 只股票 → 样本不足分支，返回原值 + warning。"""
    query.open_db(market_db)
    try:
        assert query.industry_source() == "sw"
        codes = ["A00001.SZ", "B00002.SZ", "C00003.SH", "D00004.SZ"]
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=codes)
        out, warns = pipeline.neutralize(s, AS_OF, codes)
        assert out.equals(s)
        assert warns
    finally:
        query.close_db()


# ══════════════ process：四步顺序 ══════════════
def _spec(neutralize: bool = True) -> FactorSpec:
    return FactorSpec(name="t", fn=lambda *a, **k: None, direction=1, category="price",
                      lookback_days=20, neutralize=neutralize)


def test_nan_survives_winsorize_and_neutralize_and_is_filled_only_at_the_end(
        monkeypatch, cross_section):
    """fillna(0) 必须在 zscore 之后。

    往输入里注入 3 个 NaN：
      1) 中性化输出里它们仍是 NaN（不是 0）；
      2) process 的最终输出里它们是 0；
      3) 其余股票的 z 值 == 只用有效样本算出来的 z 值 ——
         如果 fillna 提前到 zscore 之前，这 3 个 0 会进入均值和标准差，
         把【所有】股票的分数都拉偏，第 3 条就会挂。
    """
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind)
    s = s.copy()
    s.iloc[[0, 5, 9]] = np.nan
    holes = s.index[[0, 5, 9]]

    resid, _ = pipeline.neutralize(pipeline.winsorize_mad(s), AS_OF, codes)
    assert resid.loc[holes].isna().all()                    # ① 中性化阶段还是 NaN

    out, _ = pipeline.process(s, AS_OF, codes, spec=_spec())
    assert (out.loc[holes] == 0.0).all()                    # ② 末端才被填成 0

    valid = resid.drop(index=holes)                         # ③ 其余股票未被那 3 个 0 拉偏
    expected = (valid - valid.mean()) / valid.std()
    pd.testing.assert_series_equal(out.drop(index=holes), expected, check_names=False)


def test_process_skips_neutralization_when_spec_says_so(monkeypatch, cross_section):
    """risk 类因子（log_mv / industry / beta）自己就是中性化的回归元，不能被自己中性化掉。"""
    s, mv, ind, codes = cross_section
    _patch_cross_section(monkeypatch, mv, ind)
    out, warns = pipeline.process(s, AS_OF, codes, spec=_spec(neutralize=False))
    assert warns == []
    expected = pipeline.zscore(pipeline.winsorize_mad(s)).fillna(0.0)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_process_surfaces_neutralize_warnings(monkeypatch):
    """降级的那一天要在 BacktestResult.warnings 里看得见，而不是无声无息。"""
    codes = _codes(20)
    mv = pd.Series(np.linspace(1e9, 1e10, 20), index=codes)
    ind = pd.Series([f"IND{i % 3}" for i in range(20)], index=codes)
    _patch_cross_section(monkeypatch, mv, ind)
    s = pd.Series(np.arange(20.0), index=codes)
    out, warns = pipeline.process(s, AS_OF, codes, spec=_spec())
    assert warns
    assert out.abs().sum() > 0                              # 没中性化，但仍然出了 zscore
