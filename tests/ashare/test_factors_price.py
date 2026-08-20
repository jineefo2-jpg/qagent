"""Task 4：六个量价因子 —— 数值断言全部对着【构造的已知序列】做闭式解。

为什么不用 market_db 做主力用例：那个 fixture 只有 29 个交易日，连 momentum_120_20
的一个窗口都装不下；而且"返回了一个形状正确的 Series"这种断言抓不到任何真实错误。
本文件每条用例钉的是一个具体的错法：

  · 区间错位     —— momentum_120_20 用 [t-120, t] 而不是 [t-120, t-20]
  · 收益口径混用 —— 波动率/Amihud 该用对数收益，反转/动量/max_ret 该用简单收益
  · 停牌污染     —— 不 ffill 则一天停牌毁掉整个窗口；ffill 过头则拿 3 天真实价硬算 20 日反转
  · 窗口边界     —— 20 日均值取了 21 天

真实取数链路（get_price_panel/get_bars/get_daily_basic 的 NaN 语义）由末尾
test_*_on_real_market_db 覆盖，不打桩，直接读 fixture 库。
"""
from __future__ import annotations
import datetime as dt
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from ashare.data import query
from ashare.factors import price as fp

AS_OF = "2024-01-05"
CODES = ["S00001.SZ", "S00002.SZ"]


# ══════════════ 构造面板的脚手架 ══════════════
def _dates(n: int) -> list[dt.date]:
    """n 个"交易日"（连续自然日即可，因子只按行序取值，不看日历）。"""
    return [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def _panel(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=_dates(n)).rename_axis("trade_date")


def _ramp(n: int, base: float = 100.0, step: float = 0.01) -> np.ndarray:
    """每日恰好上涨 step 的复利价格路径：p_i = base·(1+step)^i。"""
    return base * (1.0 + step) ** np.arange(n)


def _patch_prices(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    """替换 get_price_panel，但【遵守 lookback】—— 因子取多少行、切哪一段都是被测行为。"""
    def _fake(as_of_date, ts_codes, field="close", lookback=250, adjust="hfq"):
        assert field == "close", f"量价因子只应取收盘价，实际取了 {field!r}"
        assert adjust == "hfq", "D8：后复权是唯一真值"
        if seen is not None:
            seen["lookback"] = lookback
        return panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
    monkeypatch.setattr(query, "get_price_panel", _fake)


def _long(panel: pd.DataFrame, ts_codes, lookback: int, col: str) -> pd.DataFrame:
    """宽表 → 长表 MultiIndex (ts_code, trade_date)，并【丢掉 NaN 行】——
    真实 SQL 查询就是这样：没数据的股票根本不出现在结果里，不会有一列 NaN 等着你。"""
    sub = panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
    out = sub.stack(future_stack=True).dropna().rename(col).to_frame()
    out.index = out.index.set_names(["trade_date", "ts_code"])
    return out.reorder_levels(["ts_code", "trade_date"]).sort_index()


def _patch_amount(monkeypatch, panel: pd.DataFrame) -> None:
    """替换 get_bars，只提供 amount 列（长表 MultiIndex，与真实返回同形）。"""
    def _fake(as_of_date, ts_codes, *, lookback=None, start=None, fields=("close",), adjust="hfq"):
        assert adjust == "hfq"
        out = _long(panel, ts_codes, lookback, "amount")
        out["is_suspended"] = False
        return out
    monkeypatch.setattr(query, "get_bars", _fake)


def _patch_turnover(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    def _fake(as_of_date, ts_codes, fields=("turnover_rate_f",), lookback=1):
        assert "turnover_rate_f" in fields, f"换手率因子应取 turnover_rate_f，实际 {fields}"
        if seen is not None:
            seen["lookback"] = lookback
        return _long(panel, ts_codes, lookback, "turnover_rate_f")
    monkeypatch.setattr(query, "get_daily_basic", _fake)


# ══════════════ 1 reversal_20 ══════════════
def test_reversal_20_closed_form_on_a_1pct_ramp(monkeypatch):
    """每日 +1% 涨 20 天 → reversal = −(1.01²⁰ − 1)。反转因子带负号，涨得多的分数低。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(200) for c in CODES}))
    out = fp.reversal_20(AS_OF, CODES)
    assert out.notna().all()
    np.testing.assert_allclose(out.to_numpy(), -(1.01 ** 20 - 1), rtol=1e-12)


def test_reversal_20_window_is_exactly_20_not_19_or_21(monkeypatch):
    """只有 p_t 与 p_{t-20} 进公式：改 p_{t-21} 不动结果，改 p_{t-20} 一定动。"""
    px = _ramp(200)
    _patch_prices(monkeypatch, _panel({c: px.copy() for c in CODES}))
    base = fp.reversal_20(AS_OF, CODES).iloc[0]

    far = px.copy(); far[-22] *= 3.0                            # p_{t-21}
    _patch_prices(monkeypatch, _panel({c: far for c in CODES}))
    assert fp.reversal_20(AS_OF, CODES).iloc[0] == pytest.approx(base)

    edge = px.copy(); edge[-21] *= 3.0                          # p_{t-20}
    _patch_prices(monkeypatch, _panel({c: edge for c in CODES}))
    assert fp.reversal_20(AS_OF, CODES).iloc[0] != pytest.approx(base)


def test_reversal_20_ffills_a_halt_instead_of_returning_nan(monkeypatch):
    """停牌日 NaN 不 ffill 就会顺着 p_t / p_{t-20} 传染成 NaN；ffill 后应取停牌前的最后真实价。"""
    px = _ramp(200)
    halted = px.copy()
    halted[-1] = np.nan                                          # t 日停牌
    halted[-21] = np.nan                                         # t-20 日也停牌
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    out = fp.reversal_20(AS_OF, CODES)
    assert out.notna().all(), "窗口内非空天数充足，停牌日应被 ffill 补上而不是传染 NaN"
    # p_t → p_{t-1}，p_{t-20} → p_{t-21}，两端各前移一天，比值仍是 1.01²⁰
    np.testing.assert_allclose(out["S00001.SZ"], -(1.01 ** 20 - 1), rtol=1e-12)


@pytest.mark.parametrize("n_real,expect_nan", [(13, False), (12, True)])
def test_reversal_20_needs_60pct_real_observations(monkeypatch, n_real, expect_nan):
    """窗口 21 天 × 60% = 12.6 → 13 天真实价才算数。12 天靠 ffill 硬算出来的是噪声，置 NaN。"""
    px = _ramp(200)
    thin = px.copy()
    thin[-21:] = np.nan
    keep = np.linspace(-21, -1, n_real).astype(int)              # 含两端，共 n_real 个真实日
    thin[keep] = px[keep]
    _patch_prices(monkeypatch, _panel({"S00001.SZ": thin, "S00002.SZ": px}))
    out = fp.reversal_20(AS_OF, CODES)
    assert bool(np.isnan(out["S00001.SZ"])) is expect_nan
    assert out.notna()["S00002.SZ"], "同一横截面里数据完整的股票不应被连坐"


# ══════════════ 2 momentum_120_20 ══════════════
def test_momentum_120_20_closed_form_is_the_100_day_span(monkeypatch):
    """区间 [t-120, t-20] 共 100 个 1% 涨幅 → 1.01¹⁰⁰ − 1（不是 1.01¹²⁰ − 1）。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(400) for c in CODES}))
    out = fp.momentum_120_20(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 1.01 ** 100 - 1, rtol=1e-12)
    assert out.iloc[0] != pytest.approx(1.01 ** 120 - 1), "算成 [t-120, t] 了：跳过最近一月是本因子的全部意义"


def test_momentum_120_20_ignores_the_last_20_days_entirely(monkeypatch):
    """t-19..t 的价格怎么改都不影响因子值 —— 这是"跳过最近一个月"的可验证形式。"""
    px = _ramp(400)
    _patch_prices(monkeypatch, _panel({c: px.copy() for c in CODES}))
    base = fp.momentum_120_20(AS_OF, CODES).iloc[0]

    wild = px.copy()
    wild[-20:] *= 10.0                                           # t-19 .. t 全部 ×10
    _patch_prices(monkeypatch, _panel({c: wild for c in CODES}))
    assert fp.momentum_120_20(AS_OF, CODES).iloc[0] == pytest.approx(base), \
        "最近 20 天的价格进了公式：区间应是 [t-120, t-20]"

    edge = px.copy()
    edge[-21] *= 10.0                                            # p_{t-20} 是区间右端点，必须生效
    _patch_prices(monkeypatch, _panel({c: edge for c in CODES}))
    assert fp.momentum_120_20(AS_OF, CODES).iloc[0] == pytest.approx(base * 10 + 9), \
        "p_{t-20} ×10 → (10·p_{t-20}/p_{t-120} − 1) = 10·(base+1) − 1"


def test_momentum_120_20_coverage_window_excludes_the_skipped_month(monkeypatch):
    """最近一月整月停牌不该让动量作废 —— 它本来就不看那一段。"""
    px = _ramp(400)
    halted = px.copy()
    halted[-20:] = np.nan
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    np.testing.assert_allclose(fp.momentum_120_20(AS_OF, CODES)["S00001.SZ"],
                               1.01 ** 100 - 1, rtol=1e-12)


def test_momentum_120_20_coverage_counts_only_the_scored_window(monkeypatch):
    """反向：区间内只有一半天数真实成交 → NaN，哪怕最近一个月天天正常交易。

    覆盖率窗口若错写成 [t-120, t]，最近一月那 20 天会把计数顶过阈值，
    于是一只三个月前长期停牌的股票凭最近的活跃拿到一个动量分 —— 而那个分
    恰恰算自它没怎么交易的那段区间。这条用例专门钉住这个方向。
    """
    px = _ramp(400)
    thin = px.copy()
    thin[-121:-20] = np.nan
    keep = np.arange(-121, -20, 2)                               # 51/101 = 50.5%，含两端点
    thin[keep] = px[keep]
    _patch_prices(monkeypatch, _panel({"S00001.SZ": thin, "S00002.SZ": px}))
    out = fp.momentum_120_20(AS_OF, CODES)
    assert np.isnan(out["S00001.SZ"]), "覆盖率窗口把被跳过的一个月也数进去了"
    assert out.notna()["S00002.SZ"]


# ══════════════ 3 volatility_60 ══════════════
def _zigzag(n: int, up: float = 1.2, base: float = 100.0) -> np.ndarray:
    """在 base 与 base·up 之间来回：对数收益恰为 ±ln(up)，简单收益是 +0.2/−1/6（两者显著不同）。"""
    out = np.full(n, base)
    out[1::2] = base * up
    return out


def test_volatility_60_is_the_sample_std_of_60_log_returns(monkeypatch):
    """闭式：60 个 ±ln(1.2)，均值 0 → std(ddof=1) = ln(1.2)·√(60/59)。"""
    _patch_prices(monkeypatch, _panel({c: _zigzag(300) for c in CODES}))
    out = fp.volatility_60(AS_OF, CODES)
    expect = math.log(1.2) * math.sqrt(60 / 59)
    np.testing.assert_allclose(out.to_numpy(), expect, rtol=1e-12)


def test_volatility_60_uses_log_not_simple_returns(monkeypatch):
    """同一路径下简单收益的 std 明显更大；两者数值必须可区分，否则口径换了没人发现。"""
    _patch_prices(monkeypatch, _panel({c: _zigzag(300) for c in CODES}))
    got = fp.volatility_60(AS_OF, CODES).iloc[0]
    simple = np.array([0.2, -1 / 6] * 30)
    assert got != pytest.approx(float(simple.std(ddof=1)), rel=1e-6), "用成简单收益了"


def test_volatility_60_uses_ddof_1(monkeypatch):
    """规格 §2.1 的分母是 59（样本标准差），不是 60。"""
    _patch_prices(monkeypatch, _panel({c: _zigzag(300) for c in CODES}))
    got = fp.volatility_60(AS_OF, CODES).iloc[0]
    assert got != pytest.approx(math.log(1.2), rel=1e-9), "分母用成 60 了（总体标准差）"


def test_volatility_60_spans_exactly_60_returns(monkeypatch):
    """第 61 根收益之外的价格不进窗口。"""
    px = _zigzag(300)
    _patch_prices(monkeypatch, _panel({c: px.copy() for c in CODES}))
    base = fp.volatility_60(AS_OF, CODES).iloc[0]
    far = px.copy(); far[-62] = 1.0                              # 只影响第 61 根收益（窗口外）
    _patch_prices(monkeypatch, _panel({c: far for c in CODES}))
    assert fp.volatility_60(AS_OF, CODES).iloc[0] == pytest.approx(base)


# ══════════════ 4 turnover_20 ══════════════
def test_turnover_20_averages_exactly_20_days(monkeypatch):
    """1..20 的等差换手率 → 均值 10.5；若多取一天（0..20）会得到 10.0。"""
    seen: dict = {}
    ramp = np.arange(0.0, 21.0)                                  # 末 20 个是 1..20
    _patch_turnover(monkeypatch, _panel({c: ramp for c in CODES}), seen)
    out = fp.turnover_20(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 10.5, rtol=1e-12)
    assert seen["lookback"] == 20


def test_turnover_20_nan_days_do_not_count_as_zero(monkeypatch):
    """停牌日在 daily_basic 里是 NaN；当成 0 会把均值拖低成一个假的"低换手"好分数。"""
    v = np.full(40, 4.0)
    v[-3:] = np.nan
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": v, "S00002.SZ": np.full(40, 4.0)}))
    np.testing.assert_allclose(fp.turnover_20(AS_OF, CODES)["S00001.SZ"], 4.0, rtol=1e-12)


def test_turnover_20_needs_60pct_real_observations(monkeypatch):
    v = np.full(40, 4.0)
    v[-20:] = np.nan
    v[-11:] = 4.0                                                # 只剩 11/20 = 55%
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": v, "S00002.SZ": np.full(40, 4.0)}))
    out = fp.turnover_20(AS_OF, CODES)
    assert np.isnan(out["S00001.SZ"])
    assert out.notna()["S00002.SZ"]


# ══════════════ 5 amihud_20 ══════════════
def test_amihud_20_closed_form(monkeypatch):
    """每日恒定 +10%、成交额恒定 1e6 → 1e9/20 · Σ ln(1.1)/1e6 = 1000·ln(1.1)。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(200, step=0.10) for c in CODES}))
    _patch_amount(monkeypatch, _panel({c: np.full(200, 1e6) for c in CODES}))
    out = fp.amihud_20(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 1e9 * math.log(1.1) / 1e6, rtol=1e-12)


def test_amihud_20_uses_log_returns(monkeypatch):
    """简单收益会给出 1000·0.10；两者差 5%，混用不会报错只会静默变数。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(200, step=0.10) for c in CODES}))
    _patch_amount(monkeypatch, _panel({c: np.full(200, 1e6) for c in CODES}))
    assert fp.amihud_20(AS_OF, CODES).iloc[0] != pytest.approx(1e9 * 0.10 / 1e6, rel=1e-6)


def test_amihud_20_zero_amount_day_is_dropped_not_infinite(monkeypatch):
    """停牌日 amount=0：|r|/0 = inf 会让整只股票的非流动性变成 inf，横截面被一只股票绑架。"""
    amt = np.full(200, 1e6)
    amt[-5] = 0.0
    _patch_prices(monkeypatch, _panel({c: _ramp(200, step=0.10) for c in CODES}))
    _patch_amount(monkeypatch, _panel({"S00001.SZ": amt, "S00002.SZ": np.full(200, 1e6)}))
    out = fp.amihud_20(AS_OF, CODES)
    assert np.isfinite(out["S00001.SZ"])
    np.testing.assert_allclose(out["S00001.SZ"], 1e9 * math.log(1.1) / 1e6, rtol=1e-12)


# ══════════════ 6 max_ret_20 ══════════════
def test_max_ret_20_picks_the_single_largest_daily_gain(monkeypatch):
    """窗口内插一根 +30%：结果就是 0.30，不是均值也不是振幅。"""
    px = _ramp(200)
    px[-5:] *= 1.30 / 1.01                                       # 第 t-4 日单日 +30%，其后保持 1%
    _patch_prices(monkeypatch, _panel({c: px for c in CODES}))
    np.testing.assert_allclose(fp.max_ret_20(AS_OF, CODES).to_numpy(), 0.30, rtol=1e-12)


def test_max_ret_20_uses_simple_returns(monkeypatch):
    """简单收益 0.30 vs 对数收益 ln(1.3)=0.262；文献口径是简单收益。"""
    px = _ramp(200)
    px[-5:] *= 1.30 / 1.01
    _patch_prices(monkeypatch, _panel({c: px for c in CODES}))
    assert fp.max_ret_20(AS_OF, CODES).iloc[0] != pytest.approx(math.log(1.3), rel=1e-6)


def test_max_ret_20_window_is_20_returns(monkeypatch):
    """t-20 日的涨幅（第 21 根收益）在窗口外，不该被选中。"""
    px = _ramp(200)
    px[-21:] *= 5.0                                              # 第 t-20 日单日 +400%，其后不变
    _patch_prices(monkeypatch, _panel({c: px for c in CODES}))
    np.testing.assert_allclose(fp.max_ret_20(AS_OF, CODES).to_numpy(), 0.01, rtol=1e-12)


# ══════════════ 契约：签名 / index / 注册元数据 ══════════════
_ALL = ["reversal_20", "momentum_120_20", "volatility_60", "turnover_20", "amihud_20", "max_ret_20"]


@pytest.mark.parametrize("name", _ALL)
def test_first_two_positional_params_are_as_of_date_and_universe(name):
    """L3 静态检查之外再钉一次：装饰器原样返回函数，签名就是运行期真签名。"""
    params = list(inspect.signature(getattr(fp, name)).parameters.values())
    assert [p.name for p in params[:2]] == ["as_of_date", "universe"]
    assert all(p.kind is p.KEYWORD_ONLY for p in params[2:]), "窗口参数必须是 keyword-only"


@pytest.mark.parametrize("name,direction,lookback", [
    ("reversal_20", 1, 30), ("momentum_120_20", 1, 130), ("volatility_60", -1, 70),
    ("turnover_20", -1, 30), ("amihud_20", 1, 30), ("max_ret_20", -1, 30)])
def test_registered_metadata_matches_the_spec_table(name, direction, lookback):
    from ashare.factors.base import get_factor
    spec = get_factor(name)
    assert spec.direction == direction
    assert spec.category == "price"
    assert spec.lookback_days == lookback
    assert spec.neutralize is True, "量价因子是 alpha，要中性化（风险因子才设 False）"
    assert spec.fn is getattr(fp, name)


@pytest.mark.parametrize("name", _ALL)
def test_index_is_exactly_universe_in_universe_order(monkeypatch, name):
    """index 必须【就是】universe 且保持传入顺序。

    只断言"⊆ universe"是不够的：amihud 把价格面板和成交额面板相除，pandas 会把两边
    的列取并集【并重新排序】—— 结果 index 悄悄变成字典序。下游 process 是按标签
    对齐的所以不会炸，但任何按位置读这个 Series 的代码都会静默错配。
    故意用非字典序的 universe。
    """
    unordered = ["S00002.SZ", "S00001.SZ"]
    only_first = {"S00001.SZ": np.full(400, 1e6)}                # 另一只在长表里完全缺席
    _patch_prices(monkeypatch, _panel({c: _ramp(400) for c in unordered}))
    _patch_amount(monkeypatch, _panel(only_first))
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": np.full(400, 3.0)}))
    out = getattr(fp, name)(AS_OF, unordered)
    assert isinstance(out, pd.Series)
    assert list(out.index) == unordered, f"{name} 的 index 不等于 universe（或顺序变了）"


@pytest.mark.parametrize("name", _ALL)
def test_history_shorter_than_the_window_is_nan_not_a_crash(monkeypatch, name):
    """回测头几天必然取不满一个窗口 —— 该返回全 NaN，不是 IndexError，也不是拿 5 天硬算。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(5) for c in CODES}))
    _patch_amount(monkeypatch, _panel({c: np.full(5, 1e6) for c in CODES}))
    _patch_turnover(monkeypatch, _panel({c: np.full(5, 3.0) for c in CODES}))
    out = getattr(fp, name)(AS_OF, CODES)
    assert list(out.index) == CODES
    assert out.isna().all(), f"{name} 用不足一个窗口的历史算出了值"


@pytest.mark.parametrize("name", _ALL)
def test_empty_universe_returns_an_empty_series(monkeypatch, name):
    """空票池（某天全池被停牌/ST 剔光）不该抛。"""
    _patch_prices(monkeypatch, _panel({c: _ramp(400) for c in CODES}))
    _patch_amount(monkeypatch, _panel({c: np.full(400, 1e6) for c in CODES}))
    _patch_turnover(monkeypatch, _panel({c: np.full(400, 3.0) for c in CODES}))
    out = getattr(fp, name)(AS_OF, [])
    assert isinstance(out, pd.Series) and len(out) == 0


@pytest.mark.parametrize("name", _ALL)
def test_a_code_absent_from_the_source_is_nan_not_missing(monkeypatch, name):
    """池内某股完全无数据（长表里根本没有它的行）→ 该股为 NaN，但仍在 index 里。
    下游 pipeline.process 按 universe 做横截面回归，少一个 index 会静默错位。"""
    dead = np.full(400, np.nan)
    _patch_prices(monkeypatch, _panel({"S00001.SZ": _ramp(400), "S00002.SZ": dead}))
    _patch_amount(monkeypatch, _panel({"S00001.SZ": np.full(400, 1e6), "S00002.SZ": dead}))
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": np.full(400, 3.0), "S00002.SZ": dead}))
    out = getattr(fp, name)(AS_OF, CODES)
    assert list(out.index) == CODES
    assert np.isnan(out["S00002.SZ"])
    assert out.notna()["S00001.SZ"]


# ══════════════ 真实取数链路（不打桩，直接读 fixture 库）══════════════
def _fixture_hfq(base: float, i: int) -> float:
    """fixture 第 i 个交易日的后复权价：原始价 base+0.1i，复权因子 1+0.01i（两者都不为 1）。"""
    return (base + 0.1 * i) * (1.0 + 0.01 * i)


def test_reversal_20_on_real_market_db(market_db):
    """29 个交易日的 fixture：B 全程正常交易，A 在 01-15~01-17 停牌。
    as_of=2024-02-02 → i=28，t-20 → i=8。fixture 的 adj_factor 不是 1，
    所以这条同时钉住"用的是后复权价"（D8）—— 用原始价会得到明显不同的数。"""
    query.open_db(market_db)
    try:
        codes = ["A00001.SZ", "B00002.SZ"]
        out = fp.reversal_20("2024-02-02", codes)
        np.testing.assert_allclose(out["B00002.SZ"],
                                   -(_fixture_hfq(20, 28) / _fixture_hfq(20, 8) - 1), rtol=1e-9)
        # A 的 3 个停牌日被 ffill 补上，两端都是真实价 → 同样的闭式
        np.testing.assert_allclose(out["A00001.SZ"],
                                   -(_fixture_hfq(10, 28) / _fixture_hfq(10, 8) - 1), rtol=1e-9)
        assert out["B00002.SZ"] != pytest.approx(-((20 + 2.8) / (20 + 0.8) - 1), rel=1e-6), \
            "拿到的是未复权价 —— D8：后复权是唯一真值"
    finally:
        query.close_db()


def test_turnover_20_on_real_market_db(market_db):
    """fixture 的 turnover_rate_f 恒为 1.2。"""
    query.open_db(market_db)
    try:
        out = fp.turnover_20("2024-02-02", ["A00001.SZ", "B00002.SZ"])
        np.testing.assert_allclose(out.to_numpy(), 1.2, rtol=1e-9)
    finally:
        query.close_db()


def test_amihud_20_on_real_market_db_survives_suspended_zero_amount(market_db):
    """A 停牌 3 天、amount=0：真实库里就有这一行，不做保护就是 inf。"""
    query.open_db(market_db)
    try:
        out = fp.amihud_20("2024-02-02", ["A00001.SZ", "B00002.SZ"])
        assert np.isfinite(out).all(), f"amount=0 的停牌日污染了结果: {out.to_dict()}"
        assert (out > 0).all()
    finally:
        query.close_db()
