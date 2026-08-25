"""Task 4：六个量价因子 —— 数值断言全部对着【构造的已知序列】做闭式解。

为什么不用 market_db 做主力用例：那个 fixture 只有 29 个交易日，连 momentum_120_20
的一个窗口都装不下；而且"返回了一个形状正确的 Series"这种断言抓不到任何真实错误。
本文件每条用例钉的是一个具体的错法：

  · 区间错位     —— momentum_120_20 用 [t-120, t] 而不是 [t-120, t-20]
  · 收益口径混用 —— 波动率/Amihud 该用对数收益，反转/动量/max_ret 该用简单收益
  · 停牌污染     —— 不 ffill 则一天停牌毁掉整个窗口；ffill 过头则拿 3 天真实价硬算 20 日反转
  · 复牌日错配   —— ffill 让复牌日的 r 变成整段停牌的累计收益（Amihud 分子分母不同区间），
                    停牌日的 r 变成 0（全窗口下跌时它会赢下 max_ret）
  · 域外取值     —— 源在停牌日给出 turnover_rate_f=0，覆盖率闸门看不见，均值被摊薄
  · 窗口边界     —— 20 日均值取了 21 天；覆盖率闸门必须钉在【恰好】的阈值上，
                    只测宽松点与极端点的话阈值可以在区间里随便挪而全绿

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
            seen["lookback"] = max(lookback, seen.get("lookback", 0))
        return panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
    monkeypatch.setattr(query, "get_price_panel", _fake)


def _long(panel: pd.DataFrame, ts_codes, lookback: int, col: str) -> pd.DataFrame:
    """宽表 → 长表 MultiIndex (ts_code, trade_date)，并【丢掉 NaN 行】——
    真实 SQL 查询就是这样：没数据的股票根本不出现在结果里，不会有一列 NaN 等着你。"""
    sub = panel.reindex(columns=list(ts_codes)).iloc[-lookback:]
    out = sub.stack(future_stack=True).dropna().rename(col).to_frame()
    out.index = out.index.set_names(["trade_date", "ts_code"])
    return out.reorder_levels(["ts_code", "trade_date"]).sort_index()


def _patch_amount(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    """替换 get_bars，只提供 amount 列（长表 MultiIndex，与真实返回同形）。"""
    def _fake(as_of_date, ts_codes, *, lookback=None, start=None, fields=("close",), adjust="hfq"):
        assert adjust == "hfq"
        if seen is not None:
            seen["lookback"] = max(lookback, seen.get("lookback", 0))
        out = _long(panel, ts_codes, lookback, "amount")
        out["is_suspended"] = False
        return out
    monkeypatch.setattr(query, "get_bars", _fake)


def _patch_turnover(monkeypatch, panel: pd.DataFrame, seen: dict | None = None) -> None:
    def _fake(as_of_date, ts_codes, fields=("turnover_rate_f",), lookback=1):
        assert "turnover_rate_f" in fields, f"换手率因子应取 turnover_rate_f，实际 {fields}"
        if seen is not None:
            seen["lookback"] = max(lookback, seen.get("lookback", 0))
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


def test_volatility_60_keeps_the_ffilled_zero_and_the_resumption_jump(monkeypatch):
    """★ 现值钉桩（不是"这样最好"）：波动率【故意】不做 Amihud/max_ret 那道复牌日掩码。

    恒定 +1% 的路径上 60 根收益全等于 a=ln(1.01)，样本标准差恰为 0。
    窗口内插一天停牌后，ffill 把那天变成 0、复牌日变成 2a，其余 58 根仍是 a：
    均值仍是 a，离差只有 ±a 两项 → std(ddof=1) = a·√(2/59)。
    掩掉这两天会退回 58 根全等的 a → 又变回 0，两个数差得一眼可见。

    「k 个 0 + 1 个累计」的平方和期望是 (k+1)σ²、分母不变，所以现实现【无偏但方差大】；
    掩码版同样无偏、样本更少。哪个更好要用真实数据量 IC 代价（Task 12），本期不动。
    这条用例的作用只是：在那之前谁改了这个行为，必须是【明知故改】。
    """
    px = _ramp(300)
    a = math.log(1.01)
    _patch_prices(monkeypatch, _panel({c: px.copy() for c in CODES}))
    assert fp.volatility_60(AS_OF, CODES).iloc[0] == pytest.approx(0.0, abs=1e-15)

    halted = px.copy()
    halted[-30] = np.nan                                         # 窗口正中一天停牌
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    out = fp.volatility_60(AS_OF, CODES)
    np.testing.assert_allclose(out["S00001.SZ"], a * math.sqrt(2 / 59), rtol=1e-12)
    assert out["S00002.SZ"] == pytest.approx(0.0, abs=1e-15), "同一横截面里没停牌的股票不该被连坐"


@pytest.mark.parametrize("n_real,expect_nan", [(37, False), (36, True)])
def test_volatility_60_needs_60pct_real_observations(monkeypatch, n_real, expect_nan):
    """窗口 61 天 × 60% = 36.6 → 37 天真实价才算数；36 天置 NaN。
    钉在【边界】而不是随便取一个宽松值：否则闸门被偷偷收紧到 45 也是绿的。"""
    px = _zigzag(300)
    thin = px.copy()
    thin[-61:] = np.nan
    keep = np.linspace(-61, -1, n_real).astype(int)
    thin[keep] = px[keep]
    _patch_prices(monkeypatch, _panel({"S00001.SZ": thin, "S00002.SZ": px}))
    out = fp.volatility_60(AS_OF, CODES)
    assert bool(np.isnan(out["S00001.SZ"])) is expect_nan
    assert out.notna()["S00002.SZ"], "同一横截面里数据完整的股票不应被连坐"


# ══════════════ 4 turnover_20 ══════════════
def test_turnover_20_averages_exactly_20_days(monkeypatch):
    """1..20 的等差换手率 → 均值 10.5；若多取一天（0..20）会得到 10.0。"""
    seen: dict = {}
    ramp = np.arange(0.0, 21.0)                                  # 末 20 个是 1..20
    _patch_turnover(monkeypatch, _panel({c: ramp for c in CODES}), seen)
    out = fp.turnover_20(AS_OF, CODES)
    np.testing.assert_allclose(out.to_numpy(), 10.5, rtol=1e-12)
    assert seen["lookback"] == 20


@pytest.mark.parametrize("filler", [np.nan, 0.0], ids=["源无行/NULL", "源给出 0"])
def test_turnover_20_a_non_trading_day_never_counts_as_low_turnover(monkeypatch, filler):
    """停牌日两种形态，都必须【不进均值】——补 0 会算出一个假的"低换手"，
    而低换手在本因子里是【好分数】（direction=-1），等于给停牌股发买入信号。

    NaN 那半 `mean` 的 skipna 本来就管；`0.0` 那半是 D9 的老问题（源会给出 vol=0 的"行"），
    `ingest_daily_basic` 把 vendor 帧原样入库、`validate.py` 也不查换手率，所以只能在因子里防。
    20 天里 8 天停牌、真实换手 5.0：不防的话算成 3.0（低估 40%），
    而且 `notna()` 是 20/20，覆盖率闸门连响都不会响。
    """
    v = np.full(40, 5.0)
    v[-8:] = filler
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": v, "S00002.SZ": np.full(40, 5.0)}))
    out = fp.turnover_20(AS_OF, CODES)
    np.testing.assert_allclose(out["S00001.SZ"], 5.0, rtol=1e-12)
    assert out["S00001.SZ"] != pytest.approx(3.0, rel=1e-6), "把停牌日按 0 摊进了 20 天"


@pytest.mark.parametrize("n_real,expect_nan", [(12, False), (11, True)])
@pytest.mark.parametrize("filler", [np.nan, 0.0], ids=["源无行/NULL", "源给出 0"])
def test_turnover_20_needs_60pct_real_observations(monkeypatch, filler, n_real, expect_nan):
    """窗口 20 天 × 60% = 12 → 12 天算数、11 天置 NaN。必须【恰好】钉在边界上：
    只测 20/17/11 的话 [12, 17] 里任何一个阈值都能全绿，闸门可以被无声收紧。
    `0.0` 那一版同时钉住「闸门建在过滤之后」—— 建在之前的话 20/20 一路放行。"""
    v = np.full(40, 4.0)
    v[-20:] = filler
    v[-n_real:] = 4.0
    _patch_turnover(monkeypatch, _panel({"S00001.SZ": v, "S00002.SZ": np.full(40, 4.0)}))
    out = fp.turnover_20(AS_OF, CODES)
    assert bool(np.isnan(out["S00001.SZ"])) is expect_nan
    assert out.notna()["S00002.SZ"], "同一横截面里数据完整的股票不应被连坐"


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


def test_amihud_20_drops_the_resumption_day_not_just_the_halted_day(monkeypatch):
    """★ 分子分母必须是【同一段区间】。

    `amount<=0` 只挡掉停牌【当日】。复牌日的成交额是正的，闸门放行 —— 但那天的 `r`
    是 ffill 后跨越整段停牌的累计收益（k 天停牌约 √k 倍虚高），配的却只有一天的成交额。
    停一天：正确 1000·ln(1.1)，只挡当日的话 (20/19)·1000·ln(1.1)，高 5.3%。
    direction=+1 → 系统性超配爱停牌的票；且这是分布【体内】的位移，MAD 去极值截不掉。
    """
    px = _ramp(200, step=0.10)
    amt = np.full(200, 1e6)
    halted = px.copy()
    halted[-10] = np.nan                                         # 窗口内一天停牌
    amt_halted = amt.copy()
    amt_halted[-10] = 0.0                                        # D9：停牌日成交额为 0
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    _patch_amount(monkeypatch, _panel({"S00001.SZ": amt_halted, "S00002.SZ": amt}))
    out = fp.amihud_20(AS_OF, CODES)
    clean = 1e9 * math.log(1.1) / 1e6
    np.testing.assert_allclose(out["S00001.SZ"], clean, rtol=1e-12)
    assert out["S00001.SZ"] != pytest.approx(clean * 20 / 19, rel=1e-6), \
        "复牌日那根跨停牌的累计收益进了分子"
    np.testing.assert_allclose(out["S00002.SZ"], clean, rtol=1e-12)


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


def test_max_ret_20_never_reports_a_day_that_did_not_trade(monkeypatch):
    """★ 全窗口下跌时，ffill 造出来的那个 0 会赢下 `max`。

    天天跌 2% 的票：干净窗口报 −0.0200；插一天停牌，不剔的话报 0.0000 ——
    一个根本没有交易的日子成了"最好的一天"。复牌日的 −3.96%（跨两天的累计跌幅）
    同样不是单日涨幅，一起剔。direction=-1 所以后果是【多罚】而非奖励，量级也小，
    但报的是一个不存在的观测。
    """
    px = 100.0 * 0.98 ** np.arange(200)
    _patch_prices(monkeypatch, _panel({c: px.copy() for c in CODES}))
    np.testing.assert_allclose(fp.max_ret_20(AS_OF, CODES).to_numpy(), -0.02, rtol=1e-12)

    halted = px.copy()
    halted[-10] = np.nan
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    out = fp.max_ret_20(AS_OF, CODES)
    np.testing.assert_allclose(out["S00001.SZ"], -0.02, rtol=1e-12)
    assert out["S00001.SZ"] != pytest.approx(0.0, abs=1e-9), "停牌日那个 ffill 出来的 0 赢下了 max"


def test_max_ret_20_still_sees_a_real_spike_next_to_a_halt(monkeypatch):
    """反向锚：剔复牌日不能顺手把真实的单日大涨也剔掉 —— 否则上一条用「一律返回 NaN」也能过。"""
    px = _ramp(200)
    px[-6:] *= 1.30 / 1.01                                       # t-5 日单日 +30%
    halted = px.copy()
    halted[-10] = np.nan                                         # 停牌在更早的一天，与大涨不相邻
    _patch_prices(monkeypatch, _panel({"S00001.SZ": halted, "S00002.SZ": px}))
    np.testing.assert_allclose(fp.max_ret_20(AS_OF, CODES).to_numpy(), 0.30, rtol=1e-12)


# ══════════════ 契约：签名 / index / 注册元数据 ══════════════
_ALL = ["reversal_20", "momentum_120_20", "volatility_60", "turnover_20", "amihud_20", "max_ret_20"]


@pytest.mark.parametrize("name", _ALL)
def test_first_two_positional_params_are_as_of_date_and_universe(name):
    """L3 静态检查之外再钉一次：装饰器原样返回函数，签名就是运行期真签名。"""
    params = list(inspect.signature(getattr(fp, name)).parameters.values())
    assert [p.name for p in params[:2]] == ["as_of_date", "universe"]
    assert all(p.kind is p.KEYWORD_ONLY for p in params[2:]), "窗口参数必须是 keyword-only"


@pytest.mark.parametrize("name,direction", [
    ("reversal_20", 1), ("momentum_120_20", 1), ("volatility_60", -1),
    ("turnover_20", -1), ("amihud_20", 1), ("max_ret_20", -1)])
def test_registered_metadata_matches_the_spec_table(name, direction):
    from ashare.factors.base import get_factor
    spec = get_factor(name)
    assert spec.direction == direction
    assert spec.category == "price"
    assert spec.neutralize is True, "量价因子是 alpha，要中性化（风险因子才设 False）"
    assert spec.fn is getattr(fp, name)


@pytest.mark.parametrize("name", _ALL)
def test_declared_lookback_days_equals_what_the_factor_actually_fetches(monkeypatch, name):
    """`lookback_days` 是【声明给 preload 用的取数区间】，必须等于因子真的取了多少天。

    之前这里是两个手打的字面量互相比（注册写 30、用例也写 30），
    而 30 = window + _BUFFER 只是巧合：改 _BUFFER 就让声明与实际脱钩，全绿。
    turnover_20 更是直接对不上 —— 声明 30，实际只向 daily_basic 要 20 天。
    改成对着【被测函数实际传给 data 层的 lookback】断言，字面量就没有说谎的空间了。
    """
    from ashare.factors.base import get_factor
    seen: dict = {}
    _patch_prices(monkeypatch, _panel({c: _ramp(400) for c in CODES}), seen)
    _patch_amount(monkeypatch, _panel({c: np.full(400, 1e6) for c in CODES}), seen)
    _patch_turnover(monkeypatch, _panel({c: np.full(400, 3.0) for c in CODES}), seen)
    getattr(fp, name)(AS_OF, CODES)
    assert seen["lookback"] == get_factor(name).lookback_days


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


def test_turnover_20_on_real_market_db_ignores_a_source_zero_on_halted_days(market_db):
    """fixture 的 `daily_basic.turnover_rate_f` 恒为 1.2 —— 【包括 A 停牌的那三天】。

    那是 fixture 的简化，不是源的行为：真实源会在停牌日给出 0（D9 的同一个毛病，
    `daily_bar` 在 ingest 里被归一了，`daily_basic` 没有）。照原样断言"两只都是 1.2"
    等于把"停牌日照样计入均值"写成了期望契约。所以这里先把那三天改写成 0 再断言 ——
    答案必须仍是 1.2（0 当没有数据），而不是 17×1.2/20 = 1.02。
    """
    import duckdb                                                # 测试侧直连：改写 fixture 库
    con = duckdb.connect(market_db)
    con.execute("UPDATE daily_basic SET turnover_rate_f = 0 WHERE ts_code = 'A00001.SZ' "
                "AND trade_date IN (DATE '2024-01-15', DATE '2024-01-16', DATE '2024-01-17')")
    assert con.execute("SELECT count(*) FROM daily_basic WHERE turnover_rate_f = 0").fetchone()[0] == 3
    con.close()

    query.open_db(market_db)
    try:
        out = fp.turnover_20("2024-02-02", ["A00001.SZ", "B00002.SZ"])
        np.testing.assert_allclose(out.to_numpy(), 1.2, rtol=1e-9)
        assert out["A00001.SZ"] != pytest.approx(17 * 1.2 / 20, rel=1e-6), \
            "停牌日的 0 被摊进了 20 天 —— 而 direction=-1，这是给停牌股发买入信号"
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
