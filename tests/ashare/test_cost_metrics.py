"""成本模型（§5.4）与绩效/因子检验指标（§4 / §9）—— 回测报告里每一个能骗人的数字。

本文件钉的是「算出来好看」的那一类失败，它们全都不抛异常：

  1. 【换手漏掉漂移归一化】全市场普涨 10%、一笔不交易，算出 10% 的换手（正确答案是 0）。
     §5.4 的初稿就漏了归一化分母，与 §9 自己的验收断言直接矛盾。
  2. 【印花税记到买入】单边 5bp 记成双边 → 往返成本凭空翻倍，或反过来漏记 → 净值虚高。
  3. 【冲击成本不封顶】微盘股一笔大单能算出 500bp 的冲击，把一条本来能跑的策略废掉；
     不封顶的另一面是【ADV20 缺失时静默按 0 冲击】—— 那才是危险的方向。
  4. 【ICIR 用朴素标准误】IC 序列有自相关，朴素 SE 把 t 高估 30%–50%，
     是把噪声因子判成有效因子的头号原因（§4.2 / §10.9）。本文件的 NW 钉子测的就是这个。
  5. 【分层单调性用 Pearson】单调但非线性的分层（真实因子的常态）会被 Pearson 打成 0.89，
     §4.3 要的是秩相关 ρ_mono。
  6. 【幽灵行业行】pandas 的 category dtype 在 groupby / value_counts 下会为
     【零观测的类别】发一行（`__OTHER__ → NaN`）。`pipeline.neutralize` 已经被
     `get_dummies` 的同一个行为咬过一次；归因表里多一行不存在的行业暴露是同一个 bug。
  7. 【残差的规模暴露不报】§3.2 选 OLS 而非 WLS 的裁决，唯一能被真实数据推翻的证据
     就是归因里这一项。不报 = 那个裁决永远无法被检验，所以它【不能藏在开关后面】。

★ fixture 里的权重 / 价格 / 收益一律取除不尽的值（global-constraints 2026-08-21）：
  0.3717 / 55.037188 这种。`shares × price / equity` 往返精确的 fixture 会让
  「按权重反推持仓」之类的变异与真实现逐位相同，测试绿的是算术不是代码。
"""
from __future__ import annotations
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from ashare.backtest import cost, metrics
from ashare.backtest.types import CostConfig

TOL = 1e-9

# ══════════════ §5.4 成本模型 ══════════════
_D = dt.date(2024, 1, 5)


def _trades(rows) -> pd.DataFrame:
    """rows: (ts_code, side, amount, adv20, range)"""
    return pd.DataFrame(
        [{"exec_date": _D, "ts_code": c, "side": s, "shares": a / 55.037188,
          "price_hfq": 55.037188, "amount": a, "adv20": adv, "range": rng}
         for c, s, a, adv, rng in rows])


def test_buy_one_million_pays_commission_but_no_stamp_duty():
    """验收：买入 100 万 → 佣金 250 元、印花税 0。过户费双边各 10 元（见下一条测试的注）。"""
    out, warns = cost.charge(_trades([("A.SZ", "BUY", 1_000_000.0, 9e9, 0.0)]), CostConfig())
    r = out.iloc[0]
    assert r["commission"] == pytest.approx(250.0, abs=TOL)
    assert r["stamp_duty"] == 0.0
    assert r["transfer_fee"] == pytest.approx(10.0, abs=TOL)
    assert r["impact"] == 0.0                      # range=0 → 冲击 0
    assert r["total_cost"] == pytest.approx(260.0, abs=TOL)
    assert warns == []


def test_sell_one_million_pays_commission_stamp_duty_and_transfer_fee():
    """验收：卖出 100 万 → 佣金 250 + 印花税 **500** + 过户费 10。

    ★ brief 的验收断言把印花税写成 5000 元，那是 50bp 而不是它自己声明的 5bp
      （`CostConfig.stamp_duty_bps=5.0`、§5.4 的 c^s 里也是 0.0005）。5000 还会让
      「一个完整往返 ≈ 0.3%」这句话变成 ≥0.55%，自相矛盾。取 500，见任务报告。
    ★ 印花税【只】在卖出侧。记到买入 = 往返成本翻倍；漏记 = 净值虚高 5bp/次。
    """
    out, _ = cost.charge(_trades([("A.SZ", "SELL", 1_000_000.0, 9e9, 0.0)]), CostConfig())
    r = out.iloc[0]
    assert r["commission"] == pytest.approx(250.0, abs=TOL)
    assert r["stamp_duty"] == pytest.approx(500.0, abs=TOL)
    assert r["transfer_fee"] == pytest.approx(10.0, abs=TOL)
    assert r["total_cost"] == pytest.approx(760.0, abs=TOL)


def test_one_round_trip_is_about_thirty_bps():
    """§5.4「一个完整往返 ≈ 0.3%」—— 佣金 5bp + 印花税 5bp + 过户费 0.2bp + 冲击 ~20bp。

    这一条是印花税量级的独立旁证：若印花税真是 50bp，往返最少 0.55%，与 §5.4 打架。
    """
    tr = _trades([("A.SZ", "BUY", 1_000_000.0, 2.6e7, 0.0517),
                  ("A.SZ", "SELL", 1_000_000.0, 2.6e7, 0.0517)])
    out, _ = cost.charge(tr, CostConfig())
    assert out["total_cost"].sum() / 1_000_000.0 == pytest.approx(0.003, abs=4e-4)


def test_impact_is_capped_at_thirty_bps_on_a_real_adv20(market_db):
    """验收「冲击成本封顶 30bp」——【真实 ADV20】，不是占位数。

    ADV20 与振幅都从库里取（这正是 Task 13 引擎要做的事，见 cost.py 的「谁来给 ADV20」）：
      · ADV20 = 信号日 T 往前 20 个交易日 amount 的均值，**剔除停牌日的 0**；
      · range = 执行日掩码的 amplitude。
    一笔吃掉 ADV20 大半的单子必须被 30bp 封住，一笔小单必须落在公式上。
    """
    from ashare.data import query
    query.open_db(market_db)
    try:
        sig, ex = dt.date(2024, 1, 31), dt.date(2024, 2, 1)
        bars = query.get_bars(sig, ["A00001.SZ"], lookback=20, fields=("amount",))
        amt = bars["amount"].astype(float)
        adv20 = float(amt[amt > 0].mean())          # ★ 停牌占位行 amount=0，不算进 ADV
        rng = float(query.get_tradable_mask(ex, ["A00001.SZ"])["amplitude"].iloc[0])
    finally:
        query.close_db()
    assert adv20 > 0 and 0 < rng < 0.1              # fixture 自检：真数不是占位

    big = 40.0 * adv20 * 1e-2 / rng                 # 保证 0.5·(amt/ADV)·rng ≫ 0.003
    small = 0.0731 * adv20
    out, warns = cost.charge(
        _trades([("A00001.SZ", "BUY", big, adv20, rng),
                 ("A00001.SZ", "BUY", small, adv20, rng)]), CostConfig())
    assert out["impact"].iloc[0] == pytest.approx(0.0030 * big, rel=1e-12)
    assert out["impact"].iloc[1] == pytest.approx(0.5 * (small / adv20) * rng * small, rel=1e-12)
    assert out["impact"].iloc[1] < 0.0030 * small   # 第二笔确实【没有】被封顶
    assert warns == []


def test_unknown_adv20_charges_the_cap_and_says_so():
    """ADV20 缺失 → 按【封顶】计冲击 + 告警，绝不按 0。

    静默按 0 是「流动性未知 ⇒ 免费成交」，方向恰好错在把净值画好看那一侧；
    按上限计只会让结果变差，且一定看得见（与 `get_tradable_mask` 的
    `limit_unknown → 两侧不可交易` 同一个保守方向）。
    """
    tr = _trades([("A.SZ", "BUY", 813_402.0, float("nan"), 0.0431),
                  ("B.SZ", "SELL", 621_907.0, 0.0, 0.0288),
                  ("C.SZ", "BUY", 402_311.0, 3.1e8, float("nan"))])
    out, warns = cost.charge(tr, CostConfig())
    assert out["impact"].to_numpy() == pytest.approx((0.0030 * out["amount"]).to_numpy(), rel=1e-12)
    assert len(warns) == 1 and "3" in warns[0]


def test_missing_impact_columns_is_an_engine_contract_error():
    """`charge` 是纯函数，ADV20 / range 由引擎附列（cost.py 模块头）。
    少列时抛，不是默默把冲击算成 0 —— 那会让整条曲线少掉全部冲击成本。"""
    tr = _trades([("A.SZ", "BUY", 1e6, 1e9, 0.03)]).drop(columns=["adv20"])
    with pytest.raises(ValueError, match="adv20"):
        cost.charge(tr, CostConfig())


def test_unknown_side_raises():
    """side 写错（'B' / 'buy' / ''）会静默跳过印花税，卖出成本凭空少 5bp。"""
    with pytest.raises(ValueError, match="side"):
        cost.charge(_trades([("A.SZ", "buy", 1e6, 1e9, 0.03)]), CostConfig())


def test_non_positive_amount_raises():
    with pytest.raises(ValueError, match="amount"):
        cost.charge(_trades([("A.SZ", "BUY", -1e6, 1e9, 0.03)]), CostConfig())


def test_multiplier_scales_every_component_and_the_cap_applies_first():
    """闸 4（成本敏感）的 multiplier=2.0 必须真的让【封顶的那些笔】也翻倍。

    先封顶再乘：否则被封顶的笔在 multiplier 下纹丝不动，闸 4 对最贵的那批交易等于没跑。
    并且四项分量之和必须等于 total_cost（只乘 total 会让明细对不上账）。
    """
    tr = _trades([("A.SZ", "SELL", 917_233.0, 1.1e6, 0.0863)])
    one, _ = cost.charge(tr, CostConfig())
    two, _ = cost.charge(tr, CostConfig(multiplier=2.0))
    assert one["impact"].iloc[0] == pytest.approx(0.0030 * 917_233.0, rel=1e-12)
    for c in ("commission", "stamp_duty", "transfer_fee", "impact", "total_cost"):
        assert two[c].iloc[0] == pytest.approx(2.0 * one[c].iloc[0], rel=1e-12)
    parts = two[["commission", "stamp_duty", "transfer_fee", "impact"]].iloc[0].sum()
    assert parts == pytest.approx(two["total_cost"].iloc[0], rel=1e-12)


def test_empty_trades_still_returns_the_cost_columns():
    """空成交表要带着列出去：下游 `groupby(exec_date)['total_cost']` 不该在无交易日 KeyError。"""
    out, warns = cost.charge(pd.DataFrame(columns=["exec_date", "ts_code", "side", "amount",
                                                   "adv20", "range"]), CostConfig())
    assert list(out.columns[-5:]) == cost.COST_COLS
    assert len(out) == 0 and warns == []


# ══════════════ §5.4 换手 = Δw，必须扣持仓自然漂移 ══════════════
def _wide(d: dict, dates) -> pd.DataFrame:
    return pd.DataFrame(d, index=pd.Index(dates, name="rebalance_date"))


def test_flat_book_in_a_ten_percent_rally_has_zero_turnover():
    """验收断言：持仓不动、股价齐涨 10% → 换手 0（不是 10%）。

    ★ 归一化分母 `1 + Σ w R` 不可省。漏掉它这一期就算出 10% 的换手，
      §5.4 的初稿正是这么写的，而它与本断言直接矛盾（portfolio.py 模块头已记一笔）。
    """
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.3717, 0.3717], "B.SZ": [0.2903, 0.2903], "C.SZ": [0.3380, 0.3380]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.10], "B.SZ": [np.nan, 0.10], "C.SZ": [np.nan, 0.10]}, dates)
    to, warns = metrics.turnover_series(w, r)
    assert to.iloc[0] == pytest.approx(1.0, abs=TOL)        # 首期建仓 = 满仓换手
    assert to.iloc[1] == pytest.approx(0.0, abs=1e-15)
    assert warns == []


def test_drift_is_computed_per_name_not_from_the_portfolio_average():
    """异质收益下的手算值。齐涨那条测试杀不掉「用组合平均收益当每只的漂移」这个变异。"""
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.3717, 0.3184], "B.SZ": [0.2903, 0.3521], "C.SZ": [0.3380, 0.3295]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.0731], "B.SZ": [np.nan, -0.0418], "C.SZ": [np.nan, 0.1129]}, dates)
    to, _ = metrics.turnover_series(w, r)
    assert to.iloc[1] == pytest.approx(0.175969329977063, abs=1e-12)   # 手算
    assert to.iloc[1] != pytest.approx(0.201066010000000, abs=1e-6)    # 漏归一化分母的那个值


def test_a_rally_does_require_trading_when_the_book_holds_cash():
    """未满仓时齐涨 10% 的正确答案【不是】0 —— 现金不涨，持仓占比会被顶上去。

    Σw=0.8 时 w·1.1/1.08 > w，维持同一个目标必须卖出。这一条防的是
    「见到齐涨就直接判 0」的偷懒实现（它会在满仓 fixture 上完美通过）。
    """
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.2913, 0.2913], "B.SZ": [0.2547, 0.2547], "C.SZ": [0.2540, 0.2540]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.10], "B.SZ": [np.nan, 0.10], "C.SZ": [np.nan, 0.10]}, dates)
    to, _ = metrics.turnover_series(w, r)
    assert to.iloc[1] == pytest.approx(0.014814814814815, abs=1e-12)


def test_a_held_name_without_a_return_is_reported_not_silently_drifted_by_zero():
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.4831, 0.4831], "B.SZ": [0.5169, 0.5169]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.0217], "B.SZ": [np.nan, np.nan]}, dates)
    _, warns = metrics.turnover_series(w, r)
    assert any("B.SZ" in w_ or "漂移" in w_ for w_ in warns)


# ══════════════ §9 净值 / 相对指标 ══════════════
_R = [0.0137, -0.0208, 0.0091, 0.0313, -0.0074, -0.0161,
      -0.0212, -0.0183, -0.0247, -0.0126,
      0.0117, 0.0205, -0.0092, 0.0061, 0.0183, 0.0074,
      -0.0119, 0.0288, -0.0056, 0.0132]
_RB = [0.0102, -0.0171, 0.0148, 0.0261, 0.0013, -0.0094,
       -0.0155, -0.0121, -0.0203, -0.0067,
       0.0089, 0.0247, -0.0038, 0.0104, 0.0126, 0.0031,
       -0.0072, 0.0219, -0.0011, 0.0178]


def _curve(rets) -> pd.Series:
    days = pd.bdate_range("2024-01-01", periods=len(rets) + 1)
    return pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(rets))]),
                     index=pd.Index([d.date() for d in days], name="trade_date"))


def _empty_positions() -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([[], []], names=["rebalance_date", "ts_code"])
    return pd.DataFrame(columns=["score", "target_weight", "filled_weight", "shares",
                                 "price_hfq", "industry"], index=idx)


def test_sharpe_calmar_mdd_are_the_hand_computed_values():
    """§9 的四个数，手算值硬编码。

    ★ 最大回撤【跨 4 个交易日复合】(9.6289%)，大于任何单日跌幅 (2.47%)：
      「MDD = 最大单日亏损」这个变异必须死在这里。
    """
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None, full=False)
    assert m["annual_return"] == pytest.approx(0.125895625089497, abs=1e-12)
    assert m["annual_vol"] == pytest.approx(0.277061819626481, abs=1e-12)
    assert m["sharpe"] == pytest.approx(0.454395431529406, abs=1e-12)
    assert m["max_drawdown"] == pytest.approx(0.096289495984263, abs=1e-12)
    assert m["calmar"] == pytest.approx(1.307469976892118, abs=1e-12)
    assert m["max_drawdown"] > 0.0247 + 1e-6            # ← 不是最大单日亏损
    assert any("基准" in w for w in warns)              # 没给基准 → 相对指标缺失要说


def test_information_ratio_is_negative_when_the_book_trails_the_benchmark():
    """IR 的【符号】必须跟着超额收益走。符号翻转是一个不抛异常、且报告照样漂亮的变异。"""
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), _curve(_RB), full=False)
    assert m["information_ratio"] == pytest.approx(-7.475392206155894, abs=1e-10)
    assert m["information_ratio"] < 0


def test_no_benchmark_leaves_ir_none_and_warns():
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None, full=False)
    assert m["information_ratio"] is None
    assert any("基准" in w for w in warns)


def test_risk_free_rate_lowers_sharpe():
    """§9 的 Sharpe 是 (R_ann − R_f)/σ。R_f 默认 0，给了就必须真的减掉。"""
    a, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None, full=False)
    b, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                           full=False, risk_free=0.0217)
    assert b["sharpe"] == pytest.approx(a["sharpe"] - 0.0217 / a["annual_vol"], abs=1e-12)


# ── full=True 的交易类指标 ──
def _positions_frame(rows, cols) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([(d, c) for d, c, *_ in rows],
                                    names=["rebalance_date", "ts_code"])
    return pd.DataFrame([r[2:] for r in rows], index=idx, columns=cols)


_POS_COLS = ["score", "target_weight", "filled_weight", "shares", "price_hfq", "industry"]
_D1, _D2 = dt.date(2024, 1, 5), dt.date(2024, 1, 12)
_POS_ROWS = [
    (_D1, "A.SZ", 1.31, 0.5083, 0.5083, 9236.0, 55.037188, "银行"),
    (_D1, "B.SZ", 0.47, 0.4917, 0.4917, 4471.0, 110.0731, "白酒"),
    (_D2, "A.SZ", 0.92, 0.3164, 0.2811, 5107.0, 57.318829, "银行"),
    (_D2, "B.SZ", 1.08, 0.6836, 0.7189, 6532.0, 108.4402, "白酒"),
]


def test_d6_slippage_is_measured_from_actual_weights_not_from_blocked_intent():
    """D6 的缺口 = Σ|目标 − 实际成交|，从【真实权重】现算。

    ★ `simulate` 的 `Σ blocked.intended_weight` 两个方向都会偏（execution.py 的交接注
      实测 0.55 vs 0.80 低估、0.70 vs 0.42 高估），所以它【不能】当缺口用。
    """
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _positions_frame(_POS_ROWS, _POS_COLS),
                           None, full=True)
    # D1 完全成交（0），D2 差 |0.3164−0.2811| + |0.6836−0.7189| = 0.0706
    assert m["d6_slippage_mean"] == pytest.approx(0.0706 / 2, abs=1e-12)
    assert m["d6_slippage_max"] == pytest.approx(0.0706, abs=1e-12)


def test_a_price_gap_does_not_get_forward_filled_into_a_fabricated_return():
    """一只票中间某期不在账本里 → 那一期的收益是【没有】，不是"和上次一样"。

    `pct_change()` 默认 ffill，会拿更早的价格补出一个编造的收益 —— 与 D9 说的
    "把停牌占位行 ffill 成假的 0 收益"是同一个错，只是发生在换手这一侧。
    """
    d3 = dt.date(2024, 1, 19)
    rows = [(_D1, "A.SZ", 1.31, 0.5083, 0.5083, 9236.0, 55.037188, "银行"),
            (_D1, "B.SZ", 0.47, 0.4917, 0.4917, 4471.0, 110.0731, "白酒"),
            (_D2, "A.SZ", 0.92, 1.0000, 1.0000, 18163.0, 57.318829, "银行"),
            (_D2, "B.SZ", 0.11, 0.0000, 0.0000, 0.0, np.nan, "白酒"),
            (d3, "A.SZ", 0.61, 0.4712, 0.4712, 8014.0, 59.117302, "银行"),
            (d3, "B.SZ", 1.44, 0.5288, 0.5288, 5127.0, 113.6094, "白酒")]
    _, warns = metrics.compute(_curve(_R), pd.DataFrame(),
                               _positions_frame(rows, _POS_COLS), None, full=True)
    assert any("B.SZ" in w for w in warns)


def test_cost_drag_is_annualised_from_the_trade_ledger():
    eq = _curve(_R)
    tr = pd.DataFrame({"exec_date": [_D1, _D1, _D2],
                       "total_cost": [3172.41, 1908.07, 2661.53]})
    # equity 的 index 是 bdate_range 从 2024-01-01 起：把成交日对齐到曲线上的真实日期
    tr["exec_date"] = [eq.index[3], eq.index[3], eq.index[8]]
    m, warns = metrics.compute(eq, tr, _positions_frame(_POS_ROWS, _POS_COLS), None, full=True)
    years = (len(eq) - 1) / 252.0
    c = (3172.41 + 1908.07) / eq.iloc[3] + 2661.53 / eq.iloc[8]
    assert m["cost_drag_annual"] == pytest.approx(c / years, rel=1e-12)
    assert not any("成交日" in w for w in warns)


def test_a_trade_dated_off_the_equity_curve_is_reported():
    eq = _curve(_R)
    tr = pd.DataFrame({"exec_date": [dt.date(1999, 1, 4)], "total_cost": [1234.5]})
    _, warns = metrics.compute(eq, tr, _positions_frame(_POS_ROWS, _POS_COLS), None, full=True)
    assert any("1999-01-04" in w for w in warns)


def test_full_false_skips_the_trade_block():
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _positions_frame(_POS_ROWS, _POS_COLS),
                           None, full=False)
    assert "turnover_annual" not in m and "d6_slippage_mean" not in m
    assert "sharpe" in m


def test_factors_used_per_period_surfaces_partial_degradation():
    """§9 诊断：每期实际用了几个因子。

    ★ `build_targets` 的 50% 覆盖率闸经 `combine` 喂进来是【二值】的
      （`process` 末尾 fillna(0) → 覆盖率恒为 100% 或 0%），探测不到部分降级。
      「12 个因子失效、拿剩下 2 个把整个账本调了一遍」只能靠这一项发现。
    """
    fu = pd.Series([14, 14, 2, 13], index=[_D1, _D2, dt.date(2024, 1, 19), dt.date(2024, 1, 26)])
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(),
                               _positions_frame(_POS_ROWS, _POS_COLS), None,
                               full=True, factors_used=fu)
    assert m["factors_used_min"] == 2
    assert m["factors_used_max"] == 14
    assert m["factors_used_median"] == pytest.approx(13.5)
    assert m["n_periods_below_half"] == 1
    assert any("2024-01-19" in w for w in warns)


def test_missing_factors_used_is_itself_reported():
    """不给这一项 = 部分降级不可见。那就必须在 warnings 里说出来。"""
    _, warns = metrics.compute(_curve(_R), pd.DataFrame(),
                               _positions_frame(_POS_ROWS, _POS_COLS), None, full=True)
    assert any("因子" in w and "每期" in w for w in warns)


# ══════════════ §4.1 IC / RankIC ══════════════
def _panel(dates, codes, values) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, codes], names=["rebalance_date", "ts_code"])
    return pd.DataFrame(values, index=idx)


_CODES = [f"{i:06d}.SZ" for i in range(10)]
_SC = [-1.31, -0.82, -0.47, -0.16, 0.09, 0.34, 0.58, 0.91, 1.27, 1.63]
_RT = [0.9137, 0.0113, 0.0186, 0.0241, 0.0307, 0.0382, 0.0455, 0.0524, 0.0613, 0.0729]


def test_rank_ic_survives_a_limit_up_streak_that_flips_pearson():
    """§4.1：主用 RankIC。一只连板把 Pearson 拽到 −0.52，而 RankIC 仍是 +0.45。

    两列都要出：报了 IC 却没报 RankIC，或反过来把 RankIC 算成 Pearson，
    在正常横截面上几乎看不出差别 —— 只有涨停连板这种 A 股常态能分开它们。
    """
    fp = _panel([_D1], _CODES, {"reversal_20": _SC})
    fr = pd.Series(_RT, index=fp.index, name="fwd_ret")
    ic, warns = metrics.ic_series(fp, fr)
    assert list(ic.columns) == ["reversal_20__ic", "reversal_20__rank_ic"]
    assert ic["reversal_20__ic"].iloc[0] == pytest.approx(-0.515105623255936, abs=1e-12)
    assert ic["reversal_20__rank_ic"].iloc[0] == pytest.approx(0.454545454545454, abs=1e-12)
    assert warns == []


def test_ic_series_covers_every_factor_and_every_date():
    dates = [_D1, _D2]
    fp = _panel(dates, _CODES, {"a": _SC * 2, "b": list(reversed(_SC)) * 2})
    fr = pd.Series(_RT * 2, index=fp.index)
    ic, _ = metrics.ic_series(fp, fr)
    assert list(ic.index) == dates
    assert list(ic.columns) == ["a__ic", "a__rank_ic", "b__ic", "b__rank_ic"]
    assert ic["a__rank_ic"].iloc[0] == pytest.approx(-ic["b__rank_ic"].iloc[0], abs=1e-12)


def test_a_cross_section_too_small_for_a_significant_rank_ic_is_dropped():
    """n=4 的秩相关最小双侧 p 是 1/12 ≈ 0.083 —— 永远够不到 5%，算出来只是噪声。

    ★ 样本量必须能分辨相邻阈值（global-constraints 2026-08-21）：这里用 n=4（剔除）
      与 n=5（保留）两组直接夹住 `_MIN_IC_OBS`，把 4 / 6 两个候选值一起杀掉。
    """
    small = _panel([_D1], _CODES[:4], {"f": [0.13, -0.27, 0.41, -0.08]})
    ic, warns = metrics.ic_series(small, pd.Series([0.017, -0.023, 0.038, -0.011],
                                                  index=small.index))
    assert np.isnan(ic["f__rank_ic"].iloc[0])
    assert any("4" in w for w in warns)

    ok = _panel([_D1], _CODES[:5], {"f": [0.13, -0.27, 0.41, -0.08, 0.22]})
    ic2, warns2 = metrics.ic_series(ok, pd.Series([0.017, -0.023, 0.038, -0.011, 0.029],
                                                 index=ok.index))
    assert np.isfinite(ic2["f__rank_ic"].iloc[0])
    assert warns2 == []


def test_a_constant_cross_section_gives_nan_ic_and_says_why():
    """`process` 末尾的 fillna(0) 会把「全部因子都被剔掉」的一天变成一列恒 0。
    corr 给 NaN 而不抛 —— 那一天的 IC 必须带着理由出现在 warnings 里。"""
    fp = _panel([_D1], _CODES, {"f": [0.0] * 10})
    ic, warns = metrics.ic_series(fp, pd.Series(_RT, index=fp.index))
    assert np.isnan(ic["f__ic"].iloc[0]) and np.isnan(ic["f__rank_ic"].iloc[0])
    assert any("常数" in w for w in warns)


# ══════════════ §4.2 ICIR + Newey-West ══════════════
def _autocorrelated_ic(n: int = 118) -> pd.Series:
    """强自相关的 IC 序列（周期 40 的正弦，n=118 故意不是整周期 → 均值不是整数）。"""
    t = np.arange(n)
    return pd.Series(0.031 + 0.047 * np.sin(2 * np.pi * t / 40.0),
                     index=pd.bdate_range("2022-01-03", periods=n))


def test_newey_west_t_is_materially_below_the_naive_t():
    """★ 本文件最重要的一条（§4.2 / §10.9）。

    IC 序列自相关时朴素标准误把 t 高估 30%–50%，是把噪声因子判成有效因子的头号原因。
    这条强自相关序列上：朴素 t = 10.08，NW t = 4.64 —— 差一倍还多。
    朴素 SE、lag=0、或者把 Bartlett 权写成 1 的实现都过不了这里。
    """
    out, warns = metrics.icir(_autocorrelated_ic())
    assert out["nw_lag"] == 4
    assert out["t_naive"] == pytest.approx(10.075182197580178, abs=1e-10)
    assert out["t_newey_west"] == pytest.approx(4.641556793001294, abs=1e-10)
    assert out["t_newey_west"] < 0.6 * out["t_naive"]
    assert warns == []


def test_newey_west_lag_is_floor_four_times_t_over_hundred_to_the_two_ninths():
    for n, lag in ((52, 3), (100, 4), (118, 4), (250, 4), (500, 5), (10, 2)):
        assert metrics.newey_west_lag(n) == lag, n


def test_our_hac_matches_statsmodels():
    """§4.2 指定的参照实现。生产路径用 numpy 直接算 Bartlett 核（6 行、零运行时依赖，
    statsmodels 缺席时 NW 钉子照样跑），这一条保证两者不会各自演化。"""
    sm = pytest.importorskip("statsmodels.api")
    ic = _autocorrelated_ic()
    lag = metrics.newey_west_lag(len(ic))
    ours = metrics._nw_se(ic.to_numpy(dtype=float), lag)
    ref = sm.OLS(ic.to_numpy(dtype=float), np.ones(len(ic))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}).bse[0]
    assert ours == pytest.approx(ref, rel=1e-12)


def test_icir_annualises_by_sqrt_fifty_two():
    out, _ = metrics.icir(_autocorrelated_ic())
    assert out["icir"] == pytest.approx(0.927495700179336, abs=1e-12)
    assert out["icir_ann"] == pytest.approx(out["icir"] * math.sqrt(52), abs=1e-12)
    assert 0.0 < out["p"] < 1e-4


def test_icir_of_a_constant_series_is_not_infinitely_significant():
    out, warns = metrics.icir(pd.Series([0.0271] * 30))
    assert out["t_newey_west"] is None or np.isnan(out["t_newey_west"])
    assert warns


def test_icir_needs_at_least_two_observations():
    out, warns = metrics.icir(pd.Series([0.0193]))
    assert np.isnan(out["std"]) or out["std"] is None
    assert warns


# ══════════════ §4.3 分层 ══════════════
def test_layered_returns_are_equal_weighted_ascending_by_score():
    """L1 = 分数最低的一层。层序倒过来 = 单调性符号整体翻转，而报告照样"好看"。"""
    codes = [f"{i:06d}.SZ" for i in range(20)]
    sc = pd.Series(np.linspace(-1.93, 2.07, 20), index=pd.MultiIndex.from_product(
        [[_D1], codes], names=["rebalance_date", "ts_code"]))
    ret = pd.Series(np.linspace(-0.0431, 0.0619, 20), index=sc.index)
    lay, warns = metrics.layered_returns(sc, ret, n_layers=10)
    assert list(lay.columns) == [f"L{i}" for i in range(1, 11)]
    assert lay["L1"].iloc[0] == pytest.approx(np.mean(ret.to_numpy()[:2]), abs=1e-12)
    assert lay["L10"].iloc[0] == pytest.approx(np.mean(ret.to_numpy()[-2:]), abs=1e-12)
    assert warns == []


def test_tied_scores_are_layered_deterministically_regardless_of_input_order():
    """★ 精确的平手是常态：`process` 末尾 fillna(0) 填的是【正好 0】、§3.1 的 MAD
    截到的是【正好】边界值。不先按 (score, ts_code) 定序，分层就随调用方给的 index
    顺序翻脸 —— D7 的「同参数同数据必复现」只是碰巧成立（portfolio.py 同一处坑）。
    """
    codes = [f"{i:06d}.SZ" for i in range(20)]
    sc_by_code = dict(zip(codes, [0.0] * 12 + [1.31, 1.31, 1.31, -0.47, -0.47, 2.03, -1.19, 0.0]))
    ret_by_code = dict(zip(codes, np.linspace(-0.0431, 0.0619, 20)))
    got = []
    for order in (codes, list(reversed(codes))):
        idx = pd.MultiIndex.from_product([[_D1], order], names=["rebalance_date", "ts_code"])
        lay, _ = metrics.layered_returns(
            pd.Series([sc_by_code[c] for c in order], index=idx),
            pd.Series([ret_by_code[c] for c in order], index=idx), n_layers=5)
        got.append(lay)
    pd.testing.assert_frame_equal(got[0], got[1])


def test_a_cross_section_smaller_than_the_layer_count_is_skipped():
    codes = [f"{i:06d}.SZ" for i in range(6)]
    sc = pd.Series(np.linspace(-1.0, 1.0, 6), index=pd.MultiIndex.from_product(
        [[_D1], codes], names=["rebalance_date", "ts_code"]))
    lay, warns = metrics.layered_returns(sc, pd.Series(np.linspace(-0.01, 0.03, 6), index=sc.index))
    assert lay.isna().all(axis=None)
    assert any("6" in w for w in warns)


_LAYER_PERIODS = 39         # ★ 故意不是 52：满 52 期时年化的指数恰好是 1，
                            #   「少了幂次、直接用累计收益」这个变异会与真实现逐位相同。


def _layer_frame(annual_by_layer) -> pd.DataFrame:
    """把 10 个【年化】收益反推成 `_LAYER_PERIODS` 期等额周收益，喂给 layer_monotonicity。"""
    per = [(1.0 + a) ** (1 / 52.0) - 1.0 for a in annual_by_layer]
    return pd.DataFrame([per] * _LAYER_PERIODS, columns=[f"L{i}" for i in range(1, 11)],
                        index=pd.bdate_range("2023-01-02", periods=_LAYER_PERIODS, freq="W-MON"))


def test_perfectly_monotone_layers_give_rho_mono_of_one():
    ann = [0.0113, 0.0217, 0.0331, 0.0429, 0.0538, 0.0641, 0.0759, 0.0863, 0.0971, 0.1087]
    out, _ = metrics.layer_monotonicity(_layer_frame(ann))
    assert out["rho_mono"] == pytest.approx(1.0, abs=1e-12)
    # 年化必须真的年化（52/39 次幂）：只断言单调性的话，漏掉幂次的实现照样 ρ=1
    assert out["layer_annual"]["L10"] == pytest.approx(0.1087, rel=1e-12)
    assert out["layer_annual"]["L1"] == pytest.approx(0.0113, rel=1e-12)
    assert out["long_short_eval_only_annual"] == pytest.approx(0.1087 - 0.0113, abs=1e-12)


def test_monotonicity_is_a_rank_correlation_not_pearson():
    """真实因子的分层几乎从不线性。这组【单调但凸】的年化收益 Spearman=1、Pearson=0.887。

    §4.3 的判据是 |ρ_mono| > 0.7；Pearson 在这里给 0.887 也过得了闸，
    所以只断言"过闸"是钉不住的 —— 必须断言它恰好等于 1。
    """
    ann = [0.0011, 0.0017, 0.0026, 0.0041, 0.0063, 0.0098, 0.0152, 0.0236, 0.0367, 0.0571]
    out, _ = metrics.layer_monotonicity(_layer_frame(ann))
    assert out["rho_mono"] == pytest.approx(1.0, abs=1e-12)
    assert pd.Series(np.arange(1.0, 11.0)).corr(pd.Series(ann)) == pytest.approx(0.887050344, abs=1e-8)


def test_a_reversed_factor_gives_rho_mono_of_minus_one():
    lay = _layer_frame([0.1087, 0.0971, 0.0863, 0.0759, 0.0641,
                        0.0538, 0.0429, 0.0331, 0.0217, 0.0113])
    out, _ = metrics.layer_monotonicity(lay)
    assert out["rho_mono"] == pytest.approx(-1.0, abs=1e-12)
    assert out["long_short_eval_only_annual"] < 0


def test_the_long_short_number_is_labelled_evaluation_only():
    """§4.3：多空只是因子评估口径，A 股融券成本与券源不支持系统性做空。
    键名本身必须带着这句话 —— 报告里出现一个叫 `long_short_return` 的数字，
    下一个人就会把它当成一条可交易的策略收益。"""
    out, _ = metrics.layer_monotonicity(_layer_frame([0.01 * i for i in range(1, 11)]))
    assert "long_short_eval_only_annual" in out
    assert not any(k in out for k in ("long_short_return", "long_short_annual", "ls_return"))


# ══════════════ §9 归因 ══════════════
_ATTR_DATES = [_D1, _D2]
_ATTR_CODES = ["A.SZ", "B.SZ", "C.SZ", "D.SZ"]


def _attr_inputs(*, industry_dtype="object", intended=True, size_vals=None, score_vals=None):
    idx = pd.MultiIndex.from_product([_ATTR_DATES, _ATTR_CODES],
                                     names=["rebalance_date", "ts_code"])
    ind = pd.Series(["银行", "白酒", "银行", "白酒"] * 2, index=idx)
    if industry_dtype == "category":
        # ★ 真实形态：risk.industry 返回 category dtype，且 __OTHER__ 在类别表里
        #   却【零观测】—— get_universe 把成分 <5 家的行业并进去，那天可能一只都没选中。
        ind = ind.astype(pd.CategoricalDtype(categories=["银行", "白酒", "__OTHER__"]))
    pos = pd.DataFrame({
        "score": score_vals if score_vals is not None else [0.31, -0.17, 0.83, -0.42] * 2,
        "target_weight": [0.2731, 0.2419, 0.2503, 0.2347] * 2,
        "filled_weight": [0.2731, 0.2419, 0.2503, 0.2347,
                          0.2117, 0.2903, 0.2641, 0.2339],
        "shares": [4183.0, 2917.0, 3311.0, 5209.0] * 2,
        "price_hfq": [55.037188, 110.0731, 33.71829, 21.44073] * 2,
        "industry": ind,
    }, index=idx)
    if intended:
        pos["intended_weight"] = [0.2731, 0.2419, 0.2503, 0.2347,
                                  0.2519, 0.2617, 0.2803, 0.2061]
    fwd = pd.Series([0.0217, -0.0143, 0.0381, -0.0092,
                     0.0119, 0.0263, -0.0071, 0.0184], index=idx)
    sc = pd.Series(score_vals if score_vals is not None else [0.31, -0.17, 0.83, -0.42] * 2,
                   index=idx)
    size = pd.Series(size_vals if size_vals is not None
                     else [24.31, 22.17, 23.09, 21.83, 24.37, 22.11, 23.14, 21.79], index=idx)
    return pos, fwd, sc, size


def test_industry_attribution_never_invents_a_zero_observation_industry():
    """★ pandas 2.3.3 实测：`v.groupby(category_series).mean()` 会为【零观测的类别】
    发一行（`__OTHER__ → NaN`），`value_counts()` 同样报 `__OTHER__ → 0`。

    `pipeline.neutralize` 已经被 `get_dummies` 的同一个行为咬过一次（那次靠 .astype(object)
    修的）。归因表里多一行不存在的行业暴露，读者会当成真的持仓。
    """
    pos, fwd, sc, size = _attr_inputs(industry_dtype="category")
    tbl, _ = metrics.attribution(pos, fwd, sc, size=size)
    inds = list(tbl.loc[tbl["block"] == "industry", "item"])
    assert sorted(inds) == sorted(["银行", "白酒"])
    assert "__OTHER__" not in inds
    assert not tbl.loc[tbl["block"] == "industry", "exposure"].isna().any()


def test_industry_exposure_and_contribution_are_per_period_averages():
    pos, fwd, sc, size = _attr_inputs()
    tbl, _ = metrics.attribution(pos, fwd, sc, size=size)
    row = tbl[(tbl["block"] == "industry") & (tbl["item"] == "银行")].iloc[0]
    assert row["exposure"] == pytest.approx((0.2731 + 0.2503 + 0.2117 + 0.2641) / 2, abs=1e-12)
    assert row["contribution"] == pytest.approx(
        (0.2731 * 0.0217 + 0.2503 * 0.0381 + 0.2117 * 0.0119 + 0.2641 * -0.0071) / 2, abs=1e-12)


def test_the_residual_size_exposure_is_always_in_the_table():
    """★ §3.2「OLS 而非 WLS」这个裁决，唯一能被真实数据推翻的证据就是这一行。

    所以它【不能】藏在开关后面，也不能因为样本不足就整行消失 —— 没有它，
    那条裁决永远无法被检验（brief 转来的第 2 条）。样本不足时报 NaN + 告警，行照在。
    """
    pos, fwd, sc, size = _attr_inputs()
    tbl, warns = metrics.attribution(pos, fwd, sc, size=size)
    items = set(tbl.loc[tbl["block"] == "style", "item"])
    assert {"size", "size_sq"} <= items

    bad = pd.Series(np.nan, index=size.index)          # 市值整列缺失 → 行仍在
    tbl2, warns2 = metrics.attribution(pos, fwd, sc, size=bad)
    style2 = tbl2[tbl2["block"] == "style"]
    assert set(style2["item"]) == {"size", "size_sq"}
    assert style2["exposure"].isna().all()
    assert any("规模" in w for w in warns2)


def test_a_planted_linear_size_tilt_shows_up_in_the_size_row():
    """残差 = 1.37×size + 噪声 → size 行的暴露必须是正的、且量级对得上。"""
    n, dates = 40, [_D1, _D2]
    codes = [f"{i:06d}.SZ" for i in range(n)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["rebalance_date", "ts_code"])
    lm = np.tile(np.linspace(19.13, 26.87, n), 2)
    z = np.concatenate([(np.linspace(19.13, 26.87, n) - np.mean(np.linspace(19.13, 26.87, n)))
                        / np.std(np.linspace(19.13, 26.87, n), ddof=1)] * 2)
    sc = pd.Series(1.37 * z, index=idx)
    pos = pd.DataFrame({"filled_weight": np.full(2 * n, 1.0 / n),
                        "industry": ["银行"] * (2 * n)}, index=idx)
    tbl, _ = metrics.attribution(pos, pd.Series(0.0, index=idx), sc, size=pd.Series(lm, index=idx))
    row = tbl[(tbl["block"] == "style") & (tbl["item"] == "size")].iloc[0]
    assert row["exposure"] == pytest.approx(1.37, rel=1e-6)


def test_a_nonlinear_size_tilt_is_caught_by_the_size_squared_row():
    """★ 推翻裁决之后的【补救方向】也必须是可测的：§3.2 说补救是加非线性规模项
    （size² 或秩变换），不是换回 WLS。一个纯 size² 的残差在线性项上暴露为 0 ——
    只报线性暴露就会得出「已经中性了」的结论。
    """
    n, dates = 40, [_D1, _D2]
    codes = [f"{i:06d}.SZ" for i in range(n)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["rebalance_date", "ts_code"])
    raw = np.linspace(19.13, 26.87, n)
    z = (raw - raw.mean()) / raw.std(ddof=1)
    sc = pd.Series(np.tile(0.83 * z ** 2, 2), index=idx)
    pos = pd.DataFrame({"filled_weight": np.full(2 * n, 1.0 / n),
                        "industry": ["银行"] * (2 * n)}, index=idx)
    tbl, _ = metrics.attribution(pos, pd.Series(0.0, index=idx), sc,
                                 size=pd.Series(np.tile(raw, 2), index=idx))
    lin = tbl[(tbl["block"] == "style") & (tbl["item"] == "size")].iloc[0]
    sq = tbl[(tbl["block"] == "style") & (tbl["item"] == "size_sq")].iloc[0]
    assert lin["exposure"] == pytest.approx(0.0, abs=1e-9)
    assert sq["exposure"] == pytest.approx(0.83, rel=1e-6)


def test_style_regression_boundary_is_thirty_observations():
    """架构 B7 的 30 —— 用 29（跳过）/ 30（算）两组直接夹住，
    不然把 `_MIN_STYLE_OBS` 改成 20 或 35 都没有任何测试会红。"""
    for n, want_nan in ((29, True), (30, False)):
        codes = [f"{i:06d}.SZ" for i in range(n)]
        idx = pd.MultiIndex.from_product([[_D1], codes], names=["rebalance_date", "ts_code"])
        raw = np.linspace(19.13, 26.87, n)
        z = (raw - raw.mean()) / raw.std(ddof=1)
        pos = pd.DataFrame({"filled_weight": np.full(n, 1.0 / n),
                            "industry": ["银行"] * n}, index=idx)
        tbl, _ = metrics.attribution(pos, pd.Series(0.0, index=idx),
                                     pd.Series(1.37 * z, index=idx),
                                     size=pd.Series(raw, index=idx))
        got = tbl[(tbl["block"] == "style") & (tbl["item"] == "size")]["exposure"].iloc[0]
        assert bool(np.isnan(got)) is want_nan, (n, got)


def test_constraint_drag_uses_the_intended_book():
    """§9 的约束拖累 = 意图账本的反事实收益 − 实际收益。

    分不清「跑输是因为信号不行」还是「因为换手约束让信号表达不出来」，
    对一个受换手约束的策略是完全相反的两个结论（brief 转来的第 8 条）。
    """
    pos, fwd, sc, size = _attr_inputs()
    tbl, _ = metrics.attribution(pos, fwd, sc, size=size)
    row = tbl[tbl["block"] == "constraint"].iloc[0]
    gap = pos["intended_weight"] - pos["filled_weight"]
    assert row["contribution"] == pytest.approx(float((gap * fwd).sum()) / 2, abs=1e-12)
    assert row["exposure"] == pytest.approx(float(gap.abs().sum()) / 2, abs=1e-12)


def test_missing_intended_weight_leaves_the_row_but_says_it_is_unattributable():
    pos, fwd, sc, size = _attr_inputs(intended=False)
    tbl, warns = metrics.attribution(pos, fwd, sc, size=size)
    row = tbl[tbl["block"] == "constraint"].iloc[0]
    assert np.isnan(row["contribution"])
    assert any("intended_weight" in w for w in warns)


def test_attribution_columns_are_fixed():
    pos, fwd, sc, size = _attr_inputs()
    tbl, _ = metrics.attribution(pos, fwd, sc, size=size)
    assert list(tbl.columns) == metrics.ATTRIBUTION_COLS
