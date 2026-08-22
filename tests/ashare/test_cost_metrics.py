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
    """rows: (ts_code, side, amount, adv20, range)

    `range` 是【信号日往前 20 个交易日的平均振幅】，不是执行日当天的
    （2026-08-22 改口径，见 cost.py 模块头 —— 当天的 high/low 在 τ 开盘时还没走出来）。
    """
    return pd.DataFrame(
        [{"exec_date": _D, "ts_code": c, "side": s, "shares": a / 55.037188,
          "price_hfq": 55.037188, "amount": a, "adv20": adv, "range": rng}
         for c, s, a, adv, rng in rows])


def test_buy_one_million_pays_commission_but_no_stamp_duty():
    """验收：买入 100 万 → 佣金 250 元、印花税 0。过户费双边各 10 元（见下一条测试的注）。

    ★ 这里【不能】用 range=0 把冲击项抹平：0 现在走封顶路径（20 天连续一字板 = 不可成交，
      不是零波动零冲击）。改用一只极活跃的票（ADV20 80 亿、20 日均振幅 2.88%），
      冲击 1.8 元 —— 0.018bp，对"佣金 250、印花税 0"这条验收毫无干扰。
    """
    out, warns = cost.charge(_trades([("A.SZ", "BUY", 1_000_000.0, 8e9, 0.0288)]), CostConfig())
    r = out.iloc[0]
    assert r["commission"] == pytest.approx(250.0, abs=TOL)
    assert r["stamp_duty"] == 0.0
    assert r["transfer_fee"] == pytest.approx(10.0, abs=TOL)
    assert r["impact"] == pytest.approx(1.8, rel=1e-12)      # 0.5×(1e6/8e9)×0.0288×1e6
    assert r["total_cost"] == pytest.approx(261.8, rel=1e-12)
    assert warns == []


def test_sell_one_million_pays_commission_stamp_duty_and_transfer_fee():
    """验收：卖出 100 万 → 佣金 250 + 印花税 **500** + 过户费 10。

    ★ brief 的验收断言把印花税写成 5000 元，那是 50bp 而不是它自己声明的 5bp
      （`CostConfig.stamp_duty_bps=5.0`、§5.4 的 c^s 里也是 0.0005）。5000 还会让
      「一个完整往返 ≈ 0.3%」这句话变成 ≥0.55%，自相矛盾。取 500，见任务报告。
    ★ 印花税【只】在卖出侧。记到买入 = 往返成本翻倍；漏记 = 净值虚高 5bp/次。
    """
    out, _ = cost.charge(_trades([("A.SZ", "SELL", 1_000_000.0, 8e9, 0.0288)]), CostConfig())
    r = out.iloc[0]
    assert r["commission"] == pytest.approx(250.0, abs=TOL)
    assert r["stamp_duty"] == pytest.approx(500.0, abs=TOL)
    assert r["transfer_fee"] == pytest.approx(10.0, abs=TOL)
    assert r["total_cost"] == pytest.approx(761.8, rel=1e-12)   # +1.8 冲击，同上一条


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

    ADV20 与振幅都从**同一个 20 日窗口**取（这正是 Task 13 引擎要做的事，
    见 cost.py 的「谁来给 ADV20」）：
      · ADV20 = 信号日 T 往前 20 个交易日 amount 的均值，**剔除停牌日的 0**；
      · range = 同一窗口 (high−low)/pre_close 的均值，同样剔停牌
        （2026-08-22 改口径：原来取的是执行日掩码的 amplitude，那是前视）。
    一笔吃掉 ADV20 大半的单子必须被 30bp 封住，一笔小单必须落在公式上。
    """
    from ashare.data import query
    query.open_db(market_db)
    try:
        sig = dt.date(2024, 1, 31)
        bars = query.get_bars(sig, ["A00001.SZ"], lookback=20,
                              fields=("amount", "high", "low", "pre_close"))
        amt = bars["amount"].astype(float)
        adv20 = float(amt[amt > 0].mean())          # ★ 停牌占位行 amount=0，不算进 ADV
        # ★ 停牌占位行的价格在 get_bars 出口已置 NaN，dropna() 即「剔停牌」
        rng = float(((bars["high"] - bars["low"]) / bars["pre_close"]).dropna().mean())
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


def _bar_window():
    """20 根 bar，其中 3 根是 D9 的停牌占位行（`get_bars` 出口把它们的价格置 NaN）。

    这是 Task 13 引擎侧口径的**可执行版本**：`range` = 这 20 根里非停牌行的
    `(high−low)/pre_close` 均值。执行日当天那根**不在**窗口里。
    """
    susp = (5, 11, 12)
    rows = []
    for i in range(20):
        p = 55.037188 + 0.31 * i
        if i in susp:
            rows.append((np.nan, np.nan, np.nan))
            continue
        h = 0.4137 + 0.0113 * i
        rows.append((p + h / 2, p - h / 2, p))
    return pd.DataFrame(rows, columns=["high", "low", "pre_close"])


def test_range_is_the_trailing_twenty_day_mean_not_the_execution_day_amplitude():
    """★ §5.4（2026-08-21 裁决）：振幅取**滞后 20 日均值**，不取执行日当天。

    09:25 集合竞价成交时当天的 high/low 还没走出来，用它就是前视 —— 而且**可被利用**：
    `volatility_60` 的 direction 是 −1，账本系统性偏好低波动的票，低波动的票当日振幅
    也低，于是成本模型读到的正是因子在下注的那份未来；闸 5 拿净额收益调参时会顺着走。

    本条把口径本身钉成可执行的（Task 13 照抄）：同一批 bar 上
      · 滞后均值 0.008961 → 冲击 210.79 元
      · 执行日当天 0.047444（消息日，5.3 倍）→ 1116.01 元
      · 把停牌占位行按 0 算进均值 0.007617 → 179.17 元（低估 15%）
    三个数互不相同，喂错哪一个都看得出来。
    """
    bars = _bar_window()
    amp = (bars["high"] - bars["low"]) / bars["pre_close"]
    assert int(amp.notna().sum()) == 17                     # fixture 自检：3 根占位行
    rng = float(amp.dropna().mean())
    rng_exec = 2.9137 / 61.4137                             # 执行日那根：一个消息日
    rng_zero_filled = float(amp.fillna(0.0).mean())         # 占位行当 0 算进去
    assert rng == pytest.approx(0.008961282257317, abs=1e-15)
    assert rng_exec > 2.0 * rng                             # 两个口径真的分得开（5.3 倍）
    assert rng_zero_filled < rng                            # 剔停牌不是可选项

    adv20, amt = 3.7194e8, 4_183_071.0
    out, warns = cost.charge(_trades([("A.SZ", "BUY", amt, adv20, rng)]), CostConfig())
    assert out["impact"].iloc[0] == pytest.approx(210.793757924241, rel=1e-12)
    assert warns == []
    # 换成另外两个口径，成本立刻不是这个数 —— 这才叫「口径被钉住了」
    for wrong, want in ((rng_exec, 1116.007621173173), (rng_zero_filled, 179.174694235605)):
        alt, _ = cost.charge(_trades([("A.SZ", "BUY", amt, adv20, wrong)]), CostConfig())
        assert alt["impact"].iloc[0] == pytest.approx(want, rel=1e-12)
        assert abs(alt["impact"].iloc[0] - out["impact"].iloc[0]) > 1.0


def test_unknown_adv20_charges_the_cap_and_says_so():
    """ADV20 缺失 → 按【封顶】计冲击 + 告警，绝不按 0。

    静默按 0 是「流动性未知 ⇒ 免费成交」，方向恰好错在把净值画好看那一侧；
    按上限计只会让结果变差，且一定看得见（与 `get_tradable_mask` 的
    `limit_unknown → 两侧不可交易` 同一个保守方向）。

    ★ 第四笔的 `range` **恰好是 0**，走的是同一条封顶路径（`rng > 0` 而不是 `>= 0`）：
      20 日振幅全为 0 意味着这 20 天每天 high==low —— 连续一字板，或整段停牌剔干净后
      无样本。那是「根本成交不了」，不是「零波动所以零冲击」。旧实现读成后者，
      恰好又倒向把净值画好看那一侧；改口径成滞后均值之后这条路径才真正走得到
      （执行日当天振幅为 0 只有一字板一种，20 日均值为 0 还多了「无有效样本」一种）。
    """
    tr = _trades([("A.SZ", "BUY", 813_402.0, float("nan"), 0.0431),
                  ("B.SZ", "SELL", 621_907.0, 0.0, 0.0288),
                  ("C.SZ", "BUY", 402_311.0, 3.1e8, float("nan")),
                  ("D.SZ", "BUY", 517_233.0, 2.7e8, 0.0)])
    out, warns = cost.charge(tr, CostConfig())
    assert out["impact"].to_numpy() == pytest.approx((0.0030 * out["amount"]).to_numpy(), rel=1e-12)
    assert len(warns) == 1 and "4 笔成交缺" in warns[0]


def test_a_cost_config_that_zeroes_or_rebates_the_fee_table_is_refused():
    """`CostConfig` 是 frozen dataclass 但不做任何校验，这道闸只能在 `charge` 里补。

    `multiplier=0` 产出一张**全零费用表** —— 一次「零成本」的回测在报告上只表现为
    「策略特别好」，不抛、不告警；负费率更进一步，变成交易返现，把净值画得更漂亮。
    闸 4（成本敏感）正是靠改 `multiplier` 工作的，手滑写 0 与写 2.0 一样容易。
    """
    tr = _trades([("A.SZ", "SELL", 1_000_000.0, 8e9, 0.0288)])
    for bad in (CostConfig(multiplier=0.0), CostConfig(multiplier=-1.0),
                CostConfig(impact_coef=-0.5), CostConfig(stamp_duty_bps=-5.0)):
        with pytest.raises(ValueError, match="CostConfig"):
            cost.charge(tr, bad)


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


def test_cash_plus_heterogeneous_returns_on_a_non_round_book():
    """★ 三个承重件同时上场的唯一一条：现金（Σw≠1）+ 异质收益 + **除不尽的 Σw**。

    齐涨那两条各自只压住一个：满仓 fixture 里 Σw 恰好 1.0000、未满仓那条恰好 0.8000，
    而分母 `1+ΣwR` 在齐涨下退化成 `1+0.10·Σw` —— 漂移、分母、逐票拆分三者当中
    任何两个的错误都能互相抵消。这里 Σw₀ = 0.7333、收益 (+7.31%, −4.18%, +11.29%)，
    三种常见错法给出三个互不相同的数：
      · 漏归一化分母 → 0.030906（差 0.8%，靠除不尽的数才分得开）
      · 漏漂移       → 0.075700（2.4 倍）
      · 用组合平均收益当每只的漂移 → 0.073191
    """
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.2913, 0.3164], "B.SZ": [0.2547, 0.2211],
               "C.SZ": [0.1873, 0.2043]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.0731], "B.SZ": [np.nan, -0.0418],
               "C.SZ": [np.nan, 0.1129]}, dates)
    to, warns = metrics.turnover_series(w, r)
    assert float(w.iloc[0].sum()) == pytest.approx(0.7333, abs=1e-12)    # fixture 自检
    assert to.iloc[0] == pytest.approx(0.7333, abs=1e-12)   # 首期从现金建仓 = Σw，不是 1
    assert to.iloc[1] == pytest.approx(0.031148419745210, abs=1e-14)
    for wrong in (0.030905680000000, 0.075700000000000, 0.073190748721994):
        assert to.iloc[1] != pytest.approx(wrong, rel=1e-6)
    assert warns == []


def test_a_held_name_without_a_return_is_reported_not_silently_drifted_by_zero():
    dates = [dt.date(2024, 1, 5), dt.date(2024, 1, 12)]
    w = _wide({"A.SZ": [0.4831, 0.4831], "B.SZ": [0.5169, 0.5169]}, dates)
    r = _wide({"A.SZ": [np.nan, 0.0217], "B.SZ": [np.nan, np.nan]}, dates)
    _, warns = metrics.turnover_series(w, r)
    assert any("B.SZ" in w_ or "漂移" in w_ for w_ in warns)


# ══════════════ §9 净值 / 相对指标 ══════════════
# ★ 末尾三日反弹（2026-08-22 补）是【必需】的，不是把曲线画长一点：
#   没有它，曲线的**全局最高点就是回撤的起点**（idx 4 = 1.032993），此后再没涨回去 ——
#   于是 `1 − V/cummax(V)` 与 `1 − V/max(V)` 在整条曲线上**逐位相同**（都是 0.09628949…），
#   而 cummax 就是最大回撤的全部内容（MDD 又喂给 Calmar，两者都是 brief 点名要手算的数）。
#   加上反弹后最高点落在**末尾**（idx 23），全局 max 的实现给 0.09964266，差 0.0034，
#   那个变异才有地方死。
_R = [0.0137, -0.0208, 0.0091, 0.0313, -0.0074, -0.0161,
      -0.0212, -0.0183, -0.0247, -0.0126,
      0.0117, 0.0205, -0.0092, 0.0061, 0.0183, 0.0074,
      -0.0119, 0.0288, -0.0056, 0.0132,
      0.0117, 0.0089, 0.0063]
_RB = [0.0102, -0.0171, 0.0148, 0.0261, 0.0013, -0.0094,
       -0.0155, -0.0121, -0.0203, -0.0067,
       0.0089, 0.0247, -0.0038, 0.0104, 0.0126, 0.0031,
       -0.0072, 0.0219, -0.0011, 0.0178,
       0.0141, 0.0106, 0.0079]      # 基准同期涨得更多 → IR 仍为负（那条测试钉的是符号）

# 回测本金（`BacktestConfig.initial_capital`）。**故意不取默认的 1_000_000**：
# `compute` 拿它把货币口径的 `total_cost` 换算成净值口径的比例（架构 §4.3 的 2026-08-21
# 裁决），取默认值的话「把 initial_capital 写死成 1e6」这个变异与真实现逐位相同。
_CAP = 1_923_517.0


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

    ★ 最大回撤【跨 6 个交易日复合】(9.6289%)，大于任何单日跌幅 (2.47%)：
      「MDD = 最大单日亏损」这个变异必须死在这里。
    """
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                               full=False, initial_capital=_CAP)
    assert m["annual_return"] == pytest.approx(0.486538765768243, abs=1e-12)
    assert m["annual_vol"] == pytest.approx(0.261814550123034, abs=1e-12)
    assert m["sharpe"] == pytest.approx(1.858333562972744, abs=1e-12)
    assert m["max_drawdown"] == pytest.approx(0.096289495984263, abs=1e-12)
    assert m["calmar"] == pytest.approx(5.052874779277721, abs=1e-12)
    assert m["max_drawdown"] > 0.0247 + 1e-6            # ← 不是最大单日亏损
    assert any("基准" in w for w in warns)              # 没给基准 → 相对指标缺失要说


def test_max_drawdown_runs_off_the_running_peak_not_the_global_maximum():
    """★ MDD 的全部内容就是 `cummax`：回撤要从**当时的**高点算起，不是从事后的最高点。

    这条曲线的最高点在**末尾**（末三日反弹），所以：
      · 正确（cummax，峰在 idx 4）→ 0.09628949598426251
      · 全局 max（峰在 idx 23）  → 0.09964266419155199 —— 把还没发生的高点当成了起点
    差 0.0034。旧 fixture 里最高点恰好【就是】回撤起点，两个实现逐位相同，
    这个变异因此一直是逃逸的；而 MDD 又是 Calmar 的分母，错了两个数一起错。
    """
    v = _curve(_R)
    assert v.to_numpy().argmax() == len(v) - 1          # fixture 自检：峰在末尾
    m, _ = metrics.compute(v, pd.DataFrame(), _empty_positions(), None,
                           full=False, initial_capital=_CAP)
    naive = float((1.0 - v.to_numpy() / v.to_numpy().max()).max())
    assert naive == pytest.approx(0.099642664191552, abs=1e-12)
    assert m["max_drawdown"] == pytest.approx(0.096289495984263, abs=1e-12)
    assert m["max_drawdown"] < naive - 1e-4


def test_information_ratio_is_negative_when_the_book_trails_the_benchmark():
    """IR 的【符号】必须跟着超额收益走。符号翻转是一个不抛异常、且报告照样漂亮的变异。"""
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), _curve(_RB),
                           full=False, initial_capital=_CAP)
    assert m["information_ratio"] == pytest.approx(-7.848763163186034, abs=1e-10)
    assert m["information_ratio"] < 0


def test_no_benchmark_leaves_ir_none_and_warns():
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                               full=False, initial_capital=_CAP)
    assert m["information_ratio"] is None
    assert any("基准" in w for w in warns)


def test_risk_free_rate_lowers_sharpe():
    """§9 的 Sharpe 是 (R_ann − R_f)/σ。R_f 默认 0，给了就必须真的减掉。"""
    a, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                           full=False, initial_capital=_CAP)
    b, _ = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                           full=False, initial_capital=_CAP, risk_free=0.0217)
    assert b["sharpe"] == pytest.approx(a["sharpe"] - 0.0217 / a["annual_vol"], abs=1e-12)


def test_a_single_point_equity_curve_has_no_metrics_at_all():
    """一个点算不出任何一期收益 —— 全部指标是 NaN，不是 0。

    ★ 这条同时是 `compute` 里两处 `if years > 0 else nan`（换手年化 / 成本拖累）的守卫：
      正因为有这个早返回，steps ≥ 1、years 恒为正，那两处今天走不到。按 2026-08-21 的
      等价变异裁决它们留着（要走两步推理、又隔着三四十行，删了下一个人会看错），
      而这条测试钉住的是让它们成为死代码的那个前提本身。
    """
    one = _curve([])            # 只有起点 1.0
    assert len(one) == 1
    m, warns = metrics.compute(one, pd.DataFrame(), _empty_positions(), None,
                               full=True, initial_capital=_CAP)
    assert m["n_days"] == 1
    for k in ("annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar"):
        assert np.isnan(m[k]), k
    assert m["information_ratio"] is None
    assert "turnover_annual" not in m and "cost_drag_annual" not in m
    assert any("不足两个点" in w for w in warns)


def test_blank_days_in_the_equity_curve_are_reported_with_their_effect_on_annualisation():
    """`dropna()` 删掉的是**日子**，而年化的分母正是走过的步数 —— 静默剔除 = 系统性高估。

    24 个交易日里挖掉 5 天：年化收益 0.4865 → 0.6596（+35%），
    而所有 warning 一字不变、`n_days` 也只是小一点。缺值本身可能没得救
    （净值那天真算不出来），但「年化被这件事抬高了」必须写在报告里。
    """
    eq = _curve(_R).copy()
    eq.iloc[[2, 5, 9, 13, 17]] = np.nan
    m, warns = metrics.compute(eq, pd.DataFrame(), _empty_positions(), None,
                               full=False, initial_capital=_CAP)
    assert m["n_days"] == 19
    assert m["annual_return"] == pytest.approx(0.659598265515683, abs=1e-12)
    hit = [w for w in warns if "缺值" in w]
    assert len(hit) == 1
    assert "5" in hit[0] and "年化" in hit[0]        # 数量 + 后果，两样都要说


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
                           None, full=True, initial_capital=_CAP)
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
                               _positions_frame(rows, _POS_COLS), None,
                               full=True, initial_capital=_CAP)
    assert any("B.SZ" in w for w in warns)


def _cost_ledger(eq) -> pd.DataFrame:
    """三笔成交、共 7742.01 元，落在净值曲线的第 3 天与第 8 天。"""
    return pd.DataFrame({"exec_date": [eq.index[3], eq.index[3], eq.index[8]],
                         "total_cost": [3172.41, 1908.07, 2661.53]})


def test_cost_drag_lands_in_the_three_to_six_percent_band_of_section_five_four():
    """★ 成本拖累必须是**比例**，§5.4 的量级是 3%–6%/年。

    `charge` 的 `total_cost` 跟着 `simulate` 的【货币】权益（那边要算
    shares = Δw·equity/price），而本函数收到的 `equity` 是初始 1.0 的【净值指数】。
    两者直接相除得到的是钱不是比例 —— 实测 85654.77，也就是「一年亏掉本金的八万倍」。

    ★ 这条测试的前一版是【自己把实现的表达式抄了一遍】（`c = cost/eq.iloc[3] + …`），
      于是它连同那个量纲错误一起绿了两轮：断言与实现共用同一个错。
      判据必须是物理量级 —— 一个能被 §5.4 独立证伪的数，不是实现的复读。
      本金取 `_CAP`（非 1e6），「把 initial_capital 写死」的变异也一并死在这里。
    """
    eq = _curve(_R)
    m, warns = metrics.compute(eq, _cost_ledger(eq), _positions_frame(_POS_ROWS, _POS_COLS),
                               None, full=True, initial_capital=_CAP)
    assert 0.03 < m["cost_drag_annual"] < 0.06                      # ← §5.4 的独立判据
    assert m["cost_drag_annual"] == pytest.approx(0.044530288275565, rel=1e-12)
    assert m["cost_total"] == pytest.approx(7742.01, abs=1e-9)      # 总额仍是【货币】
    assert not any("成交日" in w for w in warns)
    assert not any("量纲" in w or "100%" in w for w in warns)


def test_the_dimensional_guard_fires_when_the_two_curves_are_not_on_the_same_scale():
    """量纲守卫：>100%/年的成本拖累永远不是真结果，而这是「两条曲线不同量纲」唯一的信号。

    传 `initial_capital=1.0` 就等于退回未修复前的除法。成本模型接没接对，本来就只有
    这一个便宜的体检指标；没有守卫时它只是报告上一个大得离谱、没人会去核的数。
    """
    eq = _curve(_R)
    m, warns = metrics.compute(eq, _cost_ledger(eq), _positions_frame(_POS_ROWS, _POS_COLS),
                               None, full=True, initial_capital=1.0)
    assert m["cost_drag_annual"] == pytest.approx(85654.766512950600, rel=1e-12)
    assert any("100%" in w for w in warns)


def test_turnover_mean_and_annual_are_the_hand_computed_values():
    """★ 这是读者拿去对 `PortfolioConstraints.max_turnover = 0.30`（周频双边）的那个数。

    两条都必须钉住具体值，只断言"存在"的话下面两个变异都能活：
      · **丢掉首期从现金建仓的那 1.0** —— 均值 0.7411 → 0.4822，年化少掉三分之二；
      · **年化 = 均值 × periods_per_year** —— 252 是【净值】的日频，换手是调仓频，
        乘出来 186.75 而不是 16.24，大了 11.5 倍（§9 的口径是「总换手 / 年数」）。
    """
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _positions_frame(_POS_ROWS, _POS_COLS),
                           None, full=True, initial_capital=_CAP)
    assert m["turnover_mean"] == pytest.approx(0.741077674456063, rel=1e-12)
    assert m["turnover_annual"] == pytest.approx(16.239267301124155, rel=1e-12)


def test_full_true_without_any_positions_says_the_two_blocks_are_missing():
    """`full=True` 却拿不到账本时，换手与 D6 缺口这两组数会**整组消失**。

    报告上少两行看不出是「整段没有调仓」还是「账本根本没传进来」—— 前者是策略结论，
    后者是接线错误，而缺省的表现完全一样。
    """
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(), _empty_positions(), None,
                               full=True, initial_capital=_CAP)
    assert "turnover_mean" not in m and "d6_slippage_mean" not in m
    assert any("positions 为空" in w for w in warns)


def test_a_trade_dated_off_the_equity_curve_is_reported():
    eq = _curve(_R)
    tr = pd.DataFrame({"exec_date": [dt.date(1999, 1, 4)], "total_cost": [1234.5]})
    _, warns = metrics.compute(eq, tr, _positions_frame(_POS_ROWS, _POS_COLS), None,
                               full=True, initial_capital=_CAP)
    assert any("1999-01-04" in w for w in warns)


def test_full_false_skips_the_trade_block():
    m, _ = metrics.compute(_curve(_R), pd.DataFrame(), _positions_frame(_POS_ROWS, _POS_COLS),
                           None, full=False, initial_capital=_CAP)
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
                               full=True, initial_capital=_CAP, factors_used=fu)
    assert m["factors_used_min"] == 2
    assert m["factors_used_max"] == 14
    assert m["factors_used_median"] == pytest.approx(13.5)
    assert m["n_periods_below_half"] == 1
    assert any("2024-01-19" in w for w in warns)
    # 没给 n_factors_configured 就只能拿观测最大值当分母 —— 这件事本身也要说出来
    assert any("n_factors_configured" in w for w in warns)
    assert "n_factors_configured" not in m


def test_uniform_degradation_is_invisible_without_the_configured_factor_count():
    """★ 分母必须是【配置了几个】，不是【观测到的最多几个】。

    14 个因子里废了 12 个、每一期都靠剩下 2 个把整个账本调一遍 —— 这是最常见的降级形态
    （一个因子少一列就是**每期**都少，不是个别期的数据缺口）。拿观测最大值当分母时
    max = 2、没有任何一期低于 max/2，`n_periods_below_half = 0`，报告干干净净。
    """
    idx = [_D1, _D2, dt.date(2024, 1, 19), dt.date(2024, 1, 26)]
    fu = pd.Series([2, 2, 2, 2], index=idx)
    m, warns = metrics.compute(_curve(_R), pd.DataFrame(),
                               _positions_frame(_POS_ROWS, _POS_COLS), None,
                               full=True, initial_capital=_CAP,
                               factors_used=fu, n_factors_configured=14)
    assert m["n_factors_configured"] == 14
    assert m["factors_used_max"] == 2
    assert m["n_periods_below_half"] == 4            # 拿 max=2 当分母时这里是 0
    assert any("配置了 14 个因子，全期最多只用上 2 个" in w for w in warns)
    assert any("4 期只用了不到一半的因子" in w for w in warns)

    # 同一组数据、不给分母 → 四期全部"健康"，一条降级告警都没有（这正是要防的那一幕）
    m2, warns2 = metrics.compute(_curve(_R), pd.DataFrame(),
                                 _positions_frame(_POS_ROWS, _POS_COLS), None,
                                 full=True, initial_capital=_CAP, factors_used=fu)
    assert m2["n_periods_below_half"] == 0
    assert not any("全期最多只用上" in w or "不到一半" in w for w in warns2)


def test_missing_factors_used_is_itself_reported():
    """不给这一项 = 部分降级不可见。那就必须在 warnings 里说出来。"""
    _, warns = metrics.compute(_curve(_R), pd.DataFrame(),
                               _positions_frame(_POS_ROWS, _POS_COLS), None,
                               full=True, initial_capital=_CAP)
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


def test_names_without_a_forward_return_are_dropped_from_both_sides_of_the_cross_section():
    """★ 有效性掩码必须**两侧同时**取 —— 只掩因子那一侧、再把收益 `fillna(0)`，
    等于给一只没有持有期收益的票编造一个"这期不赚不亏"，凭空造出 IC。

    更坏的是覆盖率闸随之失明：`_MIN_IC_OBS` 会在**没收缩过的**横截面上判，
    于是 10 只里 6 只没有收益的一天照样过闸，然后拿 4 只算出一个 IC 报上去。
    这个变异实测通过了全部 51 条旧用例。
    """
    fp = _panel([_D1], _CODES, {"f": _SC})
    fr = pd.Series(_RT, index=fp.index).copy()
    fr.iloc[[1, 3, 4, 6, 7, 9]] = np.nan                    # 10 只里 6 只没有持有期收益
    ic, warns = metrics.ic_series(fp, fr)
    assert np.isnan(ic["f__ic"].iloc[0]) and np.isnan(ic["f__rank_ic"].iloc[0])
    assert any("有效横截面 4 只" in w for w in warns)        # ← 判在【剩下的 4 只】上
    # 反面：只缺 5 只时剩 5 只，恰好够 `_MIN_IC_OBS`，必须算得出来
    fr2 = pd.Series(_RT, index=fp.index).copy()
    fr2.iloc[[1, 3, 4, 6, 7]] = np.nan
    ic2, warns2 = metrics.ic_series(fp, fr2)
    assert np.isfinite(ic2["f__rank_ic"].iloc[0]) and warns2 == []


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
    """★ n=2 是 `min(n−1, ·)` 那道上夹**最紧**的一点：公式给 floor(1.6768)=1，n−1 也是 1。

    再往上 n−1 涨得比 4(T/100)^{2/9} 快得多，所以上夹在 n≥2 上恒不生效 —— 拆掉它是一个
    **等价变异**（跑过，存活）。按 2026-08-21 的裁决它留着：「floor(4(T/100)^{2/9}) ≤ T−1
    对所有 T≥2 成立」不是读者扫一眼能确认的事，删了下一个人会以为 lag 可以超过样本长度。
    这里把最紧的那两点钉进表里，至少让它有据可查。
    """
    for n, lag in ((2, 1), (3, 1), (52, 3), (100, 4), (118, 4), (250, 4), (500, 5), (10, 2)):
        assert metrics.newey_west_lag(n) == lag, n
    assert metrics.newey_west_lag(1) == 0 and metrics.newey_west_lag(0) == 0


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


def test_the_icir_annualisation_frequency_is_a_parameter_not_a_constant():
    """52 只是**默认值**（IC 是调仓频）。`layer_monotonicity` 早就收 `periods_per_year`，
    邻居可配而这里写死，正是任务报告拿来挑 §9 毛病的那种自相矛盾。
    月频调仓的策略把 52 当 12 用，ICIR_ann 会大 2.08 倍。"""
    out, _ = metrics.icir(_autocorrelated_ic(), periods_per_year=12)
    assert out["icir_ann"] == pytest.approx(out["icir"] * math.sqrt(12), abs=1e-12)
    assert out["icir_ann"] != pytest.approx(out["icir"] * math.sqrt(52), rel=1e-6)


def test_an_ic_mean_above_zero_point_one_five_is_flagged_as_probable_look_ahead():
    """★ §4.2「声称 RankIC > 0.15 的，先去查前视偏差」—— 这句话原先没有任何实现。

    它是 `ic_series` **结构上看不见**的那一类错的唯一探测器：调用方若把 R_t 标成 t 日
    （而不是 t→t+1），索引照样对得齐、每期横截面照样够、常数闸也过 —— 算出来的是
    "当期收益解释当期因子"，IC 高得离谱而全链路一声不吭。全市场单因子周频的现实量级是
    0.02–0.06，`icir` 手上正好有 mean，这道闸只能长在这里。
    """
    t = np.arange(60)
    aligned = pd.Series(0.037 + 0.021 * np.sin(2 * np.pi * t / 17.0))
    out, warns = metrics.icir(aligned)
    assert abs(out["mean"]) < 0.15
    assert not any("前视" in w for w in warns)

    # 同一条序列整体抬到 0.31 —— 对齐错了的典型量级
    out2, warns2 = metrics.icir(aligned + 0.273)
    assert out2["mean"] == pytest.approx(out["mean"] + 0.273, abs=1e-12)
    hit = [w for w in warns2 if "前视" in w]
    assert len(hit) == 1 and "0.15" in hit[0]


def test_dropped_non_finite_periods_are_counted_not_just_dropped():
    """780 周掉 300 周是真实运行里的常态（横截面不足 / 全常数，`ic_series` 置的 NaN），
    而报告上只会出现一个 n=480。剩下多少要说，掉了多少也要说。"""
    ic = _autocorrelated_ic(40)
    ic.iloc[[3, 7, 11, 19, 28, 33, 37]] = np.nan
    out, warns = metrics.icir(ic)
    assert out["n"] == 33
    hit = [w for w in warns if "非有限" in w]
    assert len(hit) == 1 and "40" in hit[0] and "7" in hit[0] and "33" in hit[0]


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


def test_a_period_with_no_layers_is_dropped_from_the_exponent_and_counted():
    """★ 年化的期数用的是**该层实际有值的期数**（`dropna()` 之后），不是表的行数。

    横截面不足的那一天整行是 NaN（`layered_returns` 就是这么写的），而全部 fixture 都是
    满表 —— 于是 `len(col)` 与 `len(lay)` 逐位相同，「用表的行数当分母」这个变异一直逃逸，
    同一个 fixture 缺口还让 `n_dropped` 那条告警从未被触发过。
    38 期算年化给 0.1087（真值），39 期给 0.10577 —— 差 2.7%，方向是系统性低估。
    """
    ann = [0.0113, 0.0217, 0.0331, 0.0429, 0.0538, 0.0641, 0.0759, 0.0863, 0.0971, 0.1087]
    lay = _layer_frame(ann)
    lay.iloc[7] = np.nan                        # 那一天有效横截面 < 10 层
    out, warns = metrics.layer_monotonicity(lay)
    assert out["layer_annual"]["L10"] == pytest.approx(0.1087, rel=1e-9)
    assert out["layer_annual"]["L1"] == pytest.approx(0.0113, rel=1e-9)
    assert out["layer_annual"]["L10"] != pytest.approx(0.105770423203089, rel=1e-6)
    assert any("1 期分层全空" in w for w in warns)


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
    """残差 = β×size → size 行的暴露 = 两期 β 的 Fama-MacBeth 均值，t 用 Newey-West。

    ★ 两期的载荷【必须不同】（1.37 与 0.42）。原 fixture 用 `np.tile(...,2)` 把同一个
      横截面铺了两遍，于是 b1 = [1.37, 1.37] —— 均值、首项、末项、中位数是同一个数，
      「`arr.mean()` → `arr[0]`」这个变异与真实现逐位相同。t_stat 更是全文件一次都没断言过。
    ★ 这个 t 是 §3.2「OLS 而非 √MV-WLS」那条裁决**唯一**能被证伪的仪器。
      「残差有 0.895 的规模暴露」不是证据，「0.895，NW t = 3.77」才是 ——
      前者可以是噪声，后者可以被拒绝。
    ★ 朴素标准误在这里给 t = 1.88（NW 的一半）。方向与 IC 序列那条相反是对的：
      两期载荷是负自相关，Bartlett 核如实地把方差收窄了。要的不是「NW 一定更小」，
      是「NW 与朴素给出的不是同一个数」。
    """
    n, dates = 40, [_D1, _D2]
    codes = [f"{i:06d}.SZ" for i in range(n)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["rebalance_date", "ts_code"])
    raw = np.linspace(19.13, 26.87, n)
    z = (raw - raw.mean()) / raw.std(ddof=1)
    sc = pd.Series(np.concatenate([1.37 * z, 0.42 * z]), index=idx)
    pos = pd.DataFrame({"filled_weight": np.full(2 * n, 1.0 / n),
                        "industry": ["银行"] * (2 * n)}, index=idx)
    tbl, _ = metrics.attribution(pos, pd.Series(0.0, index=idx), sc,
                                 size=pd.Series(np.tile(raw, 2), index=idx))
    row = tbl[(tbl["block"] == "style") & (tbl["item"] == "size")].iloc[0]
    assert row["exposure"] == pytest.approx(0.895, rel=1e-9)        # (1.37+0.42)/2
    assert row["exposure"] != pytest.approx(1.37, rel=1e-3)         # ← 不是第一期的值
    assert row["t_stat"] == pytest.approx(3.768421052631579, rel=1e-9)
    assert row["t_stat"] != pytest.approx(1.884210526315789, rel=1e-3)   # ← 不是朴素 SE
    assert np.isfinite(row["t_stat"])                               # ← 也不是"懒得算就 NaN"
    assert np.isnan(row["contribution"])        # 风格块只报暴露，贡献留给行业块


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
    """★ `exposure` 也必须是 NaN，不是 0。

    这一列的语义是【被换手预算裁掉的调仓规模】，`0.0` 读起来就是「预算一点没裁」——
    恰好是这一行存在的目的所要防的那个误读，而且它长得和一个合法结果一模一样。
    只断言 contribution 的话，「算不出来就填 0」这个变异从 exposure 那半溜过去。
    """
    pos, fwd, sc, size = _attr_inputs(intended=False)
    tbl, warns = metrics.attribution(pos, fwd, sc, size=size)
    row = tbl[tbl["block"] == "constraint"].iloc[0]
    assert np.isnan(row["contribution"])
    assert np.isnan(row["exposure"])
    assert any("intended_weight" in w for w in warns)


def test_scores_that_are_only_the_book_are_flagged_as_the_wrong_instrument():
    """★ §3.2 的证伪仪器指错了对象，会照样给出一个看起来很正常的小数字。

    `scores` 必须是**全股票池**的合成分数。只传账本里那几十只时，style/size 量到的是
    【账本自己的规模倾斜】，而 §3.2 问的是「残差在整个横截面上还是不是规模的代理」。
    `portfolio.build_targets` 有一模一样的一道闸（scores 短于 industry ⇒ 调用方先
    dropna 过了，覆盖率闸失明），这里照它的样子补。
    """
    pos, fwd, sc, size = _attr_inputs()                     # sc 与 pos 同索引 = 就是账本
    _, warns = metrics.attribution(pos, fwd, sc, size=size)
    assert any("全股票池" in w for w in warns)

    # 全池：给它一批**不在账本里**的票，行数超过 positions → 不再告警
    extra = pd.MultiIndex.from_product([_ATTR_DATES, [f"X{i:04d}.SZ" for i in range(60)]],
                                       names=["rebalance_date", "ts_code"])
    sc_all = pd.concat([sc, pd.Series(np.linspace(-1.7, 1.9, len(extra)), index=extra)])
    size_all = pd.concat([size, pd.Series(np.linspace(19.13, 26.87, len(extra)), index=extra)])
    _, warns2 = metrics.attribution(pos, fwd, sc_all, size=size_all)
    assert not any("全股票池" in w for w in warns2)


def test_a_holding_without_a_forward_return_is_counted_as_zero_contribution_not_as_missing():
    """★ 措辞就是这条测试的全部内容：`(w*R).groupby().sum()` **跳过** NaN。

    「按缺失计」读作"不知道"，`0` 读作"这只票这一期不赚不亮" —— 后者会把该行业的
    contribution 往 0 拉，而 exposure 一分不少。两列在这些行上口径不一致，
    告警必须说出是哪一种，否则读者会拿一个被稀释过的贡献去和别的行业比。
    """
    pos, fwd, sc, size = _attr_inputs()
    fwd = fwd.copy()
    fwd.iloc[[0, 2]] = np.nan                               # 两只银行股第一期没有持有期收益
    tbl, warns = metrics.attribution(pos, fwd, sc, size=size)
    row = tbl[(tbl["block"] == "industry") & (tbl["item"] == "银行")].iloc[0]
    # exposure 照算全部四行；contribution 只剩后两行 —— 贡献被摊薄，暴露没有
    assert row["exposure"] == pytest.approx((0.2731 + 0.2503 + 0.2117 + 0.2641) / 2, abs=1e-12)
    assert row["contribution"] == pytest.approx(
        (0.2117 * 0.0119 + 0.2641 * -0.0071) / 2, abs=1e-12)
    hit = [w for w in warns if "持有期收益" in w]
    assert len(hit) == 1 and "按 0 计入" in hit[0] and "2" in hit[0]


def test_attribution_columns_are_fixed():
    pos, fwd, sc, size = _attr_inputs()
    tbl, _ = metrics.attribution(pos, fwd, sc, size=size)
    assert list(tbl.columns) == metrics.ATTRIBUTION_COLS
