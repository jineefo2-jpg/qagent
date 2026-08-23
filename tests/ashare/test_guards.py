"""Task 14：防自欺五闸（`ashare/backtest/guards.py`，算法说明书 §8）。

这五个函数是整个平台存在的理由：前面十三个任务产出一条净值曲线，这里判断那条曲线
算不算数。**一道该拦没拦的闸比没有这道闸更坏** —— 它把一条烂策略洗成"走过严格流程"的
结论。所以本文件的用例分成两类，第二类才是重点：

  · 「该过的过」—— 平常路径，一条就够；
  · 「该拦的拦」—— 每一条都对着一个具体的失效模式，且那个失效模式**看起来像成功**：
      · 样本内 Sharpe 为负时，`SR_oos ≥ 0.6·SR_is` 会把 −0.2 判成"守住了 60%"；
      · 真实 Sharpe 是 NaN 时，`SR_b ≥ NaN` 恒为假 → p = 1/(n+1) → **噪声被判显著**；
      · 置换里出现 NaN 时，NaN 不计入 `#{SR_b ≥ SR_real}` → p 被系统性压小；
      · 邻域平均 Sharpe ≤ 0 时，PeakRatio 变成负数 → `< 1.3` → **孤峰过闸**；
      · θ* 自己混进邻域时，均值被自己拉高 → PeakRatio 变小 → **尖峰过闸**；
      · 闸 4 的 multiplier 没传下去时，压力测试跑的是基线本身 —— 且结果完全正常。

★ 闸 3 的两条硬约束各有一条用例，都不能用"看起来像打乱了"糊弄过去：
  · **p 值分布**：真随机分数下 p 必须近似均匀（`test_p_values_are_uniform_...`）。
    任何偏置（置换不真随机 / `>` 与 `≥` 写反 / 少一个 +1）都会让这个闸对**每一条**
    被检验的策略把噪声判成显著，且永远不会有人发现。
  · **同日横截面内置换**：驱动【真引擎】，spy 记下每次置换后的分数，断言其
    `trade_date` 集合恒为当天（分数值里编了日期，跨天置换会把别的日子的值带进来）。

fixture 的数字一律取除不尽的值（global-constraints ★）。两处例外是**边界**用例
（PeakRatio 恰为 1.3、SR_oos 恰为 0.6·SR_is）：那里要的正是逐位相等，
`<` 与 `≤` 的区别只在这一个点上看得见，用除不尽的数就永远测不到它。
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from ashare.backtest import engine, execution, guards
from ashare.backtest.guards import (GateResult, gate1_out_of_sample, gate2_walk_forward,
                                    gate3_shuffle, gate4_cost_stress, gate5_param_plateau,
                                    run_all_gates)
from ashare.backtest.types import BacktestConfig, CostConfig, PortfolioConstraints
from ashare.data.query import QueryError

SNAP = "snap-9c4e17"
IS_END = dt.date(2019, 12, 31)


# ══════════════ 假回测：gates 只读 metrics / warnings / 两个 D7 指纹 ══════════════

@dataclass
class FakeRun:
    """`BacktestResult` 里 gates 真正读到的那一小块。字段少是有意的：
    多一个字段就多一条"闸依赖了结果的什么"的暗线。"""
    metrics: dict
    warnings: list
    param_hash: str
    data_snapshot_id: str
    engine_version: str
    config: BacktestConfig


def fake_run(cfg, *, sharpe=1.0, ir=0.5, warnings=(), engine_version="p2-engine-1") -> FakeRun:
    return FakeRun(metrics={"sharpe": sharpe, "information_ratio": ir},
                   warnings=list(warnings), param_hash=cfg.param_hash(),
                   data_snapshot_id=SNAP, engine_version=engine_version, config=cfg)


class Runner:
    """记录每次调用的 cfg。`fn(cfg, i)` 给出该次的结果。"""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list = []

    def __call__(self, cfg, **kw):
        self.calls.append(cfg)
        return self.fn(cfg, len(self.calls) - 1)

    @property
    def n(self) -> int:
        return len(self.calls)


@pytest.fixture
def runner(monkeypatch):
    def install(fn) -> Runner:
        r = Runner(fn)
        monkeypatch.setattr(guards, "run_backtest", r)
        return r
    return install


class _Spec:
    """`get_factor` 的最小替身：闸 2/闸 5 只问 `default_params`（θ* 的注册默认值）。"""

    def __init__(self, **default_params):
        self.default_params = dict(default_params)
        self.lookback_days = 250


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    specs = {"mom": _Spec(window=60), "rev": _Spec(window=20)}
    monkeypatch.setattr(guards, "get_factor", lambda n: specs[n])
    return specs


def make_cfg(**kw) -> BacktestConfig:
    base = dict(start=dt.date(2010, 1, 1), end=dt.date(2024, 12, 31),
                factors=(("mom", 1.0), ("rev", 0.7)),
                constraints=PortfolioConstraints(top_n=37, max_turnover=0.29),
                initial_capital=3_141_593.0)
    base.update(kw)
    return BacktestConfig(**base)


def only(runner_, **where) -> list:
    """按 (start, end) 挑出调用。"""
    return [c for c in runner_.calls
            if all(getattr(c, k) == v for k, v in where.items())]


# ══════════════ 闸 1 · 样本外单次检验 ══════════════

def test_out_of_sample_splits_at_the_2019_boundary_and_runs_each_side_once(runner):
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=(1.3717 if i == 0 else 0.9013)))
    res = gate1_out_of_sample(make_cfg())

    assert isinstance(res, GateResult) and res.name == "gate1"
    assert [(c.start, c.end) for c in r.calls] == [
        (dt.date(2010, 1, 1), IS_END), (dt.date(2020, 1, 1), dt.date(2024, 12, 31))]
    assert res.detail["in_sample"]["sharpe"] == 1.3717
    assert res.detail["out_of_sample"]["sharpe"] == 0.9013
    assert res.passed is True


def test_exactly_six_tenths_of_the_in_sample_sharpe_still_passes(runner):
    """§8 写的是 `SR_oos ≥ 0.6·SR_is`。`>` 与 `≥` 只在这一个点上分得开，
    所以 fixture 在这里**必须**逐位相等（用除不尽的数就永远走不到边界）。"""
    sr_is = 1.3717
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=(sr_is if i == 0 else 0.6 * sr_is)))
    res = gate1_out_of_sample(make_cfg())

    assert res.detail["threshold"] == 0.6 * sr_is
    assert res.passed is True
    assert r.n == 2


def test_a_hair_below_six_tenths_fails(runner):
    sr_is = 1.3717
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=(sr_is if i == 0 else 0.6 * sr_is - 1e-9)))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is False
    assert r.n == 2


def test_a_non_positive_in_sample_sharpe_cannot_certify_anything(runner):
    """样本内 Sharpe ≤ 0 时 `0.6 × SR_is` 是个**负**门槛：SR_oos = −0.2 会
    "≥ −0.30" 地通过。策略在样本内就不赚钱，这道闸没有任何可比的基准。"""
    runner(lambda cfg, i: fake_run(cfg, sharpe=(-0.5031 if i == 0 else -0.2017)))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is False
    assert "样本内" in res.note


@pytest.mark.parametrize("bad_side", [0, 1])
def test_a_non_finite_sharpe_is_not_a_pass_and_says_so(runner, bad_side):
    """NaN 那一侧要报「算不出」。少了有限性守卫时 NaN 会掉进「样本内 Sharpe ≤ 0」那一支 ——
    结论碰巧还是"不过"，但理由是错的（NaN 既不 ≤ 0 也不 > 0），下一个读报告的人会去查
    一个根本不存在的负 Sharpe。"""
    runner(lambda cfg, i: fake_run(cfg, sharpe=(float("nan") if i == bad_side else 0.9013)))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is False
    assert "算不出" in res.note


def test_two_infinite_sharpes_do_not_certify_each_other(runner):
    """`inf ≥ 0.6 × inf` 为真 —— 有限性守卫**唯一**独占的那条路径。
    ann_vol=0 时 `metrics` 给的是 NaN 不是 inf，所以这条路今天走不到；
    钉住它是为了让守卫别被当成冗余删掉（2026-08-21 等价变异裁决：留守卫 + 补测试）。"""
    runner(lambda cfg, i: fake_run(cfg, sharpe=float("inf")))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is False
    assert "算不出" in res.note


def test_engine_version_is_recorded_beside_the_fingerprint_never_inside_it(runner):
    """§8 闸 1「分家」那一侧：`engine_version` 进哈希 = 每次引擎升级白送一次样本外机会。
    两次运行给不同的 engine_version，闸报出的 `param_hash` 必须一个字节都不变。"""
    cfg = make_cfg()
    r = runner(lambda cfg_, i: fake_run(cfg_, sharpe=1.3717,
                                        engine_version=f"p2-engine-{i + 7}"))
    res = gate1_out_of_sample(cfg)

    is_cfg, oos_cfg = r.calls
    assert res.detail["in_sample"]["param_hash"] == is_cfg.param_hash()
    assert res.detail["out_of_sample"]["param_hash"] == oos_cfg.param_hash()
    # 并列记录（台账要写），但两个指纹都不含它
    assert res.detail["out_of_sample"]["engine_version"] == "p2-engine-8"
    assert res.detail["out_of_sample"]["data_snapshot_id"] == SNAP
    assert is_cfg.param_hash() != oos_cfg.param_hash()      # 区间不同 = 两次不同的运行


def test_gate1_needs_data_past_the_out_of_sample_boundary(runner):
    r = runner(lambda cfg, i: fake_run(cfg))
    res = gate1_out_of_sample(make_cfg(end=dt.date(2018, 6, 29)))

    assert res.passed is False and r.n == 0
    assert "2019-12-31" in res.note


def test_gate1_surfaces_the_duplicate_fingerprint_warning_from_the_ledger(runner):
    """`append_oos_run` 发现同指纹会喊「⚠ D7 重复指纹」。闸 1 判过而这条不见了，
    等于把污染洗掉 —— 它必须出现在返回值里。"""
    dup = "⚠ D7 重复指纹：param_hash=abc + data_snapshot_id=snap 已在样本外台账里出现过"
    runner(lambda cfg, i: fake_run(cfg, sharpe=(1.3717 if i == 0 else 1.2011),
                                   warnings=[dup] if i == 1 else []))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is True
    assert dup in res.detail["warnings"]
    assert "⚠" in res.note


# ══════════════ 闸 2 · Walk-forward ══════════════

# 逐折 × 逐参数的训练集 Sharpe。argmax 逐折不同，且**与测试集的 argmax 不同** ——
# 「用测试集选参数」这个泄漏一旦发生，选出来的 θ̂ 会变，用例立刻红。
_TRAIN_SR = {0: {20: 0.9013, 60: 1.4177, 250: 0.7331},
             1: {20: 0.8117, 60: 1.3313, 250: 1.0937},
             2: {20: 1.2231, 60: 1.1873, 250: 0.6619},
             3: {20: 0.7717, 60: 1.5041, 250: 1.1109},
             4: {20: 0.9931, 60: 1.2777, 250: 0.8803}}
_TEST_SR = {0: {20: 2.7183, 60: 0.3010, 250: 2.3026},
            1: {20: 2.6180, 60: 0.4771, 250: 2.1972},
            2: {20: 0.6931, 60: 0.5772, 250: 2.9957},
            3: {20: 2.4849, 60: 0.6021, 250: 2.0794},
            4: {20: 2.1401, 60: 0.7781, 250: 1.9459}}

_FOLDS = [(dt.date(2010 + k, 1, 1), dt.date(2014 + k, 12, 31),
           dt.date(2015 + k, 1, 1), dt.date(2015 + k, 12, 31)) for k in range(5)]


def _wf_runner(window_of_fold=None):
    """按 (start, end) 判断这次是选参跑还是测试跑，给出对应的 Sharpe。"""
    def fn(cfg, i):
        w = cfg.factor_param_override.get("mom", {}).get("window", 60)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                return fake_run(cfg, sharpe=_TRAIN_SR[k][w])
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=_TEST_SR[k][w])
        raise AssertionError(f"闸 2 跑了一个既不是训练窗也不是测试窗的区间: "
                             f"{cfg.start}~{cfg.end}")
    return fn


def test_walk_forward_selects_the_parameter_on_the_training_window_only(runner):
    """泄漏就发生在这条缝上：θ̂ 必须是**训练集** Sharpe 的 argmax。
    fixture 让两边的 argmax 逐折都不同，拿测试集选参会选出另一个数。"""
    r = runner(_wf_runner())
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [60, 60, 20, 60, 60]
    # 每折：3 次选参（都在训练窗内）+ 1 次测试
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
        sel = only(r, start=tr_s, end=tr_e)
        assert len(sel) == 3
        assert {c.factor_param_override["mom"]["window"] for c in sel
                if c.factor_param_override} <= {20, 250}      # 60 = 注册默认，不写 override
        test_calls = only(r, start=te_s, end=te_e)
        assert len(test_calls) == 1
        got = test_calls[0].factor_param_override.get("mom", {}).get("window", 60)
        assert got == res.detail["folds"][k]["theta"]["mom.window"]


def test_walk_forward_never_lets_a_selection_run_see_past_its_training_window(runner):
    r = runner(_wf_runner())
    gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    for c in r.calls:
        fold = next(f for f in _FOLDS if c.end in (f[1], f[3]))
        assert c.end <= fold[3]
        if c.end == fold[1]:                    # 选参跑：绝不许碰测试窗的任何一天
            assert c.end < fold[2]


def test_walk_forward_stays_inside_the_in_sample_period(runner):
    """折窗必须钉在样本内。在 2020+ 上滚动着选参数，本身就是在样本外调参（D7）——
    而且每一折都会往 `docs/oos-runs.md` 里写一行，把真正那一行埋掉。"""
    r = runner(_wf_runner())
    gate2_walk_forward(make_cfg(end=dt.date(2024, 12, 31)),
                       grid={"mom.window": [20, 60, 250]})

    assert r.n == 20                                    # 5 折 × (3 选参 + 1 测试)
    assert max(c.end for c in r.calls) == IS_END


def test_an_order_of_magnitude_parameter_jump_fails_the_walk_forward(runner):
    """§8 的例子：动量窗口在 20 与 250 之间反复横跳即为不稳定。"""
    jumpy = {0: {20: 2.7183, 60: 0.9013, 250: 0.3010},
             1: {20: 0.4771, 60: 0.8117, 250: 2.6180},
             2: {20: 2.3026, 60: 1.0937, 250: 0.5772},
             3: {20: 0.6931, 60: 1.1109, 250: 2.4849},
             4: {20: 2.1401, 60: 0.9931, 250: 0.7781}}

    def fn(cfg, i):
        w = cfg.factor_param_override.get("mom", {}).get("window", 60)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                return fake_run(cfg, sharpe=jumpy[k][w])
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=0.5)
        raise AssertionError("窗口不对")

    r = runner(fn)
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [20, 250, 20, 250, 20]
    assert res.detail["ranges"]["mom.window"]["ratio"] == 12.5
    assert res.passed is False
    assert r.n == 20


def test_a_parameter_that_barely_moves_passes_the_walk_forward(runner):
    steady = {0: {54: 0.9013, 60: 1.4177, 78: 0.7331},
              1: {54: 1.3313, 60: 1.0937, 78: 0.8117},
              2: {54: 0.6619, 60: 1.2231, 78: 1.1873},
              3: {54: 0.7717, 60: 1.5041, 78: 1.1109},
              4: {54: 1.2777, 60: 0.9931, 78: 0.8803}}

    def fn(cfg, i):
        w = cfg.factor_param_override.get("mom", {}).get("window", 60)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                return fake_run(cfg, sharpe=steady[k][w])
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=0.4321)
        raise AssertionError("窗口不对")

    runner(fn)
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [54, 60, 78]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [60, 54, 60, 60, 54]
    assert res.detail["ranges"]["mom.window"]["ratio"] == pytest.approx(60 / 54, rel=0, abs=1e-12)
    assert res.passed is True
    # 测试集 Sharpe 是 walk-forward 的产出，不能因为不参与判定就不产出
    assert [f["test_sharpe"] for f in res.detail["folds"]] == [0.4321] * 5


def test_the_walk_forward_stability_threshold_bites_at_exactly_three_times(runner):
    """阈值 3.0 是**本实现**选的（§8 只给了 20 ↔ 250 这个例子，没给数），
    所以它必须被钉住：60/20 恰好 3.0 算稳定，把阈值收到 2.9 同一组 θ̂ 就不过。
    这里刻意用逐位精确的 3.0 —— `>` 与 `≥` 只在这一个点上分得开。"""
    sr = {0: {20: 1.4177, 60: 0.9013}, 1: {20: 0.8117, 60: 1.3313},
          2: {20: 1.2231, 60: 0.6619}, 3: {20: 0.7717, 60: 1.5041},
          4: {20: 1.2777, 60: 0.9931}}

    def fn(cfg, i):
        w = cfg.factor_param_override.get("mom", {}).get("window", 60)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                return fake_run(cfg, sharpe=sr[k][w])
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=0.4321)
        raise AssertionError("窗口不对")

    runner(fn)
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60]})
    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [20, 60, 20, 60, 20]
    assert res.detail["ranges"]["mom.window"]["ratio"] == 3.0
    assert res.passed is True

    runner(fn)
    assert gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60]},
                              max_param_ratio=2.9).passed is False


def test_a_non_numeric_parameter_falls_back_to_exact_agreement(runner):
    """`weighting` 这种取值没有"量级"，max/min 无从算起 —— 判据退化成"每折都选了同一个"。
    不退化的话这一格会直接抛 TypeError，把一道闸的结论变成一条异常。"""
    sr = {"equal": 0.9013, "risk_parity": 1.4177}

    def fn(cfg, i):
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                w = sr[cfg.constraints.weighting]
                return fake_run(cfg, sharpe=(w if k % 2 else 1.9013 - w))
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=0.4321)
        raise AssertionError("窗口不对")

    runner(fn)
    res = gate2_walk_forward(make_cfg(), grid={"weighting": ["equal", "risk_parity"]})

    assert [f["theta"]["weighting"] for f in res.detail["folds"]] == [
        "equal", "risk_parity", "equal", "risk_parity", "equal"]
    assert res.detail["ranges"]["weighting"]["ratio"] is None
    assert res.passed is False


def test_walk_forward_without_a_grid_is_reported_as_not_run(runner):
    r = runner(_wf_runner())
    res = gate2_walk_forward(make_cfg())

    assert res.passed is False and r.n == 0
    assert "网格" in res.note


def test_walk_forward_needs_at_least_one_complete_fold(runner):
    r = runner(_wf_runner())
    res = gate2_walk_forward(make_cfg(start=dt.date(2017, 1, 1)),
                             grid={"mom.window": [20, 60]})

    assert res.passed is False and r.n == 0
    assert "折" in res.note


# ══════════════ 闸 3 · Shuffle 置换检验 ══════════════

def _shuffle_runner(real, perms):
    """第 0 次是真回测，其后依次是置换对照。"""
    def fn(cfg, i):
        return fake_run(cfg, sharpe=(real if i == 0 else perms[i - 1]))
    return fn


def test_shuffle_p_value_counts_ties_and_keeps_both_plus_ones(runner):
    """`p = (1 + #{b: SR_b ≥ SR_real}) / (n + 1)`。三处都能独立地把噪声判成显著：
    `>` 漏掉持平的那次、分子少 1、分母少 1。"""
    real = 1.4142
    perms = [real, 2.7183, 1.6180, 0.5772, -0.3010, 0.1234, 0.9876, -1.1111, 0.4321]
    runner(_shuffle_runner(real, perms))
    res = gate3_shuffle(make_cfg(), n=9, seed=11)

    assert res.detail["n_ge"] == 3                  # 含那次持平
    assert res.detail["p_value"] == 0.4             # (1+3)/(9+1)
    assert res.passed is False


def test_shuffle_rejects_the_null_when_the_score_predicts_the_future(runner):
    real = 2.9013
    perms = [0.4771 - 0.01 * i for i in range(39)]
    runner(_shuffle_runner(real, perms))
    res = gate3_shuffle(make_cfg(), n=39, seed=3)

    assert res.detail["n_ge"] == 0
    assert res.detail["p_value"] == 0.025           # 1/40
    assert res.passed is True


def test_p_values_are_uniform_under_a_truly_random_score(runner, monkeypatch):
    """★ 本任务最重要的一条。真随机分数下 H0 成立，`SR_real` 与 200 个 `SR_b`
    可交换，于是 p 必须在 [0,1] 上近似均匀。**任何**偏置都会让这个闸对每一条
    被检验的策略把噪声判成显著，而且没有任何症状。

    n = 39 是有意的：p 的取值格点是 k/40，`p < 0.05` 等价于 `p = 1/40`，
    H0 下的真实拒绝率因此正好是 2.5%（n = 19 时最小的 p 就是 0.05，
    严格小于永远不成立，这条用例会退化成恒真）。
    """
    rng = np.random.default_rng(20260822)
    ps: list = []
    for _ in range(200):
        r = Runner(lambda cfg, i: fake_run(cfg, sharpe=float(rng.normal())))
        monkeypatch.setattr(guards, "run_backtest", r)
        ps.append(gate3_shuffle(make_cfg(), n=39, seed=int(rng.integers(1 << 30))
                                ).detail["p_value"])

    arr = np.array(ps)
    # 经验分布函数逐点贴住 y = x（离散均匀的理论值就是 x），200 个样本 σ ≈ 0.035
    for x in (0.25, 0.5, 0.75):
        assert abs(float((arr <= x).mean()) - x) <= 0.10, f"p 值在 {x} 处偏离均匀"
    assert 0.01 <= float((arr < 0.05).mean()) <= 0.15    # 真值 5%
    assert arr.min() < 0.10 and arr.max() > 0.90         # 没有塌缩到某一端
    assert 0.42 <= float(arr.mean()) <= 0.58


def test_a_p_value_of_exactly_five_percent_does_not_pass(runner):
    """§8 的通过标准是 `p < 0.05`。n = 19 时 p 的最小取值恰好是 1/20 = 0.05，
    `<` 与 `≤` 只在这一个点上分得开 —— 这里刻意让它落在那一点上。"""
    runner(_shuffle_runner(2.9013, [0.4771 - 0.01 * i for i in range(19)]))
    res = gate3_shuffle(make_cfg(), n=19, seed=1)

    assert res.detail["n_ge"] == 0
    assert res.detail["p_value"] == 0.05
    assert res.passed is False


def test_shuffle_forces_diagnostics_off_on_every_single_run(runner):
    """架构 A3：200 次 × 60 s = 3.3 小时，这道闸就没人跑了。
    `is False` 而不是 `not ...`：`compute_diagnostics` 只有两种取值，
    但"顺手传了个 0"与"真的关了"在报告里长得一样。"""
    r = runner(_shuffle_runner(1.0, [0.5] * 5))
    gate3_shuffle(make_cfg(compute_diagnostics=True), n=5, seed=2)

    assert r.n == 6
    assert all(c.compute_diagnostics is False for c in r.calls)


def test_every_permutation_gets_its_own_seed_and_nothing_else_changes(runner):
    """闸 3 把"怎么打乱"整个交给引擎的 `shuffle_seed`（引擎只在同日横截面内置换）。
    如果它还顺手改了别的字段，对照组就不再是同一个策略的零假设样本。"""
    cfg = make_cfg(compute_diagnostics=True)
    r = runner(_shuffle_runner(1.0, [0.5] * 7))
    gate3_shuffle(cfg, n=7, seed=41)

    base = r.calls[0]
    assert base.shuffle_seed is None
    seeds = [c.shuffle_seed for c in r.calls[1:]]
    assert len(set(seeds)) == 7 and None not in seeds
    for c in r.calls[1:]:
        assert replace(c, shuffle_seed=None) == base       # 只差一个 seed


def test_a_config_that_already_carries_a_shuffle_seed_is_reset_for_the_baseline(runner):
    """基线若自己就是一次置换，闸 3 拿噪声当"真实值"去比噪声 —— 检验的对象整个错了，
    而 p 值看起来完全正常。降级必须可见：重跑之外还要出一条告警。"""
    r = runner(_shuffle_runner(1.4142, [0.5] * 4))
    res = gate3_shuffle(make_cfg(shuffle_seed=99), n=4, seed=7)

    assert r.calls[0].shuffle_seed is None
    assert any("shuffle_seed=99" in w for w in res.detail["warnings"])


def test_a_non_finite_real_sharpe_can_never_be_declared_significant(runner):
    """`SR_b ≥ NaN` 恒为假 → n_ge = 0 → p = 1/(n+1) = 0.005 → **显著**。
    一条算不出 Sharpe 的净值曲线会被这道闸判成"击败了全部 200 个对照"。"""
    r = runner(_shuffle_runner(float("nan"), [0.5] * 20))
    res = gate3_shuffle(make_cfg(), n=20, seed=5)

    assert res.passed is False
    assert res.detail.get("p_value") is None
    assert r.n == 1                                  # 真实 Sharpe 都没有，不必再跑 20 次


def test_non_finite_permutation_sharpes_count_against_significance(runner):
    """置换里的 NaN 不能悄悄从分子里消失 —— 那个方向恰好是把 p 压小。"""
    real = 1.0
    perms = [1.7321, 2.2361, 1.4142, float("nan"), float("nan"),
             0.3010, 0.4771, 0.6021, 0.6990]
    runner(_shuffle_runner(real, perms))
    res = gate3_shuffle(make_cfg(), n=9, seed=13)

    assert res.detail["n_nonfinite"] == 2
    assert res.detail["n_ge"] == 5                   # 3 个真的更高 + 2 个算不出的
    assert res.detail["p_value"] == 0.6              # (1+5)/10
    assert any("非有限" in w for w in res.detail["warnings"])


# ── 同日横截面内置换：驱动真引擎 ──────────────────────────────────────────────

_G_DAYS = [dt.date(2024, 5, 6) + dt.timedelta(days=i) for i in range(16)]
_G_WEEKLY = _G_DAYS[4::5]                       # 3 个调仓日
_G_CODES = ["AAA.SZ", "BBB.SZ", "CCC.SH", "DDD.SZ"]
_G_BASE = {"AAA.SZ": 55.037188, "BBB.SZ": 13.884723, "CCC.SH": 87.219461, "DDD.SZ": 7.302957}
_G_OPEN_K = {"AAA.SZ": 0.993713, "BBB.SZ": 1.004271, "CCC.SH": 0.987619, "DDD.SZ": 1.011533}
_G_IND = {"AAA.SZ": "银行", "BBB.SZ": "白酒", "CCC.SH": "钢铁", "DDD.SZ": "煤炭"}
# 分数值里编进了调仓日的序号（十位以上）：跨天置换会把别的日子的值带进来，一眼看得出
_G_FRAC = {"AAA.SZ": 1.618034, "BBB.SZ": 2.718282, "CCC.SH": 3.141593, "DDD.SZ": 0.577216}


def _g_close(code: str, d: dt.date) -> float:
    return _G_BASE[code] * (1.00713 ** _G_DAYS.index(d)) + 0.0031 * _G_DAYS.index(d)


def _g_open(code: str, d: dt.date) -> float:
    return _g_close(code, d) * (_G_OPEN_K[code] + 0.000317 * _G_DAYS.index(d))


class _GuardQuery:
    """驱动真引擎所需的最小 `query` 表面（4 只票、16 个交易日、无停牌无退市）。"""

    QueryError = QueryError

    def snapshot_id(self, *, pin: bool = False) -> str:
        return SNAP

    def preload(self, start, end, tables=()):
        pass

    def get_trade_dates(self, as_of_date, *, start=None, freq="D"):
        days = [d for d in _G_DAYS if d <= as_of_date and (start is None or d >= start)]
        return days if freq == "D" else [d for d in _G_WEEKLY if d in days]

    def next_trade_date(self, as_of_date, n: int = 1):
        later = [d for d in _G_DAYS if d > as_of_date]
        return later[n - 1] if len(later) >= n else None

    def get_universe(self, as_of_date, **k):
        return list(_G_CODES)

    def get_industry(self, as_of_date, ts_codes=None, level="l1", **k):
        codes = list(ts_codes if ts_codes is not None else _G_CODES)
        return pd.Series([_G_IND[c] for c in codes], index=codes, name="sw_l1")

    def get_price_panel(self, as_of_date, ts_codes, field="close", lookback=250, adjust="hfq"):
        get = {"close": _g_close, "open": _g_open}[field]
        days = [d for d in _G_DAYS if d <= as_of_date][-lookback:]
        return pd.DataFrame({c: [get(c, d) for d in days] for c in ts_codes},
                            index=days).rename_axis("trade_date")

    def get_bars(self, as_of_date, ts_codes, *, lookback=None, start=None,
                 fields=("open", "high", "low", "close", "vol", "amount"), adjust="hfq"):
        days = [d for d in _G_DAYS if d <= as_of_date]
        days = days[-lookback:] if lookback else days
        rows, idx = [], []
        for c in ts_codes:
            for d in days:
                px = _g_close(c, d)
                rows.append({"open": _g_open(c, d), "high": px * 1.02341, "low": px * 0.98117,
                             "close": px, "pre_close": px * 0.99411, "vol": 913_517.0,
                             "amount": 8_137_211.0 + 1000.0 * _G_DAYS.index(d),
                             "is_suspended": False})
                idx.append((c, d))
        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
            idx, names=["ts_code", "trade_date"]))
        return df[[*fields, "is_suspended"]]

    def get_index_bars(self, as_of_date, index_code, lookback=250, fields=("close",)):
        days = [d for d in _G_DAYS if d <= as_of_date][-lookback:]
        return pd.DataFrame({"close": [3011.37 * (1.00291 ** i) for i in range(len(days))]},
                            index=days).rename_axis("trade_date")[list(fields)]

    def get_tradable_mask(self, exec_date, ts_codes):
        rows = [(c, True, True, "", _g_open(c, exec_date), _g_close(c, exec_date), 9.1e6, 0.02137)
                for c in ts_codes]
        return pd.DataFrame(rows, columns=["ts_code", "can_buy", "can_sell", "reason",
                                           "open_hfq", "close_hfq", "amount", "amplitude"]
                            ).set_index("ts_code")


def test_the_permutation_stays_inside_one_days_cross_section(monkeypatch):
    """★ 跨时间置换会破坏市场整体涨跌的时序结构 → 对照组分布失真 → 检验结论无效。

    分数值里编了调仓日序号，spy 记下**置换之后**交给 `build_targets` 的那一列：
    每次置换的 `trade_date` 集合必须恒为当天那一个。
    """
    q = _GuardQuery()
    runs: list = []

    def fake_combine(weights, as_of_date, universe):
        k = _G_WEEKLY.index(as_of_date)
        return pd.Series([10.0 * k + _G_FRAC[c] for c in universe],
                         index=list(universe), name="score", dtype=float), []

    real_bt = engine.build_targets

    def spy_targets(scores, *a, **kw):
        runs[-1].append(scores.copy())
        return real_bt(scores, *a, **kw)

    monkeypatch.setattr(engine, "query", q)
    monkeypatch.setattr(execution, "get_tradable_mask", q.get_tradable_mask)
    monkeypatch.setattr(engine, "combine", fake_combine)
    monkeypatch.setattr(engine, "get_factor", lambda n: _Spec())
    monkeypatch.setattr(engine, "build_targets", spy_targets)
    monkeypatch.setattr(engine, "append_oos_run", lambda r: (False, []))

    def tracking(cfg, **kw):
        runs.append([])
        return engine.run_backtest(cfg, **kw)

    monkeypatch.setattr(guards, "run_backtest", tracking)

    cfg = BacktestConfig(start=_G_DAYS[0], end=_G_DAYS[-1], factors=(("mom", 1.0),),
                         constraints=PortfolioConstraints(top_n=2, max_single=0.6,
                                                          max_industry=1.0, max_turnover=0.37),
                         initial_capital=3_141_593.0, compute_diagnostics=False)
    gate3_shuffle(cfg, n=2, seed=17)

    assert len(runs) == 3                       # 1 次真回测 + 2 次置换，闸没有提前退出
    for r in runs:
        assert len(r) == len(_G_WEEKLY)
    for j, scores in enumerate(runs[0]):        # 真回测这一遍原样不动
        assert list(scores.to_numpy()) == [10.0 * j + _G_FRAC[c] for c in _G_CODES]

    permuted_any = False
    for run in runs[1:]:
        for j, scores in enumerate(run):
            assert list(scores.index) == _G_CODES               # 池子没动
            days = {int(v // 10) for v in scores.to_numpy()}    # 值里编着的调仓日序号
            assert days == {j}, f"置换跨到了别的调仓日: {days}"
            same_day = sorted(10.0 * j + _G_FRAC[c] for c in _G_CODES)
            assert sorted(scores.to_numpy()) == same_day        # 只是换了顺序
            permuted_any |= list(scores.to_numpy()) != [10.0 * j + _G_FRAC[c] for c in _G_CODES]
    assert permuted_any, "两次置换 × 两个调仓日都是恒等排列 —— 根本没打乱"


# ══════════════ 闸 4 · 成本敏感性 ══════════════

def test_cost_stress_doubles_the_multiplier_on_the_way_down(runner):
    r = runner(lambda cfg, i: fake_run(cfg, ir=0.3717))
    res = gate4_cost_stress(make_cfg())

    assert r.n == 1
    assert r.calls[0].cost.multiplier == 2.0
    assert res.detail["stressed"]["cost_multiplier"] == 2.0
    assert res.detail["stressed"]["information_ratio"] == 0.3717
    assert res.passed is True


def test_cost_stress_doubles_whatever_the_config_already_had(runner):
    """§8 说的是"成本参数整体**翻倍**"，不是"设成 2.0"。基线本来就调过成本时，
    设成 2.0 可能反而是**减压**（基线 3.0 → 2.0）。"""
    r = runner(lambda cfg, i: fake_run(cfg, ir=0.2011))
    gate4_cost_stress(make_cfg(cost=CostConfig(multiplier=1.5)))

    assert r.calls[0].cost.multiplier == 3.0


def test_cost_stress_must_be_a_different_d7_fingerprint_than_the_baseline(runner):
    r = runner(lambda cfg, i: fake_run(cfg, ir=0.3717))
    cfg = make_cfg()
    res = gate4_cost_stress(cfg)

    assert r.calls[0].param_hash() != cfg.param_hash()
    assert res.detail["baseline_param_hash"] == cfg.param_hash()
    assert res.detail["stressed"]["param_hash"] == r.calls[0].param_hash()


def test_a_stress_that_does_not_change_the_fingerprint_is_no_stress_at_all(runner):
    """multiplier=1.0 产出与基线**逐位相同**的一次运行，报告上完全正常 ——
    「闸 4 跑过了」于是变成一句谎话。指纹一样就必须当场判死。"""
    r = runner(lambda cfg, i: fake_run(cfg, ir=0.9013))
    res = gate4_cost_stress(make_cfg(), multiplier=1.0)

    assert res.passed is False and r.n == 0
    assert "指纹" in res.note


def test_cost_stress_fails_when_doubling_eats_the_excess(runner):
    runner(lambda cfg, i: fake_run(cfg, ir=-0.0731))
    assert gate4_cost_stress(make_cfg()).passed is False


def test_zero_excess_is_not_positive_excess(runner):
    runner(lambda cfg, i: fake_run(cfg, ir=0.0))
    assert gate4_cost_stress(make_cfg()).passed is False


def test_cost_stress_cannot_judge_without_a_benchmark(runner):
    """`metrics.compute` 拿不到基准时 `information_ratio` 是 None。
    "算不出超额"不是"超额为正"。"""
    runner(lambda cfg, i: fake_run(cfg, ir=None))
    res = gate4_cost_stress(make_cfg())

    assert res.passed is False
    assert "基准" in res.note or "超额" in res.note


# ══════════════ 闸 5 · 参数高原 ══════════════

def _plateau_runner(star, neighbours: dict):
    def fn(cfg, i):
        w = cfg.factor_param_override.get("mom", {}).get("window")
        return fake_run(cfg, sharpe=(star if w is None else neighbours[w]))
    return fn


def test_a_spike_in_parameter_space_fails_the_plateau_gate(runner):
    """SR* = 2.713，邻域均值 2.00603 → PeakRatio 1.3524 ≥ 1.3。
    邻域取"均值"而不是中位数/最大值：这组数在中位数（1.2911）与最大值（1.0037）下都会**过闸**。"""
    r = runner(_plateau_runner(2.713, {42: 1.2137, 78: 2.1013, 90: 2.7031}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78, 90]})

    assert r.n == 4                                      # θ* + 3 个邻域点
    assert res.detail["mean_neighbour"] == pytest.approx(2.0060333333333333, rel=0, abs=1e-12)
    assert res.detail["peak_ratio"] == pytest.approx(1.35242020, rel=0, abs=1e-8)
    assert res.passed is False


def test_a_flat_plateau_passes(runner):
    """这组数在"取邻域最小值"下 PeakRatio = 1.357 会误判成尖峰。"""
    runner(_plateau_runner(1.2231, {42: 0.9013, 78: 1.4177, 90: 1.3313}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78, 90]})

    assert res.detail["peak_ratio"] == pytest.approx(1.0052046, rel=0, abs=1e-6)
    assert res.passed is True


def test_a_peak_ratio_of_exactly_one_point_three_fails(runner):
    """§8 的通过标准是 `PeakRatio < 1.3`。`<` 与 `≤` 只在这一点上分得开，
    所以这里刻意用逐位精确的数（0.75 + 1.25 = 2.0，均值恰为 1.0）。"""
    runner(_plateau_runner(1.3, {42: 0.75, 78: 1.25}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78]})

    assert res.detail["peak_ratio"] == 1.3
    assert res.passed is False


def test_theta_star_is_never_counted_inside_its_own_neighbourhood(runner):
    """±30% 网格**必然**包含 θ* 自己（60 就是注册默认值）。把它算进邻域会把均值
    朝 SR* 拉高、PeakRatio 变小 —— 只会把尖峰洗成高原。这组数正好跨过 1.3：
    排除 θ* 时 1.3524（不过），算进去 1.2273（过）。"""
    r = runner(_plateau_runner(2.713, {42: 1.2137, 78: 2.1013, 90: 2.7031}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 60, 78, 90]})

    assert r.n == 4                                     # 60 = θ*，不重复跑
    assert sorted(res.detail["neighbourhood"]) == ["mom.window=42", "mom.window=78",
                                                   "mom.window=90"]
    assert res.passed is False


def test_a_neighbourhood_that_loses_money_cannot_certify_a_plateau(runner):
    """邻域平均 Sharpe ≤ 0 时 PeakRatio 是**负数**，`< 1.3` 恒真 ——
    一座周围全是坑的孤峰会拿到满分。"""
    runner(_plateau_runner(1.9013, {42: -0.7331, 78: -0.4177, 90: -0.2011}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78, 90]})

    assert res.passed is False
    assert "邻域" in res.note


def test_the_plateau_grid_stays_inside_the_in_sample_window(runner):
    """在样本外扫 ±30% 网格 = 拿样本外调参（D7）。"""
    r = runner(_plateau_runner(1.2231, {42: 1.1873, 78: 1.3041}))
    gate5_param_plateau(make_cfg(end=dt.date(2024, 12, 31)), {"mom.window": [42, 78]})

    assert {c.end for c in r.calls} == {IS_END}


def test_the_grid_sweeps_into_factor_param_override_with_distinct_fingerprints(runner):
    r = runner(_plateau_runner(1.2231, {42: 1.1873, 78: 1.3041}))
    gate5_param_plateau(make_cfg(), {"mom.window": [42, 78]})

    assert [c.factor_param_override for c in r.calls] == [
        {}, {"mom": {"window": 42}}, {"mom": {"window": 78}}]
    assert len({c.param_hash() for c in r.calls}) == 3


def test_a_grid_key_that_names_nothing_is_rejected(runner):
    """键写错 = 网格扫的是空气，而闸照样给出一个 PeakRatio。"""
    runner(_plateau_runner(1.0, {}))
    for bad in ({"mom.windwo": [42, 78]}, {"top_nn": [30, 40]}, {"nosuch.window": [1, 2]}):
        res = gate5_param_plateau(make_cfg(), bad)
        assert res.passed is False and "网格" in res.note


def test_the_grid_can_also_sweep_a_portfolio_constraint(runner):
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=(1.2231 if cfg.constraints.top_n == 37
                                                    else 1.1873)))
    res = gate5_param_plateau(make_cfg(), {"top_n": [26, 37, 48]})

    assert [c.constraints.top_n for c in r.calls] == [37, 26, 48]
    assert res.passed is True


# ══════════════ run_all_gates ══════════════

def _all_pass(monkeypatch):
    """把五个闸都替成"过"，便于单独考察编排本身。"""
    for g in guards.GATE_NAMES:
        monkeypatch.setitem(guards.GATES, g,
                            lambda cfg, _n=g, **kw: GateResult(_n, True, {}, "ok"))


def test_every_gate_always_appears_in_the_result(monkeypatch):
    _all_pass(monkeypatch)
    out = run_all_gates(make_cfg())

    assert list(out) == list(guards.GATE_NAMES)
    assert all(r.passed for r in out.values())


def test_unselected_gates_are_reported_as_not_run_never_omitted(monkeypatch):
    """U5：悄悄不返回读起来就是"没问题"。"""
    _all_pass(monkeypatch)
    out = run_all_gates(make_cfg(), gates=["gate1"])

    assert list(out) == list(guards.GATE_NAMES)
    assert out["gate1"].passed is True
    for g in ("gate2", "gate3", "gate4", "gate5"):
        assert out[g].passed is False
        assert out[g].note == "未运行"


def test_a_failing_gate_does_not_stop_the_ones_after_it(monkeypatch):
    """操作员要一次看到全貌 —— 前面挂了就不跑后面的话，报告里"没跑"与"过了"长得一样。"""
    _all_pass(monkeypatch)
    monkeypatch.setitem(guards.GATES, "gate1",
                        lambda cfg, **kw: GateResult("gate1", False, {}, "挂了"))
    out = run_all_gates(make_cfg())

    assert [r.passed for r in out.values()] == [False, True, True, True, True]


def test_a_crashing_gate_is_reported_as_a_failure_not_swallowed(monkeypatch):
    _all_pass(monkeypatch)

    def boom(cfg, **kw):
        raise ZeroDivisionError("邻域均值为 0")

    monkeypatch.setitem(guards.GATES, "gate3", boom)
    out = run_all_gates(make_cfg())

    assert out["gate3"].passed is False
    assert out["gate3"].note != "未运行"
    assert "ZeroDivisionError" in out["gate3"].note and "邻域均值为 0" in out["gate3"].note
    assert all(out[g].passed for g in ("gate1", "gate2", "gate4", "gate5"))


def test_an_unknown_gate_name_is_rejected_up_front(monkeypatch):
    _all_pass(monkeypatch)
    with pytest.raises(ValueError, match="gate6"):
        run_all_gates(make_cfg(), gates=["gate1", "gate6"])


def test_run_all_gates_hands_the_grid_to_the_two_gates_that_need_one(monkeypatch):
    seen: dict = {}
    for g in guards.GATE_NAMES:
        monkeypatch.setitem(guards.GATES, g,
                            lambda cfg, _n=g, **kw: (seen.__setitem__(_n, kw),
                                                     GateResult(_n, True, {}, "ok"))[1])
    grid = {"mom.window": [42, 78]}
    run_all_gates(make_cfg(), grid=grid)

    assert seen["gate2"]["grid"] == grid
    assert seen["gate5"]["grid"] == grid
    assert seen["gate1"] == {} and seen["gate3"] == {} and seen["gate4"] == {}
