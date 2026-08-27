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
      · 闸 4 的 multiplier 没传下去时，压力测试跑的是基线本身 —— 且结果完全正常；
      · θ̂ 逐折不动而测试窗净值一路阴跌时，「参数稳定」把**只在拟合窗内成立**洗成认证
        （2026-08-24 评审 C1，闸 2 判拼接样本外业绩、不判参数离散度）；
      · ⚠ 级告警被引擎追加在**末尾**、普通告警上百条时，一刀切的封顶恰好吃掉它（C2）；
      · 邻域里算不出的点被剔出均值时，分母变小 → 尖峰借势洗成高原（C3）。

★ 闸 3 的两条硬约束各有一条用例，都不能用"看起来像打乱了"糊弄过去：
  · **p 值分布**：真随机分数下 p 必须近似均匀（`test_p_values_are_uniform_...`）。
    任何偏置（置换不真随机 / `>` 与 `≥` 写反 / 少一个 +1）都会让这个闸对**每一条**
    被检验的策略把噪声判成显著，且永远不会有人发现。
  · **同日横截面内置换**：驱动【真引擎】，spy 记下每次置换后的分数，断言其
    `trade_date` 集合恒为当天（分数值里编了日期，跨天置换会把别的日子的值带进来）。

fixture 的数字一律取除不尽的值（global-constraints ★）。例外是**边界**用例
（PeakRatio 恰为 1.3、闸 1 与闸 2 的 SR_oos 恰为 0.6·SR_is）：那里要的正是逐位相等，
`<` 与 `≤` 的区别只在这一个点上看得见，用除不尽的数就永远测不到它。
闸 2 的边界用桩控制拼接 Sharpe（真 metrics 凑不出逐位相等），真算路径另有两条用例
（调用形状逐参数钉死 + §9 公式的独立 oracle）。
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
    多一个字段就多一条"闸依赖了结果的什么"的暗线。
    `equity` 是闸 2 拼接样本外的原料（其余闸不许碰它 —— 默认 None，碰了当场炸）。"""
    metrics: dict
    warnings: list
    param_hash: str
    data_snapshot_id: str
    engine_version: str
    config: BacktestConfig
    equity: object = None


def fake_run(cfg, *, use_store=False, sharpe=1.0, ir=0.5, warnings=(), engine_version="p2-engine-1",
             equity=None) -> FakeRun:
    return FakeRun(metrics={"sharpe": sharpe, "information_ratio": ir},
                   warnings=list(warnings), param_hash=cfg.param_hash(),
                   data_snapshot_id=SNAP, engine_version=engine_version, config=cfg,
                   equity=equity)


def _eq(start, rets):
    """测试窗净值：1.0 起步逐日乘 (1+r)。日期只要单调即可 —— 闸 2 按 252 年化，不读间隔。"""
    idx = [start + dt.timedelta(days=j) for j in range(len(rets) + 1)]
    vals = [1.0]
    for r_ in rets:
        vals.append(vals[-1] * (1.0 + r_))
    return pd.Series(vals, index=idx, name="equity")


def _sharpe_252(rets):
    """§9 公式的独立实现（纯 stdlib，oracle 用）：`(∏(1+r))^(252/D) − 1` 除以 `σ·√252`。"""
    steps = len(rets)
    prod = math.prod(1.0 + r_ for r_ in rets)
    ann = prod ** (252.0 / steps) - 1.0
    mean = sum(rets) / steps
    var = sum((r_ - mean) ** 2 for r_ in rets) / (steps - 1)
    return ann / (math.sqrt(var) * math.sqrt(252.0))


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
    """`get_factor` 的最小替身：闸 2/闸 5 只问 `default_params`（θ* 的注册默认值）；
    闸接 use_store 后引擎的 preload_window 还会问 `param_hash`（给个稳定桩即可）。"""

    def __init__(self, **default_params):
        self.default_params = dict(default_params)
        self.lookback_days = 250

    def param_hash(self) -> str:
        return "spec-" + "-".join(f"{k}{v}" for k, v in sorted(self.default_params.items()))


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


def test_gate1_needs_an_in_sample_window_too(runner):
    """起点已在 2019-12-31 之后：0.6 倍门槛没有基准，一次回测都不该跑。"""
    r = runner(lambda cfg, i: fake_run(cfg))
    res = gate1_out_of_sample(make_cfg(start=dt.date(2020, 3, 2)))

    assert res.passed is False and r.n == 0
    assert "样本内区间为空" in res.note


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


def test_the_duplicate_fingerprint_flag_survives_the_warning_cap(runner):
    """评审 C2 重放：引擎把「⚠ D7 重复指纹」追加在告警列表**末尾**（`append_oos_run` 是
    `run_backtest` 的最后一步），而 `combine`/`build_targets` 的普通告警逐日带文本、
    上百条是常态 —— 一刀切的 20 条封顶砍掉的恰好是那条 ⚠：闸 1 报「通过」，
    污染标记从 note 和 detail 里**同时**消失。⚠ 必须无条件保留，封顶只对其余生效，
    而且被砍的条数照样要说出来。"""
    noise = [f"2015-0{1 + i % 9}-1{i % 3} 合成剔除 {i}/16 个因子并按剩余权重重新归一"
             for i in range(25)]
    dup = "⚠ D7 重复指纹：param_hash=abc + data_snapshot_id=snap 已在样本外台账里出现过"
    runner(lambda cfg, i: fake_run(cfg, sharpe=(1.3717 if i == 0 else 1.2011),
                                   warnings=(noise if i == 0 else [dup])))
    res = gate1_out_of_sample(make_cfg())

    assert res.passed is True
    assert dup in res.detail["warnings"]                # 旧代码：⚠ 在第 26 位，被砍
    assert "⚠" in res.note
    plain = [w for w in res.detail["warnings"] if not w.startswith("⚠") and "未列出" not in w]
    assert len(plain) == 20
    assert any("另有 5 条" in w for w in res.detail["warnings"])


# ══════════════ 闸 2 · Walk-forward ══════════════

# 逐折 × 逐参数的训练集 Sharpe。argmax 逐折不同 —— 「用测试集选参数」这个泄漏
# 一旦发生，选出来的 θ̂ 会变，用例立刻红。
_TRAIN_SR = {0: {20: 0.9013, 60: 1.4177, 250: 0.7331},
             1: {20: 0.8117, 60: 1.3313, 250: 1.0937},
             2: {20: 1.2231, 60: 1.1873, 250: 0.6619},
             3: {20: 0.7717, 60: 1.5041, 250: 1.1109},
             4: {20: 0.9931, 60: 1.2777, 250: 0.8803}}

_FOLDS = [(dt.date(2010 + k, 1, 1), dt.date(2014 + k, 12, 31),
           dt.date(2015 + k, 1, 1), dt.date(2015 + k, 12, 31)) for k in range(5)]
IS_WIN = (dt.date(2010, 1, 1), IS_END)

# 各折测试窗的逐日收益（除不尽、逐折逐日都不同 —— 归一化/约分吸收不掉）。
# 拼起来的 Sharpe 远高于默认 sr_is 的 0.6 倍门槛，所以默认走「过」的那条路。
_FOLD_RETS = {k: [0.0121 + 0.0007 * k, -0.0053 + 0.0003 * k, 0.0097 - 0.0004 * k]
              for k in range(5)}


def _wf_runner(train=None, sr_is=1.1013, fold_rets=None, test_sr=0.4321, fold_equity=None):
    """样本内窗一次 + 每折（g 次选参 + 1 次测试）。
    per-fold 的 `test_sr`（默认 0.4321，**低于**一切门槛）与测试窗净值故意背离：
    任何「拿逐折 Sharpe 当判据」的实现在「净值强、逐折 Sharpe 弱」的用例里会判反。"""
    train = train if train is not None else _TRAIN_SR
    fold_rets = fold_rets if fold_rets is not None else _FOLD_RETS
    fold_equity = fold_equity or {}

    def fn(cfg, i):
        if (cfg.start, cfg.end) == IS_WIN:
            return fake_run(cfg, sharpe=sr_is)
        w = cfg.factor_param_override.get("mom", {}).get("window", 60)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                return fake_run(cfg, sharpe=train[k][w])
            if (cfg.start, cfg.end) == (te_s, te_e):
                eq = fold_equity.get(k, _eq(te_s, fold_rets[k]))
                return fake_run(cfg, sharpe=test_sr, equity=eq)
        raise AssertionError(f"闸 2 跑了一个既不是样本内窗、训练窗也不是测试窗的区间: "
                             f"{cfg.start}~{cfg.end}")
    return fn


def test_walk_forward_selects_the_parameter_on_the_training_window_only(runner):
    """泄漏就发生在这条缝上：θ̂ 必须是**训练集** Sharpe 的 argmax。"""
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
    assert [f["test_sharpe"] for f in res.detail["folds"]] == [0.4321] * 5


def test_walk_forward_never_lets_a_selection_run_see_past_its_training_window(runner):
    r = runner(_wf_runner())
    gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    for c in r.calls:
        if (c.start, c.end) == IS_WIN:          # 样本内那一次（闸 1 的口径），不属于任何折
            continue
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

    assert r.n == 21                    # 1 次样本内 + 5 折 × (3 选参 + 1 测试)
    assert max(c.end for c in r.calls) == IS_END


def test_gate2s_in_sample_side_is_gate1s_config_bit_for_bit(runner):
    """裁决原文：「样本内一侧就用闸 1 已经跑过的那一次」。「拼起来的样本内」不存在 ——
    训练窗逐年重叠，串联会把 2011–2018 每年数进去多次。config 必须与闸 1 的 `is_cfg`
    逐位相同（`param_hash` 相等）：指纹一致才谈得上结果可互换、将来可被缓存去重。"""
    cfg = make_cfg()
    r = runner(_wf_runner(sr_is=1.3717))
    res = gate2_walk_forward(cfg, grid={"mom.window": [20, 60, 250]})

    is_calls = [c for c in r.calls if (c.start, c.end) == IS_WIN]
    assert len(is_calls) == 1
    assert is_calls[0].param_hash() == replace(cfg, compute_diagnostics=False,
                                               end=IS_END).param_hash()
    assert res.detail["in_sample"]["param_hash"] == is_calls[0].param_hash()
    assert res.detail["in_sample"]["sharpe"] == 1.3717


def test_the_pooled_curve_is_the_fold_returns_chained_in_order(runner, monkeypatch):
    """拼接的定义：各折测试窗的**逐日收益**按折序串联，1.0 锚点起步，
    Sharpe 交给引擎同一个 `metrics.compute`（full=False / 252 / 无基准）——
    评审确认过的调用形状，这里逐参数钉死。"""
    seen: dict = {}
    real = guards._metrics_compute

    def spy(eq, trades, positions, bench, **kw):
        seen["eq"], seen["bench"], seen["kw"] = eq, bench, kw
        return real(eq, trades, positions, bench, **kw)

    monkeypatch.setattr(guards, "_metrics_compute", spy)
    cfg = make_cfg()
    runner(_wf_runner())
    res = gate2_walk_forward(cfg, grid={"mom.window": [20, 60, 250]})

    assert seen["kw"] == {"full": False, "initial_capital": cfg.initial_capital,
                          "periods_per_year": 252}
    assert seen["bench"] is None
    v = seen["eq"].to_numpy(dtype=float)
    assert v[0] == 1.0                                   # 锚点：第一段测试窗前的 1.0
    exp = [x for k in range(5) for x in _FOLD_RETS[k]]
    assert len(v) == 16                                  # 1 + 5 折 × 3 个收益
    assert np.allclose(v[1:] / v[:-1] - 1.0, exp, rtol=1e-12, atol=0)
    assert res.detail["n_oos_returns"] == 15


def test_the_pooled_sharpe_matches_an_independent_section_9_computation(runner):
    """真 `metrics.compute` 走全程，oracle 是 §9 公式的纯 stdlib 独立实现。
    训练窗净值不进拼接、θ* 不混入 —— 任何多拼/漏拼都会改变这个数。"""
    sr_is = 1.1013
    exp = _sharpe_252([x for k in range(5) for x in _FOLD_RETS[k]])
    assert exp >= 0.6 * sr_is                            # fixture 自检：走「过」的那条路
    runner(_wf_runner(sr_is=sr_is))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.detail["sharpe_oos_pooled"] == pytest.approx(exp, rel=1e-9, abs=0)
    assert res.detail["threshold"] == 0.6 * sr_is
    assert res.passed is True


def test_exactly_six_tenths_of_the_in_sample_sharpe_passes_gate2(runner, monkeypatch):
    """§8 修正案复用闸 1 的 `≥ 0.6`。`>` 与 `≥` 只在这一点分得开，fixture 必须逐位相等 ——
    真 metrics 里凑不出逐位相等，所以边界用桩钉；真算路径由上面两条钉。"""
    sr_is = 1.3717
    monkeypatch.setattr(guards, "_metrics_compute",
                        lambda *a, **k: ({"sharpe": 0.6 * sr_is}, []))
    runner(_wf_runner(sr_is=sr_is))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.detail["threshold"] == 0.6 * sr_is
    assert res.passed is True


def test_a_hair_below_six_tenths_fails_gate2(runner, monkeypatch):
    sr_is = 1.3717
    monkeypatch.setattr(guards, "_metrics_compute",
                        lambda *a, **k: ({"sharpe": 0.6 * sr_is - 1e-9}, []))
    runner(_wf_runner(sr_is=sr_is))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False


def test_stable_parameters_with_a_dead_oos_curve_fail_the_walk_forward(runner):
    """评审 C1 点名必须翻转的 fixture：θ̂ 几乎不动（54↔60）、训练 Sharpe 0.90–1.50，
    而测试窗净值一路阴跌。旧判据（max/min ≤ 3.0）把「只在拟合窗内成立」认证成「稳定」；
    修正案下它判不过 —— 参数稳不稳只是失败时的一种解释，不是判据。"""
    steady = {0: {54: 0.9013, 60: 1.4177, 78: 0.7331},
              1: {54: 1.3313, 60: 1.0937, 78: 0.8117},
              2: {54: 0.6619, 60: 1.2231, 78: 1.1873},
              3: {54: 0.7717, 60: 1.5041, 78: 1.1109},
              4: {54: 1.2777, 60: 0.9931, 78: 0.8803}}
    dead = {k: [-0.0021 - 0.0003 * k, 0.0007 - 0.0002 * k, -0.0031 + 0.0001 * k]
            for k in range(5)}
    sr_is = 1.5041
    assert _sharpe_252([x for k in range(5) for x in dead[k]]) < 0.6 * sr_is    # fixture 自检
    runner(_wf_runner(train=steady, sr_is=sr_is, fold_rets=dead))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [54, 60, 78]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [60, 54, 60, 60, 54]
    assert res.detail["theta_spread"]["mom.window"] == pytest.approx(60 / 54, rel=0, abs=1e-12)
    assert res.detail["theta_flips"]["mom.window"] == 2   # [60,54,60,60,54]：−,+,(0),− → 2
    assert res.passed is False


def test_parameter_dispersion_goes_to_detail_and_no_longer_fails_the_gate(runner):
    """裁决：判在业绩上，不判在参数离散度上。20 ↔ 250 横跳（spread 12.5、flips 3）
    照样能过 —— 只要拼接样本外真的赚钱；离散度进 detail 供人解读，不设阈值。
    per-fold 的 test_sharpe（0.4321，低于门槛）故意与净值背离：
    「拿逐折 Sharpe 平均当判据」的实现在这里会判反。"""
    jumpy = {0: {20: 2.7183, 60: 0.9013, 250: 0.3010},
             1: {20: 0.4771, 60: 0.8117, 250: 2.6180},
             2: {20: 2.3026, 60: 1.0937, 250: 0.5772},
             3: {20: 0.6931, 60: 1.1109, 250: 2.4849},
             4: {20: 2.1401, 60: 0.9931, 250: 0.7781}}
    strong = {k: [0.0141 + 0.0007 * k, -0.0053 + 0.0003 * k, 0.0119 - 0.0004 * k]
              for k in range(5)}
    sr_is = 0.9013
    assert _sharpe_252([x for k in range(5) for x in strong[k]]) >= 0.6 * sr_is
    runner(_wf_runner(train=jumpy, sr_is=sr_is, fold_rets=strong))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [20, 250, 20, 250, 20]
    assert res.detail["theta_spread"]["mom.window"] == 12.5
    assert res.detail["theta_flips"]["mom.window"] == 3
    assert res.passed is True


def test_monotone_drift_shows_spread_but_zero_flips(runner):
    """裁决原文：单调漂移可能是真实的 regime 变化，振荡才是拟合噪声 —— 两个量分开报。
    [20,60,60,250,250] 的 spread 同为 12.5，flips 却是 0：混着报就分不出这两种形状。"""
    drift = {0: {20: 1.4177, 60: 0.9013, 250: 0.7331},
             1: {20: 0.8117, 60: 1.3313, 250: 1.0937},
             2: {20: 0.6619, 60: 1.2231, 250: 1.1873},
             3: {20: 0.7717, 60: 1.1109, 250: 1.5041},
             4: {20: 0.9931, 60: 0.8803, 250: 1.2777}}
    runner(_wf_runner(train=drift))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert [f["theta"]["mom.window"] for f in res.detail["folds"]] == [20, 60, 60, 250, 250]
    assert res.detail["theta_spread"]["mom.window"] == 12.5
    assert res.detail["theta_flips"]["mom.window"] == 0


def test_the_stability_threshold_knob_is_gone(runner):
    """评审 C1：`max_param_ratio` 是被裁掉的判据留下的旋钮，留着就是邀请谁拧回来。
    传它必须 TypeError，而不是被静默吞掉。"""
    runner(_wf_runner())
    with pytest.raises(TypeError):
        gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60]}, max_param_ratio=3.0)
    assert not hasattr(guards, "_WF_MAX_PARAM_RATIO")


def test_a_non_numeric_parameter_has_no_spread_and_the_verdict_rides_on_the_curve(runner):
    """`weighting` 没有「量级」：spread/flips 报 None（逐折取值在 folds 里看得到），
    判定完全落在拼接样本外的业绩上 —— 逐折取值不一致不再是失败理由。"""
    sr = {"equal": 0.9013, "risk_parity": 1.4177}

    def fn(cfg, i):
        if (cfg.start, cfg.end) == IS_WIN:
            return fake_run(cfg, sharpe=1.1013)
        for k, (tr_s, tr_e, te_s, te_e) in enumerate(_FOLDS):
            if (cfg.start, cfg.end) == (tr_s, tr_e):
                w = sr[cfg.constraints.weighting]
                return fake_run(cfg, sharpe=(w if k % 2 else 1.9013 - w))
            if (cfg.start, cfg.end) == (te_s, te_e):
                return fake_run(cfg, sharpe=0.4321, equity=_eq(te_s, _FOLD_RETS[k]))
        raise AssertionError("窗口不对")

    runner(fn)
    res = gate2_walk_forward(make_cfg(), grid={"weighting": ["equal", "risk_parity"]})

    assert [f["theta"]["weighting"] for f in res.detail["folds"]] == [
        "equal", "risk_parity", "equal", "risk_parity", "equal"]
    assert res.detail["theta_spread"]["weighting"] is None
    assert res.detail["theta_flips"]["weighting"] is None
    assert res.passed is True


def test_a_fold_that_cannot_select_a_parameter_fails_the_gate(runner):
    """评审 I1 重放：候选全 NaN 的折被悄悄丢掉后，旧 note 说「5 折全部选参完毕」。
    缺折必须判死，并把 4/5 说出来 —— 4 折说不出「5 折」的结论。"""
    train = {k: dict(_TRAIN_SR[k]) for k in range(5)}
    train[2] = {20: float("nan"), 60: float("nan"), 250: float("nan")}
    runner(_wf_runner(train=train))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False
    assert "4/5" in res.note
    assert res.detail["n_chosen"] == 4 and res.detail["n_folds"] == 5
    assert res.detail["folds"][2]["theta"] is None
    assert any(w.startswith("⚠") for w in res.detail["warnings"])
    assert "⚠" in res.note


def test_a_fold_whose_test_window_yields_no_curve_fails_the_gate(runner):
    """I1 的另一半：选出了 θ̂ 但测试窗只有一个净值点（给不出一个收益）——
    拼接样本外缺一段，不能当它不存在。"""
    fold_rets = {k: list(_FOLD_RETS[k]) for k in range(5)}
    fold_rets[3] = []                                   # 单点净值
    runner(_wf_runner(fold_rets=fold_rets))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False
    assert "净值不足两个点" in res.note and "2018-01-01" in res.note


def test_a_hole_inside_a_test_curve_poisons_the_pool_instead_of_bridging_it(runner):
    """净值中间的 NaN 若被 `pct_change` 的默认 ffill 桥掉，缺口两侧会缝出一段编造的收益
    （metrics.py 给同一个坑写过注释）。这里必须走「算不出 → 判不过」。"""
    holed = pd.Series([1.0, 1.0037, float("nan"), 1.0091, 1.0113],
                      index=[_FOLDS[1][2] + dt.timedelta(days=j) for j in range(5)])
    runner(_wf_runner(fold_equity={1: holed}))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False
    assert "缺口" in res.note and "2 个算不出" in res.note


def test_a_zero_vol_pooled_curve_cannot_pass_gate2(runner):
    """常数收益 → σ=0 → `metrics.compute` 给 NaN。算不出 ≠ 达标。"""
    runner(_wf_runner(fold_rets={k: [0.0037, 0.0037, 0.0037] for k in range(5)}))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False
    assert "算不出" in res.note


def test_a_non_positive_in_sample_sharpe_stops_gate2_before_the_folds(runner):
    """闸 1 的同一条理由：0.6 × 负数是个负门槛。且必须**先**跑样本内 —— 判死时
    5 折 × 3 候选的 15 次运行一次都不必花。"""
    r = runner(_wf_runner(sr_is=-0.5031))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False and r.n == 1
    assert "样本内" in res.note and "负数" in res.note


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_an_uncomputable_in_sample_sharpe_stops_gate2_before_the_folds(runner, bad):
    r = runner(_wf_runner(sr_is=bad))
    res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60, 250]})

    assert res.passed is False and r.n == 1
    assert "算不出" in res.note


def test_zero_or_negative_window_widths_are_rejected_up_front(runner):
    """`_folds(test_years=0)` 的前滚步长为 0，永不推进 —— `run_all_gates` 的 try/except
    接不住一个挂死的循环。必须在入口拒绝。"""
    r = runner(_wf_runner())
    for kw in ({"test_years": 0}, {"train_years": 0}, {"test_years": -3}):
        res = gate2_walk_forward(make_cfg(), grid={"mom.window": [20, 60]}, **kw)
        assert res.passed is False
        assert "未运行" in res.note and "1 年" in res.note
    assert r.n == 0


def test_gate2_rejects_a_grid_that_names_nothing_or_lists_nothing(runner):
    r = runner(_wf_runner())
    for bad in ({"mom.windwo": [42]}, {"mom.window": []}):
        res = gate2_walk_forward(make_cfg(), grid=bad)
        assert res.passed is False and "网格" in res.note
    assert r.n == 0


def test_a_leap_day_start_clamps_to_feb_28_instead_of_crashing(runner):
    """2012-02-29 + 5 年不存在：`_plus_years` 退到 2 月 28 日再切折。
    挂在闸 2 上测 —— 折窗切分是它的私事，公共可见的是训练/测试窗端点。"""
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=1.0721 + 0.0137 * i,
                                       equity=_eq(cfg.start, [0.0021, -0.0013, 0.0031])))
    res = gate2_walk_forward(make_cfg(start=dt.date(2012, 2, 29)),
                             grid={"mom.window": [20, 60]})

    wins = {(c.start, c.end) for c in r.calls}
    assert (dt.date(2012, 2, 29), dt.date(2017, 2, 27)) in wins      # 训练窗 1
    assert (dt.date(2017, 2, 28), dt.date(2018, 2, 27)) in wins      # 测试窗 1
    assert (dt.date(2013, 2, 28), dt.date(2018, 2, 27)) in wins      # 训练窗 2（再次退位）
    assert isinstance(res, GateResult) and res.name == "gate2"


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

    1000 次重复而不是 200（评审 M5）：拒绝率的下界 0.01 卡的正是 n=19 那种
    「永远拒绝不了」的退化，200 次时观测值 0.0100 恰好压线（σ≈0.011，纯运气）；
    1000 次时 σ≈0.005，真值 0.025 离下界 3σ —— 界不动，样本量把余量挣出来。
    """
    rng = np.random.default_rng(20260822)
    ps: list = []
    for _ in range(1000):
        r = Runner(lambda cfg, i: fake_run(cfg, sharpe=float(rng.normal())))
        monkeypatch.setattr(guards, "run_backtest", r)
        ps.append(gate3_shuffle(make_cfg(), n=39, seed=int(rng.integers(1 << 30))
                                ).detail["p_value"])

    arr = np.array(ps)
    # 经验分布函数逐点贴住 y = x（离散均匀的理论值就是 x），1000 个样本 σ ≈ 0.016
    for x in (0.25, 0.5, 0.75):
        assert abs(float((arr <= x).mean()) - x) <= 0.10, f"p 值在 {x} 处偏离均匀"
    assert 0.01 <= float((arr < 0.05).mean()) <= 0.15    # 真值 2.5%（= 1/40 那一格）
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
    w = next(w for w in res.detail["warnings"] if "shuffle_seed=99" in w)
    assert w.startswith("⚠")            # 闸自己喊的降级不比引擎的低一级：_note 只顶 ⚠
    assert "⚠" in res.note


def test_a_non_finite_real_sharpe_can_never_be_declared_significant(runner):
    """`SR_b ≥ NaN` 恒为假 → n_ge = 0 → p = 1/(n+1) = 0.005 → **显著**。
    一条算不出 Sharpe 的净值曲线会被这道闸判成"击败了全部 200 个对照"。"""
    r = runner(_shuffle_runner(float("nan"), [0.5] * 20))
    res = gate3_shuffle(make_cfg(), n=20, seed=5)

    assert res.passed is False
    # 键必须**存在**且为 None：`.get(...) is None` 连「键根本没写」也放过（评审 minor）
    assert "p_value" in res.detail and res.detail["p_value"] is None
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
    w = next(w for w in res.detail["warnings"] if "非有限" in w)
    assert w.startswith("⚠") and "⚠" in res.note     # 有效样本变少是 ⚠ 级降级


def test_a_permutation_count_below_one_is_refused_up_front(runner):
    """没有这道守卫：n=−2 → `range(−2)` 一个对照都不跑 → p = (1+0)/(−2+1) = −1.0 →
    「−1.0 < 0.05」→ 一条什么都没检验的曲线被判**显著**（评审 minor 重放）。"""
    r = runner(_shuffle_runner(1.4142, []))
    res = gate3_shuffle(make_cfg(), n=-2)

    assert res.passed is False and r.n == 0
    assert "未运行" in res.note


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

    def last_data_date(self):
        return _G_DAYS[-1]          # 这个假世界里日历与行情同长

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

    def fake_combine(weights, as_of_date, universe, *, use_store=False):
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


@pytest.mark.parametrize("m", [0.0, 0.5031, float("nan")])
def test_a_multiplier_below_one_is_relief_not_stress(runner, m):
    """评审 I4 重放：multiplier=0.0 以**零成本**跑一遍并报「闸 4 通过」。更糟的是闸 4
    不钉样本内，扫 multiplier 等于一格一个新指纹的样本外运行，台账的重复检查永远不响。
    加压不是减压：< 1 当场拒绝，一次回测都不跑（NaN 也走这支 —— `not ≥ 1`）。"""
    r = runner(lambda cfg, i: fake_run(cfg, ir=0.9013))
    res = gate4_cost_stress(make_cfg(), multiplier=m)

    assert res.passed is False and r.n == 0
    assert "减压" in res.note


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


def test_a_peak_ratio_a_hair_under_one_point_three_passes(runner):
    """评审 I2：1.3 此前只被上方钉住，(1.005, 1.3] 里任何阈值（1.1、1.25 的变异）
    都能溜过整套用例。照闸 1 的样子从两侧钉死：均值恰为 1.0，比值就是 SR* 自己。"""
    star = 1.3 - 1e-9
    runner(_plateau_runner(star, {42: 0.75, 78: 1.25}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78]})

    assert res.detail["peak_ratio"] == star
    assert res.passed is True


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_theta_star_sharpe_never_certifies_a_plateau(runner, bad):
    """闸 1 M05 的同形（评审 I3）：sr* = −inf 时 `−inf < 1.3` 恒真 → 邻域为正就满分过闸。
    守卫独占这条路径 —— 钉住它，别让它被当成冗余删掉（2026-08-21 等价变异裁决）。"""
    runner(_plateau_runner(bad, {42: 0.9013, 78: 1.1873}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78]})

    assert res.passed is False
    assert "无从算起" in res.note


def test_a_neighbourhood_where_nothing_evaluates_cannot_certify(runner):
    """`not ok` 那一支（评审 I3 点名的零覆盖分支）：邻域全算不出 → 无从判高原。"""
    runner(_plateau_runner(1.2231, {42: float("nan"), 78: float("nan")}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78]})

    assert res.passed is False
    assert "没有一个点算得出" in res.note


def test_one_unevaluable_neighbour_fails_the_gate_instead_of_shrinking_the_mean(runner):
    """评审 C3 重放：SR* = 1.3553，三个邻域里一个 NaN。旧代码把 NaN 剔出均值 →
    分母只剩俩 → 比值 1.1295 →「通过、高原」，note 干干净净。方向不可知的降级必须判死
    （拿不准就判不过），且 note 要点出是谁算不出。"""
    runner(_plateau_runner(1.3553, {42: 1.2011, 78: float("nan"), 90: 1.1987}))
    res = gate5_param_plateau(make_cfg(), {"mom.window": [42, 78, 90]})

    assert res.passed is False
    assert "1 个算不出" in res.note and "mom.window=78" in res.note
    assert res.detail["peak_ratio"] is None and res.detail["mean_neighbour"] is None


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
    """邻域两点取**不同**的除不尽值（评审 minor：原 fixture 两点同值，mean↔min↔max
    在这条用例上不可分辨，违反「会被约分/归一吸收的量要互不相同」的纪律）。"""
    sr = {37: 1.2231, 26: 1.1873, 48: 1.3341}
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=sr[cfg.constraints.top_n]))
    res = gate5_param_plateau(make_cfg(), {"top_n": [26, 37, 48]})

    assert [c.constraints.top_n for c in r.calls] == [37, 26, 48]
    assert res.detail["mean_neighbour"] == pytest.approx((1.1873 + 1.3341) / 2, rel=0,
                                                         abs=1e-12)
    assert res.passed is True


def test_plateau_without_a_grid_is_reported_as_not_run(runner):
    r = runner(_plateau_runner(1.0, {}))
    res = gate5_param_plateau(make_cfg())

    assert res.passed is False and r.n == 0
    assert "网格" in res.note


def test_the_plateau_gate_needs_an_in_sample_window(runner):
    """起点在 2019-12-31 之后：网格扫描只能落在样本外，那就是拿样本外调参（D7）。"""
    r = runner(_plateau_runner(1.0, {}))
    res = gate5_param_plateau(make_cfg(start=dt.date(2020, 3, 2)), {"mom.window": [42]})

    assert res.passed is False and r.n == 0
    assert "样本外" in res.note


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


def test_the_default_run_all_gates_call_reaches_every_gate_with_no_grid(runner):
    """**默认**的 `run_all_gates(cfg)`（评审点名的零覆盖路径）：不给 grid，真闸全跑一遍，
    闸 2 / 闸 5 必须以「没有参数网格」现身 —— 消失或误报通过都等于"没问题"。"""
    r = runner(lambda cfg, i: fake_run(cfg, sharpe=1.1013 + 0.0001 * i, ir=0.3717))
    out = run_all_gates(make_cfg())

    assert list(out) == list(guards.GATE_NAMES)
    # 闸 1 过（样本外 1.1014 ≥ 0.6×1.1013）；闸 3 不过（对照逐个更高，p=1.0）；
    # 闸 4 过（ir>0）；闸 2/闸 5 没网格，判不过
    assert [g.passed for g in out.values()] == [True, False, False, True, False]
    assert "网格" in out["gate2"].note and "网格" in out["gate5"].note
    assert out["gate3"].detail["p_value"] == 1.0
    assert r.n == 204                   # 闸1 2 次 + 闸3 (1+200) 次 + 闸4 1 次
