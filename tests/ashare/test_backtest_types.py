"""回测数据结构（架构 §4.3）—— param_hash 是 D7 的执行机制，summary 是 REST/LLM 的返回预算。

两条主线：
  1. `param_hash` 少哈希一个字段 = 两组【真不同】的参数撞成一个指纹 →「样本外已经跑过」的
     闸门永远不触发，人在样本外数据上调参却以为没有。所以这里的字段表带完整性守卫：
     BacktestConfig 加了字段而没在表里登记，测试直接失败。
  2. `summary()` 走 REST 与 Agent 工具层，超预算的"精简版"要么被截断要么撑爆上下文，
     所以用【真实规模】的结果对象量字节数 —— 空结果测出来的 3 KB 是废话。
"""
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from ashare.backtest.types import (BacktestConfig, BacktestResult, CostConfig,
                                   PortfolioConstraints)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# 一个【全字段都非默认】的配置：跨进程/跨版本指纹稳定性都钉在它身上。
PINNED_CFG = BacktestConfig(
    start=dt.date(2010, 1, 1), end=dt.date(2019, 12, 31),
    factors=(("reversal_20", 0.6), ("turnover_20", 0.4)),
    constraints=PortfolioConstraints(top_n=30, weighting="risk_parity", max_single=0.04,
                                     max_industry=0.15, max_turnover=0.25),
    cost=CostConfig(commission_bps=3.0, stamp_duty_bps=5.0, transfer_bps=0.1,
                    impact_coef=0.4, impact_cap_bps=25.0, multiplier=2.0),
    macro_timing=True, position_floor=0.3, position_cap=0.9,
    benchmark="000300.SH", initial_capital=500_000.0,
    compute_diagnostics=False, shuffle_seed=7,
    factor_param_override={"reversal_20": {"window": 10}},
)

BASE = BacktestConfig(start=dt.date(2015, 1, 1), end=dt.date(2018, 12, 31),
                      factors=(("a", 0.5), ("b", 0.5)))


# ── param_hash 字段覆盖：表 + 完整性守卫 ────────────────────────────────

# 影响结果 → 必须进哈希。值都与 BASE 不同。
_HASHED = {
    "start": dt.date(2015, 1, 2),
    "end": dt.date(2019, 1, 1),
    "factors": (("a", 0.5), ("b", 0.6)),
    "constraints": PortfolioConstraints(top_n=30),
    "cost": CostConfig(multiplier=2.0),
    "macro_timing": True,
    "position_floor": 0.25,
    "position_cap": 0.9,
    "benchmark": "000300.SH",
    "initial_capital": 2_000_000.0,
    "shuffle_seed": 0,                                  # 闸 3：打乱分数 → 另一次实验
    "factor_param_override": {"a": {"window": 5}},
}
# 唯一豁免：只决定"算不算诊断"，不改 equity/trades/positions/metrics 里的任何一个数。
_NOT_HASHED = {"compute_diagnostics": False}


def test_the_field_table_covers_every_config_field():
    """加了新字段却忘了登记 → 它默默不进哈希 = D7 失效。这条测试就是那道提醒。"""
    assert set(_HASHED) | set(_NOT_HASHED) == {f.name for f in dataclasses.fields(BacktestConfig)}


@pytest.mark.parametrize("name", sorted(_HASHED))
def test_every_result_affecting_field_changes_the_hash(name):
    changed = dataclasses.replace(BASE, **{name: _HASHED[name]})
    assert changed.param_hash() != BASE.param_hash(), f"{name} 不在 param_hash 里"


@pytest.mark.parametrize("name", sorted(f.name for f in dataclasses.fields(CostConfig)))
def test_every_cost_field_is_nested_into_the_hash(name):
    """闸 4（成本翻倍）的对照组必须与基线不同指纹，否则闸 4 记录下来的是它自己的基线。"""
    bumped = dataclasses.replace(BASE.cost, **{name: getattr(BASE.cost, name) + 1.0})
    assert dataclasses.replace(BASE, cost=bumped).param_hash() != BASE.param_hash()


@pytest.mark.parametrize("name", sorted(f.name for f in dataclasses.fields(PortfolioConstraints)))
def test_every_constraint_field_is_nested_into_the_hash(name):
    old = getattr(BASE.constraints, name)
    new = "risk_parity" if isinstance(old, str) else old + 1
    bumped = dataclasses.replace(BASE.constraints, **{name: new})
    assert dataclasses.replace(BASE, constraints=bumped).param_hash() != BASE.param_hash()


def test_compute_diagnostics_is_the_only_exclusion():
    assert dataclasses.replace(BASE, compute_diagnostics=False).param_hash() == BASE.param_hash()


def test_shuffle_seed_none_zero_and_one_are_three_different_runs():
    """seed=0 与 seed=None 是"打乱过"与"没打乱"，撞哈希等于把闸 3 的对照组记成真回测。"""
    hs = {dataclasses.replace(BASE, shuffle_seed=s).param_hash() for s in (None, 0, 1)}
    assert len(hs) == 3


# ── 书写顺序 / 浮点噪声 ────────────────────────────────────────────────

def test_keyword_order_does_not_change_the_hash():
    a = BacktestConfig(start=dt.date(2015, 1, 1), end=dt.date(2018, 12, 31),
                       factors=(("a", 0.5),), benchmark="000300.SH", initial_capital=5e5)
    b = BacktestConfig(initial_capital=5e5, benchmark="000300.SH",
                       factors=(("a", 0.5),), end=dt.date(2018, 12, 31), start=dt.date(2015, 1, 1))
    assert a.param_hash() == b.param_hash()


def test_factor_order_does_not_change_the_hash():
    """combine 是 Σ wᵢ·dirᵢ·zᵢ（计划 Task 7），加法可交换 → 换个书写顺序是同一个策略。
    不排序的话 D7 台账里会多出一条根本不存在的"新实验"。"""
    a = dataclasses.replace(BASE, factors=(("a", 0.3), ("b", 0.7)))
    b = dataclasses.replace(BASE, factors=(("b", 0.7), ("a", 0.3)))
    assert a.param_hash() == b.param_hash()
    # 但权重真变了必须变指纹 —— 别把排序做成"因子内容也不管"
    assert dataclasses.replace(BASE, factors=(("b", 0.6), ("a", 0.4))).param_hash() != a.param_hash()


def test_nested_override_key_order_does_not_change_the_hash():
    a = dataclasses.replace(BASE, factor_param_override={"a": {"x": 1, "y": 2}, "b": {}})
    b = dataclasses.replace(BASE, factor_param_override={"b": {}, "a": {"y": 2, "x": 1}})
    assert a.param_hash() == b.param_hash()
    assert dataclasses.replace(BASE, factor_param_override={"a": {"x": 1, "y": 3}, "b": {}}) \
        .param_hash() != a.param_hash()


def test_float_noise_below_the_quantum_does_not_change_the_hash():
    """0.1+0.2 != 0.3 是浮点，不是两组参数。1e-12 以下的差改不动任何一个回测数字
    （权重量级 1e-1，股数按 100 股取整），"什么都没变但指纹变了"和漏哈希一样有害。"""
    noisy = dataclasses.replace(BASE, factors=(("a", 0.1 + 0.2), ("b", 0.7)))
    clean = dataclasses.replace(BASE, factors=(("a", 0.3), ("b", 0.7)))
    assert noisy.param_hash() == clean.param_hash()
    # 量级之上的差别照样要区分
    assert dataclasses.replace(BASE, factors=(("a", 0.3 + 1e-9), ("b", 0.7))).param_hash() \
        != clean.param_hash()


def test_unserializable_override_raises_instead_of_silently_hashing_its_id():
    """object() 的 repr 带内存地址：吞掉就等于指纹随进程变，D7 台账全是一次性记录。"""
    cfg = dataclasses.replace(BASE, factor_param_override={"a": {"cb": object()}})
    with pytest.raises(TypeError):
        cfg.param_hash()


# ── 指纹格式：跨进程 + 跨版本 ──────────────────────────────────────────

def test_param_hash_is_16_hex_chars():
    h = PINNED_CFG.param_hash()
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_param_hash_is_identical_in_fresh_interpreters_with_different_hash_seeds():
    """PYTHONHASHSEED 随机化下 hash()/set 迭代序都会变 —— 指纹只能来自 canonical JSON。
    这条挂 = 同一组参数今天明天两个指纹，D7 的"跑过没有"永远查不到。"""
    snippet = ("from tests.ashare.test_backtest_types import PINNED_CFG;"
               "print(PINNED_CFG.param_hash())")
    got = {subprocess.run([sys.executable, "-c", snippet], cwd=_REPO_ROOT, check=True,
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip()
           for seed in ("0", "1", "random")}
    assert got == {PINNED_CFG.param_hash()}


def test_param_hash_of_the_pinned_config_is_frozen():
    """回归钉子：这个值一变，docs/oos-runs.md 里所有历史 (param_hash, snapshot_id) 记录
    就再也匹配不上现在的配置 —— 每条"样本外已跑过"都失效。改它必须是有意的。"""
    assert PINNED_CFG.param_hash() == "c883e4b45ab1adf3"


# ── summary()：真实规模下的字节预算 ────────────────────────────────────

_METRICS = {                                    # 规格 §5.4 的四类指标
    "ann_return": 0.183412, "ann_vol": 0.221078, "sharpe": 0.829511, "calmar": 0.610233,
    "max_drawdown": -0.301244, "max_drawdown_start": dt.date(2015, 6, 12),
    "max_drawdown_end": dt.date(2016, 1, 28), "monthly_win_rate": 0.575,
    "excess_return_csi985": 0.092133, "excess_return_hs300": 0.110387,
    "information_ratio": 0.881244, "tracking_error": 0.104512,
    "ic_mean": 0.041233, "rank_ic_mean": 0.048811, "icir": 0.412344, "ic_win_rate": 0.623122,
    "layer_monotonicity": 0.854533, "ann_turnover": 12.433, "avg_holdings": 49.31,
    "cost_drag_pct": 0.041244, "newey_west_t": 2.7133,
}


def _realistic_result(*, n_warnings: int = 60, metrics=None,
                      factors=None, config=None) -> BacktestResult:
    """2010–2019 周频、50 只持仓的一次真实规模回测：约 2.4k 净值点 / 26k 持仓行 / 20k 笔成交。"""
    days = pd.bdate_range("2010-01-04", "2019-12-31")
    equity = pd.Series([1.0 + i * 0.0011 for i in range(len(days))], index=days)
    rebal = list(days[::5])
    codes = [f"{600000 + i}.SH" for i in range(50)]
    idx = pd.MultiIndex.from_product([rebal, codes], names=["rebalance_date", "ts_code"])
    positions = pd.DataFrame(
        {"score": 0.5, "target_weight": 0.02, "filled_weight": 0.02,
         "shares": 1000, "price_hfq": 12.34, "industry": "银行"}, index=idx)
    trades = pd.DataFrame(
        {"exec_date": [rebal[i % len(rebal)] for i in range(20_000)],
         "ts_code": [codes[i % 50] for i in range(20_000)],
         "side": "BUY", "shares": 1000, "price_hfq": 12.34, "amount": 12340.0,
         "commission": 3.1, "stamp_duty": 0.0, "transfer_fee": 0.1,
         "impact": 1.2, "total_cost": 4.4})
    blocked = pd.DataFrame({"exec_date": rebal[:200], "ts_code": "600001.SH",
                            "intended_side": "BUY", "intended_weight": 0.02,
                            "reason": "涨停不可买入"})
    return BacktestResult(
        config=config or dataclasses.replace(
            BASE, factors=tuple(factors or ((f"factor_{i:02d}", 1 / 16) for i in range(16)))),
        param_hash="a1b2c3d4e5f60718", data_snapshot_id="20191231-7f3a9c2b",
        engine_version="0.3.0", started_at=dt.datetime(2026, 8, 20, 9, 30, 1),
        elapsed_sec=58.3312, equity=equity, positions=positions, trades=trades, blocked=blocked,
        metrics=_METRICS if metrics is None else metrics,
        ic=pd.DataFrame({"reversal_20__ic": [0.01] * len(rebal)}, index=rebal),
        layers=pd.DataFrame({f"L{i}": [0.001] * len(rebal) for i in range(1, 11)}, index=rebal),
        attribution=pd.DataFrame({"银行": [0.002] * len(rebal)}, index=rebal),
        warnings=[f"2015-{1 + i % 12:02d}-{1 + i % 28:02d} 中性化秩亏：行业哑变量共线剔除 3 列，"
                  f"因子 factor_{i:02d} 当日残差自由度不足，该日该因子作废" for i in range(n_warnings)])


def _size(d: dict) -> int:
    return len(json.dumps(d, ensure_ascii=False).encode("utf-8"))


def test_summary_of_a_realistic_multi_year_run_fits_the_rest_budget():
    """空结果测 3 KB 是废话：十年周频 + 60 条中文告警才是真实规模。"""
    s = _realistic_result().summary()
    assert _size(s) < 3072, f"summary 实际 {_size(s)} 字节"
    # 紧凑分隔符（starlette 实际用的）自然更小，一并确认
    assert len(json.dumps(s, ensure_ascii=False, separators=(",", ":")).encode()) < 3072


def test_summary_shows_what_it_truncated():
    """告警条数有【固定上限】，不是"能塞多少塞多少"：Agent 工具层要的是可预期的载荷大小，
    顺带把兜底循环的迭代次数摁住（10 万条告警不该导致 10 万次 json.dumps）。"""
    s = _realistic_result(n_warnings=60).summary()
    assert s["warnings_total"] == 60
    assert len(s["warnings"]) == 5
    assert "60" in s["warnings_note"]                     # 丢了多少必须写在输出里


def test_summary_sheds_warnings_before_metrics_when_both_overflow():
    """砍的顺序是有意的：metrics 是用户问的那个答案，warnings 是诊断线索。
    固定上限只是把量级摁住，真顶到预算时得有人继续砍 —— 这条测的就是那个"继续"。"""
    s = _realistic_result(n_warnings=60,
                          metrics={f"指标_{i:02d}": i * 1.111111 for i in range(70)}).summary()
    assert _size(s) < 3072, f"summary 实际 {_size(s)} 字节"
    assert len(s["warnings"]) < 5 and "60" in s["warnings_note"]
    assert "_dropped" not in s["metrics"] and len(s["metrics"]) == 70


def test_summary_keeps_all_warnings_when_they_fit():
    s = _realistic_result(n_warnings=2).summary()
    assert len(s["warnings"]) == 2 and "warnings_note" not in s


def test_summary_carries_the_d7_identity():
    """D7：param_hash 与 data_snapshot_id 缺一不可；engine_version 决定语义版本。"""
    s = _realistic_result().summary()
    assert s["param_hash"] == "a1b2c3d4e5f60718"
    assert s["data_snapshot_id"] == "20191231-7f3a9c2b"
    assert s["engine_version"] == "0.3.0"


def test_summary_marks_a_shuffle_control_run():
    """闸 3 的 200 次对照跑出来的 summary 不能长得像一次真回测。"""
    cfg = dataclasses.replace(BASE, shuffle_seed=42)
    assert _realistic_result(config=cfg).summary()["shuffle_seed"] == 42
    assert _realistic_result().summary()["shuffle_seed"] is None


def test_summary_reports_cost_multiplier_for_gate4():
    cfg = dataclasses.replace(BASE, cost=CostConfig(multiplier=2.0))
    assert _realistic_result(config=cfg).summary()["cost_multiplier"] == 2.0


def test_summary_is_json_serializable_the_way_starlette_does_it():
    """starlette 的 JSONResponse 用 allow_nan=False 且不带 default：
    一个 NaN 指标或一个 numpy 标量就是 500，而不是"数字难看一点"。"""
    metrics = dict(_METRICS, calmar=float("nan"), sharpe=math.inf,
                   n_rebalances=pd.Series([1, 2, 3]).sum())      # numpy 标量
    s = _realistic_result(metrics=metrics).summary()
    json.dumps(s, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert s["metrics"]["calmar"] is None and s["metrics"]["sharpe"] is None
    assert s["metrics"]["n_rebalances"] == 6
    assert s["metrics"]["max_drawdown_start"] == "2015-06-12"


def test_summary_stays_in_budget_against_a_pathological_result():
    """metrics 与 factors 是别的任务给的自由结构 —— 预算保证不能建立在"它们不会太大"上。"""
    s = _realistic_result(metrics={f"metric_{i:03d}": i * 1.111111 for i in range(500)},
                          factors=tuple((f"very_long_factor_name_{i:03d}" * 20, 0.005)
                                        for i in range(200))).summary()
    assert _size(s) < 3072, f"summary 实际 {_size(s)} 字节"
    assert "500" in json.dumps(s["metrics"], ensure_ascii=False)     # 砍了多少要看得见
    assert s["n_factors"] == 200 and len(s["factors"]) == 10         # 同上：截断可见


def test_summary_survives_an_empty_run():
    """全部调仓日都不可交易（D6）也要能返回 —— 这类结果恰恰是最需要被看见的。"""
    empty = BacktestResult(
        config=BASE, param_hash="0" * 16, data_snapshot_id="empty", engine_version="0.3.0",
        started_at=dt.datetime(2026, 8, 20, 9, 30), elapsed_sec=0.4,
        equity=pd.Series(dtype=float), positions=pd.DataFrame(), trades=pd.DataFrame(),
        blocked=pd.DataFrame(), metrics={})
    s = empty.summary()
    assert s["n_days"] == 0 and s["n_trades"] == 0 and s["equity_final"] is None
    assert s["warnings"] == [] and s["diagnostics"] == {"ic": False, "layers": False,
                                                        "attribution": False}


def test_summary_reports_which_diagnostics_exist():
    s = _realistic_result().summary()
    assert s["diagnostics"] == {"ic": True, "layers": True, "attribution": True}


# ── 默认值（P2 阶段口径）────────────────────────────────────────────────

def test_macro_timing_defaults_to_false():
    """架构 §4.3 的字面量写 True，同一行注释写"P2 阶段默认 False"；宏观层属 P3，
    默认 True 会让 P2 的每次回测都去调一个还不存在的择时层。计划口径为准。"""
    assert BASE.macro_timing is False


def test_documented_defaults_are_not_drifting():
    """成本数值是规格 §5.3 算出来的往返 0.3%，约束是 §5.3 的组合口径。别人改一个默认值
    = 悄悄改掉所有未显式传参的历史回测。"""
    c = CostConfig()
    assert (c.commission_bps, c.stamp_duty_bps, c.transfer_bps) == (2.5, 5.0, 0.1)
    assert (c.impact_coef, c.impact_cap_bps, c.multiplier) == (0.5, 30.0, 1.0)
    p = PortfolioConstraints()
    assert (p.top_n, p.weighting) == (50, "equal")
    assert (p.max_single, p.max_industry, p.max_turnover) == (0.05, 0.20, 0.30)
    assert (BASE.position_floor, BASE.position_cap) == (0.20, 1.00)
    assert (BASE.benchmark, BASE.initial_capital) == ("000985.CSI", 1_000_000.0)
    assert BASE.compute_diagnostics is True and BASE.shuffle_seed is None
    assert dict(BASE.factor_param_override) == {}


def test_configs_are_frozen():
    """配置在回测过程中被改一下，param_hash 记的就是另一组参数（D7 溯源断链）。"""
    for obj in (BASE, BASE.cost, BASE.constraints):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.multiplier = 2.0
