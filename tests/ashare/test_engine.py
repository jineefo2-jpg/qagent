"""Task 13：回测引擎编排（`ashare/backtest/engine.py` + `store.py`）。

引擎自己不做任何策略决策 —— 它把十一个模块接在一起。所以本文件守的不是算法，
是**接缝**：每一处接错都会产出一条看起来完全正常、却在撒谎的净值曲线。

十五条接缝，每条一到两个用例（编号与任务书一致）：

  1  `prev_weights` 每期由持仓重算，不是把上期 `build_targets` 的返回值喂回来
  2  两个 `equity`：`simulate` 收货币、`BacktestResult.equity` 是初始 1.0 的净值指数
  3  `simulate` 收的是**股数**，不是权重
  4/5 数据中断日（`targets is None`）仍然要调 `simulate`，退市照样强平
  6  `scores` 索引是**完整股票池**、算不出的位置留 NaN（先 dropna 会让覆盖率闸失明）
  7  `positions.intended_weight` = 换手裁剪【之前】的目标（`build_targets` 的第二个返回值）
  8  `metrics.compute` 的因子分母是【配置的】个数，不是观测最大值
  9  `adv20` / `range` 由引擎附列，`range` 是**滞后 20 日**平均振幅、剔 D9 占位行
  10 （未落地，见模块尾 ★）
  11 全程钉住快照，收尾再核一次；引擎不得中途 `open_db`
  12 `engine_version` 与 `(param_hash, data_snapshot_id)` 并列记账，**不进 hash**
  13 `run_backtest()` 不接受调仓频率参数
  14 空股票池当日显式跳过并告警（`compute_factor` 对空池是抛的）
  15 `preload` 在入口调一次，起点 = start − max(lookback_days)

架构 §4.3 的五条裁决（2026-08-21）各自的钉子：
  ① `equity` 日频盯市 —— 周频采样看不见周内低点，MDD 被低估（Calmar 的分母）
  ② `compute_diagnostics=True` 必须真的产出 ic / layers / attribution
  ③ `build_targets` 返回三元组，引擎不再造反事实（丢那次的 warnings = 降级不可见）
  ④ `metrics.compute(full=True)` 恒真，不由 `compute_diagnostics` 驱动
  ⑤ `save`/`load` 一半在 `backtest_run` 表、一半在 sidecar，任一半丢字段都要红

fixture 的股数 / 价格 / 净值一律取**除不尽**的值（global-constraints ★）：
`shares × price / equity` 这一层往返换算太多，整齐的数字会让「按权重反推股数」
「把上期目标当 prev_w」这类变异与真实现逐位相同，测试绿的是算术不是代码。

★ 未落地的第 10 条：任务书要求「`derived_store.read_factor_values` 冷库的逐因子告警
  不要无条件汇入 `BacktestResult.warnings`」，前提是引擎走「优先 store.read」的缓存路径。
  本实现**没有**这条路径 —— `factors/` 里没有 store 版的 `combine`，在 `engine.py` 里
  现写一个等于把 alpha 白名单、覆盖率闸、按剩余权重重新归一（全是策略）搬进编排层，
  既撞架构 A5，又把钱的路径分了叉。理由与去处见任务报告。
"""
from __future__ import annotations

import datetime as dt
import inspect
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from ashare.backtest import engine, execution, store as bt_store
from ashare.backtest.engine import run_backtest
from ashare.backtest.types import BacktestConfig, CostConfig, PortfolioConstraints
from ashare.data.query import QueryError

MASK_COLS = ["can_buy", "can_sell", "reason", "open_hfq", "close_hfq", "amount", "amplitude"]

# ── 除不尽的世界（global-constraints ★）──────────────────────────────────────
DAYS = [dt.date(2024, 3, 4) + dt.timedelta(days=i) for i in range(30)]
WEEKLY = DAYS[4::5]                       # 6 个调仓日，最后一个的 next_trade_date 是 None
CODES = ["AAA.SZ", "BBB.SZ", "CCC.SH", "DDD.SZ"]
CAPITAL = 3_141_593.0                     # 本金也除不尽

_BASE = {"AAA.SZ": 55.037188, "BBB.SZ": 13.884723, "CCC.SH": 87.219461, "DDD.SZ": 7.302957}
# 开盘/收盘的比值**逐票不同**。全票同一个比值时它就是一个纯量，
# 会在 `shares × price / Σ(shares × price)` 里整个约掉 ——「拿开盘价度量 T 收盘权重」
# 这个变异因此与真实现逐位相同（实测：第一轮它就是这么逃掉的）。
_OPEN_K = {"AAA.SZ": 0.993713, "BBB.SZ": 1.004271, "CCC.SH": 0.987619, "DDD.SZ": 1.011533}
_IND = {"AAA.SZ": "银行", "BBB.SZ": "白酒", "CCC.SH": "钢铁", "DDD.SZ": "煤炭"}
# 分数固定不变：账本的变化只能来自价格漂移与换手预算，不能来自打分翻脸
_SCORES = {"AAA.SZ": 1.618034, "BBB.SZ": 0.577216, "CCC.SH": -0.301030, "DDD.SZ": -1.414214}


# 逐票不同的 log_mv（风格归因的规模回归元）—— 与 `pipeline.neutralize` 减掉的同一个定义
_LOG_MV = {"AAA.SZ": 23.417231, "BBB.SZ": 21.098344, "CCC.SH": 24.771903, "DDD.SZ": 20.334517}

# 诊断要够宽的横截面：IC 至少 5 只、分层 10 层、风格回归 30 只（`metrics` 的三个下限）。
# 默认池仍是那 4 只 —— 只有诊断用例把池换宽（`world["q"].universe` 是逐日覆盖的）。
WIDE = [f"W{i:02d}.SZ" for i in range(32)]
for _i, _c in enumerate(WIDE):
    _BASE[_c] = 31.415927 + _i * 7.208710
    _OPEN_K[_c] = 0.991037 + _i * 0.000713           # 逐票不同，不会在归一化里约掉
    _IND[_c] = ("银行", "白酒", "钢铁", "煤炭")[_i % 4]
    _SCORES[_c] = 2.302585 - _i * 0.057722
    _LOG_MV[_c] = 22.026465 + _i * 0.141421

# date -> 乘数。只给某一天砸个坑，用来分辨「日频盯市」与「按调仓日采样」：
# 坑挖在非调仓、非执行日上，周频曲线上它根本不存在。
_DIP: dict = {}


def close_px(code: str, d: dt.date) -> float:
    i = DAYS.index(d)
    return (_BASE[code] * (1.00713 ** i) + 0.0031 * i) * _DIP.get(d, 1.0)


def open_px(code: str, d: dt.date) -> float:
    # 开收比**逐票且逐日**都不同。只逐票不同还不够：持有期收益是 open(τ₂)/open(τ₁)，
    # 一个只随股票变化的常数在这个比值里【整个约掉】—— 于是「拿收盘价算持有期收益」
    # 这个变异与真实现逐位相同（global-constraints ★「会被约分的量」的时间序列版）。
    return close_px(code, d) * (_OPEN_K[code] + 0.000317 * DAYS.index(d))


class FakeQuery:
    """引擎用到的 `query` 表面。每个调用都记下来，接缝才测得到。"""

    QueryError = QueryError

    def __init__(self):
        self.snapshot = "snap-A1b2"
        self.pin_calls: list = []
        self.snapshot_calls = 0
        self.preload_args: list = []
        self.open_db_calls = 0
        self.universe: dict = {}          # date -> list[str]（缺省全池）
        self.suspended: set = set()       # (code, date)
        self.delisted: set = set()        # (code, date)
        self.sealed_up: set = set()       # (code, date) 一字涨停，买不进
        self.placeholder: set = set()     # (code, date) D9 占位行（is_suspended + vol=0）
        self.zero_vol: set = set()        # (code, date) 源自己给 vol=0，但 is_suspended=False
        self.index_rows = True
        self.mask_calls: list = []
        self.bars_calls: list = []

    # ── 连接 / 快照 ──
    def snapshot_id(self, *, pin: bool = False) -> str:
        self.snapshot_calls += 1
        self.pin_calls.append(pin)
        return self.snapshot

    def open_db(self, *a, **k):
        self.open_db_calls += 1

    def preload(self, start, end, tables=("daily_bar", "daily_basic")):
        self.preload_args.append((start, end, tuple(tables)))

    # ── 日历 ──
    def get_trade_dates(self, as_of_date, *, start=None, freq="D"):
        days = [d for d in DAYS if d <= as_of_date and (start is None or d >= start)]
        return days if freq == "D" else [d for d in WEEKLY if d in days]

    def next_trade_date(self, as_of_date, n: int = 1):
        later = [d for d in DAYS if d > as_of_date]
        return later[n - 1] if len(later) >= n else None

    # ── 池 / 元数据 ──
    def get_universe(self, as_of_date, **k):
        return list(self.universe.get(as_of_date, CODES))

    def get_industry(self, as_of_date, ts_codes=None, level="l1", **k):
        codes = list(ts_codes if ts_codes is not None else CODES)
        return pd.Series([_IND[c] for c in codes], index=codes, name="sw_l1")

    # ── 行情 ──
    def get_price_panel(self, as_of_date, ts_codes, field="close", lookback=250, adjust="hfq"):
        # ★ 必须真的按 `field` 分流：假实现无论问什么都给收盘价的话，
        #   「拿开盘价当 T 收盘权重的分母」这个变异永远杀不掉（实测第一轮它就这么逃的）。
        get = {"close": close_px, "open": open_px}[field]
        codes = list(ts_codes)
        days = [d for d in DAYS if d <= as_of_date][-lookback:]
        data = {c: [np.nan if (c, d) in self.suspended else get(c, d) for d in days]
                for c in codes}
        return pd.DataFrame(data, index=days).rename_axis("trade_date")

    def get_bars(self, as_of_date, ts_codes, *, lookback=None, start=None,
                 fields=("open", "high", "low", "close", "vol", "amount"), adjust="hfq"):
        self.bars_calls.append({"as_of": as_of_date, "codes": list(ts_codes),
                                "lookback": lookback, "fields": tuple(fields)})
        days = [d for d in DAYS if d <= as_of_date]
        if lookback:
            days = days[-lookback:]
        rows, idx = [], []
        for c in ts_codes:
            for d in days:
                ph = (c, d) in self.placeholder or (c, d) in self.zero_vol
                px = close_px(c, d)
                # D9 占位行：high == low == pre_close（振幅 0）、vol = 0、amount = 0。
                # `zero_vol` 那一批 is_suspended 是 False —— 源自己给了 vol=0，D9 说它同样算停牌。
                rec = {"open": px, "high": px, "low": px, "close": px, "pre_close": px,
                       "vol": 0.0, "amount": 0.0,
                       "is_suspended": (c, d) in self.placeholder} if ph else {
                    "open": open_px(c, d), "high": px * 1.02341, "low": px * 0.98117,
                    "close": px, "pre_close": px * 0.99411,
                    "vol": 913_517.0, "amount": 8_137_211.0 + 1000.0 * DAYS.index(d),
                    "is_suspended": False}
                rows.append(rec)
                idx.append((c, d))
        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["ts_code", "trade_date"]))
        return df[[*fields, "is_suspended"]]

    def get_index_bars(self, as_of_date, index_code, lookback=250, fields=("close",)):
        if not self.index_rows:
            return pd.DataFrame(columns=list(fields)).rename_axis("trade_date")
        days = [d for d in DAYS if d <= as_of_date][-lookback:]
        return pd.DataFrame({"close": [3011.37 * (1.00291 ** i) for i in range(len(days))]},
                            index=days).rename_axis("trade_date")[list(fields)]

    # ── 执行时点 ──
    def get_tradable_mask(self, exec_date, ts_codes):
        self.mask_calls.append((exec_date, list(ts_codes)))
        rows = []
        for c in ts_codes:
            if (c, exec_date) in self.delisted:
                last = close_px(c, exec_date)
                rows.append((c, False, True, "delisted", last, last, np.nan, np.nan))
            elif (c, exec_date) in self.suspended:
                # D9 占位行：OHLC = 前收。掩码两侧 False，但**价格是有的**（真实现如此）——
                # 否则 `simulate` 的无价守卫会把每一次停牌都当成数据有洞。
                last = max(d for d in DAYS if d < exec_date and (c, d) not in self.suspended)
                rows.append((c, False, False, "suspended", close_px(c, last),
                             close_px(c, last), 0.0, 0.0))
            elif (c, exec_date) in self.sealed_up:
                rows.append((c, False, True, "limit_up_seal", open_px(c, exec_date),
                             close_px(c, exec_date), 9.1e6, 0.0))
            else:
                rows.append((c, True, True, "", open_px(c, exec_date), close_px(c, exec_date),
                             9.1e6, 0.02137))
        return pd.DataFrame(rows, columns=["ts_code", *MASK_COLS]).set_index("ts_code")


class _Spec:
    def __init__(self, lookback_days: int):
        self.lookback_days = lookback_days


@pytest.fixture
def world(monkeypatch):
    """装好假 query / 假 combine / 假 get_factor，并把每个接缝上的调用录下来。"""
    q = FakeQuery()
    rec: dict = {"combine": [], "targets": [], "simulate": [], "metrics": [], "oos": [],
                 "panel": [], "factor": []}
    scores_by_date: dict = {}
    # max = 251，preload 起点由它决定
    lookbacks = {"f_alpha": 66, "f_beta": 251, "f_gamma": 30}
    dropped_by_date: dict = {}

    def fake_combine(weights, as_of_date, universe, *, use_store=False):
        rec["combine"].append({"weights": dict(weights), "date": as_of_date,
                               "universe": list(universe)})
        base = scores_by_date.get(as_of_date, _SCORES)
        s = pd.Series([base.get(c, float("nan")) for c in universe],
                      index=list(universe), name="score", dtype=float)
        warns = []
        k = dropped_by_date.get(as_of_date, 0)
        if k:
            warns.append(f"{as_of_date} 合成剔除 {k}/{len(weights)} 个因子并按剩余权重重新归一：x")
        return s, warns

    real_bt = engine.build_targets
    real_sim = engine.simulate
    real_metrics = engine.metrics.compute

    def spy_targets(scores, target_position, prev_weights, industry, constraints):
        out = real_bt(scores, target_position, prev_weights, industry, constraints)
        rec["targets"].append({"scores": scores.copy(), "pi": target_position,
                               "prev": pd.Series(prev_weights).copy(),
                               "constraints": constraints, "out": out[0], "intended": out[1]})
        return out

    def fake_panel(names, as_of_date, universe, *, processed=True, use_store=False):
        """IC 面板：逐因子一列，列间用除不尽的系数拉开（同一列会让两个因子 IC 相同）。"""
        rec["panel"].append({"names": list(names), "date": as_of_date, "processed": processed})
        base = scores_by_date.get(as_of_date, _SCORES)
        return pd.DataFrame(
            {n: [base.get(c, float("nan")) * (1.0 + j * 0.318310) - j * 0.161803
                 for c in universe] for j, n in enumerate(names)},
            index=list(universe)), []

    def fake_factor(name, as_of_date, universe, *, processed=True, **kw):
        rec["factor"].append({"name": name, "date": as_of_date, "processed": processed})
        return pd.Series([_LOG_MV[c] for c in universe], index=list(universe),
                         name=name, dtype=float), []

    def spy_sim(exec_date, targets, prev_holdings, equity, *, signal_date):
        out = real_sim(exec_date, targets, prev_holdings, equity, signal_date=signal_date)
        rec["simulate"].append({"exec_date": exec_date, "targets": targets,
                                "prev_holdings": pd.Series(prev_holdings).copy(),
                                "equity": equity, "signal_date": signal_date,
                                "holdings": out[1]})
        return out

    def spy_metrics(*a, **kw):
        rec["metrics"].append({"args": a, "kwargs": kw})
        return real_metrics(*a, **kw)

    monkeypatch.setattr(engine, "query", q)
    monkeypatch.setattr(execution, "get_tradable_mask", q.get_tradable_mask)
    monkeypatch.setattr(engine, "combine", fake_combine)
    monkeypatch.setattr(engine, "get_factor", lambda n: _Spec(lookbacks[n]))
    monkeypatch.setattr(engine, "compute_panel", fake_panel)
    monkeypatch.setattr(engine, "compute_factor", fake_factor)
    monkeypatch.setattr(engine, "build_targets", spy_targets)
    monkeypatch.setattr(engine, "simulate", spy_sim)
    monkeypatch.setattr(engine.metrics, "compute", spy_metrics)
    monkeypatch.setattr(engine, "append_oos_run",
                        lambda r: (rec["oos"].append(r), (False, []))[1])

    return {"q": q, "rec": rec, "scores": scores_by_date, "lookbacks": lookbacks,
            "dropped": dropped_by_date}


def make_cfg(**kw) -> BacktestConfig:
    base = dict(
        start=DAYS[0], end=DAYS[-1],
        factors=(("f_alpha", 1.0), ("f_beta", 1.0)),
        constraints=PortfolioConstraints(top_n=2, weighting="equal", max_single=0.6,
                                         max_industry=1.0, max_turnover=0.37),
        cost=CostConfig(), initial_capital=CAPITAL, compute_diagnostics=False)
    base.update(kw)
    return BacktestConfig(**base)


def real_target_calls(rec) -> list:
    """裁决 ③ 之后每次 `build_targets` 都是一次真实调仓（不再有 max_turnover=inf 的反事实）。"""
    return list(rec["targets"])


# ══════════════ 冒烟：整段跑得通 ══════════════

def test_multi_period_run_produces_a_coherent_result(world):
    res = run_backtest(make_cfg())

    assert res.param_hash == make_cfg().param_hash()
    assert res.data_snapshot_id == world["q"].snapshot
    assert res.engine_version == engine.ENGINE_VERSION
    # 6 个周频日，最后一个的 next_trade_date 是 None → 5 期真正成交
    assert len(real_target_calls(world["rec"])) == 5
    # 日频（裁决 ①）：首个调仓日到日历末端，每个交易日一格 —— 不是 6 个调仓日采样点
    assert list(res.equity.index) == [d for d in DAYS if d >= WEEKLY[0]]
    assert res.equity.iloc[0] == 1.0                 # 建仓前全是现金
    assert len(res.trades) > 0
    assert set(res.positions.index.get_level_values(0)) == set(WEEKLY[:5])
    assert res.metrics["n_trades"] == len(res.trades)

    # `price_hfq` 存 **T 收盘**后复权价：`metrics` 报的换手要与 `build_targets` 的
    # 换手上限同口径（都在 T 收盘度量）。存 τ 开盘价，报告里的换手就和预算对不上。
    pos = res.positions.xs(WEEKLY[1], level=0)
    for c in pos.index:
        assert float(pos.loc[c, "price_hfq"]) == pytest.approx(close_px(c, WEEKLY[1]),
                                                               rel=1e-12)
        assert float(pos.loc[c, "industry"] == _IND[c])


def test_calendar_end_returns_none_and_the_run_ends_cleanly(world):
    res = run_backtest(make_cfg())          # WEEKLY[-1] == DAYS[-1]，其后无交易日
    assert WEEKLY[-1] not in res.positions.index.get_level_values(0)
    assert any("日历末端" in w for w in res.warnings)


def test_progress_is_reported_once_per_rebalance_date(world):
    seen: list = []
    run_backtest(make_cfg(), on_progress=lambda i, n: seen.append((i, n)))
    # 末期在日历尽头 break，进度停在 5/6 —— 停在哪儿本身就是「回测没跑满」的信号
    assert seen == [(i, len(WEEKLY)) for i in range(1, len(WEEKLY))]


# ══════════════ 1 · prev_weights 每期重算 ══════════════

def test_prev_weights_are_recomputed_from_holdings_not_fed_back(world):
    """把上期 `build_targets` 的返回值喂回来，等于假装账本没随价格漂移过。"""
    res = run_backtest(make_cfg())
    calls = real_target_calls(world["rec"])
    sims = world["rec"]["simulate"]

    assert calls[0]["prev"].empty                      # 首期空账本
    for k in range(1, len(calls)):
        prev, last_targets = calls[k]["prev"], calls[k - 1]["out"]
        assert not prev.empty
        common = prev.index.intersection(last_targets.index)
        assert len(common) > 0
        assert not np.allclose(prev.reindex(common).to_numpy(),
                               last_targets.reindex(common).to_numpy(), atol=1e-9)

    # 第二期做**逐位精确**的复算：现金流与持仓都能从公开返回值里还原出来。
    # 只钉「与股数成正比」是不够的 —— 那样「拿 τ 开盘权益当分母」这个变异活得下来，
    # 而它正是把两个 equity 口径混起来的那一步。
    t = WEEKLY[1]
    first = res.trades[res.trades["exec_date"] == DAYS[DAYS.index(WEEKLY[0]) + 1]]
    amt, side = first["amount"].astype(float), first["side"].astype(str)
    cash = CAPITAL - float(amt[side == "BUY"].sum()) + float(amt[side == "SELL"].sum()) \
        - float(first["total_cost"].sum())
    held = sims[0]["holdings"]
    px = pd.Series({c: close_px(c, t) for c in held.index})       # T 收盘，不是 τ 开盘
    equity_t = cash + float((held * px).sum())
    want = held * px / equity_t
    got = calls[1]["prev"]
    # rtol=0 不可省：权重量级 0.5、开收比差 ~0.6%，np.allclose 默认的 rtol=1e-5
    # 恰好宽到让「拿开盘价度量 prev_w」这个变异活下来（实测第一轮它就这么逃的）。
    assert np.allclose(got.reindex(want.index).to_numpy(), want.to_numpy(),
                       rtol=0.0, atol=1e-12)
    open_based = held * pd.Series({c: open_px(c, t) for c in held.index})
    assert not np.allclose(got.reindex(want.index).to_numpy(),
                           (open_based / (cash + float(open_based.sum()))).to_numpy(),
                           rtol=0.0, atol=1e-9), "用 τ 开盘价度量 prev_w = 把隔夜跳空算成一笔调仓"


def test_prev_weights_track_a_price_move_between_periods(world):
    """两期之间只有价格在动，`prev_w` 就必须跟着动 —— 这是「重算 vs 喂回」的判据。"""
    run_backtest(make_cfg())
    calls = real_target_calls(world["rec"])
    p1, p2 = calls[1]["prev"], calls[2]["prev"]
    common = p1.index.intersection(p2.index)
    assert len(common) > 0
    assert not np.allclose(p1.reindex(common).to_numpy(), p2.reindex(common).to_numpy(), atol=1e-9)


# ══════════════ 2 · 两个 equity ══════════════

def test_simulate_gets_currency_equity_while_result_equity_is_a_net_value_index(world):
    res = run_backtest(make_cfg())
    first = world["rec"]["simulate"][0]
    assert first["equity"] == pytest.approx(CAPITAL)          # 货币口径，首期 = 本金
    assert res.equity.iloc[0] == 1.0                           # 净值指数
    assert 0.1 < float(res.equity.iloc[-1]) < 10.0
    for s in world["rec"]["simulate"]:
        assert s["equity"] > CAPITAL / 10                       # 一直是钱，不是权重


def test_metrics_receives_initial_capital_so_cost_drag_is_a_ratio(world):
    res = run_backtest(make_cfg())
    kw = world["rec"]["metrics"][0]["kwargs"]
    assert kw["initial_capital"] == CAPITAL
    # 净值是日频 → 252（裁决 ①）。留着 52 的话年化收益 / Sharpe 全按周频算，
    # 而喂进去的是一条日频曲线（§9：一律按入参序列自己的频率）。
    assert kw["periods_per_year"] == engine._PERIODS_PER_YEAR == 252
    assert res.metrics["cost_total"] > 1.0                      # 钱
    assert 0.0 < res.metrics["cost_drag_annual"] < 1.0          # 比例
    assert not any("量纲" in w for w in res.warnings)
    # 净值按 exec_date 索引 —— 对不齐的话成本一分都进不了拖累，而指标照出
    assert not any("不在净值曲线上" in w for w in res.warnings)


def _cash_before(res, exec_day: dt.date) -> float:
    """本金 − 之前所有买入 + 之前所有卖出 − 之前所有费用。全部从公开返回值还原。"""
    tr = res.trades[res.trades["exec_date"] < exec_day]
    amt, side = tr["amount"].astype(float), tr["side"].astype(str)
    return (CAPITAL - float(amt[side == "BUY"].sum()) + float(amt[side == "SELL"].sum())
            - float(tr["total_cost"].sum()))


def test_the_equity_handed_to_simulate_is_the_book_valued_at_tau_open(world):
    """货币权益 = 现金 + 持仓 × τ 开盘价。传错这个数，`simulate` 的 Δw 就全错。"""
    res = run_backtest(make_cfg())
    sims = world["rec"]["simulate"]
    for k in range(1, len(sims)):
        s, held = sims[k], sims[k - 1]["holdings"]
        px = pd.Series({c: open_px(c, s["exec_date"]) for c in held.index})
        want = _cash_before(res, s["exec_date"]) + float((held * px).sum())
        assert s["equity"] == pytest.approx(want, rel=1e-12)
        assert s["equity"] != pytest.approx(CAPITAL, rel=1e-6)      # 账本真的在漂移


def test_a_delisted_holding_is_valued_at_the_b8_haircut_not_the_open(world):
    """退市股按开盘价估值会先虚增权益，再在成交（`close_hfq × 0.5`）时莫名亏掉一笔。"""
    victim = "AAA.SZ"
    exec_day = DAYS[DAYS.index(WEEKLY[2]) + 1]
    world["q"].delisted.add((victim, exec_day))
    res = run_backtest(make_cfg())

    s = [x for x in world["rec"]["simulate"] if x["exec_date"] == exec_day][0]
    held = s["prev_holdings"]
    assert victim in held.index
    cash = _cash_before(res, exec_day)
    haircut = pd.Series({c: (close_px(c, exec_day) * 0.5 if c == victim
                             else open_px(c, exec_day)) for c in held.index})
    naive = pd.Series({c: open_px(c, exec_day) for c in held.index})
    assert s["equity"] == pytest.approx(cash + float((held * haircut).sum()), rel=1e-12)
    assert s["equity"] != pytest.approx(cash + float((held * naive).sum()), rel=1e-9)


def _cash_after(res, day: dt.date) -> float:
    """本金 − 截至 day（含）的买入 + 卖出 − 费用。全部从公开返回值还原。"""
    tr = res.trades[res.trades["exec_date"] <= day]
    amt, side = tr["amount"].astype(float), tr["side"].astype(str)
    return (CAPITAL - float(amt[side == "BUY"].sum()) + float(amt[side == "SELL"].sum())
            - float(tr["total_cost"].sum()))


def test_the_net_value_index_is_marked_every_trading_day_at_the_close(world):
    """裁决 ①：净值 = (现金 + Σ 持仓 × 当日**收盘**价) / 本金，每个交易日一格。

    旧口径把 `(τ 开盘权益 − 费用)/本金` 记在执行日上、其余日子干脆没有 —— 那条曲线
    看不见任何周内波动，最大回撤系统性偏小，而 MDD 正是 Calmar 的分母。
    """
    res = run_backtest(make_cfg())
    sims = world["rec"]["simulate"]
    for k, s in enumerate(sims):
        ex = s["exec_date"]
        nxt = sims[k + 1]["exec_date"] if k + 1 < len(sims) else None
        cash, held = _cash_after(res, ex), s["holdings"]
        for d in [x for x in DAYS if x >= ex and (nxt is None or x < nxt)]:
            px = pd.Series({c: close_px(c, d) for c in held.index})
            assert float(res.equity.loc[d]) == pytest.approx(
                (cash + float((held * px).sum())) / CAPITAL, rel=1e-12)
        # 执行日那一格也是【收盘】价，不是「τ 开盘权益 − 费用」（旧口径差一个日内涨跌）
        cost = float(res.trades[res.trades["exec_date"] == ex]["total_cost"].sum())
        assert float(res.equity.loc[ex]) != pytest.approx((s["equity"] - cost) / CAPITAL,
                                                          rel=1e-9)


def test_a_sold_off_name_stops_marking_the_curve(world):
    """持仓是阶梯函数：ffill 之前必须先 fillna(0)，否则卖掉的票会继续替净值赚钱。"""
    _rotation(world, WEEKLY[2])          # 排名翻转 + 放开预算 → 上期那两只被卖光
    res = run_backtest(make_cfg(constraints=PortfolioConstraints(
        top_n=2, weighting="equal", max_single=0.6, max_industry=1.0, max_turnover=5.0)))
    sims = world["rec"]["simulate"]
    gone = [(k, c) for k in range(1, len(sims))
            for c in sims[k - 1]["holdings"].index if c not in sims[k]["holdings"].index]
    assert gone, "fixture 里没有任何一只票被卖光，这条用例什么都没钉住"
    k, code = gone[0]
    ex = sims[k]["exec_date"]
    held = sims[k]["holdings"]
    px = pd.Series({c: close_px(c, ex) for c in held.index})
    assert code not in held.index
    assert float(res.equity.loc[ex]) == pytest.approx(
        (_cash_after(res, ex) + float((held * px).sum())) / CAPITAL, rel=1e-12)


def test_the_daily_index_covers_an_exec_date_past_the_end(world):
    """末个调仓日落在区间末尾时 τ = T+1 已经越过 `end`，而那一笔仍然成交 ——
    日频索引不盖住它，这天的成本就「不在净值曲线上」，拖累一分都算不进去。"""
    res = run_backtest(make_cfg(end=WEEKLY[4]))
    last_exec = DAYS[DAYS.index(WEEKLY[4]) + 1]

    assert last_exec > WEEKLY[4]
    assert res.equity.index[-1] == last_exec
    assert (res.trades["exec_date"] == last_exec).any()
    assert not any("不在净值曲线上" in w for w in res.warnings)


def test_an_intra_week_trough_shows_up_in_the_max_drawdown(world):
    """裁决 ①：坑挖在【非调仓、非执行】日上 —— 按调仓频采样的曲线上它根本不存在。"""
    dip = DAYS[7]
    execs = [DAYS[DAYS.index(w) + 1] for w in WEEKLY[:5]]
    assert dip not in WEEKLY and dip not in execs
    _DIP[dip] = 0.791300
    try:
        res = run_backtest(make_cfg())
    finally:
        _DIP.clear()

    assert float(res.equity.loc[dip]) < 0.85
    assert res.metrics["max_drawdown"] > 0.15
    # 同一次运行，只按执行日采样 —— 回撤几乎为 0，Calmar 因此虚高
    weekly = res.equity.reindex(execs)
    assert float((1.0 - weekly / weekly.cummax()).max()) < 0.02


# ══════════════ 3 · simulate 收股数 ══════════════

def test_simulate_receives_share_counts_not_weights(world):
    run_backtest(make_cfg())
    sims = world["rec"]["simulate"]
    assert sims[0]["prev_holdings"].empty
    for k in range(1, len(sims)):
        prev = sims[k]["prev_holdings"]
        assert not prev.empty
        assert prev.abs().max() > 100.0                          # 股数量级，不是 ≤1 的权重
        pd.testing.assert_series_equal(prev, sims[k - 1]["holdings"], check_names=False)


# ══════════════ 4 / 5 · 数据中断日 ══════════════

def _outage(world, on_date):
    world["scores"][on_date] = {c: float("nan") for c in CODES}


def test_outage_day_still_calls_simulate_with_targets_none(world):
    _outage(world, WEEKLY[2])
    res = run_backtest(make_cfg())
    hit = [s for s in world["rec"]["simulate"] if s["signal_date"] == WEEKLY[2]]
    assert len(hit) == 1, "中断日跳过 simulate = 退市股永远留在账上（§5.5 幽灵资产）"
    assert hit[0]["targets"] is None
    assert any("数据中断" in w for w in res.warnings)


def test_outage_day_still_liquidates_a_delisting(world):
    """中断 + 退市同日：账本必须清掉那只票，否则 §5.5 的洞就开回来了。"""
    victim = "AAA.SZ"
    # 先让它建上仓（前两期正常），第三期中断且当日退市
    _outage(world, WEEKLY[2])
    exec_day = DAYS[DAYS.index(WEEKLY[2]) + 1]
    world["q"].delisted.add((victim, exec_day))
    res = run_backtest(make_cfg())

    held_before = world["rec"]["simulate"][1]["holdings"]
    assert victim in held_before.index and held_before[victim] > 0
    held_after = world["rec"]["simulate"][2]["holdings"]
    assert victim not in held_after.index
    assert any("退市清仓" in w for w in res.warnings)
    sold = res.trades[(res.trades["ts_code"] == victim) & (res.trades["side"] == "SELL")
                      & (res.trades["exec_date"] == exec_day)]
    assert len(sold) == 1
    # B8 折价（`close_hfq × 0.5`）—— 按开盘价清仓会系统性高估退市股的回收
    assert float(sold["price_hfq"].iloc[0]) == pytest.approx(close_px(victim, exec_day) * 0.5,
                                                            rel=1e-12)


# ══════════════ 6 · scores 索引 = 完整股票池 ══════════════

def test_scores_reach_build_targets_on_the_full_universe_with_nan_holes(world):
    """先 dropna 会让 `build_targets` 的 50% 覆盖率闸永远读到 100%。"""
    world["scores"][WEEKLY[1]] = {"AAA.SZ": 1.618034, "BBB.SZ": 0.577216}   # 另两只算不出
    run_backtest(make_cfg())
    call = [c for c in real_target_calls(world["rec"]) if len(c["scores"]) and
            c["scores"].isna().any()]
    assert call, "NaN 占位被剔掉了：覆盖率闸从此永远读 100%"
    sc = call[0]["scores"]
    assert list(sc.index) == CODES
    assert int(sc.isna().sum()) == 2


def test_a_low_coverage_day_is_read_as_an_outage_end_to_end(world):
    """4 只票只剩 1 只有分数 → 25% < 50% → 中断（而不是拿 1 只票建满仓）。"""
    world["scores"][WEEKLY[1]] = {"AAA.SZ": 1.618034}
    res = run_backtest(make_cfg())
    hit = [s for s in world["rec"]["simulate"] if s["signal_date"] == WEEKLY[1]]
    assert hit[0]["targets"] is None
    assert any("覆盖率" in w for w in res.warnings)
    # 意图账本那次反事实调用会**逐字**产出同一条告警。把它也汇进来，
    # 报告里每一条降级都变成两条，`summary()` 的 5 条预算一半是回声。
    assert sum("评分覆盖率" in w for w in res.warnings) == 1


# ══════════════ 7 · intended_weight ══════════════

def _rotation(world, on_date):
    """把排名整个翻过来：账本要全额换手 2.0，而预算只有 0.37 —— 裁剪必然发生。"""
    world["scores"][on_date] = {c: -v for c, v in _SCORES.items()}


def test_positions_carry_the_pre_clip_intended_weight(world):
    _rotation(world, WEEKLY[2])
    res = run_backtest(make_cfg())
    assert "intended_weight" in res.positions.columns
    rot = res.positions.xs(WEEKLY[2], level=0)
    # 意图：翻转后的前两名各 0.5；实际：换手预算只放行一部分
    assert float(rot["intended_weight"].sum()) == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(rot["intended_weight"].to_numpy(), rot["target_weight"].to_numpy())
    top2 = set(rot["intended_weight"].sort_values(ascending=False).index[:2])
    assert top2 == {"CCC.SH", "DDD.SZ"}
    # 归因才分得清「信号不行」与「换手预算把信号卡住了」
    assert not any("没有 intended_weight" in w for w in res.warnings)
    assert any("换手预算" in w for w in res.warnings)


def test_intended_weight_is_the_second_return_value_not_a_second_call(world):
    """裁决 ③：`build_targets` 每期只调一次。

    上一版用 `max_turnover=inf` 再调一次来造反事实，代价是**必须丢掉那次的 warnings**
    （否则每条降级报两遍）—— 丢 warning 本身就违反「降级必须可见」。
    """
    _rotation(world, WEEKLY[2])
    res = run_backtest(make_cfg())
    calls = world["rec"]["targets"]
    assert len(calls) == 5, "每个调仓日恰好一次 build_targets"
    assert all(np.isfinite(c["constraints"].max_turnover) for c in calls), \
        "又出现了 max_turnover=inf 的反事实调用"

    # 落进 positions 的就是那一次的第二个返回值，不是引擎另算的一份
    want = calls[2]["intended"]
    got = res.positions.xs(WEEKLY[2], level=0)["intended_weight"]
    assert np.allclose(want.reindex(got.index).fillna(0.0).to_numpy(), got.to_numpy(),
                       rtol=0.0, atol=1e-12)
    # 「拿 final 当 intended」必须死：旋转日两者本来就不同
    assert not np.allclose(got.to_numpy(),
                           calls[2]["out"].reindex(got.index).fillna(0.0).to_numpy(),
                           rtol=0.0, atol=1e-9)


def test_a_name_the_budget_refused_entirely_still_carries_its_intent(world):
    """被换手预算【整只】挡下的票：final == 0 且 prev == 0 —— 只按 final 筛就把它
    从账本里裁掉，于是 `intended_weight` 恰好在它唯一有意义的那几行上整行消失。"""
    _rotation(world, WEEKLY[2])
    res = run_backtest(make_cfg())
    rot = res.positions.xs(WEEKLY[2], level=0)
    refused = rot[(rot["target_weight"].abs() < 1e-12) & (rot["intended_weight"] > 1e-12)]
    assert len(refused) >= 1, "旋转日没有任何一只票被整只挡下，这条用例什么都没钉住"
    assert (refused["filled_weight"].abs() < 1e-12).all()


# ══════════════ 8 · 因子分母 = 配置数 ══════════════

def test_metrics_gets_the_configured_factor_count_as_the_denominator(world):
    # ★ 必须是 3 个因子剔 1 个。2 剔 1 时「活下来几个」与「剔掉几个」都是 1 ——
    #   「把分子读成分母」这个变异与真实现逐位相同（实测第一轮它就这么逃的）。
    world["dropped"][WEEKLY[1]] = 1
    res = run_backtest(make_cfg(factors=(("f_alpha", 1.0), ("f_beta", 1.0), ("f_gamma", 1.0))))
    kw = world["rec"]["metrics"][0]["kwargs"]
    assert kw["n_factors_configured"] == 3
    fu = kw["factors_used"]
    assert fu is not None
    assert int(fu.loc[WEEKLY[0]]) == 3
    assert int(fu.loc[WEEKLY[1]]) == 2
    # `combine` 的降级也必须自己出现在报告里，不只是被折成一个计数
    assert any("合成剔除 1/3" in w for w in res.warnings)


def test_uniform_degradation_is_flagged_because_the_denominator_is_configured(world):
    """每期都只活 1 个：观测最大值也是 1，用它当分母一句告警都没有。"""
    for d in WEEKLY:
        world["dropped"][d] = 1
    res = run_backtest(make_cfg())
    assert res.metrics["n_factors_configured"] == 2
    assert any("全期最多只用上" in w for w in res.warnings)


def test_the_dropped_factor_warning_regex_matches_the_real_combine():
    """`factors.base.combine` 的告警文本是本引擎唯一能读到「活了几个因子」的通道。
    那句话改了措辞，`factors_used` 会静默退回「全都在用」—— 乐观的那一侧。"""
    from ashare.factors import base

    saved = dict(base.FACTOR_REGISTRY)
    base.FACTOR_REGISTRY.clear()
    try:
        @base.factor(name="_t13_good", direction=1, category="price", lookback_days=5,
                     neutralize=False, min_coverage=0.1)
        def _good(as_of_date, universe):
            return pd.Series([0.31831, 0.57722, -1.20206], index=list(universe))

        @base.factor(name="_t13_good2", direction=-1, category="price", lookback_days=5,
                     neutralize=False, min_coverage=0.1)
        def _good2(as_of_date, universe):
            return pd.Series([-0.86603, 0.41421, 2.71828], index=list(universe))

        @base.factor(name="_t13_sparse", direction=1, category="price", lookback_days=5,
                     neutralize=False, min_coverage=0.9)
        def _sparse(as_of_date, universe):
            return pd.Series([0.69315, float("nan"), float("nan")], index=list(universe))

        _, warns = base.combine({"_t13_good": 1.0, "_t13_good2": 1.0, "_t13_sparse": 1.0},
                                dt.date(2024, 3, 8), ["X.SZ", "Y.SZ", "Z.SH"])
    finally:
        base.FACTOR_REGISTRY.clear()
        base.FACTOR_REGISTRY.update(saved)

    # 3 个剔 1 个：分子 2、分母 3，两者分得开（2 剔 1 时它们都是 1）
    assert engine._factors_used(3, warns) == 2


# ══════════════ 9 · adv20 / range ══════════════

def test_trades_carry_adv20_and_a_lagged_20_day_amplitude(world):
    res = run_backtest(make_cfg())
    assert {"adv20", "range"} <= set(res.trades.columns)
    assert res.trades["adv20"].notna().all()
    assert res.trades["range"].notna().all()

    row = res.trades.iloc[0]
    t = WEEKLY[0]
    days = [d for d in DAYS if d <= t][-20:]
    exp_rng = float(np.mean([(close_px(row.ts_code, d) * 1.02341 - close_px(row.ts_code, d) * 0.98117)
                             / (close_px(row.ts_code, d) * 0.99411) for d in days]))
    assert row["range"] == pytest.approx(exp_rng, rel=1e-12)
    # 执行日当天的振幅（掩码的 amplitude）是**前视**，不能是它
    assert row["range"] != pytest.approx(0.02137, rel=1e-6)


def test_the_impact_window_is_read_at_the_signal_date_not_the_exec_date(world):
    run_backtest(make_cfg())
    calls = [c for c in world["q"].bars_calls if c["lookback"] == engine._IMPACT_WINDOW]
    assert calls
    for c in calls:
        assert c["as_of"] in WEEKLY                      # 信号日 T，不是 τ
        assert {"high", "low", "pre_close"} <= set(c["fields"])


def test_d9_placeholder_rows_are_dropped_from_adv20_and_range(world):
    """占位行振幅 0 / 成交额 0：混进去会把振幅均值拖低（低估成本）、把 ADV20 拉低（高估冲击）。"""
    t = WEEKLY[0]
    ph_days = [DAYS[1], DAYS[3]]                 # 5 根 K 线里两根是占位行
    for c in CODES:
        for d in ph_days:
            world["q"].placeholder.add((c, d))
    res = run_backtest(make_cfg())

    row = res.trades.iloc[0]
    live = [d for d in DAYS if d <= t][-20:]
    live = [d for d in live if (row.ts_code, d) not in world["q"].placeholder]
    exp_rng = float(np.mean([(close_px(row.ts_code, d) * 1.02341 - close_px(row.ts_code, d) * 0.98117)
                             / (close_px(row.ts_code, d) * 0.99411) for d in live]))
    exp_adv = float(np.mean([8_137_211.0 + 1000.0 * DAYS.index(d) for d in live]))
    assert row["range"] == pytest.approx(exp_rng, rel=1e-12)
    assert row["adv20"] == pytest.approx(exp_adv, rel=1e-12)


def test_source_rows_with_zero_volume_count_as_suspended_too(world):
    """D9：源给出 `vol=0` 的行同样算停牌。只看 `is_suspended` 会把振幅 0 的那几根
    放进均值 —— 成本算低，方向恰好在把净值画好看那一侧。"""
    t = WEEKLY[0]
    for c in CODES:
        for d in (DAYS[0], DAYS[2]):
            world["q"].zero_vol.add((c, d))
    res = run_backtest(make_cfg())

    row = res.trades.iloc[0]
    live = [d for d in DAYS if d <= t and (row.ts_code, d) not in world["q"].zero_vol]
    exp = float(np.mean([(close_px(row.ts_code, d) * 1.02341 - close_px(row.ts_code, d) * 0.98117)
                         / (close_px(row.ts_code, d) * 0.99411) for d in live]))
    assert row["range"] == pytest.approx(exp, rel=1e-12)
    all_days = [d for d in DAYS if d <= t]
    assert exp != pytest.approx(exp * len(live) / len(all_days), rel=1e-9)


def test_missing_impact_data_is_charged_at_the_cap_not_at_zero(world):
    """所有 20 日都是占位行 → 没有可用样本 → `charge` 按 30bp 封顶收费并告警。"""
    for c in CODES:
        for d in DAYS:
            world["q"].placeholder.add((c, d))
    res = run_backtest(make_cfg())
    assert res.trades["adv20"].isna().all()
    assert any("封顶" in w for w in res.warnings)


def test_a_suspended_holding_still_gets_a_t_close_price(world):
    """`build_targets` **拒收** NaN 的 prev_weights（「敞口算不出来」≠「没有敞口」）——
    停牌持仓的价必须由引擎 ffill 到最后一个有效收盘，否则整段回测在这里炸。"""
    t = WEEKLY[2]
    for d in [x for x in DAYS if WEEKLY[1] < x <= t]:
        world["q"].suspended.add(("AAA.SZ", d))
    res = run_backtest(make_cfg())

    prev = real_target_calls(world["rec"])[2]["prev"]
    assert "AAA.SZ" in prev.index and np.isfinite(prev["AAA.SZ"])
    pos = res.positions.xs(t, level=0)
    assert float(pos.loc["AAA.SZ", "price_hfq"]) == pytest.approx(close_px("AAA.SZ", WEEKLY[1]),
                                                                  rel=1e-12)

    # 日频盯市同样要 ffill：不 ffill 的话停牌那几天这只票在 `sum` 里被跳过（等于一文不值），
    # 净值曲线上凭空多出一段「跌了又涨」的假回撤，而它只是没开盘。
    d = DAYS[DAYS.index(WEEKLY[1]) + 3]
    assert ("AAA.SZ", d) in world["q"].suspended
    held = world["rec"]["simulate"][1]["holdings"]
    assert "AAA.SZ" in held.index
    px = pd.Series({c: close_px(c, WEEKLY[1] if c == "AAA.SZ" else d) for c in held.index})
    assert float(res.equity.loc[d]) == pytest.approx(
        (_cash_after(res, d) + float((held * px).sum())) / CAPITAL, rel=1e-12)


# ══════════════ D6 证据链：blocked 必须一路带出来 ══════════════

def test_blocked_rows_survive_into_the_result(world):
    """每期都恰好达成目标权重的回测，描述的是一个不存在的市场。"""
    for d in DAYS:
        world["q"].sealed_up.add(("AAA.SZ", d))       # 一字涨停，买不进
    res = run_backtest(make_cfg())
    assert len(res.blocked) > 0
    assert set(res.blocked.columns) == {"exec_date", "ts_code", "intended_side",
                                        "intended_weight", "reason"}
    assert (res.blocked["reason"] == "limit_up_seal").any()
    assert "AAA.SZ" not in set(res.trades[res.trades["side"] == "BUY"]["ts_code"])
    assert res.metrics["d6_slippage_max"] > 0         # 缺口现算，不求和 intended_weight


# ══════════════ 11 · 快照钉住 ══════════════

def test_the_snapshot_is_pinned_once_at_entry_and_verified_at_the_end(world):
    res = run_backtest(make_cfg())
    assert world["q"].pin_calls[0] is True
    assert world["q"].pin_calls.count(True) == 1
    assert world["q"].pin_calls[-1] is False           # 收尾复核
    assert world["q"].snapshot_calls >= 2
    assert res.data_snapshot_id == world["q"].snapshot


def test_the_engine_never_reopens_the_database_mid_run(world):
    """钉住期间 `open_db` 会抛；而它【曾经】是静默换库的后门，引擎不能碰。"""
    run_backtest(make_cfg())
    assert world["q"].open_db_calls == 0


def test_a_snapshot_change_mid_run_aborts_instead_of_producing_a_result(world):
    q = world["q"]
    orig = q.snapshot_id

    def flip(*, pin=False):
        out = orig(pin=pin)
        if q.snapshot_calls == 1:              # 钉住之后、跑到一半换库
            q.snapshot = "snap-B9c8"
        return out

    q.snapshot_id = flip
    with pytest.raises(QueryError, match="快照"):
        run_backtest(make_cfg())


# ══════════════ 12 · engine_version 不进 hash ══════════════

def test_engine_version_is_recorded_beside_the_hash_never_inside_it(world, monkeypatch):
    a = run_backtest(make_cfg())
    monkeypatch.setattr(engine, "ENGINE_VERSION", "p2-engine-99")
    b = run_backtest(make_cfg())
    assert a.param_hash == b.param_hash, "engine_version 进了 hash = 每次引擎升级白送一次样本外"
    assert a.engine_version != b.engine_version
    assert b.engine_version == "p2-engine-99"


# ══════════════ 13 · 没有调仓频率参数 ══════════════

def test_macro_timing_is_refused_rather_than_silently_degraded(world):
    """静默退化成恒定满仓 = param_hash 写着择时、跑的却是另一个策略（D7 台账失真）。"""
    with pytest.raises(ValueError, match="macro_timing"):
        run_backtest(make_cfg(macro_timing=True))


def test_run_backtest_takes_no_rebalance_frequency_argument():
    sig = inspect.signature(run_backtest)
    # use_store 是架构 §4.3 补裁 ①（2026-08-24）钦定的 kwarg：不进 config/指纹、不改变结果。
    # 本测试锁的是「没有调仓频率旋钮」，不是「永不加参数」。
    assert list(sig.parameters) == ["config", "on_progress", "use_store"]
    banned = ("freq", "rebalance", "weekly", "period")
    assert not [p for p in sig.parameters if any(b in p.lower() for b in banned)]


# ══════════════ 14 · 空股票池 ══════════════

def test_an_empty_universe_day_is_skipped_loudly_not_fatally(world):
    world["q"].universe[WEEKLY[1]] = []
    res = run_backtest(make_cfg())
    assert not any(c["date"] == WEEKLY[1] for c in world["rec"]["combine"]), \
        "空池喂进 compute_factor 会抛，一天坏数据就炸掉十五年"
    assert not any(s["signal_date"] == WEEKLY[1] for s in world["rec"]["simulate"])
    assert any("股票池为空" in w for w in res.warnings)
    assert len(real_target_calls(world["rec"])) == 4       # 5 期减掉这一天


# ══════════════ 15 · preload ══════════════

def test_preload_is_called_once_and_covers_the_longest_factor_lookback(world):
    run_backtest(make_cfg())
    assert len(world["q"].preload_args) == 1
    start, end, tables = world["q"].preload_args[0]
    assert end == DAYS[-1]
    assert "daily_bar" in tables
    # 日历只有 30 天而 max(lookback_days)=251 → 只能钉到数据起点
    assert start == DAYS[0]


def test_preload_start_backs_off_by_the_longest_lookback(world):
    world["lookbacks"]["f_beta"] = 9
    world["lookbacks"]["f_alpha"] = 4
    run_backtest(make_cfg(start=DAYS[14]))
    start, _, _ = world["q"].preload_args[0]
    assert start == DAYS[14 - 9 + 1], "少 preload 一天，滚动窗口就少一天，因子静默变形"
    assert start < DAYS[14]


# ══════════════ 时序语义钉子（brief 验收）══════════════

def test_fills_happen_at_the_next_day_open_and_ignore_the_signal_day_close(world,
                                                                          monkeypatch):
    res = run_backtest(make_cfg())
    tr = res.trades[res.trades["exec_date"] == DAYS[DAYS.index(WEEKLY[0]) + 1]]
    assert len(tr) > 0
    for _, r in tr.iterrows():
        assert r["price_hfq"] == pytest.approx(open_px(r["ts_code"], r["exec_date"]), rel=1e-12)
        assert r["price_hfq"] != pytest.approx(close_px(r["ts_code"], WEEKLY[0]), rel=1e-6)

    # 把 T 日收盘价整体改掉 —— 成交价一分不动（证明不是收盘价成交）
    prices = dict(res.trades.set_index(["exec_date", "ts_code"])["price_hfq"])
    world2 = world
    monkeypatch.setitem(_BASE, "AAA.SZ", _BASE["AAA.SZ"])      # 占位，避免全局污染
    orig_panel = world2["q"].get_price_panel

    def bumped(as_of_date, ts_codes, field="close", lookback=250, adjust="hfq"):
        return orig_panel(as_of_date, ts_codes, field, lookback, adjust) * 1.31427

    world2["q"].get_price_panel = bumped
    res2 = run_backtest(make_cfg())
    for (d, c), px in dict(res2.trades.set_index(["exec_date", "ts_code"])["price_hfq"]).items():
        if (d, c) in prices:
            assert px == pytest.approx(prices[(d, c)], rel=1e-12)


# ══════════════ ② · 诊断三块必须真的产出 ══════════════

def _diag_cfg(**kw) -> BacktestConfig:
    return make_cfg(compute_diagnostics=True, **kw)


def _wide(world) -> list:
    """把股票池换成 36 只 —— 4 只横截面下 IC / 分层 / 风格回归全部退化成 NaN，
    「三块产出了」就只剩形状，测不出里面有没有数。"""
    pool = CODES + WIDE
    for d in DAYS:
        world["q"].universe[d] = pool
    return pool


def test_diagnostics_are_actually_produced_not_a_warning(world):
    """裁决 ②：`compute_diagnostics=True` 返回 `None` + 一条 warning 是不合格的静止态。"""
    _wide(world)
    res = run_backtest(_diag_cfg())

    assert res.ic is not None and res.layers is not None and res.attribution is not None
    assert not any("不产出" in w for w in res.warnings)
    # 5 期成交、4 期算得出持有期收益（末期没有下一个执行日）
    assert list(res.ic.index) == list(WEEKLY[:4])
    assert WEEKLY[4] not in res.ic.index
    assert list(res.ic.columns) == ["f_alpha__ic", "f_alpha__rank_ic",
                                    "f_beta__ic", "f_beta__rank_ic"]
    assert list(res.layers.columns) == [f"L{i}" for i in range(1, 11)]
    # 形状对而里面全是 NaN，等于「算了但什么都没算出来」——与没算只差一句话
    assert res.ic.notna().all().all()
    assert res.layers.notna().all().all()


def test_attribution_always_carries_the_size_row_that_can_falsify_the_ols_ruling(world):
    """§3.2 选 OLS 而非 √MV-WLS，**只能靠这一行被证伪** —— 它不能藏在任何开关后面，
    样本不足也只报 NaN 不删行。"""
    _wide(world)
    res = run_backtest(_diag_cfg())
    att = res.attribution.set_index(["block", "item"])
    rows = set(att.index)

    assert ("style", "size") in rows
    assert ("style", "size_sq") in rows          # 补救方向是非线性规模项，不是换回 WLS
    assert ("constraint", "turnover_budget") in rows
    assert [b for b, _ in rows].count("industry") >= 1
    # 36 只横截面 ≥ `metrics._MIN_STYLE_OBS`，所以这一行必须是【真数】不是占位 NaN
    assert np.isfinite(float(att.loc[("style", "size"), "exposure"]))
    assert not any("未被检验" in w for w in res.warnings)


def test_the_turnover_drag_row_is_non_zero_when_the_budget_actually_bites(world):
    """裁决 ②×③ 的合流：归因分得清「信号不行」与「换手把信号卡住了」，
    前提是 `intended_weight` 真的落进了 positions（自查出的那个 index bug）。"""
    _wide(world)
    _rotation(world, WEEKLY[2])
    res = run_backtest(_diag_cfg())
    row = res.attribution[(res.attribution["block"] == "constraint")
                          & (res.attribution["item"] == "turnover_budget")]

    assert len(row) == 1
    # 逐位钉住：Σ|intended − filled| / 期数，且**只数有持有期收益的那几期**
    dates = list(WEEKLY[:4])
    pos = res.positions[res.positions.index.get_level_values(0).isin(dates)]
    want = float((pos["intended_weight"] - pos["filled_weight"]).abs().sum()) / len(dates)
    assert float(row["exposure"].iloc[0]) == pytest.approx(want, rel=1e-12)
    assert want > 1e-9, "换手约束拖累恒为 0 = intended_weight 没落进它唯一有意义的那几行"
    assert not any("没有 intended_weight" in w for w in res.warnings)


def test_the_size_regressor_is_the_raw_log_mv_the_same_one_neutralize_subtracts(world):
    """`pipeline.neutralize` 减掉的是 `risk.log_mv` 的**原始值**。这里取 processed 的话
    两份定义分家，归因可以报「规模暴露已清零」而账本实际带着倾斜。"""
    run_backtest(_diag_cfg())
    calls = world["rec"]["factor"]

    assert calls and {c["name"] for c in calls} == {engine._SIZE_FACTOR} == {"log_mv"}
    assert all(c["processed"] is False for c in calls)
    # IC 面板反过来要【处理后】的 z 值（原始值量纲不一，秩相关也会被离群值带偏）
    assert all(c["processed"] is True for c in world["rec"]["panel"])
    assert all(c["names"] == ["f_alpha", "f_beta"] for c in world["rec"]["panel"])


def test_the_holding_period_return_runs_open_to_open_between_two_exec_days(world):
    """§5.1：持有期收益 = 下一执行日开盘 / 本执行日开盘 − 1。用收盘价算会把 T+1 那天的
    日内涨跌算进信号的功劳里 —— 而那段行情在成交价之后才发生。"""
    pool = _wide(world)
    res = run_backtest(_diag_cfg())
    t, t2 = WEEKLY[0], WEEKLY[1]
    e1 = DAYS[DAYS.index(t) + 1]
    e2 = DAYS[DAYS.index(t2) + 1]
    want = pd.Series({c: open_px(c, e2) / open_px(c, e1) - 1.0 for c in pool})

    # IC 是分数与持有期收益的秩相关：用同一份收益重算，必须逐位相同
    sc = pd.Series({c: _SCORES[c] for c in pool})
    assert float(res.ic.loc[t, "f_alpha__rank_ic"]) == pytest.approx(
        float(sc.corr(want, method="spearman")), rel=1e-12)
    close_based = pd.Series({c: close_px(c, e2) / close_px(c, e1) - 1.0 for c in pool})
    assert not np.allclose(want.to_numpy(), close_based.to_numpy(), rtol=0.0, atol=1e-9)
    assert float(res.ic.loc[t, "f_alpha__rank_ic"]) != pytest.approx(
        float(sc.corr(close_based, method="spearman")), rel=1e-9)


def test_a_single_rebalance_period_says_it_could_not_diagnose(world):
    """一期算不出持有期收益。`None` 必须配一条 warning —— 「没算」与「算出来是空的」
    在返回值上长得一模一样。"""
    res = run_backtest(_diag_cfg(start=DAYS[0], end=WEEKLY[0]))

    assert (res.ic, res.layers, res.attribution) == (None, None, None)
    assert any("不足两个执行日" in w for w in res.warnings)


def test_diagnostics_off_costs_nothing_and_computes_nothing(world):
    res = run_backtest(make_cfg())                 # compute_diagnostics=False
    assert (res.ic, res.layers, res.attribution) == (None, None, None)
    assert world["rec"]["panel"] == [] and world["rec"]["factor"] == []


# ══════════════ ④ · metrics.compute(full=True) 恒真 ══════════════

def test_metrics_is_always_full_regardless_of_compute_diagnostics(world):
    """裁决 ④：`compute_diagnostics` 不进 `param_hash` 的前提是它**只能新增**三块诊断。
    接到 `full=` 上，`full=False` 会删掉换手 / 成本拖累 / D6 缺口 ——
    同一个 D7 指纹映到两套 metrics 键集。"""
    off = run_backtest(make_cfg())
    n = len(world["rec"]["metrics"])
    on = run_backtest(_diag_cfg())

    assert all(c["kwargs"]["full"] is True for c in world["rec"]["metrics"])
    assert set(off.metrics) == set(on.metrics)
    assert {"turnover_annual", "cost_drag_annual", "d6_slippage_max"} <= set(off.metrics)
    assert off.param_hash == on.param_hash          # 两次运行共用一个指纹，所以数必须一样
    assert off.metrics == on.metrics
    assert len(world["rec"]["metrics"]) == n + 1


# ══════════════ 结构 ══════════════

def test_engine_is_orchestration_sized():
    src = pathlib.Path(engine.__file__).read_text(encoding="utf-8").splitlines()
    assert len(src) <= 400, f"engine.py {len(src)} 行 > 400（架构 A5：编排层）"


def test_engine_and_store_do_not_import_duckdb():
    """L1：落库只能经 `ashare/data/**`。闸是粗粒度的 —— 能 import duckdb 就能
    `connect('market.duckdb')` 然后 SELECT 未经掩码的原始行（D2 要挡的正是这个）。"""
    import ast

    for mod in (engine, bt_store):
        tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "duckdb" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "duckdb"


# ══════════════ store：save / load / run_id ══════════════

def test_run_id_carries_both_halves_of_the_d7_fingerprint(world):
    res = run_backtest(make_cfg())
    run_id = bt_store.make_run_id(res)
    assert res.param_hash in run_id
    assert res.data_snapshot_id in run_id
    assert res.started_at.strftime("%Y%m%dT%H%M%S") in run_id


@pytest.fixture
def saved_env(tmp_path, monkeypatch):
    """派生库路径是相对的 —— chdir 到 tmp，`backtest_run` 表才落在 tmp 里。"""
    pytest.importorskip("duckdb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bt_store, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def test_save_load_round_trips_every_field(world, saved_env):
    """裁决 ⑤：标量在 `backtest_run` 表、帧在 sidecar，**任一半漏一个字段都要红**。

    `ic` / `layers` / `attribution` / `warnings` 都有默认值 —— 少存一个不会抛，
    只会安静地读回 None，所以这条用例逐字段比。
    """
    _wide(world)
    res = run_backtest(make_cfg(compute_diagnostics=True))
    run_id = bt_store.make_run_id(res)
    bt_store.save(res, run_id)
    back = bt_store.load(run_id)

    for f in ("param_hash", "data_snapshot_id", "engine_version", "started_at",
              "elapsed_sec", "config", "warnings"):
        assert getattr(back, f) == getattr(res, f), f
    # metrics 里有 NaN（MDD=0 时的 Calmar），`nan == nan` 是 False —— 逐键比
    assert set(back.metrics) == set(res.metrics)
    for k, v in res.metrics.items():
        got = back.metrics[k]
        assert got == v or (isinstance(v, float) and np.isnan(v) and np.isnan(got)), k
    pd.testing.assert_series_equal(back.equity, res.equity)
    for f in ("positions", "trades", "blocked", "ic", "layers", "attribution"):
        assert getattr(back, f) is not None, f
        pd.testing.assert_frame_equal(getattr(back, f), getattr(res, f))


def test_the_backtest_run_table_actually_gets_a_row(world, saved_env):
    """schema 里躺着一张没有写入方的表，比没有这张表更糟：「按 run_id 查历史运行」
    看起来可用，实际永远查不到。"""
    from ashare.data import derived_store

    res = run_backtest(make_cfg())
    run_id = bt_store.make_run_id(res)
    bt_store.save(res, run_id)
    row = derived_store.read_backtest_run(run_id)

    assert row is not None
    assert (row["param_hash"], row["data_snapshot_id"]) == (res.param_hash,
                                                            res.data_snapshot_id)
    assert row["engine_version"] == res.engine_version
    assert row["started_at"] == res.started_at
    assert row["elapsed_sec"] == pytest.approx(res.elapsed_sec, rel=1e-12)
    assert res.param_hash in row["config_json"] or str(res.config.start) in row["config_json"]
    assert "sharpe" in row["metrics_json"]
    # `is_oos` 与 `append_oos_run` 共用 `OOS_CUTOFF`：两处各写一个「样本外」的定义，
    # 台账与库表就会各说各话。
    assert res.config.end > bt_store.OOS_CUTOFF and row["is_oos"] is True


def test_write_backtest_run_refuses_a_row_missing_a_d7_fingerprint(saved_env):
    """`param_hash` / `data_snapshot_id` 撞 NOT NULL，其余缺键会在写入深处炸出一个
    没有上下文的错误 —— 两种都比「在调用点说清楚缺了什么」难查。"""
    from ashare.data import derived_store

    row = {c: "x" for c in derived_store.BACKTEST_RUN_COLUMNS}
    del row["data_snapshot_id"]
    with pytest.raises(ValueError, match="data_snapshot_id"):
        derived_store.write_backtest_run(row)


def test_saving_the_same_run_id_twice_replaces_the_row(world, saved_env):
    """`run_id` 里含 `started_at`，撞主键只可能是同一次运行重存 —— DO NOTHING 会让
    第二次 save 静默无效。"""
    from ashare.data import derived_store

    res = run_backtest(make_cfg())
    run_id = bt_store.make_run_id(res)
    bt_store.save(res, run_id)
    res.metrics = {**res.metrics, "sharpe": 42.0}
    bt_store.save(res, run_id)

    assert "42.0" in derived_store.read_backtest_run(run_id)["metrics_json"]
    assert bt_store.load(run_id).metrics["sharpe"] == 42.0


def test_load_refuses_a_half_written_run(world, saved_env):
    """两半各存各的 —— 只剩一半时必须抛，不能拿默认值把另一半补出来。"""
    res = run_backtest(make_cfg())
    run_id = bt_store.make_run_id(res)
    bt_store.save(res, run_id)
    (saved_env / "runs" / f"{run_id}.pkl").unlink()

    with pytest.raises(FileNotFoundError):
        bt_store.load(run_id)
    with pytest.raises(FileNotFoundError):
        bt_store.load(run_id + "-never-saved")


def test_load_refuses_a_run_whose_table_row_is_gone(world, saved_env):
    """另一半：sidecar 还在、`backtest_run` 行没了。拿默认值把 param_hash 补成空串，
    读回来的就是一次【无法溯源】的运行 —— D7 的两个指纹正是从这一行来的。"""
    from ashare.data import _db

    res = run_backtest(make_cfg())
    run_id = bt_store.make_run_id(res)
    bt_store.save(res, run_id)
    (saved_env / _db.DEFAULT_DERIVED_PATH).unlink()

    assert (saved_env / "runs" / f"{run_id}.pkl").exists()
    with pytest.raises(FileNotFoundError):
        bt_store.load(run_id)


def test_load_refuses_a_run_id_that_walks_out_of_the_runs_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(bt_store, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(ValueError, match="run_id"):
        bt_store.load("../../etc/passwd")      # 验在碰库之前 —— 否则先拼一次 SQL 再拒


# ══════════════ store：样本外台账（D7）══════════════

def _oos_result(world, monkeypatch, tmp_path, end, **kw):
    monkeypatch.setattr(bt_store, "OOS_LOG_PATH", tmp_path / "oos-runs.md")
    monkeypatch.setattr(engine, "append_oos_run", bt_store.append_oos_run)
    return run_backtest(make_cfg(end=end, **kw))


def test_in_sample_runs_do_not_touch_the_oos_ledger(world, monkeypatch, tmp_path):
    monkeypatch.setattr(bt_store, "OOS_LOG_PATH", tmp_path / "oos-runs.md")
    res = run_backtest(make_cfg())
    res.config = BacktestConfig(**{**res.config.__dict__, "end": dt.date(2018, 6, 29)})
    appended, warns = bt_store.append_oos_run(res)
    assert appended is False
    assert warns and "样本外" in warns[0]
    assert not (tmp_path / "oos-runs.md").exists()


def test_an_out_of_sample_run_is_logged_with_both_fingerprints_and_the_engine_version(
        world, monkeypatch, tmp_path):
    res = _oos_result(world, monkeypatch, tmp_path, DAYS[-1])
    text = (tmp_path / "oos-runs.md").read_text(encoding="utf-8")
    assert res.param_hash in text
    assert res.data_snapshot_id in text
    assert res.engine_version in text
    assert "未跑" in text                      # 五闸都没跑，必须列进备注


def test_a_second_run_on_the_same_fingerprint_is_flagged_as_contamination(
        world, monkeypatch, tmp_path):
    _oos_result(world, monkeypatch, tmp_path, DAYS[-1])
    res2 = run_backtest(make_cfg())            # 同参数 + 同快照的第二次样本外运行
    appended, warns = bt_store.append_oos_run(res2)
    assert appended is True                    # 照记 —— 台账不能因为难看就漏记
    assert any("重复指纹" in w for w in warns)
    text = (tmp_path / "oos-runs.md").read_text(encoding="utf-8")
    assert text.count(res2.param_hash) == 3    # 首次 + 第二次运行自动记的 + 显式这次
    assert "⚠ 重复指纹（D7 污染）" in text
    # 引擎自己也把这条污染警告汇进了结果（不靠调用方另外去查台账）
    assert any("重复指纹" in w for w in res2.warnings)


def test_shuffle_controls_stay_out_of_the_ledger(world, monkeypatch, tmp_path):
    """闸 3 要跑 200 次置换，每次一行会把真正的样本外记录埋掉。"""
    monkeypatch.setattr(bt_store, "OOS_LOG_PATH", tmp_path / "oos-runs.md")
    res = run_backtest(make_cfg(shuffle_seed=7))
    appended, warns = bt_store.append_oos_run(res)
    assert appended is False
    assert any("shuffle" in w or "置换" in w for w in warns)


def test_shuffle_seed_actually_permutes_the_cross_section(world):
    """`shuffle_seed` 进了 `param_hash`。它若是个空转的开关，闸 3 的 200 次对照
    就是 200 次真回测，`p` 恒为 1/201 —— 一道永远「通过」的闸。"""
    plain = run_backtest(make_cfg())
    n_plain = len(real_target_calls(world["rec"]))
    shuffled = run_backtest(make_cfg(shuffle_seed=20260822))
    got = real_target_calls(world["rec"])[n_plain:]

    same_multiset = 0
    permuted = 0
    for c in got:
        sc = c["scores"]
        assert sorted(sc.dropna().tolist()) == sorted(_SCORES[c] for c in CODES)
        same_multiset += 1
        if not np.allclose(sc.to_numpy(), [_SCORES[x] for x in sc.index]):
            permuted += 1
    assert same_multiset == len(got)
    assert permuted >= 1, "shuffle_seed 是个空转的开关"
    assert not plain.positions["target_weight"].equals(shuffled.positions["target_weight"])


def test_the_same_shuffle_seed_reproduces_the_same_run(world):
    a = run_backtest(make_cfg(shuffle_seed=20260822))
    b = run_backtest(make_cfg(shuffle_seed=20260822))
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.param_hash == b.param_hash


# ══════════════ 真库集成（brief 验收：promote 换库 → QueryError）══════════════

@pytest.fixture
def real_engine_env(market_db, tmp_path, monkeypatch):
    """把一个合成 alpha 因子注册进真注册表，对着 fixture 真库跑整段回测。"""
    duckdb = pytest.importorskip("duckdb")
    from ashare.data import query as real_query
    from ashare.factors import base

    monkeypatch.chdir(tmp_path)
    saved = dict(base.FACTOR_REGISTRY)
    base.FACTOR_REGISTRY.clear()

    @base.factor(name="_t13_rev", direction=-1, category="price", lookback_days=6,
                 neutralize=False, min_coverage=0.0)
    def _rev(as_of_date, universe):
        panel = real_query.get_price_panel(as_of_date, list(universe), "close", lookback=6)
        return (panel.iloc[-1] / panel.iloc[0] - 1.0).reindex(list(universe))

    real_query.open_db(market_db)
    yield market_db
    real_query.close_db()
    base.FACTOR_REGISTRY.clear()
    base.FACTOR_REGISTRY.update(saved)


def _real_cfg() -> BacktestConfig:
    return BacktestConfig(
        start=dt.date(2023, 12, 25), end=dt.date(2024, 2, 2),
        factors=(("_t13_rev", 1.0),),
        constraints=PortfolioConstraints(top_n=2, max_single=0.75, max_industry=1.0,
                                         max_turnover=0.55),
        initial_capital=1_337_711.0, compute_diagnostics=False)


def test_end_to_end_against_the_real_fixture_database(real_engine_env, monkeypatch):
    monkeypatch.setattr(engine, "append_oos_run", lambda r: (False, []))
    res = run_backtest(_real_cfg())
    assert len(res.equity) >= 3
    assert res.equity.iloc[0] == 1.0
    assert len(res.positions) > 0
    assert {"intended_weight", "target_weight", "filled_weight", "price_hfq"} \
        <= set(res.positions.columns)
    assert {"adv20", "range", "total_cost"} <= set(res.trades.columns)
    assert res.data_snapshot_id and len(res.data_snapshot_id) == 16


def test_swapping_the_database_mid_run_raises_because_the_snapshot_is_pinned(
        real_engine_env, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "append_oos_run", lambda r: (False, []))
    shadow = str(tmp_path / "shadow.duckdb")
    shutil.copy(real_engine_env, shadow)

    def swap(i, n):
        if i == 1:
            os.replace(shadow, real_engine_env)      # promote：路径不变、inode 变

    with pytest.raises(QueryError):
        run_backtest(_real_cfg(), on_progress=swap)
