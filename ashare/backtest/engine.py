"""回测引擎 —— 唯一公开入口 `run_backtest`（架构 §4.3 / 规格 §5.3）。

本文件**只做编排**：把 query / factors / portfolio / execution / cost / metrics 接起来。
凡是一个读者会称之为「策略」的判断（选谁、给多少权重、能不能成交、收多少费、
指标怎么算）都不在这里 —— 架构 A5 说得很直白，靠把逻辑挪走来凑「核心 400 行」
是文字游戏，所以这里的口径是**可检查的**：本文件里找不到一条策略决策。

接缝上最容易撒谎的十来处，逐条钉在这里（每条都有对应用例）：

★ 1【两个 equity 不是一个】（架构 §4.3，2026-08-21 裁决）
  `simulate(equity=)` 是**货币**（它要做 `shares = Δw·equity/price`），
  `BacktestResult.equity` 是**初始 1.0 的净值指数**。`charge` 的 `total_cost` 跟前者。
  两者相除得到的是钱不是比例 —— 所以 `metrics.compute` 必须收 `initial_capital`。

★ 2【`prev_weights` 每期从持仓重算】
  `prev_w = 持仓股数 × T 收盘后复权价 / T 收盘权益`。把上期 `build_targets` 的返回值
  喂回来是最自然也最错的一步：那等于宣称账本一直贴在目标上、价格从没漂移过，
  换手因此系统性偏小，而换手直接进成本、成本直接进净值。

★ 3【`simulate` 收的是股数】
  权重在 τ 开盘要重算一次漂移（隔夜跳空），只有股数能让它自己算。改成传权重，
  跳空就被当成一笔交易做掉了。

★ 4【数据中断 ≠ 什么都不做】
  `build_targets` 返回 `None` 时仍要调 `simulate(targets=None)`：读作「按 τ 开盘持平，
  只执行强制退出」。跳过 `simulate` 会让中断日撞上的退市股永远留在账上（§5.5 幽灵资产），
  而那条「退市清仓」的 warning 照发不误 —— 声称了一件没发生的事。

★ 5【`scores` 必须是完整股票池 + NaN 占位】
  调用方先 `dropna()` 的话，`build_targets` 的 50% 覆盖率闸永远读到 100%，直接失明。

★ 6【`positions.intended_weight` = 换手裁剪之前的目标】
  没有它，净值曲线不可归因：分不清「信号不行」与「换手预算让信号表达不出来」，
  而对一个受换手约束的策略，这是完全相反的两个结论。
  取法是**再调一次 `build_targets`、只放开 `max_turnover`** —— 不在本文件里重算
  top-N + 行业上限（那是策略，且会与 portfolio.py 分叉）。

★ 7【`range` 取滞后 20 日平均振幅，不是执行日当天】（§5.4，2026-08-21 裁决）
  09:25 集合竞价成交时当天的最高最低价还没发生，用它是前视 —— 而且**可被利用**：
  `volatility_60` 的 direction 是 −1，组合系统性偏好低波动的票，低波动的票当日振幅也低，
  于是成本模型读到的正是因子在下注的那份未来。D9 占位行（`vol=0`）振幅为 0，
  混进均值会把成本算低，与 ADV20 已有的剔停牌规则同源，必须一起剔。

★ 8【`engine_version` 与 `param_hash` 并列，绝不进 hash】（§8 闸 1「分家」那一侧）
  塞进 hash 的后果是每次引擎升级都白送一次样本外机会 —— 正是 D7 要挡的污染，
  只是从另一扇门进来。同理 `run_backtest()` **不接受**调仓频率参数：周频是产品决策，
  当参数会让日频与周频两次运行共用一个指纹。真要可调，它得进 `BacktestConfig`。

★ 9【钉住快照，收尾复核】
  `snapshot_id(pin=True)` 之后 promote 换库会抛而不是静默重连；本文件因此**不得**
  中途 `open_db()`（钉住期间它同样抛，而它曾经是静默换库的后门）。
  ⚠ 钉子活得比本函数长，只有 `query.close_db()` 解得开 —— 与 `factors.store.build`
  同一个交接：长驻进程跑完要自己 close。

已知边界（都是有意的，不是漏做）：
  · `equity` 按**调仓频**（周）采样，不是架构 §4.3 写的日频 —— 逐日给账本估值要
    每天一次全池取价，而本引擎唯一需要价格的时点是 T 收盘与 τ 开盘。
    `metrics.compute` 因此收 `periods_per_year=52`（§9：年化一律按入参序列自己的频率）。
  · `ic` / `layers` / `attribution` 不产出（`compute_diagnostics=True` 时告警说明）：
    IC 面板要按因子逐列重算一遍，分层与归因要全池的持有期收益 —— 那是独立的一层，
    塞进编排层就是把 A5 的口径变成文字游戏。
  · 不建模部分成交、不做整手取整（见 `execution.simulate` 的「本模型的天花板」）。
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import re
import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import query
from ..factors.base import combine, get_factor
from . import metrics
from .cost import charge
from .execution import _DELIST_HAIRCUT, simulate
from .portfolio import build_targets
from .store import append_oos_run
from .types import BacktestConfig, BacktestResult

# 引擎语义变更时手工 bump。★ 绝不进 param_hash（模块头 ★8）。
ENGINE_VERSION = "p2-engine-1"

# 净值按调仓频采样（模块头「已知边界」），§9 的年化因子随之取 52。
_PERIODS_PER_YEAR = 52
# 冲击成本的回看窗口（§5.4：ADV20 与滞后振幅同一个 20 日窗）
_IMPACT_WINDOW = 20
# T 收盘取价的回看：停牌持仓要 ffill 到最后一个有效收盘（`build_targets` 不收 NaN）。
# 60 个交易日足够覆盖一个持有期内的停牌；仍取不到就让 `simulate` 的无价守卫去炸。
_PRICE_LOOKBACK = 60
_PRELOAD_TABLES = ("daily_bar", "daily_basic")

POSITION_COLS = ["score", "target_weight", "intended_weight", "filled_weight",
                 "shares", "price_hfq", "industry"]

# `factors.base.combine` 的逐因子剔除只在这句 warning 里露面（返回值是合成后的一列分数）。
# 它是「这一期活了几个因子」唯一的通道 —— 措辞改了就会静默退回「全都在用」，
# 而那是乐观的那一侧。`test_engine.py` 有一条用例拿真 `combine` 钉住这个正则。
_DROPPED_RE = re.compile(r"合成剔除 (\d+)/(\d+) 个因子")


def _factors_used(n_configured: int, warns: Sequence[str]) -> int:
    for w in warns:
        m = _DROPPED_RE.search(w)
        if m:
            return int(m.group(2)) - int(m.group(1))
    return n_configured


def _preload_start(cfg: BacktestConfig) -> _dt.date:
    """`start` 往前退 max(lookback_days) 个交易日。**这是 `lookback_days` 唯一的消费者**。

    多退是安全的（缓存大一点），少退不是：滚动窗口少一天，因子静默变形。
    """
    lb = max(get_factor(n).lookback_days for n, _ in cfg.factors)
    days = query.get_trade_dates(cfg.start)          # 日历上 ≤ start 的全部交易日
    if not days:
        return cfg.start
    return days[-lb] if len(days) >= lb else days[0]


def _tclose(as_of: _dt.date, codes: Sequence[str]) -> pd.Series:
    """T 日收盘后复权价。停牌日 `get_bars` 给 NaN，这里 ffill 到最后一个有效收盘。

    ffill 不是美化：`build_targets` 明确拒收 NaN 的 `prev_weights`（「敞口算不出来」
    ≠「没有敞口」），停牌持仓的价必须由调用方定，定不出来才该炸。
    """
    codes = list(codes)
    if not codes:
        return pd.Series(dtype=float)
    panel = query.get_price_panel(as_of, codes, "close", lookback=_PRICE_LOOKBACK)
    if not len(panel):
        return pd.Series(float("nan"), index=codes, dtype=float)
    return panel.ffill().iloc[-1].reindex(codes).astype(float)


def _value_at_open(exec_date: _dt.date, holdings: pd.Series, cash: float) -> float:
    """τ 开盘时点的**货币**权益 = 现金 + 持仓市值（模块头 ★1）。

    退市股按 `close_hfq × _DELIST_HAIRCUT` 估值 —— 与 `simulate` 的成交价**同一个数**
    （常量从那边导入，不抄字面量）：用开盘价估会先虚增权益、再在成交时莫名亏掉一笔。
    ponytail: 与 `simulate` 各查一次同一份掩码。合并要么把估值搬进 execution、
    要么让 simulate 多返回一列，两者都比省一次查询贵。
    取不到价的持仓在 `.sum()` 里被跳过 —— 故意的：`simulate` 随后会用同一份掩码
    报出那一只的代码与 reason，比这里抛一个「equity=nan」有用得多。
    """
    if not len(holdings):
        return cash
    m = query.get_tradable_mask(exec_date, list(holdings.index)).reindex(holdings.index)
    px = m["open_hfq"].astype(float).where(
        m["reason"].astype(str) != "delisted", m["close_hfq"].astype(float) * _DELIST_HAIRCUT)
    return cash + float((holdings * px).sum())


def _with_impact(signal_date: _dt.date, trades: pd.DataFrame) -> pd.DataFrame:
    """给成交表附 `adv20` / `range`（`cost.charge` 缺列直接抛，模块头 ★7）。

    两列同窗口、同剔除规则：信号日往前 20 个交易日，**剔掉 D9 占位行**
    （`is_suspended` 或 `vol == 0` —— 源给出 vol=0 的行同样算停牌）。
    占位行的振幅恒为 0、成交额恒为 0，留着会同时低估成本和低估流动性。
    """
    out = trades.copy()
    codes = sorted(set(out["ts_code"])) if len(out) else []
    if not codes:
        out["adv20"] = pd.Series(dtype=float)
        out["range"] = pd.Series(dtype=float)
        return out
    bars = query.get_bars(signal_date, codes, lookback=_IMPACT_WINDOW,
                          fields=("high", "low", "pre_close", "vol", "amount"))
    live = bars[(~bars["is_suspended"].astype(bool)) & (bars["vol"].astype(float) > 0)]
    adv = live["amount"].astype(float).groupby(level="ts_code").mean()
    amp = ((live["high"].astype(float) - live["low"].astype(float))
           / live["pre_close"].astype(float)).groupby(level="ts_code").mean()
    out["adv20"] = out["ts_code"].map(adv).astype(float)
    out["range"] = out["ts_code"].map(amp).astype(float)
    return out


def _benchmark(cfg: BacktestConfig, index: pd.Index) -> Optional[pd.Series]:
    n = len(query.get_trade_dates(cfg.end, start=cfg.start, freq="D"))
    bars = query.get_index_bars(cfg.end, cfg.benchmark, lookback=max(n, 2), fields=("close",))
    if not len(bars):
        return None
    # 不 ffill：对不齐的日子留 NaN，由 `metrics.compute` 出声。补出来的基准点会让
    # 超额收益凭空少一段波动，而那正是信息比率的分母。
    return bars["close"].astype(float).reindex(index)


def run_backtest(config: BacktestConfig,
                 *, on_progress: Optional[Callable[[int, int], None]] = None
                 ) -> BacktestResult:
    """跑一次回测。**没有调仓频率入参**（模块头 ★8）：周频是产品决策不是旋钮。

    Args:
        config: 策略口径。`param_hash()` 是它的 D7 指纹。
        on_progress: `(已完成期数, 总期数)`，每个调仓日调一次（含被跳过的空池日）。

    Returns:
        `BacktestResult`。`equity` 是**净值指数**（初始 1.0，按调仓频采样），
        `positions` / `trades` / `blocked` 是逐期明细，`warnings` 汇了全链路的降级。

    Raises:
        `query.QueryError`: 运行途中数据库被换掉（快照钉住）或首尾快照不一致。
        `ValueError`: `macro_timing=True`（宏观择时层属 P3，还不存在）。
    """
    cfg = config
    started_at = _dt.datetime.now()
    t0 = time.perf_counter()
    if cfg.macro_timing:
        raise ValueError("macro_timing=True 需要 P3 的宏观择时层（§7.1），本期未实现。"
                         "静默退化成满仓 = param_hash 写着择时而跑的是恒定仓位（D7 台账失真）")
    if not cfg.factors:
        raise ValueError("config.factors 为空：没有因子就没有合成分数")
    weights = dict(cfg.factors)
    pi = float(cfg.position_cap)            # macro_timing=False → 恒定满仓（§7.1）
    rng = None if cfg.shuffle_seed is None else np.random.default_rng(cfg.shuffle_seed)
    # 反事实账本：只放开换手预算，其余约束逐字段不动（模块头 ★6）
    unclipped = _dc.replace(cfg.constraints, max_turnover=float("inf"))

    # ★ 先钉住，再取任何数（模块头 ★9）。钉住之后本函数不再 open_db。
    snapshot = query.snapshot_id(pin=True)
    query.preload(_preload_start(cfg), cfg.end, _PRELOAD_TABLES)
    dates = query.get_trade_dates(cfg.end, start=cfg.start, freq="W")

    cash = float(cfg.initial_capital)
    holdings = pd.Series(dtype=float, name="shares").rename_axis("ts_code")
    warns: list = []
    eq_idx: list = list(dates[:1])
    eq_val: list = [1.0] if dates else []
    pos_frames: list = []
    trade_frames: list = []
    blocked_frames: list = []
    used: dict = {}

    for i, t in enumerate(dates, 1):
        exec_date = query.next_trade_date(t)
        if exec_date is None:
            warns.append(f"{t}: 日历末端没有下一个交易日，回测在此结束"
                         f"（D6：T 日收盘算信号、T+1 开盘成交）")
            break
        universe = query.get_universe(t)
        if not universe:
            # `compute_factor` 对空池是**抛**的（问一个空横截面要因子是调用方的 bug）。
            # 一天坏数据不该炸掉十五年，所以这个可预期的情形由引擎显式接住并出声。
            warns.append(f"{t}: 股票池为空，跳过该调仓日（当日不调仓、不成交）")
            if on_progress is not None:
                on_progress(i, len(dates))
            continue

        scores, w = combine(weights, t, universe)       # index = 完整池，算不出的留 NaN（★5）
        warns += w
        used[t] = _factors_used(len(weights), w)
        if rng is not None:                             # 闸 3：同日横截面内置换，不跨时间
            scores = pd.Series(rng.permutation(scores.to_numpy()),
                               index=scores.index, name="score")
        industry = query.get_industry(t, universe)

        held = list(holdings.index)
        px_t = _tclose(t, held)
        equity_t = cash + float((holdings * px_t).sum())
        prev_w = (holdings * px_t / equity_t).rename(None) if held else pd.Series(dtype=float)

        targets, w2 = build_targets(scores, pi, prev_w, industry, cfg.constraints)
        warns += w2
        # 反事实的 warning **不汇入**：它与上面那次逐条重复，只少了换手那几条。
        intended, _ = build_targets(scores, pi, prev_w, industry, unclipped)

        equity_open = _value_at_open(exec_date, holdings, cash)
        trades, new_holdings, blocked, w3 = simulate(
            exec_date, targets, holdings, equity_open, signal_date=t)
        warns += w3
        priced, w4 = charge(_with_impact(t, trades), cfg.cost)
        warns += w4

        total_cost = float(priced["total_cost"].sum()) if len(priced) else 0.0
        if len(priced):
            side = priced["side"].astype(str)
            amt = priced["amount"].astype(float)
            cash += float(amt[side == "SELL"].sum()) - float(amt[side == "BUY"].sum())
        cash -= total_cost
        # 同价成交是价值中性的，费用是唯一的漏损 —— 与 `cash + Σ 持仓 × 同一价格` 恒等。
        eq_idx.append(exec_date)
        eq_val.append((equity_open - total_cost) / float(cfg.initial_capital))

        # ★ 意图账本也要进 idx：被换手预算挡下的票**不在**交出的账本里，只按 targets
        #   建 index 会把 `intended_weight` 恰好在它唯一有意义的那几行上截掉 ——
        #   于是「换手约束拖累」永远算成 0，而那正是这一列存在的理由。
        idx = pd.Index(prev_w.index).union(new_holdings.index)
        if targets is not None:
            idx = idx.union(targets.index).union(intended.index)
        px_all = px_t.reindex(idx)
        gap = [c for c in idx if c not in px_t.index]
        if gap:
            px_all.loc[gap] = _tclose(t, gap)
        shares = new_holdings.reindex(idx).fillna(0.0)
        filled = shares * px_all / equity_t
        pos_frames.append(pd.DataFrame({
            "score": scores.reindex(idx).to_numpy(),
            # 中断日没有目标账本：意图就是「持平」，交出的也是持平，缺口因此确实是 0。
            # 这一天为什么不调仓，由 `build_targets` 的「数据中断」warning 说明。
            "target_weight": (filled if targets is None else
                              targets.reindex(idx).fillna(0.0)).to_numpy(),
            "intended_weight": (filled if targets is None else
                                intended.reindex(idx).fillna(0.0)).to_numpy(),
            "filled_weight": filled.to_numpy(),
            "shares": shares.to_numpy(),
            "price_hfq": px_all.to_numpy(),          # T 收盘（与换手上限同口径）
            "industry": industry.reindex(idx).to_numpy(),
        }, index=pd.MultiIndex.from_product([[t], list(idx)],
                                            names=["rebalance_date", "ts_code"])))
        trade_frames.append(priced)
        blocked_frames.append(blocked)
        holdings = new_holdings
        if on_progress is not None:
            on_progress(i, len(dates))

    # ★ 收尾复核（模块头 ★9）：钉子挡的是换文件，这一句还挡「同一个文件被原地改写」。
    if query.snapshot_id() != snapshot:
        raise query.QueryError(
            f"首尾数据快照不一致（{snapshot} → {query.snapshot_id()}）：本次运行横跨两份"
            f"数据，只记一个 data_snapshot_id 就是 D7 失效 —— 请重跑")

    equity = pd.Series(eq_val, index=pd.Index(eq_idx, name="trade_date"),
                       dtype=float, name="equity")
    positions = (pd.concat(pos_frames) if pos_frames else
                 pd.DataFrame(columns=POSITION_COLS,
                              index=pd.MultiIndex.from_arrays(
                                  [[], []], names=["rebalance_date", "ts_code"])))
    trades = (pd.concat(trade_frames, ignore_index=True) if trade_frames
              else pd.DataFrame(columns=["exec_date", "ts_code", "side", "shares",
                                         "price_hfq", "amount"]))
    blocked = (pd.concat(blocked_frames, ignore_index=True) if blocked_frames
               else pd.DataFrame(columns=["exec_date", "ts_code", "intended_side",
                                          "intended_weight", "reason"]))

    # ★ `full` **不**接 `compute_diagnostics`，尽管两处文档都把它俩指向「8s 档」。
    #   `BacktestConfig.param_hash` 把 `compute_diagnostics` 排除在指纹之外，合法性条件
    #   写得很死：它「只能【新增】ic/layers/attribution，不得改动 metrics 里的任何一个数」。
    #   `full=False` 会让 metrics 少掉换手 / 成本拖累 / D6 缺口几项 —— 那就是两次结果不同的
    #   运行共用一个 D7 指纹。真正吃时间的是逐日算因子，不是这几行 groupby。
    m, w5 = metrics.compute(
        equity, trades, positions, _benchmark(cfg, equity.index),
        full=True, initial_capital=cfg.initial_capital,
        periods_per_year=_PERIODS_PER_YEAR,
        factors_used=pd.Series(used, dtype=float) if used else None,
        # ★ 分母是【配置了几个】，不是观测最大值：[2,2,2,2] 的 max 就是 2，
        #   于是「14 个因子每期都只活 2 个」这种齐步降级一句告警都没有。
        n_factors_configured=len(cfg.factors))
    warns += w5
    if cfg.compute_diagnostics:
        warns.append("compute_diagnostics=True，但本引擎版本不产出 ic / layers / attribution"
                     "（见模块头「已知边界」）—— 三者为 None，不是「诊断算出来是空的」")

    result = BacktestResult(
        config=cfg, param_hash=cfg.param_hash(), data_snapshot_id=snapshot,
        engine_version=ENGINE_VERSION, started_at=started_at,
        elapsed_sec=time.perf_counter() - t0,
        equity=equity, positions=positions, trades=trades, blocked=blocked,
        metrics=m, warnings=warns)
    # D7 台账由代码写（U6：人工记录必然漏记，漏记即失效）。是否真写、为什么不写，
    # 都从 warning 通道回来 —— 「这次没记」本身就是要看得见的降级。
    _, w6 = append_oos_run(result)
    result.warnings += w6
    return result
