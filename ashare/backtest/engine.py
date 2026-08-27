"""回测引擎 —— 唯一公开入口 `run_backtest`（架构 §4.3 / 规格 §5.3）。

本文件**只做编排**：把 query / factors / portfolio / execution / cost / metrics 接起来。
凡是一个读者会称之为「策略」的判断（选谁、给多少权重、能不能成交、收多少费、指标怎么算）
都不在这里 —— 架构 A5 的口径是可检查的：本文件里找不到一条策略决策。

接缝上最容易撒谎的十一处（每条都有用例；★1/★6/★10/★11 的完整论证在架构 §4.3 的裁决框里）：

★ 1【两个 equity 不是一个】`simulate(equity=)` 是**货币**（它要做 `shares = Δw·equity/price`），
  `BacktestResult.equity` 是**初始 1.0 的净值指数**；`charge` 的 `total_cost` 跟前者，两者
  相除得到的是钱不是比例 —— 所以 `metrics.compute` 必须收 `initial_capital`。

★ 2【`prev_weights` 每期从持仓重算】`prev_w = 持仓股数 × T 收盘后复权价 / T 收盘权益`。喂
  回上期 `build_targets` 的返回值等于宣称账本一直贴在目标上、价格从没漂移过，换手因此
  系统性偏小 —— 而换手直接进成本、成本直接进净值。

★ 3【`simulate` 收的是股数】权重在 τ 开盘要重算一次漂移（隔夜跳空），只有股数能让它自己
  算 —— 改成传权重，跳空就被当成一笔交易做掉了。

★ 4【数据中断 ≠ 什么都不做】`build_targets` 给 `None` 时仍要调 `simulate(targets=None)`
  （「τ 开盘持平 + 只做强制退出」）。跳过它，中断日撞上的退市股就永远留在账上（§5.5 幽灵
  资产），而「退市清仓」那条 warning 照发不误 —— 声称了一件没发生的事。

★ 5【`scores` 必须是完整股票池 + NaN 占位】调用方先 `dropna()` 的话，`build_targets`
  的 50% 覆盖率闸永远读到 100%，直接失明。

★ 6【`positions.intended_weight` = 换手裁剪之前的目标】没有它，净值曲线分不清「信号不行」
  与「换手预算让信号表达不出来」—— 对受换手约束的策略这是相反的两个结论。它由
  `build_targets` 的**第二个返回值**直接给出（裁决 ③）：本文件既不重算 top-N + 行业上限
  （那是策略，会与 portfolio.py 分叉），也不用 `max_turnover=inf` 造反事实（丢掉那次的
  warnings 就是降级不可见）。

★ 7【`range` 取滞后 20 日平均振幅，不是执行日当天】（§5.4）09:25 成交时当天的最高最低价
  还没发生，用它是前视，而且**可被利用**：`volatility_60` 方向为 −1，组合系统性偏好低波动
  的票，它们当日振幅也低 —— 成本模型于是读到因子正在下注的那份未来。D9 占位行
  （`vol=0`）振幅恒 0，混进均值把成本算低，与 ADV20 的剔停牌规则同源。

★ 8【`engine_version` 与 `param_hash` 并列，绝不进 hash】（§8 闸 1「分家」那一侧）塞进
  hash 等于每次引擎升级白送一次样本外机会。同理 `run_backtest()` **不接受**调仓频率参数：
  周频是产品决策，当参数会让日频与周频两次运行共用一个指纹。

★ 9【钉住快照，收尾复核】`snapshot_id(pin=True)` 之后 promote 换库会抛而不是静默重连，
  本文件因此**不得**中途 `open_db()`（它曾经是静默换库的后门）。⚠ 钉子活得比本函数长，
  只有 `query.close_db()` 解得开 —— 长驻进程跑完要自己 close。

★ 10【`equity` 是日频盯市，不是调仓频采样】（裁决 ①）按调仓频采样**看不见周内的低点**，
  于是低估最大回撤 —— 而 MDD 是 Calmar 的分母，偏差朝着「好看」那一侧。
  ⚠ 这条曲线**不回流任何决策**：权重仍在 T 收盘度量、成交仍在 τ 开盘，纯报告口径。

★ 11【`compute_diagnostics=True` 就必须真的产出三块】（裁决 ②）尤其 `attribution`：§3.2 选
  OLS 而非 Barra 的 √MV-WLS，那条裁决**只能靠它被证伪** —— 不产出而只发一条 warning，
  等于把可检验的断言降格成空话。持有期收益按 §5.1 取两个执行日的开盘价。

已知边界（有意的）：不建模部分成交、不做整手取整（见 `execution.simulate`）。
"""
from __future__ import annotations

import datetime as _dt
import re
import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import query
from ..factors import store as factor_store
from ..factors.base import combine, compute_factor, compute_panel, get_factor
from . import metrics
from .cost import charge
from .execution import _DELIST_HAIRCUT, BLOCKED_COLS, TRADE_COLS, simulate
from .portfolio import build_targets
from .store import append_oos_run
from .types import BacktestConfig, BacktestResult

# 引擎语义变更时手工 bump。★ 绝不进 param_hash（模块头 ★8）。
ENGINE_VERSION = "p2-engine-1"

# 净值日频故年化 252（★10）；IC / 分层是调仓频，年化留在 `metrics` 那边各自的 52 上
# （§9：一律按入参序列自己的频率 —— 两个频率不同的序列不能共用一个年化因子）。
_PERIODS_PER_YEAR = 252
# 风格归因的规模回归元，取**原始值**：必须与 `pipeline.neutralize` 减掉的是同一个定义，
# 两份各自演化的实现会让 §3.2「OLS 而非 WLS」的裁决永远无法证伪。
_SIZE_FACTOR = "log_mv"
_IMPACT_WINDOW = 20      # 冲击成本的回看窗（§5.4：ADV20 与滞后振幅同一个 20 日窗）
_PRICE_LOOKBACK, _PRICE_LOOKBACK_DEEP = 60, 1600   # 60 日 ffill 常规停牌；深窗兜 2015 潮的数百日长停
_PRELOAD_TABLES = ("daily_bar", "daily_basic")

POSITION_COLS = ["score", "target_weight", "intended_weight", "filled_weight", "shares",
                 "price_hfq", "industry"]

# `combine` 的逐因子剔除只在这句 warning 里露面（返回值是合成后的一列分数），它是
# 「这一期活了几个因子」唯一的通道 —— 措辞一改就静默退回「全都在用」，而那是乐观的
# 那一侧。`test_engine.py` 有一条用例拿真 `combine` 钉住这个正则。
_DROPPED_RE = re.compile(r"合成剔除 (\d+)/(\d+) 个因子")


def _factors_used(n_configured: int, warns: Sequence[str]) -> int:
    m = next((x for x in map(_DROPPED_RE.search, warns) if x), None)
    return n_configured if m is None else int(m.group(2)) - int(m.group(1))


def _preload_start(cfg: BacktestConfig) -> _dt.date:
    """`start` 往前退 max(lookback_days) 个交易日（`lookback_days` 唯一的消费者）。
    多退只是缓存大一点，少退是滚动窗口少一天 —— 因子静默变形。"""
    lb = max(get_factor(n).lookback_days for n, _ in cfg.factors)
    days = query.get_trade_dates(cfg.start)          # 日历上 ≤ start 的全部交易日
    return days[max(0, len(days) - lb)] if days else cfg.start


def _tclose(as_of: _dt.date, codes: Sequence[str]) -> pd.Series:
    """T 日收盘后复权价。停牌日 `get_bars` 给 NaN，这里 ffill 到最后一个有效收盘 ——
    `build_targets` 明确拒收 NaN 的 `prev_weights`（「敞口算不出来」≠「没有敞口」）。
    仍缺者（长停牌，真库验收 600827 死在 2015）升级 `_PRICE_LOOKBACK_DEEP` 再查。"""
    out = pd.Series(float("nan"), index=pd.Index(list(codes)), dtype=float)
    for lb in (_PRICE_LOOKBACK, _PRICE_LOOKBACK_DEEP):
        miss = out.index[out.isna()].tolist()
        if miss and len(p := query.get_price_panel(as_of, miss, "close", lookback=lb)):
            out.update(p.ffill().iloc[-1].reindex(miss).astype(float))
    return out


def _value_at_open(exec_date: _dt.date, holdings: pd.Series, cash: float) -> float:
    """τ 开盘时点的**货币**权益 = 现金 + 持仓市值（模块头 ★1）。

    退市股按 `close_hfq × _DELIST_HAIRCUT` 估值 —— 与 `simulate` 的成交价**同一个数**
    （常量从那边导入，不抄字面量）：用开盘价估会先虚增权益、再在成交时莫名亏掉一笔。
    取不到价的持仓在 `.sum()` 里被跳过 —— 故意的：`simulate` 随后会用同一份掩码报出
    那一只的代码与 reason，比这里抛一个「equity=nan」有用得多。
    ponytail: 与 `simulate` 各查一次同一份掩码，合并比省这次查询贵。
    """
    if not len(holdings):
        return cash
    m = query.get_tradable_mask(exec_date, list(holdings.index)).reindex(holdings.index)
    px = m["open_hfq"].astype(float).where(
        m["reason"].astype(str) != "delisted", m["close_hfq"].astype(float) * _DELIST_HAIRCUT)
    return cash + float((holdings * px).sum())


def _with_impact(signal_date: _dt.date, trades: pd.DataFrame) -> pd.DataFrame:
    """给成交表附 `adv20` / `range`（`cost.charge` 缺列直接抛，模块头 ★7）。

    两列同窗口、同剔除规则：信号日往前 20 个交易日，**剔掉 D9 占位行**（`is_suspended`
    或 `vol == 0`，源给出 vol=0 的行同样算停牌）—— 占位行振幅恒 0、成交额恒 0，
    留着会同时低估成本与流动性。
    """
    out = trades.copy()
    codes = sorted(set(out["ts_code"])) if len(out) else []
    if not codes:
        return out.assign(adv20=pd.Series(dtype=float), range=pd.Series(dtype=float))
    bars = query.get_bars(signal_date, codes, lookback=_IMPACT_WINDOW,
                          fields=("high", "low", "pre_close", "vol", "amount"))
    live = bars[(~bars["is_suspended"].astype(bool)) & (bars["vol"].astype(float) > 0)]
    adv = live["amount"].astype(float).groupby(level="ts_code").mean()
    amp = ((live["high"].astype(float) - live["low"].astype(float))
           / live["pre_close"].astype(float)).groupby(level="ts_code").mean()
    out["adv20"] = out["ts_code"].map(adv).astype(float)
    out["range"] = out["ts_code"].map(amp).astype(float)
    return out


def _daily_equity(days: Sequence[_dt.date], marks: Sequence, capital: float) -> pd.Series:
    """逐交易日盯市：`(现金 + Σ 持仓 × 当日收盘后复权价) / 本金`（模块头 ★10）。

    `marks` = 每个执行日成交后的 `(exec_date, cash, holdings)`，升序。持仓与现金都是阶梯
    函数（只在执行日跳变），`ffill` 到每个交易日；建仓前全是现金 → 1.0。停牌日收盘为 NaN，
    同样 ffill 到最后一个有效收盘（与 `_tclose` 同一条理由）。
    ★ `fillna(0.0)` 必须在 `ffill` **之前**：某只票这期不在账本里读作「持有 0 股」，让那个空位
      去 ffill 会把上期股数带下去 —— 卖掉的票继续替净值曲线赚钱。"""
    idx = pd.Index(days, name="trade_date")
    if not len(idx) or not marks:
        return pd.Series(1.0, index=idx, dtype=float, name="equity")
    held = pd.DataFrame({d: h for d, _, h in marks}).T.fillna(0.0).reindex(idx).ffill().fillna(0.0)
    cash = pd.Series({d: c for d, c, _ in marks}).reindex(idx).ffill().fillna(capital)
    px = query.get_price_panel(idx[-1], list(held.columns), "close",
                               lookback=len(idx)).reindex(idx).ffill()
    return (((held * px).sum(axis=1) + cash) / capital).rename("equity")


def _open_px(exec_date: _dt.date, codes: Sequence[str]) -> pd.Series:
    """执行日的开盘后复权价（成交价口径），index = codes。"""
    p = query.get_price_panel(exec_date, list(codes), "open", lookback=1)
    return (p.iloc[-1] if len(p) else pd.Series(float("nan"), index=list(codes))).astype(float)


def _diagnose(periods: Sequence, positions: pd.DataFrame) -> tuple:
    """IC / 分层 / 归因（模块头 ★11）。返回 `(ic, layers, attribution, warnings)`。

    `periods` 逐期一条 `(调仓日, 执行日, 股票池, 合成分数, 因子面板, log_mv)`。
    持有期收益 = 下一执行日开盘 / 本执行日开盘 − 1（§5.1），所以**末期整期不进诊断**：
    它没有下一个执行日，留一行 NaN 会让 `ic_series` 报「横截面不足」——把样本尽头读成缺口。
    """
    names = ["rebalance_date", "ts_code"]
    pairs = list(zip(periods, periods[1:]))
    if not pairs:
        return None, None, None, ["可诊断的调仓期不足两个执行日，一期持有期收益都算不出："
                                  "三块诊断为 None —— 是「没算」，不是「算出来是空的」"]

    def stack(k: int) -> pd.Series:         # 逐期的第 k 项摞成 (调仓日, ts_code) 的长表
        return pd.concat({p[0]: p[k] for p, _ in pairs}, names=names)

    ret = pd.concat({p[0]: _open_px(n[1], p[2]) / _open_px(p[1], p[2]) - 1.0
                     for p, n in pairs}, names=names)
    scores = stack(3)
    # 归因按期数取平均，所以只喂有持有期收益的那几期；带上末期会把每一行都摊薄一档。
    pos = positions[positions.index.get_level_values(0).isin([p[0] for p, _ in pairs])]
    ic, w1 = metrics.ic_series(stack(4), ret)
    layers, w2 = metrics.layered_returns(scores, ret)
    attribution, w3 = metrics.attribution(pos, ret, scores, size=stack(5))
    return ic, layers, attribution, w1 + w2 + w3


def _benchmark(cfg: BacktestConfig, index: pd.Index) -> Optional[pd.Series]:
    if not len(index):
        return None
    bars = query.get_index_bars(index[-1], cfg.benchmark, lookback=max(len(index), 2),
                                fields=("close",))
    if not len(bars):
        return None
    # 不 ffill：对不齐的日子留 NaN，由 `metrics.compute` 出声。补出来的基准点会让
    # 超额收益凭空少一段波动，而那正是信息比率的分母。
    return bars["close"].astype(float).reindex(index)


def run_backtest(config: BacktestConfig, *, on_progress: Optional[Callable[[int, int], None]] = None,
                 use_store: bool = False) -> BacktestResult:
    """跑一次回测。**没有调仓频率入参**（模块头 ★8）：周频是产品决策不是旋钮。

    Args:
        config: 策略口径。`param_hash()` 是它的 D7 指纹。
        on_progress: 每期一次 `(已完成, 总)`；use_store: 缓存快路径（补裁 ①：kwarg 不进 config/指纹）。

    Returns:
        `BacktestResult`。`equity` 是**日频净值指数**（初始 1.0），其余是逐期明细，
        `warnings` 汇了全链路的降级。

    Raises:
        `query.QueryError`: 运行途中数据库被换掉（快照钉住）或首尾快照不一致。
    """
    cfg = config
    started_at, t0 = _dt.datetime.now(), time.perf_counter()
    if not cfg.factors:
        raise ValueError("config.factors 为空：没有因子就没有合成分数")
    weights = dict(cfg.factors)
    pi = float(cfg.position_cap)            # macro_timing=False → 恒定满仓（§7.1）
    macro_short: set = set()
    if cfg.macro_timing:                    # 惰性导入避环：strategy→backtest 方向的依赖已存在
        from ..strategy.macro import position_for
    rng = None if cfg.shuffle_seed is None else np.random.default_rng(cfg.shuffle_seed)

    # ★ 先钉住，再取任何数（模块头 ★9）。钉住之后本函数不再 open_db。
    snapshot = query.snapshot_id(pin=True)
    query.preload(_preload_start(cfg), cfg.end, _PRELOAD_TABLES)
    dates = query.get_trade_dates(cfg.end, start=cfg.start, freq="W")
    use_store and factor_store.preload_window({n: get_factor(n).param_hash() for n, _ in cfg.factors}, dates)

    cash = float(cfg.initial_capital)
    holdings = pd.Series(dtype=float, name="shares").rename_axis("ts_code")
    warns: list = []
    marks: list = []            # (exec_date, cash, holdings) —— 日频盯市的台阶（★10）
    diag: list = []             # compute_diagnostics=True 时逐期的诊断入参（★11）
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

        scores, w = combine(weights, t, universe, use_store=use_store)   # index = 完整池（★5）
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

        if cfg.macro_timing:                # π 变化照常消耗换手预算（P3 计划 V5：择时降仓就是卖出）
            pi, _ws = position_for(t, floor=cfg.position_floor, cap=cfg.position_cap); macro_short.update(_ws)
        targets, intended, w2 = build_targets(scores, pi, prev_w, industry, cfg.constraints)
        warns += w2

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
        marks.append((exec_date, cash, new_holdings))

        # `intended` 与 `targets` 共用一个 index（portfolio 保证）—— 被换手预算整只挡下的
        # 票因此不会从账本里消失，「换手约束拖累」那一行才量得到东西。
        idx = pd.Index(prev_w.index).union(new_holdings.index)
        if targets is not None:
            idx = idx.union(targets.index)
        px_all = _tclose(t, idx)
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
        if cfg.compute_diagnostics:
            # IC 面板要的是**处理后**的逐因子列（`combine` 只交出合成后的一列），
            # 所以这里按因子重算一遍；`log_mv` 取原始值 —— 与中性化减掉的同一个定义。
            panel, wp = compute_panel([n for n, _ in cfg.factors], t, universe, processed=True, use_store=use_store)
            size, ws = compute_factor(_SIZE_FACTOR, t, universe, processed=False)
            warns += wp + ws
            diag.append((t, exec_date, universe, scores, panel, size))
        if on_progress is not None:
            on_progress(i, len(dates))

    # ★ 收尾复核（模块头 ★9）：钉子挡的是换文件，这一句还挡「同一个文件被原地改写」。
    if query.snapshot_id() != snapshot:
        raise query.QueryError(
            f"首尾数据快照不一致（{snapshot} → {query.snapshot_id()}）：本次运行横跨两份"
            f"数据，只记一个 data_snapshot_id 就是 D7 失效 —— 请重跑")

    # 日频索引要盖住最后一个执行日：调仓日落在区间末尾时 τ = T+1 可能已越过 `end`（那一笔
    # 仍然成交），少了它这天的成本就「不在净值曲线上」，拖累一分都算不进去。
    last = max([cfg.end] + [m[0] for m in marks])
    days = query.get_trade_dates(last, start=dates[0]) if dates else []
    equity = _daily_equity(days, marks, float(cfg.initial_capital))
    empty_mi = pd.MultiIndex.from_arrays([[], []], names=["rebalance_date", "ts_code"])
    positions = (pd.concat(pos_frames) if pos_frames
                 else pd.DataFrame(columns=POSITION_COLS, index=empty_mi))
    trades = (pd.concat(trade_frames, ignore_index=True) if trade_frames
              else pd.DataFrame(columns=list(TRADE_COLS)))
    blocked = (pd.concat(blocked_frames, ignore_index=True) if blocked_frames
               else pd.DataFrame(columns=list(BLOCKED_COLS)))

    # ★ `full=True` **恒真，绝不接 `compute_diagnostics`**（§4.3 裁决 ④）。`param_hash`
    #   排除 `compute_diagnostics` 的合法性条件写得很死：它「只能【新增】
    #   ic/layers/attribution，不得改动 metrics 里的任何一个数」，而 `full=False` 会
    #   **删掉**换手 / 成本拖累 / D6 缺口 —— 同一个 D7 指纹映到两套 metrics 键集。
    #   `full=False` 只留给临时分析，不由任何配置字段驱动。
    m, w5 = metrics.compute(
        equity, trades, positions, _benchmark(cfg, equity.index),
        full=True, initial_capital=cfg.initial_capital,
        periods_per_year=_PERIODS_PER_YEAR,
        factors_used=pd.Series(used, dtype=float) if used else None,
        # ★ 分母是【配置了几个】不是观测最大值：[2,2,2,2] 的 max 就是 2，于是
        #   「14 个因子每期都只活 2 个」这种齐步降级一句告警都没有。
        n_factors_configured=len(cfg.factors))
    warns += w5

    ic = layers = attribution = None
    if cfg.compute_diagnostics:
        ic, layers, attribution, wd = _diagnose(diag, positions)
        warns += wd

    macro_short and warns.append(f"宏观择时：{sorted(macro_short)} 分位窗不足 5 年，按 0.5 中性参与打分")
    result = BacktestResult(
        config=cfg, param_hash=cfg.param_hash(), data_snapshot_id=snapshot,
        engine_version=ENGINE_VERSION, started_at=started_at,
        elapsed_sec=time.perf_counter() - t0,
        equity=equity, positions=positions, trades=trades, blocked=blocked,
        metrics=m, ic=ic, layers=layers, attribution=attribution, warnings=warns)
    # D7 台账由代码写（U6：人工记录必然漏记，漏记即失效）。是否真写、为什么不写，
    # 都从 warning 通道回来 —— 「这次没记」本身就是要看得见的降级。
    _, w6 = append_oos_run(result)
    result.warnings += w6
    return result
