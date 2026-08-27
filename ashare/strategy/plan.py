"""调仓清单生成（规格 §6.3 / P3 Task 3）—— 整个系统的终点产物。

本模块零 DML（分层 L4）：落库与导出在 CLI（Task 4）经 ashare.data.ledger_store。
生成期间 `snapshot_id(pin=True)` 钉住（Global Constraints）；钉子活得比本函数长，
CLI 收尾要自己 `query.close_db()`（与引擎 ★9 同一条契约）。

三个口径决定（都写进 JSON，让读者不用猜）：
  · `current_weight` 基准 = **持仓市值占比（不含现金）** —— 系统不掌握用户现金余额
    （A 股不接券商 API 是 N2 的设计取舍），估值价用 τ 日掩码的 `pre_close_raw`
    （= T 日原始收盘，与限价带同一次查询）。
  · 换手预算的 `prev_weights` = 现状市值占比 × 上一份清单的 `target_position`
    （没有上一份就用本期 π）—— 把「不含现金」的书面权重折回全权益口径的近似。
  · 限价带 = `pre_close_raw × (1 ± 0.5 × ATR20)`，再夹进当日涨跌停（超出涨跌停的
    限价单交易所直接拒单，夹是可执行性不是口径修改）；ATR20 是**振幅比率**的 20 日
    均值（(high−low)/pre_close 对复权不变，剔停牌与 vol=0 占位行 —— 引擎 ★7 同源）。
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, List, Optional

import pandas as pd

from ashare.backtest.portfolio import build_targets
from ashare.data import ledger_store, query
from ashare.factors.base import compute_panel, get_factor
from ashare.factors.base import combine as _combine
from .macro import macro_score

STRATEGY_VERSION = "p3-strategy-1"
AS_OF_NOTE = "T 日收盘后计算"
EXECUTE_NOTE = "T+1 09:15-09:25 集合竞价阶段挂限价单，09:25 统一按开盘价撮合"
UNCALIBRATED_WARNING = ("⚠ 持仓未校准：上一期清单之后没有对账/确认记录，"
                        "current_weight 按已有台账推演，可靠性下降（规格 §6.4）")
_BAND_K = 0.5                    # 限价带半宽 = 0.5 × ATR20（规格 §6.3 price_basis 原文）
_CONTRIB_TOP = 3


def _atr20_ratio(as_of: _dt.date, codes: List[str]) -> pd.Series:
    """20 日均振幅比率（复权不变量）。剔停牌/vol=0 占位行（引擎 ★7 的同一条剔除规则）。"""
    if not codes:
        return pd.Series(dtype=float)
    bars = query.get_bars(as_of, codes, lookback=20, fields=("high", "low", "pre_close", "vol"))
    live = bars[(~bars["is_suspended"].astype(bool)) & (bars["vol"].astype(float) > 0)]
    return ((live["high"].astype(float) - live["low"].astype(float))
            / live["pre_close"].astype(float)).groupby(level="ts_code").mean()


def _band(pc: float, atr: float, lim_dn: float, lim_up: float) -> Optional[List[float]]:
    """[前收×(1−k·ATR), 前收×(1+k·ATR)] ∩ [跌停, 涨停]，tick 取整到分。算不出返回 None。"""
    if not (pc and pc == pc and pc > 0) or not (atr == atr):
        return None
    lo, hi = pc * (1 - _BAND_K * atr), pc * (1 + _BAND_K * atr)
    if lim_dn == lim_dn:
        lo = max(lo, lim_dn)
    if lim_up == lim_up:
        hi = min(hi, lim_up)
    return [round(lo, 2), round(hi, 2)]


def _contrib(panel: pd.DataFrame, weights: Dict[str, float], code: str) -> Dict[str, float]:
    """|w·d·z| 前 3 的因子及其带符号贡献（与 combine 的合成项同构）。"""
    if code not in panel.index:
        return {}
    row = panel.loc[code]
    items = []
    for n, w in weights.items():
        z = row.get(n)
        if z is None or (isinstance(z, float) and math.isnan(z)):
            continue
        items.append((n, float(w) * get_factor(n).direction * float(z)))
    items.sort(key=lambda kv: -abs(kv[1]))
    return {n: round(v, 4) for n, v in items[:_CONTRIB_TOP]}


def build_rebalance_plan(as_of_date, config) -> dict:
    """§6.3 契约 + `data_snapshot_id`（架构 §4.4）。数据中断（build_targets 返回 None）
    直接抛 —— 一份基于中断数据的清单比没有清单更坏。"""
    as_of = query.norm_date(as_of_date)
    snapshot = query.snapshot_id(pin=True)
    tau = query.next_trade_date(as_of)
    if tau is None:
        raise ValueError(f"{as_of} 之后没有交易日历覆盖：定不出执行日（先跑 daily 增量补日历）")

    weights = dict(config.factors)
    universe = query.get_universe(as_of)
    scores, warns = _combine(weights, as_of, universe, use_store=True)

    macro: Optional[dict] = None
    if config.macro_timing:
        macro = macro_score(as_of)
        pi = float(config.position_floor) + (float(config.position_cap)
                                             - float(config.position_floor)) * macro["score"]
    else:
        pi = float(config.position_cap)

    # ── 现状持仓（口径见模块头）──
    led_date, led_rows = ledger_store.latest_positions()
    prior = ledger_store.latest_signal_plan()
    calibrated = True
    if prior is not None:
        calibrated = led_date is not None and led_date >= query.norm_date(prior["as_of"])
    held = {r["ts_code"]: float(r["shares"]) for r in led_rows}

    mask = query.get_tradable_mask(tau, sorted(set(universe) | set(held)))
    cur_w: Dict[str, float] = {}
    if held:
        vals = {c: sh * float(mask.loc[c, "pre_close_raw"]) for c, sh in held.items()
                if c in mask.index and mask.loc[c, "pre_close_raw"] == mask.loc[c, "pre_close_raw"]}
        total = sum(vals.values())
        if total > 0:
            cur_w = {c: v / total for c, v in vals.items()}

    prior_pi = float(prior["target_position"]) if prior else pi
    prev_w = pd.Series({c: w * prior_pi for c, w in cur_w.items()}, dtype=float)
    industry = query.get_industry(as_of)
    targets, intended, w_bt = build_targets(scores, pi, prev_w, industry, config.constraints)
    warns += w_bt
    if targets is None:
        raise ValueError(f"{as_of} 组合构建中断（{'; '.join(w_bt) or '原因见 warnings'}）——"
                         f"基于中断数据的清单比没有清单更坏，本期不出清单")

    # ── 逐单组装 ──
    active = sorted(set(targets[targets > 0].index) | set(cur_w))
    atr = _atr20_ratio(as_of, active)
    names = query.get_stock_basic(as_of, active)["name"] if active else pd.Series(dtype=object)
    panel, wp = compute_panel(list(weights), as_of, universe, use_store=True)
    warns += wp

    orders: list = []
    excluded: list = []
    for code in active:
        tgt = float(targets.get(code, 0.0))
        cur = float(cur_w.get(code, 0.0))
        action = "BUY" if cur == 0 else ("SELL" if tgt == 0 else "ADJUST")
        m = mask.loc[code] if code in mask.index else None
        if action == "BUY" and m is not None and not bool(m["can_buy"]):
            excluded.append({"ts_code": code, "reason": str(m["reason"]),
                             "note": "τ 日不可买（§6.2：剔除预期一字涨停等不可成交标的）"})
            continue
        urgency, note = "normal", None
        if action != "BUY" and m is not None and not bool(m["can_sell"]):
            urgency, note = "blocked", f"τ 日不可卖（{m['reason']}），顺延执行"
        band = None if m is None else _band(float(m["pre_close_raw"]), float(atr.get(code, float("nan"))),
                                            float(m["limit_down_raw"]), float(m["limit_up_raw"]))
        orders.append({"ts_code": code,
                       "name": str(names.get(code, "?")),
                       "action": action,
                       "current_weight": round(cur, 6),
                       "target_weight": round(tgt, 6),
                       "limit_price_range": band,
                       "price_basis": "前收 ± 0.5 × ATR20，夹进当日涨跌停",
                       "factor_contrib": _contrib(panel, weights, code),
                       "urgency": urgency, **({"note": note} if note else {})})

    plan_warns = ([] if calibrated else [UNCALIBRATED_WARNING]) + [w for w in warns if w.lstrip().startswith("⚠")]
    return {"as_of": str(as_of), "as_of_note": AS_OF_NOTE,
            "execute_on": f"{tau}T09:15:00+08:00", "execute_note": EXECUTE_NOTE,
            "target_position": round(pi, 4),
            "macro_score": None if macro is None else macro["scores"],
            "position_calibrated": calibrated,
            "current_weight_basis": "持仓市值占比（不含现金；系统不掌握现金余额）",
            "orders": orders, "excluded": excluded,
            "warnings": plan_warns,
            "strategy_version": STRATEGY_VERSION,
            "param_hash": config.param_hash(), "data_snapshot_id": snapshot}
