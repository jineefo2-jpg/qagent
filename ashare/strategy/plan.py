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

import argparse
import datetime as _dt
import json
import math
import pathlib
from dataclasses import replace as _dc_replace
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ashare.backtest.portfolio import build_targets
from ashare.backtest.types import BacktestConfig, PortfolioConstraints
from ashare.data import ledger_store, query
from ashare.factors.base import ALPHA_CATEGORIES, compute_panel, get_factor, list_factors
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
    data_end = query.last_data_date()
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

    # ★ 实盘出清单时执行日 τ 在【未来】，那天没有任何行情：掩码若按 τ 求值会把整个股票池
    #   判成 no_quote 而全数剔除 —— 2026-08-28 实测 0 笔订单 / 50 只剔除，而报告读起来像
    #   「策略今天什么都不该买」。把「未知」当成「不可交易」是错的。
    #   改在 T 日求值，这正是 §6.2「剔除次日【预期】一字涨停」里「预期」二字的落点 ——
    #   T 日一字封死是次日一字最好的可得预测。历史回放（τ 已有行情）路径分毫不动。
    live = data_end is not None and tau > data_end
    mask = query.get_tradable_mask(as_of if live else tau, sorted(set(universe) | set(held)))
    px_col = "close_raw" if live else "pre_close_raw"      # 两者都是【τ 的前收】
    cur_w: Dict[str, float] = {}
    if held:
        vals = {c: sh * float(mask.loc[c, px_col]) for c, sh in held.items()
                if c in mask.index and mask.loc[c, px_col] == mask.loc[c, px_col]}
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
        # 实盘不夹涨跌停：τ 的涨跌停由 T 收盘推出，此刻按 T 自己的板去夹会夹错。
        # ATR 带（典型 ±1~2%）几乎不可能越过 ±10%，越过也由券商侧拒单，不至于成交在坏价。
        lo_lim, hi_lim = ((float("nan"), float("nan")) if live or m is None
                          else (float(m["limit_down_raw"]), float(m["limit_up_raw"])))
        band = None if m is None else _band(float(m[px_col]), float(atr.get(code, float("nan"))),
                                            lo_lim, hi_lim)
        orders.append({"ts_code": code,
                       "name": str(names.get(code, "?")),
                       "action": action,
                       "current_weight": round(cur, 6),
                       "target_weight": round(tgt, 6),
                       "limit_price_range": band,
                       "price_basis": ("前收 ± 0.5 × ATR20（执行日涨跌停尚未确定，未夹）" if live
                                       else "前收 ± 0.5 × ATR20，夹进当日涨跌停"),
                       "factor_contrib": _contrib(panel, weights, code),
                       "urgency": urgency, **({"note": note} if note else {})})

    plan_warns = ([] if calibrated else [UNCALIBRATED_WARNING]) + [w for w in warns if w.lstrip().startswith("⚠")]
    if live:
        plan_warns.append(f"执行日 {tau} 尚无行情：可交易性与限价带按 {as_of} 收盘推断"
                          f"（次日一字涨停/停牌属预测，非事实）")
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


# ══════════════ CLI（Task 4）—— 唯一写库点 ══════════════
_VALIDATION_WINDOW = (_dt.date(2010, 1, 1), _dt.date(2019, 12, 31))
# ★ 2026-08-28 样本外仪式的判定③ 把生产配置定死在「A 臂」：恒定仓位 0.8、不开宏观择时
#   （宏观层因 Sharpe 未超恒定仓位而关停，规格 §6.1）。这个哈希是**过了闸 1 的那次回测**
#   的指纹 —— 下面的 `default_config()` 必须逐位复现它，否则每晚的信号就来自一个
#   没有通过样本外验证的配置。测试 `test_default_config_is_the_validated_arm` 钉住。
#   D7 只给一次样本外机会且已用尽：改因子/约束/仓位 = 这个哈希对不上 = 信号失去背书，
#   那是需要人明确承认的事，不是可以顺手改掉的默认值。
VALIDATED_PARAM_HASH = "ab102248f1b54f7f"
PRODUCTION_POSITION_CAP = 0.8


def default_config(*, macro_timing: bool = False, top_n: Optional[int] = None,
                   position_cap: Optional[float] = None) -> BacktestConfig:
    """生产策略配置 = 样本外仪式的胜出臂（见 `VALIDATED_PARAM_HASH`）。

    ★ start/end 有意钉在 **P2 验收窗口（2010–2019）**：BacktestConfig 的 param_hash
    覆盖 start/end，钉死后清单的指纹与「验证过这套参数」的那次回测**同指纹** ——
    D7 的连续性从回测台账一路接到清单台账。改窗口 = 另一套策略 = 新指纹，这是特性。"""
    alphas = tuple((s.name, 1.0) for s in list_factors() if s.category in ALPHA_CATEGORIES)
    kw: dict = {"start": _VALIDATION_WINDOW[0], "end": _VALIDATION_WINDOW[1],
                "factors": alphas, "macro_timing": macro_timing,
                "position_cap": PRODUCTION_POSITION_CAP}
    if top_n is not None:
        kw["constraints"] = _dc_replace(PortfolioConstraints(), top_n=top_n)
    if position_cap is not None:
        kw["position_cap"] = position_cap
    return BacktestConfig(**kw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ashare.strategy.plan",
        description="生成 §6.3 调仓清单：落 ledger 库 + 导出 JSON。落库只发生在这里（L4）。")
    ap.add_argument("--as-of", default=None, help="信号日（T，收盘后），YYYY-MM-DD")
    ap.add_argument("--nightly", action="store_true",
                    help="定时链模式（21:30 增量之后调用）：非交易日/非周频调仓日安静跳过；"
                         "调仓日先 build 当日因子（V6：只 build 这一天）再出清单，build 失败不出清单")
    ap.add_argument("--macro-timing", action="store_true", help="启用宏观择时层（默认恒定 position_cap）")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--position-cap", type=float, default=None)
    ap.add_argument("--out", default="out/signals", help="JSON 导出目录")
    a = ap.parse_args(argv)

    cfg = default_config(macro_timing=a.macro_timing, top_n=a.top_n, position_cap=a.position_cap)
    query.open_db()
    try:
        as_of = a.as_of
        if a.nightly:
            today = _dt.date.today()
            tds = query.get_trade_dates(today)
            if not tds or tds[-1] != today:
                print(f"[plan] nightly：{today} 非交易日，跳过"); return 0
            if query.get_trade_dates(today, freq="W")[-1] != today:
                print(f"[plan] nightly：{today} 非周频调仓日，跳过"); return 0
            as_of = str(today)
            from ashare.factors import store as _fstore     # 惰性：仅 nightly 需要
            counts, bw = _fstore.build([n for n, _ in cfg.factors], [today])
            print(f"[plan] nightly：当日因子已落库 {sum(counts.values())} 行"
                  + (f"，{len(bw)} 条告警" if bw else ""))
        elif as_of is None:
            ap.error("--as-of 与 --nightly 必须给一个")
        plan = build_rebalance_plan(as_of, cfg)
        overwrote = ledger_store.save_signal_plan(plan)
        out_dir = pathlib.Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"{plan['as_of']}.json"
        f.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[plan] {plan['as_of']} → 执行 {plan['execute_on']}  π={plan['target_position']}")
        print(f"[plan] 订单 {len(plan['orders'])} 笔 / 剔除 {len(plan['excluded'])} 只"
              f" / 校准={'是' if plan['position_calibrated'] else '否'}")
        for w in plan["warnings"]:
            print(f"[plan] {w}")
        if overwrote:
            print(f"[plan] ⚠ 同 (as_of, param_hash) 已存在，幂等覆盖旧行（参数没变，这不是新实验）")
        print(f"[plan] 已落库 ledger + 导出 {f}")
        return 0
    finally:
        query.close_db()        # 解钉（引擎 ★9 同一条契约：钉子活得比 build 长，调用方收尾）


if __name__ == "__main__":
    raise SystemExit(main())
