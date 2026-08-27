"""宏观择时层（规格 §6.1 / P3 Task 1）：5 个 PIT 指标 → 滚动 5 年分位 → 总仓位 π。

三条铁规矩（Global Constraints 原文）：
  · 禁 ML（规格 N5）；
  · 禁全样本分位数 —— 分位窗是【滚动 5 年 = 60 个月】，窗不足记 0.5 中性并打旗，不缩窗
    （缩窗分位在样本早期剧烈抖动）；
  · 一切取数经 ashare.data.query（PIT：宏观按 publish_date，行情按交易日当日可得）。

5 指标全部构造成「高 → 加仓」（规格 §6.1 的方向列），因此统一取原值分位，无需逐指标翻符号：
  erp           = 100 / PE_TTM(中证全指) − 10Y 国债收益率     （百分点口径：1/PE 换算成 %）
  m1_m2_gap     = M1 同比 − M2 同比
  tsf_yoy_chg   = 社融存量同比的 3 个月变化（按月历对齐后差分 —— 缺月得 NaN，不许错位）
  north_flow_60 = 60 交易日北向净流入 / 全市场流通市值（V3：Σ₆₀ Δ(hold×circ) 望远镜式
                  收缩成 N(t) − N(t−60)，全市场加总里缺持仓记录 = 未持有 = 0 ——
                  这是【加总】不是逐股因子，B5 的「不填 0」不适用于它）
  trend_ma200   = 中证全指 close / MA200

分位 → 分数（V2 裁决）：pct < 30% → 0；30% ≤ pct ≤ 70% → 0.5；pct > 70% → 1。
pct 定义 = 当期值在【含当期】的 60 个月窗内的 ≤ 秩占比。
π = 0.2 + 0.8 × mean(五项分数) —— 下限 20% 防空仓错过 V 型反弹（规格原文）。
"""
from __future__ import annotations

import calendar
import datetime as _dt
from typing import Dict, Optional

import pandas as pd

from ashare.data import query

INDICATORS = ("erp", "m1_m2_gap", "tsf_yoy_chg", "north_flow_60", "trend_ma200")
_WINDOW = 60                     # 滚动 5 年 = 60 个月
_LO, _HI = 0.3, 0.7              # V2：30/70 切点，中档覆盖 40% 历史状态
_NORTH_FROM = _dt.date(2016, 12, 5)      # 北向持股数据起点（与 pipeline.HK_HOLD_FROM 同值）
_DAILY_LOOKBACK = 4400           # 覆盖 2008 起的日频序列（ERP/国债/趋势的原料）


def _month_end(d: _dt.date) -> _dt.date:
    return _dt.date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def _to_month_last(s: pd.Series) -> pd.Series:
    """日频 → 月频：每个日历月取最后一个观测，标签统一为日历月末（与宏观 period 对齐）。"""
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    keys = [_month_end(d) for d in s.index]
    return pd.Series(s.values, index=keys, dtype=float).groupby(level=0).last()


def _grid_shift(s: pd.Series, months: int) -> pd.Series:
    """按【月历】取 months 个月前的值（不是位置 shift：源缺一个月，位置差分会整体错位，
    与 get_financial_ttm「按日历取上年同期」同一条教训）。"""
    idx = [_month_end(_dt.date(d.year + (d.month - 1 - months) // 12,
                               (d.month - 1 - months) % 12 + 1, 1)) for d in s.index]
    return pd.Series(s.reindex(idx).to_numpy(), index=s.index, dtype=float)


def _north_flow_monthly(as_of: _dt.date) -> pd.Series:
    """每个月末交易日 t 的 (N(t) − N(t−60)) / C(t)。N = Σ hold_ratio% × circ_mv，
    C = Σ circ_mv。逐月末 3 次横截面查询，一次全历史调用约数秒 —— 周频信号可担。"""
    tds = query.get_trade_dates(as_of)
    if not tds:
        return pd.Series(dtype=float)
    ends = _to_month_last(pd.Series(range(len(tds)), index=tds))     # 月末交易日的【位置】
    codes = list(query.get_stock_basic(as_of).index)

    def _cross(d: _dt.date) -> "tuple[float, float]":
        ratio = (query.get_money_flow(d, codes, ("hk_hold_ratio",), lookback=1)["hk_hold_ratio"]
                 .droplevel("trade_date").astype(float))
        circ = query.get_daily_basic(d, codes, ("circ_mv",))["circ_mv"].astype(float)
        n = float((ratio.reindex(circ.index).fillna(0.0) / 100.0 * circ).sum())
        return n, float(circ.sum())

    out: dict = {}
    for label, pos in ends.items():
        pos = int(pos)
        t = tds[pos]
        if pos < 60 or t < _NORTH_FROM or tds[pos - 60] < _NORTH_FROM:
            continue                                   # 数据未开始 / 回看窗踩到起点之前
        n_t, c_t = _cross(t)
        n_p, _ = _cross(tds[pos - 60])
        if c_t > 0:
            out[label] = (n_t - n_p) / c_t
    return pd.Series(out, dtype=float).sort_index()


def macro_indicators(as_of_date) -> pd.DataFrame:
    """5 指标的月频原值面板：index = 日历月末，columns = INDICATORS。
    各指标独立取自己 PIT 可见的序列，缺的月份留 NaN（展示与打分都按列自理，不互相填补）。"""
    as_of = query.norm_date(as_of_date)
    mac = query.get_macro(as_of, ["m1_yoy", "m2_yoy", "tsf_stock_yoy"], lookback_periods=240)
    m1m2 = (mac["m1_yoy"] - mac["m2_yoy"]).dropna().astype(float)
    m1m2.index = [_month_end(d) for d in m1m2.index]
    tsf = mac["tsf_stock_yoy"].dropna().astype(float)
    tsf.index = [_month_end(d) for d in tsf.index]
    tsf_chg = (tsf - _grid_shift(tsf, 3)).dropna()

    cn10 = query.get_macro(as_of, ["cn10y"], lookback_periods=_DAILY_LOOKBACK)["cn10y"]
    bars = query.get_index_bars(as_of, "000985.CSI", lookback=_DAILY_LOOKBACK,
                                fields=("close", "pe_ttm"))
    close = bars["close"].astype(float)
    pe_m = _to_month_last(bars["pe_ttm"].astype(float))
    erp = (100.0 / pe_m - _to_month_last(cn10.astype(float))).dropna()
    trend = _to_month_last((close / close.rolling(200).mean()).dropna())

    return pd.DataFrame({"erp": erp, "m1_m2_gap": m1m2, "tsf_yoy_chg": tsf_chg,
                         "north_flow_60": _north_flow_monthly(as_of),
                         "trend_ma200": trend}).sort_index()


def _pct_in_window(s: pd.Series) -> Optional[float]:
    """当期值在含当期的 60 个月窗内的 ≤ 秩占比；窗不足返回 None（调用方记 0.5 打旗）。"""
    s = s.dropna()
    if len(s) < _WINDOW:
        return None
    w = s.iloc[-_WINDOW:]
    return float((w <= w.iloc[-1]).mean())


def _box(pct: float) -> float:
    """V2：<30% → 0；30–70%（含端点）→ 0.5；>70% → 1。"""
    return 0.0 if pct < _LO else (0.5 if pct <= _HI else 1.0)


def position_for(as_of_date, *, floor: float = 0.2, cap: float = 1.0) -> "tuple[float, list]":
    """引擎接口（P3 Task 2）：π = floor + (cap − floor) × score。默认参数即规格 §6.1 的
    `20% + 80% × score`；floor/cap 来自 BacktestConfig（两者都在 param_hash 里）。
    返回 (π, window_short) —— 旗子由引擎跨期汇总成一条告警，不逐期刷屏。"""
    got = macro_score(as_of_date)
    return float(floor) + (float(cap) - float(floor)) * got["score"], got["window_short"]


def macro_score(as_of_date) -> Dict:
    """{"scores": {指标: 0|0.5|1}, "score": 均值, "position": 0.2+0.8×score,
    "window_short": [窗不足记 0.5 的指标], "percentiles": {指标: 分位或 None}}"""
    df = macro_indicators(as_of_date)
    scores: dict = {}
    pcts: dict = {}
    short: list = []
    for name in INDICATORS:
        pct = _pct_in_window(df[name]) if name in df.columns else None
        pcts[name] = pct
        if pct is None:
            scores[name] = 0.5
            short.append(name)
        else:
            scores[name] = _box(pct)
    score = sum(scores.values()) / len(INDICATORS)
    return {"as_of": str(query.norm_date(as_of_date)), "scores": scores,
            "percentiles": pcts, "score": score,
            "position": 0.2 + 0.8 * score, "window_short": short}
