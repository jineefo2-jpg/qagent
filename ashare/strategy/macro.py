"""宏观择时层（规格 §6.1 / P3 Task 1；Task 8 增量缓存重构，公开 API 与语义不变）。

三条铁规矩（Global Constraints 原文）：禁 ML（N5）；禁全样本分位 —— 滚动 5 年 = 60 个月，
窗不足记 0.5 中性打旗不缩窗；一切取数经 ashare.data.query（PIT）。

5 指标全部构造成「高 → 加仓」，统一取原值分位（切点 30/70，V2）：
  erp / m1_m2_gap / tsf_yoy_chg / north_flow_60 / trend_ma200（定义见各构造函数）。

── 增量缓存（2026-08-27 用户裁决：不许逐调用全量重建）──
老路径每次 macro_score 重建全部历史（北向 160 个月末 × 3 次横截面查询），回测里
511 个调仓日 × 秒级 = 小时级瓶颈。现在：
  · 原料按【快照】缓存一次（月频宏观带 __publish_date、日频序列、北向日度聚合 ——
    后者下沉为 query.get_north_aggregate 的一条 SQL）；
  · 每个 as_of 只做【可见性切片】：月频按 publish_date ≤ as_of（PIT 不破），
    日频按 trade_date ≤ as_of；毫秒级；
  · 失效键 = query.snapshot_id()（约 1ms）：每日增量 promote 后自动重建一次。
已知近似（写给读者，不是藏着）：缓存存的是【建缓存时点】各期的最新值；某期在
as_of 与建缓存日之间发生数值修订时，切片值 = 修订后值而非 as_of 当时值。
月频宏观修订罕见且幅度小；要逐版本精确需 query 暴露多版本接口，本期不做。
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
_NORTH_FROM = _dt.date(2016, 12, 5)      # 北向数据起点（与 pipeline.HK_HOLD_FROM 同值）
_DAILY_LOOKBACK = 4400           # 覆盖 2008 起的日频序列

_CACHE: dict = {}                # {"snap","built_to","mac","cn10","close","pe","north"}
_SCORE_MEMO: dict = {}           # 单条：{"key": (as_of, snap), "val": dict}


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
    """按【月历】取 months 个月前的值（不是位置 shift：源缺一个月，位置差分会整体错位）。"""
    idx = [_month_end(_dt.date(d.year + (d.month - 1 - months) // 12,
                               (d.month - 1 - months) % 12 + 1, 1)) for d in s.index]
    return pd.Series(s.reindex(idx).to_numpy(), index=s.index, dtype=float)


def _ensure_cache(as_of: _dt.date) -> str:
    """原料缓存：快照一致且覆盖 as_of 即复用；否则以 as_of 为终点重建一次。返回快照。"""
    snap = query.snapshot_id()
    if (_CACHE.get("snap") == snap and _CACHE.get("built_to") is not None
            and as_of <= _CACHE["built_to"]):
        return snap
    mac = query.get_macro(as_of, ["m1_yoy", "m2_yoy", "tsf_stock_yoy"], lookback_periods=240)
    cn10 = query.get_macro(as_of, ["cn10y"], lookback_periods=_DAILY_LOOKBACK)["cn10y"].dropna()
    bars = query.get_index_bars(as_of, "000985.CSI", lookback=_DAILY_LOOKBACK,
                                fields=("close", "pe_ttm"))
    north = query.get_north_aggregate(as_of)
    _CACHE.update(snap=snap, built_to=as_of, mac=mac,
                  cn10=cn10.astype(float),
                  close=bars["close"].astype(float),
                  pe=bars["pe_ttm"].astype(float),
                  north=north)
    return snap


def _visible_monthly(col: str, as_of: _dt.date) -> pd.Series:
    """月频列按 publish_date ≤ as_of 的可见性切片，index 归一到日历月末。"""
    mac = _CACHE["mac"]
    pub = mac[f"{col}__publish_date"]
    vis = mac[col][[p is not None and not pd.isna(p) and p <= as_of for p in pub]].dropna()
    vis.index = [_month_end(d) for d in vis.index]
    return vis.astype(float)


def _north_flow_monthly(as_of: _dt.date) -> pd.Series:
    """每个月末交易日 t 的 (N(t) − N(t−60)) / C(t)（V3 望远镜口径），全部来自缓存的
    日度聚合 —— 零查询。缺持仓记录 = 未持有 = 0（加总语义，非逐股因子）。"""
    n = _CACHE["north"]
    n = n[n.index <= as_of]
    if n.empty:
        return pd.Series(dtype=float)
    pos = _to_month_last(pd.Series(range(len(n)), index=n.index))       # 月末交易日的位置
    days = list(n.index)
    out: dict = {}
    for label, p in pos.items():
        p = int(p)
        if p < 60 or days[p] < _NORTH_FROM or days[p - 60] < _NORTH_FROM:
            continue
        c_t = float(n["circ_mv_total"].iloc[p])
        if c_t > 0:
            out[label] = (float(n["north_mv"].iloc[p]) - float(n["north_mv"].iloc[p - 60])) / c_t
    return pd.Series(out, dtype=float).sort_index()


def macro_indicators(as_of_date) -> pd.DataFrame:
    """5 指标的月频原值面板：index = 日历月末，columns = INDICATORS。
    各指标独立取自己 PIT 可见的序列，缺的月份留 NaN（展示与打分都按列自理）。"""
    as_of = query.norm_date(as_of_date)
    _ensure_cache(as_of)

    m1 = _visible_monthly("m1_yoy", as_of)
    m2 = _visible_monthly("m2_yoy", as_of)
    m1m2 = (m1 - m2).dropna()
    tsf = _visible_monthly("tsf_stock_yoy", as_of)
    tsf_chg = (tsf - _grid_shift(tsf, 3)).dropna()

    cn10 = _CACHE["cn10"];  cn10 = cn10[cn10.index <= as_of]
    close = _CACHE["close"]; close = close[close.index <= as_of]
    pe = _CACHE["pe"];       pe = pe[pe.index <= as_of]
    erp = (100.0 / _to_month_last(pe) - _to_month_last(cn10)).dropna()   # 百分点口径
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
    as_of = query.norm_date(as_of_date)
    snap = _ensure_cache(as_of)
    if _SCORE_MEMO.get("key") == (as_of, snap):
        return dict(_SCORE_MEMO["val"])
    df = macro_indicators(as_of)
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
    out = {"as_of": str(as_of), "scores": scores, "percentiles": pcts, "score": score,
           "position": 0.2 + 0.8 * score, "window_short": short}
    _SCORE_MEMO["key"], _SCORE_MEMO["val"] = (as_of, snap), dict(out)
    return out
