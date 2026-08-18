# ashare/data/limits.py
"""涨跌停价规则兜底（架构师 B2）。

Tushare stk_limit 有积分门槛，拿不到时用规则算。D6 直接依赖这个函数。

★ 核心原则：算不出的一律返回 (None, None, 'unknown')，由上层判为【不可交易】。
  宁可少一次成交机会，绝不能假设可交易 —— 后者会在回测里凭空生成
  现实中不存在的成交，是单向乐观偏差。
"""
from __future__ import annotations
import datetime as _dt

CHINEXT_20PCT_FROM = _dt.date(2020, 8, 24)     # 创业板涨跌幅由 10% 改 20%
STAR_OPEN_DATE = _dt.date(2019, 7, 22)         # 科创板开板
NEW_LISTING_GRACE_DAYS = 10                    # 上市初期规则复杂，一律 unknown


def _board(ts_code: str) -> str:
    code, _, ex = ts_code.partition(".")
    if ex == "BJ" or code.startswith(("43", "83", "87", "88")):
        return "BSE"
    if code.startswith("688"):
        return "STAR"
    if code.startswith("300"):
        return "CHINEXT"
    return "MAIN"


def compute_limits(ts_code: str, trade_date: _dt.date, pre_close: float | None,
                   list_date: _dt.date | None, status: str) -> tuple[float | None, float | None, str]:
    """返回 (limit_up, limit_down, source)。source ∈ {'rule', 'unknown'}。"""
    if pre_close is None or pre_close <= 0:
        return None, None, "unknown"

    # 退市整理期：涨跌幅规则历经多次变更，且流动性枯竭，一律不交易
    if status == "DELIST_PERIOD":
        return None, None, "unknown"

    # 上市初期：主板首日无涨跌幅、科创创业前 5 日无限制、北交所首日无限制
    if list_date is not None and (trade_date - list_date).days < NEW_LISTING_GRACE_DAYS:
        return None, None, "unknown"

    board = _board(ts_code)
    if board == "BSE":
        pct = 0.30
    elif board == "STAR":
        if trade_date < STAR_OPEN_DATE:
            return None, None, "unknown"
        pct = 0.20
    elif board == "CHINEXT":
        pct = 0.20 if trade_date >= CHINEXT_20PCT_FROM else 0.10
    else:
        pct = 0.10

    # ST 一律 5%（各板块统一），且优先级高于板块规则
    if status in ("ST", "*ST"):
        pct = 0.05

    # A 股涨跌停价按四舍五入到 0.01 元
    return round(pre_close * (1 + pct), 2), round(pre_close * (1 - pct), 2), "rule"
