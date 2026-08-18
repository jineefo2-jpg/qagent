"""涨跌停价规则兜底（架构师 B2）。

Tushare stk_limit 有积分门槛，拿不到时用规则算。D6 直接依赖这个函数。

★ 核心原则：算不出的一律返回 (None, None, 'unknown')，由上层判为【不可交易】。
  宁可少一次成交机会，绝不能假设可交易 —— 后者会在回测里凭空生成
  现实中不存在的成交，是单向乐观偏差。

★ 舍入必须是十进制四舍五入（交易所规则），不能用 round()：
  round() 是二进制浮点 + 银行家舍入，约 4% 的 pre_close 会差 0.01，
  且绝大多数偏向把涨跌停带算宽 → 真实一字板被判成可交易 → 幽灵成交。
"""
from __future__ import annotations
import datetime as _dt
from decimal import Decimal, ROUND_HALF_UP

CHINEXT_20PCT_FROM = _dt.date(2020, 8, 24)     # 创业板涨跌幅由 10% 改 20%（同日引入 ST 制度）
STAR_OPEN_DATE = _dt.date(2019, 7, 22)         # 科创板开板
NEW_LISTING_GRACE_DAYS = 20                    # 自然日；须覆盖"最长假期 + 5 个交易日"（春节/国庆前上市的新股）

_CENT = Decimal("0.01")


def _round_half_up(x: float, pct: float) -> float:
    return float((Decimal(str(x)) * (Decimal("1") + Decimal(str(pct)))).quantize(_CENT, ROUND_HALF_UP))


def _board(ts_code: str) -> str:
    code, _, ex = ts_code.partition(".")
    if ex == "BJ" or code.startswith(("43", "83", "87", "88")):
        return "BSE"
    if code.startswith("68"):                  # 688xxx 科创板 + 689xxx 科创板 CDR
        return "STAR"
    if code.startswith("30"):                  # 300xxx / 301xxx / 302xxx 均为创业板
        return "CHINEXT"
    return "MAIN"


def compute_limits(ts_code: str, trade_date: _dt.date, pre_close: float | None,
                   list_date: _dt.date | None, status: str) -> tuple[float | None, float | None, str]:
    """返回 (limit_up, limit_down, source)。source ∈ {'rule', 'unknown'}。"""
    # not (x > 0) 同时挡住 None 之外的 NaN：NaN > 0 为 False，而 NaN <= 0 也是 False（那条守卫会漏）
    if pre_close is None or not (pre_close > 0):
        return None, None, "unknown"

    # 退市整理期：涨跌幅规则历经多次变更，且流动性枯竭，一律不交易
    if status == "DELIST_PERIOD":
        return None, None, "unknown"

    # 上市初期：主板首日无涨跌幅、科创创业前 5 日无限制、北交所首日无限制。
    # list_date 未知时无法排除"正处于上市初期"，同样保守处理。
    if list_date is None or (trade_date - list_date).days < NEW_LISTING_GRACE_DAYS:
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

    # ST / *ST 5% 是【主板】规则。科创板 / 注册制后创业板 / 北交所的风险警示股
    # 沿用本板涨跌幅（20% / 20% / 30%），交易所特别规定无 ST 例外。
    # 注册制前创业板不存在 ST 制度，无需分支。
    if status in ("ST", "*ST") and board == "MAIN":
        pct = 0.05

    return _round_half_up(pre_close, pct), _round_half_up(pre_close, -pct), "rule"
