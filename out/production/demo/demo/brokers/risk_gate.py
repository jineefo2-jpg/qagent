"""
预交易风控（pre-trade risk gate）。

调用时机：用户点"确认下单"后、向 broker.place_order 前。
任何一条规则不通过就阻断订单。

所有阈值可通过环境变量覆盖（不改代码即可调）：
    RISK_MAX_SINGLE_PCT       默认 0.20 （单笔上限占净值的比例）
    RISK_MAX_SINGLE_PCT_ETF   默认 0.50 （ETF 标的可放宽）
    RISK_DOUBLE_CONFIRM_PCT   默认 0.50 （超过此比例需要前端二次确认）
    RISK_MAX_DAILY_ORDERS     默认 20   （每日下单笔数上限）
    RISK_MAX_DAILY_CANCEL_RATE 默认 0.40（每日撤单率上限）
    RISK_BLOCK_MARKET_ORDER   默认 1    （是否禁用市价单）
    RISK_WHITELIST            默认见下方常量
"""
from __future__ import annotations

import os
import time
import datetime
from dataclasses import dataclass
from typing import List, Optional

from cache import cache
from .base import OrderIntent, OrderSide, OrderType, AccountInfo


# ════════════════════════════════════════════════════════════
# 配置（环境变量优先，否则用默认值）
# ════════════════════════════════════════════════════════════

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# 默认 ETF 列表（用来识别"可放宽到 50%"的标的）
DEFAULT_ETF_SET = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VEA", "VWO",
    "EEM", "GLD", "SLV", "USO", "TLT", "HYG", "LQD", "XLF",
    "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    "ARKK", "QLD", "TQQQ", "SQQQ", "UVXY",
}

# 默认白名单（个股 + ETF）
DEFAULT_WHITELIST = {
    # ETF
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
    # 七巨头
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    # 其他高流动性蓝筹
    "BRK.B", "JPM", "JNJ", "V", "MA", "UNH", "HD", "PG",
    "AMD", "NFLX", "AVGO", "ORCL", "CRM", "ADBE",
}


def _load_whitelist() -> set:
    """从环境变量 RISK_WHITELIST 加载，逗号分隔；空时用默认"""
    env = os.getenv("RISK_WHITELIST", "").strip()
    if not env:
        return set(DEFAULT_WHITELIST)
    return {s.strip().upper() for s in env.split(",") if s.strip()}


def _load_etf_set() -> set:
    env = os.getenv("RISK_ETF_LIST", "").strip()
    if not env:
        return set(DEFAULT_ETF_SET)
    return {s.strip().upper() for s in env.split(",") if s.strip()}


# ════════════════════════════════════════════════════════════
# 结果对象
# ════════════════════════════════════════════════════════════

@dataclass
class RiskCheckResult:
    passed: bool
    reasons: List[str]                    # 失败原因（passed=False 时填充）
    warnings: List[str]                   # 通过但有警告
    needs_double_confirm: bool = False    # 是否需要前端二次确认
    notional_pct: float = 0.0             # 本单占净值比例（便于前端展示）

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "needs_double_confirm": self.needs_double_confirm,
            "notional_pct": round(self.notional_pct, 4),
        }


# ════════════════════════════════════════════════════════════
# 日内订单统计（持久化到 cache）
# ════════════════════════════════════════════════════════════

def _today_key(device_id: str) -> str:
    today = datetime.date.today().strftime("%Y%m%d")
    return f"quant:risk:daily:{device_id}:{today}"


def _load_daily_stats(device_id: str) -> dict:
    d = cache.get(_today_key(device_id)) or {}
    d.setdefault("orders_placed", 0)
    d.setdefault("orders_canceled", 0)
    return d


def _save_daily_stats(device_id: str, stats: dict):
    cache.set(_today_key(device_id), stats, ttl=86400 * 2)


def record_order_placed(device_id: str):
    s = _load_daily_stats(device_id)
    s["orders_placed"] = int(s.get("orders_placed", 0)) + 1
    _save_daily_stats(device_id, s)


def record_order_canceled(device_id: str):
    s = _load_daily_stats(device_id)
    s["orders_canceled"] = int(s.get("orders_canceled", 0)) + 1
    _save_daily_stats(device_id, s)


def get_daily_stats(device_id: str) -> dict:
    return _load_daily_stats(device_id)


# ════════════════════════════════════════════════════════════
# 主入口：风控检查
# ════════════════════════════════════════════════════════════

def check_order(
    intent: OrderIntent,
    account: AccountInfo,
    device_id: str,
    double_confirmed: bool = False,
) -> RiskCheckResult:
    """
    执行预交易风控。

    参数:
        intent:    待下单意图
        account:   当前账户信息（取 equity 作为净值参考）
        device_id: 用于日内统计
        double_confirmed: 用户是否已勾选"我确认大额订单"
    """
    reasons: List[str] = []
    warnings: List[str] = []

    # 配置
    max_single_pct = _env_float("RISK_MAX_SINGLE_PCT", 0.20)
    max_etf_pct = _env_float("RISK_MAX_SINGLE_PCT_ETF", 0.50)
    double_confirm_pct = _env_float("RISK_DOUBLE_CONFIRM_PCT", 0.50)
    max_daily_orders = _env_int("RISK_MAX_DAILY_ORDERS", 20)
    max_cancel_rate = _env_float("RISK_MAX_DAILY_CANCEL_RATE", 0.40)
    block_market = _env_bool("RISK_BLOCK_MARKET_ORDER", True)
    whitelist = _load_whitelist()
    etf_set = _load_etf_set()

    # ── 1. 订单类型 ──
    if block_market and intent.order_type == OrderType.MARKET:
        reasons.append(
            "市价单已被风控禁用（防滑点）。请改用限价单。"
            "如需开启，设置 RISK_BLOCK_MARKET_ORDER=0"
        )

    # ── 2. 白名单 ──
    sym = intent.symbol.upper()
    if sym not in whitelist:
        reasons.append(
            f"标的 {sym} 不在白名单。当前白名单：{', '.join(sorted(whitelist)[:10])}..."
            "（共 {} 个）。可通过 RISK_WHITELIST 环境变量扩充".format(len(whitelist))
        )

    # ── 3. 单笔金额上限 ──
    # 估算名义金额：优先用 limit_price；没有就跳过（市价单已被禁，理论上不会到这）
    notional = (intent.qty * intent.limit_price) if intent.limit_price else 0.0
    equity = account.equity or 0.0
    notional_pct = (notional / equity) if equity > 0 else 0.0

    is_etf = sym in etf_set
    cap = max_etf_pct if is_etf else max_single_pct

    if equity > 0 and notional_pct > cap:
        reasons.append(
            f"单笔金额 ${notional:,.2f} 占净值 {notional_pct*100:.1f}%，"
            f"超过{'ETF ' if is_etf else ''}上限 {cap*100:.0f}%。"
            "如需放宽，设置 RISK_MAX_SINGLE_PCT / RISK_MAX_SINGLE_PCT_ETF"
        )

    # ── 4. 资金充足性（仅买入）──
    if intent.side == OrderSide.BUY and notional > 0:
        if notional > account.buying_power:
            reasons.append(
                f"资金不足：本单需 ${notional:,.2f}，"
                f"可用购买力 ${account.buying_power:,.2f}"
            )

    # ── 5. 日内订单数上限 ──
    stats = _load_daily_stats(device_id)
    placed = int(stats.get("orders_placed", 0))
    canceled = int(stats.get("orders_canceled", 0))
    if placed >= max_daily_orders:
        reasons.append(
            f"今日已下单 {placed} 笔，达到上限 {max_daily_orders}。"
            "明日恢复，或调高 RISK_MAX_DAILY_ORDERS"
        )

    # ── 6. 日内撤单率 ──
    if placed > 0:
        cancel_rate = canceled / placed
        if cancel_rate > max_cancel_rate:
            reasons.append(
                f"今日撤单率 {cancel_rate*100:.1f}%（{canceled}/{placed}），"
                f"超过上限 {max_cancel_rate*100:.0f}%。继续高频撤单可能被券商风控冻结账户"
            )
        elif cancel_rate > max_cancel_rate * 0.8:
            warnings.append(
                f"今日撤单率 {cancel_rate*100:.1f}% 接近上限 {max_cancel_rate*100:.0f}%"
            )

    # ── 7. 二次确认（>50% 需要在弹窗里额外勾选）──
    needs_double = (equity > 0 and notional_pct >= double_confirm_pct)
    if needs_double and not double_confirmed:
        reasons.append(
            f"本单占净值 {notional_pct*100:.1f}%，达到大额订单阈值"
            f"（≥{double_confirm_pct*100:.0f}%），需要在弹窗中勾选「我确认大额下单」"
        )

    passed = len(reasons) == 0
    return RiskCheckResult(
        passed=passed,
        reasons=reasons,
        warnings=warnings,
        needs_double_confirm=needs_double,
        notional_pct=notional_pct,
    )


# ════════════════════════════════════════════════════════════
# 风控当前配置快照（前端"设置"页用）
# ════════════════════════════════════════════════════════════

def current_config() -> dict:
    return {
        "max_single_pct": _env_float("RISK_MAX_SINGLE_PCT", 0.20),
        "max_single_pct_etf": _env_float("RISK_MAX_SINGLE_PCT_ETF", 0.50),
        "double_confirm_pct": _env_float("RISK_DOUBLE_CONFIRM_PCT", 0.50),
        "max_daily_orders": _env_int("RISK_MAX_DAILY_ORDERS", 20),
        "max_daily_cancel_rate": _env_float("RISK_MAX_DAILY_CANCEL_RATE", 0.40),
        "block_market_order": _env_bool("RISK_BLOCK_MARKET_ORDER", True),
        "whitelist_size": len(_load_whitelist()),
        "whitelist_sample": sorted(_load_whitelist())[:15],
    }
