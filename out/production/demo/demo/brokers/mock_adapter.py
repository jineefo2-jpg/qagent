"""
MockAdapter —— 纯本地虚拟交易（C1 方案）。

每个登录用户独立的虚拟账户：
  - 起始 $100,000 虚拟现金
  - 持仓、订单、资金完全本地账本（Redis）
  - 价格来自 market_quote 的真实美股报价 → 持仓市值、限价撮合都跟着市场走
  - 与 Alpaca 完全解耦：不发任何 API，不受限额/盘外影响

用户隔离：通过 thread-local 拿到当前请求的 device_id（登录用户为 "u:google:..."），
该字符串作为账本 namespace。
"""
from __future__ import annotations

import os
import time
import uuid
import datetime
from typing import List, Optional

from cache import cache
from .base import (
    BrokerAdapter, BrokerError, BrokerRejectedError,
    AccountInfo, Position, OrderIntent, OrderResult,
    OrderSide, OrderType, OrderStatus,
    MockCredentials,
)


# 起始虚拟现金，可被环境变量覆盖(向后兼容路径)
INITIAL_CASH = float(os.getenv("MOCK_INITIAL_CASH", "100000").strip() or "100000")


# ════════════════════════════════════════════════════════════
# 用户 namespace 提取
# ════════════════════════════════════════════════════════════

def _current_user_ns() -> str:
    """
    从 thread-local 读 device_id 作为账本 namespace。
    server.py 的 chat 路由把它设成 _scope_id(user, x_device_id)，
    登录用户为 "u:google:...", 匿名用户为 device_id。
    """
    try:
        from quant_agent import _get_request_device_id
        ns = _get_request_device_id() or "default"
        return ns
    except Exception:
        return "default"


# ════════════════════════════════════════════════════════════
# Redis key 规则
# ════════════════════════════════════════════════════════════

def _k_cash(ns: str) -> str:
    return f"quant:vcash:{ns}"

def _k_positions(ns: str) -> str:
    return f"quant:vpos:{ns}"

def _k_orders_idx(ns: str) -> str:
    return f"quant:vorders:{ns}"

def _k_order(ns: str, oid: str) -> str:
    return f"quant:vorder:{ns}:{oid}"


# ════════════════════════════════════════════════════════════
# 价格抓取（用于撮合）
# ════════════════════════════════════════════════════════════

def _current_price(symbol: str, fresh: bool = False) -> Optional[float]:
    """
    通过 market_quote 拉实时价；失败返回 None。
    fresh=True 时绕开 30s 缓存（持仓估值/撮合判断用）。
    """
    try:
        from quant_agent import market_quote
        q = market_quote(symbol, skip_cache=fresh)
        if q and q.get("success") and q.get("price") is not None:
            return float(q["price"])
    except Exception:
        return None
    return None


# ════════════════════════════════════════════════════════════
# 账本读写（小工具）
# ════════════════════════════════════════════════════════════

def _load_cash(ns: str) -> float:
    """新用户首次访问初始化为 INITIAL_CASH"""
    v = cache.get(_k_cash(ns))
    if v is None:
        cache.set(_k_cash(ns), INITIAL_CASH, ttl=None)
        return INITIAL_CASH
    return float(v)


def _save_cash(ns: str, v: float):
    cache.set(_k_cash(ns), float(v), ttl=None)


def _load_positions(ns: str) -> dict:
    """返回 {symbol: {qty, avg_entry_price, total_cost}}"""
    return dict(cache.get(_k_positions(ns)) or {})


def _save_positions(ns: str, positions: dict):
    cache.set(_k_positions(ns), positions, ttl=None)


def _load_order_ids(ns: str) -> list:
    return list(cache.get(_k_orders_idx(ns)) or [])


def _save_order_ids(ns: str, ids: list):
    cache.set(_k_orders_idx(ns), ids, ttl=None)


def _load_order(ns: str, oid: str) -> Optional[dict]:
    d = cache.get(_k_order(ns, oid))
    return d if isinstance(d, dict) else None


def _save_order(ns: str, oid: str, d: dict):
    cache.set(_k_order(ns, oid), d, ttl=None)


# ════════════════════════════════════════════════════════════
# 撮合逻辑
# ════════════════════════════════════════════════════════════

def _can_fill(side: str, limit_price: float, market_price: float) -> bool:
    """限价单触发判断"""
    if side == "buy":
        return market_price <= limit_price
    return market_price >= limit_price


def _try_fill_order(ns: str, order: dict) -> bool:
    """
    尝试用当前市价撮合一笔 new 状态的订单。返回是否成功 fill。
    fill 会同步更新账本（现金/持仓）和订单本身。
    """
    if order.get("status") != "new":
        return False
    side = order.get("side")
    symbol = order.get("symbol")
    qty = float(order.get("qty", 0))
    limit_price = float(order.get("limit_price", 0))
    if qty <= 0 or limit_price <= 0:
        return False

    # 撮合用最新价，绕开 30s 缓存
    mp = _current_price(symbol, fresh=True)
    if mp is None:
        return False
    if not _can_fill(side, limit_price, mp):
        return False

    # 成交价：买入用 min(limit, market)，卖出用 max(limit, market)
    fill_price = min(limit_price, mp) if side == "buy" else max(limit_price, mp)
    notional = qty * fill_price

    if side == "buy":
        # 现金扣减：实际成交价（可能比预扣的更低，差额返还）
        reserved = float(order.get("reserved_cash", qty * limit_price))
        cash = _load_cash(ns)
        # 释放 reserved，再扣实际
        cash += reserved
        cash -= notional
        _save_cash(ns, cash)
        # 持仓更新
        positions = _load_positions(ns)
        pos = positions.get(symbol, {"qty": 0, "avg_entry_price": 0, "total_cost": 0})
        new_total_cost = float(pos["total_cost"]) + notional
        new_qty = float(pos["qty"]) + qty
        positions[symbol] = {
            "qty": new_qty,
            "avg_entry_price": new_total_cost / new_qty if new_qty > 0 else 0,
            "total_cost": new_total_cost,
        }
        _save_positions(ns, positions)
    else:  # sell
        positions = _load_positions(ns)
        pos = positions.get(symbol)
        if not pos or float(pos.get("qty", 0)) < qty:
            # 持仓不足，标记 rejected
            order["status"] = "rejected"
            order["rejected_reason"] = "持仓不足"
            return False
        new_qty = float(pos["qty"]) - qty
        # 同比减少成本（按平均成本结算）
        if float(pos["qty"]) > 0:
            cost_per_share = float(pos["total_cost"]) / float(pos["qty"])
        else:
            cost_per_share = float(pos.get("avg_entry_price", 0))
        new_total_cost = max(0.0, float(pos["total_cost"]) - cost_per_share * qty)
        if new_qty <= 0.0001:
            positions.pop(symbol, None)
        else:
            positions[symbol] = {
                "qty": new_qty,
                "avg_entry_price": new_total_cost / new_qty if new_qty > 0 else 0,
                "total_cost": new_total_cost,
            }
        _save_positions(ns, positions)
        # 现金入账
        cash = _load_cash(ns)
        cash += notional
        _save_cash(ns, cash)

    # 标记订单 filled
    order["status"] = "filled"
    order["filled_qty"] = qty
    order["filled_avg_price"] = fill_price
    order["filled_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return True


# ════════════════════════════════════════════════════════════
# MockAdapter
# ════════════════════════════════════════════════════════════

class MockAdapter(BrokerAdapter):
    name = "mock-paper"

    def __init__(self, credentials: Optional[MockCredentials] = None):
        """
        X2:凭证可选注入。MockAdapter 的用户隔离继续走 thread-local namespace
        (见 _current_user_ns),仅 initial_cash 通过 credentials 传入。
        无 credentials 时回落到模块级 INITIAL_CASH(向后兼容路径)。
        """
        if credentials is not None and not isinstance(credentials, MockCredentials):
            raise BrokerError(
                f"MockAdapter requires MockCredentials, got {type(credentials).__name__}"
            )
        self._initial_cash = (
            credentials.initial_cash if credentials is not None else INITIAL_CASH
        )

    def is_configured(self) -> bool:
        return True

    def ping(self) -> bool:
        return True

    # ── 账户 ──

    def get_account(self) -> AccountInfo:
        ns = _current_user_ns()
        cash = _load_cash(ns)
        positions = _load_positions(ns)

        # 实时计算总市值（绕开缓存，每次刷新都拉新价）
        market_value = 0.0
        for sym, p in positions.items():
            mp = _current_price(sym, fresh=True)
            if mp is not None:
                market_value += mp * float(p.get("qty", 0))
            else:
                # 拉不到价就用平均成本估
                market_value += float(p.get("avg_entry_price", 0)) * float(p.get("qty", 0))

        equity = cash + market_value
        return AccountInfo(
            cash=cash,
            buying_power=cash,           # 不做杠杆，购买力 = 现金
            equity=equity,
            currency="USD",
            account_id=ns,
            status="ACTIVE",
            raw={"backend": "mock", "initial_cash": self._initial_cash},
        )

    def list_positions(self) -> List[Position]:
        ns = _current_user_ns()
        positions = _load_positions(ns)
        out = []
        for sym, p in positions.items():
            qty = float(p.get("qty", 0))
            avg = float(p.get("avg_entry_price", 0))
            total_cost = float(p.get("total_cost", 0))
            mp = _current_price(sym, fresh=True)
            if mp is None:
                mp = avg
            market_value = mp * qty
            unrealized = market_value - total_cost
            pct = (unrealized / total_cost * 100) if total_cost > 0 else 0
            out.append(Position(
                symbol=sym,
                qty=qty,
                avg_entry_price=avg,
                market_value=market_value,
                unrealized_pl=unrealized,
                unrealized_pl_pct=pct,
                current_price=mp,
            ))
        return out

    # ── 订单 ──

    def place_order(self, intent: OrderIntent) -> OrderResult:
        ns = _current_user_ns()
        symbol = intent.symbol
        side = intent.side.value
        qty = float(intent.qty)
        limit_price = float(intent.limit_price or 0)

        if intent.order_type != OrderType.LIMIT:
            raise BrokerRejectedError("MockAdapter 仅支持限价单")
        if limit_price <= 0:
            raise BrokerRejectedError("limit_price 必须为正")

        # ── 资金/持仓预检 ──
        if side == "buy":
            notional = qty * limit_price
            cash = _load_cash(ns)
            if notional > cash + 0.01:
                raise BrokerRejectedError(
                    f"现金不足：需 ${notional:,.2f}，可用 ${cash:,.2f}")
        else:  # sell
            positions = _load_positions(ns)
            held = float(positions.get(symbol, {}).get("qty", 0))
            if qty > held + 0.0001:
                raise BrokerRejectedError(
                    f"持仓不足：要卖 {qty}，持有 {held}")

        # ── 创建订单 ──
        order_id = "vord_" + uuid.uuid4().hex[:12]
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        reserved_cash = qty * limit_price if side == "buy" else 0

        order = {
            "order_id": order_id,
            "intent_id": intent.intent_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": "limit",
            "limit_price": limit_price,
            "status": "new",
            "filled_qty": 0,
            "filled_avg_price": None,
            "submitted_at": now_iso,
            "filled_at": None,
            "reserved_cash": reserved_cash,
        }

        # ── 资金预扣（buy）/ 持仓预占（sell 不扣，撮合时再扣）──
        if side == "buy":
            cash = _load_cash(ns)
            _save_cash(ns, cash - reserved_cash)

        # ── 尝试立刻撮合 ──
        _try_fill_order(ns, order)

        # ── 落库 ──
        _save_order(ns, order_id, order)
        ids = _load_order_ids(ns)
        ids.insert(0, order_id)
        _save_order_ids(ns, ids[:500])  # 保留 500 条

        return self._order_to_result(order)

    def cancel_order(self, broker_order_id: str) -> bool:
        ns = _current_user_ns()
        order = _load_order(ns, broker_order_id)
        if not order:
            raise BrokerError(f"订单不存在: {broker_order_id}")
        st = order.get("status")
        if st in ("filled", "canceled", "rejected", "expired"):
            raise BrokerRejectedError(f"订单已 {st}，不能撤")

        # 退还预扣的现金（仅 buy）
        if order.get("side") == "buy":
            reserved = float(order.get("reserved_cash", 0))
            if reserved > 0:
                cash = _load_cash(ns)
                _save_cash(ns, cash + reserved)

        order["status"] = "canceled"
        order["canceled_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _save_order(ns, broker_order_id, order)
        return True

    def get_order(self, broker_order_id: str) -> OrderResult:
        ns = _current_user_ns()
        order = _load_order(ns, broker_order_id)
        if not order:
            raise BrokerError(f"订单不存在: {broker_order_id}")
        # 主动尝试撮合一次
        if order.get("status") == "new":
            if _try_fill_order(ns, order):
                _save_order(ns, broker_order_id, order)
        return self._order_to_result(order)

    def list_orders(self, status: Optional[str] = None,
                     limit: int = 50) -> List[OrderResult]:
        ns = _current_user_ns()
        ids = _load_order_ids(ns)
        out = []
        for oid in ids[:limit]:
            order = _load_order(ns, oid)
            if not order:
                continue
            # 对每个 new 状态的订单尝试撮合一次
            if order.get("status") == "new":
                if _try_fill_order(ns, order):
                    _save_order(ns, oid, order)
            # 过滤
            st = order.get("status")
            if status == "open" and st not in ("new",):
                continue
            if status == "closed" and st not in ("filled", "canceled", "rejected", "expired"):
                continue
            out.append(self._order_to_result(order))
        return out

    # ── 工具方法 ──

    def reset_account(self) -> None:
        """清空当前用户的所有账本数据，恢复到初始 $100k"""
        ns = _current_user_ns()
        for oid in _load_order_ids(ns):
            cache.delete(_k_order(ns, oid))
        cache.delete(_k_orders_idx(ns))
        cache.delete(_k_positions(ns))
        cache.delete(_k_cash(ns))

    def _order_to_result(self, order: dict) -> OrderResult:
        side = OrderSide.BUY if order.get("side") == "buy" else OrderSide.SELL
        st = order.get("status", "new")
        status_map = {
            "new": OrderStatus.NEW,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return OrderResult(
            broker_order_id=order["order_id"],
            intent_id=order.get("intent_id"),
            symbol=order["symbol"],
            side=side,
            qty=float(order.get("qty", 0)),
            filled_qty=float(order.get("filled_qty", 0)),
            order_type=OrderType.LIMIT,
            limit_price=float(order["limit_price"]) if order.get("limit_price") else None,
            status=status_map.get(st, OrderStatus.NEW),
            filled_avg_price=order.get("filled_avg_price"),
            submitted_at=order.get("submitted_at"),
            raw={"backend": "mock", "user_ns": _current_user_ns()},
        )
