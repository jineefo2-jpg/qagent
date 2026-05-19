"""
Alpaca paper trading 适配器。

依赖：pip install alpaca-py

凭证从环境变量读：
    ALPACA_API_KEY        Trading API Key ID（PK 开头）
    ALPACA_API_SECRET     Secret Key
    ALPACA_BASE_URL       https://paper-api.alpaca.markets （默认）

文档：https://docs.alpaca.markets/docs/trading-api
"""
from __future__ import annotations

import os
from typing import List, Optional

from .base import (
    BrokerAdapter, BrokerError, BrokerAuthError,
    BrokerNetworkError, BrokerRejectedError,
    AccountInfo, Position, OrderIntent, OrderResult,
    OrderSide, OrderType, OrderStatus,
)


# ════════════════════════════════════════════════════════════
# Alpaca SDK 惰性导入（缺包时给出友好提示）
# ════════════════════════════════════════════════════════════

def _import_alpaca():
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (
            LimitOrderRequest, MarketOrderRequest, GetOrdersRequest,
        )
        from alpaca.trading.enums import (
            OrderSide as AlpacaSide,
            TimeInForce,
            OrderStatus as AlpacaStatus,
            QueryOrderStatus,
        )
        return {
            "TradingClient": TradingClient,
            "LimitOrderRequest": LimitOrderRequest,
            "MarketOrderRequest": MarketOrderRequest,
            "GetOrdersRequest": GetOrdersRequest,
            "AlpacaSide": AlpacaSide,
            "TimeInForce": TimeInForce,
            "AlpacaStatus": AlpacaStatus,
            "QueryOrderStatus": QueryOrderStatus,
        }
    except ImportError as e:
        raise BrokerError(
            "缺少 alpaca-py 依赖，请运行: pip install alpaca-py"
        ) from e


# Alpaca 订单状态 → 我们的统一状态映射
_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.NEW,
    "pending_replace": OrderStatus.NEW,
    "calculated": OrderStatus.NEW,
    "stopped": OrderStatus.CANCELED,
    "held": OrderStatus.NEW,
}


def _map_status(raw) -> OrderStatus:
    """
    Alpaca 返回的 status 是 enum，str() 给出 "OrderStatus.FILLED" 这种带前缀的字符串。
    必须先剥掉前缀拿到纯 value 再查表。
    """
    if raw is None:
        return OrderStatus.NEW
    # 优先用 enum 自带的 .value（Alpaca SDK 的 OrderStatus 是 StrEnum）
    if hasattr(raw, "value"):
        s = str(raw.value).lower()
    else:
        s = str(raw).lower()
    # 兜底：剥掉 "orderstatus." 前缀
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return _STATUS_MAP.get(s, OrderStatus.NEW)


# ════════════════════════════════════════════════════════════
# AlpacaAdapter
# ════════════════════════════════════════════════════════════

class AlpacaAdapter(BrokerAdapter):
    name = "alpaca-paper"

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "").strip()
        self.api_secret = os.getenv("ALPACA_API_SECRET", "").strip()
        self.base_url = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ).strip()
        self._client = None
        self._sdk = None

    # ── 元信息 ──

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.is_configured():
            raise BrokerAuthError(
                "Alpaca 凭证未配置。请在 .env 设置 ALPACA_API_KEY / ALPACA_API_SECRET"
            )
        self._sdk = _import_alpaca()
        try:
            # paper=True 让 SDK 用 paper endpoint
            self._client = self._sdk["TradingClient"](
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper="paper" in self.base_url,
            )
        except Exception as e:
            raise BrokerAuthError(f"Alpaca 客户端初始化失败: {e}") from e
        return self._client

    def ping(self) -> bool:
        try:
            self.get_account()
            return True
        except Exception:
            return False

    # ── 账户 ──

    def get_account(self) -> AccountInfo:
        client = self._ensure_client()
        try:
            acc = client.get_account()
        except Exception as e:
            raise self._wrap_error(e)
        return AccountInfo(
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            equity=float(acc.equity),
            currency=getattr(acc, "currency", "USD") or "USD",
            account_id=str(getattr(acc, "id", "")),
            status=str(getattr(acc, "status", "")),
            raw={"account_number": str(getattr(acc, "account_number", ""))},
        )

    def list_positions(self) -> List[Position]:
        client = self._ensure_client()
        try:
            positions = client.get_all_positions()
        except Exception as e:
            raise self._wrap_error(e)
        out = []
        for p in positions:
            out.append(Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
                current_price=float(getattr(p, "current_price", 0) or 0) or None,
            ))
        return out

    # ── 订单 ──

    def place_order(self, intent: OrderIntent) -> OrderResult:
        client = self._ensure_client()
        sdk = self._sdk

        # Alpaca SDK 要求枚举对象
        alpaca_side = (
            sdk["AlpacaSide"].BUY if intent.side == OrderSide.BUY
            else sdk["AlpacaSide"].SELL
        )
        tif = sdk["TimeInForce"].DAY

        try:
            if intent.order_type == OrderType.LIMIT:
                req = sdk["LimitOrderRequest"](
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=alpaca_side,
                    time_in_force=tif,
                    limit_price=intent.limit_price,
                    client_order_id=intent.intent_id,
                )
            else:
                req = sdk["MarketOrderRequest"](
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=alpaca_side,
                    time_in_force=tif,
                    client_order_id=intent.intent_id,
                )
            order = client.submit_order(req)
        except Exception as e:
            raise self._wrap_error(e)

        return self._order_to_result(order, intent.intent_id)

    def cancel_order(self, broker_order_id: str) -> bool:
        client = self._ensure_client()
        try:
            client.cancel_order_by_id(broker_order_id)
            return True
        except Exception as e:
            raise self._wrap_error(e)

    def get_order(self, broker_order_id: str) -> OrderResult:
        client = self._ensure_client()
        try:
            order = client.get_order_by_id(broker_order_id)
        except Exception as e:
            raise self._wrap_error(e)
        return self._order_to_result(order)

    def list_orders(self, status: Optional[str] = None,
                     limit: int = 50) -> List[OrderResult]:
        client = self._ensure_client()
        sdk = self._sdk
        # status 映射
        q_status = sdk["QueryOrderStatus"].ALL
        if status == "open":
            q_status = sdk["QueryOrderStatus"].OPEN
        elif status == "closed":
            q_status = sdk["QueryOrderStatus"].CLOSED

        try:
            req = sdk["GetOrdersRequest"](status=q_status, limit=limit)
            orders = client.get_orders(filter=req)
        except Exception as e:
            raise self._wrap_error(e)
        return [self._order_to_result(o) for o in orders]

    # ── 内部辅助 ──

    def _order_to_result(self, order, intent_id: Optional[str] = None) -> OrderResult:
        """Alpaca Order 对象 → 统一 OrderResult"""
        side = (OrderSide.BUY if str(order.side).lower().endswith("buy")
                else OrderSide.SELL)
        order_type = (OrderType.LIMIT if str(order.order_type).lower() == "limit"
                       else OrderType.MARKET)
        return OrderResult(
            broker_order_id=str(order.id),
            intent_id=intent_id or str(getattr(order, "client_order_id", "") or ""),
            symbol=order.symbol,
            side=side,
            qty=float(order.qty),
            filled_qty=float(order.filled_qty or 0),
            order_type=order_type,
            limit_price=float(order.limit_price) if order.limit_price else None,
            status=_map_status(order.status),
            filled_avg_price=(float(order.filled_avg_price)
                              if order.filled_avg_price else None),
            submitted_at=(str(order.submitted_at) if order.submitted_at else None),
            raw={"id": str(order.id)},
        )

    def _wrap_error(self, e: Exception) -> BrokerError:
        """把 Alpaca SDK 的异常归类成我们的异常"""
        msg = str(e)
        low = msg.lower()
        if "unauthorized" in low or "forbidden" in low or "invalid api key" in low:
            return BrokerAuthError(f"Alpaca 鉴权失败：{msg}")
        if ("timeout" in low or "connection" in low or "network" in low
                or "name resolution" in low):
            return BrokerNetworkError(f"Alpaca 网络错误：{msg}")
        if ("insufficient" in low or "buying power" in low
                or "not tradable" in low or "rejected" in low):
            return BrokerRejectedError(f"Alpaca 拒单：{msg}")
        return BrokerError(f"Alpaca 错误：{msg}")
