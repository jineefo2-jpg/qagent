"""
订单意图（OrderIntent）的持久化包装。

生命周期：
  1. Agent 调用 place_order_intent → save_intent
  2. 前端弹窗读取 → get_intent
  3. 用户确认 → 后端 confirm 路由 pop_intent + broker.place_order
  4. 5 分钟内未确认 → TTL 自动过期（前端再次确认会失败，必须重新生成）
"""
from __future__ import annotations

from typing import Optional

from cache import cache
from .base import OrderIntent, OrderSide, OrderType


INTENT_TTL_SECONDS = 300   # 5 分钟过期


def _key(device_id: str, intent_id: str) -> str:
    return f"quant:intent:{device_id}:{intent_id}"


def save_intent(device_id: str, intent: OrderIntent) -> None:
    cache.set(_key(device_id, intent.intent_id),
              intent.to_dict(), ttl=INTENT_TTL_SECONDS)


def get_intent(device_id: str, intent_id: str) -> Optional[OrderIntent]:
    d = cache.get(_key(device_id, intent_id))
    if not d:
        return None
    return _from_dict(d)


def pop_intent(device_id: str, intent_id: str) -> Optional[OrderIntent]:
    """取出并删除（确认下单时用，防止重复提交）"""
    d = cache.get(_key(device_id, intent_id))
    if not d:
        return None
    cache.delete(_key(device_id, intent_id))
    return _from_dict(d)


def _from_dict(d: dict) -> OrderIntent:
    return OrderIntent(
        intent_id=d["intent_id"],
        symbol=d["symbol"],
        side=OrderSide(d["side"]),
        qty=float(d["qty"]),
        order_type=OrderType(d["order_type"]),
        limit_price=(float(d["limit_price"]) if d.get("limit_price") is not None
                      else None),
        time_in_force=d.get("time_in_force", "day"),
        notes=d.get("notes", ""),
        estimated_cost=d.get("estimated_cost"),
        created_at=d.get("created_at", 0.0),
    )
