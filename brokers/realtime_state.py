"""
brokers/realtime_state.py — per-user real-time state + pub/sub for SSE.

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  TigerPushClient (background thread, c2)                         │
  │    on order_changed  → state.update_order(user_scope, order)     │
  │    on position_changed → state.update_position(...)              │
  │    on asset_changed  → state.update_account(...)                 │
  │    on quote_changed  → state.update_quote(...)                   │
  └────────────┬─────────────────────────────────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  RealtimeStateStore  (this file)                                 │
  │   {user_scope: _UserState{                                       │
  │      account, positions{sym}, orders{id}, quotes{sym},           │
  │      subscribers: [Queue, ...]                                   │
  │   }}                                                             │
  └────────────┬─────────────────────────────────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  SSE handler (c3) — server.py /api/broker/stream                 │
  │    q = state.subscribe(user_scope)                               │
  │    while True:                                                   │
  │       event = q.get(timeout=...)                                 │
  │       yield SSE-formatted line                                   │
  └──────────────────────────────────────────────────────────────────┘

Thread safety:
  - One RLock per UserState protects mutation + subscriber list.
  - Updates from the Tiger background thread are non-blocking for slow
    consumers: queue full → mark subscriber dead → remove.
  - Initial snapshot is sent on subscribe so the UI shows last-known
    state immediately (no blank period before first push arrives).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional


@dataclass
class _UserState:
    user_scope: str
    account: Optional[dict] = None
    positions: Dict[str, dict] = field(default_factory=dict)   # symbol → position dict
    orders: Dict[str, dict] = field(default_factory=dict)      # order_id → order dict
    quotes: Dict[str, dict] = field(default_factory=dict)      # symbol → quote dict
    last_updated: float = field(default_factory=time.time)
    subscribers: List[Queue] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)


class RealtimeStateStore:
    """Thread-safe per-user state + pub/sub for SSE subscribers."""

    def __init__(self) -> None:
        self._users: Dict[str, _UserState] = {}
        self._global_lock = threading.Lock()

    # ────────────────────────────────────────────────────────────
    # Subscription
    # ────────────────────────────────────────────────────────────

    def subscribe(self, user_scope: str, queue_max: int = 200) -> Queue:
        """
        Register a new SSE subscriber for this user_scope. Returns a Queue
        the caller polls. The initial snapshot (last-known account /
        positions / orders / quotes) is enqueued immediately so the UI has
        something to render before the first push tick arrives.
        """
        us = self._get_or_create(user_scope)
        q: Queue = Queue(maxsize=queue_max)
        with us.lock:
            us.subscribers.append(q)
            if us.account is not None:
                q.put(("account", dict(us.account)))
            for pos in us.positions.values():
                q.put(("position", dict(pos)))
            for order in us.orders.values():
                q.put(("order", dict(order)))
            for quote in us.quotes.values():
                q.put(("quote", dict(quote)))
        return q

    def unsubscribe(self, user_scope: str, q: Queue) -> None:
        us = self._users.get(user_scope)
        if us is None:
            return
        with us.lock:
            try:
                us.subscribers.remove(q)
            except ValueError:
                pass

    def subscriber_count(self, user_scope: str) -> int:
        us = self._users.get(user_scope)
        if us is None:
            return 0
        with us.lock:
            return len(us.subscribers)

    # ────────────────────────────────────────────────────────────
    # Mutations (called from TigerPushClient callbacks in c2)
    # ────────────────────────────────────────────────────────────

    def update_account(self, user_scope: str, account: dict) -> None:
        us = self._get_or_create(user_scope)
        with us.lock:
            us.account = dict(account)
            us.last_updated = time.time()
            self._broadcast(us, "account", account)

    def update_position(self, user_scope: str, position: dict) -> None:
        """
        Update a position. `position` dict must include 'symbol'.
        qty=0 → position closed → removed from state.
        """
        symbol = position.get("symbol")
        if not symbol:
            return
        us = self._get_or_create(user_scope)
        with us.lock:
            try:
                qty = float(position.get("qty", 0) or 0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty == 0:
                us.positions.pop(symbol, None)
            else:
                us.positions[symbol] = dict(position)
            us.last_updated = time.time()
            self._broadcast(us, "position", position)

    def update_order(self, user_scope: str, order: dict) -> None:
        oid = order.get("broker_order_id") or order.get("id")
        if not oid:
            return
        us = self._get_or_create(user_scope)
        with us.lock:
            us.orders[str(oid)] = dict(order)
            us.last_updated = time.time()
            self._broadcast(us, "order", order)

    def update_quote(self, user_scope: str, quote: dict) -> None:
        """
        High-frequency tick update. Doesn't bump last_updated (would defeat
        the purpose of last_updated for diagnostics).
        """
        symbol = quote.get("symbol")
        if not symbol:
            return
        us = self._get_or_create(user_scope)
        with us.lock:
            us.quotes[symbol] = dict(quote)
            self._broadcast(us, "quote", quote)

    # ────────────────────────────────────────────────────────────
    # Diagnostics
    # ────────────────────────────────────────────────────────────

    def snapshot(self, user_scope: str) -> dict:
        us = self._users.get(user_scope)
        if us is None:
            return {"user_scope": user_scope, "exists": False}
        with us.lock:
            return {
                "user_scope": user_scope,
                "exists": True,
                "account": dict(us.account) if us.account else None,
                "positions": [dict(p) for p in us.positions.values()],
                "orders": [dict(o) for o in us.orders.values()],
                "quote_count": len(us.quotes),
                "subscriber_count": len(us.subscribers),
                "last_updated": us.last_updated,
            }

    def clear(self) -> None:
        """Test helper. Production code MUST NOT call this."""
        with self._global_lock:
            self._users.clear()

    # ────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────

    def _get_or_create(self, user_scope: str) -> _UserState:
        with self._global_lock:
            us = self._users.get(user_scope)
            if us is None:
                us = _UserState(user_scope=user_scope)
                self._users[user_scope] = us
            return us

    def _broadcast(self, us: _UserState, event_type: str, payload: dict) -> None:
        """
        Push (event_type, payload) to every subscriber of `us`.
        Slow subscribers (full queue) get dropped — we never block the
        producer thread. Caller must hold us.lock.
        """
        dead = []
        for q in us.subscribers:
            try:
                q.put_nowait((event_type, dict(payload)))
            except Exception:
                # queue.Full (most common) — slow consumer, drop them
                dead.append(q)
        for q in dead:
            try:
                us.subscribers.remove(q)
            except ValueError:
                pass


# Module-level singleton — same pattern as credentials_store.store.
state = RealtimeStateStore()
