"""
brokers/tiger_push.py — Tiger WebSocket push integration (X6 c2).

Wraps tigeropen's PushClient, runs it in its own (SDK-managed) background
thread, and routes every event into the RealtimeStateStore singleton.

Lifecycle:
  hub.start(user_scope, credentials)    # creates + connects + subscribes
  hub.stop(user_scope)                  # unsubscribes + disconnects + drops
  hub.is_running(user_scope)            # bool
  hub.subscribe_quotes(user_scope, symbols)   # opt-in real-time quote ticks

Each user_scope owns exactly one PushClient instance. The SDK handles the
network I/O thread; we only set callback properties and forward to the
state store. Errors inside callbacks are swallowed (logged at most) — we
must never let a single bad event kill the connection.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from .base import BrokerError, TigerCredentials
from .realtime_state import state as _state


log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# SDK lazy import (tests monkeypatch this to return mocks)
# ════════════════════════════════════════════════════════════

def _import_push_sdk():
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.push.push_client import PushClient
        return {
            "TigerOpenClientConfig": TigerOpenClientConfig,
            "PushClient": PushClient,
        }
    except ImportError as e:
        raise BrokerError(
            "tigeropen 缺失,无法启动推送。pip install tigeropen>=3.2.0,<4.0.0"
        ) from e


def _strip_pem_markers(pem: str) -> str:
    """tigeropen 期望纯 base64,见 tiger_adapter 同名函数。"""
    body = [
        ln for ln in pem.strip().splitlines()
        if "BEGIN" not in ln and "END" not in ln
    ]
    return "".join(body).strip()


# ════════════════════════════════════════════════════════════
# Single-user wrapper around PushClient
# ════════════════════════════════════════════════════════════

class TigerPushClientWrapper:
    """One Tiger PushClient + normalization callbacks for one user_scope."""

    def __init__(self, user_scope: str, credentials: TigerCredentials) -> None:
        self.user_scope = user_scope
        self.tiger_id = credentials.tiger_id
        self.account = credentials.account
        self.license = credentials.license
        self._private_key_body = _strip_pem_markers(credentials.private_key)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()

    # ── public lifecycle ────────────────────────────────────

    def start(self) -> None:
        """
        Construct + connect the PushClient and hook callbacks.
        Subscribes run inside connect_callback so they only fire after the
        WebSocket handshake completes (otherwise SDK silently drops them).
        """
        with self._lock:
            if self._client is not None:
                return
            sdk = _import_push_sdk()

            config = sdk["TigerOpenClientConfig"]()
            config.private_key = self._private_key_body
            config.tiger_id = self.tiger_id
            config.account = self.account
            config.license = self.license

            # socket_host_port is ('ssl', host, port) for ssl conns
            shp = config.socket_host_port
            if isinstance(shp, (list, tuple)) and len(shp) >= 3:
                _, host, port = shp[0], shp[1], shp[2]
                use_ssl = (str(shp[0]).lower() == "ssl")
            else:
                # Fallback if SDK shape changes
                host, port, use_ssl = "openapi.tigerfintech.com", 9883, True

            client = sdk["PushClient"](host, port, use_ssl=use_ssl)

            # Hook callbacks BEFORE connect so we don't miss the connect frame
            client.connect_callback = self._on_connect
            client.disconnect_callback = self._on_disconnect
            client.error_callback = self._on_error
            client.asset_changed = self._on_asset_changed
            client.position_changed = self._on_position_changed
            client.order_changed = self._on_order_changed
            client.quote_changed = self._on_quote_changed

            client.connect(self.tiger_id, self._private_key_body)
            self._client = client
            # _connected flips True inside on_connect_callback once SDK
            # finishes the handshake; not here.

    def stop(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
        if client is None:
            return
        # Best-effort cleanup — never raise out of stop()
        for fn in (
            lambda: client.unsubscribe_asset(),
            lambda: client.unsubscribe_position(),
            lambda: client.unsubscribe_order(),
            lambda: client.disconnect(),
        ):
            try:
                fn()
            except Exception:
                pass

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to real-time quote ticks for the given symbols.
        Safe to call before connect completes; tigeropen queues it."""
        with self._lock:
            client = self._client
        if client is None or not symbols:
            return
        try:
            client.subscribe_quote(list(symbols))
        except Exception:
            log.warning("tiger_push: subscribe_quote failed", exc_info=True)

    # ── connection lifecycle callbacks ──────────────────────

    def _on_connect(self, frame=None) -> None:
        """Fires on the SDK thread once the WebSocket handshake completes."""
        with self._lock:
            client = self._client
            if client is None:
                return
            self._connected = True
        # Subscribe AFTER connect — SDK silently drops pre-connect subscribes
        try:
            client.subscribe_asset(account=self.account)
            client.subscribe_position(account=self.account)
            client.subscribe_order(account=self.account)
        except Exception:
            log.warning("tiger_push: post-connect subscribe failed", exc_info=True)

    def _on_disconnect(self, frame=None) -> None:
        with self._lock:
            self._connected = False

    def _on_error(self, frame=None) -> None:
        # tigeropen's reconnect logic handles transient errors; we just log
        log.warning("tiger_push: error frame for %s: %r", self.user_scope, frame)

    # ── event callbacks (route to RealtimeStateStore) ───────

    def _on_asset_changed(self, frame) -> None:
        try:
            _state.update_account(self.user_scope, self._normalize_asset(frame))
        except Exception:
            log.warning("tiger_push: asset normalize failed", exc_info=True)

    def _on_position_changed(self, frame) -> None:
        try:
            _state.update_position(self.user_scope, self._normalize_position(frame))
        except Exception:
            log.warning("tiger_push: position normalize failed", exc_info=True)

    def _on_order_changed(self, frame) -> None:
        try:
            _state.update_order(self.user_scope, self._normalize_order(frame))
        except Exception:
            log.warning("tiger_push: order normalize failed", exc_info=True)

    def _on_quote_changed(self, frame) -> None:
        try:
            _state.update_quote(self.user_scope, self._normalize_quote(frame))
        except Exception:
            log.warning("tiger_push: quote normalize failed", exc_info=True)

    # ── normalization helpers (frame → dict) ────────────────

    @staticmethod
    def _normalize_asset(frame) -> dict:
        cash = float(getattr(frame, "cash", 0) or 0)
        gross = float(getattr(frame, "gross_position_value", 0) or 0)
        return {
            "cash": cash,
            "buying_power": float(getattr(frame, "buying_power", cash) or cash),
            "equity": cash + gross,
            "currency": str(getattr(frame, "currency", "USD") or "USD"),
            "account_id": str(getattr(frame, "account", "") or ""),
        }

    @staticmethod
    def _normalize_position(frame) -> dict:
        contract = getattr(frame, "contract", None)
        symbol = (
            getattr(contract, "symbol", None)
            or getattr(frame, "symbol", None)
            or ""
        )
        qty = float(getattr(frame, "quantity", 0) or 0)
        avg = float(getattr(frame, "average_cost", 0) or 0)
        market_value = float(
            getattr(frame, "market_value", 0) or (qty * avg)
        )
        unrealized = float(getattr(frame, "unrealized_pnl", 0) or 0)
        cost = qty * avg
        pct = (unrealized / cost * 100) if cost > 0 else 0.0
        return {
            "symbol": str(symbol),
            "qty": qty,
            "avg_entry_price": avg,
            "market_value": market_value,
            "unrealized_pl": unrealized,
            "unrealized_pl_pct": pct,
            "current_price": float(getattr(frame, "market_price", avg) or avg),
        }

    @staticmethod
    def _normalize_order(frame) -> dict:
        contract = getattr(frame, "contract", None)
        symbol = (
            getattr(contract, "symbol", None)
            or getattr(frame, "symbol", None)
            or ""
        )
        oid = getattr(frame, "id", None) or getattr(frame, "order_id", None)
        return {
            "broker_order_id": str(oid) if oid is not None else "",
            "symbol": str(symbol),
            "side": str(getattr(frame, "action", "") or "").lower(),
            "qty": float(getattr(frame, "quantity", 0) or 0),
            "filled_qty": float(getattr(frame, "filled", 0) or 0),
            "limit_price": getattr(frame, "limit_price", None),
            "status": str(getattr(frame, "status", "") or ""),
            "submitted_at": str(getattr(frame, "order_time", "") or ""),
        }

    @staticmethod
    def _normalize_quote(frame) -> dict:
        return {
            "symbol": str(getattr(frame, "symbol", "") or ""),
            "price": float(getattr(frame, "latest_price", 0) or 0),
            "change": float(getattr(frame, "change", 0) or 0),
            "change_pct": float(getattr(frame, "change_pct", 0) or 0),
            "volume": int(getattr(frame, "volume", 0) or 0),
            "timestamp": int(getattr(frame, "timestamp", 0) or 0),
        }


# ════════════════════════════════════════════════════════════
# Hub — one wrapper per user_scope
# ════════════════════════════════════════════════════════════

class TigerPushHub:
    """Manages per-user_scope PushClient lifecycles. Module-level singleton."""

    def __init__(self) -> None:
        self._clients: Dict[str, TigerPushClientWrapper] = {}
        self._lock = threading.Lock()

    def start(self, user_scope: str, credentials: TigerCredentials) -> TigerPushClientWrapper:
        with self._lock:
            existing = self._clients.get(user_scope)
            if existing is not None:
                return existing
            wrapper = TigerPushClientWrapper(user_scope, credentials)
            self._clients[user_scope] = wrapper
        # Start outside the global lock — connect() blocks until handshake
        # in some SDK paths and we don't want to serialize all users on it.
        try:
            wrapper.start()
        except Exception:
            # Roll back so a future retry isn't blocked by a dead wrapper
            with self._lock:
                self._clients.pop(user_scope, None)
            raise
        return wrapper

    def stop(self, user_scope: str) -> None:
        with self._lock:
            wrapper = self._clients.pop(user_scope, None)
        if wrapper is not None:
            wrapper.stop()

    def is_running(self, user_scope: str) -> bool:
        with self._lock:
            wrapper = self._clients.get(user_scope)
        return wrapper is not None and wrapper.is_connected()

    def subscribe_quotes(self, user_scope: str, symbols: List[str]) -> None:
        with self._lock:
            wrapper = self._clients.get(user_scope)
        if wrapper is not None:
            wrapper.subscribe_quotes(symbols)

    def get(self, user_scope: str) -> Optional[TigerPushClientWrapper]:
        with self._lock:
            return self._clients.get(user_scope)

    def clear(self) -> None:
        """Test helper. Stops every connection."""
        with self._lock:
            wrappers = list(self._clients.values())
            self._clients.clear()
        for w in wrappers:
            try:
                w.stop()
            except Exception:
                pass


# Module-level singleton — same pattern as credentials_store.store
hub = TigerPushHub()
