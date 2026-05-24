"""
Tiger Brokers (老虎证券) 适配器 —— paper / 环球账户模拟环境。

依赖: pip install tigeropen>=3.2.0,<4.0.0
凭证由 BrokerRegistry 注入,通过 UI / CLI 绑定。
不会读 env (per ADR-0001 addendum)。

注意:
  * Tiger 用 RSA 私钥认证,**不是** OAuth。私钥永远不应出现在日志中。
  * 本 adapter 默认走"环球账户模拟",对应 broker_bindings.env='paper'。
  * 真实下单已被 CLAUDE.md 安全规则禁止;此处仅实现接口契约。
"""
from __future__ import annotations

from typing import List, Optional

from .base import (
    AccountInfo,
    BrokerAdapter,
    BrokerAuthError,
    BrokerError,
    BrokerNetworkError,
    BrokerRejectedError,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TigerCredentials,
    redact_credentials,
)


# ════════════════════════════════════════════════════════════
# SDK lazy import (跟 alpaca_adapter 一个模式)
# ════════════════════════════════════════════════════════════

def _import_tiger():
    """Return the bits of tigeropen we need, or raise BrokerError on missing dep."""
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
        from tigeropen.common.consts import Language
        from tigeropen.trade.domain.order import Order
        return {
            "TigerOpenClientConfig": TigerOpenClientConfig,
            "TradeClient": TradeClient,
            "Language": Language,
            "Order": Order,
        }
    except ImportError as e:
        raise BrokerError(
            "缺少 tigeropen 依赖,请运行: pip install tigeropen>=3.2.0,<4.0.0"
        ) from e


# Tiger 订单状态 → 我们的统一状态
_STATUS_MAP = {
    "Initial": OrderStatus.NEW,
    "PendingNew": OrderStatus.NEW,
    "New": OrderStatus.NEW,
    "Held": OrderStatus.NEW,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELED,
    "PendingCancel": OrderStatus.NEW,
    "Inactive": OrderStatus.CANCELED,
    "Rejected": OrderStatus.REJECTED,
    "Expired": OrderStatus.EXPIRED,
    "Replaced": OrderStatus.NEW,
}


def _map_status(raw) -> OrderStatus:
    if raw is None:
        return OrderStatus.NEW
    s = str(raw)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return _STATUS_MAP.get(s, OrderStatus.NEW)


def _strip_pem_markers(pem: str) -> str:
    """
    tigeropen's signature_utils.load_private_key does `base64.b64decode(private_key)`
    directly on its input — so it expects RAW BASE64, NOT a full PEM string.
    Feeding it `-----BEGIN...` lines causes "Incorrect padding" errors.

    This helper strips:
      - any line containing BEGIN / END
      - all whitespace and newlines (b64decode tolerates whitespace but we be tidy)
    Returns just the base64 body as a single string.
    """
    body_lines = [
        ln for ln in pem.strip().splitlines()
        if "BEGIN" not in ln and "END" not in ln
    ]
    return "".join(body_lines).strip()


# ════════════════════════════════════════════════════════════
# TigerAdapter
# ════════════════════════════════════════════════════════════

class TigerAdapter(BrokerAdapter):
    name = "tiger-paper"

    def __init__(self, credentials: TigerCredentials):
        if not isinstance(credentials, TigerCredentials):
            raise BrokerError(
                f"TigerAdapter requires TigerCredentials, got {type(credentials).__name__}"
            )
        self.tiger_id = credentials.tiger_id
        # NOTE: private_key stored in-memory only; never logged or stringified.
        self._private_key = credentials.private_key
        self.account = credentials.account
        self.license = credentials.license
        self._client = None
        self._sdk = None

    def is_configured(self) -> bool:
        return bool(self.tiger_id and self._private_key and self.account)

    def __repr__(self) -> str:
        # Defensive: ensure private_key never appears via default dataclass-like repr.
        return f"<TigerAdapter tiger_id={self.tiger_id!r} account={self.account!r}>"

    # ── client construction (lazy) ──────────────────────────

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.is_configured():
            raise BrokerAuthError(
                "Tiger 凭证未配置完整 (tiger_id / private_key / account 必须都有)"
            )
        self._sdk = _import_tiger()
        try:
            config = self._sdk["TigerOpenClientConfig"]()
            # IMPORTANT: tigeropen expects raw base64 here, not full PEM.
            # If we passed the PEM verbatim, signature_utils.load_private_key()
            # calls base64.b64decode() on it directly and explodes with
            # "Incorrect padding" because BEGIN/END lines aren't base64.
            config.private_key = _strip_pem_markers(self._private_key)
            config.tiger_id = self.tiger_id
            config.account = self.account
            config.license = self.license
            config.language = self._sdk["Language"].zh_CN
            self._client = self._sdk["TradeClient"](config)
        except Exception as e:
            raise BrokerAuthError(f"Tiger 客户端初始化失败: {type(e).__name__}") from e
        return self._client

    def ping(self) -> bool:
        try:
            self.get_account()
            return True
        except Exception:
            return False

    # ── 账户 ────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        client = self._ensure_client()
        try:
            assets = client.get_assets(account=self.account)
        except Exception as e:
            raise self._wrap_error(e) from e

        # tigeropen 的 get_assets 返回 list[PortfolioAccount]
        if not assets:
            raise BrokerNetworkError("Tiger get_assets 返回空")
        acct = assets[0]
        # SDK 内部用 summary 字段;某些版本路径不同,我们用 getattr 防御
        summary = getattr(acct, "summary", acct)

        cash = float(getattr(summary, "cash", 0) or 0)
        buying_power = float(getattr(summary, "buying_power", cash) or cash)
        gross_position_value = float(getattr(summary, "gross_position_value", 0) or 0)
        equity = cash + gross_position_value

        return AccountInfo(
            cash=cash,
            buying_power=buying_power,
            equity=equity,
            currency=str(getattr(summary, "currency", "USD") or "USD"),
            account_id=self.account,
            status=str(getattr(summary, "status", "ACTIVE") or "ACTIVE"),
            raw={"backend": "tiger", "license": self.license},
        )

    def list_positions(self) -> List[Position]:
        client = self._ensure_client()
        try:
            positions = client.get_positions(account=self.account)
        except Exception as e:
            raise self._wrap_error(e) from e

        out: List[Position] = []
        for p in positions or []:
            symbol = getattr(p, "contract", None)
            sym_str = getattr(symbol, "symbol", None) or str(symbol)
            qty = float(getattr(p, "quantity", 0) or 0)
            avg = float(getattr(p, "average_cost", 0) or 0)
            market_value = float(getattr(p, "market_value", qty * avg) or (qty * avg))
            unrealized = float(getattr(p, "unrealized_pnl", market_value - qty * avg) or 0)
            cost = qty * avg
            pct = (unrealized / cost * 100) if cost > 0 else 0.0
            out.append(Position(
                symbol=sym_str,
                qty=qty,
                avg_entry_price=avg,
                market_value=market_value,
                unrealized_pl=unrealized,
                unrealized_pl_pct=pct,
                current_price=float(getattr(p, "market_price", avg) or avg),
            ))
        return out

    # ── 订单 ────────────────────────────────────────────────

    def place_order(self, intent: OrderIntent) -> OrderResult:
        client = self._ensure_client()
        if intent.order_type != OrderType.LIMIT:
            # CLAUDE.md trading safety: 市价单默认禁(滑点保护)
            raise BrokerRejectedError("TigerAdapter 默认仅支持限价单 (market 已被风控禁)")
        try:
            # 1) create_order 在本地组装一个 Order 对象 (不发请求)
            order = client.create_order(
                account=self.account,
                contract=client.get_contract(symbol=intent.symbol),
                action="BUY" if intent.side == OrderSide.BUY else "SELL",
                order_type="LMT",
                quantity=int(intent.qty),
                limit_price=float(intent.limit_price or 0),
            )
            # 2) place_order 发请求,**返回 Optional[int]** = broker_order_id
            order_id = client.place_order(order)
        except Exception as e:
            raise self._wrap_error(e) from e

        if order_id is None:
            raise BrokerNetworkError("Tiger place_order 返回 None (未拿到 broker_order_id)")

        # Tiger 在 place_order 后会把 id 回写到 order 对象,但 filled/status 还未更新。
        # 我们只信任明确拿到的 broker_order_id;其余字段读 order 上的当前快照,
        # 后续 get_order(broker_order_id) 可以拿到最新成交状态。
        return OrderResult(
            broker_order_id=str(order_id),
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=float(intent.qty),
            filled_qty=float(getattr(order, "filled", 0) or 0),
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            status=_map_status(getattr(order, "status", "New")),
            filled_avg_price=getattr(order, "avg_fill_price", None) or None,
            submitted_at=str(getattr(order, "order_time", "") or ""),
            raw={"backend": "tiger"},
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        client = self._ensure_client()
        try:
            client.cancel_order(account=self.account, id=int(broker_order_id))
            return True
        except Exception as e:
            raise self._wrap_error(e) from e

    def get_order(self, broker_order_id: str) -> OrderResult:
        client = self._ensure_client()
        try:
            order = client.get_order(account=self.account, id=int(broker_order_id))
        except Exception as e:
            raise self._wrap_error(e) from e
        if order is None:
            raise BrokerError(f"Tiger 未找到订单 id={broker_order_id}")
        return self._to_order_result(order)

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[OrderResult]:
        client = self._ensure_client()
        try:
            orders = client.get_orders(account=self.account, limit=limit)
        except Exception as e:
            raise self._wrap_error(e) from e
        results = [self._to_order_result(o) for o in (orders or [])]
        if status and status != "all":
            results = [r for r in results if r.status.value == status]
        return results

    # ── helpers ─────────────────────────────────────────────

    def _to_order_result(self, order) -> OrderResult:
        contract = getattr(order, "contract", None)
        sym = getattr(contract, "symbol", None) or str(contract)
        action = str(getattr(order, "action", "BUY"))
        side = OrderSide.BUY if "BUY" in action.upper() else OrderSide.SELL
        otype = OrderType.LIMIT if str(getattr(order, "order_type", "LMT")).upper().startswith("LMT") else OrderType.MARKET
        return OrderResult(
            broker_order_id=str(getattr(order, "id", "") or getattr(order, "order_id", "")),
            intent_id=None,
            symbol=sym,
            side=side,
            qty=float(getattr(order, "quantity", 0) or 0),
            # NOTE: Tiger SDK 的字段叫 `filled` (不是 `filled_quantity`)
            filled_qty=float(getattr(order, "filled", 0) or 0),
            order_type=otype,
            limit_price=getattr(order, "limit_price", None),
            status=_map_status(getattr(order, "status", "New")),
            filled_avg_price=getattr(order, "avg_fill_price", None) or None,
            submitted_at=str(getattr(order, "order_time", "") or ""),
            raw={"backend": "tiger"},
        )

    def _wrap_error(self, exc: Exception) -> BrokerError:
        """
        Categorize a Tiger SDK exception while preserving the message for
        debugging. Credential-shaped content (PEM blocks, long base64 blobs)
        is redacted in the wrapped message so private-key bytes never leak.

        Callers should `raise self._wrap_error(e) from e` so the original
        traceback (including the un-redacted local frame, not the wrapped
        message) is preserved for in-process debugging while the public
        exception remains safe.
        """
        name = type(exc).__name__
        msg = redact_credentials(str(exc))
        low = msg.lower()
        if ("sign" in low or "auth" in low or "permission" in low
                or "private key" in low or "invalid token" in low):
            return BrokerAuthError(f"Tiger {name}: {msg}")
        if ("timeout" in low or "network" in low or "connect" in low
                or "name resolution" in low or "unreachable" in low):
            return BrokerNetworkError(f"Tiger {name}: {msg}")
        if ("reject" in low or "insufficient" in low or "not eligible" in low
                or "buying power" in low or "not tradable" in low):
            return BrokerRejectedError(f"Tiger {name}: {msg}")
        return BrokerError(f"Tiger {name}: {msg}")
