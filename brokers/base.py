"""
BrokerAdapter 抽象基类 + 通用数据模型。

所有具体券商实现（Alpaca / 富途 / IBKR / QMT）都遵循这套接口，
让 Agent 层与具体券商解耦。
"""
from __future__ import annotations

import abc
import enum
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ════════════════════════════════════════════════════════════
# Credentials —— 凭证类型层(X2 引入)
# ════════════════════════════════════════════════════════════
# 设计原则:
#   1. 类型安全(每个 broker 一个子类,字段明确)
#   2. 不可变(frozen=True),防止运行时偷改 api_key
#   3. 不实现 __repr__/__str__ 时的脱敏 —— 我们靠"绝不让凭证进日志"
#      的工程纪律来保证,而不是依赖 dataclass 的默认 repr
# X3 会在 credentials_store.py 里加密存储这些对象的序列化结果。

@dataclass(frozen=True)
class Credentials:
    """所有 broker 凭证子类的抽象基类。子类必须设 broker_type。"""
    broker_type: str = ""


@dataclass(frozen=True)
class MockCredentials(Credentials):
    """MockAdapter 凭证。Mock 没有真实凭证,只是配置项。"""
    broker_type: str = "mock"
    initial_cash: float = 100000.0


@dataclass(frozen=True)
class AlpacaCredentials(Credentials):
    """Alpaca paper trading 凭证。base_url 默认 paper —— live URL 由 CLAUDE.md 安全规则禁止。"""
    broker_type: str = "alpaca"
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"


# ════════════════════════════════════════════════════════════
# Credential redaction (shared by adapter _wrap_error implementations)
# ════════════════════════════════════════════════════════════

import re as _re

# Match an entire PEM block (BEGIN ... END) including newlines in between.
_PEM_BLOCK_RE = _re.compile(
    r"-----BEGIN[^-\n]{0,40}-----.*?-----END[^-\n]{0,40}-----",
    _re.DOTALL,
)
# Match long contiguous base64-looking blobs (50+ chars of base64 alphabet).
# Catches stray RSA bodies that escaped a PEM block.
_BASE64_BLOB_RE = _re.compile(r"[A-Za-z0-9+/=]{50,}")


def redact_credentials(text: str, max_len: int = 400) -> str:
    """
    Sanitize a string before it goes into an exception message / log / response.

    Removes:
      - any -----BEGIN ... -----END----- PEM block (RSA, EC, etc.)
      - any 50+ char base64-looking run (catches naked key bodies)

    Truncates to `max_len` to bound exception size.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _PEM_BLOCK_RE.sub("[REDACTED-PEM]", text)
    text = _BASE64_BLOB_RE.sub("[REDACTED-BASE64]", text)
    if len(text) > max_len:
        text = text[:max_len] + "...[truncated]"
    return text


@dataclass(frozen=True)
class TigerCredentials(Credentials):
    """
    Tiger Brokers 凭证 (老虎证券)。

    Tiger 不支持 OAuth,只能用 RSA 私钥认证。整套凭证由用户在
    https://www.itigerup.com/openapi/ 生成密钥对后上传到本应用 (ADR-0001 §5)。

    字段:
      tiger_id:     开发者账号 ID (类似 "20151024")
      private_key:  PEM 格式 RSA 私钥(整段,含 BEGIN/END 头尾)
      account:      交易账户号(模拟账户以 'U' 开头)
      license:      持牌主体 (TBNZ/TBAU/TBSG/TBHK),默认 TBNZ(新西兰,覆盖最广)

    paper/live 的区分**不在这里**,在 broker_bindings.env 列。本应用当前只
    允许 env='paper'(对应 Tiger 的"环球账户模拟")。
    """
    broker_type: str = "tiger"
    tiger_id: str = ""
    private_key: str = ""
    account: str = ""
    license: str = "TBNZ"


# ════════════════════════════════════════════════════════════
# 异常
# ════════════════════════════════════════════════════════════

class BrokerError(Exception):
    """所有 broker 操作异常的基类"""


class BrokerAuthError(BrokerError):
    """凭证缺失或无效"""


class BrokerNetworkError(BrokerError):
    """网络 / 超时"""


class BrokerRejectedError(BrokerError):
    """券商拒绝订单（资金不足 / 标的不可交易 等）"""


# ════════════════════════════════════════════════════════════
# 枚举
# ════════════════════════════════════════════════════════════

class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, enum.Enum):
    NEW = "new"                 # 已提交，未成交
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING = "pending"         # 已生成 intent，未确认


# ════════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════════

@dataclass
class OrderIntent:
    """
    订单意图 —— Agent 生成、用户确认前的"待办订单"。
    intent_id 是我们自己造的；只有用户确认后才会发到券商拿到真实 order_id。
    """
    intent_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    notes: str = ""               # Agent 给的下单理由
    estimated_cost: Optional[float] = None  # qty * limit_price，便于前端展示
    created_at: float = field(default_factory=time.time)

    @classmethod
    def new(
        cls,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        notes: str = "",
    ) -> "OrderIntent":
        try:
            side_enum = OrderSide(side.lower())
            type_enum = OrderType(order_type.lower())
        except ValueError as e:
            raise BrokerError(f"无效参数: {e}") from e

        if type_enum == OrderType.LIMIT and limit_price is None:
            raise BrokerError("限价单必须提供 limit_price")
        if qty <= 0:
            raise BrokerError("qty 必须为正数")

        est = (qty * limit_price) if limit_price else None
        return cls(
            intent_id=f"int_{uuid.uuid4().hex[:12]}",
            symbol=symbol.upper(),
            side=side_enum,
            qty=qty,
            order_type=type_enum,
            limit_price=limit_price,
            notes=notes,
            estimated_cost=est,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        return d


@dataclass
class OrderResult:
    """券商真正下单后返回的订单状态"""
    broker_order_id: str
    intent_id: Optional[str]
    symbol: str
    side: OrderSide
    qty: float
    filled_qty: float
    order_type: OrderType
    limit_price: Optional[float]
    status: OrderStatus
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[str] = None
    raw: dict = field(default_factory=dict)  # 保留原始返回，便于排查

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        d["status"] = self.status.value
        # raw 字段可能很大，前端不需要
        d.pop("raw", None)
        return d


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    current_price: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AccountInfo:
    cash: float
    buying_power: float
    equity: float               # 净值（现金 + 持仓市值）
    currency: str = "USD"
    account_id: str = ""
    status: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


# ════════════════════════════════════════════════════════════
# BrokerAdapter 抽象基类
# ════════════════════════════════════════════════════════════

class BrokerAdapter(abc.ABC):
    """
    所有券商适配器的统一接口。
    实现类应该 raise BrokerAuthError / BrokerNetworkError / BrokerRejectedError，
    便于上层做友好提示。
    """

    name: str = "abstract"

    # ── 元信息 ──
    @abc.abstractmethod
    def is_configured(self) -> bool:
        """凭证是否齐全（不发实际请求）"""

    @abc.abstractmethod
    def ping(self) -> bool:
        """连通性 + 鉴权校验（轻量请求）"""

    # ── 账户 ──
    @abc.abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abc.abstractmethod
    def list_positions(self) -> List[Position]: ...

    # ── 订单 ──
    @abc.abstractmethod
    def place_order(self, intent: OrderIntent) -> OrderResult:
        """真正提交订单到券商。调用方必须先做风控。"""

    @abc.abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool: ...

    @abc.abstractmethod
    def get_order(self, broker_order_id: str) -> OrderResult: ...

    @abc.abstractmethod
    def list_orders(self, status: Optional[str] = None,
                     limit: int = 50) -> List[OrderResult]: ...
