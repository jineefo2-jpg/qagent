"""
brokers — 券商接入层

设计目标:
  - 不同券商(Alpaca / Mock / Tiger / IBKR)实现同一个 BrokerAdapter 接口
  - 上层(quant_agent / server)用 `brokers.registry.get_current_broker()` 拿 adapter
  - 凭证由 BrokerRegistry 注入(X2 暂从 env;X3 起从 credentials_store)
"""
from typing import Optional
from .base import (
    BrokerAdapter,
    BrokerError,
    BrokerAuthError,
    BrokerNetworkError,
    BrokerRejectedError,
    Credentials,
    MockCredentials,
    AlpacaCredentials,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    AccountInfo,
)

__all__ = [
    "BrokerAdapter",
    "BrokerError",
    "BrokerAuthError",
    "BrokerNetworkError",
    "BrokerRejectedError",
    "Credentials",
    "MockCredentials",
    "AlpacaCredentials",
    "OrderIntent",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Position",
    "AccountInfo",
    "get_broker",
]


def get_broker(name: Optional[str] = None) -> BrokerAdapter:
    """
    [Transitional shim · X2]
    向后兼容入口。X2 commit 2 会把所有调用点迁到 `brokers.registry.get_current_broker`,
    之后此函数会加 DeprecationWarning(X2 commit 2)和最终删除(后续版本)。

    行为等价于旧版:按当前 thread-local 用户 + BROKER_MODE env 解析 adapter。
    """
    from .registry import get_current_broker
    return get_current_broker(broker_type=name)
