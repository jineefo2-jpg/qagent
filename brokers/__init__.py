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
    [Deprecated · X2 commit 2 onwards]
    向后兼容入口。生产代码已全部迁到 `brokers.registry.get_current_broker`。
    保留此函数仅为防止外部脚本/旧分支断裂;新代码不要使用。

    行为等价于旧版:按当前 thread-local 用户 + BROKER_MODE env 解析 adapter。
    """
    import warnings
    warnings.warn(
        "brokers.get_broker() is deprecated. "
        "Use brokers.registry.get_current_broker() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .registry import get_current_broker
    return get_current_broker(broker_type=name)
