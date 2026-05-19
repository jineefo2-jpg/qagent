"""
brokers — 券商接入层

设计目标：
  - 不同券商（Alpaca / 富途 / IBKR / QMT）实现同一个 BrokerAdapter 接口
  - 上层（quant_agent / server）只依赖接口，不依赖具体券商
  - 当前 Phase 0 只实现 Alpaca paper trading
"""
from .base import (
    BrokerAdapter,
    BrokerError,
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
    "OrderIntent",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Position",
    "AccountInfo",
    "get_broker",
]


def get_broker(name: str = "alpaca") -> BrokerAdapter:
    """
    工厂方法：按名字返回 BrokerAdapter 实例。
    后续接入富途/IBKR 时在这里加分支即可。
    """
    name = (name or "").lower()
    if name in ("alpaca", "alpaca-paper", ""):
        from .alpaca_adapter import AlpacaAdapter
        return AlpacaAdapter()
    raise BrokerError(f"未知 broker: {name}")
