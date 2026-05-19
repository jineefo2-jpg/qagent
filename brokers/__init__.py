"""
brokers — 券商接入层

设计目标：
  - 不同券商（Alpaca / Mock / 富途 / IBKR / QMT）实现同一个 BrokerAdapter 接口
  - 上层（quant_agent / server）只依赖接口
  - 默认 BROKER_MODE=mock，每个用户独立虚拟账户
  - 设 BROKER_MODE=alpaca 可切回真实 Alpaca paper 账户（所有用户共享一个）
"""
from typing import Optional
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


def get_broker(name: Optional[str] = None) -> BrokerAdapter:
    """
    工厂方法。
    - 不传 name 时按环境变量 BROKER_MODE 选择，默认 'mock'（虚拟账户）
    - name='mock' → MockAdapter（纯本地虚拟交易，多用户隔离）
    - name='alpaca' → AlpacaAdapter（真实 Alpaca paper 账户，所有用户共享一个账户）
    """
    import os as _os
    if not name:
        name = _os.getenv("BROKER_MODE", "mock").strip().lower() or "mock"
    name = name.lower()

    if name in ("mock", "virtual", "paper-mock"):
        from .mock_adapter import MockAdapter
        return MockAdapter()
    if name in ("alpaca", "alpaca-paper"):
        from .alpaca_adapter import AlpacaAdapter
        return AlpacaAdapter()
    raise BrokerError(f"未知 broker: {name}")


