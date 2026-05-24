"""
BrokerRegistry —— 每用户、每 broker 的 adapter 工厂 + 缓存(X2 引入)。

ADR-0001 第 2、4 节落地:
  - 调用 `BrokerRegistry.get(user_id, broker_type, label=None)` 拿 adapter
  - 缓存命中直接返回(60 秒 TTL,本 commit 暂不计时,下个 commit 接 invalidate hook)
  - 凭证当前从 env 取(transitional);X3 会换成 credentials_store
  - adapter 缓存 key 为 (user_id, broker_type, label or "")

线程安全说明:
  Python GIL 保证 dict 的 set/pop 原子,X2 不加显式锁;
  双重创建(同一 key 几乎同时被两个线程请求)的代价是多构造一个 adapter,
  最后一个 set 胜出 —— 行为正确,只是浪费一次构造,X3 评估是否要加锁。

调用约定:
  - 业务代码用 `get_current_broker(broker_type?, label?)`,user_id 自动从
    thread-local 取(`quant_agent._get_request_device_id()`)
  - 显式 user_id 的场景(后台任务、测试)直接调 `_registry.get(...)`
"""
from __future__ import annotations

import os
from typing import Optional

from .base import (
    BrokerAdapter,
    BrokerError,
    Credentials,
    MockCredentials,
    AlpacaCredentials,
)


# ════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════

class BrokerRegistry:
    def __init__(self) -> None:
        self._cache: dict[tuple, BrokerAdapter] = {}

    def get(
        self,
        user_id: str,
        broker_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> BrokerAdapter:
        broker_type = (broker_type or _default_broker_type()).lower()
        cache_key = (user_id or "default", broker_type, label or "")
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        creds = _load_credentials_transitional(broker_type)
        adapter = _build_adapter(creds)
        self._cache[cache_key] = adapter
        return adapter

    def invalidate(
        self,
        user_id: str,
        broker_type: str,
        label: Optional[str] = None,
    ) -> None:
        """X3 钩子:credentials_store 更新/删除凭证后必须调它。"""
        self._cache.pop((user_id or "default", broker_type.lower(), label or ""), None)

    def clear(self) -> None:
        """测试 / 热重载用。生产路径不应调。"""
        self._cache.clear()


# 模块级单例
_registry = BrokerRegistry()


# ════════════════════════════════════════════════════════════
# 上层便捷入口
# ════════════════════════════════════════════════════════════

def get_current_broker(
    broker_type: Optional[str] = None,
    label: Optional[str] = None,
) -> BrokerAdapter:
    """
    解析当前请求上下文(thread-local device_id)→ adapter。
    X2 commit 2 把所有 `get_broker()` 调用点迁到这里。
    """
    user_id = _resolve_current_user_id()
    return _registry.get(user_id, broker_type=broker_type, label=label)


def _resolve_current_user_id() -> str:
    """
    从 quant_agent 的 thread-local 取 device_id;失败回落到 'default'。
    "default" 保证脚本 / 测试场景仍能工作(行为与旧 `get_broker()` 等价)。
    """
    try:
        from quant_agent import _get_request_device_id  # circular import 防御:延迟导入
        return _get_request_device_id() or "default"
    except Exception:
        return "default"


# ════════════════════════════════════════════════════════════
# Transitional 凭证加载(X3 替换)
# ════════════════════════════════════════════════════════════

def _default_broker_type() -> str:
    return (os.getenv("BROKER_MODE", "mock") or "mock").strip().lower()


def _load_credentials_transitional(broker_type: str) -> Credentials:
    """
    X2 过渡版本:从 env 拿凭证(与旧 get_broker 等价)。
    X3 会替换成 `credentials_store.load(user_id, broker_type, label)`。
    """
    if broker_type in ("mock", "virtual", "paper-mock"):
        return MockCredentials(
            initial_cash=float(
                (os.getenv("MOCK_INITIAL_CASH", "100000") or "100000").strip()
            ),
        )
    if broker_type in ("alpaca", "alpaca-paper"):
        return AlpacaCredentials(
            api_key=os.getenv("ALPACA_API_KEY", "").strip(),
            api_secret=os.getenv("ALPACA_API_SECRET", "").strip(),
            base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ).strip(),
        )
    raise BrokerError(f"Unknown broker_type: {broker_type!r}")


def _build_adapter(creds: Credentials) -> BrokerAdapter:
    """Credentials → Adapter 的分派。新 broker 接入时(X4 Tiger)在这里加分支。"""
    if isinstance(creds, MockCredentials):
        from .mock_adapter import MockAdapter
        return MockAdapter(creds)
    if isinstance(creds, AlpacaCredentials):
        from .alpaca_adapter import AlpacaAdapter
        return AlpacaAdapter(creds)
    raise BrokerError(f"No adapter for credentials type: {type(creds).__name__}")
