"""因子库。注册层在 `base`；具体因子实现（price / fundamental / flow / risk）后续任务补。"""
from __future__ import annotations

from .base import FACTOR_REGISTRY, FactorSpec, factor, get_factor, list_factors

__all__ = ["FACTOR_REGISTRY", "FactorSpec", "factor", "get_factor", "list_factors"]
