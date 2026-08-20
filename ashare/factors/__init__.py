"""因子库。注册层在 `base`，四个因子模块在 `price` / `fundamental` / `flow` / `risk`。

★ 下面那行 import 是整个平台最便宜、也最容易被"顺手清理掉"的一道保险，别删：
  `@factor` **只在模块被导入时**才注册。少一个模块 → `FACTOR_REGISTRY` 里就少一类因子，
  而缺因子是【静默失败】—— combine 拿不到值 → 合成分数全 NaN → build_targets 返回空
  → 净值一条直线。读起来像"策略这段时间没信号"，而不是"代码没装配"。
  `tests/ashare/test_factors_flow_risk.py` 在【子进程】里钉住了这件事（本进程里别的
  测试模块早就直接 import 过 price，在本进程断言等于把最需要保护的那件事测掉）。
"""
from __future__ import annotations

from .base import FACTOR_REGISTRY, FactorSpec, factor, get_factor, list_factors
from . import price, fundamental, flow, risk    # noqa: F401  —— 导入即注册，见上

__all__ = ["FACTOR_REGISTRY", "FactorSpec", "factor", "get_factor", "list_factors",
           "price", "fundamental", "flow", "risk"]
