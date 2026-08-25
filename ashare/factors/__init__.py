"""因子库。注册层在 `base`，四个因子模块在 `price` / `fundamental` / `flow` / `risk`。

★ 下面那行 import 是整个平台最便宜、也最容易被"顺手清理掉"的一道保险，别删：
  `@factor` **只在模块被导入时**才注册。少一个模块 → `FACTOR_REGISTRY` 里就少一类因子，
  而缺因子是【静默失败】—— combine 拿不到值 → 合成分数全 NaN → build_targets 返回空
  → 净值一条直线。读起来像"策略这段时间没信号"，而不是"代码没装配"。
  `tests/ashare/test_factors_flow_risk.py` 在【子进程】里钉住了这件事（本进程里别的
  测试模块早就直接 import 过 price，在本进程断言等于把最需要保护的那件事测掉）。
"""
from __future__ import annotations

from .base import (FACTOR_REGISTRY, FactorSpec, combine, compute_factor, compute_panel,
                   factor, get_factor, list_factors)
from . import price, fundamental, flow, risk    # noqa: F401  —— 导入即注册，见上

# ★ 三个计算入口必须出现在这里：包的公开面就是下一个人照抄的那条路。
#   只导出 get_factor 的话，`get_factor(n).fn(as_of, universe)` 是一行合法的公开写法，
#   而它一次绕过【四】道闸：_checked_universe（18 个因子唯一的校验点）、
#   reindex(codes)（因子契约允许返回子集，少几行 = 落库写短行）、
#   spec.default_params（"缓存写着 window=5、内容是 99"，而 param_hash 是 factor_value 主键）、
#   available_from 短路。走 compute_factor 才有这四道。
__all__ = ["FACTOR_REGISTRY", "FactorSpec", "factor", "get_factor", "list_factors",
           "compute_factor", "compute_panel", "combine",
           "price", "fundamental", "flow", "risk"]
