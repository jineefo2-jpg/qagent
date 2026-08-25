"""回测引擎。本任务只交付输入输出数据结构（`types`）；组合/成交/成本/指标/五闸见后续任务。"""
from __future__ import annotations

from .types import BacktestConfig, BacktestResult, CostConfig, PortfolioConstraints

__all__ = ["BacktestConfig", "BacktestResult", "CostConfig", "PortfolioConstraints"]
