"""策略层（P3）：宏观择时 + 清单生成。一切取数经 ashare.data.query（D1/D2），
本包无任何 DML（写库只发生在 CLI 入口经 ledger_store，分层检查 L4 强制）。"""
from __future__ import annotations

from .macro import macro_indicators, macro_score  # noqa: F401


def build_rebalance_plan(as_of_date, config) -> dict:
    """§6.3 调仓清单契约（架构 §4.4 占位，P3 Task 3 实现；JSON 必须带
    param_hash 与 data_snapshot_id 双指纹）。"""
    raise NotImplementedError("P3 Task 3 落地前不可调用")
