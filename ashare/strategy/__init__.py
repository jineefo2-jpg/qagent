"""策略层（P3）：宏观择时 + 清单生成。一切取数经 ashare.data.query（D1/D2），
本包无任何 DML（写库只发生在 CLI 入口经 ledger_store，分层检查 L4 强制）。"""
from __future__ import annotations

from .macro import macro_indicators, macro_score, position_for  # noqa: F401
from .plan import build_rebalance_plan  # noqa: F401
