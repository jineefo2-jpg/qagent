"""
Append-only audit log for the broker subsystem (ADR-0001 §3 + §8).

Hard rules from CLAUDE.md (Trading safety · Audit trail):
  - Audit writes MUST NOT be silenced by env flags or try/except.
  - A failed audit write is a code path that needs fixing, not swallowing.
  - Any row with `actor='llm' AND action ∈ {bind, unbind, rotate}` is a
    CRITICAL incident — the LLM may only ever trigger 'use'.

This module is intentionally tiny: one function, no batching, no buffering.
If write throughput becomes a problem (it won't at SaaS scale of dozens of
users), revisit then — not now.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

import sqlite3

from . import _db


# Public enums (kept as plain strings to match SQL CHECK constraints exactly)
ACTORS = frozenset({"user", "llm", "system", "rotation"})
ACTIONS = frozenset({"bind", "unbind", "use", "rotate", "fail", "read"})


def audit_log(
    actor: str,
    action: str,
    user_id: Optional[str] = None,
    binding_id: Optional[int] = None,
    detail: Optional[str] = None,
    success: bool = True,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Insert one append-only audit row. Returns the new row id.

    Also emits one line of structured JSON to stderr — that line doubles as
    a stream for external log aggregators (per ADR-0001 §8).

    Raises:
      ValueError       — actor or action is not in the enum
      sqlite3.Error    — DB write failed; caller MUST let this propagate
                         (no try/except wrap allowed in the broker subsystem)
    """
    if actor not in ACTORS:
        raise ValueError(f"audit: actor must be one of {ACTORS!r}, got {actor!r}")
    if action not in ACTIONS:
        raise ValueError(f"audit: action must be one of {ACTIONS!r}, got {action!r}")

    ts = int(time.time())
    success_int = 1 if success else 0

    c = conn or _db.init()
    cur = c.execute(
        """
        INSERT INTO broker_audit_log
            (ts, user_id, binding_id, actor, action, detail, success)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, user_id, binding_id, actor, action, detail, success_int),
    )
    row_id = cur.lastrowid

    # Stream-friendly structured log line (no secrets — detail is caller-controlled)
    _emit_stream({
        "ts": ts,
        "user_id": user_id,
        "binding_id": binding_id,
        "actor": actor,
        "action": action,
        "detail": detail,
        "success": bool(success),
        "row_id": row_id,
    })

    return row_id


def _emit_stream(payload: dict[str, Any]) -> None:
    """One-line JSON to stderr. Kept private; never called outside this module."""
    try:
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Stderr write failure is best-effort; the DB row is the source of truth.
        pass
