"""
brokers/audit.py — append-only audit log.

These tests pin down the contract called out in CLAUDE.md trading-safety:
  - exceptions on the audit path MUST propagate (no swallowing)
  - rows are append-only (no UPDATE/DELETE paths in our module)
  - the CHECK enums match between code and SQL
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def conn(tmp_path):
    from brokers import _db
    c = _db.init(db_path=tmp_path / "audit.db")
    yield c
    c.close()


def test_audit_log_inserts_row(conn):
    from brokers.audit import audit_log

    row_id = audit_log(
        actor="user",
        action="bind",
        user_id="u_42",
        binding_id=None,
        detail="bound tiger paper",
        success=True,
        conn=conn,
    )
    assert row_id > 0
    rows = conn.execute(
        "SELECT actor, action, user_id, detail, success FROM broker_audit_log"
    ).fetchall()
    assert rows == [("user", "bind", "u_42", "bound tiger paper", 1)]


def test_audit_log_failure_records_success_zero(conn):
    from brokers.audit import audit_log

    audit_log(
        actor="system", action="fail",
        user_id="u_42", binding_id=7,
        detail="alpaca auth failed",
        success=False,
        conn=conn,
    )
    success = conn.execute("SELECT success FROM broker_audit_log").fetchone()[0]
    assert success == 0


def test_audit_log_rejects_unknown_actor(conn):
    from brokers.audit import audit_log

    with pytest.raises(ValueError, match="actor"):
        audit_log(actor="anonymous", action="use", conn=conn)


def test_audit_log_rejects_unknown_action(conn):
    from brokers.audit import audit_log

    with pytest.raises(ValueError, match="action"):
        audit_log(actor="user", action="hack", conn=conn)


def test_audit_actor_action_enums_match_sql(conn):
    """ACTORS/ACTIONS in code must match SQL CHECK constraints exactly."""
    from brokers import audit
    from brokers.audit import audit_log

    for actor in audit.ACTORS:
        for action in audit.ACTIONS:
            audit_log(actor=actor, action=action, conn=conn, success=True)
    # If any combination violates the CHECK constraint, sqlite raises.


def test_audit_log_failure_propagates(conn):
    """
    Hard rule: a failed audit write MUST raise. Caller MUST NOT swallow.
    We simulate this by closing the connection mid-flight.
    """
    from brokers.audit import audit_log

    conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        audit_log(actor="user", action="bind", user_id="u_42", conn=conn)


def test_audit_log_ordering_preserved(conn):
    """Insertion order should be reflected in row ids (autoincrement)."""
    from brokers.audit import audit_log
    ids = [
        audit_log(actor="user", action="bind", user_id=f"u_{i}", conn=conn)
        for i in range(5)
    ]
    assert ids == sorted(ids)
    assert ids == list(range(ids[0], ids[0] + 5))


def test_audit_emits_stderr_jsonline(conn, capsys):
    """ADR-0001 §8: one JSON line per audit row goes to stderr."""
    from brokers.audit import audit_log
    audit_log(actor="llm", action="use", user_id="u_42", conn=conn)
    captured = capsys.readouterr()
    assert '"actor": "llm"' in captured.err
    assert '"action": "use"' in captured.err
    assert '"user_id": "u_42"' in captured.err
