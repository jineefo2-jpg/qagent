"""
brokers/_db.py — schema application + idempotency.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    # Filter out SQLite auto-indexes (sqlite_autoindex_*)
    return {row[0] for row in cur.fetchall() if not row[0].startswith("sqlite_")}


def test_init_creates_both_tables(tmp_path):
    from brokers import _db

    db = tmp_path / "test.db"
    conn = _db.init(db_path=db)
    tables = _table_names(conn)

    assert "broker_bindings" in tables
    assert "broker_audit_log" in tables
    conn.close()


def test_init_creates_expected_indexes(tmp_path):
    from brokers import _db

    conn = _db.init(db_path=tmp_path / "test.db")
    idx = _index_names(conn)

    assert "idx_audit_user_ts" in idx
    assert "idx_audit_ts" in idx
    assert "idx_bindings_user" in idx
    conn.close()


def test_init_is_idempotent(tmp_path):
    """Calling init() twice on the same path must not error."""
    from brokers import _db

    db = tmp_path / "test.db"
    conn1 = _db.init(db_path=db)
    # Insert a row, then re-init — schema script must not wipe it.
    conn1.execute(
        "INSERT INTO broker_audit_log (ts, actor, action, success) "
        "VALUES (?, ?, ?, ?)",
        (1000, "system", "rotate", 1),
    )
    conn1.close()

    conn2 = _db.init(db_path=db)
    rows = conn2.execute("SELECT COUNT(*) FROM broker_audit_log").fetchone()
    assert rows[0] == 1  # IF NOT EXISTS preserved data
    conn2.close()


def test_wal_journal_mode(tmp_path):
    from brokers import _db

    conn = _db.init(db_path=tmp_path / "test.db")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    conn.close()


def test_check_constraint_rejects_invalid_actor(tmp_path):
    from brokers import _db

    conn = _db.init(db_path=tmp_path / "test.db")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO broker_audit_log (ts, actor, action, success) "
            "VALUES (?, ?, ?, ?)",
            (1000, "hacker", "bind", 1),  # invalid actor
        )
    conn.close()


def test_unique_user_broker_label(tmp_path):
    from brokers import _db

    conn = _db.init(db_path=tmp_path / "test.db")
    conn.execute(
        """INSERT INTO broker_bindings
        (user_id, broker_type, label, encrypted_credential, dek_wrapped, kek_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("u_42", "tiger", "main", b"x", b"y", 1, 1000),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO broker_bindings
            (user_id, broker_type, label, encrypted_credential, dek_wrapped, kek_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("u_42", "tiger", "main", b"x", b"y", 1, 2000),  # same (user, type, label)
        )
    conn.close()


def test_explicit_path_does_not_pollute_default(tmp_path):
    """Tests using db_path must not touch the production-pathed singleton."""
    from brokers import _db

    _db.close_default()  # ensure clean slate
    conn = _db.init(db_path=tmp_path / "test.db")
    assert _db._default_conn is None  # singleton untouched
    conn.close()
