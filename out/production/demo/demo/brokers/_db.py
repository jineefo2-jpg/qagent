"""
SQLite connector for the broker credentials & audit store.

Design choices:
  - WAL journal mode for concurrent readers + one writer (small-SaaS scale).
  - `check_same_thread=False` because FastAPI handlers cross threads.
    SQLite itself serializes writes via its own mutex; we rely on that.
  - `isolation_level=None` → autocommit. We don't run multi-statement
    transactions yet; if X3 c2 needs them, add a small `transaction()` ctx
    manager here.
  - Schema is applied via `executescript(_schema.sql)`, which is idempotent
    thanks to `IF NOT EXISTS` everywhere — no migration runner needed yet.

Tests must call `init(db_path=<tmp>)` with their own path; production code
calls `init()` (no arg) which uses the module-global connection.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "brokers.db"
_SCHEMA_PATH = _MODULE_DIR / "_schema.sql"

_lock = threading.RLock()
_default_conn: Optional[sqlite3.Connection] = None


def init(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection with schema applied.

    - No arg → singleton connection on `data/brokers.db`, created on first
      call. Subsequent calls return the same connection.
    - Explicit `db_path` → fresh connection on the given path (used by tests
      and tools). Caller owns its lifetime.
    """
    if db_path is None:
        return _default_connection()
    return _open(db_path)


def _default_connection() -> sqlite3.Connection:
    global _default_conn
    with _lock:
        if _default_conn is not None:
            return _default_conn
        _default_conn = _open(_DEFAULT_DB_PATH)
        return _default_conn


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,           # autocommit
        check_same_thread=False,        # FastAPI threads
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """
    Forward-only schema migrations for existing databases. New columns
    declared in _schema.sql don't land via CREATE TABLE IF NOT EXISTS,
    so we ALTER TABLE ADD COLUMN here, idempotently.

    Each migration MUST be:
      - Idempotent (safe to run on a fresh DB AND an already-migrated DB)
      - Forward-compatible (never drops or renames existing data)
      - Backed by an ADR if it changes a safety-relevant column
    """
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(broker_bindings)")
    }
    # ADR-0002: live_orders_enabled flag
    if "live_orders_enabled" not in cols:
        conn.execute(
            "ALTER TABLE broker_bindings "
            "ADD COLUMN live_orders_enabled INTEGER NOT NULL DEFAULT 0"
        )


def close_default() -> None:
    """Release the singleton. Mainly for tests / clean shutdown."""
    global _default_conn
    with _lock:
        if _default_conn is not None:
            _default_conn.close()
            _default_conn = None
