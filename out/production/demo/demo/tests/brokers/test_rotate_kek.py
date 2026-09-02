"""
brokers/rotate_kek.py — KEK rotation tool.

Verifies the rotation playbook end-to-end:
  1. Bind under v1.
  2. Add v2; v1 still in env.
  3. Run rotate_kek.main(--apply). All bindings now wrapped with v2.
  4. Old credentials are still loadable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def two_kek_env(monkeypatch, tmp_path):
    """Set up v1 + v2 KEKs and an isolated DB."""
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("BROKER_KEK_v2", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "rotate.db")

    yield tmp_path

    _db.close_default()


def test_dry_run_does_not_modify_rows(two_kek_env, monkeypatch, capsys):
    """First bind a row wrapped with v1 (by hiding v2), then dry-run."""
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers import rotate_kek, _db

    # Hide v2 to force v1 wrapping
    saved_v2 = os.environ.pop("BROKER_KEK_v2")
    try:
        store.bind("u_42", "alpaca", "main",
                   AlpacaCredentials(api_key="k", api_secret="s"),
                   actor="user")
    finally:
        os.environ["BROKER_KEK_v2"] = saved_v2

    conn = _db.init()
    before = conn.execute(
        "SELECT kek_version, dek_wrapped FROM broker_bindings"
    ).fetchone()
    assert before[0] == 1

    rc = rotate_kek.main([])  # dry-run (no --apply)
    assert rc == 0

    after = conn.execute(
        "SELECT kek_version, dek_wrapped FROM broker_bindings"
    ).fetchone()
    assert after[0] == 1
    assert after[1] == before[1]

    out = capsys.readouterr().out
    assert "Dry-run only" in out


def test_apply_rotates_all_bindings(two_kek_env):
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers import rotate_kek, _db

    # Bind two rows under v1
    saved_v2 = os.environ.pop("BROKER_KEK_v2")
    try:
        store.bind("u_alice", "alpaca", "main",
                   AlpacaCredentials(api_key="ak", api_secret="as"),
                   actor="user")
        store.bind("u_bob", "alpaca", "main",
                   AlpacaCredentials(api_key="bk", api_secret="bs"),
                   actor="user")
    finally:
        os.environ["BROKER_KEK_v2"] = saved_v2

    conn = _db.init()
    versions_before = [
        r[0] for r in conn.execute("SELECT kek_version FROM broker_bindings")
    ]
    assert versions_before == [1, 1]

    rc = rotate_kek.main(["--apply"])
    assert rc == 0

    versions_after = [
        r[0] for r in conn.execute("SELECT kek_version FROM broker_bindings")
    ]
    assert versions_after == [2, 2]

    # Credentials still load correctly under the new wrap
    loaded = store.load("u_alice", "alpaca", "main", actor="system")
    assert loaded.api_key == "ak"
    loaded = store.load("u_bob", "alpaca", "main", actor="system")
    assert loaded.api_key == "bk"

    # Audit rows for rotation
    rotate_rows = conn.execute(
        "SELECT COUNT(*) FROM broker_audit_log "
        "WHERE actor='rotation' AND action='rotate' AND success=1"
    ).fetchone()[0]
    assert rotate_rows == 2


def test_apply_no_op_when_all_current(two_kek_env, capsys):
    """If every binding is already at the current version, exit cleanly."""
    from brokers.base import AlpacaCredentials
    from brokers.credentials_store import store
    from brokers import rotate_kek

    # Both KEKs present — bind under current (v2)
    store.bind("u_42", "alpaca", "main",
               AlpacaCredentials(api_key="k", api_secret="s"),
               actor="user")

    rc = rotate_kek.main(["--apply"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing to do" in out
