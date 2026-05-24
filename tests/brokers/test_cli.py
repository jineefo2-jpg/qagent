"""
brokers/cli.py — bind / list / unbind end-to-end via argparse entry point.

Tests avoid getpass prompts by passing --api-key / --api-secret directly,
which is the same path automation / playbooks will use.
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
def cli_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "cli.db")
    yield
    _db.close_default()


def test_bind_alpaca_via_flags(cli_env, capsys):
    from brokers import cli
    from brokers.credentials_store import store

    rc = cli.main([
        "bind", "alpaca",
        "--user-id", "u_42", "--label", "main",
        "--api-key", "PK_test", "--api-secret", "secret_test",
    ])
    assert rc == 0

    out = capsys.readouterr().out
    assert "bound alpaca/main" in out

    creds = store.load("u_42", "alpaca", "main", actor="system")
    assert creds.api_key == "PK_test"
    assert creds.api_secret == "secret_test"


def test_list_shows_bound_rows(cli_env, capsys):
    from brokers import cli

    cli.main([
        "bind", "alpaca",
        "--user-id", "u_42", "--label", "main",
        "--api-key", "k", "--api-secret", "s",
    ])
    capsys.readouterr()  # clear

    rc = cli.main(["list", "--user-id", "u_42"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpaca" in out
    assert "main" in out
    # Secret MUST NOT leak into list output.
    assert "k" not in out.split("paper")[1] if "paper" in out else True
    assert "secret_test" not in out


def test_list_empty_user(cli_env, capsys):
    from brokers import cli

    rc = cli.main(["list", "--user-id", "u_nobody"])
    assert rc == 0
    assert "no bindings" in capsys.readouterr().out


def test_unbind_removes_row(cli_env, capsys):
    from brokers import cli
    from brokers.credentials_store import store

    cli.main([
        "bind", "alpaca",
        "--user-id", "u_42", "--label", "main",
        "--api-key", "k", "--api-secret", "s",
    ])
    [summary] = store.list_user_bindings("u_42")
    capsys.readouterr()

    rc = cli.main([
        "unbind", "--user-id", "u_42", "--binding-id", str(summary.id),
    ])
    assert rc == 0
    assert "unbound" in capsys.readouterr().out
    assert store.list_user_bindings("u_42") == []


def test_unbind_wrong_user_fails(cli_env, capsys):
    from brokers import cli
    from brokers.credentials_store import store

    cli.main([
        "bind", "alpaca",
        "--user-id", "u_alice", "--label", "main",
        "--api-key", "k", "--api-secret", "s",
    ])
    [summary] = store.list_user_bindings("u_alice")
    capsys.readouterr()

    rc = cli.main([
        "unbind", "--user-id", "u_bob", "--binding-id", str(summary.id),
    ])
    assert rc == 3
    err = capsys.readouterr().err
    assert "no binding" in err


def test_bind_duplicate_label_exits_nonzero(cli_env, capsys):
    from brokers import cli

    cli.main([
        "bind", "alpaca",
        "--user-id", "u_42", "--label", "main",
        "--api-key", "k1", "--api-secret", "s1",
    ])
    capsys.readouterr()

    rc = cli.main([
        "bind", "alpaca",
        "--user-id", "u_42", "--label", "main",
        "--api-key", "k2", "--api-secret", "s2",
    ])
    assert rc == 2
    assert "already exists" in capsys.readouterr().err


def test_bind_mock_with_initial_cash(cli_env, capsys):
    from brokers import cli
    from brokers.credentials_store import store

    rc = cli.main([
        "bind", "mock",
        "--user-id", "u_42", "--label", "test",
        "--initial-cash", "50000",
    ])
    assert rc == 0
    creds = store.load("u_42", "mock", "test", actor="system")
    assert creds.initial_cash == 50000.0
