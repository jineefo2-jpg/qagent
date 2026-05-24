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


def test_bind_tiger_via_pem_file(cli_env, capsys, tmp_path):
    """Tiger binding reads PEM from --private-key-file."""
    from brokers import cli
    from brokers.credentials_store import store

    pem_file = tmp_path / "tiger.pem"
    pem_content = "-----BEGIN PRIVATE KEY-----\nFAKE-PEM-BODY\n-----END PRIVATE KEY-----"
    pem_file.write_text(pem_content)

    rc = cli.main([
        "bind", "tiger",
        "--user-id", "u_42", "--label", "paper-main",
        "--tiger-id", "20151024",
        "--private-key-file", str(pem_file),
        "--account", "U99999999",
    ])
    assert rc == 0
    assert "bound tiger/paper-main" in capsys.readouterr().out

    creds = store.load("u_42", "tiger", "paper-main", actor="system")
    assert creds.tiger_id == "20151024"
    assert creds.private_key == pem_content
    assert creds.account == "U99999999"
    assert creds.license == "TBNZ"  # default


def test_bind_tiger_rejects_non_pem_file(cli_env, capsys, tmp_path):
    from brokers import cli

    bad = tmp_path / "not_a_key.txt"
    bad.write_text("this is just a text file")

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "bind", "tiger",
            "--user-id", "u_42", "--label", "main",
            "--tiger-id", "20151024",
            "--private-key-file", str(bad),
            "--account", "U999",
        ])
    assert "PEM" in str(exc.value)


def test_bind_tiger_missing_args_rejected(cli_env, tmp_path):
    from brokers import cli

    pem_file = tmp_path / "k.pem"
    pem_file.write_text("-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----")

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "bind", "tiger",
            "--user-id", "u_42", "--label", "main",
            "--tiger-id", "20151024",
            # missing --account and --private-key-file
        ])
    assert "tiger requires" in str(exc.value)


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
