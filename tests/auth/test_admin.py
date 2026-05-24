"""
auth/admin.py — ADMIN_EMAILS-based admin gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_user(email: str):
    from auth.users import User
    return User(
        user_id=f"google:{email}",
        email=email, name="Test User", avatar_url="",
        provider="google", created_at=0.0, last_login_at=0.0,
    )


def test_no_env_means_no_admins(monkeypatch):
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    from auth.admin import is_admin

    assert is_admin(_make_user("anyone@example.com")) is False


def test_listed_email_is_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "alice@example.com,bob@example.com")
    from auth.admin import is_admin

    assert is_admin(_make_user("alice@example.com")) is True
    assert is_admin(_make_user("bob@example.com")) is True


def test_email_matching_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "Alice@example.com")
    from auth.admin import is_admin

    assert is_admin(_make_user("alice@example.com")) is True
    assert is_admin(_make_user("ALICE@EXAMPLE.COM")) is True


def test_unlisted_email_is_not_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "alice@example.com")
    from auth.admin import is_admin

    assert is_admin(_make_user("eve@example.com")) is False


def test_empty_email_is_not_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "alice@example.com")
    from auth.admin import is_admin

    assert is_admin(_make_user("")) is False


def test_require_admin_returns_user_when_allowed(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "alice@example.com")
    from auth.admin import require_admin

    u = _make_user("alice@example.com")
    # Call the dep directly (FastAPI Depends becomes a kwarg).
    result = require_admin(user=u)
    assert result is u


def test_require_admin_raises_403_when_not(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "alice@example.com")
    from auth.admin import require_admin
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        require_admin(user=_make_user("eve@example.com"))
    assert exc.value.status_code == 403


def test_require_admin_raises_when_env_empty(monkeypatch):
    """No one is admin when ADMIN_EMAILS is unset — deny by default."""
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    from auth.admin import require_admin
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        require_admin(user=_make_user("alice@example.com"))
    assert exc.value.status_code == 403
