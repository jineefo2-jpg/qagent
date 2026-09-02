"""
Admin gate based on `ADMIN_EMAILS` env (comma-separated, case-insensitive).

Used by management endpoints like `/metrics/brokers` that should not be
reachable by ordinary logged-in users.

Behaviour:
  - ADMIN_EMAILS unset or empty → no one is admin (deny by default).
  - Logged-out request → 401 from require_user (FastAPI handles).
  - Logged-in user whose email is not on the list → 403 here.
  - Logged-in user whose email matches (case-insensitive) → returned through.
"""
from __future__ import annotations

import os
from typing import Set

from fastapi import Depends, HTTPException

from .deps import require_user
from .users import User


def _admin_emails() -> Set[str]:
    raw = os.getenv("ADMIN_EMAILS", "") or ""
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin(user: User) -> bool:
    """Pure function — handy for templates / non-FastAPI callers."""
    email = (user.email or "").strip().lower()
    return bool(email and email in _admin_emails())


def require_admin(user: User = Depends(require_user)) -> User:
    """FastAPI dependency: 403 if user is not in ADMIN_EMAILS."""
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
