"""
CredentialsStore —— per-user broker credentials, encrypted at rest (ADR-0001 §3-4).

Public API:
    store.bind(user_id, broker_type, label, creds, *, actor, env='paper') -> int
    store.load(user_id, broker_type, label=None, *, actor) -> Optional[Credentials]
    store.list_user_bindings(user_id) -> list[BindingSummary]
    store.unbind(binding_id, user_id, *, actor) -> bool
    store.get_default_label(user_id, broker_type) -> Optional[str]

Wire-up:
    `bind` and `unbind` invalidate the BrokerRegistry's adapter cache via
    late import (avoids a circular import at module load).

    The audit log is written *inside* each mutating call. If the audit insert
    raises, the surrounding operation is aborted — per CLAUDE.md the audit
    trail MUST NOT be silenced. We don't have a transaction wrapper yet, so
    the audit row is appended AFTER the data row is committed; partial state
    (data without audit) is impossible only if both share a connection in
    autocommit mode with the audit row last — which is what we do here.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import sqlite3

from . import _db, audit, crypto
from .base import (
    AlpacaCredentials,
    Credentials,
    MockCredentials,
)


# ════════════════════════════════════════════════════════════
# Public summary type (no ciphertext)
# ════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BindingSummary:
    id: int
    user_id: str
    broker_type: str
    label: str
    env: str
    created_at: int
    last_used_at: Optional[int]


class CredentialsStoreError(Exception):
    """Bind/unbind/load errors. Message never includes plaintext or keys."""


# ════════════════════════════════════════════════════════════
# Credentials (de)serialization
# ════════════════════════════════════════════════════════════

def _serialize(creds: Credentials) -> bytes:
    """Convert a Credentials dataclass to JSON bytes for encryption."""
    return json.dumps(asdict(creds), ensure_ascii=False).encode("utf-8")


def _credentials_class_for(broker_type: str) -> type[Credentials]:
    """
    Find the Credentials subclass whose default `broker_type` field matches.
    New brokers (X4 Tiger) automatically pick up by subclassing Credentials.
    """
    for cls in Credentials.__subclasses__():
        field = cls.__dataclass_fields__.get("broker_type")
        if field is not None and field.default == broker_type:
            return cls
    raise CredentialsStoreError(
        f"No Credentials subclass registered for broker_type={broker_type!r}"
    )


def _deserialize(broker_type: str, plaintext: bytes) -> Credentials:
    """Inverse of _serialize. Raises CredentialsStoreError on unknown type."""
    cls = _credentials_class_for(broker_type)
    data = json.loads(plaintext.decode("utf-8"))
    # Drop broker_type from the dict (it's the default on the dataclass)
    data.pop("broker_type", None)
    try:
        return cls(**data)
    except TypeError as e:
        # Stored creds shape doesn't match current class — schema drift?
        raise CredentialsStoreError(
            f"Stored {broker_type} credential shape does not match current "
            f"{cls.__name__}: {e}"
        ) from e


# ════════════════════════════════════════════════════════════
# Store
# ════════════════════════════════════════════════════════════

class CredentialsStore:
    """All mutating methods take `actor` keyword-only — caller MUST declare
    whether the action is initiated by a 'user' click, 'system' code path,
    'rotation' tool, or (CRITICAL) by the 'llm'."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        # If None, init() resolves to the singleton on each call (allows
        # tests to patch _db._DEFAULT_DB_PATH before first use).
        self._conn = conn

    def _c(self) -> sqlite3.Connection:
        return self._conn if self._conn is not None else _db.init()

    # ── bind ────────────────────────────────────────────────

    def bind(
        self,
        user_id: str,
        broker_type: str,
        label: str,
        creds: Credentials,
        *,
        actor: str,
        env: str = "paper",
    ) -> int:
        if not user_id or not label:
            raise CredentialsStoreError("user_id and label are required")
        if env not in ("paper", "live"):
            raise CredentialsStoreError(f"env must be 'paper' or 'live', got {env!r}")
        if creds.broker_type != broker_type:
            raise CredentialsStoreError(
                f"creds.broker_type={creds.broker_type!r} mismatches "
                f"broker_type argument={broker_type!r}"
            )

        plaintext = _serialize(creds)
        blob = crypto.encrypt(plaintext)
        ts = int(time.time())

        c = self._c()
        try:
            cur = c.execute(
                """
                INSERT INTO broker_bindings
                    (user_id, broker_type, label, env,
                     encrypted_credential, dek_wrapped, kek_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, broker_type, label, env,
                 blob.ciphertext, blob.dek_wrapped, blob.kek_version, ts),
            )
        except sqlite3.IntegrityError as e:
            # UNIQUE(user_id, broker_type, label) violation OR CHECK fail
            audit.audit_log(
                actor=actor, action="bind",
                user_id=user_id, binding_id=None,
                detail=f"{broker_type}/{label}: integrity error",
                success=False, conn=c,
            )
            raise CredentialsStoreError(
                f"Binding ({broker_type}, label={label!r}) already exists for this user"
            ) from e

        binding_id = cur.lastrowid
        audit.audit_log(
            actor=actor, action="bind",
            user_id=user_id, binding_id=binding_id,
            detail=f"{broker_type}/{label}/env={env}",
            success=True, conn=c,
        )

        # Drop any stale adapter in the registry's cache (late import: cycle break)
        from . import registry
        registry._registry.invalidate(user_id, broker_type, label)
        # Also invalidate label=None bucket — default may have changed.
        registry._registry.invalidate(user_id, broker_type, None)

        return binding_id

    # ── load ────────────────────────────────────────────────

    def load(
        self,
        user_id: str,
        broker_type: str,
        label: Optional[str] = None,
        *,
        actor: str,
    ) -> Optional[Credentials]:
        c = self._c()
        if label is None:
            row = c.execute(
                """
                SELECT id, encrypted_credential, dek_wrapped, kek_version, label
                FROM broker_bindings
                WHERE user_id = ? AND broker_type = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (user_id, broker_type),
            ).fetchone()
        else:
            row = c.execute(
                """
                SELECT id, encrypted_credential, dek_wrapped, kek_version, label
                FROM broker_bindings
                WHERE user_id = ? AND broker_type = ? AND label = ?
                """,
                (user_id, broker_type, label),
            ).fetchone()

        if row is None:
            return None

        binding_id, ciphertext, dek_wrapped, kek_version, actual_label = row

        try:
            plaintext = crypto.decrypt(crypto.EncryptedBlob(
                ciphertext=ciphertext,
                dek_wrapped=dek_wrapped,
                kek_version=kek_version,
            ))
        except Exception:
            # Decrypt failures are audited as 'fail' (success=0) then re-raised.
            audit.audit_log(
                actor=actor, action="fail",
                user_id=user_id, binding_id=binding_id,
                detail=f"{broker_type}/{actual_label}: decrypt failed",
                success=False, conn=c,
            )
            raise

        try:
            creds = _deserialize(broker_type, plaintext)
        finally:
            # Zero the plaintext buffer ASAP. It's a `bytes`, so we can't
            # mutate; the best we can do is drop the reference.
            del plaintext

        # Touch last_used_at + audit 'use'
        c.execute(
            "UPDATE broker_bindings SET last_used_at = ? WHERE id = ?",
            (int(time.time()), binding_id),
        )
        audit.audit_log(
            actor=actor, action="use",
            user_id=user_id, binding_id=binding_id,
            detail=f"{broker_type}/{actual_label}",
            success=True, conn=c,
        )
        return creds

    # ── list ────────────────────────────────────────────────

    def list_user_bindings(self, user_id: str) -> List[BindingSummary]:
        c = self._c()
        rows = c.execute(
            """
            SELECT id, user_id, broker_type, label, env, created_at, last_used_at
            FROM broker_bindings
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()
        return [
            BindingSummary(
                id=r[0], user_id=r[1], broker_type=r[2], label=r[3],
                env=r[4], created_at=r[5], last_used_at=r[6],
            )
            for r in rows
        ]

    # ── unbind ──────────────────────────────────────────────

    def unbind(
        self,
        binding_id: int,
        user_id: str,
        *,
        actor: str,
    ) -> bool:
        c = self._c()
        # Look up first so audit + invalidate know the broker_type/label.
        row = c.execute(
            "SELECT broker_type, label FROM broker_bindings WHERE id = ? AND user_id = ?",
            (binding_id, user_id),
        ).fetchone()
        if row is None:
            audit.audit_log(
                actor=actor, action="unbind",
                user_id=user_id, binding_id=binding_id,
                detail="not found / wrong user",
                success=False, conn=c,
            )
            return False

        broker_type, label = row
        c.execute(
            "DELETE FROM broker_bindings WHERE id = ? AND user_id = ?",
            (binding_id, user_id),
        )
        audit.audit_log(
            actor=actor, action="unbind",
            user_id=user_id, binding_id=binding_id,
            detail=f"{broker_type}/{label}",
            success=True, conn=c,
        )

        from . import registry
        registry._registry.invalidate(user_id, broker_type, label)
        registry._registry.invalidate(user_id, broker_type, None)
        return True

    # ── default label ──────────────────────────────────────

    def get_default_label(self, user_id: str, broker_type: str) -> Optional[str]:
        """The first-bound label for this (user, broker). For c2 there is no
        explicit 'set default' concept; oldest binding wins."""
        c = self._c()
        row = c.execute(
            """
            SELECT label FROM broker_bindings
            WHERE user_id = ? AND broker_type = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (user_id, broker_type),
        ).fetchone()
        return row[0] if row else None


# Module-level singleton (matches `_db`'s pattern).
store = CredentialsStore()
