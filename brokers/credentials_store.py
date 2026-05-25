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

from . import _db, audit, crypto, metrics
from .base import (
    AlpacaCredentials,
    Credentials,
    MockCredentials,
    TigerCredentials,
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
    # ADR-0002: per-binding opt-in for live order placement
    live_orders_enabled: bool = False


class CredentialsStoreError(Exception):
    """Bind/unbind/load errors. Message never includes plaintext or keys."""


# ════════════════════════════════════════════════════════════
# Credentials factory from API payload (used by REST endpoints)
# ════════════════════════════════════════════════════════════

def build_credentials(broker_type: str, payload: dict) -> Credentials:
    """
    Construct the right Credentials subclass from a generic dict payload.
    Used by the /api/broker/bindings endpoint to turn JSON into a typed
    Credentials object. Raises ValueError on unknown broker_type so the
    HTTP layer can return 400.
    """
    if not isinstance(payload, dict):
        raise ValueError("credentials payload must be an object")

    if broker_type == "alpaca":
        return AlpacaCredentials(
            api_key=str(payload.get("api_key", "")).strip(),
            api_secret=str(payload.get("api_secret", "")).strip(),
            base_url=str(payload.get(
                "base_url", "https://paper-api.alpaca.markets",
            )).strip(),
        )
    if broker_type == "tiger":
        return TigerCredentials(
            tiger_id=str(payload.get("tiger_id", "")).strip(),
            private_key=str(payload.get("private_key", "")),
            account=str(payload.get("account", "")).strip(),
            license=str(payload.get("license", "TBNZ")).strip() or "TBNZ",
        )
    if broker_type == "mock":
        return MockCredentials(
            initial_cash=float(payload.get("initial_cash", 100000.0)),
        )
    raise ValueError(f"Unsupported broker_type: {broker_type!r}")


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
            metrics.incr("broker_bind_total", broker=broker_type, result="fail")
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
        metrics.incr("broker_bind_total", broker=broker_type, result="ok")

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
            metrics.incr("broker_use_total", broker=broker_type, result="fail")
            metrics.incr("broker_auth_fail_total", broker=broker_type)
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
        metrics.incr("broker_use_total", broker=broker_type, result="ok")
        return creds

    # ── list ────────────────────────────────────────────────

    def list_user_bindings(self, user_id: str) -> List[BindingSummary]:
        c = self._c()
        rows = c.execute(
            """
            SELECT id, user_id, broker_type, label, env, created_at,
                   last_used_at, live_orders_enabled
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
                live_orders_enabled=bool(r[7]),
            )
            for r in rows
        ]

    # ── ADR-0002: live trading opt-in flag (per binding) ─────────

    def is_live_orders_enabled(self, binding_id: int, user_id: str) -> bool:
        """Read-only check. Returns False for paper bindings, unknown ids,
        or wrong-user attempts. The caller MUST use this before letting any
        order touch a live binding."""
        c = self._c()
        row = c.execute(
            "SELECT env, live_orders_enabled FROM broker_bindings "
            "WHERE id = ? AND user_id = ?",
            (binding_id, user_id),
        ).fetchone()
        if row is None:
            return False
        env, flag = row
        if env != "live":
            return False
        return bool(flag)

    def set_live_orders_enabled(
        self,
        binding_id: int,
        user_id: str,
        enabled: bool,
        *,
        actor: str,
        ack: str = "",
    ) -> bool:
        """
        Flip the live_orders_enabled flag on a binding owned by user_id.
        Returns True on success, False if the binding doesn't exist OR is
        not owned by user_id OR has env='paper' (the flag is meaningless
        for paper bindings, refuse).

        Per ADR-0002, the caller (HTTP layer) MUST require an explicit ack
        string when enabling live orders. We record it in the audit row.

        Per CLAUDE.md "live trading" rule, the LLM tool layer MUST NOT
        ever call this — `actor='llm'` is rejected here as a fail-safe.
        """
        if actor == "llm":
            raise CredentialsStoreError(
                "live_orders_enabled MUST NOT be flipped by the LLM tool path"
            )
        c = self._c()
        # Find the binding + verify ownership + verify env='live'
        row = c.execute(
            "SELECT broker_type, label, env FROM broker_bindings "
            "WHERE id = ? AND user_id = ?",
            (binding_id, user_id),
        ).fetchone()
        if row is None:
            audit.audit_log(
                actor=actor, action="fail",
                user_id=user_id, binding_id=binding_id,
                detail="set_live_orders_enabled: not found / wrong user",
                success=False, conn=c,
            )
            return False
        broker_type, label, env = row
        if env != "live":
            audit.audit_log(
                actor=actor, action="fail",
                user_id=user_id, binding_id=binding_id,
                detail=f"set_live_orders_enabled refused: env={env!r}",
                success=False, conn=c,
            )
            return False

        new_val = 1 if enabled else 0
        c.execute(
            "UPDATE broker_bindings SET live_orders_enabled = ? "
            "WHERE id = ? AND user_id = ?",
            (new_val, binding_id, user_id),
        )
        audit.audit_log(
            actor=actor, action="use",
            user_id=user_id, binding_id=binding_id,
            detail=(
                f"{broker_type}/{label}: live_orders_enabled = {new_val} "
                f"ack={ack[:80]!r}"
            ),
            success=True, conn=c,
        )

        # Drop registry cache so the next request rebuilds the adapter
        # (currently a no-op since adapter doesn't read this flag, but
        # keeps the invariant "binding state change → cache invalidate").
        from . import registry
        registry._registry.invalidate(user_id, broker_type, label)
        registry._registry.invalidate(user_id, broker_type, None)
        return True

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
            metrics.incr("broker_unbind_total", broker="unknown", result="fail")
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
        metrics.incr("broker_unbind_total", broker=broker_type, result="ok")

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
