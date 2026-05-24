"""
KEK rotation tool — re-wrap every binding's DEK with the current KEK.

Workflow:
  1. Generate a new KEK and add it to .env at the next version:
        python -m brokers.gen_kek --version 2 --write-env
  2. Keep the previous KEK in .env so this tool can still unwrap old DEKs.
  3. Run rotation:
        python -m brokers.rotate_kek            # dry-run summary
        python -m brokers.rotate_kek --apply    # actually re-wrap
  4. After 30 days (per ADR-0001 §4), remove the old KEK from .env.

Each successful re-wrap writes an audit row (actor='rotation', action='rotate').
"""
from __future__ import annotations

import argparse
import sys

# Load .env first so BROKER_KEK_v* env vars are present.
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--apply", action="store_true",
        help="Actually re-wrap rows. Omit for a dry-run summary.",
    )
    args = p.parse_args(argv)

    # Late import so .env loads first.
    from . import _db, audit, crypto, metrics

    try:
        current_version, _ = crypto._current_kek()
    except crypto.CryptoError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    conn = _db.init()
    rows = conn.execute(
        "SELECT id, encrypted_credential, dek_wrapped, kek_version, user_id, broker_type "
        "FROM broker_bindings"
    ).fetchall()

    candidates = [r for r in rows if r[3] != current_version]

    print(f"Current KEK: v{current_version}")
    print(f"Total bindings: {len(rows)}")
    print(f"Bindings to re-wrap: {len(candidates)}")
    if not candidates:
        print("Nothing to do.")
        return 0

    # Show summary
    by_version: dict[int, int] = {}
    for r in candidates:
        by_version[r[3]] = by_version.get(r[3], 0) + 1
    for v, n in sorted(by_version.items()):
        print(f"   v{v} → v{current_version}: {n} bindings")

    if not args.apply:
        print()
        print("Dry-run only. Re-run with --apply to actually rotate.")
        return 0

    print()
    success = 0
    fail = 0
    for binding_id, ciphertext, dek_wrapped, old_version, user_id, broker_type in candidates:
        try:
            blob = crypto.EncryptedBlob(
                ciphertext=ciphertext,
                dek_wrapped=dek_wrapped,
                kek_version=old_version,
            )
            new_blob = crypto.rewrap(blob)
            conn.execute(
                "UPDATE broker_bindings SET dek_wrapped = ?, kek_version = ? WHERE id = ?",
                (new_blob.dek_wrapped, new_blob.kek_version, binding_id),
            )
            audit.audit_log(
                actor="rotation", action="rotate",
                user_id=user_id, binding_id=binding_id,
                detail=f"{broker_type}: v{old_version} → v{current_version}",
                success=True, conn=conn,
            )
            metrics.incr("broker_rotate_total", result="ok")
            success += 1
        except Exception as e:
            audit.audit_log(
                actor="rotation", action="rotate",
                user_id=user_id, binding_id=binding_id,
                detail=f"{broker_type}: rotate failed: {type(e).__name__}",
                success=False, conn=conn,
            )
            metrics.incr("broker_rotate_total", result="fail")
            fail += 1
            print(f"   FAIL binding {binding_id}: {e}", file=sys.stderr)

    print(f"✅ Re-wrapped {success} bindings to v{current_version}"
          + (f", {fail} failed" if fail else ""))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
