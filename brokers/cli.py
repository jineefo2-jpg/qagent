"""
CLI for broker credential management.

Usage:
    python -m brokers.cli bind alpaca --user-id u_xyz --label main \
            [--api-key K --api-secret S]
    python -m brokers.cli list --user-id u_xyz
    python -m brokers.cli unbind --user-id u_xyz --binding-id 7

Designed as a fallback for the era before the UI bind wizard (X5) lands,
and as the canonical interface for diagnostic / rotation playbooks.

Security notes:
  - Secrets read via getpass when --api-key / --api-secret are omitted
    (never echoed to stdout, never on argv, never in shell history).
  - Audit rows are written by the underlying store with actor='user'.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from typing import Optional

# Ensure .env is loaded so BROKER_KEK_v1 is available.
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass


def _build_creds(broker_type: str, args: argparse.Namespace):
    """Construct the right Credentials subclass; prompt for missing secrets."""
    from .base import AlpacaCredentials, MockCredentials

    if broker_type == "alpaca":
        api_key = args.api_key or getpass.getpass("Alpaca API Key: ")
        api_secret = args.api_secret or getpass.getpass("Alpaca API Secret: ")
        if not api_key or not api_secret:
            raise SystemExit("ERROR: alpaca requires both --api-key and --api-secret")
        return AlpacaCredentials(api_key=api_key, api_secret=api_secret)

    if broker_type == "mock":
        return MockCredentials(initial_cash=float(args.initial_cash or 100000.0))

    # X4 will add tiger here.
    raise SystemExit(
        f"ERROR: broker_type {broker_type!r} not yet supported by the CLI."
    )


def cmd_bind(args: argparse.Namespace) -> int:
    from .credentials_store import store, CredentialsStoreError

    creds = _build_creds(args.broker_type, args)
    try:
        binding_id = store.bind(
            user_id=args.user_id,
            broker_type=args.broker_type,
            label=args.label,
            creds=creds,
            actor="user",
            env=args.env,
        )
    except CredentialsStoreError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"✅ bound {args.broker_type}/{args.label} for {args.user_id} "
        f"(id={binding_id}, env={args.env})"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .credentials_store import store

    rows = store.list_user_bindings(args.user_id)
    if not rows:
        print(f"(no bindings for {args.user_id})")
        return 0

    print(f"{'id':>4}  {'broker':<10} {'label':<20} {'env':<6} {'last_used':<12}")
    print("─" * 60)
    for r in rows:
        last = str(r.last_used_at) if r.last_used_at else "never"
        print(f"{r.id:>4}  {r.broker_type:<10} {r.label:<20} {r.env:<6} {last:<12}")
    return 0


def cmd_unbind(args: argparse.Namespace) -> int:
    from .credentials_store import store

    ok = store.unbind(args.binding_id, args.user_id, actor="user")
    if ok:
        print(f"✅ unbound id={args.binding_id} for {args.user_id}")
        return 0
    print(
        f"ERROR: no binding id={args.binding_id} owned by {args.user_id}",
        file=sys.stderr,
    )
    return 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brokers.cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bind", help="Bind a brokerage account for a user")
    pb.add_argument("broker_type", choices=["alpaca", "mock"],
                    help="(tiger lands in X4)")
    pb.add_argument("--user-id", required=True)
    pb.add_argument("--label", default="main")
    pb.add_argument("--env", default="paper", choices=["paper", "live"])
    pb.add_argument("--api-key", default=None, help="(alpaca; will prompt if omitted)")
    pb.add_argument("--api-secret", default=None,
                    help="(alpaca; will prompt if omitted)")
    pb.add_argument("--initial-cash", default=None, help="(mock; default 100000)")
    pb.set_defaults(func=cmd_bind)

    pl = sub.add_parser("list", help="List bindings for a user (no secrets shown)")
    pl.add_argument("--user-id", required=True)
    pl.set_defaults(func=cmd_list)

    pu = sub.add_parser("unbind", help="Delete a binding by id (must own it)")
    pu.add_argument("--user-id", required=True)
    pu.add_argument("--binding-id", required=True, type=int)
    pu.set_defaults(func=cmd_unbind)

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
