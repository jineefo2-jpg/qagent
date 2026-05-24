"""
Generate a fresh KEK for the broker credentials store.

Usage:
    python -m brokers.gen_kek
    python -m brokers.gen_kek --version 2   # for rotation

The output line is meant to be copied into `.env`. Never paste the key
anywhere else — chat history, GitHub issues, etc. — and never commit the
.env file.
"""
from __future__ import annotations

import argparse
import sys

from cryptography.fernet import Fernet


def _next_default_version() -> int:
    """If BROKER_KEK_v1 already exists in env, suggest v2, etc."""
    import os
    used = set()
    for name in os.environ:
        if name.startswith("BROKER_KEK_v"):
            try:
                used.add(int(name[len("BROKER_KEK_v"):]))
            except ValueError:
                pass
    if not used:
        return 1
    return max(used) + 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version", type=int, default=None,
        help="KEK version number (default: auto-detect next free slot)",
    )
    args = p.parse_args(argv)

    version = args.version if args.version is not None else _next_default_version()
    if version < 1:
        print("ERROR: --version must be >= 1", file=sys.stderr)
        return 2

    key = Fernet.generate_key().decode("ascii")

    print(f"# ✅ Generated KEK for broker credential store.")
    print(f"# Add ONE of these to your .env (do not commit!):")
    print()
    print(f"BROKER_KEK_v{version}={key}")
    print()
    print(f"# Restart the server after adding.")
    print(f"# To rotate later: re-run with --version {version + 1}, then run")
    print(f"#   python -m brokers.rotate_kek")
    return 0


if __name__ == "__main__":
    sys.exit(main())
