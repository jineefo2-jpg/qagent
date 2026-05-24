"""
Generate a fresh KEK for the broker credentials store.

Usage:
    python -m brokers.gen_kek                       # print key, copy manually
    python -m brokers.gen_kek --write-env           # write into ./.env
    python -m brokers.gen_kek --version 2 --write-env   # rotation

`--write-env` will REPLACE an empty `BROKER_KEK_v<n>=` placeholder line if
it finds one, otherwise append a new line. It refuses to overwrite an
existing non-empty value (so an accidental re-run cannot trash a live key).
Never paste the key anywhere else — chat history, GitHub issues, etc.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


def _next_default_version() -> int:
    """If BROKER_KEK_v1 already exists in env, suggest v2, etc."""
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


def _write_to_env(env_path: Path, version: int, key: str) -> str:
    """
    Write BROKER_KEK_v<version>=<key> to .env.

    Returns one of: 'replaced placeholder', 'appended'.
    Raises FileNotFoundError if .env is missing,
           RuntimeError if version is already set with a non-empty value.
    """
    if not env_path.exists():
        raise FileNotFoundError(
            f"{env_path} does not exist. Copy .env.example to .env first."
        )

    prefix = f"BROKER_KEK_v{version}="
    text = env_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    placeholder_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        # Ignore commented lines
        if s.startswith("#"):
            continue
        if s.startswith(prefix):
            value = s[len(prefix):].strip()
            if value:
                raise RuntimeError(
                    f"BROKER_KEK_v{version} is already set in {env_path}. "
                    f"Refusing to overwrite. To rotate, re-run with "
                    f"--version {version + 1}."
                )
            placeholder_idx = i

    new_line = f"{prefix}{key}"
    if placeholder_idx is not None:
        lines[placeholder_idx] = new_line
        action = "replaced placeholder"
    else:
        lines.append(new_line)
        action = "appended"

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return action


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version", type=int, default=None,
        help="KEK version number (default: auto-detect next free slot)",
    )
    p.add_argument(
        "--write-env", action="store_true",
        help="Write directly to ./.env instead of printing to stdout.",
    )
    args = p.parse_args(argv)

    version = args.version if args.version is not None else _next_default_version()
    if version < 1:
        print("ERROR: --version must be >= 1", file=sys.stderr)
        return 2

    key = Fernet.generate_key().decode("ascii")

    if args.write_env:
        try:
            action = _write_to_env(_ENV_PATH, version, key)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        # IMPORTANT: never echo the actual key to stdout in this branch.
        print(f"✅ {action} BROKER_KEK_v{version} in {_ENV_PATH}")
        print(f"   Restart the server to pick up the new key.")
        print(f"   To rotate later: re-run with --version {version + 1} --write-env,")
        print(f"   then `python -m brokers.rotate_kek`.")
        return 0

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
