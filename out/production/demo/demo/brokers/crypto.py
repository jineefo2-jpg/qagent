"""
Envelope encryption for broker credentials (ADR-0001 §2 + §4).

Layered design:
  - **DEK** (Data Encryption Key): generated fresh per binding. Encrypts the
    actual credential blob with Fernet (AES-128-CBC + HMAC-SHA256 under the hood).
  - **KEK** (Key Encryption Key): pulled from env vars `BROKER_KEK_v1`,
    `BROKER_KEK_v2`, ... Encrypts the per-binding DEK.

Why two layers:
  - Rotating the KEK only requires re-wrapping each DEK (small, fast),
    not re-encrypting potentially large credential blobs.
  - A leaked DB is useless without at least one valid KEK env var.

Operator workflow:
  1. Generate the first KEK: `python -m brokers.gen_kek` → copy line into .env
  2. Server starts; encrypt() picks the highest-versioned KEK.
  3. Later, to rotate: generate KEK_v2, `python -m brokers.rotate_kek`,
     after 30 days delete KEK_v1.

Important:
  - Plaintext credential bytes are decrypted only inside `decrypt()` and the
    caller's adapter constructor — never logged, never persisted.
  - `EncryptedBlob` is a plain NamedTuple of bytes/int; safe to store in SQLite.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, NamedTuple, Tuple

from cryptography.fernet import Fernet, InvalidToken


# ════════════════════════════════════════════════════════════
# Public types
# ════════════════════════════════════════════════════════════

class EncryptedBlob(NamedTuple):
    """The three columns we store per credential row."""
    ciphertext: bytes      # credential bytes encrypted by DEK
    dek_wrapped: bytes     # DEK bytes encrypted by KEK[kek_version]
    kek_version: int


class CryptoError(Exception):
    """Anything wrong with KEK env / wrapping / unwrapping. Never includes
    plaintext or key material in its message."""


# ════════════════════════════════════════════════════════════
# KEK discovery
# ════════════════════════════════════════════════════════════

_KEK_PREFIX = "BROKER_KEK_v"
_KEK_RE = re.compile(rf"^{re.escape(_KEK_PREFIX)}(\d+)$")


def _list_keks() -> List[Tuple[int, bytes]]:
    """
    Discover every `BROKER_KEK_v<n>` env var. Returns list of (version, key)
    sorted descending by version (so the current/latest KEK is first).
    """
    out: List[Tuple[int, bytes]] = []
    for name, value in os.environ.items():
        m = _KEK_RE.match(name)
        if not m:
            continue
        if not value or not value.strip():
            continue
        try:
            version = int(m.group(1))
        except ValueError:
            continue
        out.append((version, value.strip().encode("utf-8")))
    if not out:
        raise CryptoError(
            "No BROKER_KEK_v* env var set. Generate one with "
            "`python -m brokers.gen_kek` and add it to your .env."
        )
    out.sort(key=lambda kv: -kv[0])
    return out


def _current_kek() -> Tuple[int, bytes]:
    """The highest-versioned KEK — used for new encryptions and rewraps."""
    return _list_keks()[0]


def _kek_fernets() -> Dict[int, Fernet]:
    """All KEKs as Fernet instances, keyed by version. Used to decrypt
    blobs that may have been wrapped by an older KEK."""
    out: Dict[int, Fernet] = {}
    for version, key in _list_keks():
        try:
            out[version] = Fernet(key)
        except Exception as e:
            raise CryptoError(f"BROKER_KEK_v{version} is not a valid Fernet key") from e
    return out


# ════════════════════════════════════════════════════════════
# encrypt / decrypt / rewrap
# ════════════════════════════════════════════════════════════

def encrypt(plaintext: bytes) -> EncryptedBlob:
    """
    Generate a fresh DEK, encrypt `plaintext` with it, wrap the DEK with the
    current KEK. Returns an EncryptedBlob ready to persist.
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("encrypt() requires bytes")
    plaintext = bytes(plaintext)

    dek = Fernet.generate_key()                      # 44 bytes urlsafe base64
    ciphertext = Fernet(dek).encrypt(plaintext)

    version, kek_bytes = _current_kek()
    try:
        kek_fernet = Fernet(kek_bytes)
    except Exception as e:
        raise CryptoError(f"BROKER_KEK_v{version} is not a valid Fernet key") from e
    dek_wrapped = kek_fernet.encrypt(dek)

    return EncryptedBlob(ciphertext=ciphertext, dek_wrapped=dek_wrapped, kek_version=version)


def decrypt(blob: EncryptedBlob) -> bytes:
    """
    Unwrap the DEK using KEK[blob.kek_version], then decrypt the ciphertext.
    Raises CryptoError if the KEK version is no longer available or if any
    Fernet token is invalid (tampering, wrong key, expired).
    """
    keks = _kek_fernets()
    if blob.kek_version not in keks:
        raise CryptoError(
            f"KEK version {blob.kek_version} is not available. "
            f"Available versions: {sorted(keks.keys())}. "
            f"If you rotated, keep the old KEK env var until rotate_kek "
            f"has re-wrapped every binding."
        )
    try:
        dek = keks[blob.kek_version].decrypt(blob.dek_wrapped)
    except InvalidToken as e:
        raise CryptoError("Failed to unwrap DEK (wrong KEK or tampered blob)") from e
    try:
        return Fernet(dek).decrypt(blob.ciphertext)
    except InvalidToken as e:
        raise CryptoError("Failed to decrypt credential (tampered ciphertext)") from e


def rewrap(blob: EncryptedBlob) -> EncryptedBlob:
    """
    Re-wrap an existing DEK with the current KEK. Used by `rotate_kek`.
    Plaintext is never touched.
    """
    keks = _kek_fernets()
    if blob.kek_version not in keks:
        raise CryptoError(
            f"Cannot rewrap: KEK version {blob.kek_version} is not available."
        )
    try:
        dek = keks[blob.kek_version].decrypt(blob.dek_wrapped)
    except InvalidToken as e:
        raise CryptoError("Failed to unwrap DEK during rewrap") from e

    new_version, new_kek = _current_kek()
    if new_version == blob.kek_version:
        # No-op: blob already wrapped with current KEK.
        return blob
    try:
        new_wrapped = Fernet(new_kek).encrypt(dek)
    except Exception as e:
        raise CryptoError(f"BROKER_KEK_v{new_version} is not a valid Fernet key") from e
    return EncryptedBlob(
        ciphertext=blob.ciphertext,
        dek_wrapped=new_wrapped,
        kek_version=new_version,
    )
