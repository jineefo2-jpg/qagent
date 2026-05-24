"""
brokers/crypto.py — Fernet envelope encryption + KEK rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def kek_v1(monkeypatch):
    """Install a single KEK v1 and clear any other KEK envs."""
    # Clear any pre-existing BROKER_KEK_v* (e.g. real .env loaded by other code)
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("BROKER_KEK_v1", key)
    return key


@pytest.fixture
def kek_v1_and_v2(monkeypatch):
    """Install two KEKs (v1 and v2); v2 is the current one."""
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    k1 = Fernet.generate_key().decode("ascii")
    k2 = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("BROKER_KEK_v1", k1)
    monkeypatch.setenv("BROKER_KEK_v2", k2)
    return k1, k2


# ─────────────────────────────────────────────────────────────────────────────
# Roundtrip & integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip(kek_v1):
    from brokers.crypto import encrypt, decrypt

    plaintext = b"super-secret-private-key-blob"
    blob = encrypt(plaintext)
    assert plaintext != blob.ciphertext  # actually encrypted
    assert decrypt(blob) == plaintext


def test_encrypt_produces_different_dek_each_call(kek_v1):
    """Two encryptions of the same plaintext yield different DEK & ciphertext."""
    from brokers.crypto import encrypt

    pt = b"same secret"
    b1 = encrypt(pt)
    b2 = encrypt(pt)
    assert b1.ciphertext != b2.ciphertext
    assert b1.dek_wrapped != b2.dek_wrapped


def test_decrypt_with_no_kek_env_raises(monkeypatch, kek_v1):
    """Server start with rows but no KEK → CryptoError. (CLAUDE.md mandate.)"""
    from brokers.crypto import encrypt, decrypt, CryptoError

    blob = encrypt(b"x")
    monkeypatch.delenv("BROKER_KEK_v1")
    with pytest.raises(CryptoError, match="No BROKER_KEK"):
        decrypt(blob)


def test_decrypt_unknown_kek_version_raises(monkeypatch, kek_v1):
    """Blob says kek_version=99 but env only has v1 → clear error."""
    from brokers.crypto import encrypt, decrypt, CryptoError, EncryptedBlob

    blob = encrypt(b"x")
    bad_blob = EncryptedBlob(
        ciphertext=blob.ciphertext,
        dek_wrapped=blob.dek_wrapped,
        kek_version=99,
    )
    with pytest.raises(CryptoError, match="KEK version 99"):
        decrypt(bad_blob)


def test_decrypt_tampered_ciphertext_raises(kek_v1):
    from brokers.crypto import encrypt, decrypt, CryptoError

    blob = encrypt(b"hello")
    tampered = blob._replace(ciphertext=blob.ciphertext[:-1] + b"!")
    with pytest.raises(CryptoError):
        decrypt(tampered)


def test_decrypt_tampered_dek_wrapped_raises(kek_v1):
    from brokers.crypto import encrypt, decrypt, CryptoError

    blob = encrypt(b"hello")
    tampered = blob._replace(dek_wrapped=blob.dek_wrapped[:-1] + b"!")
    with pytest.raises(CryptoError):
        decrypt(tampered)


# ─────────────────────────────────────────────────────────────────────────────
# KEK rotation
# ─────────────────────────────────────────────────────────────────────────────

def test_current_kek_is_highest_version(kek_v1_and_v2):
    """When both v1 and v2 are set, encrypt() must use v2."""
    from brokers.crypto import encrypt

    blob = encrypt(b"x")
    assert blob.kek_version == 2


def test_blob_wrapped_by_v1_still_decrypts_after_v2_added(monkeypatch):
    """Old blobs must remain readable while keeping both KEKs around."""
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)

    # Step 1: only v1 exists, encrypt
    k1 = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("BROKER_KEK_v1", k1)
    from brokers.crypto import encrypt, decrypt
    blob = encrypt(b"hello")
    assert blob.kek_version == 1

    # Step 2: v2 added; v1 still present
    k2 = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("BROKER_KEK_v2", k2)
    assert decrypt(blob) == b"hello"


def test_rewrap_changes_version_keeps_plaintext(kek_v1_and_v2):
    """rewrap() preserves plaintext but updates kek_version + dek_wrapped."""
    from brokers.crypto import encrypt, decrypt, rewrap

    # Force encrypt with v1 by temporarily hiding v2
    import os
    k2 = os.environ.pop("BROKER_KEK_v2")
    try:
        blob_v1 = encrypt(b"hello")
        assert blob_v1.kek_version == 1
    finally:
        os.environ["BROKER_KEK_v2"] = k2

    # Now rotate to v2
    blob_v2 = rewrap(blob_v1)
    assert blob_v2.kek_version == 2
    assert blob_v2.ciphertext == blob_v1.ciphertext       # untouched
    assert blob_v2.dek_wrapped != blob_v1.dek_wrapped     # re-wrapped
    assert decrypt(blob_v2) == b"hello"


def test_rewrap_is_noop_when_already_current(kek_v1):
    from brokers.crypto import encrypt, rewrap

    blob = encrypt(b"x")
    again = rewrap(blob)
    assert again == blob


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_encrypt_rejects_non_bytes(kek_v1):
    from brokers.crypto import encrypt

    with pytest.raises(TypeError):
        encrypt("not bytes")  # type: ignore[arg-type]


def test_encrypt_accepts_bytearray(kek_v1):
    from brokers.crypto import encrypt, decrypt

    data = bytearray(b"mutable")
    blob = encrypt(data)
    assert decrypt(blob) == b"mutable"


def test_invalid_kek_value_raises(monkeypatch):
    """If BROKER_KEK_v1 is set but not a valid Fernet key, fail clearly."""
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", "this-is-not-a-fernet-key")
    from brokers.crypto import encrypt, CryptoError
    with pytest.raises(CryptoError, match="not a valid Fernet key"):
        encrypt(b"x")
