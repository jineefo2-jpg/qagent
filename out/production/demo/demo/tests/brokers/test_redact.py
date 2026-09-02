"""
brokers.base.redact_credentials — shared helper used by adapter _wrap_error
to keep private-key bytes out of exception messages and logs.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_redact_strips_pem_block():
    from brokers.base import redact_credentials

    src = (
        "error before\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICXAIBAAKBgQDIyVcAAQUq7Q\n"
        "TWxqMDe44XOlS/yiTbcN/eZqAA==\n"
        "-----END RSA PRIVATE KEY-----\n"
        "error after"
    )
    out = redact_credentials(src)
    assert "MIICXAIBAAKBgQDIyVcAAQUq7Q" not in out
    assert "TWxqMDe44XOlS" not in out
    assert "[REDACTED-PEM]" in out
    assert "error before" in out
    assert "error after" in out


def test_redact_strips_pkcs8_block():
    from brokers.base import redact_credentials

    src = (
        "-----BEGIN PRIVATE KEY-----\n"
        "AAAA\nBBBB\nCCCC\n"
        "-----END PRIVATE KEY-----"
    )
    out = redact_credentials(src)
    assert "[REDACTED-PEM]" in out
    assert "AAAA" not in out


def test_redact_strips_long_base64_blob():
    """Even when not wrapped in PEM markers, long base64 runs are stripped."""
    from brokers.base import redact_credentials

    blob = "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDIyVcAAQUq7Q"
    assert len(blob) >= 50
    src = f"sign failed for key {blob} please retry"
    out = redact_credentials(src)
    assert blob not in out
    assert "[REDACTED-BASE64]" in out


def test_redact_leaves_short_strings_alone():
    """Short identifiers like account numbers or UUIDs are NOT redacted."""
    from brokers.base import redact_credentials

    src = "request 991485da-578e-11f1-b1bb-6ac052fd811e failed"
    # UUID is 36 chars, below the 50 threshold
    out = redact_credentials(src)
    assert "991485da" in out
    assert "[REDACTED" not in out


def test_redact_truncates_overlong_messages():
    """Use chars outside the base64 alphabet so truncation is the only thing
    that shortens the input (avoids the long-blob redaction kicking in first)."""
    from brokers.base import redact_credentials

    src = "! " * 600  # 1200 chars of non-base64 content
    out = redact_credentials(src, max_len=400)
    assert len(out) <= 400 + len("...[truncated]")
    assert out.endswith("...[truncated]")


def test_redact_handles_non_string_input():
    from brokers.base import redact_credentials

    class Weird:
        def __str__(self):
            return "weird obj"
    assert redact_credentials(Weird()) == "weird obj"
    assert redact_credentials(42) == "42"
    assert redact_credentials(None) == "None"
