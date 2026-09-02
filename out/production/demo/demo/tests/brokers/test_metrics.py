"""
brokers/metrics.py — in-process counter store + on-demand gauges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_counters():
    from brokers import metrics
    metrics.reset()
    yield
    metrics.reset()


def test_incr_starts_at_one():
    from brokers.metrics import incr, snapshot

    incr("broker_bind_total", broker="alpaca", result="ok")
    snap = snapshot()
    items = [c for c in snap["counters"]
             if c["metric"] == "broker_bind_total"
             and c["labels"] == {"broker": "alpaca", "result": "ok"}]
    assert items == [{"metric": "broker_bind_total",
                      "labels": {"broker": "alpaca", "result": "ok"},
                      "value": 1}]


def test_incr_accumulates():
    from brokers.metrics import incr, snapshot

    for _ in range(5):
        incr("broker_use_total", broker="alpaca", result="ok")
    item = next(c for c in snapshot()["counters"]
                if c["metric"] == "broker_use_total")
    assert item["value"] == 5


def test_labels_are_distinct():
    """Different label combinations get separate counter slots."""
    from brokers.metrics import incr, snapshot

    incr("broker_bind_total", broker="alpaca", result="ok")
    incr("broker_bind_total", broker="alpaca", result="fail")
    incr("broker_bind_total", broker="tiger", result="ok")
    counters = snapshot()["counters"]
    relevant = [c for c in counters if c["metric"] == "broker_bind_total"]
    assert len(relevant) == 3
    assert all(c["value"] == 1 for c in relevant)


def test_snapshot_includes_gauges():
    from brokers.metrics import snapshot

    snap = snapshot()
    assert "gauges" in snap
    assert "broker_active_bindings" in snap["gauges"]
    assert "broker_kek_age_days" in snap["gauges"]
    assert "uptime_seconds" in snap


def test_active_bindings_counts_real_rows(monkeypatch, tmp_path):
    """The gauge must reflect actual DB state."""
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "metrics.db")

    from brokers.credentials_store import store
    from brokers.base import AlpacaCredentials
    from brokers.metrics import snapshot

    assert snapshot()["gauges"]["broker_active_bindings"] == 0
    store.bind("u_42", "alpaca", "main",
               AlpacaCredentials(api_key="k", api_secret="s"),
               actor="user")
    assert snapshot()["gauges"]["broker_active_bindings"] == 1

    _db.close_default()


def test_credentials_store_increments_metrics(monkeypatch, tmp_path):
    """End-to-end: store.bind triggers metrics counter."""
    import os
    for name in list(os.environ):
        if name.startswith("BROKER_KEK_v"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROKER_KEK_v1", Fernet.generate_key().decode("ascii"))

    from brokers import _db, metrics
    _db.close_default()
    monkeypatch.setattr(_db, "_DEFAULT_DB_PATH", tmp_path / "metrics2.db")

    from brokers.credentials_store import store
    from brokers.base import AlpacaCredentials

    metrics.reset()
    store.bind("u_42", "alpaca", "main",
               AlpacaCredentials(api_key="k", api_secret="s"),
               actor="user")

    snap = metrics.snapshot()
    bind_ok = [c for c in snap["counters"]
               if c["metric"] == "broker_bind_total"
               and c["labels"] == {"broker": "alpaca", "result": "ok"}]
    assert bind_ok == [{"metric": "broker_bind_total",
                        "labels": {"broker": "alpaca", "result": "ok"},
                        "value": 1}]

    _db.close_default()
