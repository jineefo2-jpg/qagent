"""
In-process metrics for the broker subsystem (ADR-0001 §8).

Tiny on purpose: a dict of counters + a few gauges computed on demand.
No external monitoring dependency. If/when the deployment grows beyond
"single host" the easy migration is to swap `incr()`'s body for a
`prometheus_client.Counter.labels(...).inc()` call.

Public surface:
    incr(metric, **labels)   — atomically bump a labeled counter
    snapshot()               — return all current counters + gauges
    reset()                  — testing only

Standard metric names (kept in sync with ADR-0001 §8):
    broker_bind_total      labels={broker, result}
    broker_use_total       labels={broker, result}
    broker_auth_fail_total labels={broker}
    broker_unbind_total    labels={broker, result}
    broker_rotate_total    labels={result}
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


_lock = threading.Lock()
_counters: Dict[Tuple[Any, ...], int] = {}
_started_at = int(time.time())


# ════════════════════════════════════════════════════════════
# Counters
# ════════════════════════════════════════════════════════════

def incr(metric: str, **labels: str) -> None:
    """Atomically increment `metric{labels}` by 1."""
    key = (metric,) + tuple(sorted(labels.items()))
    with _lock:
        _counters[key] = _counters.get(key, 0) + 1


def reset() -> None:
    """Test helper. Production code MUST NOT call this."""
    with _lock:
        _counters.clear()


# ════════════════════════════════════════════════════════════
# Gauges (computed on demand — no background thread)
# ════════════════════════════════════════════════════════════

def _active_bindings_count() -> int:
    """Total non-deleted bindings across all users."""
    try:
        from . import _db
        c = _db.init()
        row = c.execute("SELECT COUNT(*) FROM broker_bindings").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return -1


def _kek_age_days() -> int:
    """
    Days since the *current* KEK was first observed. We don't track KEK
    creation time directly, so we infer from the earliest binding wrapped by
    the current KEK version. Returns -1 if KEK env / DB is unavailable.
    """
    try:
        from . import _db, crypto
        current_version, _ = crypto._current_kek()
        c = _db.init()
        row = c.execute(
            "SELECT MIN(created_at) FROM broker_bindings WHERE kek_version = ?",
            (current_version,),
        ).fetchone()
        if not row or row[0] is None:
            return 0  # KEK is set but no bindings yet
        age_sec = int(time.time()) - int(row[0])
        return max(0, age_sec // 86400)
    except Exception:
        return -1


# ════════════════════════════════════════════════════════════
# Snapshot
# ════════════════════════════════════════════════════════════

def snapshot() -> dict:
    """Return the full metrics state as a JSON-safe dict."""
    with _lock:
        items: List[dict] = []
        for key, value in _counters.items():
            metric = key[0]
            labels = dict(key[1:])
            items.append({"metric": metric, "labels": labels, "value": value})

    return {
        "counters": items,
        "gauges": {
            "broker_active_bindings": _active_bindings_count(),
            "broker_kek_age_days": _kek_age_days(),
        },
        "uptime_seconds": int(time.time() - _started_at),
    }
