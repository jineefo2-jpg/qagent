"""
brokers/realtime_state.py — per-user state + pub/sub contract.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from queue import Empty

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def store():
    from brokers.realtime_state import RealtimeStateStore
    return RealtimeStateStore()


# ─────────────────────────────────────────────────────────────────────────────
# Basic mutation + broadcast
# ─────────────────────────────────────────────────────────────────────────────

def test_update_account_broadcasts_to_subscriber(store):
    q = store.subscribe("u:alice")
    store.update_account("u:alice", {"cash": 100000, "equity": 100000})
    event_type, payload = q.get(timeout=1)
    assert event_type == "account"
    assert payload["cash"] == 100000


def test_update_position_keeps_latest(store):
    store.subscribe("u:alice")  # required so the store creates the slot
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 10, "market_value": 1500})
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 20, "market_value": 3000})
    snap = store.snapshot("u:alice")
    assert len(snap["positions"]) == 1
    assert snap["positions"][0]["qty"] == 20


def test_update_position_qty_zero_removes(store):
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 10})
    assert len(store.snapshot("u:alice")["positions"]) == 1
    # close the position
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 0})
    assert len(store.snapshot("u:alice")["positions"]) == 0


def test_update_order_keyed_by_broker_order_id(store):
    store.update_order("u:alice", {
        "broker_order_id": "12345", "symbol": "AAPL", "status": "new",
    })
    store.update_order("u:alice", {
        "broker_order_id": "12345", "symbol": "AAPL", "status": "filled",
    })
    snap = store.snapshot("u:alice")
    assert len(snap["orders"]) == 1
    assert snap["orders"][0]["status"] == "filled"


def test_update_quote_keeps_latest_per_symbol(store):
    store.update_quote("u:alice", {"symbol": "AAPL", "price": 150})
    store.update_quote("u:alice", {"symbol": "AAPL", "price": 151})
    store.update_quote("u:alice", {"symbol": "GOOG", "price": 200})
    snap = store.snapshot("u:alice")
    assert snap["quote_count"] == 2  # AAPL + GOOG


def test_update_with_missing_required_field_is_noop(store):
    """Defensive: bad payload from PushClient shouldn't blow up.
    These no-op updates don't even create the user_scope entry, which is
    fine — the snapshot just reports `exists: False`."""
    store.update_position("u:alice", {"qty": 10})    # no symbol
    store.update_order("u:alice", {"status": "new"}) # no id
    store.update_quote("u:alice", {"price": 150})    # no symbol
    snap = store.snapshot("u:alice")
    assert snap["exists"] is False

    # And once a real update lands, the state is correct (no leaked junk)
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 10})
    snap2 = store.snapshot("u:alice")
    assert snap2["exists"] is True
    assert len(snap2["positions"]) == 1
    assert snap2["orders"] == []
    assert snap2["quote_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Subscription semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_subscribe_returns_initial_snapshot(store):
    """New subscribers must receive the last-known state immediately so UI
    has something to render before the first push tick arrives."""
    store.update_account("u:alice", {"cash": 100000})
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 10})
    store.update_position("u:alice", {"symbol": "GOOG", "qty": 5})

    q = store.subscribe("u:alice")
    received = []
    while True:
        try:
            received.append(q.get_nowait())
        except Empty:
            break

    types = [t for t, _ in received]
    assert types.count("account") == 1
    assert types.count("position") == 2


def test_multiple_subscribers_all_receive_updates(store):
    q1 = store.subscribe("u:alice")
    q2 = store.subscribe("u:alice")
    store.update_account("u:alice", {"cash": 50000})
    e1 = q1.get(timeout=1)
    e2 = q2.get(timeout=1)
    assert e1 == e2 == ("account", {"cash": 50000})


def test_unsubscribe_removes_queue(store):
    q = store.subscribe("u:alice")
    assert store.subscriber_count("u:alice") == 1
    store.unsubscribe("u:alice", q)
    assert store.subscriber_count("u:alice") == 0


def test_unsubscribe_unknown_is_noop(store):
    from queue import Queue
    store.unsubscribe("u:never_existed", Queue())  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Per-user isolation (critical security invariant)
# ─────────────────────────────────────────────────────────────────────────────

def test_users_dont_see_each_others_updates(store):
    qa = store.subscribe("u:alice")
    qb = store.subscribe("u:bob")
    store.update_account("u:alice", {"cash": 100000})
    # Alice's subscriber gets the event
    assert qa.get(timeout=1) == ("account", {"cash": 100000})
    # Bob's subscriber gets nothing
    with pytest.raises(Empty):
        qb.get(timeout=0.1)


def test_subscribe_for_user_b_does_not_leak_user_a_initial_snapshot(store):
    store.update_account("u:alice", {"cash": 100000})
    store.update_position("u:alice", {"symbol": "AAPL", "qty": 10})
    qb = store.subscribe("u:bob")
    with pytest.raises(Empty):
        qb.get(timeout=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness: slow consumer doesn't block producer
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_consumer_gets_dropped(store):
    """If a subscriber's queue fills up, we drop them so the push thread
    never blocks."""
    q = store.subscribe("u:alice", queue_max=3)
    # Fill the queue: account at subscribe time is None so no initial.
    for i in range(10):
        store.update_account("u:alice", {"cash": i})
    # The slow consumer should now be removed.
    assert store.subscriber_count("u:alice") == 0


def test_fast_consumer_unaffected_by_slow_consumer(store):
    slow = store.subscribe("u:alice", queue_max=2)
    fast = store.subscribe("u:alice", queue_max=200)

    for i in range(20):
        store.update_account("u:alice", {"cash": i})

    # Fast consumer should have everything
    received = []
    while True:
        try:
            received.append(fast.get_nowait())
        except Empty:
            break
    assert len(received) == 20

    # Slow consumer was removed; producer never blocked.
    assert store.subscriber_count("u:alice") == 1  # only fast remains


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safety smoke (best-effort, hits multiple writers + readers)
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_updates_dont_lose_state(store):
    """Two writer threads, one reader thread, no exceptions, final count exact."""
    q = store.subscribe("u:alice", queue_max=10000)

    def writer(start):
        for i in range(50):
            store.update_position("u:alice", {
                "symbol": f"SYM{start + i}", "qty": 10,
            })

    t1 = threading.Thread(target=writer, args=(0,))
    t2 = threading.Thread(target=writer, args=(1000,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    snap = store.snapshot("u:alice")
    # 100 distinct symbols → 100 positions
    assert len(snap["positions"]) == 100
