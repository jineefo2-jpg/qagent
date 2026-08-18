from __future__ import annotations
import json, time
from ashare.data.sources._ratelimit import TokenBucket


def test_bucket_allows_burst_up_to_capacity(tmp_path):
    b = TokenBucket(calls_per_min=60, state_path=str(tmp_path / "s.json"))
    t0 = time.time()
    for _ in range(5):
        b.acquire()
    assert time.time() - t0 < 0.5, "首批调用不应被限速"


def test_bucket_throttles_beyond_rate(tmp_path):
    b = TokenBucket(calls_per_min=60, state_path=str(tmp_path / "s.json"), capacity=2)
    b.acquire(); b.acquire()
    t0 = time.time()
    b.acquire()                       # 第 3 次必须等 ≈1s（60/min → 1 token/s）
    assert time.time() - t0 >= 0.9


def test_bucket_state_persists_across_instances(tmp_path):
    """跨进程复用：ingest 分批跑，第二个进程不能把配额当全新的。"""
    p = str(tmp_path / "s.json")
    b1 = TokenBucket(calls_per_min=60, state_path=p, capacity=2)
    b1.acquire(); b1.acquire()
    b2 = TokenBucket(calls_per_min=60, state_path=p, capacity=2)
    t0 = time.time()
    b2.acquire()
    assert time.time() - t0 >= 0.9, "新实例必须读到已消耗的配额"
    assert json.loads(open(p).read())["tokens"] >= 0
