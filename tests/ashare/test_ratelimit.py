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
    assert json.loads(open(p).read())["tokens"] < 1.0, "耗尽状态必须真的落盘"


def test_save_preserves_other_keys_in_state_file(tmp_path):
    """rate_state.json 同时承载 Task 0 的探测结果与 calls_per_min；
    令牌桶只能合并写自己的两个键，不得覆写整文件。"""
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"calls_per_min": 50, "probed_at": "x", "results": [1]}))
    b = TokenBucket(calls_per_min=50, state_path=str(p))
    b.acquire()
    d = json.loads(p.read_text())
    assert d["calls_per_min"] == 50 and d["probed_at"] == "x" and d["results"] == [1]
    assert "tokens" in d and "updated_at" in d


def test_tushare_call_retries_on_rate_limit_and_raises_on_permission(tmp_path, monkeypatch):
    """_call：分钟限频文案 → 等待重试；无权限 → 立刻抛；网络错误 → 退避重试后成功。"""
    from ashare.data.sources import tushare as ts_mod
    import pandas as pd
    if ts_mod._ts is None:
        import pytest as _pt; _pt.skip("tushare 未安装")
    monkeypatch.setattr(ts_mod.time, "sleep", lambda s: None)                 # 不真等
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy")
    monkeypatch.setattr(ts_mod._ts, "pro_api", lambda tok: object())
    src = ts_mod.TushareSource(state_path=str(tmp_path / "rs.json"))

    class Pro:
        def __init__(self, fails): self.fails = list(fails); self.calls = 0
        def daily(self, **kw):
            self.calls += 1
            if self.fails:
                raise RuntimeError(self.fails.pop(0))
            return pd.DataFrame({"trade_date": ["20240102"], "close": [1.0]})

    src._pro = Pro(["抱歉，您每分钟最多访问该接口500次，权限的具体详情访问：x", "connection reset"])
    out = src.daily(ts_code="600519.SH")
    assert len(out) == 1 and src._pro.calls == 3

    src._pro = Pro(["抱歉，您没有访问该接口的权限，权限的具体详情访问：x"])
    import pytest as _pt
    with _pt.raises(RuntimeError, match="没有访问该接口的权限"):
        src.daily(ts_code="600519.SH")
    assert src._pro.calls == 1                                                # 不重试

    src._pro = Pro(["e1", "e2", "e3", "e4"])                                   # 超过 MAX_RETRIES → 抛最后一个
    with _pt.raises(RuntimeError, match="e4"):
        src.daily(ts_code="600519.SH")
