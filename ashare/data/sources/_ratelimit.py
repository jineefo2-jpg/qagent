# ashare/data/sources/_ratelimit.py
"""持久化令牌桶。Tushare 限频按账号算，跨进程共享 —— 状态必须落盘，
否则分批 ingest 时第二个进程会把配额当全新的，直接撞限频。"""
from __future__ import annotations
import json, pathlib, time


class TokenBucket:
    def __init__(self, calls_per_min: int, state_path: str, capacity: int | None = None) -> None:
        self.rate = calls_per_min / 60.0          # tokens/sec
        self.capacity = float(capacity if capacity is not None else calls_per_min)
        self.state_path = pathlib.Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> tuple[float, float]:
        try:
            d = json.loads(self.state_path.read_text())
            return float(d["tokens"]), float(d["updated_at"])
        except Exception:                          # 首次运行 / 文件损坏 → 满桶
            return self.capacity, time.time()

    def _save(self, tokens: float, now: float) -> None:
        self.state_path.write_text(json.dumps({"tokens": tokens, "updated_at": now}))

    def acquire(self) -> None:
        """阻塞直到拿到一个 token。"""
        while True:
            tokens, updated = self._load()
            now = time.time()
            tokens = min(self.capacity, tokens + (now - updated) * self.rate)
            if tokens >= 1.0:
                self._save(tokens - 1.0, now)
                return
            self._save(tokens, now)
            time.sleep(max(0.05, (1.0 - tokens) / self.rate))
