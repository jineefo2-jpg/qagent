"""
统一缓存层：Redis 优先，无 Redis 时降级到内存字典。

环境变量：
    REDIS_URL=redis://localhost:6379/0   （设置则用 Redis）
    （未设置或连接失败则用内存）

用法：
    from cache import cache
    cache.set("key", {"data": 1}, ttl=300)
    cache.get("key")
"""
import os
import json
import time
import fnmatch
from typing import Any, Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class MemoryCache:
    """单进程内存缓存（带 TTL）"""

    def __init__(self):
        self._data = {}  # {key: (value, expire_ts | None)}

    def get(self, key: str) -> Any:
        item = self._data.get(key)
        if not item:
            return None
        value, expire = item
        if expire is not None and time.time() > expire:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expire = (time.time() + ttl) if ttl else None
        self._data[key] = (value, expire)

    def delete(self, key: str):
        self._data.pop(key, None)

    def keys(self, pattern: str = "*") -> list:
        # 清掉过期再返回
        now = time.time()
        expired = [k for k, (_, e) in self._data.items()
                    if e is not None and e < now]
        for k in expired:
            self._data.pop(k, None)
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    def info(self) -> dict:
        return {"backend": "memory", "size": len(self._data)}


class RedisCache:
    """Redis 后端"""

    def __init__(self, client):
        self.r = client

    def get(self, key: str):
        v = self.r.get(key)
        if v is None:
            return None
        try:
            return json.loads(v)
        except (TypeError, json.JSONDecodeError):
            return v  # 直接返回原始字符串

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            data = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            data = str(value)
        if ttl:
            self.r.setex(key, ttl, data)
        else:
            self.r.set(key, data)

    def delete(self, key: str):
        self.r.delete(key)

    def keys(self, pattern: str = "*") -> list:
        return list(self.r.keys(pattern))

    def info(self) -> dict:
        try:
            return {
                "backend": "redis",
                "size": self.r.dbsize(),
                "url": self._url,
            }
        except Exception:
            return {"backend": "redis", "size": "?"}


def init_cache():
    """工厂：按 REDIS_URL 初始化，失败降级到内存"""
    url = os.getenv("REDIS_URL", "").strip()

    if url and _REDIS_AVAILABLE:
        try:
            client = _redis_lib.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_keepalive=True,
            )
            client.ping()
            cache_obj = RedisCache(client)
            cache_obj._url = url
            print(f"✅ Cache backend: Redis @ {url}")
            return cache_obj
        except Exception as e:
            print(f"⚠️  Redis 连接失败 ({type(e).__name__}: {e})，降级到内存缓存")

    if url and not _REDIS_AVAILABLE:
        print("⚠️  设置了 REDIS_URL 但未安装 redis 包，pip install redis")

    print("ℹ️  Cache backend: in-memory (单进程，不持久化)")
    return MemoryCache()


# 模块加载时初始化一次
cache = init_cache()
