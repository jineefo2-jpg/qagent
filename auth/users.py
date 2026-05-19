"""
用户数据存储（Redis-backed）。

user_id 格式：{provider}:{provider_sub}
  例：google:1234567890   github:5678
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

from cache import cache


@dataclass
class User:
    user_id: str          # "google:..." / "github:..."
    email: str
    name: str
    avatar_url: str
    provider: str         # "google" / "github"
    created_at: float     # unix ts
    last_login_at: float

    def to_public_dict(self) -> dict:
        """暴露给前端的字段（不含敏感信息）"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "provider": self.provider,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "User":
        return cls(
            user_id=d["user_id"],
            email=d.get("email", ""),
            name=d.get("name", ""),
            avatar_url=d.get("avatar_url", ""),
            provider=d.get("provider", ""),
            created_at=float(d.get("created_at", 0)),
            last_login_at=float(d.get("last_login_at", 0)),
        )


def _user_key(user_id: str) -> str:
    return f"quant:user:{user_id}"


def get_user(user_id: str) -> Optional[User]:
    d = cache.get(_user_key(user_id))
    if not d:
        return None
    try:
        return User.from_dict(d)
    except Exception:
        return None


def upsert_user(
    provider: str,
    provider_sub: str,
    email: str,
    name: str,
    avatar_url: str,
) -> User:
    """
    OAuth 登录成功后的"建/更新"用户。已存在则刷新 last_login_at + 资料；
    不存在则新建。
    """
    user_id = f"{provider}:{provider_sub}"
    now = time.time()
    existing = get_user(user_id)
    if existing:
        existing.email = email or existing.email
        existing.name = name or existing.name
        existing.avatar_url = avatar_url or existing.avatar_url
        existing.last_login_at = now
        cache.set(_user_key(user_id), existing.to_dict(), ttl=None)
        return existing

    user = User(
        user_id=user_id,
        email=email or "",
        name=name or "",
        avatar_url=avatar_url or "",
        provider=provider,
        created_at=now,
        last_login_at=now,
    )
    cache.set(_user_key(user_id), user.to_dict(), ttl=None)
    return user
