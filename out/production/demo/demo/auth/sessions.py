"""
Web Session（浏览器 cookie 会话）。

设计：
  - cookie 名 qa_auth
  - 值 = 随机 32 字节 hex token（不放用户 ID，避免泄露后映射 = 暴露用户）
  - token → user_id 的映射存 Redis，TTL 7 天
  - 退出登录：删 Redis 映射 + Set-Cookie 清空
"""
from __future__ import annotations

import secrets
from typing import Optional

from cache import cache


COOKIE_NAME = "qa_auth"
COOKIE_MAX_AGE = 7 * 86400        # 7 天


def _session_key(token: str) -> str:
    return f"quant:web_session:{token}"


def create_web_session(user_id: str) -> str:
    """生成新 token 并落 Redis；返回 token（调用方塞进 Set-Cookie）"""
    token = secrets.token_urlsafe(32)
    cache.set(_session_key(token), {"user_id": user_id}, ttl=COOKIE_MAX_AGE)
    return token


def get_user_id_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    d = cache.get(_session_key(token))
    if not d:
        return None
    return d.get("user_id")


def revoke_session(token: str) -> None:
    if token:
        cache.delete(_session_key(token))
