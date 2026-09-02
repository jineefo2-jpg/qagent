"""
FastAPI 依赖注入：从 cookie 解析当前登录用户。

current_user:   可选；未登录返回 None
require_user:   必需；未登录抛 401（用在交易类路由）
"""
from __future__ import annotations

from typing import Optional

from fastapi import Cookie, HTTPException, status

from .sessions import COOKIE_NAME, get_user_id_from_token
from .users import User, get_user


def current_user(
    qa_auth: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> Optional[User]:
    """
    用法：def my_route(user = Depends(current_user)):
      - 未登录 → user is None
      - 登录   → user 是 User 对象
    """
    if not qa_auth:
        return None
    user_id = get_user_id_from_token(qa_auth)
    if not user_id:
        return None
    return get_user(user_id)


def require_user(
    qa_auth: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> User:
    """登录强校验：未登录直接 401。给交易路由用。"""
    user = current_user(qa_auth=qa_auth)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要登录才能访问交易功能",
        )
    return user
