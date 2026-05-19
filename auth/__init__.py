"""
auth — OAuth (Google + GitHub) 账户体系

模块组成：
  users.py    用户数据（Redis 存储）
  sessions.py 浏览器会话 token + Cookie 签发
  oauth.py    OAuth 客户端（authlib）
  deps.py     FastAPI 依赖注入：current_user / require_user
"""
from .users import User, get_user, upsert_user
from .sessions import (
    create_web_session, get_user_id_from_token, revoke_session,
    COOKIE_NAME, COOKIE_MAX_AGE,
)
from .deps import current_user, require_user
from .oauth import oauth_client, configured_providers

__all__ = [
    "User", "get_user", "upsert_user",
    "create_web_session", "get_user_id_from_token", "revoke_session",
    "COOKIE_NAME", "COOKIE_MAX_AGE",
    "current_user", "require_user",
    "oauth_client", "configured_providers",
]
