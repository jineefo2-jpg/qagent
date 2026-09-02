"""
OAuth 客户端（Authlib）。

支持 Google + GitHub。当 .env 没配某家的 ClientID/Secret 时，
configured_providers() 不会返回该家，前端按钮自动变灰。
"""
from __future__ import annotations

import os
from typing import List

# 防御性 dotenv 加载：当本模块被 server.py 之外的入口 import 时
# （CLI 测试、单元测试、Celery worker 等）也能拿到 .env 里的 OAuth 凭证。
# 已加载的 env 不会被覆盖（override=False），所以 server.py 的早期加载仍然优先。
try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path
    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from authlib.integrations.starlette_client import OAuth


# 全局唯一 OAuth 客户端
oauth_client = OAuth()


def _register_google():
    cid = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if not (cid and secret):
        return False
    oauth_client.register(
        name="google",
        client_id=cid,
        client_secret=secret,
        # OIDC discovery
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return True


def _register_github():
    cid = os.getenv("GITHUB_CLIENT_ID", "").strip()
    secret = os.getenv("GITHUB_CLIENT_SECRET", "").strip()
    if not (cid and secret):
        return False
    oauth_client.register(
        name="github",
        client_id=cid,
        client_secret=secret,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )
    return True


# 模块加载时注册（缺凭证的自动跳过）
_GOOGLE_OK = _register_google()
_GITHUB_OK = _register_github()


def configured_providers() -> List[str]:
    """返回当前可用的 OAuth 提供方列表（供前端渲染按钮）"""
    out = []
    if _GOOGLE_OK:
        out.append("google")
    if _GITHUB_OK:
        out.append("github")
    return out


def redirect_base() -> str:
    """构造 callback URL 用的 base，默认 localhost:8001"""
    return os.getenv("OAUTH_REDIRECT_BASE", "http://localhost:8001").rstrip("/")
