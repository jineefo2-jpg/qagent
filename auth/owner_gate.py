"""单所有者锁定（Owner Lock，2026-08-26）。

整个部署只允许一个固定账号 + 一个固定访问密钥使用：
  - 账号：OWNER_EMAIL（默认 jineefo2@gmail.com，环境变量可换人，**不能置空**）
  - 密钥：AGENT_ACCESS_KEY（服务端环境变量；前端登录时输入一次、存浏览器，
    之后每个 /api 请求经 X-Agent-Key 头带上）

三道闸，缺一不可：
  1. 登录入口（邮箱验证码 / OAuth 回调）拒绝给非所有者发码 / 建会话 —— UX 层，
     错误信息友好；
  2. 本模块的 HTTP 中间件 —— **最后一道、全局单点**：受保护路径上逐请求校验
     「密钥一致（常量时间比较）∧ 会话用户是所有者」，任何新加路由自动被罩住，
     忘了在路由上挂 Depends 也越不过去；
  3. AGENT_ACCESS_KEY 未配置 → 全部拒绝（fail-closed），错误信息说清缺什么。
     生成：python3 -c "import secrets; print(secrets.token_urlsafe(32))"

env 逐请求读取而非 import 时固化：测试 monkeypatch.setenv 即生效，运维换 key
不用改代码重启逻辑。开销是一次 dict 查找，忽略不计。
"""
from __future__ import annotations

import hmac
import os

from fastapi.responses import JSONResponse

from .deps import current_user

# 无需密钥/登录即可访问的路径：页面壳 + 静态资源 + 登录流程自身 +
# 前端判断登录态所需的两个只读端点。/docs 与 /openapi.json 不在此列（同样上锁）。
_OPEN_EXACT = {"/", "/brokers", "/favicon.ico", "/api/me", "/api/auth/providers"}
_OPEN_PREFIX = ("/static/", "/auth/")

KEY_HEADER = "x-agent-key"


def owner_email() -> str:
    return os.environ.get("OWNER_EMAIL", "jineefo2@gmail.com").strip().lower()


def is_owner(email: str | None) -> bool:
    return (email or "").strip().lower() == owner_email()


def _is_open(path: str) -> bool:
    return path in _OPEN_EXACT or path.startswith(_OPEN_PREFIX)


def key_ok(supplied: str | None) -> bool:
    """常量时间比较访问密钥。AGENT_ACCESS_KEY 未配置 → 恒 False（fail-closed）。"""
    real = os.environ.get("AGENT_ACCESS_KEY", "")
    return bool(real) and hmac.compare_digest((supplied or "").encode(), real.encode())


def check_request(path: str, supplied_key: str | None, cookie_token: str | None):
    """纯判定：放行返回 None，拒绝返回 (status_code, error_msg)。"""
    if _is_open(path):
        return None
    if not os.environ.get("AGENT_ACCESS_KEY", ""):
        return 503, ("服务端未配置 AGENT_ACCESS_KEY，所有功能已锁定（fail-closed）。"
                     "生成: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\" "
                     "后写入 .env 并重启")
    if not key_ok(supplied_key):
        return 403, "访问密钥缺失或不正确（X-Agent-Key）"
    user = current_user(qa_auth=cookie_token)
    if user is None:
        return 401, "需要登录"
    if not is_owner(user.email):
        return 403, f"此部署仅限所有者账号使用，{user.email} 无权访问"
    return None


def install(app) -> None:
    """挂到 FastAPI app 上。放在其他中间件之后注册 = 请求时最先执行。"""

    @app.middleware("http")
    async def _owner_gate(request, call_next):
        # key 取头或 cookie（二选一即可）：EventSource 发不了自定义头，只能靠 cookie 带
        supplied = request.headers.get(KEY_HEADER) or request.cookies.get("agent_key")
        verdict = check_request(request.url.path, supplied,
                                request.cookies.get("qa_auth"))
        if verdict is not None:
            status, msg = verdict
            return JSONResponse(status_code=status, content={"error": msg, "owner_lock": True})
        return await call_next(request)
