"""
邮箱 + 6 位验证码登录（无需任何第三方 OAuth，国内可用）。

流程:
  1. POST /auth/email/send_code {email}
     → 生成 6 位码 → SMTP 发邮件 → Redis 存 (TTL 5 分钟)
  2. POST /auth/email/verify {email, code}
     → 校验 → upsert_user → 签 cookie

防滥用：
  - 同一邮箱 60s 内不能重发
  - 同一邮箱 1h 内最多 5 次
  - 失败验证 5 次后该 code 失效（强制重发）

SMTP 环境变量（任一邮箱服务商均可，本文件以 QQ 邮箱为例）:
  SMTP_HOST=smtp.qq.com
  SMTP_PORT=465
  SMTP_USER=you@qq.com
  SMTP_PASS=XXXXX        ← QQ 邮箱叫"授权码"，不是登录密码
  SMTP_FROM=QuantAgent <you@qq.com>   ← 可选；默认用 SMTP_USER
  SMTP_SSL=1             ← 1=SSL(465 端口), 0=STARTTLS(587 端口)
"""
from __future__ import annotations

import os
import re
import time
import random
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional, Tuple

from cache import cache


# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════

CODE_TTL = 300                # 验证码 5 分钟过期
RESEND_COOLDOWN = 60          # 60 秒重发冷却
HOURLY_SEND_LIMIT = 5         # 每小时同邮箱最多发 5 次
MAX_VERIFY_FAILS = 5          # 同邮箱验证失败 5 次后强制重发
EMAIL_PATTERN = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')


def _smtp_configured() -> bool:
    return all([
        os.getenv("SMTP_HOST", "").strip(),
        os.getenv("SMTP_USER", "").strip(),
        os.getenv("SMTP_PASS", "").strip(),
    ])


def is_email_login_enabled() -> bool:
    """对外：是否启用邮箱登录（前端按钮亮灭）"""
    return _smtp_configured()


def _valid_email(s: str) -> bool:
    return bool(EMAIL_PATTERN.match((s or "").strip().lower()))


# ════════════════════════════════════════════════════════════
# Redis key 模式
# ════════════════════════════════════════════════════════════

def _code_key(email: str) -> str:
    return f"quant:vcode:{email.lower()}"


def _cooldown_key(email: str) -> str:
    return f"quant:vcode_cd:{email.lower()}"


def _hourly_key(email: str) -> str:
    return f"quant:vcode_h:{email.lower()}:{int(time.time() // 3600)}"


# ════════════════════════════════════════════════════════════
# 发码
# ════════════════════════════════════════════════════════════

def request_code(email: str) -> Tuple[bool, str]:
    """
    生成并发送验证码。返回 (success, message)。
    """
    if not _valid_email(email):
        return False, "邮箱格式不正确"
    email = email.strip().lower()

    if not _smtp_configured():
        return False, "服务端未配置 SMTP，无法发送验证码"

    # ── 冷却检查 ──
    if cache.get(_cooldown_key(email)):
        return False, f"请求过于频繁，请 {RESEND_COOLDOWN} 秒后再试"

    # ── 每小时次数检查 ──
    hourly = int(cache.get(_hourly_key(email)) or 0)
    if hourly >= HOURLY_SEND_LIMIT:
        return False, f"该邮箱本小时已达发送上限（{HOURLY_SEND_LIMIT} 次）"

    # ── 生成 + 落库 ──
    code = f"{random.randint(0, 999999):06d}"
    cache.set(_code_key(email), {"code": code, "fails": 0}, ttl=CODE_TTL)
    cache.set(_cooldown_key(email), 1, ttl=RESEND_COOLDOWN)
    cache.set(_hourly_key(email), hourly + 1, ttl=3700)

    # ── SMTP 发送 ──
    try:
        _send_email(email, code)
    except Exception as e:
        # 发送失败：删掉冷却让用户能立刻重试
        cache.delete(_cooldown_key(email))
        return False, f"邮件发送失败: {e}"

    return True, f"验证码已发送，{CODE_TTL // 60} 分钟内有效"


def _send_email(to_email: str, code: str):
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "465").strip() or "465")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    use_ssl = os.getenv("SMTP_SSL", "1").strip().lower() in ("1", "true", "yes")
    from_addr = os.getenv("SMTP_FROM", "").strip() or f"QuantAgent <{user}>"

    msg = EmailMessage()
    msg["Subject"] = f"【QuantAgent】验证码 {code}"
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(
        f"您的 QuantAgent 登录验证码：\n\n"
        f"    {code}\n\n"
        f"5 分钟内有效。如非本人操作请忽略此邮件。\n"
    )
    # 简单 HTML 版本
    msg.add_alternative(f"""\
<!DOCTYPE html>
<html><body style="font-family:-apple-system,sans-serif;background:#f5f5f5;padding:20px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:8px;padding:28px;">
    <h2 style="margin:0 0 16px;color:#111;">QuantAgent 登录验证码</h2>
    <p style="color:#555;line-height:1.6;margin:0 0 18px;">您的登录验证码：</p>
    <div style="font-size:30px;letter-spacing:6px;font-weight:600;text-align:center;
                background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;
                padding:14px;color:#0369a1;margin-bottom:18px;">{code}</div>
    <p style="color:#888;font-size:13px;line-height:1.6;margin:0;">
      此验证码 5 分钟内有效，请勿向他人透露。<br>
      如非本人操作，请忽略此邮件。
    </p>
  </div>
</body></html>""", subtype="html")

    if use_ssl:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=15) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.ehlo()
            s.starttls(context=ssl.create_default_context())
            s.login(user, password)
            s.send_message(msg)


# ════════════════════════════════════════════════════════════
# 验码
# ════════════════════════════════════════════════════════════

def verify_code(email: str, code: str) -> Tuple[bool, str]:
    """
    校验验证码。成功 → 删 key 返回 True；失败 → 失败计数 +1。
    """
    if not _valid_email(email):
        return False, "邮箱格式不正确"
    if not code or not re.match(r'^\d{6}$', code.strip()):
        return False, "验证码必须是 6 位数字"

    email = email.strip().lower()
    code = code.strip()

    record = cache.get(_code_key(email))
    if not record:
        return False, "验证码已过期或不存在，请重新获取"

    stored_code = record.get("code") if isinstance(record, dict) else None
    fails = int(record.get("fails", 0)) if isinstance(record, dict) else 0

    if fails >= MAX_VERIFY_FAILS:
        cache.delete(_code_key(email))
        return False, "验证失败次数过多，请重新获取验证码"

    if stored_code != code:
        # 失败计数 +1（不重置 TTL，保留剩余有效时间）
        cache.set(_code_key(email),
                  {"code": stored_code, "fails": fails + 1},
                  ttl=CODE_TTL)
        return False, f"验证码错误（还可尝试 {MAX_VERIFY_FAILS - fails - 1} 次）"

    # 成功 → 消耗 code，防重放
    cache.delete(_code_key(email))
    return True, "验证成功"
