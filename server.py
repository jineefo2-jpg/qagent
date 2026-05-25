# ============================================================
#  QuantAgent Web Server
#  ─────────────────────────────────────────────
#  FastAPI 服务 + SSE 流式响应 + 会话管理
#
#  启动：
#    pip install -r requirements.txt
#    python server.py
#  访问：http://localhost:8001
# ============================================================

# ⚠️ 必须最先加载 .env：auth/oauth.py 在 import 时就读 GOOGLE_CLIENT_ID
# 等环境变量；如果 dotenv 比它晚执行，OAuth 注册会拿到空字符串。
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

import json
import uuid
import hashlib
import asyncio
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Depends
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import os as _os
import inspect as _inspect
from cache import cache  # Redis 或内存缓存

# OAuth 账户体系
from auth import (
    User, current_user, require_user, require_admin,
    oauth_client, configured_providers,
    upsert_user, create_web_session, get_user_id_from_token,
    revoke_session, COOKIE_NAME, COOKIE_MAX_AGE,
    is_email_login_enabled, email_request_code, email_verify_code,
)
from auth.oauth import redirect_base

# ── 入口结果缓存配置 ──
# 同样的 query+上下文，5 分钟内直接重放（跳过 LLM）
_ANSWER_CACHE_TTL = int(_os.getenv("ANSWER_CACHE_TTL", "300"))
_ANSWER_CACHE_ENABLE = _os.getenv("ANSWER_CACHE_ENABLE", "1").lower() in ("1", "true", "yes")
# 是否缓存包含错误的回复（默认不缓存，避免把异常态固化）
_ANSWER_CACHE_SKIP_ON_ERROR = True

# 框架切换：USE_LANGGRAPH=1 启用 LangGraph 版主循环
_USE_LG = _os.getenv("USE_LANGGRAPH", "").strip().lower() in ("1", "true", "yes", "on")

if _USE_LG:
    from quant_agent_lg import stream_quant_agent_lg as stream_quant_agent
    print("🌐 Agent 主循环: LangGraph 版（quant_agent_lg，async token 流式）")
else:
    from quant_agent import stream_quant_agent
    print("⚙️  Agent 主循环: 原生 ReAct 版（quant_agent）")

# 让 broker_account / place_order_intent / cancel_order 工具能拿到当前请求的 device_id + 鉴权状态
from quant_agent import _set_request_device_id, _set_request_authenticated

# 检测 stream_quant_agent 是同步还是异步 generator
_IS_ASYNC_AGENT = _inspect.isasyncgenfunction(stream_quant_agent)

# 当 X-Device-Id 缺失时的兜底（不应该出现，前端总会发）
_DEFAULT_DEVICE = "default"


def _scope_id(user: Optional[User], x_device_id: Optional[str]) -> str:
    """
    会话作用域标识。登录用户用 user_id（跨设备共享），匿名用 device_id。
    在 Redis key 里就用这一串，存储层无感知。
    """
    if user is not None:
        return f"u:{user.user_id}"
    return x_device_id or _DEFAULT_DEVICE


def _is_user_authenticated(user: Optional[User]) -> bool:
    return user is not None


def _setup_broker_context(user: Optional[User],
                          x_device_id: Optional[str] = None):
    """
    在调任意 broker API 之前必须先注入 user namespace（MockAdapter 靠这个隔离用户）。
    chat 路由里已经设过，但 broker_status/orders/positions 等独立路由也得设。
    """
    _set_request_device_id(_scope_id(user, x_device_id))
    _set_request_authenticated(_is_user_authenticated(user))

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="QuantAgent Web", version="1.0")

# SessionMiddleware 给 authlib 用（存储 OAuth state，防 CSRF）
# 注：与我们自己的 quant:session 用途不同，这是 starlette 内置的 cookie session
_APP_SECRET = _os.getenv("APP_SECRET_KEY", "").strip()
if not _APP_SECRET:
    _APP_SECRET = secrets.token_hex(32)
    print("⚠️  未设置 APP_SECRET_KEY，本次启动生成临时密钥（重启后 OAuth state 会失效）")
app.add_middleware(SessionMiddleware, secret_key=_APP_SECRET,
                    max_age=600, same_site="lax")


# ════════════════════════════════════════════════════════════
# 启动预热：把首次请求的高延迟成本前置到启动阶段
#   - bge 嵌入模型加载（~3-5s）
#   - DeepSeek HTTPS 连接建立 + TLS 握手（~300-800ms）
# 受环境变量 WARMUP=0 关闭
# ════════════════════════════════════════════════════════════

@app.on_event("startup")
async def _warmup():
    if _os.getenv("WARMUP", "1").lower() in ("0", "false", "no"):
        print("⚪ 启动预热已禁用 (WARMUP=0)")
        return

    import time as _t
    t0 = _t.monotonic()

    # 1. 预热 bge 嵌入模型 + bge-reranker（首次 search 不再卡）
    async def _warm_embedder():
        try:
            loop = asyncio.get_event_loop()
            def _load():
                from rag.indexer import get_embedder, get_reranker
                emb = get_embedder()
                emb.encode(["预热"], normalize_embeddings=True)
                # 同步加载 reranker（cross-encoder）
                rr = get_reranker()
                if rr is not None:
                    rr.predict([("预热查询", "预热文档")], show_progress_bar=False)
                return True
            await loop.run_in_executor(None, _load)
            print(f"🔥 嵌入 + 精排模型预热完成 ({_t.monotonic()-t0:.1f}s)")
        except Exception as e:
            print(f"⚠️  嵌入模型预热失败（不影响主流程）: {e}")

    # 2. 预热 DeepSeek 连接（TLS + 连接池）
    async def _warm_llm():
        try:
            loop = asyncio.get_event_loop()
            def _ping():
                from quant_agent import client, DEEPSEEK_MODEL
                client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    timeout=10,
                )
                return True
            await loop.run_in_executor(None, _ping)
            print(f"🔥 DeepSeek 连接预热完成")
        except Exception as e:
            print(f"⚠️  DeepSeek 预热失败（不影响主流程）: {e}")

    # 并发执行（互不阻塞）
    await asyncio.gather(_warm_embedder(), _warm_llm())
    print(f"✅ 启动预热全部完成，总耗时 {_t.monotonic()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# 会话存储（Redis 或内存，跨进程/重启持久）
# ════════════════════════════════════════════════════════════

SESSION_TTL = 86400 * 7    # 会话保留 7 天


def _index_key(device_id: str) -> str:
    """每个设备一份会话 ID 索引"""
    return f"quant:session:index:{device_id or _DEFAULT_DEVICE}"


class Session:
    """单个会话：LLM 消息历史 + UI 展示历史，全部 JSON 可序列化"""

    def __init__(self, sid: str, title: str = "新对话"):
        self.id = sid
        self.title = title
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.messages = []   # DeepSeek/OpenAI API 历史
        self.display = []    # UI 展示历史

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "messages": self.messages,
            "display": self.display,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        s = cls(data["id"], data.get("title", "新对话"))
        s.created_at = data.get("created_at", s.created_at)
        s.messages = data.get("messages", [])
        s.display = data.get("display", [])
        return s

    def to_summary(self):
        return {
            "id": self.id, "title": self.title,
            "created_at": self.created_at,
            "message_count": len(self.display),
        }

    def to_detail(self):
        return {
            "id": self.id, "title": self.title,
            "created_at": self.created_at,
            "display": self.display,
        }


def _session_key(device_id: str, sid: str) -> str:
    """每个设备的会话存储 key"""
    return f"quant:session:{device_id or _DEFAULT_DEVICE}:{sid}"


def save_session(device_id: str, s: Session):
    """持久化到 Redis/内存，同时维护该设备的索引"""
    cache.set(_session_key(device_id, s.id), s.to_dict(), ttl=SESSION_TTL)
    idx_key = _index_key(device_id)
    index = cache.get(idx_key) or []
    if s.id not in index:
        index.append(s.id)
        cache.set(idx_key, index, ttl=SESSION_TTL)


def load_session(device_id: str, sid: str) -> Optional[Session]:
    data = cache.get(_session_key(device_id, sid))
    if not data:
        return None
    return Session.from_dict(data)


def get_session(device_id: str, sid: str) -> Session:
    s = load_session(device_id, sid)
    if not s:
        raise HTTPException(404, f"Session {sid} not found")
    return s


def delete_session_storage(device_id: str, sid: str):
    cache.delete(_session_key(device_id, sid))
    idx_key = _index_key(device_id)
    index = cache.get(idx_key) or []
    if sid in index:
        index.remove(sid)
        cache.set(idx_key, index, ttl=SESSION_TTL)


# ════════════════════════════════════════════════════════════
# 入口结果缓存（重放）
# ════════════════════════════════════════════════════════════

def _answer_cache_key(user_msg: str, prior_messages: list) -> str:
    """
    根据 (本次用户消息 + 最近 3 轮上下文) 计算 hash。
    上下文加进来：避免不同会话语境下问"价格"返回错回答。
    """
    # 取最后 3 条非 system 消息作为上下文指纹
    tail = []
    for m in prior_messages[-6:]:   # 最多看 6 条（≈3 轮）
        if m.get("role") == "system":
            continue
        role = m.get("role", "")
        content = m.get("content", "") or ""
        # tool 消息只取 name，避免大 JSON 影响 hash 稳定性
        if role == "tool":
            tail.append(f"tool:{m.get('name', '')}")
        else:
            tail.append(f"{role}:{content[:200]}")
    fingerprint = "\n".join(tail) + "\n@@" + user_msg
    h = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    mode = "lg" if _USE_LG else "react"
    return f"quant:answer:{mode}:{h}"


def _is_cacheable_turn(turn: dict) -> bool:
    """只缓存 完整、无错误、有最终内容 的回复"""
    if not turn.get("content"):
        return False
    if _ANSWER_CACHE_SKIP_ON_ERROR:
        # 工具有错误时不缓存（用户可能想重试看新结果）
        if any(tc.get("is_error") for tc in turn.get("tool_calls", [])):
            return False
        if turn.get("content", "").startswith("❌"):
            return False
    return True


def _replay_events_from_turn(turn: dict):
    """
    把已缓存的 assistant_turn 转成与实时流相同的事件序列。
    顺序：tool_call/tool_result 各一对 → content_delta（一次性整段）→ final → suggestions
    """
    # 1. 工具调用回放
    for tc in turn.get("tool_calls", []) or []:
        yield {
            "type": "tool_call",
            "name": tc.get("name", ""),
            "input": tc.get("input", {}),
            "id": tc.get("id", ""),
            "cached": True,
        }
        yield {
            "type": "tool_result",
            "name": tc.get("name", ""),
            "result": tc.get("result") or {},
            "is_error": tc.get("is_error", False),
            "cached": True,
        }
    # 2. 最终内容（一次性发出，前端 _raw 直接拿到完整文本）
    content = turn.get("content", "") or ""
    if content:
        yield {"type": "content_reset"}
        yield {"type": "content_delta", "text": content, "iteration": turn.get("iterations", 0)}
    # 3. final
    yield {
        "type": "final",
        "text": content,
        "text_html": turn.get("content_html", ""),
        "iterations": turn.get("iterations", 0),
        "cached": True,
    }
    # 4. 建议
    if turn.get("suggestions"):
        yield {"type": "suggestions", "items": turn["suggestions"]}


# ════════════════════════════════════════════════════════════
# API 路由
# ════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    attachment: Optional[dict] = None  # {"name", "size_kb", "chunks"}


@app.get("/api/sessions")
def list_sessions(x_device_id: Optional[str] = Header(None),
                   user: Optional[User] = Depends(current_user)):
    """列出当前作用域（登录用户 or 匿名设备）的会话"""
    device_id = _scope_id(user, x_device_id)
    idx_key = _index_key(device_id)
    index = cache.get(idx_key) or []
    items = []
    stale = []
    for sid in index:
        s = load_session(device_id, sid)
        if s:
            items.append(s.to_summary())
        else:
            stale.append(sid)
    if stale:
        fresh = [i for i in index if i not in stale]
        cache.set(idx_key, fresh, ttl=SESSION_TTL)
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@app.post("/api/sessions")
def create_session(x_device_id: Optional[str] = Header(None),
                    user: Optional[User] = Depends(current_user)):
    """新建会话，绑定到当前作用域"""
    device_id = _scope_id(user, x_device_id)
    sid = uuid.uuid4().hex[:10]
    session = Session(sid)
    save_session(device_id, session)
    return session.to_detail()


@app.get("/api/sessions/{sid}")
def session_detail(sid: str, x_device_id: Optional[str] = Header(None),
                    user: Optional[User] = Depends(current_user)):
    device_id = _scope_id(user, x_device_id)
    return get_session(device_id, sid).to_detail()


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str, x_device_id: Optional[str] = Header(None),
                    user: Optional[User] = Depends(current_user)):
    device_id = _scope_id(user, x_device_id)
    delete_session_storage(device_id, sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, body: ChatRequest,
         x_device_id: Optional[str] = Header(None),
         user: Optional[User] = Depends(current_user)):
    """发送消息，SSE 流式返回 Agent 执行过程"""
    device_id = _scope_id(user, x_device_id)
    authenticated = _is_user_authenticated(user)
    # 让交易工具能感知当前请求的身份（设备 + 是否登录）
    _set_request_device_id(device_id)
    _set_request_authenticated(authenticated)
    session = get_session(device_id, sid)

    if not body.message.strip():
        raise HTTPException(400, "message 不能为空")

    # 首条用户消息作为会话标题
    if not session.display:
        session.title = body.message[:30] + ("..." if len(body.message) > 30 else "")

    # ── 入口结果缓存：在追加用户消息前计算 key（基于历史上下文 + 本次消息）──
    cache_key = (_answer_cache_key(body.message, session.messages)
                 if _ANSWER_CACHE_ENABLE else None)
    cached_turn = cache.get(cache_key) if cache_key else None

    # 1. 用户消息进入两条历史
    session.messages.append({"role": "user", "content": body.message})
    display_user = {
        "role": "user",
        "content": body.message,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if body.attachment:
        display_user["attachment"] = body.attachment
    session.display.append(display_user)

    # 用户消息阶段先持久化一次（防止网络中断丢失）
    save_session(device_id, session)

    # 2. 准备本轮 assistant 的展示记录（边流边填充）
    assistant_turn = {
        "role": "assistant",
        "content": "",
        "thoughts": [],
        "tool_calls": [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iterations": 0,
    }

    def _process_event(event):
        """把 stream 事件归档到 assistant_turn，并返回要 yield 的 SSE 字符串"""
        if event["type"] == "thought":
            assistant_turn["thoughts"].append({
                "iteration": event.get("iteration", 0),
                "text": event["text"],
            })
        elif event["type"] == "tool_call":
            assistant_turn["tool_calls"].append({
                "name": event["name"],
                "input": event["input"],
                "id": event["id"],
                "result": None,
                "is_error": False,
            })
        elif event["type"] == "tool_result":
            for tc in reversed(assistant_turn["tool_calls"]):
                if (tc["name"] == event["name"]
                        and tc["result"] is None):
                    tc["result"] = event["result"]
                    tc["is_error"] = event["is_error"]
                    break
        elif event["type"] == "final":
            assistant_turn["content"] = event["text"]
            if event.get("text_html"):
                assistant_turn["content_html"] = event["text_html"]
            assistant_turn["iterations"] = event.get("iterations", 0)
        elif event["type"] == "suggestions":
            assistant_turn["suggestions"] = event["items"]
        elif event["type"] == "error":
            assistant_turn["content"] = f"❌ {event['error']}"
        return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    async def event_stream_async():
        # 关键：generator 可能跑在不同线程/任务里，重新注入 thread-local
        # 否则 _is_request_authenticated() 会读到默认值 False → 交易工具被错误拒绝
        _set_request_device_id(device_id)
        _set_request_authenticated(authenticated)
        # ── 命中缓存：直接重放，跳过 LLM ──
        if cached_turn:
            try:
                for event in _replay_events_from_turn(cached_turn):
                    yield _process_event(event)
                # 缓存命中也要把这一 turn 追加到会话历史
                # （重新生成 timestamp，避免显示陈旧时间）
                replay_turn = dict(cached_turn)
                replay_turn["timestamp"] = datetime.now().isoformat(timespec="seconds")
                replay_turn["cached"] = True
                session.display.append(replay_turn)
                # 同步 LLM 视角的 messages（让后续追问能基于这次回复）
                session.messages.append({"role": "assistant",
                                          "content": cached_turn.get("content", "")})
                save_session(device_id, session)
                yield f"data: {json.dumps({'type': 'done', 'cached': True})}\n\n"
                return
            except Exception:
                # 重放失败就走真实路径
                pass
        try:
            async for event in stream_quant_agent(session.messages):
                yield _process_event(event)
            session.display.append(assistant_turn)
            save_session(device_id, session)
            # 仅缓存"完整成功"的回复
            if cache_key and _is_cacheable_turn(assistant_turn):
                cache.set(cache_key, assistant_turn, ttl=_ANSWER_CACHE_TTL)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            try:
                save_session(device_id, session)
            except Exception:
                pass
            err = {"type": "error", "error": f"服务端异常: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    def event_stream():
        # 关键：sync generator 会被 StreamingResponse 扔到线程池跑（与 chat() 不同线程）
        # 必须在新线程里重新设置 thread-local，否则交易工具会被误判为"未登录"
        _set_request_device_id(device_id)
        _set_request_authenticated(authenticated)
        # ── 命中缓存：直接重放 ──
        if cached_turn:
            try:
                for event in _replay_events_from_turn(cached_turn):
                    payload = json.dumps(event, ensure_ascii=False, default=str)
                    yield f"data: {payload}\n\n"
                replay_turn = dict(cached_turn)
                replay_turn["timestamp"] = datetime.now().isoformat(timespec="seconds")
                replay_turn["cached"] = True
                session.display.append(replay_turn)
                session.messages.append({"role": "assistant",
                                          "content": cached_turn.get("content", "")})
                save_session(device_id, session)
                yield f"data: {json.dumps({'type': 'done', 'cached': True})}\n\n"
                return
            except Exception:
                pass
        try:
            for event in stream_quant_agent(session.messages):
                # ── 同步更新展示历史 ──
                if event["type"] == "thought":
                    assistant_turn["thoughts"].append({
                        "iteration": event["iteration"],
                        "text": event["text"],
                    })

                elif event["type"] == "tool_call":
                    assistant_turn["tool_calls"].append({
                        "name": event["name"],
                        "input": event["input"],
                        "id": event["id"],
                        "result": None,
                        "is_error": False,
                    })

                elif event["type"] == "tool_result":
                    # 回填到最近一次同名调用
                    for tc in reversed(assistant_turn["tool_calls"]):
                        if (tc["name"] == event["name"]
                                and tc["result"] is None):
                            tc["result"] = event["result"]
                            tc["is_error"] = event["is_error"]
                            break

                elif event["type"] == "final":
                    assistant_turn["content"] = event["text"]
                    # 服务端预渲染的 HTML（老浏览器零依赖显示用）
                    if event.get("text_html"):
                        assistant_turn["content_html"] = event["text_html"]
                    assistant_turn["iterations"] = event["iterations"]

                elif event["type"] == "suggestions":
                    assistant_turn["suggestions"] = event["items"]

                elif event["type"] == "error":
                    assistant_turn["content"] = f"❌ {event['error']}"

                # ── SSE 推送 ──
                payload = json.dumps(event, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"

            # 流结束，存档 + 持久化
            session.display.append(assistant_turn)
            save_session(device_id, session)
            if cache_key and _is_cacheable_turn(assistant_turn):
                cache.set(cache_key, assistant_turn, ttl=_ANSWER_CACHE_TTL)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            # 异常也尝试保存当前状态
            try:
                save_session(device_id, session)
            except Exception:
                pass
            err = {"type": "error", "error": f"服务端异常: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    # 按 stream_quant_agent 是 async 还是 sync 选用对应包装
    return StreamingResponse(
        event_stream_async() if _IS_ASYNC_AGENT else event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ════════════════════════════════════════════════════════════
# RAG 文档管理：上传 PDF + 知识库统计
# ════════════════════════════════════════════════════════════

DOCS_DIR = BASE_DIR / "docs"
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20MB


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 并立即索引到向量库"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "只支持 PDF 文件")

    contents = await file.read()
    if len(contents) > MAX_PDF_SIZE:
        raise HTTPException(413, f"文件超过 {MAX_PDF_SIZE // 1024 // 1024}MB 限制")
    if len(contents) < 100:
        raise HTTPException(400, "文件内容为空或损坏")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCS_DIR / file.filename
    target.write_bytes(contents)

    # 调用 RAG indexer 入库
    try:
        from rag.indexer import index_pdf
        chunks_added = index_pdf(target)
    except Exception as e:
        # 索引失败也保留文件，但告知用户
        return {
            "success": False,
            "filename": file.filename,
            "size_kb": round(len(contents) / 1024, 1),
            "error": f"索引失败: {e}",
            "hint": "文件已保存到 docs/，可手动运行 `python -m rag.indexer`",
        }

    return {
        "success": True,
        "filename": file.filename,
        "size_kb": round(len(contents) / 1024, 1),
        "chunks_added": chunks_added,
        "message": f"已索引 {chunks_added} 个段落",
    }


@app.get("/api/kb/stats")
def kb_stats():
    """返回知识库统计：文档列表 + 总 chunk 数"""
    try:
        from rag.retriever import get_collection_stats
        return get_collection_stats()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/kb/doc/{filename}")
def delete_kb_doc(filename: str):
    """删除知识库中某个文档（同时删 PDF 文件和向量）"""
    try:
        from rag.indexer import get_collection
        coll = get_collection()
        # Chroma 不能按 metadata 直接 delete，先查 ids
        doc_stem = filename.rsplit('.', 1)[0]
        data = coll.get(where={"doc_name": {"$eq": doc_stem}})
        if data['ids']:
            coll.delete(ids=data['ids'])
        pdf_path = DOCS_DIR / filename
        if pdf_path.exists():
            pdf_path.unlink()
        return {"success": True, "deleted_chunks": len(data['ids']),
                "filename": filename}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════
# 交易 API（Alpaca paper trading）
#   GET    /api/broker/status              连通性 + 配置快照
#   GET    /api/orders/intents/{intent_id} 查意图详情（前端弹窗用）
#   POST   /api/orders/confirm             用户确认 → 风控 → 下单
#   POST   /api/orders/cancel/{order_id}   撤单
#   GET    /api/orders                     订单列表
#   GET    /api/positions                  持仓
# ════════════════════════════════════════════════════════════

from pydantic import Field as _Field


class ConfirmOrderRequest(BaseModel):
    intent_id: str
    double_confirmed: bool = False     # 大额订单需勾选确认才放行


@app.get("/api/broker/status")
def broker_status(x_device_id: Optional[str] = Header(None),
                   user: User = Depends(require_user)):
    """检查 broker 是否就绪：凭证 + 连通性 + 风控当前配置"""
    _setup_broker_context(user, x_device_id)
    try:
        from brokers.registry import get_current_broker
        from brokers.risk_gate import current_config
        broker = get_current_broker()
        configured = broker.is_configured()
        account = None
        ping_ok = False
        err = None
        if configured:
            try:
                acc = broker.get_account()
                account = acc.to_dict()
                ping_ok = True
            except Exception as e:
                err = str(e)
        return {
            "broker": broker.name,
            "configured": configured,
            "ping_ok": ping_ok,
            "account": account,
            "error": err,
            "risk_config": current_config(),
        }
    except Exception as e:
        return {"configured": False, "error": str(e)}


@app.get("/api/orders/intents/{intent_id}")
def get_order_intent(intent_id: str,
                      x_device_id: Optional[str] = Header(None),
                      user: User = Depends(require_user)):
    """前端弹窗加载意图详情 + 当前账户快照 + 预检风控"""
    _setup_broker_context(user, x_device_id)
    device_id = _scope_id(user, x_device_id)
    try:
        from brokers.intent_store import get_intent
        from brokers.registry import get_current_broker
        from brokers.risk_gate import check_order
    except Exception as e:
        raise HTTPException(500, f"broker 模块未就绪: {e}")

    intent = get_intent(device_id, intent_id)
    if not intent:
        raise HTTPException(404, "意图不存在或已过期（5 分钟超时）")

    broker = get_current_broker()
    if not broker.is_configured():
        raise HTTPException(400, "Alpaca 凭证未配置")

    try:
        acc = broker.get_account()
        check = check_order(intent, acc, device_id, double_confirmed=False)
    except Exception as e:
        raise HTTPException(502, f"账户查询失败: {e}")

    return {
        "intent": intent.to_dict(),
        "account": acc.to_dict(),
        "risk_check": check.to_dict(),
    }


@app.post("/api/orders/confirm")
def confirm_order(body: ConfirmOrderRequest,
                   x_device_id: Optional[str] = Header(None),
                   user: User = Depends(require_user)):
    """
    用户在弹窗点"确认下单"后调用。
    流程：pop 意图 → 风控 → 真实下单 → 返回订单状态
    """
    _setup_broker_context(user, x_device_id)
    device_id = _scope_id(user, x_device_id)
    try:
        from brokers.intent_store import pop_intent, save_intent
        from brokers import BrokerError
        from brokers.registry import get_current_broker
        from brokers.risk_gate import check_order, record_order_placed
    except Exception as e:
        raise HTTPException(500, f"broker 模块未就绪: {e}")

    intent = pop_intent(device_id, body.intent_id)
    if not intent:
        raise HTTPException(404, "意图不存在或已过期，请让 Agent 重新生成")

    broker = get_current_broker()
    if not broker.is_configured():
        raise HTTPException(400, "Alpaca 凭证未配置")

    # 取当前账户
    try:
        acc = broker.get_account()
    except Exception as e:
        # 回写意图以便用户重试（不计为消耗）
        save_intent(device_id, intent)
        raise HTTPException(502, f"账户查询失败: {e}")

    # 风控
    check = check_order(intent, acc, device_id,
                         double_confirmed=body.double_confirmed)
    if not check.passed:
        # 风控阻断，意图丢弃（防止用户绕过风控反复试）
        return {
            "success": False,
            "blocked_by_risk_gate": True,
            "risk_check": check.to_dict(),
            "message": "; ".join(check.reasons),
        }

    # 真实下单
    try:
        result = broker.place_order(intent)
        record_order_placed(device_id)
    except BrokerError as e:
        return {"success": False, "error": str(e),
                "risk_check": check.to_dict()}

    return {
        "success": True,
        "order": result.to_dict(),
        "risk_check": check.to_dict(),
        "warnings": check.warnings,
    }


@app.post("/api/orders/cancel/{broker_order_id}")
def api_cancel_order(broker_order_id: str,
                      x_device_id: Optional[str] = Header(None),
                      user: User = Depends(require_user)):
    _setup_broker_context(user, x_device_id)
    device_id = _scope_id(user, x_device_id)
    try:
        from brokers import BrokerError
        from brokers.registry import get_current_broker
        from brokers.risk_gate import record_order_canceled
        broker = get_current_broker()
        if not broker.is_configured():
            raise HTTPException(400, "Alpaca 凭证未配置")
        broker.cancel_order(broker_order_id)
        record_order_canceled(device_id)
        return {"success": True, "broker_order_id": broker_order_id}
    except BrokerError as e:
        return {"success": False, "error": str(e)}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/orders")
def list_orders(status: str = "open", limit: int = 50,
                 x_device_id: Optional[str] = Header(None),
                 user: User = Depends(require_user)):
    _setup_broker_context(user, x_device_id)
    try:
        from brokers import BrokerError
        from brokers.registry import get_current_broker
        broker = get_current_broker()
        if not broker.is_configured():
            return {"success": False, "error": "Alpaca 凭证未配置"}
        orders = broker.list_orders(status=status, limit=limit)
        return {"success": True, "count": len(orders),
                 "orders": [o.to_dict() for o in orders]}
    except BrokerError as e:
        return {"success": False, "error": str(e)}


@app.get("/api/positions")
def list_positions_api(x_device_id: Optional[str] = Header(None),
                       user: User = Depends(require_user)):
    _setup_broker_context(user, x_device_id)
    try:
        from brokers import BrokerError
        from brokers.registry import get_current_broker
        broker = get_current_broker()
        if not broker.is_configured():
            return {"success": False, "error": "Alpaca 凭证未配置"}
        positions = broker.list_positions()
        return {"success": True, "count": len(positions),
                 "positions": [p.to_dict() for p in positions]}
    except BrokerError as e:
        return {"success": False, "error": str(e)}


@app.post("/api/broker/reset_account")
def reset_account(x_device_id: Optional[str] = Header(None),
                   user: User = Depends(require_user)):
    """
    重置当前用户的虚拟账户：清空所有持仓/订单，现金归零再回到 $100,000。
    仅 mock 模式可用。
    """
    _setup_broker_context(user, x_device_id)
    try:
        from brokers.registry import get_current_broker
        broker = get_current_broker()
        if broker.name != "mock-paper":
            return {"success": False,
                    "error": f"当前 broker={broker.name} 不支持重置（只有 mock 模式可重置）"}
        # MockAdapter 上有 reset_account 方法
        if hasattr(broker, "reset_account"):
            broker.reset_account()
            return {"success": True, "message": "虚拟账户已重置"}
        return {"success": False, "error": "当前 broker 不支持重置"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════
# 长记忆 API（Phase 1：结构化档案）
#   GET    /api/memory/profile          读当前用户档案
#   PUT    /api/memory/profile          全量更新（覆盖）
#   PATCH  /api/memory/profile          增量更新（merge）
#   DELETE /api/memory/profile          清空档案
# ════════════════════════════════════════════════════════════


class ProfileUpdateRequest(BaseModel):
    updates: dict


@app.get("/api/memory/profile")
def api_get_memory_profile(user: User = Depends(require_user)):
    from memory import get_profile, profile_summary_text
    p = get_profile(user.user_id)
    return {
        "profile": p.to_dict(),
        "summary": profile_summary_text(p),
    }


@app.patch("/api/memory/profile")
def api_patch_memory_profile(body: ProfileUpdateRequest,
                              user: User = Depends(require_user)):
    """增量更新 —— 只覆盖 updates 里出现的字段"""
    from memory import update_profile_fields, profile_summary_text
    try:
        p = update_profile_fields(user.user_id, body.updates or {})
        return {
            "success": True,
            "profile": p.to_dict(),
            "summary": profile_summary_text(p),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/memory/profile")
def api_delete_memory_profile(user: User = Depends(require_user)):
    """清空档案"""
    from memory import clear_profile
    clear_profile(user.user_id)
    return {"success": True}


# ── Layer 2: 情景记忆 ──

@app.get("/api/memory/episodic")
def api_list_episodic(limit: int = 50, user: User = Depends(require_user)):
    """列出当前用户的所有情景记忆"""
    from memory import list_memories
    items = list_memories(user.user_id, limit=int(limit))
    return {"success": True, "count": len(items), "memories": items}


@app.delete("/api/memory/episodic/{memory_id}")
def api_delete_episodic(memory_id: str, user: User = Depends(require_user)):
    """删除一条情景记忆"""
    from memory import delete_memory
    ok = delete_memory(user.user_id, memory_id)
    if not ok:
        return {"success": False, "error": "记忆不存在或不属于当前用户"}
    return {"success": True}


@app.delete("/api/memory/episodic")
def api_clear_episodic(user: User = Depends(require_user)):
    """清空当前用户全部情景记忆"""
    from memory import clear_user_memories
    n = clear_user_memories(user.user_id)
    return {"success": True, "deleted_count": n}


@app.get("/api/memory/stats")
def api_memory_stats(user: User = Depends(require_user)):
    """记忆使用情况统计（UI 展示）"""
    from memory import memory_stats
    return memory_stats(user.user_id)


@app.post("/api/memory/prune")
def api_memory_prune(days: int = 180,
                     user: User = Depends(require_user)):
    """手动触发：清理 N 天未访问的旧记忆（默认 180 天）"""
    from memory import prune_stale_memories
    return prune_stale_memories(user_id=user.user_id, days=int(days))


# ════════════════════════════════════════════════════════════
# OAuth 账户路由
#   /auth/{provider}/login      302 跳转去 OAuth consent
#   /auth/{provider}/callback   处理回调，签发 cookie session
#   /auth/logout                清除 cookie
#   /api/me                     当前登录信息
#   /api/auth/providers         前端拿可用登录方式
# ════════════════════════════════════════════════════════════

ALLOWED_EMAILS_RAW = _os.getenv("AUTH_EMAIL_WHITELIST", "").strip()
ALLOWED_EMAILS = {e.strip().lower() for e in ALLOWED_EMAILS_RAW.split(",")
                   if e.strip()}


def _email_allowed(email: str) -> bool:
    """空白名单 = 开放注册；否则按集合校验"""
    if not ALLOWED_EMAILS:
        return True
    return (email or "").lower() in ALLOWED_EMAILS


@app.get("/api/auth/providers")
def api_auth_providers():
    """前端用：知道哪些登录按钮需要点亮"""
    providers = list(configured_providers())
    if is_email_login_enabled():
        providers.append("email")
    return {
        "providers": providers,
        "whitelist_enabled": bool(ALLOWED_EMAILS),
    }


@app.get("/api/me")
def api_me(user: Optional[User] = Depends(current_user)):
    if user is None:
        raise HTTPException(401, "未登录")
    return user.to_public_dict()


class EmailCodeRequest(BaseModel):
    email: str


class EmailVerifyRequest(BaseModel):
    email: str
    code: str


@app.post("/auth/email/send_code")
def auth_email_send_code(body: EmailCodeRequest):
    """发送邮箱验证码"""
    ok, msg = email_request_code(body.email)
    if not ok:
        return {"success": False, "error": msg}
    return {"success": True, "message": msg}


@app.post("/auth/email/verify")
def auth_email_verify(body: EmailVerifyRequest):
    """校验验证码 → 通过则签 cookie"""
    if not _email_allowed(body.email):
        return {"success": False, "error": "该邮箱未被授权登录"}
    ok, msg = email_verify_code(body.email, body.code)
    if not ok:
        return {"success": False, "error": msg}

    # 通过：建/更新用户
    email = body.email.strip().lower()
    user = upsert_user(
        provider="email",
        provider_sub=email,
        email=email,
        name=email.split("@")[0],
        avatar_url="",
    )
    web_token = create_web_session(user.user_id)

    from fastapi.responses import JSONResponse
    r = JSONResponse({
        "success": True,
        "user": user.to_public_dict(),
    })
    r.set_cookie(
        key=COOKIE_NAME,
        value=web_token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_os.getenv("OAUTH_COOKIE_SECURE", "").lower() in ("1", "true"),
        path="/",
    )
    return r


@app.get("/auth/{provider}/login")
async def auth_login(provider: str, request: Request):
    if provider not in configured_providers():
        raise HTTPException(404, f"OAuth provider '{provider}' 未配置")
    client = oauth_client.create_client(provider)
    redirect_uri = f"{redirect_base()}/auth/{provider}/callback"
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/auth/{provider}/callback")
async def auth_callback(provider: str, request: Request):
    if provider not in configured_providers():
        raise HTTPException(404, f"OAuth provider '{provider}' 未配置")
    client = oauth_client.create_client(provider)

    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        return RedirectResponse(
            url=f"/?auth_error={_url_safe(str(e))}",
            status_code=302,
        )

    # 从 token 里解析 userinfo
    try:
        if provider == "google":
            info = token.get("userinfo") or {}
            if not info:
                # 兜底：调 userinfo endpoint
                resp = await client.get("userinfo", token=token)
                info = resp.json()
            sub = str(info.get("sub", ""))
            email = info.get("email", "")
            name = info.get("name", "") or email.split("@")[0]
            avatar = info.get("picture", "")

        elif provider == "github":
            # GitHub: 走 /user
            resp = await client.get("user", token=token)
            info = resp.json()
            sub = str(info.get("id", ""))
            name = info.get("name") or info.get("login", "")
            avatar = info.get("avatar_url", "")
            email = info.get("email", "")
            # 邮箱可能是 private，需要再调 /user/emails
            if not email:
                try:
                    er = await client.get("user/emails", token=token)
                    emails = er.json() or []
                    primary = next((e for e in emails
                                     if e.get("primary") and e.get("verified")), None)
                    email = (primary or (emails[0] if emails else {})).get("email", "")
                except Exception:
                    pass
        else:
            raise HTTPException(400, "unknown provider")

        if not sub:
            raise ValueError("OAuth 返回缺少 sub/id 字段")

    except Exception as e:
        return RedirectResponse(
            url=f"/?auth_error={_url_safe('解析用户信息失败: ' + str(e))}",
            status_code=302,
        )

    # 邮箱白名单（空白名单 = 不限制）
    if not _email_allowed(email):
        return RedirectResponse(
            url=f"/?auth_error={_url_safe('该邮箱未被授权登录')}",
            status_code=302,
        )

    user = upsert_user(provider=provider, provider_sub=sub,
                        email=email, name=name, avatar_url=avatar)
    web_token = create_web_session(user.user_id)

    resp = RedirectResponse(url="/?login=ok", status_code=302)
    resp.set_cookie(
        key=COOKIE_NAME,
        value=web_token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_os.getenv("OAUTH_COOKIE_SECURE", "").lower() in ("1", "true"),
        path="/",
    )
    return resp


@app.post("/auth/logout")
def auth_logout(request: Request):
    """登出：清 Redis token + 清 cookie"""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        revoke_session(token)
    resp = {"ok": True}
    # 用 JSONResponse 让我们能 set cookie
    from fastapi.responses import JSONResponse
    r = JSONResponse(resp)
    r.delete_cookie(COOKIE_NAME, path="/")
    return r


@app.get("/auth/logout")
def auth_logout_get(request: Request):
    """GET 登出：浏览器直接访问 /auth/logout 也能登出 + 跳回首页"""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        revoke_session(token)
    resp = RedirectResponse(url="/?logout=ok", status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


def _url_safe(s: str) -> str:
    import urllib.parse as _up
    return _up.quote(s[:200], safe="")


# ════════════════════════════════════════════════════════════
# 静态文件 + 首页
# ════════════════════════════════════════════════════════════

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"error": "static/index.html 不存在，请先创建前端文件"}
    return FileResponse(index_file)


@app.get("/brokers")
def brokers_page():
    """券商绑定页面 (X5)"""
    f = STATIC_DIR / "brokers.html"
    if not f.exists():
        raise HTTPException(404, "static/brokers.html missing")
    return FileResponse(f)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sessions": len(cache.keys("quant:session:*")),
        "time": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/me")
def api_me(user: Optional[User] = Depends(current_user)):
    """
    当前登录用户的简略信息。
    - 登录:返回 user 对象 {user_id, email, name, avatar_url, provider}
    - 未登录:返回 null (200)
    契约和 index.html 现有 bootstrapAuth() 期望保持一致。
    """
    if user is None:
        return None
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "provider": user.provider,
    }


@app.get("/metrics/brokers")
def metrics_brokers(_admin: User = Depends(require_admin)):
    """
    Admin-only metrics for the broker subsystem (ADR-0001 §8).
    Gated by ADMIN_EMAILS env. Returns counters + gauges as plain JSON.
    """
    from brokers.metrics import snapshot
    return snapshot()


# ════════════════════════════════════════════════════════════
# Broker bindings — per-user OAuth-style binding management (X5)
# ════════════════════════════════════════════════════════════

class BrokerBindRequest(BaseModel):
    broker_type: str            # "alpaca" | "tiger" | "mock"
    label: str = "main"
    env: str = "paper"
    credentials: dict           # shape depends on broker_type (see credentials_store.build_credentials)


@app.post("/api/broker/bindings", status_code=201)
def api_create_binding(body: BrokerBindRequest,
                       x_device_id: Optional[str] = Header(None),
                       user: User = Depends(require_user)):
    """
    Bind a brokerage account to the logged-in user.
    Tests the credentials in-memory first (adapter.ping()) — only persists
    if the test passes. So no half-bound state on bad creds.

    NOTE: the stored `user_id` is `_scope_id(user, x_device_id)` (i.e. the
    "u:<user_id>" form), NOT raw user.user_id, so that subsequent broker
    tool calls — which use the same _scope_id via _setup_broker_context —
    can find this binding.
    """
    from brokers.credentials_store import store, build_credentials, CredentialsStoreError
    from brokers.registry import _registry, _build_adapter
    from brokers.base import BrokerError, redact_credentials

    user_scope = _scope_id(user, x_device_id)

    # 1. Build typed creds from JSON
    try:
        creds = build_credentials(body.broker_type, body.credentials)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if body.env not in ("paper", "live"):
        raise HTTPException(400, "env must be 'paper' or 'live'")
    # CLAUDE.md trading-safety: live is forbidden until a future ADR.
    if body.env == "live":
        raise HTTPException(400, "live trading is currently disabled (see ADR-0001)")

    # 2. Cheap pre-flight: validate creds SHAPE only — no network call.
    # The full SDK ping has moved to step 5 below, so we only ever create
    # ONE Tiger adapter per binding (avoids the double-allocation of
    # tigeropen's lru_cached RSA state that triggers a malloc abort on
    # macOS Python 3.9).
    try:
        shape_check = _build_adapter(creds)
    except BrokerError as e:
        raise HTTPException(400, redact_credentials(str(e)))
    if not shape_check.is_configured():
        raise HTTPException(400, "missing required credential fields")
    # Drop the throwaway adapter right away — it never touched the network
    # so it holds no C-level state worth keeping.
    del shape_check

    # 3. Persist (under the same user_scope the broker tool path will use)
    try:
        binding_id = store.bind(
            user_id=user_scope,
            broker_type=body.broker_type,
            label=body.label,
            creds=creds,
            actor="user",
            env=body.env,
        )
    except CredentialsStoreError as e:
        # UNIQUE constraint / validation failures map to 409
        raise HTTPException(409, str(e))

    # 4. Drop any stale cached adapter for this key so step 5 builds fresh.
    _registry.invalidate(user_scope, body.broker_type, body.label)
    _registry.invalidate(user_scope, body.broker_type, None)

    # 5. Test the binding via the PRODUCTION resolve path. The adapter built
    # here is cached in the registry and reused by every subsequent request
    # — exactly one adapter per binding for its lifetime, so tigeropen's
    # internal crypto cache never sees a double-construct/double-destruct.
    try:
        adapter = _registry.get(user_scope, body.broker_type, body.label)
        if not adapter.ping():
            raise HTTPException(
                422, f"{body.broker_type} ping returned False (credentials rejected)"
            )
    except HTTPException:
        # Rollback the persisted binding so the user can fix + retry without
        # hitting a UNIQUE-label collision.
        store.unbind(binding_id, user_scope, actor="user")
        _registry.invalidate(user_scope, body.broker_type, body.label)
        _registry.invalidate(user_scope, body.broker_type, None)
        raise
    except Exception as e:
        store.unbind(binding_id, user_scope, actor="user")
        _registry.invalidate(user_scope, body.broker_type, body.label)
        _registry.invalidate(user_scope, body.broker_type, None)
        raise HTTPException(422, f"binding test failed: {redact_credentials(str(e))}")

    return {
        "id": binding_id,
        "broker_type": body.broker_type,
        "label": body.label,
        "env": body.env,
    }


@app.get("/api/broker/bindings")
def api_list_bindings(x_device_id: Optional[str] = Header(None),
                      user: User = Depends(require_user)):
    """List the current user's broker bindings. No secrets in the response."""
    from brokers.credentials_store import store
    user_scope = _scope_id(user, x_device_id)
    rows = store.list_user_bindings(user_scope)
    return [
        {
            "id": r.id,
            "broker_type": r.broker_type,
            "label": r.label,
            "env": r.env,
            "created_at": r.created_at,
            "last_used_at": r.last_used_at,
        }
        for r in rows
    ]


@app.delete("/api/broker/bindings/{binding_id}")
def api_delete_binding(binding_id: int,
                       x_device_id: Optional[str] = Header(None),
                       user: User = Depends(require_user)):
    """Delete a binding owned by the current user. 404 if not found / not theirs."""
    from brokers.credentials_store import store
    user_scope = _scope_id(user, x_device_id)
    ok = store.unbind(binding_id, user_scope, actor="user")
    if not ok:
        raise HTTPException(404, "binding not found")

    # Tear down any running Tiger push connection for this binding
    try:
        from brokers.tiger_push import hub
        if hub.is_running(user_scope):
            hub.stop(user_scope)
    except Exception:
        pass

    return {"ok": True}


# ════════════════════════════════════════════════════════════
# Real-time broker stream (X6 c3) — SSE push from RealtimeStateStore
# ════════════════════════════════════════════════════════════

_PUSH_GRACE_PERIOD_SEC = int(_os.getenv("BROKER_PUSH_GRACE_SEC", "60"))


def _ensure_tiger_push_started(user_scope: str) -> bool:
    """
    Lazily start a TigerPushHub connection for this user if they have a
    tiger binding. Returns True if a push connection is running (now or
    already was).
    """
    from brokers.tiger_push import hub
    from brokers.credentials_store import store

    if hub.is_running(user_scope):
        return True

    bindings = [
        b for b in store.list_user_bindings(user_scope)
        if b.broker_type == "tiger"
    ]
    if not bindings:
        return False

    binding = bindings[0]
    creds = store.load(user_scope, "tiger", binding.label, actor="system")
    if creds is None:
        return False
    try:
        hub.start(user_scope, creds)
    except Exception:
        return False
    return True


def _populate_initial_state(user_scope: str) -> None:
    """
    Best-effort: populate RealtimeStateStore from the current REST snapshot
    so the UI has data before the first push tick arrives.

    Reads via TigerAdapter (cached for 30s, so cheap on repeat connect).
    Errors are silenced — push will fill in within seconds anyway.
    """
    from brokers.registry import _registry
    from brokers.realtime_state import state

    try:
        adapter = _registry.get(user_scope, broker_type="tiger")
    except Exception:
        return

    try:
        acct = adapter.get_account()
        state.update_account(user_scope, acct.to_dict())
    except Exception:
        pass
    try:
        for p in adapter.list_positions():
            state.update_position(user_scope, p.to_dict())
    except Exception:
        pass
    try:
        for o in adapter.list_orders(limit=50):
            state.update_order(user_scope, o.to_dict())
    except Exception:
        pass


def _schedule_push_stop_if_idle(user_scope: str) -> None:
    """After the last SSE client disconnects, wait grace period then stop."""
    import threading
    from brokers.realtime_state import state
    from brokers.tiger_push import hub

    def check():
        if state.subscriber_count(user_scope) == 0:
            try:
                hub.stop(user_scope)
            except Exception:
                pass

    threading.Timer(_PUSH_GRACE_PERIOD_SEC, check).start()


@app.get("/api/broker/stream")
async def broker_stream(
    request: Request,
    x_device_id: Optional[str] = Header(None),
    user: User = Depends(require_user),
):
    """
    Server-Sent Events stream of real-time broker updates for the
    current user. Frontend uses EventSource('/api/broker/stream').

    Events emitted:
      event: open       (one-shot, on connection)
      event: account    {cash, buying_power, equity, ...}
      event: position   {symbol, qty, market_value, unrealized_pl, ...}
      event: order      {broker_order_id, symbol, status, filled_qty, ...}
      event: quote      {symbol, price, change, change_pct, ...}
      event: error      {message}

    Plus ":ping" comment lines every ~15s to detect dead clients.
    """
    import asyncio
    import json as _json
    from queue import Empty
    from brokers.realtime_state import state

    user_scope = _scope_id(user, x_device_id)
    _setup_broker_context(user, x_device_id)

    # 1. Start push connection if user has a tiger binding (lazy)
    has_push = _ensure_tiger_push_started(user_scope)

    # 2. Populate initial state from REST so UI has data before first push
    if has_push:
        await asyncio.to_thread(_populate_initial_state, user_scope)

    # 3. Subscribe to state events
    queue = state.subscribe(user_scope)

    async def event_generator():
        try:
            # Initial 'open' so the client knows the stream is alive
            yield (
                "event: open\n"
                f"data: {_json.dumps({'has_push': has_push, 'user_scope': user_scope})}\n\n"
            )

            ping_interval = 15.0
            last_ping = asyncio.get_event_loop().time()

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_type, payload = await asyncio.to_thread(
                        queue.get, True, 1.0,
                    )
                    yield (
                        f"event: {event_type}\n"
                        f"data: {_json.dumps(payload, default=str)}\n\n"
                    )
                except Empty:
                    now = asyncio.get_event_loop().time()
                    if now - last_ping >= ping_interval:
                        yield ": ping\n\n"
                        last_ping = now
        finally:
            state.unsubscribe(user_scope, queue)
            _schedule_push_stop_if_idle(user_scope)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ════════════════════════════════════════════════════════════
# 启动入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    _host = _os.getenv("HOST", "0.0.0.0")
    _port = int(_os.getenv("PORT", "5000"))
    print("🚀 QuantAgent Web Server")
    print(f"   访问: http://localhost:{_port}")
    print(f"   API:  http://localhost:{_port}/docs")
    print(f"   绑券商: http://localhost:{_port}/brokers")
    print(f"   (overide with HOST=... PORT=... env)")
    uvicorn.run(app, host=_host, port=_port, log_level="info")
