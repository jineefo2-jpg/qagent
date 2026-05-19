# ============================================================
#  QuantAgent Web Server
#  ─────────────────────────────────────────────
#  FastAPI 服务 + SSE 流式响应 + 会话管理
#
#  启动：
#    pip install fastapi uvicorn anthropic
#    export ANTHROPIC_API_KEY="sk-ant-..."
#    python server.py
#  访问：http://localhost:8000
# ============================================================

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os as _os
import inspect as _inspect
from cache import cache  # Redis 或内存缓存

# 框架切换：USE_LANGGRAPH=1 启用 LangGraph 版主循环
_USE_LG = _os.getenv("USE_LANGGRAPH", "").strip().lower() in ("1", "true", "yes", "on")

if _USE_LG:
    from quant_agent_lg import stream_quant_agent_lg as stream_quant_agent
    print("🌐 Agent 主循环: LangGraph 版（quant_agent_lg，async token 流式）")
else:
    from quant_agent import stream_quant_agent
    print("⚙️  Agent 主循环: 原生 ReAct 版（quant_agent）")

# 检测 stream_quant_agent 是同步还是异步 generator
_IS_ASYNC_AGENT = _inspect.isasyncgenfunction(stream_quant_agent)

# 当 X-Device-Id 缺失时的兜底（不应该出现，前端总会发）
_DEFAULT_DEVICE = "default"

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="QuantAgent Web", version="1.0")


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
# API 路由
# ════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    attachment: Optional[dict] = None  # {"name", "size_kb", "chunks"}


@app.get("/api/sessions")
def list_sessions(x_device_id: Optional[str] = Header(None)):
    """列出当前设备的所有会话（最新在前）"""
    device_id = x_device_id or _DEFAULT_DEVICE
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
def create_session(x_device_id: Optional[str] = Header(None)):
    """新建会话，绑定到当前设备"""
    device_id = x_device_id or _DEFAULT_DEVICE
    sid = uuid.uuid4().hex[:10]
    session = Session(sid)
    save_session(device_id, session)
    return session.to_detail()


@app.get("/api/sessions/{sid}")
def session_detail(sid: str, x_device_id: Optional[str] = Header(None)):
    device_id = x_device_id or _DEFAULT_DEVICE
    return get_session(device_id, sid).to_detail()


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str, x_device_id: Optional[str] = Header(None)):
    device_id = x_device_id or _DEFAULT_DEVICE
    delete_session_storage(device_id, sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, body: ChatRequest,
         x_device_id: Optional[str] = Header(None)):
    """发送消息，SSE 流式返回 Agent 执行过程"""
    device_id = x_device_id or _DEFAULT_DEVICE
    session = get_session(device_id, sid)

    if not body.message.strip():
        raise HTTPException(400, "message 不能为空")

    # 首条用户消息作为会话标题
    if not session.display:
        session.title = body.message[:30] + ("..." if len(body.message) > 30 else "")

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
        try:
            async for event in stream_quant_agent(session.messages):
                yield _process_event(event)
            session.display.append(assistant_turn)
            save_session(device_id, session)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            try:
                save_session(device_id, session)
            except Exception:
                pass
            err = {"type": "error", "error": f"服务端异常: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    def event_stream():
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sessions": len(cache.keys("quant:session:*")),
        "time": datetime.now().isoformat(timespec="seconds"),
    }


# ════════════════════════════════════════════════════════════
# 启动入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🚀 QuantAgent Web Server")
    print("   访问: http://localhost:5000")
    print("   API:  http://localhost:5000/docs")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
