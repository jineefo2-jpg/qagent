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

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quant_agent import stream_quant_agent

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="QuantAgent Web", version="1.0")


# ════════════════════════════════════════════════════════════
# 会话存储（内存版，重启即失）
#   生产环境可替换为 Redis / SQLite / Postgres
# ════════════════════════════════════════════════════════════

class Session:
    """单个会话：完整 Claude 消息历史 + 用于 UI 展示的简化历史"""

    def __init__(self, sid: str, title: str = "新对话"):
        self.id = sid
        self.title = title
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.messages = []   # Claude API 历史（含 tool_use / tool_result 原始对象）
        self.display = []    # UI 展示历史（纯字典，可序列化）

    def to_summary(self):
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "message_count": len(self.display),
        }

    def to_detail(self):
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "display": self.display,
        }


SESSIONS: dict[str, Session] = {}


def get_session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, f"Session {sid} not found")
    return s


# ════════════════════════════════════════════════════════════
# API 路由
# ════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    attachment: Optional[dict] = None  # {"name", "size_kb", "chunks"}


@app.get("/api/sessions")
def list_sessions():
    """列出所有会话（最新在前）"""
    items = [s.to_summary() for s in SESSIONS.values()]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@app.post("/api/sessions")
def create_session():
    """新建会话"""
    sid = uuid.uuid4().hex[:10]
    session = Session(sid)
    SESSIONS[sid] = session
    return session.to_detail()


@app.get("/api/sessions/{sid}")
def session_detail(sid: str):
    """获取单个会话详情（含历史消息）"""
    return get_session(sid).to_detail()


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    """删除会话"""
    SESSIONS.pop(sid, None)
    return {"ok": True}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, body: ChatRequest):
    """
    发送消息，SSE 流式返回 Agent 执行过程。

    前端按行解析 `data: {...}\\n\\n`，事件类型见 stream_quant_agent。
    完成后服务端会自动把这一轮追加到 session.display。
    """
    session = get_session(sid)

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

    # 2. 准备本轮 assistant 的展示记录（边流边填充）
    assistant_turn = {
        "role": "assistant",
        "content": "",
        "thoughts": [],
        "tool_calls": [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iterations": 0,
    }

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
                    assistant_turn["iterations"] = event["iterations"]

                elif event["type"] == "error":
                    assistant_turn["content"] = f"❌ {event['error']}"

                # ── SSE 推送 ──
                payload = json.dumps(event, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"

            # 流结束，存档
            session.display.append(assistant_turn)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            err = {"type": "error", "error": f"服务端异常: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
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
        "sessions": len(SESSIONS),
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
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
