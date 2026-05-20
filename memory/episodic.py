"""
情景记忆（Layer 2）——自由文本事实/分析/决策，向量化存 Chroma。

复用 rag/indexer.py 的嵌入模型 + Chroma client，单独一个 collection。

记忆类型 (type)：
  - "fact"      用户陈述的事实/约束（不适合塞 profile schema 的）
  - "analysis"  Agent 给出的重要分析结论（用户未来可能回顾）
  - "decision"  交易决策的理由（下单理由等）

用户隔离：metadata.user_id 严格匹配（去掉 u: 前缀的 user_id 字符串）。

隐私护栏：写入前正则筛掉手机号/身份证/银行卡号（直接拒绝，不做 redact）。
"""
from __future__ import annotations

import re
import uuid
import time
import datetime
from typing import List, Dict, Optional


# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════

USER_MEMORY_COLLECTION = "user_memory_bge_small_zh"

VALID_TYPES = {"fact", "analysis", "decision"}

MIN_CONTENT_LEN = 5            # 太短的不值得存
MAX_CONTENT_LEN = 1500         # 太长截断
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.5        # 相似度 (1 - cosine_distance) 低于此不召回

# Phase 4 参数
DEDUP_SIMILARITY = 0.85        # 写入时若已有相似度 > 此值的记忆 → 拒绝写入
STALE_AFTER_DAYS = 180         # 超过此天数未被访问 → 视为过期，清理时删除


# 隐私正则（防止 LLM 把用户敏感信息存入向量库）
_RE_PHONE_CN = re.compile(r'1[3-9]\d{9}')
_RE_ID_CN = re.compile(r'\d{17}[\dXx]')      # 18 位身份证
_RE_CARD = re.compile(r'\d{13,19}')           # 银行卡 13-19 位
_RE_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def _has_sensitive(text: str) -> Optional[str]:
    """返回触发的敏感类型，无则返回 None"""
    if _RE_ID_CN.search(text):
        return "身份证号"
    if _RE_PHONE_CN.search(text):
        return "手机号"
    if _RE_CARD.search(text) and not _RE_ID_CN.search(text):
        # 银行卡跟身份证段位重叠，先识身份证再判卡
        return "银行卡号"
    return None


# ════════════════════════════════════════════════════════════
# Chroma collection（复用 rag/indexer 的客户端）
# ════════════════════════════════════════════════════════════

_user_mem_collection = None


def _get_collection():
    """惰性获取 user_memory collection；复用 rag client"""
    global _user_mem_collection
    if _user_mem_collection is not None:
        return _user_mem_collection
    from rag.indexer import _build_chroma_client
    # 单独维护一个全局 client 引用，独立于 rag 那个，避免互相干扰
    import chromadb
    from chromadb.config import Settings
    from rag.config import DB_DIR
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    _user_mem_collection = client.get_or_create_collection(
        name=USER_MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _user_mem_collection


def _reset_collection():
    """异常恢复 hook（client 被关时让下次重建）"""
    global _user_mem_collection
    _user_mem_collection = None


def _embed(texts: List[str]) -> List[List[float]]:
    """复用 rag 的嵌入模型"""
    from rag.indexer import get_embedder
    embedder = get_embedder()
    return embedder.encode(texts, normalize_embeddings=True).tolist()


# ════════════════════════════════════════════════════════════
# 写入
# ════════════════════════════════════════════════════════════

def record_memory(
    user_id: str,
    content: str,
    mem_type: str = "fact",
    session_id: str = "",
) -> dict:
    """
    存一条情景记忆。返回 {success, memory_id} 或 {success: False, error}。
    """
    if not user_id:
        return {"success": False, "error": "user_id 不能为空"}
    if not content or not content.strip():
        return {"success": False, "error": "content 不能为空"}
    content = content.strip()[:MAX_CONTENT_LEN]
    if len(content) < MIN_CONTENT_LEN:
        return {"success": False, "error": f"content 太短（< {MIN_CONTENT_LEN} 字）"}

    if mem_type not in VALID_TYPES:
        return {"success": False,
                "error": f"mem_type 必须是 {sorted(VALID_TYPES)} 之一"}

    sensitive = _has_sensitive(content)
    if sensitive:
        return {"success": False,
                "error_type": "privacy_blocked",
                "error": f"内容包含{sensitive}，已拒绝写入。"
                          "请改成不含敏感信息的描述。"}

    try:
        coll = _get_collection()
        vec = _embed([content])[0]

        # ── Phase 4: 去重检测 ──
        # 写入前先查同用户里是否已有相似度 > DEDUP_SIMILARITY 的记忆
        if coll.count() > 0:
            dup_res = coll.query(
                query_embeddings=[vec],
                n_results=1,
                where={"user_id": str(user_id)},
            )
            dists = (dup_res.get("distances") or [[]])[0]
            if dists:
                top_score = max(0.0, 1.0 - float(dists[0]))
                if top_score >= DEDUP_SIMILARITY:
                    existing_id = (dup_res.get("ids") or [[]])[0][0]
                    existing_doc = (dup_res.get("documents") or [[]])[0][0]
                    return {
                        "success": False,
                        "error_type": "duplicate",
                        "error": f"已有高度相似的记忆（相似度 {top_score:.2f}），跳过写入",
                        "duplicate_of": existing_id,
                        "duplicate_preview": existing_doc[:80],
                    }

        memory_id = "mem_" + uuid.uuid4().hex[:16]
        now = time.time()
        # summary 截前 80 字，UI 列表展示用
        summary = content[:80] + ("..." if len(content) > 80 else "")
        metadata = {
            "user_id": str(user_id),
            "type": mem_type,
            "session_id": str(session_id or ""),
            "created_at": now,
            "last_accessed_at": now,    # Phase 4: 用于过期判断
            "access_count": 0,           # Phase 4: 被召回次数
            "summary": summary,
        }
        coll.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[vec],
            metadatas=[metadata],
        )
        return {"success": True, "memory_id": memory_id,
                "summary": summary, "type": mem_type}
    except Exception as e:
        return {"success": False, "error": f"存储失败: {e}"}


# ════════════════════════════════════════════════════════════
# 检索
# ════════════════════════════════════════════════════════════

def search_memories(
    user_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[Dict]:
    """
    按相似度检索当前用户的情景记忆。仅返回 score >= min_score 的。
    每条返回：{id, content, type, score, created_at, summary}
    """
    if not user_id or not query or not query.strip():
        return []
    top_k = max(1, min(int(top_k), 20))
    try:
        coll = _get_collection()
        # 没有记录就直接返回，避免无意义查询
        if coll.count() == 0:
            return []
        vec = _embed([query])[0]
        res = coll.query(
            query_embeddings=[vec],
            n_results=top_k,
            where={"user_id": str(user_id)},
        )
    except Exception as e:
        msg = str(e).lower()
        if "closed" in msg:
            _reset_collection()
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []
    hit_ids = []
    hit_metas = []
    for i in range(len(docs)):
        # cosine 距离 → 相似度
        dist = float(dists[i]) if dists else 1.0
        score = max(0.0, 1.0 - dist)
        if score < min_score:
            continue
        meta = metas[i] or {}
        out.append({
            "id": ids[i],
            "content": docs[i],
            "type": meta.get("type", "fact"),
            "session_id": meta.get("session_id", ""),
            "created_at": meta.get("created_at", 0),
            "summary": meta.get("summary", docs[i][:80]),
            "score": round(score, 4),
        })
        hit_ids.append(ids[i])
        hit_metas.append(dict(meta))

    # ── Phase 4: 命中后更新 last_accessed_at + access_count ──
    if hit_ids:
        try:
            now = time.time()
            for m in hit_metas:
                m["last_accessed_at"] = now
                m["access_count"] = int(m.get("access_count", 0)) + 1
            coll.update(ids=hit_ids, metadatas=hit_metas)
        except Exception:
            pass   # 更新失败不影响检索本身

    return out


# ════════════════════════════════════════════════════════════
# 列表（UI 用） + 删除
# ════════════════════════════════════════════════════════════

def list_memories(user_id: str, limit: int = 50) -> List[Dict]:
    """列出当前用户的所有记忆（按时间倒序）"""
    if not user_id:
        return []
    try:
        coll = _get_collection()
        data = coll.get(where={"user_id": str(user_id)}, limit=limit * 3)
    except Exception:
        return []
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    ids = data.get("ids") or []
    items = []
    for i in range(len(metas)):
        meta = metas[i] or {}
        items.append({
            "id": ids[i],
            "content": docs[i],
            "type": meta.get("type", "fact"),
            "session_id": meta.get("session_id", ""),
            "created_at": meta.get("created_at", 0),
            "summary": meta.get("summary", docs[i][:80]),
        })
    # 按时间倒序，截断
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return items[:limit]


def delete_memory(user_id: str, memory_id: str) -> bool:
    """删除单条记忆（验证 user_id 防止跨用户删）"""
    if not user_id or not memory_id:
        return False
    try:
        coll = _get_collection()
        # 校验该 memory 确实属于该用户
        data = coll.get(ids=[memory_id])
        if not data.get("ids"):
            return False
        meta = (data.get("metadatas") or [{}])[0]
        if meta.get("user_id") != str(user_id):
            return False
        coll.delete(ids=[memory_id])
        return True
    except Exception:
        return False


def prune_stale_memories(user_id: str = None,
                          days: int = STALE_AFTER_DAYS) -> dict:
    """
    清理过期记忆（超过 `days` 天未被访问的）。
    user_id 为 None 时全局清理（管理员维护用）；指定 user_id 时只清该用户。
    返回 {checked, removed, kept}
    """
    try:
        coll = _get_collection()
        where = {"user_id": str(user_id)} if user_id else None
        data = coll.get(where=where) if where else coll.get()
    except Exception:
        return {"checked": 0, "removed": 0, "kept": 0}

    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    cutoff = time.time() - days * 86400

    to_delete = []
    for i, m in enumerate(metas):
        m = m or {}
        # 用 last_accessed_at 兜底到 created_at
        last_ts = float(m.get("last_accessed_at") or m.get("created_at", 0))
        if last_ts and last_ts < cutoff:
            to_delete.append(ids[i])

    if to_delete:
        try:
            coll.delete(ids=to_delete)
        except Exception:
            pass

    return {
        "checked": len(ids),
        "removed": len(to_delete),
        "kept": len(ids) - len(to_delete),
        "cutoff_days": days,
    }


def memory_stats(user_id: str) -> dict:
    """返回该用户记忆统计（UI 展示用）"""
    try:
        coll = _get_collection()
        data = coll.get(where={"user_id": str(user_id)})
    except Exception:
        return {"total": 0}
    metas = data.get("metadatas") or []
    by_type = {"fact": 0, "analysis": 0, "decision": 0}
    total_access = 0
    oldest = None
    newest = None
    for m in metas:
        m = m or {}
        t = m.get("type", "fact")
        if t in by_type:
            by_type[t] += 1
        total_access += int(m.get("access_count", 0))
        ts = float(m.get("created_at", 0))
        if ts:
            oldest = ts if oldest is None else min(oldest, ts)
            newest = ts if newest is None else max(newest, ts)
    return {
        "total": len(metas),
        "by_type": by_type,
        "total_access": total_access,
        "oldest_created_at": oldest,
        "newest_created_at": newest,
    }


def clear_user_memories(user_id: str) -> int:
    """删除指定用户全部记忆，返回删除条数"""
    if not user_id:
        return 0
    try:
        coll = _get_collection()
        data = coll.get(where={"user_id": str(user_id)})
        ids = data.get("ids") or []
        if ids:
            coll.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════
# 给 system prompt 用的格式化
# ════════════════════════════════════════════════════════════

def memories_to_prompt_text(memories: List[Dict]) -> str:
    """把召回的记忆格式化成 system prompt 片段"""
    if not memories:
        return ""
    TYPE_CN = {"fact": "事实", "analysis": "分析", "decision": "决策"}
    lines = []
    for m in memories:
        ts = m.get("created_at", 0)
        if ts:
            dt = datetime.datetime.fromtimestamp(ts).strftime("%m-%d")
        else:
            dt = "?"
        type_label = TYPE_CN.get(m.get("type", "fact"), m.get("type", ""))
        content = m.get("content", "")
        lines.append(f"- [{dt} · {type_label}] {content}")
    return "【相关历史记忆】\n" + "\n".join(lines)
