"""
RAG 检索接口 —— 被 quant_agent.py 作为工具调用。

两阶段检索：
  1. bi-encoder（bge-small-zh）KNN 召回 top_k * MULTIPLIER 候选
  2. cross-encoder（bge-reranker-base）对候选重打分，取 top_k
"""
from typing import Optional

from .config import (
    DEFAULT_TOP_K, COLLECTION_NAME,
    RERANK_ENABLED, RERANK_CANDIDATES_MULTIPLIER, RERANK_MIN_SCORE,
)
from .indexer import (
    get_embedder, get_collection, reset_chroma_client, get_reranker,
)


def _rerank(query: str, candidates: list) -> list:
    """
    对候选 chunk 列表用 cross-encoder 重打分。
    candidates: [{doc, meta, distance}, ...]
    返回按新分数降序排序的列表（每条新增 rerank_score 字段）。

    若 reranker 未加载/加载失败，直接返回原列表（按 distance 升序）。
    """
    if not candidates:
        return candidates
    if not RERANK_ENABLED:
        return candidates

    reranker = get_reranker()
    if reranker is None:
        return candidates    # fallback: 仅嵌入排序

    pairs = [(query, c["doc"]) for c in candidates]
    try:
        scores = reranker.predict(pairs, show_progress_bar=False).tolist()
    except Exception as e:
        # 推理出错也降级
        print(f"⚠️  rerank 失败，回退到 embedding 排序: {e}")
        return candidates

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    # 按 rerank_score 降序，过滤太低的
    candidates = [c for c in candidates if c["rerank_score"] >= RERANK_MIN_SCORE]
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def _is_client_closed(err: Exception) -> bool:
    msg = str(err).lower()
    return ("client has been closed" in msg
            or "client is closed" in msg
            or "closed" in msg and "client" in msg)


def search_research_docs(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: Optional[str] = None,
) -> dict:
    """
    在已索引的研报/财报库中检索相关段落。
    遇到「httpx client closed」类错误会自动重建 chroma client 重试一次。
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空"}

    def _do_search():
        coll = get_collection()
        if coll.count() == 0:
            return {
                "success": False,
                "error_type": "empty_kb",
                "error": "知识库为空",
                "hint": "把 PDF 放到 demo/docs/ 目录，运行 `python -m rag.indexer`",
            }
        embedder = get_embedder()
        query_vec = embedder.encode([query], normalize_embeddings=True).tolist()[0]
        where = {"doc_name": {"$eq": doc_filter}} if doc_filter else None
        final_k = max(1, min(int(top_k), 10))
        # 启用 rerank 时拉更多候选（用 reranker 精排回 final_k）
        n_candidates = final_k * RERANK_CANDIDATES_MULTIPLIER if RERANK_ENABLED else final_k
        return coll.query(
            query_embeddings=[query_vec],
            n_results=min(n_candidates, max(coll.count(), 1)),
            where=where,
        )

    # 第 1 次尝试
    try:
        res = _do_search()
        if isinstance(res, dict) and res.get("error_type") == "empty_kb":
            return res
        coll = get_collection()
    except Exception as e1:
        if _is_client_closed(e1):
            # 重建 client 重试一次
            try:
                reset_chroma_client()
                res = _do_search()
                if isinstance(res, dict) and res.get("error_type") == "empty_kb":
                    return res
                coll = get_collection()
            except Exception as e2:
                return {"success": False,
                        "error_type": "client_closed_retry_failed",
                        "error": f"检索失败（已重试）: {e2}",
                        "hint": "服务重启或稍后再试"}
        else:
            return {"success": False, "error": f"检索失败: {e1}"}

    docs       = res.get("documents", [[]])[0]
    metas      = res.get("metadatas", [[]])[0]
    distances  = res.get("distances", [[]])[0]

    if not docs:
        return {
            "success": True,
            "query": query,
            "results": [],
            "note": "未检索到相关段落",
        }

    # ── 整理候选 → 喂 reranker ──
    candidates = []
    for doc, meta, dist in zip(docs, metas, distances):
        candidates.append({
            "doc": doc.strip(),
            "meta": meta or {},
            "distance": float(dist) if dist is not None else 1.0,
        })

    reranked = _rerank(query, candidates)
    final_k = max(1, min(int(top_k), 10))
    reranked = reranked[:final_k]

    results = []
    for c in reranked:
        meta = c["meta"]
        relevance = round(1 - c["distance"], 4)
        item = {
            "doc_name":  meta.get("doc_name"),
            "page":      meta.get("page"),
            "source":    meta.get("source"),
            "content":   c["doc"],
            "relevance": relevance,
            "citation":  f"《{meta.get('doc_name')}》第 {meta.get('page')} 页",
        }
        if "rerank_score" in c:
            item["rerank_score"] = round(c["rerank_score"], 4)
        results.append(item)

    return {
        "success":     True,
        "query":       query,
        "results":     results,
        "result_count": len(results),
        "kb_total":    coll.count(),
        "rerank":      RERANK_ENABLED and (get_reranker() is not None),
        "data_source": f"内部文档库（{COLLECTION_NAME}）",
        "disclaimer":  "引用内容来自已入库 PDF，请核对原文准确性",
    }


def get_collection_stats() -> dict:
    """返回库统计：文档列表 + 总 chunk 数（用于调试和 UI 展示）"""
    try:
        coll = get_collection()
        total = coll.count()
        if total == 0:
            return {"success": True, "kb_empty": True,
                    "total_chunks": 0, "documents": []}

        # 抓所有 metadata（小库时可接受）
        all_data = coll.get(include=["metadatas"])
        doc_pages = {}
        for m in all_data["metadatas"]:
            name = m.get("doc_name", "?")
            doc_pages.setdefault(name, set()).add(m.get("page"))

        documents = [
            {"name": name, "pages": len(pages), "chunks": sum(
                1 for m in all_data["metadatas"] if m.get("doc_name") == name
            )}
            for name, pages in sorted(doc_pages.items())
        ]
        return {
            "success": True,
            "total_chunks": total,
            "document_count": len(documents),
            "documents": documents,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
