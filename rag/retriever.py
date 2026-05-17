"""
RAG 检索接口 —— 被 quant_agent.py 作为工具调用。
"""
from typing import Optional

from .config import DEFAULT_TOP_K, COLLECTION_NAME
from .indexer import get_embedder, get_collection


def search_research_docs(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: Optional[str] = None,
) -> dict:
    """
    在已索引的研报/财报库中检索相关段落。

    Args:
        query: 自然语言查询（如"茅台一季度营收增速"、"中信对新能源车的判断"）
        top_k: 返回相关段落数，默认 3
        doc_filter: 可选，限定在指定文档内搜索（按文件名前缀匹配）

    Returns:
        {"success": True, "results": [{doc_name, page, content, score}, ...]}
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空"}

    try:
        coll = get_collection()
    except Exception as e:
        return {"success": False,
                "error": f"无法连接知识库: {e}",
                "hint": "先运行 `python -m rag.indexer` 建立索引"}

    if coll.count() == 0:
        return {
            "success": False,
            "error_type": "empty_kb",
            "error": "知识库为空",
            "hint": "把 PDF 放到 demo/docs/ 目录，运行 "
                    "`python -m rag.indexer` 建索引",
        }

    # 嵌入查询
    embedder = get_embedder()
    query_vec = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    # 检索参数
    where = {"doc_name": {"$eq": doc_filter}} if doc_filter else None
    n = max(1, min(int(top_k), 10))

    try:
        res = coll.query(
            query_embeddings=[query_vec],
            n_results=n,
            where=where,
        )
    except Exception as e:
        return {"success": False, "error": f"检索失败: {e}"}

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

    results = []
    for doc, meta, dist in zip(docs, metas, distances):
        results.append({
            "doc_name":  meta.get("doc_name"),
            "page":      meta.get("page"),
            "source":    meta.get("source"),
            "content":   doc.strip(),
            "relevance": round(1 - dist, 4),  # cosine distance → 相似度
            "citation":  f"《{meta.get('doc_name')}》第 {meta.get('page')} 页",
        })

    return {
        "success":     True,
        "query":       query,
        "results":     results,
        "result_count": len(results),
        "kb_total":    coll.count(),
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
