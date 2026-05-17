"""RAG 模块：研报/财报 PDF 检索"""
from .retriever import search_research_docs, get_collection_stats

__all__ = ["search_research_docs", "get_collection_stats"]
