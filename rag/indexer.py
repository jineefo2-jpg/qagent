"""
PDF → Chroma 向量库的索引构建。

用法：
    python -m rag.indexer                  # 全量重建
    python -m rag.indexer --add 新文档.pdf  # 单文件增量
"""
import os
import sys
import hashlib
import argparse
from pathlib import Path

# HF 镜像支持（大陆部署必备）—— 必须在 import sentence_transformers 之前
if not os.getenv("HF_ENDPOINT"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

from .config import (
    DOCS_DIR, DB_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
    COLLECTION_NAME,
)


# ════════════════════════════════════════════════════════════
# PDF 解析与分块
# ════════════════════════════════════════════════════════════

def extract_pdf_pages(pdf_path: Path):
    """打开 PDF，返回 [(page_num, text), ...] 列表"""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()
        if text:
            pages.append((i + 1, text))  # 页码 1-indexed
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int):
    """简单的字符级分块（中文不需要 token 分词）"""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks(pdf_path: Path):
    """
    把一个 PDF 解析成可入库的 chunks。
    返回 [(chunk_id, content, metadata), ...]
    """
    doc_name = pdf_path.stem
    pages = extract_pdf_pages(pdf_path)
    out = []

    for page_num, page_text in pages:
        for idx, c in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
            c = c.strip()
            if len(c) < MIN_CHUNK_LEN:
                continue
            # 确定性 ID：同文档同位置重建时能去重
            cid = hashlib.md5(
                f"{doc_name}|{page_num}|{idx}|{c[:50]}".encode()
            ).hexdigest()[:16]
            out.append((
                cid,
                c,
                {
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_idx": idx,
                    "source": pdf_path.name,
                }
            ))
    return out


# ════════════════════════════════════════════════════════════
# Chroma + 嵌入模型
# ════════════════════════════════════════════════════════════

_embedder = None
_client = None


def get_embedder():
    """惰性加载嵌入模型（首次约 5-10 秒）"""
    global _embedder
    if _embedder is None:
        print(f"⏳ 加载嵌入模型 {EMBEDDING_MODEL}（首次下载约 100MB）...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ 模型已加载，向量维度 {_embedder.get_sentence_embedding_dimension()}")
    return _embedder


def _build_chroma_client():
    """构造 Chroma client；独立函数便于在异常时重建"""
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(
            anonymized_telemetry=False,  # 禁用 posthog 避免 httpx 残留
            allow_reset=True,
        ),
    )


def get_collection():
    """获取或创建 Chroma collection；client 被关时自动重建"""
    global _client
    if _client is None:
        _client = _build_chroma_client()
    try:
        return _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        # "Cannot send a request, as the client has been closed" 等场景
        msg = str(e).lower()
        if "closed" in msg or "client" in msg:
            # 重建一次再试
            _client = _build_chroma_client()
            return _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        raise


def reset_chroma_client():
    """外部异常恢复 hook：让下次 get_collection 重建实例"""
    global _client
    _client = None


# ════════════════════════════════════════════════════════════
# 索引构建入口
# ════════════════════════════════════════════════════════════

def index_pdf(pdf_path: Path):
    """单个 PDF 入库"""
    if not pdf_path.exists():
        print(f"❌ 文件不存在: {pdf_path}")
        return 0

    print(f"\n📄 处理 {pdf_path.name}")
    chunks = build_chunks(pdf_path)
    if not chunks:
        print(f"   ⚠️  无可索引文本，跳过")
        return 0

    print(f"   → 分块 {len(chunks)} 条，计算嵌入向量...")
    embedder = get_embedder()
    coll = get_collection()

    ids       = [c[0] for c in chunks]
    contents  = [c[1] for c in chunks]
    metadatas = [c[2] for c in chunks]
    embeddings = embedder.encode(contents, show_progress_bar=False,
                                  normalize_embeddings=True).tolist()

    # upsert 支持增量更新
    coll.upsert(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"   ✅ 已入库 {len(chunks)} chunks")
    return len(chunks)


def index_all(reset: bool = False):
    """扫描 docs/ 目录全量索引"""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建文档目录: {DOCS_DIR}")

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  {DOCS_DIR} 下没有 PDF 文件")
        print(f"   把研报/财报 PDF 放进该目录后重新运行")
        return

    if reset:
        print("🗑  清空旧索引...")
        try:
            client = chromadb.PersistentClient(path=str(DB_DIR))
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    print(f"📚 发现 {len(pdfs)} 份 PDF，开始索引...")
    total = 0
    for p in pdfs:
        total += index_pdf(p)

    coll = get_collection()
    print(f"\n{'='*50}")
    print(f"✨ 全部完成。库内 chunk 总数: {coll.count()}")
    print(f"   本次新增/更新: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", help="增量索引指定 PDF 文件路径")
    parser.add_argument("--reset", action="store_true", help="清空后全量重建")
    args = parser.parse_args()

    if args.add:
        index_pdf(Path(args.add))
    else:
        index_all(reset=args.reset)
