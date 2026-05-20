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
    COLLECTION_NAME, RERANKER_MODEL,
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


# ════════════════════════════════════════════════════════════
# Recursive Character 分块（中英文混合，金融文档优化）
#
# 核心思想：按"语义重要性"分级使用分隔符，能用段落切就不用句子切。
#   1. 先按高级分隔符（段落/句号）切到全部片段 ≤ chunk_size
#   2. 把小片段合并为 ~chunk_size 的 chunk
#   3. 相邻 chunk 之间保留 overlap（保持上下文连续）
#
# 关键改进点（vs 旧字符级硬切）：
#   - 句子末尾才切，不再"句子中间"截断
#   - markdown 表格行（用 \n 分隔）能被识别
#   - 小数（"15.5%"）不会被英文句号切开（要求 ". " 才算句末）
#   - 段落空行（\n\n）优先级最高，最大程度保留逻辑单元
# ════════════════════════════════════════════════════════════

# 中英金融文档常见分隔符（按优先级降序；空字符串是字符级兜底）
DEFAULT_SEPARATORS = [
    "\n\n",       # 段落
    "\n",         # 换行（表格行也走这个）
    "。", "！", "？",   # 中文句末
    "；",          # 中文分号
    ". ",         # 英文句末（带空格，避免 15.5% 被切）
    "? ", "! ", "; ",   # 英文标点带空格
    "，", ", ",   # 逗号
    " ",          # 空格
    "",           # 字符级兜底
]


def _split_keep_separator(text: str, sep: str) -> list:
    """按 sep 切分，把分隔符保留在前一段的尾部"""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    if len(parts) <= 1:
        return parts
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, separators: list, chunk_size: int) -> list:
    """递归切分到所有片段 ≤ chunk_size"""
    if len(text) <= chunk_size:
        return [text]

    # 找到本轮要用的分隔符（首个文本里出现的，或最终空字符串）
    sep = ""
    next_seps = separators
    for i, s in enumerate(separators):
        if s == "":
            sep = s
            next_seps = []
            break
        if s in text:
            sep = s
            next_seps = separators[i + 1:]
            break

    # 字符级兜底：定长硬切
    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    splits = _split_keep_separator(text, sep)

    # 对仍然超长的片段继续递归切
    result = []
    for s in splits:
        if len(s) <= chunk_size:
            result.append(s)
        elif next_seps:
            result.extend(_recursive_split(s, next_seps, chunk_size))
        else:
            # 没分隔符了 → 字符硬切
            for k in range(0, len(s), chunk_size):
                result.append(s[k:k + chunk_size])
    return result


def _merge_with_overlap(pieces: list, chunk_size: int, overlap: int) -> list:
    """
    贪心合并小片段为 chunk，相邻 chunk 之间字符级 overlap。

    严格按字符长度（不按 piece 边界）控制 overlap，避免单 piece 过大导致
    chunk 实际长度超过 chunk_size。
    """
    if not pieces:
        return []
    overlap = max(0, min(overlap, max(1, chunk_size - 1)))   # 防呆

    chunks = []
    current = ""

    for p in pieces:
        if len(current) + len(p) <= chunk_size:
            current += p
            continue
        # 当前 chunk 满了
        if current:
            chunks.append(current)
            # 从末尾按字符取 overlap 长度作为下一个 chunk 的起点
            current = current[-overlap:] if overlap > 0 else ""
        # 加上新片段；若新片段自己超长（_recursive_split 兜底过了，这里再保险）
        if len(current) + len(p) <= chunk_size:
            current += p
        else:
            # 当前已有 overlap + 新片段超长 → 先提交 overlap+部分，再剩余字符级硬切
            remaining = chunk_size - len(current)
            current += p[:remaining]
            chunks.append(current)
            current = p[remaining:]
            # 若剩余仍超长，字符级硬切
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                tail = current[chunk_size - overlap:] if overlap > 0 else ""
                current = tail + ""   # 继续在 current 里挂下一片
                break

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Recursive Character 切分（保留旧函数名，平滑替换）。
    自动按 "段落 → 句号 → 分号 → 逗号 → 空格 → 字符" 多级回退。
    """
    if not text or not text.strip():
        return []

    # 1. 递归切到所有片段 ≤ chunk_size
    pieces = _recursive_split(text, DEFAULT_SEPARATORS, chunk_size)

    # 2. 贪心合并 + 加 overlap
    chunks = _merge_with_overlap(pieces, chunk_size, overlap)

    # 3. 清洗空白
    return [c.strip() for c in chunks if c.strip()]


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
_reranker = None
_reranker_failed = False   # 加载失败标记：避免每次检索都重试


def get_embedder():
    """惰性加载嵌入模型（首次约 5-10 秒）"""
    global _embedder
    if _embedder is None:
        print(f"⏳ 加载嵌入模型 {EMBEDDING_MODEL}（首次下载约 100MB）...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ 模型已加载，向量维度 {_embedder.get_sentence_embedding_dimension()}")
    return _embedder


def get_reranker():
    """
    惰性加载 cross-encoder 精排模型（首次下载 ~110MB）。
    依次尝试 HF mirror → 官方 → ModelScope；都挂掉时 fallback 到仅嵌入排序。
    """
    global _reranker, _reranker_failed
    if _reranker_failed:
        return None
    if _reranker is not None:
        return _reranker

    from sentence_transformers import CrossEncoder

    # 依次尝试不同的 endpoint
    endpoints = [
        ("HF mirror（国内代理）", "https://hf-mirror.com"),
        ("HuggingFace 官方", "https://huggingface.co"),
    ]

    print(f"⏳ 加载精排模型 {RERANKER_MODEL}...")
    last_err = None
    for label, ep in endpoints:
        try:
            os.environ["HF_ENDPOINT"] = ep
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            print(f"✅ 精排模型已加载（来源: {label}）")
            return _reranker
        except Exception as e:
            print(f"  ⚠️  {label} 失败: {str(e)[:120]}")
            last_err = e
            continue

    print(f"⚠️  所有镜像均失败（将仅用 embedding 排序）: {last_err}")
    _reranker_failed = True
    return None


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
