"""RAG 模块共享配置"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DOCS_DIR = BASE_DIR / "docs"      # 用户放 PDF 的位置
DB_DIR   = BASE_DIR / "rag_db"    # Chroma 持久化目录

# 嵌入模型（中文研报场景，bge-small 最佳性价比）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 分块策略
CHUNK_SIZE    = 500    # 每块约 500 字符
CHUNK_OVERLAP = 50     # 相邻块重叠 50 字符
MIN_CHUNK_LEN = 30     # 太短的块（页眉/页码）丢弃

# 检索默认参数
DEFAULT_TOP_K = 3
COLLECTION_NAME = "research_docs"

# Reranker（cross-encoder，精排）
# 流程：bi-encoder 召回 top_k * MULTIPLIER 候选 → cross-encoder 重打分 → 取 top_k
RERANKER_MODEL = "BAAI/bge-reranker-base"   # 110MB，中文友好
RERANK_ENABLED = True                         # 全局开关
RERANK_CANDIDATES_MULTIPLIER = 5              # 召回 top_k * 5 喂给 reranker
RERANK_MIN_SCORE = -10.0                       # reranker logit 阈值，过低不返回（保守）
