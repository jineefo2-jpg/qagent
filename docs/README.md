# 文档知识库

把要让 Agent 引用的 PDF 放到这个目录。

## 支持的文件类型
- `.pdf`（研报、财报、行业报告、政策文件）

## 文件命名建议
文件名会被 Agent 用作引用标识，建议规范命名：
- ✅ `中信证券_茅台2026Q1点评.pdf`
- ✅ `贵州茅台_2024年报.pdf`
- ❌ `download (3).pdf`

## 索引命令

```bash
# 全量重建（首次或大改动后用）
python -m rag.indexer --reset

# 增量（加新文件时用）
python -m rag.indexer

# 加单个文件
python -m rag.indexer --add /path/to/某研报.pdf
```

## 索引后效果
- 文件存进 `demo/rag_db/`（Chroma 向量库）
- Agent 调用 `search_research_docs(query)` 时自动检索
- 每条结果返回 `citation` 字段，便于 Claude 在报告里引用「《研报名》第 X 页」

## 容量参考
- 一份 30 页的研报 ≈ 150 chunks ≈ 0.5MB 索引
- 100 份研报库 ≈ 50MB
