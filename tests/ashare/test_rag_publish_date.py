"""Task 15：RAG chunk 的 publish_date（U4 / 架构 A6）。只测纯解析 + build_chunks 的 metadata 键；不碰 chroma。

rag/__init__.py 会 eager import retriever→indexer→sentence_transformers（重依赖，且本机 transformers 版本可能不兼容）。
publish_date.py 是纯函数，这里按文件路径加载，不触发包 __init__；build_chunks 测试则 importorskip 整个 indexer。
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location(
    "rag_publish_date_standalone", Path(__file__).resolve().parents[2] / "rag" / "publish_date.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)                       # type: ignore[union-attr]
resolve_publish_date = _mod.resolve_publish_date


def test_filename_iso_date(tmp_path):
    assert resolve_publish_date(tmp_path / "研报_茅台_2026-05-18.pdf") == ("2026-05-18", "filename")


def test_filename_compact_and_underscore(tmp_path):
    assert resolve_publish_date(tmp_path / "20260518_茅台深度.pdf") == ("2026-05-18", "filename")
    assert resolve_publish_date(tmp_path / "茅台_2026_05_18.pdf") == ("2026-05-18", "filename")


def test_filename_invalid_date_is_unknown(tmp_path):
    assert resolve_publish_date(tmp_path / "报告_2026-13-45.pdf") == ("", "unknown")
    assert resolve_publish_date(tmp_path / "报告_600519.pdf") == ("", "unknown")     # 股票代码不是日期


def test_no_date_is_unknown_not_guessed(tmp_path):
    assert resolve_publish_date(tmp_path / "测试研报_茅台.pdf") == ("", "unknown")


def test_override_json_wins_over_filename(tmp_path):
    (tmp_path / "publish_dates.json").write_text(json.dumps({"研报_2026-05-18.pdf": "2026-05-20"}), encoding="utf-8")
    assert resolve_publish_date(tmp_path / "研报_2026-05-18.pdf") == ("2026-05-20", "override")


def test_override_by_stem_and_bad_override_ignored(tmp_path):
    (tmp_path / "publish_dates.json").write_text(json.dumps({"无日期报告": "2025-01-31", "坏的": "not-a-date"}), encoding="utf-8")
    assert resolve_publish_date(tmp_path / "无日期报告.pdf") == ("2025-01-31", "override")
    assert resolve_publish_date(tmp_path / "坏的.pdf") == ("", "unknown")


def test_build_chunks_carries_publish_date(monkeypatch, tmp_path):
    """indexer.build_chunks 的每个 chunk metadata 都带 publish_date / publish_date_source。"""
    indexer = pytest.importorskip("rag.indexer")          # 需要 fitz / chromadb / sentence_transformers
    monkeypatch.setattr(indexer, "extract_pdf_pages", lambda p: [(1, "这是一段足够长的测试文本。" * 30)])
    monkeypatch.setattr(indexer, "MIN_CHUNK_LEN", 1)
    chunks = indexer.build_chunks(tmp_path / "研报_茅台_2026-05-18.pdf")
    assert chunks
    for _, _, meta in chunks:
        assert meta["publish_date"] == "2026-05-18" and meta["publish_date_source"] == "filename"
        assert meta["source"] == "研报_茅台_2026-05-18.pdf"                 # 既有键不变
    chunks2 = indexer.build_chunks(tmp_path / "无日期.pdf")
    assert all(m["publish_date"] == "" and m["publish_date_source"] == "unknown" for _, _, m in chunks2)
