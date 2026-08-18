from __future__ import annotations
import pathlib
import pytest


@pytest.fixture
def tmp_db(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "test_market.duckdb")
