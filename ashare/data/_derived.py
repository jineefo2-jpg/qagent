"""derived 库的连接与 schema 管理 —— _db.py 的姊妹文件，管第二个数据库。

★ 为什么是两个库：market 库由 promote.py 用 os.replace 原子替换，
  因子值若住在里面，每次 promote 都被连锅端走；而且只有 P2 写 derived。
★ 与 _db.py 同一条约束：同一进程内对同一文件不能同时持有 connect_write 与 connect_read
  （DuckDB: "Can't open a connection to same database file with a different configuration"）。
"""
from __future__ import annotations
import pathlib

import duckdb

from . import _db
from ._db import DEFAULT_DERIVED_PATH   # 路径常量单一来源，别在这里再抄一遍字面量

_SCHEMA_SQL = pathlib.Path(__file__).with_name("derived_schema.sql")


def connect_write(path: str | None = None) -> duckdb.DuckDBPyConnection:
    """可写连接。只有因子计算与回测引擎允许调用。"""
    return _db.connect_write(path or DEFAULT_DERIVED_PATH)


def connect_read(path: str | None = None) -> duckdb.DuckDBPyConnection:
    """只读连接。D1 的最硬一层 —— DuckDB 在 read_only 连接上执行任何 DML 直接抛异常，
    不依赖任何人的自觉。"""
    p = path or DEFAULT_DERIVED_PATH
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(f"derived 库不存在: {p}（先算因子 / 跑回测落库）")
    return duckdb.connect(p, read_only=True)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """建表。幂等（全部 CREATE TABLE IF NOT EXISTS）。"""
    conn.execute(_SCHEMA_SQL.read_text(encoding="utf-8"))
