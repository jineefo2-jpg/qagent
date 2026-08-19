"""DuckDB 连接与 schema 管理。ingest 是唯一写者，query 只用 read_only 连接。

★ 同一进程内对同一文件不能同时持有 connect_write 与 connect_read
  （DuckDB: "Can't open a connection to same database file with a different configuration"）。
  写者与读者必须先后而非并存 —— 架构文档 §10.2 的影子文件 + 原子替换正是为此。"""
from __future__ import annotations
import pathlib

import duckdb

SCHEMA_VERSION = 2       # v2: + industry_member
_SCHEMA_SQL = pathlib.Path(__file__).with_name("schema.sql")

DEFAULT_MARKET_PATH = "data/ashare_market.duckdb"
DEFAULT_DERIVED_PATH = "data/ashare_derived.duckdb"


def connect_write(path: str) -> duckdb.DuckDBPyConnection:
    """可写连接。只有 ingest.py / promote.py 允许调用。"""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def connect_read(path: str) -> duckdb.DuckDBPyConnection:
    """只读连接。D1 的最硬一层 —— DuckDB 在 read_only 连接上执行任何 DML 直接抛异常，
    不依赖任何人的自觉。query.py 只能用这个。"""
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"数据库不存在: {path}（先跑 python -m ashare.data.pipeline full）")
    return duckdb.connect(path, read_only=True)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """建表。幂等（全部 CREATE TABLE IF NOT EXISTS，加表向前兼容）。
    库里版本【高于】代码版本 → 拒绝：旧代码写新库会静默漏列/漏表。"""
    have = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "_meta" in have:
        row = conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()
        if row and int(row[0]) > SCHEMA_VERSION:
            raise RuntimeError(f"数据库 schema_version={row[0]} 高于代码 SCHEMA_VERSION={SCHEMA_VERSION}，请升级代码")
    conn.execute(_SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [str(SCHEMA_VERSION)],
    )
