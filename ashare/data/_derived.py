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

# 路径常量单一来源在 _db，这里再导出一次：调用方只需要认识 _derived 一个模块
DEFAULT_DERIVED_PATH = _db.DEFAULT_DERIVED_PATH

_SCHEMA_SQL = pathlib.Path(__file__).with_name("derived_schema.sql")

# ★ 因子值的覆盖语义（Task 8 的 store 应导入而不是自己写一遍）：
#   重算 = 覆盖，且 snapshot_id 必须跟着更新。若写入方发的是 ON CONFLICT DO NOTHING，
#   留下的是陈旧值【配陈旧 snapshot_id】—— read() 会认为它属于当前快照并放行，静默喂进回测。
UPSERT_FACTOR_VALUE = """
INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code,
                          raw_value, processed_value, snapshot_id)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (factor_name, param_hash, trade_date, ts_code) DO UPDATE SET
  raw_value       = excluded.raw_value,
  processed_value = excluded.processed_value,
  snapshot_id     = excluded.snapshot_id
"""


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
