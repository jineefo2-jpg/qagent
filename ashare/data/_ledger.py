"""ledger 库（信号/持仓/确认）的连接与 schema 管理 —— _db.py 的第三个姊妹。

★ 与 derived 的本质区别：derived 是缓存（可 rm 重算），本库是【不可重算的用户资产】，
  因此带 market 同款的 schema_version 守卫 —— 库版本高于代码即拒开，加表走迁移不走 rm。
★ 同一进程内对同一文件不能同时持有 connect_write 与 connect_read（DuckDB 限制，同 _db.py）。
"""
from __future__ import annotations
import pathlib

import duckdb

DEFAULT_LEDGER_PATH = "data/ashare_ledger.duckdb"
SCHEMA_VERSION = 1
_SCHEMA_SQL = pathlib.Path(__file__).with_name("ledger_schema.sql")


def connect_write(path: str | None = None) -> duckdb.DuckDBPyConnection:
    """可写连接。只有 strategy CLI 与 server 的回写端点允许调用（D1：LLM 层无写句柄）。"""
    p = path or DEFAULT_LEDGER_PATH
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(p)


def connect_read(path: str | None = None) -> duckdb.DuckDBPyConnection:
    """只读连接。库不存在 = 还没生成过任何信号，FileNotFoundError 由调用方转译。"""
    p = path or DEFAULT_LEDGER_PATH
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(f"ledger 库不存在: {p}（先跑 python -m ashare.strategy.plan 生成首份清单）")
    return duckdb.connect(p, read_only=True)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """建表。幂等；库版本高于代码版本 → 拒绝（旧代码写新库会静默漏列/漏表，同 _db.init_schema）。"""
    have = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "_meta" in have:
        row = conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()
        if row and int(row[0]) > SCHEMA_VERSION:
            raise RuntimeError(f"ledger schema_version={row[0]} 高于代码 SCHEMA_VERSION={SCHEMA_VERSION}，请升级代码")
    conn.execute(_SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
                 "ON CONFLICT (key) DO UPDATE SET value = excluded.value", [str(SCHEMA_VERSION)])
