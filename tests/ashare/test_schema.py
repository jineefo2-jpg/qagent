from __future__ import annotations
import pytest
duckdb = pytest.importorskip("duckdb")
from ashare.data import _db

EXPECTED_TABLES = {
    "calendar", "stock_basic", "stock_status", "daily_bar", "daily_basic",
    "financial_pit", "macro_indicator", "money_flow", "index_daily", "ingest_log",
}


def test_init_schema_creates_all_tables(tmp_db):
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    got = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert EXPECTED_TABLES <= got, f"缺表: {EXPECTED_TABLES - got}"
    conn.close()


def test_init_schema_is_idempotent(tmp_db):
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    _db.init_schema(conn)          # 第二次不得抛
    conn.close()


def test_financial_pit_pk_includes_ann_date(tmp_db):
    """D3：ann_date 必须在主键里，否则同报告期的多次披露会互相覆盖。"""
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    conn.execute("""INSERT INTO financial_pit (ts_code, ann_date, end_date, report_type,
                    update_flag, n_income_attr_p) VALUES
                    ('600519.SH','2021-04-01','2020-12-31','1',0, 100.0),
                    ('600519.SH','2021-04-28','2020-12-31','1',0, 101.0)""")
    n = conn.execute("SELECT count(*) FROM financial_pit").fetchone()[0]
    assert n == 2, "同报告期不同公告日必须能共存，否则 PIT 无从谈起"
    conn.close()


def test_read_only_connection_rejects_write(tmp_db):
    """D1 的最硬一层：只读连接上任何 DML 都必须抛异常。"""
    w = _db.connect_write(tmp_db); _db.init_schema(w); w.close()
    r = _db.connect_read(tmp_db)
    with pytest.raises(Exception):
        r.execute("INSERT INTO calendar (trade_date, is_open) VALUES ('2024-01-02', TRUE)")
    r.close()
