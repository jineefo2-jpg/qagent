"""derived 库（因子值 + 回测运行）的 schema 与连接约束。

market 库由 promote.py 原子替换（os.replace），因子值若住在里面每次 promote 都会被抹掉，
所以 derived 必须是独立文件。本文件守的就是这条边界，外加 D1 的只读硬保证。
"""
from __future__ import annotations
import datetime as dt
import pathlib

import pytest
duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, _derived

REPO = pathlib.Path(__file__).resolve().parents[2]
FACTOR_PK = ["factor_name", "param_hash", "trade_date", "ts_code"]


@pytest.fixture
def derived_db(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "derived.duckdb")


def _pk_columns(conn, table: str) -> list[str]:
    row = conn.execute(
        "SELECT constraint_column_names FROM duckdb_constraints() "
        "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'", [table]).fetchone()
    assert row, f"{table} 没有主键"
    return list(row[0])


def test_init_schema_creates_both_tables(derived_db):
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    got = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"factor_value", "backtest_run"} <= got, f"缺表: {{'factor_value', 'backtest_run'}} - {got}"
    conn.close()


def test_init_schema_is_idempotent(derived_db):
    """幂等的关键不是"第二次不抛"，是"第二次不清空缓存"。
    有人把 CREATE TABLE IF NOT EXISTS 改成 CREATE OR REPLACE TABLE，只断言不抛的测试照样过，
    而每次 init 都会抹掉全部已算因子值 —— 那才是这个测试要拦的回归。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    conn.execute("INSERT INTO factor_value VALUES ('f', 'p', DATE '2024-01-02', 'A.SZ', 1.0, 0.5, 'snap')")
    _derived.init_schema(conn)          # 第二次不得抛，也不得清空
    assert conn.execute("SELECT count(*) FROM factor_value").fetchone()[0] == 1
    conn.close()


def test_factor_value_pk_excludes_snapshot_id(derived_db):
    """主键是 (factor_name, param_hash, trade_date, ts_code)。
    snapshot_id 进主键 = 同一因子同一天在库里堆 N 份，取数要先挑快照，回测静默取到哪份全看运气。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    assert _pk_columns(conn, "factor_value") == FACTOR_PK
    conn.close()


def test_recompute_under_new_snapshot_overwrites(derived_db):
    """同因子同参同日同股在新快照下【覆盖】旧值，snapshot_id 列记录这行是哪份数据算出来的。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    ins = ("INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, "
           "raw_value, processed_value, snapshot_id) VALUES (?, ?, ?, ?, ?, ?, ?) "
           "ON CONFLICT (factor_name, param_hash, trade_date, ts_code) DO UPDATE SET "
           "raw_value = excluded.raw_value, processed_value = excluded.processed_value, "
           "snapshot_id = excluded.snapshot_id")
    conn.execute(ins, ["mom20", "ph1", "2024-01-05", "600519.SH", 1.0, 0.5, "snap_a"])
    conn.execute(ins, ["mom20", "ph1", "2024-01-05", "600519.SH", 2.0, 0.9, "snap_b"])

    rows = conn.execute("SELECT raw_value, processed_value, snapshot_id FROM factor_value").fetchall()
    assert rows == [(2.0, 0.9, "snap_b")], "换快照必须覆盖而不是堆两行"
    conn.close()


def test_factor_value_pk_is_enforced(derived_db):
    """主键不是注释：不带 ON CONFLICT 的重复写入必须撞约束，而不是悄悄多一行。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    ins = ("INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, "
           "raw_value, snapshot_id) VALUES (?, ?, ?, ?, ?, ?)")
    conn.execute(ins, ["mom20", "ph1", "2024-01-05", "600519.SH", 1.0, "snap_a"])
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(ins, ["mom20", "ph1", "2024-01-05", "600519.SH", 2.0, "snap_b"])
    conn.close()


def test_factor_value_keys_are_independent(derived_db):
    """换参数 / 换日期 / 换股票都是不同的行 —— 主键 4 列缺一不可。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    conn.executemany(
        "INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, raw_value, snapshot_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [("mom20", "ph1", "2024-01-05", "600519.SH", 1.0, "s"),
         ("mom20", "ph2", "2024-01-05", "600519.SH", 1.0, "s"),     # 换 param_hash
         ("mom20", "ph1", "2024-01-08", "600519.SH", 1.0, "s"),     # 换 trade_date
         ("mom20", "ph1", "2024-01-05", "000001.SZ", 1.0, "s"),     # 换 ts_code
         ("mom60", "ph1", "2024-01-05", "600519.SH", 1.0, "s")])    # 换 factor_name
    assert conn.execute("SELECT count(*) FROM factor_value").fetchone()[0] == 5
    conn.close()


def test_backtest_run_columns_and_pk(derived_db):
    """回测运行表：run_id 主键，param_hash 与 data_snapshot_id 同在（D7 缺一不可）。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    assert _pk_columns(conn, "backtest_run") == ["run_id"]
    cols = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'backtest_run'").fetchall()}
    assert {"run_id", "param_hash", "data_snapshot_id", "engine_version", "started_at",
            "elapsed_sec", "config_json", "metrics_json", "is_oos"} <= cols, f"缺列: {cols}"

    conn.execute("INSERT INTO backtest_run (run_id, param_hash, data_snapshot_id, is_oos) "
                 "VALUES ('r1', 'ph1', 'snap_a', TRUE)")
    with pytest.raises(duckdb.ConstraintException):
        conn.execute("INSERT INTO backtest_run (run_id, param_hash, data_snapshot_id, is_oos) "
                     "VALUES ('r1', 'ph2', 'snap_b', FALSE)")
    conn.close()


def test_read_only_connection_rejects_write(derived_db):
    """D1 的最硬一层：只读连接上任何 DML 都必须抛异常。
    精确到 DuckDB 的只读异常 —— 缺表 / SQL 笔误也抛 Exception，那不算证明。"""
    w = _derived.connect_write(derived_db); _derived.init_schema(w); w.close()
    r = _derived.connect_read(derived_db)
    with pytest.raises(duckdb.InvalidInputException, match="read-only"):
        r.execute("INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, raw_value) "
                  "VALUES ('mom20', 'ph1', '2024-01-05', '600519.SH', 1.0)")
    r.close()


def test_read_only_connection_can_read(derived_db):
    w = _derived.connect_write(derived_db)
    _derived.init_schema(w)
    w.execute("INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, raw_value, snapshot_id) "
              "VALUES ('mom20', 'ph1', '2024-01-05', '600519.SH', 1.5, 'snap_a')")
    w.close()
    r = _derived.connect_read(derived_db)
    assert r.execute("SELECT raw_value FROM factor_value").fetchone()[0] == 1.5
    r.close()


def test_connect_read_missing_file_raises(tmp_path):
    """不存在时给明确报错，而不是 DuckDB 建个空库让上层拿到 0 行因子还以为算完了。"""
    with pytest.raises(FileNotFoundError):
        _derived.connect_read(str(tmp_path / "nope.duckdb"))


def test_path_defaults_to_default_derived_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = _derived.connect_write()          # path=None → DEFAULT_DERIVED_PATH
    _derived.init_schema(conn)
    conn.close()
    assert (tmp_path / _derived.DEFAULT_DERIVED_PATH).exists()


def test_derived_path_is_a_separate_file_from_market():
    """promote.py 用 os.replace 整体换掉 market 库；因子值住在里面会被一起换掉。"""
    assert _derived.DEFAULT_DERIVED_PATH != _db.DEFAULT_MARKET_PATH


def test_gitignore_covers_derived_db():
    assert "data/ashare_derived.duckdb*" in (REPO / ".gitignore").read_text(encoding="utf-8")


def test_d7_provenance_columns_are_not_null(derived_db):
    """D7 说 param_hash 与 data_snapshot_id「缺一不可」。PK 列由 DuckDB 隐式 NOT NULL —— 恰好
    保护了【缓存键】而漏掉【溯源列】。缺了 snapshot_id 的行无法追溯来源，且 store.read 的
    `snapshot_id = ?` 对 NULL 恒为 NULL，这些行会从每次读取里静默消失而不是报错。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute("INSERT INTO factor_value (factor_name, param_hash, trade_date, ts_code, raw_value) "
                     "VALUES ('f', 'p', DATE '2024-01-02', 'A.SZ', 1.0)")          # 无 snapshot_id
    with pytest.raises(duckdb.ConstraintException):
        conn.execute("INSERT INTO backtest_run (run_id, param_hash) VALUES ('r1', 'p1')")   # 无 data_snapshot_id
    with pytest.raises(duckdb.ConstraintException):
        conn.execute("INSERT INTO backtest_run (run_id, data_snapshot_id) VALUES ('r2', 's1')")  # 无 param_hash
    conn.close()


def test_upsert_factor_value_overwrites_value_and_snapshot(derived_db):
    """覆盖语义必须活在 ashare/ 里而不是测试里：写入方若发 DO NOTHING，会留下陈旧值【配陈旧
    snapshot_id】，read() 会认为它属于当前快照并放行 —— 静默把另一批数据算出的因子喂进回测。"""
    conn = _derived.connect_write(derived_db)
    _derived.init_schema(conn)
    key = ("f", "p", dt.date(2024, 1, 2), "A.SZ")
    conn.execute(_derived.UPSERT_FACTOR_VALUE, [*key, 1.0, 0.5, "snap_a"])
    conn.execute(_derived.UPSERT_FACTOR_VALUE, [*key, 2.0, 0.9, "snap_b"])
    assert conn.execute("SELECT raw_value, processed_value, snapshot_id FROM factor_value").fetchall() \
        == [(2.0, 0.9, "snap_b")]
    conn.close()
