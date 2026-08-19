"""Task 10：get_universe / explain_universe / get_stock_basic / get_industry（D5）。

market_db 里：A 正常（01-15~01-17 停牌）；B 01-10~01-19 为 ST；C 2024-01-24 退市；D 2023-12-01 上市（次新）。
成交额 A > B > D > C。
"""
from __future__ import annotations
import datetime as dt
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import query

D = dt.date


@pytest.fixture
def q(market_db):
    query.open_db(market_db)
    yield query
    query.close_db()


# ══════════════ get_universe：六步剔除 ══════════════
def test_seasoning_drops_recent_ipo(q):
    u = q.get_universe("2024-01-05")
    assert "D00004.SZ" not in u                          # 上市 35 天 < 250
    assert "D00004.SZ" in q.get_universe("2024-01-05", min_list_days=30)


def test_st_period_is_excluded_only_inside_the_window(q):
    assert "B00002.SZ" in q.get_universe("2024-01-05")   # ST 前
    assert "B00002.SZ" not in q.get_universe("2024-01-12")  # ST 中
    assert "B00002.SZ" in q.get_universe("2024-01-25")   # 摘帽后
    assert "B00002.SZ" in q.get_universe("2024-01-12", exclude_st=False)


def test_suspended_is_excluded(q):
    assert "A00001.SZ" not in q.get_universe("2024-01-16")
    assert "A00001.SZ" in q.get_universe("2024-01-16", exclude_suspended=False)
    assert "A00001.SZ" in q.get_universe("2024-01-18")


def test_delisted_is_excluded_after_delist_date(q):
    assert "C00003.SH" in q.get_universe("2024-01-23")
    assert "C00003.SH" not in q.get_universe("2024-01-25")


def test_spec_assertion_no_lookahead_in_universe(q):
    """规格 §11：池内不含 list_date > as_of-250d 或 delist_date <= as_of 的股票。"""
    as_of = D(2024, 1, 25)
    u = q.get_universe(as_of)
    basic = q.get_stock_basic(as_of)
    for code in u:
        assert basic.loc[code, "list_date"] <= as_of - dt.timedelta(days=250)
        assert basic.loc[code, "delist_date"] is None or basic.loc[code, "delist_date"] > as_of


def test_liquidity_quantile_is_computed_after_hard_filters(q):
    """01-25：硬性剔除后池 = {A, B}（C 已退市、D 次新）。剔后 50% → 只剩 A。
    若分位在硬性剔除前算，底部 50% 是 {C, D}，A、B 都留下 —— 结果不同。"""
    assert q.get_universe("2024-01-25", liquidity_drop_pct=0.5) == ["A00001.SZ"]
    assert q.get_universe("2024-01-25", liquidity_drop_pct=0.0) == ["A00001.SZ", "B00002.SZ"]


def test_default_liquidity_drop_floor(q):
    """N=3 时 floor(3×0.2)=0，默认不剔；这是 floor 语义的钉子。"""
    assert q.get_universe("2024-01-05") == ["A00001.SZ", "B00002.SZ", "C00003.SH"]


def test_markets_filter(q):
    assert q.get_universe("2024-01-05", min_list_days=1, markets=["创业板"]) == ["D00004.SZ"]


def test_returns_sorted_and_empty_is_list(q):
    u = q.get_universe("2024-01-05")
    assert u == sorted(u)
    assert q.get_universe("2024-01-05", markets=["北交所"]) == []


# ══════════════ explain_universe ══════════════
def test_explain_matches_get_universe_and_gives_reason(q):
    ex = q.explain_universe("2024-01-12")
    assert list(ex.columns) == ["step1_listed", "step2_seasoned", "step3_not_st", "step4_tradable",
                                "step5_market", "step6_liquid", "included", "drop_reason"]
    assert sorted(ex.index[ex.included]) == q.get_universe("2024-01-12")
    assert ex.loc["B00002.SZ", "drop_reason"] == "st"
    assert ex.loc["D00004.SZ", "drop_reason"] == "seasoning"
    assert ex.loc["A00001.SZ", "drop_reason"] == ""


def test_explain_reason_for_delisted_and_suspended(q):
    ex = q.explain_universe("2024-01-25")
    assert ex.loc["C00003.SH", "drop_reason"] == "delisted"
    ex16 = q.explain_universe("2024-01-16")
    assert ex16.loc["A00001.SZ", "drop_reason"] == "suspended"


# ══════════════ get_stock_basic / get_industry ══════════════
def test_get_stock_basic_shape(q):
    b = q.get_stock_basic("2024-01-05")
    assert list(b.columns) == ["symbol", "name", "sw_l1", "sw_l2", "sw_l3", "market", "list_date", "delist_date", "is_hs"]
    assert b.loc["A00001.SZ", "sw_l1"] == "银行" and b.loc["A00001.SZ", "market"] == "主板"
    sub = q.get_stock_basic("2024-01-05", ts_codes=["B00002.SZ"])
    assert list(sub.index) == ["B00002.SZ"]


def test_get_industry_pools_small_industries(q):
    ind = q.get_industry("2024-01-05")
    assert set(ind.unique()) == {"__OTHER__"}          # 每个行业 < 5 家
    ind1 = q.get_industry("2024-01-05", min_members=1)
    assert ind1["A00001.SZ"] == "银行" and ind1["B00002.SZ"] == "食品饮料"
    assert q.get_industry("2024-01-05", level="l2", min_members=1)["A00001.SZ"] == "股份制银行"


def test_get_industry_is_pit_by_in_out_date(market_db):
    """行业变更历史：在 out_date 之前用旧行业，之后用新行业。"""
    from ashare.data import _db
    w = _db.connect_write(market_db)
    w.execute("UPDATE industry_member SET out_date = DATE '2024-01-10' WHERE ts_code='B00002.SZ'")
    w.execute("INSERT INTO industry_member VALUES ('B00002.SZ', '医药生物', '中药', '中药', DATE '2024-01-11', NULL)")
    w.close()
    query.open_db(market_db)
    try:
        assert query.get_industry("2024-01-10", min_members=1)["B00002.SZ"] == "食品饮料"
        assert query.get_industry("2024-01-11", min_members=1)["B00002.SZ"] == "医药生物"
    finally:
        query.close_db()


def test_universe_functions_reject_bad_dates(q):
    with pytest.raises(query.AsOfDateError):
        q.get_universe("2030-01-01")
    with pytest.raises(query.AsOfDateError):
        q.get_industry("nope")


def test_overlapping_status_rows_do_not_crash_latest_segment_wins(market_db):
    """schema 允许区间重叠（PK 只含 start_date）；ingest 原样拷贝 namechange。取 start_date 最新一段。"""
    from ashare.data import _db
    w = _db.connect_write(market_db)
    w.execute("INSERT INTO stock_status VALUES ('A00001.SZ', DATE '2020-01-01', NULL, 'NORMAL')")   # 与既有 2010 段重叠
    w.execute("INSERT INTO stock_status VALUES ('A00001.SZ', DATE '2024-01-04', NULL, 'ST')")        # 更新的一段：ST
    w.close()
    query.open_db(market_db)
    try:
        assert "A00001.SZ" not in query.get_universe("2024-01-05")
        assert "A00001.SZ" in query.get_universe("2024-01-03")
    finally:
        query.close_db()


def test_overlapping_industry_rows_latest_in_date_wins(market_db):
    from ashare.data import _db
    w = _db.connect_write(market_db)
    w.execute("INSERT INTO industry_member VALUES ('A00001.SZ', '医药生物', '中药', '中药', DATE '2020-01-01', NULL)")
    w.close()
    query.open_db(market_db)
    try:
        assert query.get_industry("2024-01-05", min_members=1)["A00001.SZ"] == "医药生物"
    finally:
        query.close_db()


def test_markets_empty_list_means_no_market(q):
    assert q.get_universe("2024-01-05", markets=[]) == []
