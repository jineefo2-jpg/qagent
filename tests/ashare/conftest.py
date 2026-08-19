from __future__ import annotations
import datetime as dt
import pathlib
import pytest


@pytest.fixture
def tmp_db(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "test_market.duckdb")


# ── 真实交易日历片段：2023-12-25 ~ 2024-02-02（含元旦休市、周末）──
_TRADING_DAYS = [
    "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29",
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
    "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26",
    "2024-01-29", "2024-01-30", "2024-01-31",
    "2024-02-01", "2024-02-02",
]
TRADING_DAYS = [dt.date.fromisoformat(s) for s in _TRADING_DAYS]


def _all_days(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


@pytest.fixture
def market_db(tmp_path: pathlib.Path) -> str:
    """一个最小但语义完整的 market 库：日历 + 4 只股票（正常但中途停牌 / ST 区间 / 已退市 / 次新）。
    写完即关闭写连接（同进程不可同时持读写连接）。返回路径。"""
    duckdb = pytest.importorskip("duckdb")
    from ashare.data import _db

    path = str(tmp_path / "market.duckdb")
    c = _db.connect_write(path)
    _db.init_schema(c)

    # 日历
    open_set = set(TRADING_DAYS)
    prev = None
    rows = []
    for d in _all_days(TRADING_DAYS[0], TRADING_DAYS[-1]):
        is_open = d in open_set
        rows.append((d, is_open, prev))
        if is_open:
            prev = d
    c.executemany("INSERT INTO calendar VALUES (?, ?, ?)", rows)

    # 股票：A 正常；B 2024-01-10~01-19 为 ST；C 2024-01-24 退市；D 次新（2023-12-01 上市）
    c.executemany(
        "INSERT INTO stock_basic (ts_code, symbol, name, industry, market, list_date, delist_date, is_hs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [("A00001.SZ", "A00001", "甲", "银行", "主板", dt.date(2010, 1, 1), None, "S"),
         ("B00002.SZ", "B00002", "乙", "白酒", "主板", dt.date(2010, 1, 1), None, "S"),
         ("C00003.SH", "C00003", "丙", "钢铁", "主板", dt.date(2010, 1, 1), dt.date(2024, 1, 24), "N"),
         ("D00004.SZ", "D00004", "丁", "银行", "创业板", dt.date(2023, 12, 1), None, "N")])
    c.executemany(
        "INSERT INTO stock_status VALUES (?, ?, ?, ?)",
        [("A00001.SZ", dt.date(2010, 1, 1), None, "NORMAL"),
         ("B00002.SZ", dt.date(2010, 1, 1), dt.date(2024, 1, 9), "NORMAL"),
         ("B00002.SZ", dt.date(2024, 1, 10), dt.date(2024, 1, 19), "ST"),
         ("B00002.SZ", dt.date(2024, 1, 20), None, "NORMAL"),
         ("C00003.SH", dt.date(2010, 1, 1), dt.date(2024, 1, 24), "NORMAL"),
         ("D00004.SZ", dt.date(2023, 12, 1), None, "NORMAL")])
    c.executemany(
        "INSERT INTO industry_member VALUES (?, ?, ?, ?, ?, ?)",
        [("A00001.SZ", "银行", "股份制银行", "股份制银行", dt.date(2010, 1, 1), None),
         ("B00002.SZ", "食品饮料", "白酒", "白酒", dt.date(2010, 1, 1), None),
         ("C00003.SH", "钢铁", "普钢", "普钢", dt.date(2010, 1, 1), None),
         ("D00004.SZ", "银行", "城商行", "城商行", dt.date(2023, 12, 1), None)])
    c.execute("INSERT INTO _meta VALUES ('industry_source', 'sw')")   # fixture 的行业是真实成分历史

    # 日线：每股每交易日一行（D9）。A 在 01-15~01-17 停牌；价格线性上涨便于断言；C 退市后无行
    bars = []
    # base_amt：A 最活跃、C 最不活跃（流动性剔除的靶子）
    for code, base, susp, base_amt in (
            ("A00001.SZ", 10.0, {dt.date(2024, 1, 15), dt.date(2024, 1, 16), dt.date(2024, 1, 17)}, 1e6),
            ("B00002.SZ", 20.0, set(), 5e5),
            ("C00003.SH", 5.0, set(), 1e4),
            ("D00004.SZ", 30.0, set(), 2e5)):
        prev_close = None
        for i, d in enumerate(TRADING_DAYS):
            if code == "C00003.SH" and d > dt.date(2024, 1, 24):
                break
            if code == "D00004.SZ" and d < dt.date(2023, 12, 1):
                continue
            px = base + i * 0.1
            if d in susp:
                bars.append((code, d, prev_close, prev_close, prev_close, prev_close, prev_close,
                             0.0, 0.0, 1.0, None, None, "unknown", True))
                continue
            pre = prev_close if prev_close is not None else px
            bars.append((code, d, px, px + 0.05, px - 0.05, px, pre,
                         1000.0 + i, base_amt * (1 + i / 10), 1.0 + i * 0.01,
                         round(pre * 1.1, 2), round(pre * 0.9, 2), "rule", False))
            prev_close = px
    c.executemany(
        "INSERT INTO daily_bar (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, "
        "adj_factor, limit_up, limit_down, limit_source, is_suspended) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", bars)

    # daily_basic：市值 —— C 最小市值（成交额在 daily_bar 的 base_amt 里分层）
    db_rows = []
    for code, mv in (("A00001.SZ", 1e6), ("B00002.SZ", 5e5), ("C00003.SH", 1e4), ("D00004.SZ", 2e5)):
        for d in TRADING_DAYS:
            if code == "C00003.SH" and d > dt.date(2024, 1, 24):
                break
            if code == "D00004.SZ" and d < dt.date(2023, 12, 1):
                continue
            db_rows.append((code, d, 1.0, 1.2, 1.0, 10.0, 11.0, 1.5, 2.0, 2.1, 1.0, 1.1,
                            mv / 10, mv / 10, mv / 12, mv, mv * 0.8))
    c.executemany(
        "INSERT INTO daily_basic (ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio, "
        "pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", db_rows)

    c.close()
    return path
