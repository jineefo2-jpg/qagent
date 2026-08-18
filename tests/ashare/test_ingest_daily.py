from __future__ import annotations
import datetime as dt
import pandas as pd
from ashare.data.ingest import normalize_daily_bar

D = dt.date
CAL = [D(2024,1,2), D(2024,1,3), D(2024,1,4), D(2024,1,5), D(2024,1,8)]
BASIC = {"ts_code": "600519.SH", "list_date": D(2001,8,27), "delist_date": None}
STATUS = [{"start_date": D(2001,8,27), "end_date": None, "status": "NORMAL"}]


def _daily(dates_prices):
    return pd.DataFrame(
        [{"ts_code": "600519.SH", "trade_date": d, "open": p, "high": p, "low": p,
          "close": p, "pre_close": p, "vol": 100.0, "amount": 1000.0} for d, p in dates_prices])


def _adj(dates):
    return pd.DataFrame([{"ts_code": "600519.SH", "trade_date": d, "adj_factor": 1.0} for d in dates])


def test_suspended_days_get_placeholder_rows():
    """★ D9 核心断言：Tushare 停牌日不返回行，normalize 必须补齐。
    不补则 rolling(20) 拿到的是 20 条记录而非 20 个交易日，因子静默污染。"""
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,5), 105.0), (D(2024,1,8), 106.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,5), D(2024,1,8)]),
                              None, CAL, BASIC, STATUS)
    assert len(out) == len(CAL), f"必须补齐到 {len(CAL)} 个交易日，实际 {len(out)}"
    sus = out[out.is_suspended]
    assert set(sus.trade_date) == {D(2024,1,3), D(2024,1,4)}


def test_placeholder_rows_carry_prev_close_and_zero_volume():
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,5), 105.0), (D(2024,1,8), 106.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,5), D(2024,1,8)]),
                              None, CAL, BASIC, STATUS).set_index("trade_date")
    r = out.loc[D(2024,1,3)]
    assert r.vol == 0 and r.amount == 0
    assert r.open == r.high == r.low == r.close == 100.0, "占位行 OHLC 全取前收"
    assert r.adj_factor == 1.0, "占位行沿用前一日复权因子"


def test_no_rows_outside_listing_window():
    """未上市/已退市区间不得补行 —— 补了就是幸存者偏差的反面（凭空造出行情）。"""
    basic = {"ts_code": "600519.SH", "list_date": D(2024,1,4), "delist_date": None}
    out = normalize_daily_bar(_daily([(D(2024,1,5), 105.0)]), _adj([D(2024,1,5)]),
                              None, CAL, basic, STATUS)
    assert out.trade_date.min() >= D(2024,1,4)
    assert len(out) == 3                       # 1/4 占位 + 1/5 实际 + 1/8 占位（1/2、1/3 在上市前，不得出现）
    assert list(out.is_suspended) == [True, False, True]
    assert pd.isna(out.iloc[0].close), "上市首日即停牌且无前收 → OHLC 为 NaN，不得编造价格"


def test_limit_falls_back_to_rule_when_api_missing():
    daily = _daily([(D(2024,1,2), 100.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), None, [D(2024,1,2)], BASIC, STATUS)
    r = out.iloc[0]
    assert r.limit_source == "rule" and round(r.limit_up, 2) == 110.0


def test_limit_prefers_api_over_rule():
    daily = _daily([(D(2024,1,2), 100.0)])
    lim = pd.DataFrame([{"ts_code": "600519.SH", "trade_date": D(2024,1,2),
                         "up_limit": 111.11, "down_limit": 88.88}])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), lim, [D(2024,1,2)], BASIC, STATUS)
    r = out.iloc[0]
    assert r.limit_source == "api" and r.limit_up == 111.11


def test_row_count_equals_trading_days_in_listing_window():
    """P1 验收断言的单元版：行数 == 在市区间交易日数，误差为 0（不是 0.1%）。"""
    daily = _daily([(D(2024,1,2), 100.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), None, CAL, BASIC, STATUS)
    assert len(out) == 5
