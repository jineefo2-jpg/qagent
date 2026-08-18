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


def test_adj_factor_is_carried_forward():
    """占位行的复权因子前推（用不同因子值才看得出方向）；『不 bfill』由 test_leading_suspension_without_seed_is_nan_not_fabricated 钉住。"""
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,5), 105.0)])
    adj = pd.DataFrame([{"ts_code": "600519.SH", "trade_date": D(2024,1,2), "adj_factor": 1.0},
                        {"ts_code": "600519.SH", "trade_date": D(2024,1,5), "adj_factor": 1.2}])
    out = normalize_daily_bar(daily, adj, None, CAL, BASIC, STATUS).set_index("trade_date")
    assert out.loc[D(2024,1,3)].adj_factor == 1.0 and out.loc[D(2024,1,4)].adj_factor == 1.0
    assert out.loc[D(2024,1,8)].adj_factor == 1.2


def test_leading_suspension_uses_seed_from_previous_batch():
    """分批拉取：批次首日停牌但上一批有前收 → 用种子前推，而不是写 NaN / 用未来值回填。"""
    daily = _daily([(D(2024,1,5), 105.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,5)]), None, CAL, BASIC, STATUS,
                              seed_close=99.0, seed_adj=0.9).set_index("trade_date")
    for d in (D(2024,1,2), D(2024,1,3), D(2024,1,4)):
        assert out.loc[d].close == 99.0 and out.loc[d].adj_factor == 0.9 and out.loc[d].is_suspended


def test_leading_suspension_without_seed_is_nan_not_fabricated():
    daily = _daily([(D(2024,1,5), 105.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,5)]), None, CAL, BASIC, STATUS).set_index("trade_date")
    assert pd.isna(out.loc[D(2024,1,2)].close) and pd.isna(out.loc[D(2024,1,2)].adj_factor)
    assert out.loc[D(2024,1,2)].limit_source == "unknown", "无前收算不出涨跌停 → unknown，不得给 NaN 冒充 rule"


def test_consecutive_suspended_days_share_last_real_close():
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,8), 108.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,8)]), None, CAL, BASIC, STATUS).set_index("trade_date")
    assert out.loc[D(2024,1,3)].close == out.loc[D(2024,1,4)].close == out.loc[D(2024,1,5)].close == 100.0


def test_api_nan_limit_is_treated_as_missing():
    """API 返回 NaN 涨跌停不能当有效值：下游 close >= NaN 恒 False → 真涨停被判可交易。"""
    daily = _daily([(D(2024,1,2), 100.0)])
    lim = pd.DataFrame([{"ts_code": "600519.SH", "trade_date": D(2024,1,2),
                         "up_limit": float("nan"), "down_limit": float("nan")}])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), lim, [D(2024,1,2)], BASIC, STATUS)
    r = out.iloc[0]
    assert r.limit_source == "rule" and r.limit_up == 110.0


def test_duplicate_source_rows_do_not_inflate_row_count():
    """Tushare 偶发重复行；行数 == 交易日数是本函数存在的唯一理由，必须守住。"""
    daily = pd.concat([_daily([(D(2024,1,2), 100.0)]), _daily([(D(2024,1,2), 100.5)])])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), None, [D(2024,1,2), D(2024,1,3)], BASIC, STATUS)
    assert len(out) == 2 and out.iloc[0].close == 100.5     # keep="last"


def test_rows_stop_at_delist_date_inclusive():
    basic = {"ts_code": "600519.SH", "list_date": D(2001,8,27), "delist_date": D(2024,1,4)}
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,3), 99.0), (D(2024,1,4), 98.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,3), D(2024,1,4)]), None, CAL, basic, STATUS)
    assert out.trade_date.max() == D(2024,1,4) and len(out) == 3


def test_parse_date_rejects_numeric():
    from ashare.data.ingest import _parse_date
    import pytest as _pt
    with _pt.raises(TypeError):
        _parse_date(20240102)
    assert _parse_date("20240102") == _parse_date("2024-01-02") == D(2024,1,2)
