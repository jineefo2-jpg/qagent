from __future__ import annotations
import datetime as dt
from ashare.data.limits import compute_limits

D = dt.date
OLD = D(2001, 8, 27)          # 足够老的上市日，避开新股宽限


def test_main_board_10pct():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, OLD, "NORMAL")
    assert (up, dn, src) == (110.0, 90.0, "rule")


def test_st_on_main_board_5pct():
    up, dn, _ = compute_limits("600519.SH", D(2024,1,10), 100.0, OLD, "ST")
    assert (up, dn) == (105.0, 95.0)


def test_star_st_on_main_board_5pct():
    up, dn, _ = compute_limits("000001.SZ", D(2024,1,10), 100.0, OLD, "*ST")
    assert (up, dn) == (105.0, 95.0)


def test_chinext_was_10pct_before_20200824():
    up, _, _ = compute_limits("300750.SZ", D(2020,8,21), 100.0, D(2018,6,11), "NORMAL")
    assert up == 110.0


def test_chinext_became_20pct_on_20200824():
    up, _, _ = compute_limits("300750.SZ", D(2020,8,24), 100.0, D(2018,6,11), "NORMAL")
    assert up == 120.0


def test_chinext_301_and_302_prefixes_are_chinext():
    """创业板代码段 300000–309999：301xxx（2020 后新股）、302xxx 都不是主板。"""
    up1, _, _ = compute_limits("301001.SZ", D(2024,1,10), 100.0, D(2021,1,1), "NORMAL")
    up2, _, _ = compute_limits("302132.SZ", D(2024,1,10), 100.0, D(2021,1,1), "NORMAL")
    assert up1 == 120.0 and up2 == 120.0


def test_star_market_20pct():
    up, _, _ = compute_limits("688981.SH", D(2021,5,10), 100.0, D(2020,7,16), "NORMAL")
    assert up == 120.0


def test_st_on_star_and_chinext_keeps_board_limit():
    """ST 5% 是主板规则。科创板 / 注册制后创业板的风险警示股沿用 20%（如 *ST紫晶 688086、*ST长动 300612）。"""
    up_star, _, _ = compute_limits("688086.SH", D(2023,3,1), 10.0, D(2020,2,26), "*ST")
    up_cx, _, _ = compute_limits("300612.SZ", D(2021,6,1), 10.0, D(2017,3,1), "*ST")
    assert up_star == 12.0 and up_cx == 12.0


def test_bse_30pct_even_when_st():
    up, _, _ = compute_limits("830799.BJ", D(2024,1,10), 100.0, D(2021,11,15), "ST")
    assert up == 130.0


def test_rounding_is_decimal_half_up_not_python_round():
    """交易所按十进制四舍五入到 0.01。round() 是二进制浮点 + 银行家舍入：
    1.45×0.9=1.305 → round 给 1.30，交易所给 1.31 → 真实跌停 1.31 被判"可交易"→ 幽灵成交。"""
    up, dn, _ = compute_limits("600000.SH", D(2024,1,10), 1.45, OLD, "NORMAL")
    assert (up, dn) == (1.60, 1.31)
    up, dn, _ = compute_limits("600000.SH", D(2024,1,10), 12.35, OLD, "NORMAL")
    assert (up, dn) == (13.59, 11.12)


def test_new_listing_returns_unknown():
    """新股上市初期（主板首日无涨跌幅 / 科创创业前 5 日无限制）：规则算不出 → unknown → 上层判不可交易。"""
    assert compute_limits("301000.SZ", D(2024,1,10), 100.0, D(2024,1,10), "NORMAL") == (None, None, "unknown")


def test_grace_covers_five_trading_days_across_long_holiday():
    """2024-04-26(周五)上市，跨五一假期后 5 月 6 日仍是第 4 个交易日 → 必须还是 unknown。"""
    assert compute_limits("688999.SH", D(2024,5,6), 100.0, D(2024,4,26), "NORMAL")[2] == "unknown"
    assert compute_limits("688999.SH", D(2024,5,20), 100.0, D(2024,4,26), "NORMAL")[2] == "rule"


def test_unknown_list_date_is_unknown():
    assert compute_limits("600000.SH", D(2024,1,10), 100.0, None, "NORMAL")[2] == "unknown"


def test_invalid_pre_close_is_unknown():
    assert compute_limits("600000.SH", D(2024,1,10), None, OLD, "NORMAL")[2] == "unknown"
    assert compute_limits("600000.SH", D(2024,1,10), 0.0, OLD, "NORMAL")[2] == "unknown"


def test_delist_period_returns_unknown():
    assert compute_limits("600519.SH", D(2024,1,10), 100.0, OLD, "DELIST_PERIOD")[2] == "unknown"


def test_star_before_open_date_is_unknown():
    assert compute_limits("688001.SH", D(2019,7,19), 100.0, D(2018,1,1), "NORMAL")[2] == "unknown"
