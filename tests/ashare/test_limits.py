from __future__ import annotations
import datetime as dt
import pytest
from ashare.data.limits import compute_limits

D = dt.date


def test_main_board_10pct():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "NORMAL")
    assert (round(up,2), round(dn,2), src) == (110.0, 90.0, "rule")


def test_st_5pct():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "ST")
    assert (round(up,2), round(dn,2)) == (105.0, 95.0)


def test_chinext_was_10pct_before_20200824():
    up, dn, _ = compute_limits("300750.SZ", D(2020,8,21), 100.0, D(2018,6,11), "NORMAL")
    assert round(up,2) == 110.0


def test_chinext_became_20pct_on_20200824():
    up, dn, _ = compute_limits("300750.SZ", D(2020,8,24), 100.0, D(2018,6,11), "NORMAL")
    assert round(up,2) == 120.0


def test_star_market_20pct():
    up, dn, _ = compute_limits("688981.SH", D(2021,5,10), 100.0, D(2020,7,16), "NORMAL")
    assert round(up,2) == 120.0


def test_bse_30pct():
    up, dn, _ = compute_limits("830799.BJ", D(2024,1,10), 100.0, D(2021,11,15), "NORMAL")
    assert round(up,2) == 130.0


def test_new_listing_returns_unknown():
    """新股上市首日无涨跌幅限制（主板）/ 前 5 日无限制（科创创业）——
    规则算不出，必须返回 unknown 让上层判为不可交易，绝不能假装 10%。"""
    up, dn, src = compute_limits("301000.SZ", D(2024,1,10), 100.0, D(2024,1,10), "NORMAL")
    assert (up, dn, src) == (None, None, "unknown")


def test_delist_period_returns_unknown():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "DELIST_PERIOD")
    assert src == "unknown"
