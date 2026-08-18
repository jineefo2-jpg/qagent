from __future__ import annotations
import datetime as dt
import pandas as pd
from ashare.data.ingest import derive_stock_status

D = dt.date


def _nc(rows):
    return pd.DataFrame(rows, columns=["ts_code", "name", "start_date", "end_date"])


def _basic(rows):
    return pd.DataFrame(rows, columns=["ts_code", "name", "list_date", "delist_date"])


def test_derives_st_period_from_name():
    nc = _nc([("000001.SZ", "深发展A",  D(2000,1,1), D(2011,12,31)),
              ("000001.SZ", "ST深发展", D(2012,1,1), D(2013,6,30)),
              ("000001.SZ", "平安银行",  D(2013,7,1), None)])
    out = derive_stock_status(nc, _basic([("000001.SZ", "平安银行", D(2000,1,1), None)]))
    st = out[out.status == "ST"]
    assert len(st) == 1
    assert st.iloc[0].start_date == D(2012,1,1) and st.iloc[0].end_date == D(2013,6,30)


def test_star_st_is_distinct_from_st():
    nc = _nc([("000002.SZ", "*ST某某", D(2015,5,1), D(2016,4,30))])
    out = derive_stock_status(nc, _basic([("000002.SZ", "某某", D(2000,1,1), None)]))
    assert set(out.status) == {"*ST"}


def test_second_st_period_is_separate_row():
    """二次戴帽：两段 ST 之间隔着 NORMAL，不能被合并成一段。"""
    nc = _nc([("000003.SZ", "ST甲", D(2012,1,1), D(2013,1,1)),
              ("000003.SZ", "甲",   D(2013,1,2), D(2015,1,1)),
              ("000003.SZ", "ST甲", D(2015,1,2), None)])
    out = derive_stock_status(nc, _basic([("000003.SZ", "ST甲", D(2000,1,1), None)]))
    assert len(out[out.status == "ST"]) == 2


def test_s_prefix_not_treated_as_st():
    """'S' 前缀是未股改，不是 ST。误判会错杀一批 2006-2007 的股票。"""
    nc = _nc([("000004.SZ", "S某某", D(2006,1,1), D(2007,1,1))])
    out = derive_stock_status(nc, _basic([("000004.SZ", "某某", D(2000,1,1), None)]))
    assert "ST" not in set(out.status) and "*ST" not in set(out.status)


def test_delist_period_from_name_suffix():
    nc = _nc([("000005.SZ", "某某退", D(2020,1,1), D(2020,2,1))])
    out = derive_stock_status(nc, _basic([("000005.SZ", "某某退", D(2000,1,1), D(2020,2,2))]))
    assert "DELIST_PERIOD" in set(out.status)


def test_stock_with_no_namechange_gets_normal_row():
    out = derive_stock_status(_nc([]), _basic([("600519.SH", "贵州茅台", D(2001,8,27), None)]))
    assert len(out) == 1 and out.iloc[0].status == "NORMAL"
