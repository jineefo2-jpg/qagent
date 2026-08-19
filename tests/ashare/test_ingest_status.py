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


def test_sh_delist_prefix_is_delist_period():
    """上交所退市整理期用『退市』前缀（退市海润/退市长油），不是深交所的『退』后缀。
    漏掉它们会让退市整理期股票直接进股票池。"""
    nc = _nc([("600401.SH", "退市海润", D(2019,6,1), D(2019,7,1))])
    out = derive_stock_status(nc, _basic([("600401.SH", "退市海润", D(2000,1,1), D(2019,7,2))]))
    assert set(out.status) == {"DELIST_PERIOD"}


def test_s_star_st_is_star_st():
    """2006–2007 未股改 + *ST：『S*ST兰宝』既不是 NORMAL 也不是普通 ST。"""
    nc = _nc([("000631.SZ", "S*ST兰宝", D(2006,5,1), D(2007,5,1))])
    out = derive_stock_status(nc, _basic([("000631.SZ", "兰宝", D(2000,1,1), None)]))
    assert set(out.status) == {"*ST"}


def test_mixed_timestamp_and_date_inputs_do_not_crash():
    """真实调用路径：basic 来自 DuckDB fetchdf()（pandas Timestamp），namechange 来自 adapter（date）。
    两者混进 start_date 一列后 sort_values 曾抛 TypeError。"""
    nc = _nc([("000001.SZ", "ST深发展", D(2012,1,1), D(2013,6,30))])
    basic = pd.DataFrame([("000001.SZ", "平安银行", pd.Timestamp("2000-01-01"), pd.NaT),
                          ("600519.SH", "贵州茅台", pd.Timestamp("2001-08-27"), pd.NaT)],
                         columns=["ts_code", "name", "list_date", "delist_date"])
    out = derive_stock_status(nc, basic)
    assert len(out) == 2
    assert all(isinstance(x, dt.date) for x in out.start_date), "输出必须统一为 datetime.date"
    assert out[out.ts_code == "600519.SH"].iloc[0].end_date is None, "NaT 必须归一化为 None"


def test_status_at_takes_latest_segment_on_overlap():
    """写入端与 query 端必须同向取最新段。反向 → 写入按 NORMAL 算 10% 涨跌停、读取按 *ST 判 5%，
    带宽差 5%，一字板检测不出来 → 回测在锁死的日子成交。"""
    from ashare.data.ingest import _status_at
    rows = [{"start_date": D(2010,1,1), "end_date": None, "status": "NORMAL"},      # 残留的全生命周期行
            {"start_date": D(2024,1,10), "end_date": D(2024,1,19), "status": "*ST"}]
    assert _status_at(rows, D(2024,1,15)) == "*ST"
    assert _status_at(rows, D(2024,1,5)) == "NORMAL"
