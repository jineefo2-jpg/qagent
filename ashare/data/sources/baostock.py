"""BaoStock adapter —— 只用于 validate.check_cross_source 的双源交叉校验（后复权收盘价）。
不参与任何 ingest 写入路径。可选依赖：pip install baostock。"""
from __future__ import annotations
import datetime as _dt

import pandas as pd

try:
    import baostock as _bs
except ImportError:                     # 可选依赖，遵循本仓库既有模式
    _bs = None


def _bs_code(ts_code: str) -> str:
    """600519.SH → sh.600519；000001.SZ → sz.000001。北交所 BaoStock 不覆盖 → 原样返回（查无结果）。"""
    code, _, ex = ts_code.partition(".")
    return f"{ex.lower()}.{code}"


class BaoStockSource:
    def __init__(self) -> None:
        if _bs is None:
            raise ImportError("需要 baostock：pip install baostock")
        lg = _bs.login()
        if lg.error_code != "0":
            raise ConnectionError(f"baostock login failed: {lg.error_msg}")

    def close(self) -> None:
        try:
            _bs.logout()
        except Exception:               # noqa: BLE001 — logout 失败不影响校验结论
            pass

    def hfq_close(self, ts_code: str, start: _dt.date, end: _dt.date) -> pd.DataFrame:
        """→ DataFrame[trade_date(date), close_hfq(float)]。adjustflag='1' = 后复权。"""
        rs = _bs.query_history_k_data_plus(
            _bs_code(ts_code), "date,close",
            start_date=start.isoformat(), end_date=end.isoformat(),
            frequency="d", adjustflag="1")
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=["trade_date", "close_hfq"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["close_hfq"] = pd.to_numeric(df["close_hfq"], errors="coerce")
        return df.dropna()
