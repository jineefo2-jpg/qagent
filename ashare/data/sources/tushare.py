# ashare/data/sources/tushare.py
"""Tushare Pro adapter。职责只有两件：限频 + 把 Tushare 的 YYYYMMDD 字符串日期
规范成 datetime.date。不做任何业务转换 —— 那是 ingest.normalize 的事。"""
from __future__ import annotations
import json, os, pathlib
from typing import Any

import pandas as pd

from ._ratelimit import TokenBucket

try:
    import tushare as _ts
except ImportError:                     # 可选依赖，遵循本仓库既有模式
    _ts = None

_DATE_COLS = ("trade_date", "ann_date", "end_date", "list_date", "delist_date",
              "start_date", "f_ann_date", "cal_date", "pretrade_date", "date")
# 注：cn_m 的 month 列是 YYYYMM，不在此列表，由 ingest 的宏观 normalize 单独处理


def _to_date(df: pd.DataFrame) -> pd.DataFrame:
    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce").dt.date
    return df


def _fmt(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


class TushareSource:
    def __init__(self, token: str | None = None, state_path: str = "data/rate_state.json") -> None:
        if _ts is None:
            raise ImportError("需要 tushare：pip install tushare")
        tok = token or os.environ.get("TUSHARE_TOKEN")
        if not tok:
            raise ValueError("未提供 TUSHARE_TOKEN")
        self._pro = _ts.pro_api(tok)
        cpm = 120
        try:
            cpm = int(json.loads(pathlib.Path(state_path).read_text()).get("calls_per_min", 120))
        except Exception:
            pass
        self._bucket = TokenBucket(cpm, state_path)

    def _call(self, api: str, **kw) -> pd.DataFrame:
        self._bucket.acquire()
        df = getattr(self._pro, api)(**{k: v for k, v in kw.items() if v is not None})
        return _to_date(df if df is not None else pd.DataFrame())

    # ── 基础 ──
    def trade_cal(self, start, end, exchange: str = "SSE") -> pd.DataFrame:
        return self._call("trade_cal", exchange=exchange,
                          start_date=_fmt(start), end_date=_fmt(end))

    def stock_basic(self) -> pd.DataFrame:
        """★ 必须三种 list_status 都拉：L 在市 / D 已退市 / P 暂停上市。
        只拉 L 就是幸存者偏差的源头（D5）。"""
        fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs"
        parts = [self._call("stock_basic", exchange="", list_status=s, fields=fields)
                 for s in ("L", "D", "P")]
        return pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")

    def namechange(self, ts_code: str | None = None) -> pd.DataFrame:
        return self._call("namechange", ts_code=ts_code,
                          fields="ts_code,name,start_date,end_date,change_reason")

    # ── 行情 ──
    def daily(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("daily", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def adj_factor(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("adj_factor", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def stk_limit(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("stk_limit", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def daily_basic(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("daily_basic", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def index_daily(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("index_daily", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    # ── 财报（PIT 的原料）──
    def fina_indicator(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("fina_indicator", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def income(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("income", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def balancesheet(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("balancesheet", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def cashflow(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("cashflow", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    # ── 资金流 / 宏观 ──
    def hk_hold(self, trade_date) -> pd.DataFrame:
        return self._call("hk_hold", trade_date=_fmt(trade_date))

    def cn_m(self, start_m: str, end_m: str) -> pd.DataFrame:
        return self._call("cn_m", start_m=start_m, end_m=end_m)

    def shibor(self, start=None, end=None) -> pd.DataFrame:
        return self._call("shibor", start_date=_fmt(start), end_date=_fmt(end))
