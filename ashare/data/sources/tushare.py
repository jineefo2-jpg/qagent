"""Tushare Pro adapter。职责只有两件：限频 + 把 Tushare 的 YYYYMMDD 字符串日期
规范成 datetime.date。不做任何业务转换 —— 那是 ingest.normalize 的事。"""
from __future__ import annotations
import json, os, pathlib, time
from typing import Any

import pandas as pd

from ._ratelimit import TokenBucket

try:
    import tushare as _ts
except ImportError:                     # 可选依赖，遵循本仓库既有模式
    _ts = None

_DATE_COLS = ("trade_date", "ann_date", "end_date", "list_date", "delist_date",
              "start_date", "f_ann_date", "cal_date", "pretrade_date", "date",
              "in_date", "out_date")
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

    MAX_RETRIES = 3

    def _call(self, api: str, **kw) -> pd.DataFrame:
        """限频 + 重试：分钟限频文案 → 等 61s 重试；其他（网络）错误指数退避 2/4/8s；
        无权限（"没有…权限" / "积分不足"）立刻抛，不重试。"""
        params = {k: v for k, v in kw.items() if v is not None}
        last: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            self._bucket.acquire()
            try:
                df = getattr(self._pro, api)(**params)
                return _to_date(df if df is not None else pd.DataFrame())
            except Exception as exc:                  # noqa: BLE001
                msg = str(exc)
                if ("没有" in msg and "权限" in msg) or "积分不足" in msg or attempt == self.MAX_RETRIES:
                    raise
                last = exc
                time.sleep(61 if "每分钟" in msg else 2 ** (attempt + 1))
        raise last  # pragma: no cover

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

    def index_dailybasic(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        """指数估值（pe_ttm 等），供 P3 的 ERP。"""
        return self._call("index_dailybasic", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    # ── 行业 ──
    def sw_members(self) -> pd.DataFrame:
        """申万成分历史 → 列 [ts_code, sw_l1, sw_l2, sw_l3, in_date, out_date]。
        优先 index_member_all（一次全量，含 L1/L2/L3 与进出日期）；权限不足时把异常原样抛出，
        由 ingest 决定是否降级——这里不吞。"""
        # ★ 必须分页：单次上限 3000 行，而全量约 5560 行 —— 不翻页会静默少掉 46%，
        #   于是近一半股票没有行业归属、在 get_industry 里落进 __OTHER__ 桶。
        #   后果不是报错而是【行业中性化失真】：一大桶票被当成同一个行业做横截面回归，
        #   行业暴露剥不干净（2026-08-31 实测：2015 年 72% 的票落在 __OTHER__）。
        #   而且每次只取"某 3000 行"，取到哪些不保证稳定 —— 参考表 diff 因此天天误报。
        pages, off = [], 0
        while True:
            d = self._call("index_member_all", is_new=None, limit=5000, offset=off)
            if d is None or not len(d):
                break
            pages.append(d)
            off += len(d)
            if off > 100000:                      # 防呆：真实量级 5-6 千行
                break
        df = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=["ts_code", "sw_l1", "sw_l2", "sw_l3", "in_date", "out_date"])
        rename = {"con_code": "ts_code", "l1_name": "sw_l1", "l2_name": "sw_l2", "l3_name": "sw_l3"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = ["ts_code", "sw_l1", "sw_l2", "sw_l3", "in_date", "out_date"]
        for c in keep:
            if c not in df.columns:
                df[c] = None
        return _to_date(df[keep])

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

    def cn_cpi(self, start_m: str, end_m: str) -> pd.DataFrame:
        return self._call("cn_cpi", start_m=start_m, end_m=end_m)

    def cn_ppi(self, start_m: str, end_m: str) -> pd.DataFrame:
        return self._call("cn_ppi", start_m=start_m, end_m=end_m)

    def cn_pmi(self, start_m: str, end_m: str) -> pd.DataFrame:
        # cn_pmi 是唯一一个输出字段全部「默认显示 N」的宏观接口（官方文档 doc_id=325）：
        # 不显式点名 fields 时服务端一列都不给，裸调返回的空列 DataFrame 会让下游
        # 取 month 列直接 KeyError（2026-08-25 全量回补实测）。cn_m/cn_cpi/cn_ppi/sf_month
        # 的核心列都是默认 Y，不需要也不要跟风加。
        return self._call("cn_pmi", start_m=start_m, end_m=end_m, fields="month,pmi010000")

    def sf_month(self, start_m: str, end_m: str) -> pd.DataFrame:
        return self._call("sf_month", start_m=start_m, end_m=end_m)

    def shibor(self, start=None, end=None) -> pd.DataFrame:
        return self._call("shibor", start_date=_fmt(start), end_date=_fmt(end))

    def cn10y(self, start=None, end=None) -> pd.DataFrame:
        """中国 10 年期国债收益率 → [period, value]。Tushare 无稳定接口，走 akshare（可选依赖）。
        不经令牌桶（不是 Tushare 配额）。"""
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("cn10y 需要 akshare：pip install akshare") from exc
        df = ak.bond_zh_us_rate(start_date=_fmt(start) or "20100101")
        col = "中国国债收益率10年"            # 精确匹配：还有一列 "中国国债收益率10年-2年"（利差），子串匹配会撞
        if col not in df.columns:
            raise KeyError(f"akshare bond_zh_us_rate 缺少列 {col!r}，实际列: {list(df.columns)}")
        out = pd.DataFrame({"period": pd.to_datetime(df["日期"]).dt.date,
                            "value": pd.to_numeric(df[col], errors="coerce")}).dropna()
        if end is not None:
            out = out[out["period"] <= pd.Timestamp(_fmt(end)).date()]
        return out.reset_index(drop=True)
