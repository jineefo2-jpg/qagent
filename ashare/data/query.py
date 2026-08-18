"""ashare 全系统【唯一】数据出口（D2）。

四条不变量（架构文档 §4.1）：
  Q1 所有 SQL 参数化（? 占位），禁止字符串拼接日期/代码
  Q2 日期入参在入口 _norm_date() → datetime.date，越界抛 AsOfDateError
  Q3 空结果返回带正确列名的空 DataFrame/Series，绝不返回 None
  Q4 本层 raise（QueryError 子类），不返回 {"error": ...}；dict 化只发生在 agent_tools

公开函数首参一律 as_of_date（唯一豁免 get_tradable_mask(exec_date, ...)），
由 scripts/check_ashare_layering.py L2 静态强制。
只用 read_only 连接（D1 最硬一层）：这里没有任何写路径。
"""
from __future__ import annotations
import datetime as _dt
import hashlib
import os
import pathlib
from typing import Sequence, Union

import duckdb
import pandas as pd

from . import _db

DateLike = Union[str, _dt.date]


# ══════════════ 异常 ══════════════
class QueryError(Exception):
    """query 层所有错误的基类。"""


class AsOfDateError(QueryError):
    """as_of 越界 / 超出数据覆盖 / 非法格式。"""


class DataGapError(QueryError):
    """请求区间内数据缺失超过容忍度。"""


class UnknownFieldError(QueryError):
    """请求了不存在的字段。"""


# ══════════════ 连接生命周期 ══════════════
_conn_obj: duckdb.DuckDBPyConnection | None = None
_conn_realpath: str | None = None
_market_path: str | None = None
_PRELOAD: dict[str, pd.DataFrame] = {}
_CAL: pd.DataFrame | None = None            # 日历缓存：trade_date, is_open


def open_db(market_path: str | None = None, derived_path: str | None = None) -> None:
    """惰性建立 read_only 连接。幂等。底层文件 realpath 变化（影子替换发生过）→ 自动重连。"""
    global _conn_obj, _conn_realpath, _market_path, _CAL
    path = market_path or os.environ.get("ASHARE_MARKET_DB") or _db.DEFAULT_MARKET_PATH
    if not pathlib.Path(path).exists():
        raise QueryError(f"market 库不存在: {path}（先跑 python -m ashare.data.ingest）")
    real = os.path.realpath(path)
    if _conn_obj is not None and _conn_realpath == real:
        return
    if _conn_obj is not None:
        _conn_obj.close()
    _conn_obj = _db.connect_read(path)
    _conn_realpath, _market_path = real, path
    _CAL = None
    _PRELOAD.clear()


def close_db() -> None:
    global _conn_obj, _conn_realpath, _CAL
    if _conn_obj is not None:
        _conn_obj.close()
    _conn_obj, _conn_realpath, _CAL = None, None, None
    _PRELOAD.clear()


def _conn() -> duckdb.DuckDBPyConnection:
    if _conn_obj is None:
        open_db(_market_path)
    assert _conn_obj is not None
    return _conn_obj


_SNAPSHOT_TABLES = ("stock_basic", "daily_bar", "daily_basic", "financial_pit",
                    "macro_indicator", "money_flow", "index_daily")


def snapshot_id() -> str:
    """数据快照指纹：sha256(文件名 + schema_version + 各表 max(_ingested_at) + count)[:16]。
    ★ 与 param_hash 一起写进 BacktestResult / docs/oos-runs.md（D7）：参数锁参数，快照锁数据。"""
    c = _conn()
    parts = [pathlib.Path(_conn_realpath or "").name]
    ver = c.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()
    parts.append(str(ver[0] if ver else "?"))
    for t in _SNAPSHOT_TABLES:
        r = c.execute(f"SELECT max(_ingested_at), count(*) FROM {t}").fetchone()   # 表名为本模块字面量
        parts.append(f"{t}:{r[0]}:{r[1]}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ══════════════ 日期规范化（Q2）══════════════
def _norm_date(x: DateLike, *, name: str = "as_of_date") -> _dt.date:
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    if isinstance(x, str):
        s = x.strip().replace("-", "")
        try:
            return _dt.datetime.strptime(s, "%Y%m%d").date()
        except ValueError as exc:
            raise AsOfDateError(f"{name} 非法日期格式: {x!r}（应为 YYYY-MM-DD 或 YYYYMMDD）") from exc
    raise AsOfDateError(f"{name} 类型不支持: {type(x).__name__}")


# ══════════════ 日历 ══════════════
def _calendar() -> pd.DataFrame:
    global _CAL
    if _CAL is None:
        _CAL = _conn().execute("SELECT trade_date, is_open FROM calendar ORDER BY trade_date").fetchdf()
        _CAL["trade_date"] = pd.to_datetime(_CAL["trade_date"]).dt.date
    return _CAL


def _check_in_calendar(d: _dt.date, name: str = "as_of_date") -> None:
    cal = _calendar()
    if cal.empty:
        raise AsOfDateError("日历为空，先 ingest calendar")
    lo, hi = cal["trade_date"].iloc[0], cal["trade_date"].iloc[-1]
    if not (lo <= d <= hi):
        raise AsOfDateError(f"{name}={d} 超出日历覆盖 [{lo}, {hi}]")


def _open_days() -> list[_dt.date]:
    cal = _calendar()
    return list(cal.loc[cal["is_open"], "trade_date"])


def is_trade_date(as_of_date: DateLike) -> bool:
    d = _norm_date(as_of_date)
    _check_in_calendar(d)
    cal = _calendar()
    row = cal[cal["trade_date"] == d]
    return bool(row["is_open"].iloc[0]) if len(row) else False


def prev_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date:
    """严格早于 as_of_date 的第 n 个交易日。"""
    d = _norm_date(as_of_date)
    _check_in_calendar(d)
    days = [x for x in _open_days() if x < d]
    if len(days) < n:
        raise AsOfDateError(f"{d} 之前不足 {n} 个交易日")
    return days[-n]


def next_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date | None:
    """严格晚于 as_of_date 的第 n 个交易日。日历未覆盖到时返回 None（不抛）——回测末端必须处理。"""
    d = _norm_date(as_of_date)
    days = [x for x in _open_days() if x > d]
    return days[n - 1] if len(days) >= n else None


def get_trade_dates(as_of_date: DateLike, *,
                    start: DateLike | None = None,
                    freq: str = "D") -> list[_dt.date]:
    """[start, as_of_date] 闭区间内的交易日。
    freq: 'D' 全部 | 'W' 每周最后一个交易日 | 'M' 每月最后一个交易日。
    'W' 即规格 §5.3 的 weekly_dates 定义 —— 唯一实现点，禁止各处自己算周末。"""
    end = _norm_date(as_of_date)
    _check_in_calendar(end)
    days = [x for x in _open_days() if x <= end]
    if start is not None:
        s = _norm_date(start, name="start")
        days = [x for x in days if x >= s]
    if freq == "D":
        return days
    if freq not in ("W", "M"):
        raise QueryError(f"freq 只能是 D/W/M，收到 {freq!r}")
    key = (lambda x: x.isocalendar()[:2]) if freq == "W" else (lambda x: (x.year, x.month))
    out: list[_dt.date] = []
    for i, x in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else None
        if nxt is None or key(nxt) != key(x):
            out.append(x)
    return out


# ══════════════ preload（骨架；行情类函数在 Task 11 接上）══════════════
_PRELOADABLE = ("daily_bar", "daily_basic", "money_flow")


def preload(start: DateLike, end: DateLike,
            tables: Sequence[str] = ("daily_bar", "daily_basic")) -> None:
    """把区间数据一次性物化进进程内缓存。回测入口调用一次；Live 路径不调用。"""
    s, e = _norm_date(start, name="start"), _norm_date(end, name="end")
    for t in tables:
        if t not in _PRELOADABLE:
            raise QueryError(f"不可 preload 的表: {t}")
        df = _conn().execute(
            f"SELECT * FROM {t} WHERE trade_date BETWEEN ? AND ? ORDER BY ts_code, trade_date",  # 表名为白名单字面量
            [s, e]).fetchdf()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        _PRELOAD[t] = df


def clear_preload() -> None:
    _PRELOAD.clear()


# ══════════════ 股票池与元数据（D5）══════════════
_ST_STATES = ("ST", "*ST", "DELIST_PERIOD")


def _stock_frame(as_of: _dt.date) -> pd.DataFrame:
    """所有股票（含已退市）+ as_of 当日的状态 / 停牌 / 20 日均成交额。一次 SQL，后续在 pandas 里分步。"""
    sql = """
    WITH st AS (
        SELECT ts_code, status FROM stock_status
        WHERE start_date <= ? AND (end_date IS NULL OR ? <= end_date)
    ),
    bar AS (
        SELECT ts_code, is_suspended FROM daily_bar WHERE trade_date = ?
    ),
    liq AS (
        SELECT ts_code, avg(amount) AS adv20 FROM (
            SELECT ts_code, amount,
                   row_number() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn
            FROM daily_bar WHERE trade_date <= ?
        ) WHERE rn <= 20 GROUP BY ts_code
    )
    SELECT b.ts_code, b.market, b.list_date, b.delist_date,
           st.status, bar.is_suspended, liq.adv20
    FROM stock_basic b
    LEFT JOIN st  ON st.ts_code  = b.ts_code
    LEFT JOIN bar ON bar.ts_code = b.ts_code
    LEFT JOIN liq ON liq.ts_code = b.ts_code
    ORDER BY b.ts_code
    """
    df = _conn().execute(sql, [as_of, as_of, as_of, as_of]).fetchdf()
    for c in ("list_date", "delist_date"):
        df[c] = [None if pd.isna(x) else pd.Timestamp(x).date() for x in df[c]]
    return df.set_index("ts_code")


def explain_universe(as_of_date: DateLike, *,
                     min_list_days: int = 250,
                     exclude_st: bool = True,
                     exclude_suspended: bool = True,
                     liquidity_drop_pct: float = 0.20,
                     markets: Sequence[str] | None = None) -> pd.DataFrame:
    """调试与验收用。index=ts_code；逐步布尔列 + included + drop_reason（首个未通过的步骤）。
    ★ 剔除顺序固定：1 退市 → 2 上市满 min_list_days → 3 非 ST → 4 当日有行且非停牌
      → 5 市场过滤 → 6 在 1–5 剩余池内算 20 日均成交额分位，剔后 liquidity_drop_pct（floor）。
    先硬性剔除、后算流动性分位 —— 顺序颠倒会让退市股/次新股参与分位计算，结果不同。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    f = _stock_frame(as_of)
    seasoned_before = as_of - _dt.timedelta(days=min_list_days)

    out = pd.DataFrame(index=f.index)
    out["step1_listed"] = [(ld is not None and ld <= as_of) and (dd is None or dd > as_of)
                           for ld, dd in zip(f["list_date"], f["delist_date"])]
    out["step2_seasoned"] = [(ld is not None and ld <= seasoned_before) for ld in f["list_date"]]
    st_flag = f["status"].isin(_ST_STATES).fillna(False)
    out["step3_not_st"] = (~st_flag) if exclude_st else True
    has_bar = f["is_suspended"].notna()
    susp = f["is_suspended"].fillna(True).astype(bool)
    out["step4_tradable"] = (has_bar & ~susp) if exclude_suspended else has_bar
    out["step5_market"] = f["market"].isin(list(markets)) if markets else True

    hard = out[["step1_listed", "step2_seasoned", "step3_not_st", "step4_tradable", "step5_market"]].all(axis=1)
    pool = f.loc[hard, "adv20"].fillna(0.0)
    n_drop = int(len(pool) * liquidity_drop_pct)
    dropped = set(pool.sort_values(kind="mergesort").index[:n_drop]) if n_drop > 0 else set()
    out["step6_liquid"] = [(c in pool.index) and (c not in dropped) for c in out.index]

    out["included"] = out[["step1_listed", "step2_seasoned", "step3_not_st",
                           "step4_tradable", "step5_market", "step6_liquid"]].all(axis=1)
    reasons = []
    for c in out.index:
        r = out.loc[c]
        if not r.step1_listed:
            dd = f.at[c, "delist_date"]
            reasons.append("delisted" if (dd is not None and dd <= as_of) else "not_listed")
        elif not r.step2_seasoned:
            reasons.append("seasoning")
        elif not r.step3_not_st:
            reasons.append("st")
        elif not r.step4_tradable:
            flag = f.at[c, "is_suspended"]
            reasons.append("no_bar" if pd.isna(flag) else ("suspended" if bool(flag) else "no_bar"))
        elif not r.step5_market:
            reasons.append("market")
        elif not r.step6_liquid:
            reasons.append("illiquid")
        else:
            reasons.append("")
    out["drop_reason"] = reasons
    return out


def get_universe(as_of_date: DateLike, *,
                 min_list_days: int = 250,
                 exclude_st: bool = True,
                 exclude_suspended: bool = True,
                 liquidity_drop_pct: float = 0.20,
                 markets: Sequence[str] | None = None) -> list[str]:
    """as_of_date 当日可交易股票池（ts_code 升序）。规则与顺序见 explain_universe。"""
    ex = explain_universe(as_of_date, min_list_days=min_list_days, exclude_st=exclude_st,
                          exclude_suspended=exclude_suspended, liquidity_drop_pct=liquidity_drop_pct,
                          markets=markets)
    return sorted(ex.index[ex["included"]].tolist())


_BASIC_COLS = ["symbol", "name", "sw_l1", "sw_l2", "sw_l3", "market", "list_date", "delist_date", "is_hs"]


def get_stock_basic(as_of_date: DateLike,
                    ts_codes: Sequence[str] | None = None) -> pd.DataFrame:
    """index=ts_code。sw_* 取 as_of_date 时点的行业（industry_member PIT）；
    name 为当前名称（历史名称仅用于推 ST 状态，已固化进 stock_status，不单独存）。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    sql = """
    SELECT b.ts_code, b.symbol, b.name, m.sw_l1, m.sw_l2, m.sw_l3, b.market,
           b.list_date, b.delist_date, b.is_hs
    FROM stock_basic b
    LEFT JOIN industry_member m
      ON m.ts_code = b.ts_code AND m.in_date <= ? AND (m.out_date IS NULL OR ? <= m.out_date)
    """
    # 区间语义与 stock_status 一致：in_date / out_date 均【含当日】
    params: list = [as_of, as_of]
    if ts_codes is not None:
        codes = list(ts_codes)
        if not codes:
            return pd.DataFrame(columns=_BASIC_COLS).rename_axis("ts_code")
        sql += " WHERE b.ts_code IN (" + ",".join("?" * len(codes)) + ")"
        params += codes
    sql += " ORDER BY b.ts_code"
    df = _conn().execute(sql, params).fetchdf()
    for c in ("list_date", "delist_date"):
        df[c] = [None if pd.isna(x) else pd.Timestamp(x).date() for x in df[c]]
    df = df.drop_duplicates("ts_code", keep="last")     # 行业区间若重叠取最新一段
    return df.set_index("ts_code")[_BASIC_COLS]


def get_industry(as_of_date: DateLike,
                 ts_codes: Sequence[str] | None = None,
                 level: str = "l1",
                 *, min_members: int = 5) -> pd.Series:
    """index=ts_code → 申万行业。★ 成分数 < min_members 的行业统一归入 '__OTHER__'，
    否则中性化 OLS 的行业 dummy 会奇异（架构 B7）。无行业记录的股票也归 '__OTHER__'。"""
    if level not in ("l1", "l2", "l3"):
        raise UnknownFieldError(f"level 只能是 l1/l2/l3，收到 {level!r}")
    b = get_stock_basic(as_of_date, ts_codes)
    s = b[f"sw_{level}"].fillna("__OTHER__").astype(str)
    counts = s.value_counts()
    small = set(counts[counts < min_members].index)
    s = s.where(~s.isin(small), "__OTHER__")
    s.name = f"sw_{level}"
    return s


# ══════════════ 行情（D8：对外只给后复权；原始价不出 query 层）══════════════
_BAR_FIELDS = ("open", "high", "low", "close", "pre_close", "vol", "amount")
_PRICE_FIELDS = ("open", "high", "low", "close", "pre_close")
_DEFAULT_BAR_FIELDS = ("open", "high", "low", "close", "vol", "amount")


def _empty_mi() -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([[], []], names=["ts_code", "trade_date"])


def _bars_raw(as_of: _dt.date, ts_codes: Sequence[str], start: _dt.date | None) -> pd.DataFrame:
    """[start, as_of] 的 daily_bar 原始行（含 adj_factor / is_suspended）。优先命中 preload。"""
    codes = list(ts_codes)
    if not codes:
        return pd.DataFrame(columns=["ts_code", "trade_date", *_BAR_FIELDS, "adj_factor", "is_suspended"])
    cached = _PRELOAD.get("daily_bar")
    if cached is not None:
        lo = cached["trade_date"].min()
        if start is not None and start >= lo and as_of <= cached["trade_date"].max():
            m = cached["ts_code"].isin(codes) & (cached["trade_date"] <= as_of) & (cached["trade_date"] >= start)
            return cached.loc[m, ["ts_code", "trade_date", *_BAR_FIELDS, "adj_factor", "is_suspended"]].copy()
    sql = ("SELECT ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, is_suspended "
           "FROM daily_bar WHERE ts_code IN (" + ",".join("?" * len(codes)) + ") AND trade_date <= ?")
    params: list = [*codes, as_of]
    if start is not None:
        sql += " AND trade_date >= ?"
        params.append(start)
    sql += " ORDER BY ts_code, trade_date"
    df = _conn().execute(sql, params).fetchdf()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _window_start(as_of: _dt.date, lookback: int | None, start: DateLike | None) -> _dt.date | None:
    """lookback 按【交易日历条数】换算成起始日（不是记录数——D9 已保证两者相等，此处仍按日历算）。
    与 start 都给时取交集（更晚者）。"""
    s = _norm_date(start, name="start") if start is not None else None
    if lookback is not None:
        if lookback <= 0:
            raise QueryError("lookback 必须为正整数")
        days = [d for d in _open_days() if d <= as_of]
        lb = days[-lookback] if len(days) >= lookback else days[0]
        s = lb if s is None else max(s, lb)
    return s


def get_bars(as_of_date: DateLike,
             ts_codes: Sequence[str],
             *,
             lookback: int | None = None,
             start: DateLike | None = None,
             fields: Sequence[str] = _DEFAULT_BAR_FIELDS,
             adjust: str = "hfq") -> pd.DataFrame:
    """长表，MultiIndex (ts_code, trade_date)，闭区间上界 = as_of_date。
    adjust: 'hfq'（默认，价格列 × adj_factor）| 'none'（原始价，仅 ingest/validate 可用）。
    ★ 永远额外返回 is_suspended 列。停牌日有行但 OHLC 为 NaN —— 填充与否由因子自己决定。
    ★ 本函数【不返回】limit_up / limit_down：复权价与原始涨跌停价比较是 bug 温床，
      涨跌停信息只能通过 get_tradable_mask 获取。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    bad = [f for f in fields if f not in _BAR_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_bars 不支持字段 {bad}（涨跌停请用 get_tradable_mask）")
    if adjust not in ("hfq", "none"):
        raise QueryError(f"adjust 只能是 hfq/none，收到 {adjust!r}")
    s = _window_start(as_of, lookback, start)
    df = _bars_raw(as_of, ts_codes, s)
    out_cols = [*fields, "is_suspended"]
    if df.empty:
        return pd.DataFrame(columns=out_cols, index=_empty_mi())
    if adjust == "hfq":
        for c in _PRICE_FIELDS:
            df[c] = df[c] * df["adj_factor"]
    sus = df["is_suspended"].astype(bool)
    df.loc[sus, list(_PRICE_FIELDS)] = float("nan")          # 输出层：停牌日价格置 NaN
    df["is_suspended"] = sus
    return df.set_index(["ts_code", "trade_date"])[out_cols]


def get_price_panel(as_of_date: DateLike,
                    ts_codes: Sequence[str],
                    field: str = "close",
                    lookback: int = 250,
                    adjust: str = "hfq") -> pd.DataFrame:
    """宽表：index=trade_date，columns=ts_code。因子计算的主力入口。停牌日为 NaN，不做前向填充。"""
    df = get_bars(as_of_date, ts_codes, lookback=lookback, fields=(field,), adjust=adjust)
    if df.empty:
        return pd.DataFrame(columns=list(ts_codes)).rename_axis("trade_date")
    panel = df[field].unstack("ts_code")
    return panel.reindex(columns=list(ts_codes))


_DAILY_BASIC_FIELDS = ("turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                       "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv")


def get_daily_basic(as_of_date: DateLike,
                    ts_codes: Sequence[str],
                    fields: Sequence[str] = ("pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate_f"),
                    lookback: int = 1) -> pd.DataFrame:
    """lookback=1 → 单日，index=ts_code；lookback>1 → MultiIndex (ts_code, trade_date)。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    bad = [f for f in fields if f not in _DAILY_BASIC_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_daily_basic 不支持字段 {bad}")
    codes = list(ts_codes)
    if not codes:
        if lookback == 1:
            return pd.DataFrame(columns=list(fields)).rename_axis("ts_code")
        return pd.DataFrame(columns=list(fields), index=_empty_mi())
    s = _window_start(as_of, lookback, None)
    cols = ", ".join(fields)                                     # 字段已白名单校验
    df = _conn().execute(
        f"SELECT ts_code, trade_date, {cols} FROM daily_basic WHERE ts_code IN ("
        + ",".join("?" * len(codes)) + ") AND trade_date BETWEEN ? AND ? ORDER BY ts_code, trade_date",
        [*codes, s, as_of]).fetchdf()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if lookback == 1:
        return df.drop(columns=["trade_date"]).set_index("ts_code")[list(fields)]
    return df.set_index(["ts_code", "trade_date"])[list(fields)]


_INDEX_FIELDS = ("open", "high", "low", "close", "vol", "amount", "pe_ttm")


def get_index_bars(as_of_date: DateLike,
                   index_code: str,
                   lookback: int = 250,
                   fields: Sequence[str] = ("close", "pe_ttm")) -> pd.DataFrame:
    """index=trade_date。用于 beta_250 中性化与 P3 宏观 ERP / trend_ma200。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    bad = [f for f in fields if f not in _INDEX_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_index_bars 不支持字段 {bad}")
    s = _window_start(as_of, lookback, None)
    cols = ", ".join(fields)
    df = _conn().execute(
        f"SELECT trade_date, {cols} FROM index_daily WHERE ts_code = ? AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", [index_code, s, as_of]).fetchdf()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.set_index("trade_date")[list(fields)]


# ══════════════ 执行时点专用（D6）—— 唯一首参不是 as_of_date 的函数 ══════════════
_MASK_COLS = ["can_buy", "can_sell", "reason", "open_hfq", "close_hfq", "amount", "amplitude"]


def get_tradable_mask(exec_date: DateLike,
                      ts_codes: Sequence[str]) -> pd.DataFrame:
    """★ 唯一合法的「首参非 as_of_date」函数。语义：回测时钟已推进到 exec_date（T+1），
    此刻读 exec_date 的盘口是「当下」不是「未来」。调用方限定：只有 ashare/backtest/** 可以调。

    index=ts_code；columns：can_buy / can_sell / reason / open_hfq / close_hfq / amount / amplitude。
    判定（全部用【原始价】在函数内部完成，原始价与涨跌停价不外泄）：
      停牌 / 无行情              → 两侧 False（suspended / no_quote）
      limit_up IS NULL           → 两侧 False（limit_unknown）★ 保守：宁可不交易，不可假设可交易
      open==limit_up 且 high==low → can_buy=False（limit_up_seal，一字涨停买不进）
      open==limit_down 且 high==low → can_sell=False（limit_down_seal，一字跌停卖不出）
      delist_date <= exec_date   → can_buy=False, can_sell=True（delisted，强制清仓路径）"""
    d = _norm_date(exec_date, name="exec_date")
    _check_in_calendar(d, name="exec_date")
    if not is_trade_date(d):
        raise AsOfDateError(f"exec_date={d} 不是交易日")
    codes = list(ts_codes)
    if not codes:
        return pd.DataFrame(columns=_MASK_COLS).rename_axis("ts_code")
    rows = _conn().execute(
        "SELECT b.ts_code, b.open, b.high, b.low, b.close, b.pre_close, b.amount, b.adj_factor, "
        "       b.limit_up, b.limit_down, b.is_suspended, s.delist_date "
        "FROM daily_bar b LEFT JOIN stock_basic s ON s.ts_code = b.ts_code "
        "WHERE b.trade_date = ? AND b.ts_code IN (" + ",".join("?" * len(codes)) + ")",
        [d, *codes]).fetchall()
    by_code = {r[0]: r for r in rows}
    out = []
    for code in codes:
        r = by_code.get(code)
        if r is None:
            out.append((code, False, False, "no_quote", float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        _, o, h, l, c, pc, amt, adj, up, dn, sus, delist = r
        delist = None if delist is None else pd.Timestamp(delist).date()
        adj = adj if adj is not None else float("nan")
        o_h = (o * adj) if o is not None else float("nan")
        c_h = (c * adj) if c is not None else float("nan")
        amp = ((h - l) / pc) if (h is not None and l is not None and pc) else float("nan")
        amt = amt if amt is not None else float("nan")

        if delist is not None and delist <= d:
            out.append((code, False, True, "delisted", o_h, c_h, amt, amp)); continue
        if sus or o is None:
            out.append((code, False, False, "suspended", o_h, c_h, amt, amp)); continue
        if up is None or dn is None:
            out.append((code, False, False, "limit_unknown", o_h, c_h, amt, amp)); continue
        one_price = (h == l)
        if one_price and abs(o - up) < 1e-9:
            out.append((code, False, True, "limit_up_seal", o_h, c_h, amt, amp)); continue
        if one_price and abs(o - dn) < 1e-9:
            out.append((code, True, False, "limit_down_seal", o_h, c_h, amt, amp)); continue
        out.append((code, True, True, "", o_h, c_h, amt, amp))
    return pd.DataFrame(out, columns=["ts_code", *_MASK_COLS]).set_index("ts_code")


# ══════════════ 财报 PIT（D3）══════════════
_FIN_FIELDS = ("total_revenue", "revenue", "operate_profit", "total_profit", "n_income", "n_income_attr_p",
               "basic_eps", "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
               "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
               "roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "current_ratio",
               "or_yoy", "netprofit_yoy", "bps")
_FIN_META = ["ann_date", "end_date", "report_type", "lag_days"]


def _fin_visible(as_of: _dt.date, codes: list[str], fields: Sequence[str], *,
                 include_restated: bool, report_type: str) -> pd.DataFrame:
    """PIT 可见的最新披露：WHERE ann_date <= as_of，同 (ts_code, end_date) 取 ann_date 最大者。
    返回列：ts_code, end_date, ann_date, report_type, <fields>。"""
    cols = ", ".join(fields)
    flag_sql = "" if include_restated else " AND update_flag = 0"
    sql = f"""
    SELECT ts_code, end_date, ann_date, report_type, {cols} FROM (
        SELECT *, row_number() OVER (PARTITION BY ts_code, end_date
                                     ORDER BY ann_date DESC, update_flag DESC) AS rn
        FROM financial_pit
        WHERE ann_date <= ? AND report_type = ? {flag_sql}
          AND ts_code IN ({",".join("?" * len(codes))})
    ) WHERE rn = 1
    ORDER BY ts_code, end_date DESC
    """
    df = _conn().execute(sql, [as_of, report_type, *codes]).fetchdf()
    for c in ("end_date", "ann_date"):
        df[c] = pd.to_datetime(df[c]).dt.date
    return df


def get_financial(as_of_date: DateLike,
                  ts_codes: Sequence[str],
                  fields: Sequence[str],
                  *,
                  n_periods: int = 1,
                  include_restated: bool = False,
                  report_type: str = "1") -> pd.DataFrame:
    """PIT 取数：WHERE ann_date <= as_of_date，按 end_date 分组取 ann_date 最大者。
    n_periods=1 → index=ts_code；>1 → MultiIndex (ts_code, end_date) 按 end_date 倒序 n 期。
    include_restated=False（默认）→ 只取 update_flag=0 的原始披露值。
      True 仅供研究「重述影响」，任何进入回测的因子都必须用 False。
    额外返回列：ann_date、end_date、report_type、lag_days(= as_of - ann_date)。
    注：不做交易日历范围检查 —— 财报的覆盖范围是公告日，PIT 过滤自然处理越界（早于最早公告 → 空）。"""
    as_of = _norm_date(as_of_date)
    bad = [f for f in fields if f not in _FIN_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_financial 不支持字段 {bad}")
    if n_periods < 1:
        raise QueryError("n_periods 必须 >= 1")
    codes = list(ts_codes)
    out_cols = [*fields, *_FIN_META]
    if not codes:
        idx = pd.Index([], name="ts_code") if n_periods == 1 else pd.MultiIndex.from_arrays([[], []], names=["ts_code", "end_date"])
        return pd.DataFrame(columns=out_cols, index=idx)
    df = _fin_visible(as_of, codes, fields, include_restated=include_restated, report_type=report_type)
    if df.empty:
        idx = pd.Index([], name="ts_code") if n_periods == 1 else pd.MultiIndex.from_arrays([[], []], names=["ts_code", "end_date"])
        return pd.DataFrame(columns=out_cols, index=idx)
    df = df.groupby("ts_code", sort=True).head(n_periods)
    df["lag_days"] = [(as_of - a).days for a in df["ann_date"]]
    if n_periods == 1:
        return df.set_index("ts_code")[out_cols]
    return df.set_index(["ts_code", "end_date"])[[c for c in out_cols if c != "end_date"]].assign(
        end_date=lambda x: x.index.get_level_values("end_date"))[out_cols]


_TTM_FLOW = ("total_revenue", "revenue", "operate_profit", "total_profit", "n_income", "n_income_attr_p",
             "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act")
_TTM_STOCK = ("total_assets", "total_liab", "total_hldr_eqy_exc_min_int")


def _year_ago(d: _dt.date) -> _dt.date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:                                  # 2 月 29 日
        return d.replace(year=d.year - 1, day=28)


def get_financial_ttm(as_of_date: DateLike,
                      ts_codes: Sequence[str],
                      field: str) -> pd.Series:
    """★ TTM 拼接在 query 层（架构 §4.1 定死）—— A 股财报是【累计口径】。
    流量科目：TTM = 最新累计 + 上年年报 − 上年同期累计（最新为年报时即年报值）；
    存量科目：期初期末均值 =（最新 + 上年同期）/ 2；
    比率科目（roe 等）不支持 TTM → UnknownFieldError。
    任一所需期次不可见（PIT）→ NaN，不外推。返回 index=ts_codes 顺序的 Series。"""
    as_of = _norm_date(as_of_date)
    if field in _TTM_FLOW:
        kind = "flow"
    elif field in _TTM_STOCK:
        kind = "stock"
    else:
        raise UnknownFieldError(f"{field!r} 不支持 TTM（流量: {_TTM_FLOW}；存量: {_TTM_STOCK}）")
    codes = list(ts_codes)
    out = pd.Series([float("nan")] * len(codes), index=pd.Index(codes, name="ts_code"), name=f"{field}_ttm", dtype=float)
    if not codes:
        return out
    vis = _fin_visible(as_of, codes, [field], include_restated=False, report_type="1")
    if vis.empty:
        return out
    for code, g in vis.groupby("ts_code"):
        by_end = dict(zip(g["end_date"], g[field]))
        latest_end = max(by_end)
        latest = by_end[latest_end]
        prev_same = by_end.get(_year_ago(latest_end))
        if kind == "stock":
            out[code] = (latest + prev_same) / 2.0 if (latest is not None and prev_same is not None) else float("nan")
            continue
        if latest_end.month == 12 and latest_end.day == 31:
            out[code] = latest if latest is not None else float("nan")
            continue
        prev_fy = by_end.get(_dt.date(latest_end.year - 1, 12, 31))
        if latest is None or prev_fy is None or prev_same is None:
            out[code] = float("nan")
        else:
            out[code] = latest + prev_fy - prev_same
    return out


# ══════════════ 宏观 PIT（D4）与资金流 ══════════════
def get_macro(as_of_date: DateLike,
              indicators: Sequence[str],
              lookback_periods: int = 60) -> pd.DataFrame:
    """WHERE publish_date <= as_of_date；同 (indicator, period) 取 publish_date 最大者。
    index=period，columns=indicator，附加列 <indicator>__publish_date 便于审计。"""
    as_of = _norm_date(as_of_date)
    inds = list(indicators)
    cols: list[str] = []
    for i in inds:
        cols += [i, f"{i}__publish_date"]
    if not inds:
        return pd.DataFrame(columns=cols).rename_axis("period")
    df = _conn().execute(f"""
        SELECT indicator, period, publish_date, value FROM (
            SELECT *, row_number() OVER (PARTITION BY indicator, period ORDER BY publish_date DESC) AS rn
            FROM macro_indicator
            WHERE publish_date <= ? AND indicator IN ({",".join("?" * len(inds))})
        ) WHERE rn = 1 ORDER BY indicator, period
        """, [as_of, *inds]).fetchdf()
    if df.empty:
        return pd.DataFrame(columns=cols).rename_axis("period")
    df["period"] = pd.to_datetime(df["period"]).dt.date
    df["publish_date"] = pd.to_datetime(df["publish_date"]).dt.date
    val = df.pivot(index="period", columns="indicator", values="value")
    pub = df.pivot(index="period", columns="indicator", values="publish_date")
    out = pd.DataFrame(index=val.index)
    for i in inds:
        out[i] = val[i] if i in val.columns else float("nan")
        out[f"{i}__publish_date"] = pub[i] if i in pub.columns else None
    out = out.sort_index().tail(lookback_periods)
    out.index.name = "period"
    return out[cols]


_MONEY_FLOW_FIELDS = ("hk_hold_ratio",)


def get_money_flow(as_of_date: DateLike,
                   ts_codes: Sequence[str],
                   fields: Sequence[str] = ("hk_hold_ratio",),
                   lookback: int = 20) -> pd.DataFrame:
    """MultiIndex (ts_code, trade_date)，按交易日历补齐 lookback 个交易日。
    ★ hk_hold_ratio 仅 2016-12 起有数据；早于该日 / 缺失日返回 NaN，不填 0（B5）——
      调用方（north_hold_chg_20 因子）靠 FactorSpec.available_from 声明。"""
    as_of = _norm_date(as_of_date)
    _check_in_calendar(as_of)
    bad = [f for f in fields if f not in _MONEY_FLOW_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_money_flow 不支持字段 {bad}")
    codes = list(ts_codes)
    days = [d for d in _open_days() if d <= as_of][-lookback:]
    idx = pd.MultiIndex.from_product([codes, days], names=["ts_code", "trade_date"])
    if not codes or not days:
        return pd.DataFrame(columns=list(fields), index=_empty_mi())
    cols = ", ".join(fields)
    df = _conn().execute(
        f"SELECT ts_code, trade_date, {cols} FROM money_flow WHERE ts_code IN ("
        + ",".join("?" * len(codes)) + ") AND trade_date BETWEEN ? AND ?",
        [*codes, days[0], as_of]).fetchdf()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.set_index(["ts_code", "trade_date"]).reindex(idx)[list(fields)]
