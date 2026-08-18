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
