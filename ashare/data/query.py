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
