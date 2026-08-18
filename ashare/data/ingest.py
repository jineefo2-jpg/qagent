# ashare/data/ingest.py
"""market 库的唯一写者。

状态机（ingest_log）：PENDING → RUNNING → DONE | RETRY | SUSPECT | FAILED
  RETRY   网络/限频类错误，可重试
  SUSPECT 拉到了数据但校验存疑（行数异常），保留数据待人工确认
  FAILED  schema 断言失败（字段缺失/改名），不重试 —— 重试只会一直错
"""
from __future__ import annotations
import datetime as _dt
import re
from typing import Any

import pandas as pd

from . import _db

# ST 判定：'ST'/'*ST' 前缀。★ 'S' 单独前缀是未股改，不是 ST。
_RE_STAR_ST = re.compile(r"^\*ST")
_RE_ST = re.compile(r"^S?ST")          # 'SST' 也是 ST（未股改 + ST）
_RE_DELIST = re.compile(r"退$")


def _classify(name: str) -> str:
    n = (name or "").replace(" ", "")
    if _RE_DELIST.search(n):
        return "DELIST_PERIOD"
    if _RE_STAR_ST.match(n):
        return "*ST"
    if _RE_ST.match(n):
        return "ST"
    return "NORMAL"


def derive_stock_status(namechange_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """由历史名称变更反推 ST 状态区间（架构师 B6 —— Tushare 无此接口）。

    边界（必须遵守，否则 D5 的股票池就是错的）：
      - 用变更【生效日】start_date，不是公告日
      - 'S' 前缀 = 未股改，不是 ST
      - 名称含 '退' = 退市整理期，单独归类
      - 无 namechange 记录的股票，补一条覆盖全生命周期的 NORMAL
    """
    rows: list[dict[str, Any]] = []

    if len(namechange_df):
        nc = namechange_df.sort_values(["ts_code", "start_date"])
        for ts_code, grp in nc.groupby("ts_code", sort=True):
            for _, r in grp.iterrows():
                rows.append({"ts_code": ts_code,
                             "start_date": r["start_date"],
                             "end_date": r.get("end_date"),
                             "status": _classify(r["name"])})

    covered = {r["ts_code"] for r in rows}
    for _, b in basic_df.iterrows():
        if b["ts_code"] not in covered:
            rows.append({"ts_code": b["ts_code"],
                         "start_date": b["list_date"],
                         "end_date": b.get("delist_date"),
                         "status": _classify(b.get("name", ""))})

    out = pd.DataFrame(rows, columns=["ts_code", "start_date", "end_date", "status"])
    return out.sort_values(["ts_code", "start_date"]).reset_index(drop=True)


# ══════════════ ingest_log 状态机 ══════════════
def set_job(conn, job_id: str, table: str, partition: str, state: str,
            *, rows: int = 0, error: str = "") -> None:
    conn.execute(
        """INSERT INTO ingest_log (job_id, table_name, partition, state, attempts,
                                   rows_written, last_error, started_at, finished_at)
           VALUES (?, ?, ?, ?, 1, ?, ?, current_timestamp,
                   CASE WHEN ? IN ('DONE','FAILED') THEN current_timestamp END)
           ON CONFLICT (job_id) DO UPDATE SET
             state = excluded.state,
             attempts = ingest_log.attempts + 1,
             rows_written = excluded.rows_written,
             last_error = excluded.last_error,
             finished_at = excluded.finished_at""",
        [job_id, table, partition, state, rows, error, state])


def job_state(conn, job_id: str) -> str | None:
    r = conn.execute("SELECT state FROM ingest_log WHERE job_id = ?", [job_id]).fetchone()
    return r[0] if r else None


def _upsert(conn, table: str, df: pd.DataFrame, pk: list[str]) -> int:
    """按主键覆盖写入。DuckDB 的 INSERT OR REPLACE 走主键冲突。"""
    if df is None or df.empty:
        return 0
    cols = [c for c in df.columns if c != "_ingested_at"]
    conn.register("_stage", df[cols])
    conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) SELECT {','.join(cols)} FROM _stage")
    conn.unregister("_stage")
    return len(df)


# ══════════════ 各表 ingest ══════════════
def ingest_calendar(conn, src, start, end) -> int:
    df = src.trade_cal(start, end)
    df = df.rename(columns={"cal_date": "trade_date", "pretrade_date": "pre_trade_date"})
    df["is_open"] = df["is_open"].astype(int).astype(bool)
    n = _upsert(conn, "calendar", df[["trade_date", "is_open", "pre_trade_date"]], ["trade_date"])
    set_job(conn, "calendar:all", "calendar", "all", "DONE", rows=n)
    return n


def ingest_stock_basic(conn, src) -> int:
    df = src.stock_basic()
    for c in ("sw_l1", "sw_l2", "sw_l3"):
        if c not in df.columns:
            df[c] = None
    cols = ["ts_code", "symbol", "name", "sw_l1", "sw_l2", "sw_l3",
            "market", "list_date", "delist_date", "is_hs"]
    n = _upsert(conn, "stock_basic", df[cols], ["ts_code"])
    set_job(conn, "stock_basic:all", "stock_basic", "all", "DONE", rows=n)
    return n


def ingest_stock_status(conn, src) -> int:
    basic = conn.execute(
        "SELECT ts_code, name, list_date, delist_date FROM stock_basic").fetchdf()
    nc = src.namechange(ts_code=None)
    status = derive_stock_status(nc, basic)
    n = _upsert(conn, "stock_status", status, ["ts_code", "start_date"])
    set_job(conn, "stock_status:all", "stock_status", "all", "DONE", rows=n)
    return n


# ══════════════ 日线：按交易日历补齐停牌占位行（D9 / 架构师 B3）══════════════
from .limits import compute_limits

_DAILY_BAR_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                   "vol", "amount", "adj_factor", "limit_up", "limit_down",
                   "limit_source", "is_suspended"]


def _status_at(status_rows: list[dict], d: _dt.date) -> str:
    for r in status_rows:
        start = r["start_date"]
        end = r.get("end_date")
        if start is not None and start <= d and (end is None or d <= end):
            return r["status"]
    return "NORMAL"


def normalize_daily_bar(daily: pd.DataFrame,
                        adj: pd.DataFrame,
                        limit: pd.DataFrame | None,
                        calendar_dates: list[_dt.date],
                        basic_row: dict,
                        status_rows: list[dict]) -> pd.DataFrame:
    """把 Tushare 三张表合成 daily_bar，并【按交易日历补齐停牌占位行】（D9 / 架构师 B3）。

    ★ 这是全套设计里最隐蔽的一个坑：Tushare `daily` 在停牌日不返回该股的行。
      不补行的话，get_bars(lookback=20) 拿到的是「最近 20 条记录」而不是
      「最近 20 个交易日」—— 一只停牌 5 天的股票，它的 reversal_20 实际覆盖
      25 个交易日。横截面因子被静默污染，且完全不报错。
    """
    ts_code = basic_row["ts_code"]
    list_date = basic_row.get("list_date")
    delist_date = basic_row.get("delist_date")

    # 1. 只保留在市区间内的交易日
    dates = [d for d in sorted(calendar_dates)
             if (list_date is None or d >= list_date)
             and (delist_date is None or d <= delist_date)]
    if not dates:
        return pd.DataFrame(columns=_DAILY_BAR_COLS)

    # 2. 以完整交易日历为骨架左连接
    frame = pd.DataFrame({"trade_date": dates})
    d = daily.copy() if daily is not None and len(daily) else pd.DataFrame(
        columns=["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"])
    frame = frame.merge(d.drop(columns=["ts_code"], errors="ignore"), on="trade_date", how="left")

    a = adj.copy() if adj is not None and len(adj) else pd.DataFrame(
        columns=["trade_date", "adj_factor"])
    frame = frame.merge(a.drop(columns=["ts_code"], errors="ignore"), on="trade_date", how="left")

    frame = frame.sort_values("trade_date").reset_index(drop=True)

    # 3. 标记停牌（daily 无行 = 停牌）并填占位值
    frame["is_suspended"] = frame["close"].isna()
    frame["adj_factor"] = frame["adj_factor"].ffill().bfill()

    prev_close = None
    for i in frame.index:
        if frame.at[i, "is_suspended"]:
            fill = prev_close
            for c in ("open", "high", "low", "close", "pre_close"):
                frame.at[i, c] = fill
            frame.at[i, "vol"] = 0.0
            frame.at[i, "amount"] = 0.0
        prev_close = frame.at[i, "close"]

    # 4. 涨跌停：API 优先，缺失走规则兜底（B2）
    lim_map: dict[_dt.date, tuple[float, float]] = {}
    if limit is not None and len(limit):
        for _, r in limit.iterrows():
            lim_map[r["trade_date"]] = (r.get("up_limit"), r.get("down_limit"))

    ups, downs, srcs = [], [], []
    for _, r in frame.iterrows():
        td = r["trade_date"]
        if td in lim_map and lim_map[td][0] is not None:
            ups.append(lim_map[td][0]); downs.append(lim_map[td][1]); srcs.append("api")
            continue
        u, dn, src = compute_limits(ts_code, td, r.get("pre_close"),
                                    list_date, _status_at(status_rows, td))
        ups.append(u); downs.append(dn); srcs.append(src)

    frame["ts_code"] = ts_code
    frame["limit_up"], frame["limit_down"], frame["limit_source"] = ups, downs, srcs
    return frame[_DAILY_BAR_COLS]


def ingest_daily_bar(conn, src, ts_code: str, start, end) -> int:
    job = f"daily_bar:{ts_code}:{start}"
    if job_state(conn, job) == "DONE":
        return 0
    set_job(conn, job, "daily_bar", ts_code, "RUNNING")
    try:
        daily = src.daily(ts_code=ts_code, start=start, end=end)
        adj = src.adj_factor(ts_code=ts_code, start=start, end=end)
        try:
            limit = src.stk_limit(ts_code=ts_code, start=start, end=end)
        except Exception:
            limit = None                     # 无权限 → 走规则兜底，不是错误

        cal = [r[0] for r in conn.execute(
            "SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date", [start, end]).fetchall()]
        basic = conn.execute(
            "SELECT ts_code, list_date, delist_date FROM stock_basic WHERE ts_code = ?",
            [ts_code]).fetchdf().to_dict("records")[0]
        status = conn.execute(
            "SELECT start_date, end_date, status FROM stock_status WHERE ts_code = ? "
            "ORDER BY start_date", [ts_code]).fetchdf().to_dict("records")

        out = normalize_daily_bar(daily, adj, limit, cal, basic, status)
        n = _upsert(conn, "daily_bar", out, ["ts_code", "trade_date"])
        set_job(conn, job, "daily_bar", ts_code, "DONE", rows=n)
        return n
    except Exception as exc:                 # noqa: BLE001
        set_job(conn, job, "daily_bar", ts_code, "RETRY", error=str(exc)[:500])
        raise
