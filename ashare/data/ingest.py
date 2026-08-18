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

from .limits import compute_limits

# ST 判定（全部按【前缀】匹配，空格已去除）：
#   *ST / S*ST      → *ST      （'S' = 未股改，可叠加在 *ST 前）
#   ST / SST        → ST
#   S 单独前缀      → 未股改，不是 ST，落 NORMAL
#   退市xxx（上交所前缀）/ xxx退（深交所后缀） → DELIST_PERIOD
_RE_STAR_ST = re.compile(r"^S?\*ST")
_RE_ST = re.compile(r"^S?ST")
_RE_DELIST = re.compile(r"^退市|退$")


def _classify(name: str) -> str:
    n = (name or "").replace(" ", "")
    if _RE_DELIST.search(n):
        return "DELIST_PERIOD"
    if _RE_STAR_ST.match(n):
        return "*ST"
    if _RE_ST.match(n):
        return "ST"
    return "NORMAL"


def _as_date(v: Any) -> _dt.date | None:
    """把 pandas Timestamp / numpy datetime64 / date / None / NaT 统一成 datetime.date 或 None。
    DuckDB fetchdf() 给的是 Timestamp，Tushare adapter 给的是 date——两者混进同一列会让 sort 抛 TypeError。"""
    if v is None or pd.isna(v):            # None / NaN / NaT（NaT 是 datetime 子类，必须先于 isinstance 判）
        return None
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    ts = pd.Timestamp(v)
    return None if pd.isna(ts) else ts.date()


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
        for _, r in namechange_df.iterrows():
            rows.append({"ts_code": r["ts_code"],
                         "start_date": _as_date(r["start_date"]),
                         "end_date": _as_date(r.get("end_date")),
                         "status": _classify(r["name"])})

    covered = {r["ts_code"] for r in rows}
    for _, b in basic_df.iterrows():
        if b["ts_code"] not in covered:
            rows.append({"ts_code": b["ts_code"],
                         "start_date": _as_date(b["list_date"]),
                         "end_date": _as_date(b.get("delist_date")),
                         "status": _classify(b.get("name", ""))})

    out = pd.DataFrame(rows, columns=["ts_code", "start_date", "end_date", "status"])
    # 统一成 datetime.date 后再排序：Timestamp 与 date 混排会 TypeError
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


_TABLES_WITH_INGESTED_AT = {"stock_basic", "daily_bar", "daily_basic", "financial_pit",
                            "macro_indicator", "money_flow", "index_daily"}


def _upsert(conn, table: str, df: pd.DataFrame) -> int:
    """按主键覆盖写入（DuckDB INSERT OR REPLACE 走主键冲突）。
    table / 列名只来自本模块的字面量，绝不接外部输入（f-string 拼 SQL 的前提）。
    带 _ingested_at 的表每次写入都刷新为 current_timestamp —— 语义是"最后写入时间"，
    query.snapshot_id() 依赖它感知数据变化。"""
    if df is None or df.empty:
        return 0
    cols = [c for c in df.columns if c != "_ingested_at"]
    ins_cols, sel_cols = list(cols), list(cols)
    if table in _TABLES_WITH_INGESTED_AT:
        ins_cols.append("_ingested_at")
        sel_cols.append("current_timestamp")
    conn.register("_stage", df[cols])
    try:
        conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(ins_cols)}) "
                     f"SELECT {','.join(sel_cols)} FROM _stage")
    finally:
        conn.unregister("_stage")
    return len(df)


# ══════════════ 各表 ingest ══════════════
def ingest_calendar(conn, src, start, end) -> int:
    df = src.trade_cal(start, end)
    df = df.rename(columns={"cal_date": "trade_date", "pretrade_date": "pre_trade_date"})
    df["is_open"] = df["is_open"].astype(int).astype(bool)
    n = _upsert(conn, "calendar", df[["trade_date", "is_open", "pre_trade_date"]])
    set_job(conn, "calendar:all", "calendar", "all", "DONE", rows=n)
    return n


def ingest_stock_basic(conn, src) -> int:
    df = src.stock_basic()
    for c in ("sw_l1", "sw_l2", "sw_l3"):
        if c not in df.columns:
            df[c] = None
    cols = ["ts_code", "symbol", "name", "sw_l1", "sw_l2", "sw_l3",
            "market", "list_date", "delist_date", "is_hs"]
    n = _upsert(conn, "stock_basic", df[cols])
    set_job(conn, "stock_basic:all", "stock_basic", "all", "DONE", rows=n)
    return n


def ingest_stock_status(conn, src) -> int:
    basic = conn.execute(
        "SELECT ts_code, name, list_date, delist_date FROM stock_basic").fetchdf()
    nc = src.namechange(ts_code=None)
    status = derive_stock_status(nc, basic)
    n = _upsert(conn, "stock_status", status)
    set_job(conn, "stock_status:all", "stock_status", "all", "DONE", rows=n)
    return n


# ══════════════ 日线：按交易日历补齐停牌占位行（D9 / 架构师 B3）══════════════
_DAILY_BAR_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                   "vol", "amount", "adj_factor", "limit_up", "limit_down",
                   "limit_source", "is_suspended"]
_PRICE_COLS = ("open", "high", "low", "close", "pre_close")


def _status_at(status_rows: list[dict], d: _dt.date) -> str:
    for r in status_rows:
        start = r["start_date"]
        end = r.get("end_date")
        if start is not None and start <= d and (end is None or d <= end):
            return r["status"]
    return "NORMAL"


def _none_if_nan(x):
    return None if x is None or pd.isna(x) else x


def normalize_daily_bar(daily: pd.DataFrame,
                        adj: pd.DataFrame,
                        limit: pd.DataFrame | None,
                        calendar_dates: list[_dt.date],
                        basic_row: dict,
                        status_rows: list[dict],
                        *,
                        seed_close: float | None = None,
                        seed_adj: float | None = None) -> pd.DataFrame:
    """把 Tushare 三张表合成 daily_bar，并【按交易日历补齐停牌占位行】（D9 / 架构师 B3）。

    ★ 这是全套设计里最隐蔽的一个坑：Tushare `daily` 在停牌日不返回该股的行。
      不补行的话，get_bars(lookback=20) 拿到的是「最近 20 条记录」而不是
      「最近 20 个交易日」—— 一只停牌 5 天的股票，它的 reversal_20 实际覆盖
      25 个交易日。横截面因子被静默污染，且完全不报错。

    seed_close / seed_adj：本批次之前最后一个非停牌日的 close / adj_factor。
      分批（按年）拉取时批次首日若停牌，没有种子就只能写 NaN——那不是错，但会
      让"上一批已知有前收"的股票丢信息；有种子就前推。永远不用 bfill（那是看未来）。
    """
    ts_code = basic_row["ts_code"]
    list_date = _as_date(basic_row.get("list_date"))
    delist_date = _as_date(basic_row.get("delist_date"))
    # fetchdf() 给 Timestamp、adapter 给 date：入口统一，后面全部按 datetime.date 比较
    status_rows = [{**r, "start_date": _as_date(r.get("start_date")),
                    "end_date": _as_date(r.get("end_date"))} for r in status_rows]
    calendar_dates = [_as_date(d) for d in calendar_dates]

    # 1. 只保留在市区间内的交易日
    dates = [d for d in sorted(calendar_dates)
             if (list_date is None or d >= list_date)
             and (delist_date is None or d <= delist_date)]
    if not dates:
        return pd.DataFrame(columns=_DAILY_BAR_COLS)

    # 2. 以完整交易日历为骨架左连接（源表重复日期取最后一条：Tushare 偶发重复行会让行数虚增）
    frame = pd.DataFrame({"trade_date": dates})
    d = daily.copy() if daily is not None and len(daily) else pd.DataFrame(
        columns=["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"])
    d = d.drop(columns=["ts_code"], errors="ignore").drop_duplicates("trade_date", keep="last")
    frame = frame.merge(d, on="trade_date", how="left")

    a = adj.copy() if adj is not None and len(adj) else pd.DataFrame(
        columns=["trade_date", "adj_factor"])
    a = a.drop(columns=["ts_code"], errors="ignore").drop_duplicates("trade_date", keep="last")
    frame = frame.merge(a, on="trade_date", how="left")

    frame = frame.sort_values("trade_date").reset_index(drop=True)
    for c in _PRICE_COLS + ("vol", "amount", "adj_factor"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")       # 统一 float，避免 object 列 ffill 的 FutureWarning

    # 3. 标记停牌（daily 无行 = 停牌）并填占位值；adj_factor 只前推，不后推
    frame["is_suspended"] = frame["close"].isna()
    if seed_adj is not None and pd.isna(frame.at[0, "adj_factor"]):
        frame.at[0, "adj_factor"] = seed_adj
    frame["adj_factor"] = frame["adj_factor"].ffill()

    prev_close = seed_close
    for i in frame.index:
        if frame.at[i, "is_suspended"]:
            for c in _PRICE_COLS:
                frame.at[i, c] = prev_close if prev_close is not None else float("nan")
            frame.at[i, "vol"] = 0.0
            frame.at[i, "amount"] = 0.0
        prev_close = _none_if_nan(frame.at[i, "close"])

    # 4. 涨跌停：API 优先（NaN 视同缺失），缺失走规则兜底（B2）
    lim_map: dict[_dt.date, tuple[float, float]] = {}
    if limit is not None and len(limit):
        for _, r in limit.iterrows():
            up, dn = _none_if_nan(r.get("up_limit")), _none_if_nan(r.get("down_limit"))
            if up is not None and dn is not None:
                lim_map[_as_date(r["trade_date"])] = (up, dn)

    ups, downs, srcs = [], [], []
    for _, r in frame.iterrows():
        td = r["trade_date"]
        if td in lim_map:
            ups.append(lim_map[td][0]); downs.append(lim_map[td][1]); srcs.append("api")
            continue
        u, dn, src = compute_limits(ts_code, td, _none_if_nan(r.get("pre_close")),
                                    list_date, _status_at(status_rows, td))
        ups.append(u); downs.append(dn); srcs.append(src)

    frame["ts_code"] = ts_code
    frame["limit_up"], frame["limit_down"], frame["limit_source"] = ups, downs, srcs
    return frame[_DAILY_BAR_COLS]


def _fetch_limit(src, ts_code: str, start, end) -> pd.DataFrame | None:
    """stk_limit 需 Tushare 积分权限。
    无权限 → 记在 src 上并返回 None（走规则兜底，且后续股票不再浪费调用）；
    其他错误（限频 / 网络）正常抛出，让任务进 RETRY —— 吞掉会把临时故障当成永久无权限。"""
    if getattr(src, "_stk_limit_denied", False):
        return None
    try:
        return src.stk_limit(ts_code=ts_code, start=start, end=end)
    except Exception as exc:                 # noqa: BLE001 — 只吃权限类，其余重新抛
        msg = str(exc)
        if "权限" in msg or "积分" in msg:
            src._stk_limit_denied = True
            return None
        raise


def _seed_before(conn, ts_code: str, start) -> tuple[float | None, float | None]:
    """本批次之前最后一个非停牌交易日的 (close, adj_factor)，供跨批次前推。"""
    row = conn.execute(
        "SELECT close, adj_factor FROM daily_bar WHERE ts_code = ? AND trade_date < ? "
        "AND NOT is_suspended ORDER BY trade_date DESC LIMIT 1", [ts_code, start]).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _parse_date(x) -> _dt.date:
    """'YYYYMMDD' / 'YYYY-MM-DD' / date / Timestamp → datetime.date。SQL 参数必须是真日期，不能是字符串。"""
    if isinstance(x, str):
        return _dt.datetime.strptime(x.replace("-", ""), "%Y%m%d").date()
    d = _as_date(x)
    if d is None:
        raise ValueError(f"无法解析日期: {x!r}")
    return d


def ingest_daily_bar(conn, src, ts_code: str, start, end) -> int:
    start_d, end_d = _parse_date(start), _parse_date(end)
    job = f"daily_bar:{ts_code}:{start_d.isoformat()}"
    if job_state(conn, job) == "DONE":
        return 0
    set_job(conn, job, "daily_bar", ts_code, "RUNNING")
    try:
        daily = src.daily(ts_code=ts_code, start=start_d, end=end_d)
        adj = src.adj_factor(ts_code=ts_code, start=start_d, end=end_d)
        limit = _fetch_limit(src, ts_code, start_d, end_d)

        cal = [r[0] for r in conn.execute(
            "SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date", [start_d, end_d]).fetchall()]
        # fetchone/fetchall 直接给 datetime.date / None（fetchdf 会给 Timestamp / NaT）
        b = conn.execute("SELECT ts_code, list_date, delist_date FROM stock_basic WHERE ts_code = ?",
                         [ts_code]).fetchone()
        if b is None:
            raise KeyError(f"stock_basic 中无 {ts_code}，先跑 ingest_stock_basic")
        basic = {"ts_code": b[0], "list_date": b[1], "delist_date": b[2]}
        status = [{"start_date": r[0], "end_date": r[1], "status": r[2]} for r in conn.execute(
            "SELECT start_date, end_date, status FROM stock_status WHERE ts_code = ? "
            "ORDER BY start_date", [ts_code]).fetchall()]
        seed_close, seed_adj = _seed_before(conn, ts_code, start_d)

        out = normalize_daily_bar(daily, adj, limit, cal, basic, status,
                                  seed_close=seed_close, seed_adj=seed_adj)
        n = _upsert(conn, "daily_bar", out)
        set_job(conn, job, "daily_bar", ts_code, "DONE", rows=n)
        return n
    except Exception as exc:                 # noqa: BLE001
        set_job(conn, job, "daily_bar", ts_code, "RETRY", error=str(exc)[:500])
        raise
