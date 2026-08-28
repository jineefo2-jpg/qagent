"""ashare 全系统【唯一】数据出口（D2）。

四条不变量（架构文档 §4.1）：
  Q1 所有 SQL 参数化（? 占位），禁止字符串拼接日期/代码
  Q2 日期入参在入口 norm_date() → datetime.date，越界抛 AsOfDateError
  Q3 空结果返回带正确列名的空 DataFrame/Series，绝不返回 None
  Q4 本层 raise（QueryError 子类），不返回 {"error": ...}；dict 化只发生在 agent_tools

公开函数首参一律 as_of_date（唯一豁免 get_tradable_mask(exec_date, ...)），
由 scripts/check_ashare_layering.py L2 静态强制。
只用 read_only 连接（D1 最硬一层）：这里没有任何写路径。
"""
from __future__ import annotations
import bisect
import datetime as _dt
import hashlib
import os
import pathlib
from typing import Sequence, Union

import duckdb
import numpy as np
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
_conn_ident: tuple[int, int] | None = None   # (st_dev, st_ino)：识别影子替换（os.replace 不改路径但换 inode）
_conn_realpath: str | None = None
_market_path: str | None = None
_PRELOAD: dict[str, pd.DataFrame] = {}
_PRELOAD_POS: dict[str, dict] = {}          # 表 → 定位索引，见 _build_preload_pos


def _build_preload_pos(df: pd.DataFrame) -> dict:
    """预加载表的定位索引。表按 (ts_code, trade_date) 排序 → 每只票一段连续区间：
    {code: (lo, hi)} + datetime64 日期数组，切片退化为每票两次二分 + 一次 take。
    替掉的三次全表 object 扫描（isin + 两次日期比较，1400 万行 × 每期数次）
    在 2026-08-27 的性能 profile 里占引擎 46% 耗时。"""
    if not len(df):
        return {"groups": {}, "dates": None, "min": None, "max": None}
    codes = df["ts_code"].to_numpy()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    groups = {codes[lo]: (int(lo), int(hi)) for lo, hi in zip(starts, ends)}
    return {"groups": groups,
            "dates": pd.to_datetime(df["trade_date"]).to_numpy(),
            "min": df["trade_date"].min(), "max": df["trade_date"].max()}


def _preload_slice(table: str, codes: Sequence[str], start, end) -> "pd.DataFrame | None":
    """[start, end] × codes 的缓存切片；未预加载或窗口越出缓存返回 None（调用方走 SQL）。
    输出行序 = (ts_code, trade_date) 升序 —— 与旧全表布尔掩码路径逐位相同。"""
    cached, pos = _PRELOAD.get(table), _PRELOAD_POS.get(table)
    if cached is None or pos is None or pos["min"] is None:
        return None
    if start is None or start < pos["min"] or end > pos["max"]:
        return None
    s64 = pd.Timestamp(start).to_datetime64()
    e64 = pd.Timestamp(end).to_datetime64()
    take: list = []
    for c in sorted(set(codes)):
        g = pos["groups"].get(c)
        if g is None:
            continue
        d = pos["dates"][g[0]:g[1]]
        i0 = g[0] + int(np.searchsorted(d, s64, "left"))
        i1 = g[0] + int(np.searchsorted(d, e64, "right"))
        if i1 > i0:
            take.append(np.arange(i0, i1))
    if not take:
        return cached.iloc[0:0]
    return cached.iloc[np.concatenate(take)]
_CAL: pd.DataFrame | None = None            # 日历缓存：trade_date, is_open
_OPEN_DAYS: list[_dt.date] | None = None    # 开市日列表缓存（回测循环里高频调用）
_pinned_ident: tuple[int, int] | None = None  # snapshot_id(pin=True) 钉住的 inode：之后换文件要抛而不是静默重连


def _file_ident(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return (st.st_dev, st.st_ino)


def open_db(market_path: str | None = None, derived_path: str | None = None) -> None:
    """惰性建立 read_only 连接。幂等。
    底层文件被【影子替换】（promote 用 os.replace：路径不变、inode 变）→ 自动重连；
    旧连接读的是已 unlink 的旧 inode，不重连会永远看旧数据。
    derived_path：P2 因子/回测结果库，本期未 ATTACH（预留参数）。"""
    global _conn_obj, _conn_ident, _conn_realpath, _market_path, _CAL, _OPEN_DAYS
    path = market_path or _market_path or os.environ.get("ASHARE_MARKET_DB") or _db.DEFAULT_MARKET_PATH
    if not pathlib.Path(path).exists():
        raise QueryError(f"market 库不存在: {path}（先跑 python -m ashare.data.pipeline full）")
    ident = _file_ident(path)
    # ★ 钉住期间不许换库（2026-08-21）：本函数会把 _conn_ident 指到新 inode，
    #   于是 _conn() 的「磁盘 ident != _conn_ident」比较又相等了，钉住检查【永远不再触发】——
    #   查询静默落到另一个数据库上，而 _pinned_ident 还宣称钉着。
    #   那正是 D7 要挡的「一次运行横跨两份数据只记一个 data_snapshot_id」，
    #   且比直接抛更坏：抛出来的错是特性，静默成功不是。要换库先 close_db()。
    if _pinned_ident is not None and ident != _pinned_ident:
        raise QueryError(
            f"当前快照已钉住（snapshot_id(pin=True)），不能在运行途中 open_db 到另一份数据库。"
            f"钉住只由 close_db() 解除。路径: {path}")
    if _conn_obj is not None and _conn_ident == ident:
        return
    if _conn_obj is not None:
        _conn_obj.close()
    _conn_obj = _db.connect_read(path)
    _conn_ident, _conn_realpath, _market_path = ident, os.path.realpath(path), path
    _CAL, _OPEN_DAYS = None, None
    _PRELOAD.clear()
    _PRELOAD_POS.clear()
    _PRELOAD_WIDE.clear()


def close_db() -> None:
    """解钉 + 断连。★ 有意【保留】`_market_path`：close 的既定契约是「解钉后查询在同一
    路径上自愈续走」（test_build_leaves_the_snapshot_pinned_until_close_db 钉着这一条，
    长驻进程 build → close → 查询自动重连新 inode 靠它）。代价：换库必须显式
    `open_db(新路径)` —— 测试里混用临时库时无参 open_db() 会回到上一个路径。"""
    global _conn_obj, _conn_ident, _conn_realpath, _CAL, _OPEN_DAYS, _pinned_ident
    if _conn_obj is not None:
        _conn_obj.close()
    _conn_obj, _conn_ident, _conn_realpath, _CAL, _OPEN_DAYS = None, None, None, None, None
    _pinned_ident = None
    _PRELOAD.clear()
    _PRELOAD_POS.clear()
    _PRELOAD_WIDE.clear()


def _conn() -> duckdb.DuckDBPyConnection:
    """每次取连接都重新 stat 一次路径（微秒级）：inode 变了就重连。
    但如果调用方已 snapshot_id(pin=True) 钉住快照，换文件必须【抛】——
    静默重连会让一次回测横跨两个数据库而只记录一个 data_snapshot_id。"""
    if _conn_obj is not None and _market_path and pathlib.Path(_market_path).exists():
        ident = _file_ident(_market_path)
        if ident != _conn_ident:
            if _pinned_ident is not None:
                raise QueryError(
                    f"数据库在运行途中被替换（promote），而当前快照已钉住。"
                    f"本次运行的结果横跨两份数据，不可信 —— 请重跑。路径: {_market_path}")
            open_db(_market_path)
    elif _conn_obj is None:
        open_db(_market_path)
    assert _conn_obj is not None
    return _conn_obj


_SNAPSHOT_TABLES = ("stock_basic", "daily_bar", "daily_basic", "financial_pit",
                    "macro_indicator", "money_flow", "index_daily")


def _compute_snapshot(conn) -> str:
    """在【给定连接】上算数据指纹。抽出来是为了让 promote 用它自己的写连接算，
    不必在同一进程里混用只读/读写连接（DuckDB 不允许）。"""
    parts: list[str] = []
    ver = conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()
    parts.append(str(ver[0] if ver else "?"))
    for t in _SNAPSHOT_TABLES:
        r = conn.execute(f"SELECT max(_ingested_at), count(*) FROM {t}").fetchone()   # 表名为本模块字面量
        parts.append(f"{t}:{r[0]}:{r[1]}")
    # 无 _ingested_at 的三张小表用【内容哈希】：count/min/max 抓不到原地修改（改一个 is_open / 改一个行业名），
    # 而那同样改变回测结果（D7）。bit_xor(hash(...)) 与行序无关且不溢出。
    for t, cols in (("stock_status", "ts_code, start_date, end_date, status"),
                    ("industry_member", "ts_code, sw_l1, sw_l2, sw_l3, in_date, out_date"),
                    ("calendar", "trade_date, is_open")):
        r = conn.execute(f"SELECT count(*), bit_xor(hash({cols})) FROM {t}").fetchone()
        parts.append(f"{t}:{r[0]}:{r[1]}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def snapshot_id(*, pin: bool = False) -> str:
    """数据快照指纹：sha256(schema_version + 各表 max(_ingested_at)+count + 三张小表内容哈希)[:16]。

    ★ 只函数化于【数据】，不含文件名/路径 —— 否则把 .bak 快照挂到别的路径重跑，
      指纹就对不上自己记录的那次运行，而 D7 要的正是"同 param_hash + 同 data_snapshot_id ⇒ 同结果"。
    ★ pin=True：钉住当前 inode。之后 promote 换了文件，_conn() 不再静默重连而是抛 QueryError ——
      否则一次回测可能横跨两个数据库、却只记录一个指纹（D7 失效）。close_db() 解钉。
      回测入口应 snapshot_id(pin=True)，结束前再取一次核对。"""
    global _pinned_ident
    c = _conn()
    if pin:
        _pinned_ident = _conn_ident
    return _compute_snapshot(c)


# ══════════════ 日期规范化（Q2）══════════════
def norm_date(x: DateLike, *, name: str = "as_of_date") -> _dt.date:
    """把 str / datetime / date 归一成 date。**公开**（2026-08-21）：
    data 层之外也要用它（`factors/store.py` 的日期列表），而私有名走 L6。
    不返回任何行情数据，故不受 L2「首参 as_of_date」约束 —— 已列进白名单。"""
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
    global _OPEN_DAYS
    if _OPEN_DAYS is None:
        cal = _calendar()
        _OPEN_DAYS = list(cal.loc[cal["is_open"], "trade_date"])
    return _OPEN_DAYS


def is_trade_date(as_of_date: DateLike) -> bool:
    d = norm_date(as_of_date)
    _check_in_calendar(d)
    cal = _calendar()
    row = cal[cal["trade_date"] == d]
    return bool(row["is_open"].iloc[0]) if len(row) else False


def prev_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date:
    """严格早于 as_of_date 的第 n 个交易日。"""
    d = norm_date(as_of_date)
    _check_in_calendar(d)
    days = _open_days()
    i = bisect.bisect_left(days, d)          # days 已排序；O(log n) 而非每次扫 4000 个日期
    if i < n:
        raise AsOfDateError(f"{d} 之前不足 {n} 个交易日")
    return days[i - n]


def next_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date | None:
    """严格晚于 as_of_date 的第 n 个交易日。日历未覆盖到时返回 None（不抛）——回测末端必须处理。"""
    d = norm_date(as_of_date)
    days = _open_days()
    i = bisect.bisect_right(days, d)
    return days[i + n - 1] if i + n - 1 < len(days) else None


def get_trade_dates(as_of_date: DateLike, *,
                    start: DateLike | None = None,
                    freq: str = "D") -> list[_dt.date]:
    """[start, as_of_date] 闭区间内的交易日。
    freq: 'D' 全部 | 'W' 每周最后一个交易日 | 'M' 每月最后一个交易日。
    'W' 即规格 §5.3 的 weekly_dates 定义 —— 唯一实现点，禁止各处自己算周末。"""
    end = norm_date(as_of_date)
    _check_in_calendar(end)
    all_days = _open_days()
    hi = bisect.bisect_right(all_days, end)
    lo = bisect.bisect_left(all_days, norm_date(start, name="start")) if start is not None else 0
    days = all_days[lo:hi]
    if freq == "D":
        return days
    if freq not in ("W", "M"):
        raise QueryError(f"freq 只能是 D/W/M，收到 {freq!r}")
    key = (lambda x: x.isocalendar()[:2]) if freq == "W" else (lambda x: (x.year, x.month))
    # 末元素是否算周期末：看日历里【下一个交易日】（可能晚于 as_of，日历是公开信息，不构成前视）
    # 是否仍在同一周期。周三 as_of 时本周还没结束 → 周三不是调仓日。日历尽头无法判断 → 算作周期末。
    tail_next = next_trade_date(end)
    out: list[_dt.date] = []
    for i, x in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else tail_next
        if nxt is None or key(nxt) != key(x):
            out.append(x)
    return out


# ══════════════ preload（骨架；行情类函数在 Task 11 接上）══════════════
_PRELOADABLE = ("daily_bar", "daily_basic")     # get_money_flow 不读缓存，列上去只会白占内存


def preload(start: DateLike, end: DateLike,
            tables: Sequence[str] = ("daily_bar", "daily_basic")) -> None:
    """把区间数据一次性物化进进程内缓存。回测入口调用一次；Live 路径不调用。"""
    s, e = norm_date(start, name="start"), norm_date(end, name="end")
    for t in tables:
        if t not in _PRELOADABLE:
            raise QueryError(f"不可 preload 的表: {t}")
        df = _conn().execute(
            f"SELECT * FROM {t} WHERE trade_date BETWEEN ? AND ? ORDER BY ts_code, trade_date",  # 表名为白名单字面量
            [s, e]).fetchdf()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        _PRELOAD[t] = df
        _PRELOAD_POS[t] = _build_preload_pos(df)
    # ★ 宽面板缓存必须跟着换窗失效（`_PRELOAD_WIDE` 的注释说的「随 _PRELOAD 同清」——
    #   之前只有 open_db / close_db / clear_preload 三处真的清，preload 换窗这一路漏了）。
    #   同进程连跑两次回测（`gate1_out_of_sample` 的 IS→OOS 正是）时，第二次会拿着
    #   第一次那个窗的面板：新窗日期整段缺失、新窗才上市的票整列不存在 → 价格静默变 NaN。
    _PRELOAD_WIDE.clear()


def last_data_date() -> "_dt.date | None":
    """行情覆盖的最后一天。**日历比它长**：`ingest_calendar` 每次预拉 today+90 天，
    于是 `next_trade_date(最后一根 K 线那天)` 会给出一个没有任何行情的「未来交易日」。
    回测拿它当执行日 → 整个持仓 `no_quote` → `simulate` 抛。库空返回 None。"""
    r = _conn().execute("SELECT max(trade_date) FROM daily_bar").fetchone()
    return None if r is None or r[0] is None else pd.Timestamp(r[0]).date()


def clear_preload() -> None:
    _PRELOAD.clear()
    _PRELOAD_POS.clear()
    _PRELOAD_WIDE.clear()


# ══════════════ 股票池与元数据（D5）══════════════
_ST_STATES = ("ST", "*ST", "DELIST_PERIOD")


def _stock_frame(as_of: _dt.date) -> pd.DataFrame:
    """所有股票（含已退市）+ as_of 当日的状态 / 停牌 / 20 日均成交额。一次 SQL，后续在 pandas 里分步。"""
    all_days = _open_days()
    days = all_days[:bisect.bisect_right(all_days, as_of)]
    win_start = days[-20] if len(days) >= 20 else (days[0] if days else as_of)
    sql = """
    WITH st AS (
        -- 区间重叠（ingest 原样拷贝 namechange，schema 允许）时取 start_date 最新的一段，避免一股多行
        SELECT ts_code, status FROM stock_status
        WHERE start_date <= ? AND (end_date IS NULL OR ? <= end_date)
        QUALIFY row_number() OVER (PARTITION BY ts_code ORDER BY start_date DESC) = 1
    ),
    bar AS (
        SELECT ts_code, is_suspended FROM daily_bar WHERE trade_date = ?
    ),
    liq AS (
        -- 20 日均成交额：D9 保证每交易日一行，[win_start, as_of] 恰为 20 个交易日；
        -- 停牌占位行 amount=0 计入均值（近期停牌的股票被压低——这是有意的）
        SELECT ts_code, avg(amount) AS adv20 FROM daily_bar
        WHERE trade_date BETWEEN ? AND ? GROUP BY ts_code
    )
    SELECT b.ts_code, b.market, b.list_date, b.delist_date,
           st.status, bar.is_suspended, liq.adv20
    FROM stock_basic b
    LEFT JOIN st  ON st.ts_code  = b.ts_code
    LEFT JOIN bar ON bar.ts_code = b.ts_code
    LEFT JOIN liq ON liq.ts_code = b.ts_code
    ORDER BY b.ts_code
    """
    df = _conn().execute(sql, [as_of, as_of, as_of, win_start, as_of]).fetchdf()
    for c in ("list_date", "delist_date"):
        # 向量化（2026-08-27 性能专项）：逐元素 Timestamp 循环在 5800 行 × 每期多调下
        # 占全窗 ~12s；.dt.date + where 输出逐位相同（date 对象 / None）。
        conv = pd.Series(pd.to_datetime(df[c]).dt.date, index=df.index, dtype=object)
        conv[df[c].isna()] = None                    # where(..., None) 会把 None 变回 NaT
        df[c] = conv
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
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if not is_trade_date(as_of):
        # 非交易日当天没有 daily_bar 行 → step4 全 False → 静默返回空池。
        # 与 get_tradable_mask 一致：宁可抛，不给一个看起来合理的空结果。
        raise AsOfDateError(f"as_of_date={as_of} 不是交易日；股票池只在交易日有定义")
    f = _stock_frame(as_of)
    seasoned_before = as_of - _dt.timedelta(days=min_list_days)

    out = pd.DataFrame(index=f.index)
    out["step1_listed"] = [(ld is not None and ld <= as_of) and (dd is None or dd > as_of)
                           for ld, dd in zip(f["list_date"], f["delist_date"])]
    out["step2_seasoned"] = [(ld is not None and ld <= seasoned_before) for ld in f["list_date"]]
    st_flag = f["status"].isin(_ST_STATES)
    out["step3_not_st"] = (~st_flag) if exclude_st else True
    has_bar = f["is_suspended"].notna()
    susp = f["is_suspended"].fillna(True).astype(bool)
    out["step4_tradable"] = (has_bar & ~susp) if exclude_suspended else has_bar
    out["step5_market"] = f["market"].isin(list(markets)) if markets is not None else True

    hard = out[["step1_listed", "step2_seasoned", "step3_not_st", "step4_tradable", "step5_market"]].all(axis=1)
    pool = f.loc[hard, "adv20"].fillna(0.0)
    n_drop = int(len(pool) * liquidity_drop_pct)
    dropped = set(pool.sort_values(kind="mergesort").index[:n_drop]) if n_drop > 0 else set()
    out["step6_liquid"] = [(c in pool.index) and (c not in dropped) for c in out.index]

    out["included"] = out[["step1_listed", "step2_seasoned", "step3_not_st",
                           "step4_tradable", "step5_market", "step6_liquid"]].all(axis=1)
    import numpy as np
    delisted = pd.Series([(dd is not None and dd <= as_of) for dd in f["delist_date"]], index=f.index)
    conds = [~out["step1_listed"] & delisted, ~out["step1_listed"],
             ~out["step2_seasoned"], ~out["step3_not_st"],
             ~out["step4_tradable"] & has_bar & susp, ~out["step4_tradable"],
             ~out["step5_market"], ~out["step6_liquid"]]
    labels = ["delisted", "not_listed", "seasoning", "st", "suspended", "no_bar", "market", "illiquid"]
    out["drop_reason"] = np.select([c.to_numpy() for c in conds], labels, default="")
    return out


_UNIVERSE_MEMO: dict = {}       # (连接 inode, 日期, 全参数) → list。inode 换（promote/重连）即整体失效


def get_universe(as_of_date: DateLike, *,
                 min_list_days: int = 250,
                 exclude_st: bool = True,
                 exclude_suspended: bool = True,
                 liquidity_drop_pct: float = 0.20,
                 markets: Sequence[str] | None = None) -> list[str]:
    """as_of_date 当日可交易股票池（ts_code 升序）。规则与顺序见 explain_universe。

    备忘录：引擎一个调仓日会经不同路径要同一个池子 3–4 次（主循环 / read_current 的
    池子核对 / 诊断），2026-08-27 性能 profile 里占 23%。键含连接 inode ——
    promote / close / 换库任何一种重连都会换 inode，缓存随之整体失效，PIT 语义不变。"""
    _conn()   # 钉住/换库检查不许被备忘录绕过（钉着且文件被换必须抛）；顺带把 _conn_ident 刷新到位
    key = (_conn_ident, norm_date(as_of_date), min_list_days, exclude_st, exclude_suspended,
           liquidity_drop_pct, tuple(markets) if markets is not None else None)
    hit = _UNIVERSE_MEMO.get(key)
    if hit is not None:
        return list(hit)
    ex = explain_universe(as_of_date, min_list_days=min_list_days, exclude_st=exclude_st,
                          exclude_suspended=exclude_suspended, liquidity_drop_pct=liquidity_drop_pct,
                          markets=markets)
    out = sorted(ex.index[ex["included"]].tolist())
    if len(_UNIVERSE_MEMO) >= 1024:                    # 回测全窗 511 期 × 1 组参数，1024 绰绰有余
        _UNIVERSE_MEMO.clear()
    _UNIVERSE_MEMO[key] = out
    return list(out)


_BASIC_COLS = ["symbol", "name", "sw_l1", "sw_l2", "sw_l3", "market", "list_date", "delist_date", "is_hs"]


def get_stock_basic(as_of_date: DateLike,
                    ts_codes: Sequence[str] | None = None) -> pd.DataFrame:
    """index=ts_code。sw_* 取 as_of_date 时点的行业（industry_member PIT）；
    name 为当前名称（历史名称仅用于推 ST 状态，已固化进 stock_status，不单独存）。"""
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    sql = """
    WITH m AS (
        -- 区间语义与 stock_status 一致：in_date / out_date 均【含当日】；重叠时取 in_date 最新一段
        SELECT ts_code, sw_l1, sw_l2, sw_l3 FROM industry_member
        WHERE in_date <= ? AND (out_date IS NULL OR ? <= out_date)
        QUALIFY row_number() OVER (PARTITION BY ts_code ORDER BY in_date DESC) = 1
    )
    SELECT b.ts_code, b.symbol, b.name, m.sw_l1, m.sw_l2, m.sw_l3, b.market,
           b.list_date, b.delist_date, b.is_hs
    FROM stock_basic b LEFT JOIN m ON m.ts_code = b.ts_code
    """
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
        # 向量化（2026-08-27 性能专项）：逐元素 Timestamp 循环在 5800 行 × 每期多调下
        # 占全窗 ~12s；.dt.date + where 输出逐位相同（date 对象 / None）。
        conv = pd.Series(pd.to_datetime(df[c]).dt.date, index=df.index, dtype=object)
        conv[df[c].isna()] = None                    # where(..., None) 会把 None 变回 NaT
        df[c] = conv
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


def industry_source() -> str | None:
    """`_meta.industry_source`：`'sw'` = 真实申万成分历史；其余（如 `'tushare_static'`）
    = 今天的行业回填到上市日（P1 在申万成分接口无权限时的降级）。无记录返回 None。

    ★ 行业中性化前【必须】查这个：在降级标签上做行业中性化就是把今天的行业分类
      用到了历史横截面上，属于前视污染。建库时的 `--allow-static-industry` 只放行建库，
      不放行中性化 —— 所以这里返回原始值，由调用方（factors/pipeline.py）拒绝。
    无位置参数，不是取数函数，L2 首参规则不适用。"""
    r = _conn().execute("SELECT value FROM _meta WHERE key='industry_source'").fetchone()
    return r[0] if r else None


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
    sl = _preload_slice("daily_bar", codes, start, as_of)
    if sl is not None:
        return sl[["ts_code", "trade_date", *_BAR_FIELDS, "adj_factor", "is_suspended"]].copy()
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
    s = norm_date(start, name="start") if start is not None else None
    if lookback is not None:
        if lookback <= 0:
            raise QueryError("lookback 必须为正整数")
        all_days = _open_days()
        days = all_days[:bisect.bisect_right(all_days, as_of)]
        if not days:
            raise AsOfDateError(f"as_of_date={as_of} 之前日历中没有交易日")
        lb = days[-lookback] if len(days) >= lookback else days[0]      # 不足 lookback 时取全部（数据起点）
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
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if not fields:
        raise UnknownFieldError("fields 不能为空")
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


_PRELOAD_WIDE: dict = {}    # ("daily_bar", field, adjust) → 全窗宽面板，随 _PRELOAD 同清


def _wide_panel(field: str, adjust: str) -> "pd.DataFrame | None":
    """daily_bar 预加载的懒构建宽面板 —— 与 get_bars 同一口径（hfq = ×adj_factor、
    停牌价置 NaN），一次 pivot（秒级、~185MB/字段）换掉引擎逐期的 unstack
    （2026-08-27 性能专项：价格面板路径占每期 ~35ms）。"""
    key = ("daily_bar", field, adjust)
    w = _PRELOAD_WIDE.get(key)
    if w is None and _PRELOAD.get("daily_bar") is not None:
        cached = _PRELOAD["daily_bar"]
        vals = cached[field].astype(float)
        if field in _PRICE_FIELDS:
            if adjust == "hfq":
                vals = vals * cached["adj_factor"].astype(float)
            vals = vals.where(~cached["is_suspended"].astype(bool))
        w = pd.DataFrame({"trade_date": cached["trade_date"],
                          "ts_code": cached["ts_code"], "v": vals}) \
              .pivot(index="trade_date", columns="ts_code", values="v")
        _PRELOAD_WIDE[key] = w
    return w


def get_price_panel(as_of_date: DateLike,
                    ts_codes: Sequence[str],
                    field: str = "close",
                    lookback: int = 250,
                    adjust: str = "hfq") -> pd.DataFrame:
    """宽表：index=trade_date，columns=ts_code。因子计算的主力入口。停牌日为 NaN，不做前向填充。"""
    codes = list(ts_codes)
    pos = _PRELOAD_POS.get("daily_bar")
    if codes and pos is not None and pos["min"] is not None \
            and field in _BAR_FIELDS and adjust in ("hfq", "none"):
        as_of = norm_date(as_of_date)
        _check_in_calendar(as_of)
        s = _window_start(as_of, lookback, None)
        if s is not None and s >= pos["min"] and as_of <= pos["max"]:
            w = _wide_panel(field, adjust)
            if w is not None:
                return w.loc[s:as_of].reindex(columns=codes)
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
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if not fields:
        raise UnknownFieldError("fields 不能为空")
    bad = [f for f in fields if f not in _DAILY_BASIC_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_daily_basic 不支持字段 {bad}")
    codes = list(ts_codes)
    if not codes:
        if lookback == 1:
            return pd.DataFrame(columns=list(fields)).rename_axis("ts_code")
        return pd.DataFrame(columns=list(fields), index=_empty_mi())
    s = _window_start(as_of, lookback, None)
    sl = _preload_slice("daily_basic", codes, s, as_of)
    if sl is not None:
        df = sl[["ts_code", "trade_date", *fields]].sort_values(["ts_code", "trade_date"]).copy()
    else:
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
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if not fields:
        raise UnknownFieldError("fields 不能为空")
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
_MASK_COLS = ["can_buy", "can_sell", "reason", "open_hfq", "close_hfq", "amount", "amplitude",
              "pre_close_raw", "limit_up_raw", "limit_down_raw", "close_raw"]   # 末四列见函数注


def get_tradable_mask(exec_date: DateLike,
                      ts_codes: Sequence[str]) -> pd.DataFrame:
    """★ 唯一合法的「首参非 as_of_date」函数。语义：回测时钟已推进到 exec_date（T+1），
    此刻读 exec_date 的盘口是「当下」不是「未来」。调用方限定：只有 ashare/backtest/**
    与 ashare/strategy/**（P3 V4 修订：清单的限价带要给用户真实可输入的价格）可以调。

    index=ts_code；columns：can_buy / can_sell / reason / open_hfq / close_hfq / amount /
    amplitude / pre_close_raw / limit_up_raw / limit_down_raw / close_raw。
    末四列是【原始价】（P3 V4 的执行层豁免通道）：只供把清单换算成券商可输入的限价
    （`close_raw` 是**次日**的前收 —— 实盘出清单时 τ 尚无行情，限价带只能由它推），
    因子与回测数学仍然禁用（D8 —— 复权价与原始价互比是 bug 温床，L5 静态守卫不变）。
    判定（全部用【原始价】在函数内部完成，判定中间量不外泄）：
      停牌 / 无行情              → 两侧 False（suspended / no_quote）
      limit_up IS NULL           → 两侧 False（limit_unknown）★ 保守：宁可不交易，不可假设可交易
      open==limit_up 且 high==low → can_buy=False（limit_up_seal，一字涨停买不进）
      open==limit_down 且 high==low → can_sell=False（limit_down_seal，一字跌停卖不出）
      delist_date <= exec_date   → can_buy=False, can_sell=True（delisted，强制清仓路径）"""
    d = norm_date(exec_date, name="exec_date")
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
    def _last_real_close(code: str, upto: _dt.date) -> float:
        r = _conn().execute(
            "SELECT close * adj_factor FROM daily_bar WHERE ts_code = ? AND trade_date <= ? "
            "AND NOT is_suspended ORDER BY trade_date DESC LIMIT 1", [code, upto]).fetchone()
        return r[0] if r and r[0] is not None else float("nan")

    by_code = {r[0]: r for r in rows}
    missing = [c for c in codes if c not in by_code]
    # 退市后 daily_bar 不再有行（ingest 只写到 delist_date 当日）；但 delist_date <= exec_date 的股票
    # 必须仍走强平路径（can_sell=True），否则持仓永远卡在旧值。清仓价用最后一根非停牌 K 线的后复权收盘价，
    # 引擎在此之上再按 B8 打折（退市整理期首日收盘 × 0.5）。
    delisted_late: dict[str, float] = {}
    if missing:
        for code, delist, last_px in _conn().execute(
                "SELECT s.ts_code, s.delist_date, "
                "  (SELECT close * adj_factor FROM daily_bar b WHERE b.ts_code = s.ts_code AND NOT b.is_suspended "
                "   AND b.trade_date <= ? ORDER BY b.trade_date DESC LIMIT 1) "
                "FROM stock_basic s WHERE s.ts_code IN (" + ",".join("?" * len(missing)) + ") "
                "AND s.delist_date IS NOT NULL AND s.delist_date <= ?", [d, *missing, d]).fetchall():
            delisted_late[code] = last_px if last_px is not None else float("nan")
    out = []
    for code in codes:
        r = by_code.get(code)
        if r is None:
            if code in delisted_late:
                px = delisted_late[code]
                out.append((code, False, True, "delisted", px, px, float("nan"), float("nan")))
            else:
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
            if sus:      # 退市日通常已无成交（退市整理期在前一日结束）→ 当天是占位行，价格是前收，不能当真实成交价
                px = _last_real_close(code, d)
                out.append((code, False, True, "delisted", px, px, float("nan"), float("nan")))
            else:
                out.append((code, False, True, "delisted", o_h, c_h, amt, amp))
            continue
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
    df = pd.DataFrame(out, columns=["ts_code", *_MASK_COLS[:7]]).set_index("ts_code")
    raw = {c: (r[5], r[8], r[9], r[4]) for c, r in by_code.items()}   # pre_close/limit_up/limit_down/close
    for i, col in enumerate(_MASK_COLS[7:]):
        df[col] = pd.Series([raw.get(c, (None,) * 4)[i] for c in df.index],
                            index=df.index, dtype=float)
    return df


# ══════════════ 财报 PIT（D3）══════════════
_FIN_FIELDS = ("total_revenue", "revenue", "operate_profit", "total_profit", "n_income", "n_income_attr_p",
               "basic_eps", "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
               "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
               "roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "current_ratio",
               "or_yoy", "netprofit_yoy", "bps")
_FIN_META = ["ann_date", "end_date", "report_type", "lag_days"]


def _fin_visible(as_of: _dt.date, codes: list[str], fields: Sequence[str], *,
                 include_restated: bool, report_type: str) -> pd.DataFrame:
    """PIT 可见性内核（2026-08-26 重裁）：按【字段】取首个非空值的披露行。

    旧判据「update_flag=0 的最新公告行」在真实 Tushare 数据上失效（P2 真库验收实测）：
      · 近年利润表数值大量只在 update_flag=1 的行上 —— flag=0 最新行的净利非空率
        2010 年 98.6% → 2023 年只剩 21.8%，ep_ttm 全市场覆盖塌到 2.8%；
      · fina_indicator 接口没有 update_flag 字段，merge 时落成 flag=0 的【稀疏壳行】
        （同日公告、字段全 None），恰好把携值的 flag=1 行遮住。
    新判据（include_restated=False，默认）对每个 (ts_code, end_date, 字段)：
      1. flag=0 的携值行优先，组内【最新】公告胜 —— 更正公告（源头认定的合法修正，
         仍标 flag=0）按既有契约要能进因子（test_two_announcements_same_period_…）；
      2. 一条 flag=0 携值行都没有时，回落到 flag=1 行，组内【最早】公告胜 ——
         最早的披露离原始值最近，重述行即使冒用原始公告日也排在真原始之后。
    PIT 不变：只看 ann_date <= as_of。回落这一步是对源头数据形状的让步，
    代价是「唯一记录只有重述版」的期次会用上重述值 —— 但那也是旧判据下
    整只 NaN 之外唯一可得的信息。
    include_restated=True（研究重述影响用）→ 不分组，取该字段【最新】公告的非空值。

    返回列：ts_code, end_date, ann_date, report_type, <fields>，
    按 (ts_code ASC, end_date DESC) 排序（get_financial 的 head(n_periods) 依赖此序）。
    行的 ann_date = 该行各字段所用公告日的最大值 —— 整行「可知」的时点。
    """
    order = ("ann_date DESC, update_flag DESC" if include_restated else
             "update_flag ASC, CASE WHEN update_flag = 0 THEN ann_date END DESC, ann_date ASC")
    place = ",".join("?" * len(codes))
    per_field = []
    for f in fields:
        sql = f"""
        SELECT ts_code, end_date, ann_date, {f} FROM (
            SELECT ts_code, end_date, ann_date, {f},
                   row_number() OVER (PARTITION BY ts_code, end_date
                                      ORDER BY {order}) AS rn
            FROM financial_pit
            WHERE ann_date <= ? AND report_type = ? AND {f} IS NOT NULL
              AND ts_code IN ({place})
        ) WHERE rn = 1
        """
        d = _conn().execute(sql, [as_of, report_type, *codes]).fetchdf()
        for c in ("end_date", "ann_date"):
            d[c] = pd.to_datetime(d[c]).dt.date
        per_field.append(d.set_index(["ts_code", "end_date"])
                          .rename(columns={"ann_date": f"__ann_{f}"}))
    m = pd.concat(per_field, axis=1)                     # (ts,end) 外连接：字段可来自不同公告
    if m.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "ann_date", "report_type", *fields])
    ann_cols = [f"__ann_{f}" for f in fields]
    m["ann_date"] = m[ann_cols].apply(
        lambda r: max(x for x in r if x is not None and not pd.isna(x)), axis=1)
    m["report_type"] = report_type
    m = m.drop(columns=ann_cols).reset_index()
    m = m.sort_values(["ts_code", "end_date"], ascending=[True, False]).reset_index(drop=True)
    return m[["ts_code", "end_date", "ann_date", "report_type", *fields]]


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
      True 仅供研究「重述影响」，任何进入回测的因子都必须用 False ——
      重述行的 ann_date 可能仍是原始公告日（Tushare 不保证），True 不是 PIT 安全的。
    额外返回列：ann_date、end_date、report_type、lag_days(= as_of - ann_date)。
    注：只返回有可见披露的股票（index 是 ts_codes 的子集）；get_financial_ttm 则对无数据的股票返回 NaN。"""
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if not fields:
        raise UnknownFieldError("fields 不能为空")
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
    return df.set_index(["ts_code", "end_date"], drop=False)[out_cols]


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
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
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
    index=period，columns=indicator，附加列 <indicator>__publish_date 便于审计。
    lookback_periods 按【每个指标各自的观测】计数（取各自最近 N 期、行取并集）——
    不能按并集行数截：混频查询时日频指标把并集灌满逐日日期，tail(N) 只剩最近
    N 个交易日，月频指标整列 NaN（2026-08-26 P1 真库验收实测）。"""
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
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
    keep = sorted({p for i in inds if i in val.columns
                   for p in val[i].dropna().tail(lookback_periods).index})
    out = out.loc[keep]
    out.index.name = "period"
    return out[cols]


def get_north_aggregate(as_of_date: DateLike) -> pd.DataFrame:
    """全市场北向持仓市值与流通市值的逐日合计（宏观层 north_flow_60 的唯一原料）。
    index=trade_date ≤ as_of，columns：north_mv（Σ ratio%/100 × circ_mv，缺持仓记录 = 未持有 = 0
    —— 这是【加总】不是逐股因子，B5 的「不填 0」不适用）、circ_mv_total（Σ circ_mv）。
    一条 SQL 聚合全历史（P3 Task 8 增量缓存的地基：逐月末 3 次横截面查询 × 511 个调仓日
    的老路径是小时级瓶颈）。日频行情当日可得，PIT 语义即 trade_date ≤ as_of。"""
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    df = _conn().execute(
        "SELECT d.trade_date, "
        "       sum(coalesce(m.hk_hold_ratio, 0) / 100.0 * d.circ_mv) AS north_mv, "
        "       sum(d.circ_mv) AS circ_mv_total "
        "FROM daily_basic d LEFT JOIN money_flow m "
        "  ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date "
        "WHERE d.trade_date <= ? GROUP BY 1 ORDER BY 1", [as_of]).fetchdf()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.set_index("trade_date")


_MONEY_FLOW_FIELDS = ("hk_hold_ratio",)


def get_money_flow(as_of_date: DateLike,
                   ts_codes: Sequence[str],
                   fields: Sequence[str] = ("hk_hold_ratio",),
                   lookback: int = 20) -> pd.DataFrame:
    """MultiIndex (ts_code, trade_date)，按交易日历补齐 lookback 个交易日。
    ★ hk_hold_ratio 仅 2016-12 起有数据；早于该日 / 缺失日返回 NaN，不填 0（B5）——
      调用方（north_hold_chg_20 因子）靠 FactorSpec.available_from 声明。"""
    as_of = norm_date(as_of_date)
    _check_in_calendar(as_of)
    if lookback < 1:
        raise QueryError("lookback 必须 >= 1")
    if not fields:
        raise UnknownFieldError("fields 不能为空")
    bad = [f for f in fields if f not in _MONEY_FLOW_FIELDS]
    if bad:
        raise UnknownFieldError(f"get_money_flow 不支持字段 {bad}")
    codes = list(ts_codes)
    all_days = _open_days()
    days = all_days[:bisect.bisect_right(all_days, as_of)][-lookback:]
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
