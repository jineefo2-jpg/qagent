"""派生库（`factor_value` / `backtest_run`）的唯一读写出口 —— 只收发 DataFrame 与基础类型。

★ 为什么这一层在 `ashare/data/` 而不是 `ashare/factors/`（架构 §4.3 + 2026-08-21 修正）：
  分层闸 L1 只允许 `ashare/data/**` `import duckdb`。闸故意是粗粒度的 ——「这个文件能不能
  import duckdb」AST 查得出来，「能 import 但只准连 derived.duckdb」查不出来。
  反过来，本模块**绝不 import `ashare.factors` / `ashare.backtest`**：底层拿到高层类型，
  `ashare/data` 就不再是能独立测试的底座。「算 → 写」的编排住在 `ashare/factors/store.py`。

★ 读取必须校验 `snapshot_id`（架构 B4，本模块存在的头号理由）：
  `factor_value` 的主键**不含** snapshot_id（`derived_schema.sql`），换数据重算是覆盖写，
  snapshot_id 只是一列。代价就是**命中判定要由读取方补上** —— 不校验就会把另一批数据
  算出的因子值静默喂进回测，产出一条好看的假净值曲线。所以：
  快照不等的行一律**当未命中**，而且当前快照由本模块自己向 `query` 要，不做成参数
  （做成参数就一定有人传错，而传错的后果恰恰是这道校验要挡的东西）。

★ L2（公开函数首参 `as_of_date`）不延伸到这里：回测读因子面板天然是读一个日期区间，
  强行套 as_of 形状就不对。前视保护在**写入时** —— 值本身是按 PIT 纪律算出来的。
"""
from __future__ import annotations
import datetime as _dt
from typing import Mapping, Optional, Sequence

import pandas as pd

from . import _derived, query

FACTOR_VALUE_COLUMNS = ("factor_name", "param_hash", "trade_date", "ts_code",
                        "raw_value", "processed_value", "snapshot_id")

COVERAGE_COLUMNS = ["factor_name", "param_hash", "first_date", "last_date",
                    "n_dates", "mean_coverage", "n_stale_dates"]


def _read_conn():
    """派生库还不存在 = 什么都没算过，不是异常（第一次跑必然如此）。"""
    try:
        return _derived.connect_read()
    except FileNotFoundError:
        return None


def _pairs_clause(param_hashes: Mapping[str, str]) -> str:
    return ", ".join(["(?, ?)"] * len(param_hashes))


def _pairs_params(param_hashes: Mapping[str, str]) -> list:
    return [x for n, h in param_hashes.items() for x in (n, h)]


def _num(x) -> Optional[float]:
    """缺失值必须落成 NULL。DuckDB 的 DOUBLE **装得下 NaN**，而 `count(raw_value)` 会把
    NaN 算作一个值 —— 写 NaN 进去，coverage_report 就永远报 100% 覆盖率。"""
    return None if pd.isna(x) else float(x)


# ══════════════ 写 ══════════════

def write_factor_values(df: pd.DataFrame) -> int:
    """UPSERT 一批因子值，返回写入行数。列见 `FACTOR_VALUE_COLUMNS`。

    覆盖语义来自 `_derived.UPSERT_FACTOR_VALUE`（值与 snapshot_id 绑在一起更新）：
    发 `DO NOTHING` 会留下陈旧值配陈旧快照，而那种行 `read_factor_values` 认得出来
    并会当未命中 —— 于是库里有行、读出来没有，谁都查不明白。
    """
    missing = [c for c in FACTOR_VALUE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"factor_value 缺列 {missing}：七列缺一不可 —— 缺 snapshot_id 撞 NOT NULL，"
                         f"缺别的只会在写入深处炸出一个没有上下文的 KeyError")
    if df.empty:
        return 0

    rows = [(fn, ph, query.norm_date(td, name="trade_date"), tc, _num(rv), _num(pv), sid)
            for fn, ph, td, tc, rv, pv, sid
            in df[list(FACTOR_VALUE_COLUMNS)].itertuples(index=False, name=None)]
    conn = _derived.connect_write()
    try:
        _derived.init_schema(conn)
        conn.executemany(_derived.UPSERT_FACTOR_VALUE, rows)
    finally:
        conn.close()
    return len(rows)


# ══════════════ 读 ══════════════

def read_factor_values(param_hashes: Mapping[str, str], date, universe: Sequence[str], *,
                       processed: bool = True) -> pd.DataFrame:
    """某一天的因子横截面。`index = universe`（顺序一致），`columns = param_hashes` 的键序。

    `param_hashes` 是 `{factor_name: param_hash}`：主键含 param_hash，只按名字读会在
    闸 5 的参数高原（同一因子的 ±30% 网格）下同时命中两代，pivot 要么炸要么静默留一代。

    ★ 一行都没命中 → 返回**带列名的空表**，绝不现算。现算与落库的口径分歧是最难查的
      一类 bug；要不要补算由调用方决定（`ashare.factors.store.build`）。
    ★ 命中的行里，快照与当前不符的**当未命中**（见模块头）。
    ★ 部分命中 → 按 universe 对齐，池里有而库里没有的票记 NaN（因子缺值本来就长这样）。
    """
    names = list(param_hashes)
    empty = pd.DataFrame(columns=names, index=pd.Index([], name="ts_code", dtype=object),
                         dtype=float)
    if not names:
        return empty

    d = query.norm_date(date, name="date")
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return empty
    try:
        # 列名是本模块的字面量二选一，不是外部输入；日期/代码/哈希全部参数化（Q1）
        col = "processed_value" if processed else "raw_value"
        got = conn.execute(
            f"SELECT factor_name, ts_code, {col} AS value FROM factor_value "
            f"WHERE trade_date = ? AND snapshot_id = ? "
            f"AND (factor_name, param_hash) IN ({_pairs_clause(param_hashes)})",
            [d, snap, *_pairs_params(param_hashes)]).fetchdf()
    finally:
        conn.close()

    if got.empty:
        return empty
    return (got.pivot(index="ts_code", columns="factor_name", values="value")
               .reindex(index=list(universe), columns=names)
               .rename_axis(columns=None))


def current_factor_dates(param_hashes: Mapping[str, str],
                         dates: Sequence[_dt.date]) -> set:
    """已算、且**整天每一行**都属于当前快照的 `(factor_name, trade_date)`。

    ★ 判据是 `bool_and` 而不是「这一天有行」：只要有一行陈旧，`read_factor_values` 就会
      把整天当未命中（它按 `snapshot_id = ?` 过滤）。用 EXISTS / count(*) > 0 判成已算，
      那几行陈旧值就永远不会被重算 —— build 说算过了、read 说没有，这一天静默消失。
    """
    if not param_hashes or len(dates) == 0:
        return set()
    ds = [query.norm_date(d, name="trade_date") for d in dates]
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return set()
    try:
        rows = conn.execute(
            f"SELECT factor_name, trade_date FROM factor_value "
            f"WHERE trade_date IN ({', '.join(['?'] * len(ds))}) "
            f"AND (factor_name, param_hash) IN ({_pairs_clause(param_hashes)}) "
            f"GROUP BY factor_name, param_hash, trade_date "
            f"HAVING bool_and(snapshot_id = ?)",
            [*ds, *_pairs_params(param_hashes), snap]).fetchall()
    finally:
        conn.close()
    return {(n, d) for n, d in rows}


def coverage_report(names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """每个 (因子, 参数) 已算的日期区间与覆盖率。`names=None` 看全库。

    ★ 覆盖率量的是 `raw_value` 的非空占比，不是 `processed_value`：处理链末端有
      `fillna(0)`，processed 永远非空 —— 拿它算，这道指标恒等于 100%，永远不会响。
    ★ `mean_coverage` 是**逐日覆盖率的均值**（与 `FactorSpec.min_coverage` 同一口径，
      那也是逐日判的），不是总行数占比：股票池逐日变化，两者不等。
    ★ `n_stale_dates`：快照与当前不符的日期数。没有它，报告会说「2010–2024、覆盖率 92%」
      而 `read` 一行都不给 —— 这一层最容易被当成灵异事件的状态，得在同一张表上看得见。
    """
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    try:
        where, params = "", [snap]
        if names is not None:
            where = f"WHERE factor_name IN ({', '.join(['?'] * len(names))})"
            params += list(names)
        rep = conn.execute(
            f"""
            WITH per_date AS (
                SELECT factor_name, param_hash, trade_date,
                       count(raw_value)::DOUBLE / count(*) AS cov,
                       bool_and(snapshot_id = ?)           AS is_current
                FROM factor_value {where}
                GROUP BY factor_name, param_hash, trade_date)
            SELECT factor_name, param_hash,
                   min(trade_date) AS first_date, max(trade_date) AS last_date,
                   count(*)        AS n_dates,
                   avg(cov)        AS mean_coverage,
                   count(*) FILTER (WHERE NOT is_current) AS n_stale_dates
            FROM per_date GROUP BY factor_name, param_hash
            ORDER BY factor_name, param_hash
            """, params).fetchdf()
    finally:
        conn.close()

    if rep.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    for c in ("first_date", "last_date"):           # DATE 经 fetchdf 会变成 Timestamp
        rep[c] = pd.to_datetime(rep[c]).dt.date
    return rep
