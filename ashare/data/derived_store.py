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

BACKTEST_RUN_COLUMNS = ("run_id", "param_hash", "data_snapshot_id", "engine_version",
                        "started_at", "elapsed_sec", "config_json", "metrics_json", "is_oos")


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


def _checked_universe(universe: Sequence[str]) -> list[str]:
    """`factors.base._checked_universe` 的两条**真正会静默出错**的检查，在这里重写一遍。

    不是复用 —— 本模块绝不 import `ashare.factors`（模块头第一条）。代价是两份代码，
    换来的是缓存路径不再是一道绕过唯一校验点的旁门：
      · 重复代码经 `reindex` 会静默复制出重复行，下游把同一只股票加权两次；
      · 空池的返回值是「空表」，与未命中【逐位相同】，于是调用方的 bug 变成一次静默重算。
    """
    codes = list(universe)
    if not codes:
        raise ValueError("universe 为空：空横截面返回的空表与未命中长得一样，"
                         "调用方会把自己的 bug 读成「这一天没算过」")
    if len(set(codes)) != len(codes):
        dup = sorted({c for c in codes if codes.count(c) > 1})
        raise ValueError(f"universe 含重复代码 {dup[:5]}：reindex 会复制出重复行，"
                         f"横截面回归会把它当两只股票加权两次")
    return codes


def _missing(param_hashes: Mapping[str, str], hit: set, d: _dt.date) -> list[str]:
    """当前快照下一行都没有的因子，一个一条 warning。

    ★ 判据是「有没有行」而不是「值是不是全 NaN」：一个合法地整天全 NULL 的因子
      （`available_from` 未到）与一个陈旧/没算过的因子，在返回的帧里【逐位相同】——
      都是一列 NaN；而只要还有别的因子命中，帧就不是 `.empty`，调用方分不出来。
      那正是 `combine`「静默剔除 + 按剩余权重重新归一」那条路的入口，
      所以这一层的降级必须自己出声（global-constraints ★）。
    """
    return [f"{d} {n}@{h}: 当前快照下没有任何行（没算过、或快照已过期），该列全 NaN"
            for n, h in param_hashes.items() if n not in hit]


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


# ══════════════ 回测运行台账（backtest_run）══════════════

def write_backtest_run(row: Mapping) -> int:
    """UPSERT 一条回测运行记录，返回写入行数（恒 1）。键见 `BACKTEST_RUN_COLUMNS`。

    ★ 覆盖语义是 `INSERT OR REPLACE`：`run_id` 里已经含了 `started_at`
      （`backtest.store.make_run_id`），撞主键只可能是**同一次运行重存一遍**，
      整行替换正是想要的；`DO NOTHING` 会让第二次 save 静默无效。
    ★ 与 `factor_value` 不同，这张表**不做快照校验**：因子值是缓存（陈旧值喂进回测
      会产出假净值曲线），而运行记录是**历史**——「这次运行用的是哪份数据」正是
      `data_snapshot_id` 这一列要说的话，不是把它筛掉的理由。
    """
    missing = [c for c in BACKTEST_RUN_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"backtest_run 缺键 {missing}：param_hash / data_snapshot_id 撞 NOT NULL，"
                         f"缺别的只会在写入深处炸出一个没有上下文的错误")
    conn = _derived.connect_write()
    try:
        _derived.init_schema(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO backtest_run ({', '.join(BACKTEST_RUN_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(BACKTEST_RUN_COLUMNS))})",
            [row[c] for c in BACKTEST_RUN_COLUMNS])
    finally:
        conn.close()
    return 1


def read_backtest_run(run_id: str) -> Optional[dict]:
    """按 `run_id` 读回一条运行记录；没有这一行（或派生库还不存在）返回 `None`。

    ★ 返回 `None` 而不是空 dict：调用方（`backtest.store.load`）要把「没落过库」
      与「落过但字段是空的」分开 —— 前者该抛 FileNotFoundError，后者是数据坏了。
    ★ 表不存在时**照抛**（老版本派生库），不吞成 `None`：那是「该 rm 掉重算」的信号
      （见 `derived_schema.sql` 的模块头），吞掉就变成一次查不出原因的「没有这条运行」。
    """
    conn = _read_conn()
    if conn is None:
        return None
    try:
        got = conn.execute(
            f"SELECT {', '.join(BACKTEST_RUN_COLUMNS)} FROM backtest_run WHERE run_id = ?",
            [run_id]).fetchall()
    finally:
        conn.close()
    return dict(zip(BACKTEST_RUN_COLUMNS, got[0])) if got else None


# ══════════════ 读 ══════════════

def read_factor_values(param_hashes: Mapping[str, str], date, universe: Sequence[str], *,
                       processed: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """某一天的因子横截面 + warnings。`index = universe`（顺序一致），`columns = param_hashes` 的键序。

    `param_hashes` 是 `{factor_name: param_hash}`：主键含 param_hash，只按名字读会在
    闸 5 的参数高原（同一因子的 ±30% 网格）下同时命中两代，pivot 要么炸要么静默留一代。

    ★ 一行都没命中 → 返回**带列名的空表**，绝不现算。现算与落库的口径分歧是最难查的
      一类 bug；要不要补算由调用方决定（`ashare.factors.store.build`）。
    ★ 命中的行里，快照与当前不符的**当未命中**（见模块头）。
    ★ 部分命中 → 按 universe 对齐，池里有而库里没有的票记 NaN（因子缺值本来就长这样）。
    ★ **逐因子**未命中会出一条 warning（2026-08-22 评审 C1）：三个因子里有一个陈旧时，
      它会被 `reindex(columns=names)` 物化成一列 NaN，而帧不是 `.empty` ——
      与「这个因子那天本来就没数据」逐位相同。本函数整件事就是判命中，
      它自己降级却不出声，等于把 D7 的破绽藏进一列看起来很正常的 NaN 里。
    """
    names = list(param_hashes)
    empty = pd.DataFrame(columns=names, index=pd.Index([], name="ts_code", dtype=object),
                         dtype=float)
    if not names:
        return empty, []

    d = query.norm_date(date, name="date")
    codes = _checked_universe(universe)
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return empty, _missing(param_hashes, set(), d)
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

    warns = _missing(param_hashes, set(got["factor_name"]), d)
    if got.empty:
        return empty, warns
    return (got.pivot(index="ts_code", columns="factor_name", values="value")
               .reindex(index=codes, columns=names)
               .rename_axis(columns=None)), warns


_WINDOW_COLS = ["factor_name", "trade_date", "ts_code", "raw_value", "processed_value"]


def read_factor_window(param_hashes: Mapping[str, str], start, end) -> pd.DataFrame:
    """[start, end] 整窗、**当前快照**下的因子长表 —— 一次连接一次查询。
    这是 `factors.store.read_current` 文档里预留的那张「批量口」变更单（2026-08-27
    性能专项兑现）：快照过滤与 `read_factor_values` 同一句，判命中语义不变，
    省掉的是逐日 511 次「开连接 + 两次单日查询」。库不存在返回空表（= 什么都没算过）。"""
    if not param_hashes:
        return pd.DataFrame(columns=_WINDOW_COLS)
    s = query.norm_date(start, name="start")
    e = query.norm_date(end, name="end")
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return pd.DataFrame(columns=_WINDOW_COLS)
    try:
        got = conn.execute(
            f"SELECT {', '.join(_WINDOW_COLS)} FROM factor_value "
            f"WHERE trade_date BETWEEN ? AND ? AND snapshot_id = ? "
            f"AND (factor_name, param_hash) IN ({_pairs_clause(param_hashes)})",
            [s, e, snap, *_pairs_params(param_hashes)]).fetchdf()
    finally:
        conn.close()
    if len(got):
        got["trade_date"] = pd.to_datetime(got["trade_date"]).dt.date
    return got


def drop_out_of_universe(param_hashes: Mapping[str, str], date, universe: Sequence[str]) -> int:
    """删掉这一天**不在当前股票池里**的旧因子值，返回删除行数。

    ★ 为什么必须删（2026-08-22 评审 I1）：`factor_value` 的行只增不减，而股票池是按
      `as_of_date` 动态生成的（D5）—— 数据一修正（改一个上市日 / 一段 ST 区间），
      某只票就可能退出某个历史日期的池子，它那行却留在库里、盖着上一个快照。
      于是 `current_factor_dates` 的 `bool_and` 对这一天**永远为假**：
      每次跑都重算、`overwrite=False` 的跳过从此永久失效，而 `read` 那边一切正常
      （它按 universe 对齐，根本看不见这行）—— 一个只表现为"缓存莫名其妙不生效"的状态。
    ★ 只删「这一批 (因子, 参数哈希) + 这一天」范围内的孤儿行：闸 5 的参数高原下
      同一因子并存多代是**正常的**，不能顺手清掉别的 param_hash。
    """
    codes = _checked_universe(universe)
    if not param_hashes:
        return 0
    conn = _derived.connect_write()
    try:
        _derived.init_schema(conn)
        return conn.execute(
            f"DELETE FROM factor_value WHERE trade_date = ? "
            f"AND (factor_name, param_hash) IN ({_pairs_clause(param_hashes)}) "
            f"AND ts_code NOT IN ({', '.join(['?'] * len(codes))})",
            [query.norm_date(date, name="trade_date"),
             *_pairs_params(param_hashes), *codes]).fetchall()[0][0]
    finally:
        conn.close()


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
    ★ 覆盖率**只数当前快照的行**（2026-08-22 评审 I1）：`read` 只服务当前快照，而
      `factor_value` 只增不减，混代统计出来的数**比实际能读到的高**。实测 3 行当前
      （1 行非空）+ 2 行非空孤儿 → 报 0.60、实际服务 0.333，而 0.60 恰是
      `min_coverage` 的默认值 —— 报告刚好过闸，能读到的横截面只有它的一半。
      整天都陈旧的日期 cov 记 NULL（`avg` 会跳过它），它由 `n_stale_dates` 那一列报，
      不混进覆盖率：一个是「没有当前数据」，一个是「当前数据很稀」。
    """
    if names is not None and not names:
        # 空列表 = 什么都没问（与 read_factor_values({}) / current_factor_dates({}) 同口径）。
        # 不挡的话 `IN ()` 会抛 duckdb.ParserException —— 一句调用方读不懂的 SQL 语法错。
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    snap = query.snapshot_id()
    conn = _read_conn()
    if conn is None:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    try:
        where, params = "", [snap, snap, snap]
        if names is not None:
            where = f"WHERE factor_name IN ({', '.join(['?'] * len(names))})"
            params += list(names)
        rep = conn.execute(
            f"""
            WITH per_date AS (
                SELECT factor_name, param_hash, trade_date,
                       (count(raw_value) FILTER (WHERE snapshot_id = ?))::DOUBLE
                           / nullif(count(*) FILTER (WHERE snapshot_id = ?), 0) AS cov,
                       bool_and(snapshot_id = ?)                                AS is_current
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
