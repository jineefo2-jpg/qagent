"""Task 8：因子落库 —— `ashare/data/derived_store.py`（L1 持 duckdb）
+ `ashare/factors/store.py`（L3 编排「算 → 写」）。

本文件守的不是「能存能取」，是**取回来的因子值确实是当前这批数据算出来的**：

  · `read` 按 `snapshot_id` 校验命中，不等的行【当未命中】—— Task 1 把 snapshot_id 定为
    列而非主键的直接代价。不校验就会把另一批数据算出的因子值静默喂进回测，
    产出一条好看的假净值曲线（架构 B4）。这是本文件最吃重的一条断言。
  · 主键命中 ≠ 缓存有效：`param_hash` 哈希 `default_params` + `neutralize` + `available_from`
    （后两个是 2026-08-22 评审 I5 折进去的 —— 它们都改变落库的值），但**因子函数体**
    不在里面（也没法哈希）。所以判定有效性只能靠 `snapshot_id` + `overwrite`，
    不能靠「这个主键下有行」。
  · `read` 未命中返回空，**不静默现算** —— 现算与落库的口径分歧是最难查的一类 bug。
  · `build` 不遍历 `FACTOR_REGISTRY`：`raw_value` 是 DOUBLE，而 `industry` 返回字符串。

fixture 的数值一律取除不尽的值（global-constraints ★）：整齐的数字会让「raw 与 processed
写反了」「zscore 没跑」这类变异与真实现逐位相同。
"""
from __future__ import annotations
import ast
import dataclasses
import datetime as dt
import os
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, _derived, derived_store, query
from ashare.factors import base, pipeline, store

D1 = dt.date(2024, 1, 5)          # 池 = A / B / C
D2 = dt.date(2024, 1, 10)         # 池 = A / C（B 此日为 ST）
D3 = dt.date(2024, 1, 25)         # 池 = A / B（C 已退市）
COLS = list(derived_store.FACTOR_VALUE_COLUMNS)

# 除不尽的原始值：zscore 之后仍然除不尽，raw / processed 两列写反了立刻看得见
_RAW = {"A00001.SZ": 0.4142135624, "B00002.SZ": -1.7320508076,
        "C00003.SH": 2.2360679775, "D00004.SZ": 0.5772156649}


# ══════════════ 脚手架 ══════════════

@pytest.fixture(autouse=True)
def clean_registry():
    """FACTOR_REGISTRY 是模块级全局；yield 出真注册表的副本供「必须对着真因子断言」的用例还原。"""
    saved = dict(base.FACTOR_REGISTRY)
    base.FACTOR_REGISTRY.clear()
    yield saved
    base.FACTOR_REGISTRY.clear()
    base.FACTOR_REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def close_market_db():
    """`build` 会 `snapshot_id(pin=True)` 钉住 inode，只有 close_db() 解钉 —— 不解会漏给下一个用例。"""
    yield
    query.close_db()


@pytest.fixture
def store_env(market_db, tmp_path, monkeypatch):
    """chdir 到 tmp：`DEFAULT_DERIVED_PATH` 是相对路径，派生库因此落在 tmp 里。"""
    monkeypatch.chdir(tmp_path)
    query.open_db(market_db)
    return market_db


def _fn(bump: float = 0.0):
    def compute(as_of_date, universe, **params):    # 收下 default_params：参数高原的用例要传
        return pd.Series([_RAW[c] + bump for c in universe], index=list(universe))
    return compute


def _register(name="f1", *, bump=0.0, category="price", **kw):
    """测试因子：neutralize=False —— 4 只票的横截面走不进 OLS（MIN_OBS=30），
    留着只会让每条用例都拖一条「样本不足」warning，遮住真正要断言的那条。"""
    kw.setdefault("neutralize", False)
    return base.factor(name=name, direction=1, category=category, lookback_days=1,
                       **kw)(_fn(bump))


def _rows(*tuples) -> pd.DataFrame:
    return pd.DataFrame(list(tuples), columns=COLS)


def _bump_snapshot(market_db: str) -> None:
    """改一行与股票池无关的 stock_status（2024-01-29 之后），只动指纹不动数据语义。"""
    query.close_db()
    w = _db.connect_write(market_db)
    w.execute("INSERT INTO stock_status VALUES ('A00001.SZ', DATE '2024-01-29', DATE '2024-01-30', 'ST')")
    w.close()
    query.open_db(market_db)


def _table() -> pd.DataFrame:
    conn = _derived.connect_read(_derived.DEFAULT_DERIVED_PATH)
    try:
        return conn.execute("SELECT * FROM factor_value ORDER BY factor_name, trade_date, ts_code").fetchdf()
    finally:
        conn.close()


# ══════════════ 1 · derived_store 读写往返 ══════════════

def test_write_then_read_roundtrip_keeps_raw_and_processed_apart(store_env):
    snap = query.snapshot_id()
    n = derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, -0.7071067812, snap),
        ("f1", "ph1", D1, "B00002.SZ", -1.7320508076, 1.4142135624, snap)))
    assert n == 2

    raw, w1 = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"],
                                               processed=False)
    proc, w2 = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"])
    assert list(raw.columns) == ["f1"] and raw.index.tolist() == ["A00001.SZ", "B00002.SZ"]
    assert raw["f1"].tolist() == [0.4142135624, -1.7320508076]
    assert proc["f1"].tolist() == [-0.7071067812, 1.4142135624]
    assert (w1, w2) == ([], []), "全部命中不该有 warning —— 会响的东西必须平时不响"


def test_write_normalizes_the_trade_date(store_env):
    """`write` 收 str 日期（`norm_date` 的契约）。直接把 '20240105' 塞进 DATE 列，
    DuckDB 抛 ConversionException（它只认 YYYY-MM-DD）—— 归一化不是装饰。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", "20240105", "A00001.SZ", 0.4142135624, 0.1, query.snapshot_id())))
    out, _ = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ"], processed=False)
    assert out["f1"].tolist() == [0.4142135624]


def test_read_returns_an_empty_frame_with_the_requested_columns_on_a_miss(store_env):
    """未命中返回空，**不现算** —— 现算与落库的口径分歧是最难查的一类 bug，
    要不要补算由调用方决定。列名必须在（Q3），否则调用方拿到的是一个没有形状的空表。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.0, query.snapshot_id())))

    out, warns = derived_store.read_factor_values({"f1": "ph1"}, D2, ["A00001.SZ", "C00003.SH"])
    assert out.empty, "另一个交易日没落过库，必须返回空而不是就地算一份"
    assert list(out.columns) == ["f1"]
    assert len(warns) == 1 and "f1" in warns[0]


def test_read_treats_a_snapshot_mismatch_as_a_miss(store_env):
    """★ 本文件最吃重的一条：这条不过 = 回测会拿到用另一批数据算出的因子值。

    Task 1 把 snapshot_id 定为列而非主键，代价就是命中判定必须由读取方补上。"""
    stale = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, -0.7071067812, stale),
        ("f1", "ph1", D1, "B00002.SZ", -1.7320508076, 1.4142135624, stale)))
    assert not derived_store.read_factor_values({"f1": "ph1"}, D1,
                                                ["A00001.SZ", "B00002.SZ"])[0].empty

    _bump_snapshot(store_env)
    assert query.snapshot_id() != stale, "前提没成立：这次改动没有改变数据指纹"

    out, warns = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"])
    assert out.empty, "陈旧快照的因子值必须当未命中，而不是照样返回"
    assert list(out.columns) == ["f1"]
    assert len(warns) == 1 and "f1" in warns[0]
    assert len(_table()) == 2, "read 不负责删行 —— 陈旧行留在库里等 build 覆盖"


def test_read_warns_for_each_factor_that_is_not_current(store_env):
    """★ 评审 C1：**逐因子**判命中。三个因子里只有一个陈旧时，帧【不是】`.empty` ——
    那一列被 `reindex(columns=names)` 物化成全 NaN，与 f3（行在、值合法地全 NULL）
    逐位相同，调用方分不出来。而它下游正是 `combine` 的「静默剔除 + 按剩余权重
    重新归一」。走到这里只需要公开调用：build f1 → promote → build f2，
    也就是「给系统新增一个因子」的正常做法。"""
    stale = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, -0.7071067812, stale),
        ("f1", "ph1", D1, "B00002.SZ", -1.7320508076, 1.4142135624, stale)))
    _bump_snapshot(store_env)
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f2", "ph1", D1, "A00001.SZ", 2.2360679775, 0.3010299957, snap),
        ("f2", "ph1", D1, "B00002.SZ", 0.5772156649, 0.6931471806, snap),
        ("f3", "ph1", D1, "A00001.SZ", None, None, snap),
        ("f3", "ph1", D1, "B00002.SZ", None, None, snap)))

    out, warns = derived_store.read_factor_values(
        {"f1": "ph1", "f2": "ph1", "f3": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"], processed=False)

    assert not out.empty, "前提：部分命中的帧非空 —— 陈旧那一列只是静默变成 NaN"
    assert list(out.columns) == ["f1", "f2", "f3"]
    assert out["f1"].isna().all() and out["f3"].isna().all()
    assert out["f2"].tolist() == [2.2360679775, 0.5772156649]
    assert len(warns) == 1 and "f1" in warns[0], warns
    assert "f3" not in warns[0], "f3 的行【在当前快照】，全 NULL 是合法缺值，不是降级"


def test_read_with_nothing_asked_returns_an_empty_frame_and_no_warnings(store_env):
    """先落一行：库不存在的话根本走不到那道空 mapping 的闸（前面就 return 了），
    而没有它 `IN ()` 会抛 duckdb.ParserException。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, query.snapshot_id())))
    out, warns = derived_store.read_factor_values({}, D1, ["A00001.SZ"])
    assert out.empty and list(out.columns) == []
    assert warns == [], "什么都没问 ≠ 什么都没命中"


def test_read_rejects_a_universe_that_base_would_reject(store_env):
    """`base._checked_universe` 自称「18 个因子的唯一校验点」，而缓存这条路是第二个入口。
    重复代码经 reindex 会静默复制出重复行（下游把同一只股票加权两次），
    空池返回的空表与「未命中」逐位相同（调用方把自己的 bug 读成"这天没算过"）。
    L1 不许本模块 import ashare.factors，所以这两条只能在 derived_store 里再写一遍。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, query.snapshot_id())))
    with pytest.raises(ValueError, match="重复"):
        derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "A00001.SZ"])
    with pytest.raises(ValueError, match="为空"):
        derived_store.read_factor_values({"f1": "ph1"}, D1, [])


def test_read_matches_param_hash_not_just_the_factor_name(store_env):
    """闸 5 的参数高原会让同一个因子名下并存多代（window=20 / 26）。
    只按名字读会同时拿到两代 → (ts_code, factor_name) 重复，pivot 要么炸要么静默留一代。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph20", D1, "A00001.SZ", 0.4142135624, -0.7071067812, snap),
        ("f1", "ph26", D1, "A00001.SZ", 2.2360679775, 1.7320508076, snap)))

    out, warns = derived_store.read_factor_values({"f1": "ph26"}, D1, ["A00001.SZ"],
                                                  processed=False)
    assert out["f1"].tolist() == [2.2360679775]
    assert len(out) == 1
    assert warns == []


def test_read_reindexes_onto_the_requested_universe(store_env):
    """index 必须逐位等于 universe（compute_factor 的契约）：
    多出来的票要丢掉，少的票补 NaN —— 否则调用方的 score / 权重会与池错位。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "D00004.SZ", 0.5772156649, 0.2, snap)))

    out, _ = derived_store.read_factor_values({"f1": "ph1"}, D1,
                                              ["C00003.SH", "A00001.SZ"], processed=False)
    assert out.index.tolist() == ["C00003.SH", "A00001.SZ"]
    assert pd.isna(out.loc["C00003.SH", "f1"])          # 池里有、库里没有 → NaN
    assert out.loc["A00001.SZ", "f1"] == 0.4142135624
    assert "D00004.SZ" not in out.index                 # 库里有、池里没有 → 丢掉


def test_write_stores_a_missing_value_as_null_not_nan(store_env):
    """DuckDB 的 DOUBLE 真的能装 NaN：`count(raw_value)` 会把它算作一个值，
    于是 coverage_report 永远报 100%。缺失必须落成 NULL。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", float("nan"), float("nan"), query.snapshot_id())))

    conn = _derived.connect_read(_derived.DEFAULT_DERIVED_PATH)
    try:
        assert conn.execute("SELECT count(raw_value), count(processed_value), count(*) "
                            "FROM factor_value").fetchone() == (0, 0, 1)
    finally:
        conn.close()


def test_write_upserts_and_carries_the_new_snapshot_id(store_env):
    """重算 = 覆盖，且 snapshot_id 跟着更新。发 DO NOTHING 会留下陈旧值配陈旧快照。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, -0.7071067812, "snap_old")))
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 2.2360679775, 1.7320508076, "snap_new")))

    t = _table()
    assert len(t) == 1
    assert t.loc[0, "raw_value"] == 2.2360679775
    assert t.loc[0, "processed_value"] == 1.7320508076
    assert t.loc[0, "snapshot_id"] == "snap_new"


def test_write_rejects_a_frame_that_is_missing_columns(store_env):
    """少了 snapshot_id 会撞 NOT NULL，少了 factor_name 只会在深处炸出一个 KeyError。
    七列一起在入口验，报缺哪一列 —— 逐列删一遍：只测 snapshot_id 那一列的话，
    另外六列从检查里删掉也没有人会发现（而且报错文本里恰好也写着 snapshot_id，
    所以断言必须匹配那个列表本身）。"""
    full = _rows(("f1", "ph1", D1, "A00001.SZ", 1.0, 2.0, "s"))
    for col in COLS:
        with pytest.raises(ValueError, match=re.escape(f"缺列 ['{col}']")):
            derived_store.write_factor_values(full.drop(columns=[col]))


def test_write_of_an_empty_frame_writes_nothing_and_creates_no_db(store_env):
    """列齐但没有行 = 没什么可写，不是错误；也不该顺手把库建出来 ——
    `current_factor_dates` / `read` 靠「库不存在」判「第一次跑」。"""
    assert derived_store.write_factor_values(_rows()) == 0
    assert not pathlib.Path(_derived.DEFAULT_DERIVED_PATH).exists()


# ══════════════ 2 · 已算日期（build 的跳过判据）══════════════

def test_current_factor_dates_lists_only_fully_current_dates(store_env):
    """★ 判据是「这一天的每一行都属于当前快照」，不是「这一天有行」。

    D1 三行里只有一行陈旧 —— `bool_or` / `EXISTS` / `count(*) > 0` 三种写法都会把它
    误判成已算，于是那两行陈旧值永远不会被重算，而 read 会把整天当未命中：
    build 说算过了、read 说没有，这一天的因子从此静默消失。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "B00002.SZ", -1.7320508076, 0.2, snap),
        ("f1", "ph1", D1, "C00003.SH", 2.2360679775, 0.3, "snap_stale"),
        ("f1", "ph1", D2, "A00001.SZ", 0.4142135624, 0.4, snap),
        ("f1", "ph1", D2, "C00003.SH", 2.2360679775, 0.5, snap)))

    got = derived_store.current_factor_dates({"f1": "ph1"}, [D1, D2, D3])
    assert got == {("f1", D2)}
    # 没问的日期不许回答：漏掉日期过滤会让 build 以为一个从没算过的区间已经算完
    assert derived_store.current_factor_dates({"f1": "ph1"}, [D3]) == set()


def test_current_factor_dates_without_a_derived_db_is_empty(store_env):
    """第一次跑时派生库还不存在 —— 「什么都没算过」，不是异常。"""
    assert not pathlib.Path(_derived.DEFAULT_DERIVED_PATH).exists()
    assert derived_store.current_factor_dates({"f1": "ph1"}, [D1]) == set()
    out, warns = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ"])
    assert out.empty
    assert len(warns) == 1, "派生库不存在 = 每个因子都没命中，同样要出声"


# ══════════════ 3 · coverage_report ══════════════

def _coverage_fixture(snap: str) -> None:
    """D1 三行缺一个（2/3），D2 两行都在（2/2）。
    行数故意不同：逐日平均 = 0.8333、总行数占比 = 0.8，两种实现分得开。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "B00002.SZ", None, 0.0, snap),
        ("f1", "ph1", D1, "C00003.SH", 2.2360679775, 0.3, snap),
        ("f1", "ph1", D2, "A00001.SZ", 0.4142135624, 0.4, snap),
        ("f1", "ph1", D2, "C00003.SH", 2.2360679775, 0.5, snap)))


def test_coverage_report_gives_the_date_range_and_the_per_date_mean(store_env):
    snap = query.snapshot_id()
    _coverage_fixture(snap)
    derived_store.write_factor_values(_rows(
        ("f2", "ph1", D3, "A00001.SZ", 0.5772156649, 0.6, snap)))     # 问 f1 就别答 f2
    rep = derived_store.coverage_report(["f1"])

    assert len(rep) == 1
    row = rep.iloc[0]
    assert row["factor_name"] == "f1" and row["param_hash"] == "ph1"
    assert row["first_date"] == D1 and row["last_date"] == D2
    assert row["n_dates"] == 2
    assert row["mean_coverage"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_coverage_report_counts_raw_not_processed(store_env):
    """`processed_value` 末端有 fillna(0)，永远非空 —— 拿它算覆盖率恒等于 100%，
    这道指标就永远不会响。"""
    _coverage_fixture(query.snapshot_id())
    assert derived_store.coverage_report(["f1"]).iloc[0]["mean_coverage"] < 1.0


def test_coverage_report_flags_stale_dates(store_env):
    """报告里「2024-01-05 起、覆盖率 83%」而 read 一行都不给，是最容易被当成灵异事件的
    一种状态。陈旧日期数必须在同一张表上看得见。"""
    _coverage_fixture(query.snapshot_id())
    assert derived_store.coverage_report(["f1"]).iloc[0]["n_stale_dates"] == 0

    _bump_snapshot(store_env)
    rep = derived_store.coverage_report(["f1"])
    assert rep.iloc[0]["n_dates"] == 2
    assert rep.iloc[0]["n_stale_dates"] == 2


def test_coverage_report_counts_only_the_current_snapshot(store_env):
    """★ 评审 I1：`read` 只服务当前快照，而 `factor_value` 只增不减 —— 混代统计报出来的
    覆盖率**比实际能读到的高**。这里 3 行当前（1 行非空）+ 2 行非空的孤儿：
    不加快照过滤报 0.60，实际能服务的是 0.333，而 0.60 恰好是 `min_coverage` 的默认值 ——
    报告刚好过闸，能读到的横截面只有它的一半。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "B00002.SZ", None, 0.2, snap),
        ("f1", "ph1", D1, "C00003.SH", None, 0.3, snap),
        ("f1", "ph1", D1, "D00004.SZ", 2.2360679775, 0.4, "snap_orphan"),
        ("f1", "ph1", D1, "E00005.SZ", 0.5772156649, 0.5, "snap_orphan")))

    rep = derived_store.coverage_report(["f1"])
    assert rep.iloc[0]["mean_coverage"] == pytest.approx(1 / 3)
    assert rep.iloc[0]["n_stale_dates"] == 1, "这一天混着别的快照的行 → 不是「整天都当前」"


def test_coverage_report_flags_a_date_whose_rows_are_only_partly_current(store_env):
    """★ 评审 I2：`_coverage_fixture` 里一天的所有行共享同一个快照，而在同质分组上
    `bool_and ≡ bool_or` —— 那条用例根本分辨不出这个谓词（`current_factor_dates`
    有混代 fixture，`coverage_report` 带的是同一个谓词的第二份拷贝）。
    换成 bool_or：一个 build 认为没算完的日期，报告说「0 个陈旧日」，两个答案静默打架。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "B00002.SZ", 2.2360679775, 0.2, "snap_stale"),
        ("f1", "ph1", D2, "A00001.SZ", 0.5772156649, 0.3, snap)))

    rep = derived_store.coverage_report(["f1"])
    assert rep.iloc[0]["n_dates"] == 2
    assert rep.iloc[0]["n_stale_dates"] == 1
    assert derived_store.current_factor_dates({"f1": "ph1"}, [D1, D2]) == {("f1", D2)}, \
        "同一个谓词的两份拷贝必须给出同一个答案"


def test_coverage_report_ignores_a_fully_stale_date_instead_of_poisoning_the_mean(store_env):
    """整天都陈旧 → 那天的覆盖率记 NULL（`avg` 会跳过），由 `n_stale_dates` 那一列去报：
    一个是「没有当前数据」，一个是「当前数据很稀」。不记 NULL 的话分母是 0，
    而 DuckDB 的 0/0 是 **NaN 不是 NULL**（同一个坑第三次）—— avg 把 NaN 传染给整个因子，
    99 天好数据配 1 天陈旧，报出来的覆盖率是 NaN。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, "snap_stale"),
        ("f1", "ph1", D2, "A00001.SZ", 2.2360679775, 0.2, query.snapshot_id())))

    rep = derived_store.coverage_report(["f1"])
    assert rep.iloc[0]["mean_coverage"] == 1.0, "只有 D2 有当前数据，覆盖率就该是 D2 的"
    assert rep.iloc[0]["n_stale_dates"] == 1


def test_coverage_report_on_an_empty_store_has_the_columns(store_env):
    rep = derived_store.coverage_report()
    assert rep.empty
    assert list(rep.columns) == ["factor_name", "param_hash", "first_date", "last_date",
                                 "n_dates", "mean_coverage", "n_stale_dates"]


def test_coverage_report_with_no_names_or_no_matching_rows_has_the_columns(store_env):
    """`coverage_report([])` 曾经抛 `duckdb.ParserException`（`IN ()`）——
    两个姊妹函数都把空输入当「什么都没问」。「库里有行但没有一行是问的那个因子」
    是另一条路：空报告同样要带列名，否则调用方拿到一个没有形状的空表。"""
    _coverage_fixture(query.snapshot_id())
    for rep in (derived_store.coverage_report([]), derived_store.coverage_report(["nope"])):
        assert rep.empty
        assert list(rep.columns) == derived_store.COVERAGE_COLUMNS


# ══════════════ 4 · store.build ══════════════

def test_build_writes_one_row_per_universe_member_and_is_idempotent(store_env):
    """验收①：两次 build 行数不翻倍，第二次返回 0。"""
    _register("f1")
    written, warns = store.build(["f1"], [D1, D2])
    assert written == {"f1": 5}, "D1 池 3 只 + D2 池 2 只"
    assert warns == []
    assert len(_table()) == 5

    again, _ = store.build(["f1"], [D1, D2])
    assert again == {"f1": 0}
    assert len(_table()) == 5


def test_build_stores_raw_and_processed_from_the_same_call(store_env):
    """两列必须是同一个因子的原始值与处理值，不能写反、也不能两列一样。"""
    _register("f1")
    store.build(["f1"], [D1])
    t = _table().set_index("ts_code")

    assert t.loc["A00001.SZ", "raw_value"] == pytest.approx(_RAW["A00001.SZ"])
    assert t.loc["B00002.SZ", "raw_value"] == pytest.approx(_RAW["B00002.SZ"])
    raw = pd.Series({c: _RAW[c] for c in ["A00001.SZ", "B00002.SZ", "C00003.SH"]})
    expect = (raw - raw.mean()) / raw.std()
    for code, z in expect.items():
        assert t.loc[code, "processed_value"] == pytest.approx(z)
    assert t.loc[:, "snapshot_id"].unique().tolist() == [query.snapshot_id()]


def test_build_keeps_each_factors_values_on_its_own_rows(store_env):
    """宽表（因子 × 股票）摊成长表时最容易出的错：两层循环的顺序对不上，
    于是 f1 的值落到 f2 的行上。逐行断言 —— 只有一个因子的用例看不见这件事。"""
    _register("f1", bump=0.0)
    _register("f2", bump=3.1415926536)
    written, _ = store.build(["f1", "f2"], [D1])

    assert written == {"f1": 3, "f2": 3}
    t = _table().set_index(["factor_name", "ts_code"])
    for code in ("A00001.SZ", "B00002.SZ", "C00003.SH"):
        assert t.loc[("f1", code), "raw_value"] == pytest.approx(_RAW[code])
        assert t.loc[("f2", code), "raw_value"] == pytest.approx(_RAW[code] + 3.1415926536)
    assert t.loc[("f1", "A00001.SZ"), "param_hash"] == base.get_factor("f1").param_hash()
    assert t.loc[("f2", "A00001.SZ"), "param_hash"] == base.get_factor("f2").param_hash()


def test_build_after_a_snapshot_change_overwrites_in_place(store_env):
    """验收②：快照变了 → 同一批行被覆盖（不是堆两代），snapshot_id 列跟着更新。"""
    _register("f1")
    store.build(["f1"], [D1])
    old_snap = query.snapshot_id()
    assert _table()["snapshot_id"].unique().tolist() == [old_snap]

    _bump_snapshot(store_env)
    written, _ = store.build(["f1"], [D1])

    assert written == {"f1": 3}, "陈旧快照必须触发重算，overwrite=False 也一样"
    t = _table()
    assert len(t) == 3, "覆盖写，不是加一代新行"
    assert t["snapshot_id"].unique().tolist() == [query.snapshot_id()] != [old_snap]


def test_build_overwrite_recomputes_a_date_that_is_already_current(store_env):
    _register("f1")
    store.build(["f1"], [D1])
    assert store.build(["f1"], [D1])[0] == {"f1": 0}
    assert store.build(["f1"], [D1], overwrite=True)[0] == {"f1": 3}
    assert len(_table()) == 3


def test_build_deletes_rows_that_left_the_universe_and_keeps_its_skip(store_env):
    """★ 评审 I1 的第二个症状：股票池按 as_of_date 动态生成（D5），数据一修正
    （改一个上市日 / 一段 ST 区间）就可能有票退出某个**历史日期**的池子 ——
    它那行留在库里盖着旧快照，于是 `current_factor_dates` 的 bool_and 对这一天
    **永远为假**：每次跑都重算，`overwrite=False` 的跳过永久失效。
    而 read 那边一切正常（它按 universe 对齐，根本看不见那行）——
    症状只有「缓存莫名其妙不生效」，是最难往这上面想的一种。"""
    _register("f1")
    ph = base.get_factor("f1").param_hash()
    derived_store.write_factor_values(_rows(
        ("f1", ph, D1, "ZZ99999.SZ", 0.4142135624, 0.1, "snap_orphan"),
        # 另一代参数的同一只票：闸 5 的 ±30% 网格下并存多代是【正常的】，不许顺手清掉
        ("f1", "ph_other", D1, "ZZ99999.SZ", 2.2360679775, 0.2, "snap_orphan")))

    written, warns = store.build(["f1"], [D1])
    assert written == {"f1": 3}
    assert any("不在当前股票池" in w for w in warns), warns
    left = _table().set_index(["factor_name", "param_hash", "ts_code"])
    assert ("f1", ph, "ZZ99999.SZ") not in left.index
    assert ("f1", "ph_other", "ZZ99999.SZ") in left.index, "只删这次要写的那一代"
    assert store.build(["f1"], [D1])[0] == {"f1": 0}, "孤儿行清掉之后，跳过必须回来"


def test_drop_out_of_universe_refuses_an_empty_universe(store_env):
    """空池在 SQL 里是 `NOT IN ()`（删掉这一天的全部行），是这条**破坏性**路径上
    blast radius 最大的一种输入 —— 靠 DuckDB 的语法错兜底太薄。"""
    with pytest.raises(ValueError, match="为空"):
        derived_store.drop_out_of_universe({"f1": "ph1"}, D1, [])


def test_build_recomputes_when_only_the_param_hash_changed(store_env):
    """★ 评审 I3：`current_factor_dates` 按 **(factor_name, param_hash) 成对**过滤。
    只按名字过滤的话，旧参数那一代（行都在、快照是当前的）会让新参数的 build 整段跳过：
    build 说算完了 `{"f1": 0}`，read 按新哈希一行都读不到，而且**永远**停在这个状态
    （跳过 → 不写 → 还是跳过）。闸 5 的 ±30% 参数网格每换一个参数都走这条路。"""
    _register("f1", window=20)
    store.build(["f1"], [D1])
    old = base.get_factor("f1").param_hash()

    base.FACTOR_REGISTRY.pop("f1")
    _register("f1", window=26)
    new = base.get_factor("f1").param_hash()
    assert new != old, "前提：default_params 进 param_hash"

    assert store.build(["f1"], [D1])[0] == {"f1": 3}
    assert not derived_store.read_factor_values({"f1": new}, D1, ["A00001.SZ"])[0].empty
    assert len(_table()) == 6, "两代并存：闸 5 的参数高原本来就要两代都在"


def test_param_hash_covers_exactly_what_changes_the_stored_values(store_env):
    """★ 评审 I5：`neutralize` 与 `available_from` 都**改变落库的值** —— 前者决定做不做
    中性化，后者把整段历史短路成全 NaN —— 却曾经不进 `param_hash`。于是改了它们，
    build 看到「主键命中 + 快照是当前的」就跳过（0 行），read 拿回改动**之前**的值：
    一次静默的假缓存命中，长得跟缓存正常工作一模一样。
    `direction` / `min_coverage` 不进：那两个由 `combine` 在读出**之后**施加，
    库里的值一个字都不变，进哈希只会凭空多算一代缓存。

    住在本文件而不是 test_factor_base：这里守的是**缓存键的语义**（Task 8 的主键），
    哈希的书写格式（canonical JSON / isoformat）那两颗钉子在 test_factor_base 里。"""
    _register("f1")                                  # _register 默认 neutralize=False
    spec = base.get_factor("f1")
    h = spec.param_hash()
    assert dataclasses.replace(spec, neutralize=True).param_hash() != h
    assert dataclasses.replace(spec, available_from=dt.date(2030, 1, 1)).param_hash() != h
    assert dataclasses.replace(spec, direction=-1).param_hash() == h
    assert dataclasses.replace(spec, min_coverage=0.99).param_hash() == h

    store.build(["f1"], [D1])
    base.FACTOR_REGISTRY["f1"] = dataclasses.replace(spec, neutralize=True)
    assert store.build(["f1"], [D1])[0] == {"f1": 3}, "改了 neutralize 必须是真未命中，不是跳过"


def test_a_pk_hit_is_not_a_valid_cache(store_env):
    """★ 剩下唯一哈希不到的东西是**因子函数体**（本来也没法哈希）——
    同一个主键下存的可以是另一种语义的值。

    所以 build 只把主键命中当成「存在某一代」：语义变了要靠 `overwrite=True` 说出来，
    库里不会自己发现。这条用例把这个代价钉住，免得下一个人以为跳过 == 已是最新。"""
    _register("f1", bump=0.0)
    before = base.get_factor("f1").param_hash()
    store.build(["f1"], [D1])

    base.FACTOR_REGISTRY.pop("f1")
    _register("f1", bump=100.0)                     # 换了函数体，参数一个字没动
    assert base.get_factor("f1").param_hash() == before, "前提：函数体不进 param_hash"

    assert store.build(["f1"], [D1])[0] == {"f1": 0}
    stale = _table().set_index("ts_code")
    assert stale.loc["A00001.SZ", "raw_value"] == pytest.approx(_RAW["A00001.SZ"])

    assert store.build(["f1"], [D1], overwrite=True)[0] == {"f1": 3}
    fresh = _table().set_index("ts_code")
    assert fresh.loc["A00001.SZ", "raw_value"] == pytest.approx(_RAW["A00001.SZ"] + 100.0)


def test_build_refuses_factors_outside_the_alpha_whitelist(store_env, clean_registry):
    """★ `raw_value` 是 DOUBLE，而 `industry` 返回 category dtype 的字符串 ——
    无脑遍历注册表要么在它这里抛，要么更糟：静默写进一列 NULL，看起来「行业因子存在且全空」。
    `log_mv` / `beta_250` 是数值存得下，但它们是中性化的回归元不是 alpha，
    拿来当信号回测【非常好看】—— 要存必须调用方明说，不能靠遍历撞上。"""
    base.FACTOR_REGISTRY.update(clean_registry)     # 对着真注册表断言，不是对着假因子

    for name in ("industry", "log_mv", "beta_250"):
        # 匹配被点名的那一个，而不是报错文本里恰好也提到的另一个名字
        with pytest.raises(ValueError, match=re.escape(f"['{name}']")):
            store.build([name], [D1])

    with pytest.raises(ValueError, match="industry"):
        store.build(sorted(base.FACTOR_REGISTRY), [D1])      # 整表遍历
    assert not pathlib.Path(_derived.DEFAULT_DERIVED_PATH).exists(), \
        "白名单必须在算任何一个因子之前全部验完，否则前半张表已经落库了"


def test_build_surfaces_the_warnings_from_compute(store_env):
    """降级必须可见（global-constraints ★）：`available_from` 未到 / 中性化被跳过
    都会产出一条看起来完全正常的净值曲线。build 吞掉 warning，落库的行就再没有
    任何地方记着那一天是降级算出来的。"""
    _register("late", available_from=dt.date(2030, 1, 1))
    written, warns = store.build(["late"], [D1])

    assert written == {"late": 3}
    assert any("available_from" in w for w in warns), warns
    assert len(warns) == 1, "一个因子一天只该来一条；raw / processed 两遍计算不能报两份"


def test_build_writes_null_for_a_factor_that_is_not_available_yet(store_env):
    """`available_from` 之前是全 NaN，落库必须是 NULL —— 落成 0 会让 coverage_report
    报 100%，也让调用方把「没有数据」读成「中性分数」。"""
    _register("late", available_from=dt.date(2030, 1, 1))
    store.build(["late"], [D1])

    conn = _derived.connect_read(_derived.DEFAULT_DERIVED_PATH)
    try:
        assert conn.execute("SELECT count(*), count(raw_value), count(processed_value) "
                            "FROM factor_value").fetchone() == (3, 0, 0)
    finally:
        conn.close()
    assert derived_store.coverage_report(["late"]).iloc[0]["mean_coverage"] == 0.0


def test_read_after_a_partial_build_does_not_fill_in_the_missing_date(store_env):
    """验收③的行为版：因子算得出来、库里没有 → 仍然返回空。
    读缓存还是现算由调用方决定，这一层不偷偷混合。"""
    _register("f1")
    store.build(["f1"], [D1])
    ph = base.get_factor("f1").param_hash()

    assert not derived_store.read_factor_values({"f1": ph}, D1, ["A00001.SZ"])[0].empty
    assert derived_store.read_factor_values({"f1": ph}, D2, ["A00001.SZ"])[0].empty


def test_build_refuses_to_straddle_two_databases(store_env, tmp_path):
    """★ D7：build 给整批行盖【同一个】snapshot_id。跑到一半撞上 promote（os.replace：
    路径不变、inode 变），不钉住就会静默重连 —— 后半程的行是另一个数据库算出来的，
    却统一盖着开跑时那个指纹，此后永远对不出来。钉住之后换库必须抛。"""
    _register("f1")
    real = query.get_universe

    def swap_the_db_then_ask(as_of_date, **kw):
        if as_of_date == D2:
            new = str(tmp_path / "promoted.duckdb")
            shutil.copyfile(store_env, new)
            w = _db.connect_write(new)
            w.execute("INSERT INTO stock_status VALUES ('A00001.SZ', DATE '2024-02-01', NULL, 'ST')")
            w.execute("CHECKPOINT")
            w.close()
            os.replace(new, store_env)
        return real(as_of_date, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(query, "get_universe", swap_the_db_then_ask)
        with pytest.raises(query.QueryError, match="钉住"):
            store.build(["f1"], [D1, D2])


def test_build_reports_progress_per_date(store_env):
    """先把 D1 算掉：跳过的日期**同样**要报进度，否则把 progress 挪进 `if todo:`
    没有任何用例会红 —— 而一次「大部分日期都命中缓存」的重跑，进度条会卡住不动。"""
    _register("f1")
    store.build(["f1"], [D1])
    seen: list[tuple[int, int]] = []
    store.build(["f1"], [D1, D2, D3], progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_build_refuses_an_empty_factor_list(store_env):
    """与 `compute_panel`（「没有因子的面板没有意义」）同口径。静默返回 `{}` 会让
    「名字列表算错成空」的调用方看到一次成功的空跑 —— 而且顺手钉住了快照。"""
    with pytest.raises(ValueError, match="names 为空"):
        store.build([], [D1])


def test_build_over_an_empty_date_range_is_a_no_op(store_env):
    """区间里一个交易日都没有（放假一周的日历切片）是正常输入，不是错误。
    没有 `current_factor_dates` 的空输入闸，这里是 `trade_date IN ()` → ParserException。"""
    _register("f1")
    store.build(["f1"], [D1])                      # 先建库：库不存在时那句 SQL 根本不会发
    assert store.build(["f1"], [])[0] == {"f1": 0}


def test_build_counts_a_repeated_date_once(store_env):
    """`done` 在循环外算一次，所以重复日期会走两遍 todo 分支：库是对的（UPSERT 幂等），
    但 written 报双倍 —— 一个对不上库的进度数字。"""
    _register("f1")
    written, _ = store.build(["f1"], [D1, D1])
    assert written == {"f1": 3}
    assert len(_table()) == 3


def test_build_leaves_the_snapshot_pinned_until_close_db(store_env, tmp_path):
    """★ 评审 I4：正常返回之后钉子**还在**，只有 `query.close_db()` 解得开。
    长驻进程里跑完 build 不 close，下一次 nightly promote 之后每一个 A 股工具都会抛
    「请重跑」。而 `open_db()` **不是解药**：它把连接指到新文件上，于是那一次检查
    静默通过 —— 钉子还举着，你却已经在读另一个数据库了（下一次 promote 照样抛）。
    解钉是调用方的事 —— 这条用例把这个契约写下来给 Task 13 的引擎作者看。"""
    def promote():
        shutil.copyfile(store_env, str(tmp_path / "promoted.duckdb"))
        os.replace(str(tmp_path / "promoted.duckdb"), store_env)  # 路径不变，inode 变

    _register("f1")
    store.build(["f1"], [D1])

    promote()
    with pytest.raises(query.QueryError, match="钉住"):
        query.get_universe(D1)

    # ★ open_db 曾是这个钉子的后门：它把 _conn_ident 指到新 inode，于是 _conn() 的
    #   「磁盘 ident != _conn_ident」又相等了，钉住检查永远不再触发 —— 查询【静默】
    #   落到另一个数据库上而 _pinned_ident 还宣称钉着。那正是 D7 要挡的东西，
    #   且比抛错更坏（抛出来的错是特性）。现在 open_db 自己先抛。
    with pytest.raises(query.QueryError, match="钉住"):
        query.open_db(store_env)
    with pytest.raises(query.QueryError, match="钉住"):
        query.get_universe(D1)                                    # 钉子仍在

    query.close_db()
    assert query.get_universe(D1)                                 # 只有 close_db() 解钉


# ══════════════ 5 · store.read_current + combine(use_store=True) ══════════════
#
# 快路径存在的理由是数字：架构 §1.1 的规模（16 因子 × 3,100 只 × 780 周）下，
# Task 14 实测【只算中性化内核】就要 11.0 s/次回测，而闸 3 要跑 200 次置换 ——
# 置换只打乱当日的**合成分数**，因子值在 201 次运行里逐位相同，于是其中
# 2,496,000 次 `compute_factor` 是纯重算（≈ 37 分钟白烧）。
#
# 但快路径的唯一契约是「不改变任何结果」，所以本节最吃重的是那条**差分**用例：
# 同一天同一个池子，`use_store=True` 与 `False` 必须【逐位】相等（`check_exact=True`
# —— `assert_series_equal` 的默认 rtol 是 1e-5，吞得下 0.6% 的系统性偏差，
# global-constraints ★ 已经被它坑过一次）。

_RAW2 = {"A00001.SZ": -0.8414709848, "B00002.SZ": 1.7724538509,
         "C00003.SH": 0.3010299957, "D00004.SZ": -2.7182818285}


def _counting(values, calls: list):
    """记账版因子体：每次被调用记一笔，用来验「命中缓存就不再算」。"""
    def compute(as_of_date, universe, **params):
        calls.append(as_of_date)
        return pd.Series([values[c] for c in universe], index=list(universe), dtype=float)
    return compute


def _register_fn(name, fn, *, direction=1, category="price", **kw):
    kw.setdefault("neutralize", False)
    base.factor(name=name, direction=direction, category=category, lookback_days=1, **kw)(fn)
    return base.get_factor(name)


def _zs(values, codes) -> pd.Series:
    """oracle：3 只票、neutralize=False 时 process 退化成 winsorize → zscore → fillna(0)。"""
    s = pd.Series([values[c] for c in codes], index=list(codes), dtype=float)
    return ((s - s.mean()) / s.std()).fillna(0.0)


def test_read_current_hands_back_raw_and_processed_in_that_order(store_env):
    """两列写反了、或两列返回同一个东西，下游看到的都是合理的浮点数：
    raw 那一列去量覆盖率（永远满 → 闸不响），processed 那一列进合成分数（没做过 zscore）。"""
    _register_fn("f1", _counting(_RAW, []))
    store.build(["f1"], [D1])
    codes = query.get_universe(D1)

    got, warns = store.read_current({"f1": base.get_factor("f1").param_hash()}, D1, codes)

    raw, proc = got["f1"]
    assert raw.tolist() == [_RAW[c] for c in codes]
    pd.testing.assert_series_equal(proc, _zs(_RAW, codes), check_names=False, check_exact=True)
    assert warns == []


def test_combine_from_the_store_is_bit_identical_to_recomputing(store_env):
    """★ 快路径的**全部**契约：同一天同一个池子，两条路给出逐位相同的分数与 warning。

    两个因子的横截面形状不同、direction 一正一负、权重 2:1 —— 缺一样，
    「缓存路径漏乘 direction / 漏乘权重 / 只用了第一个因子」这类变异就与真实现相等。"""
    ca, cb = [], []
    _register_fn("f1", _counting(_RAW, ca))
    _register_fn("f2", _counting(_RAW2, cb), direction=-1)
    store.build(["f1", "f2"], [D1])
    codes = query.get_universe(D1)

    fresh, wf = base.combine({"f1": 2.0, "f2": 1.0}, D1, codes)
    spent = (len(ca), len(cb))
    cached, wc = base.combine({"f1": 2.0, "f2": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(cached, fresh, check_exact=True)
    assert (wc, wf) == ([], [])
    assert (len(ca), len(cb)) == spent, "命中缓存就不该再调因子函数"
    assert fresh.std() > 0, "对照前提：合成分数不是一列常数，否则任何变异都相等"


def test_a_store_hit_does_not_run_the_processing_chain(store_env, monkeypatch):
    """★ 省下来的就是这一步。`pipeline.process` = 去极值 → 横截面 OLS 中性化 → zscore，
    16 因子 × 3,100 只 × 780 周实测 11.0 s/次；闸 3 的 200 次里 99.5% 是它。
    「读了库、又照样跑一遍 process」在数字上与命中【完全一样】，只有钟表看得见。"""
    _register_fn("f1", _counting(_RAW, []))
    store.build(["f1"], [D1])
    codes = query.get_universe(D1)
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    monkeypatch.setattr(pipeline, "process",
                        lambda *a, **k: pytest.fail("命中缓存不该再跑处理链"))
    out, warns = base.combine({"f1": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert warns == []


def test_a_partial_hit_computes_the_missing_factor_instead_of_nan_filling_it(store_env):
    """★ 只 build 了 f1、合成要 f1+f2 —— 这是「给系统新增一个因子」的正常状态。

    缺的那个若跟着一起读回来，会被 `reindex(columns=...)` 物化成**一列 NaN**，
    与「这个因子那天本来就没数据」逐位相同，然后被覆盖率闸静默剔出分母：
    合成分数从两个因子变成一个，而 warning 说的是「覆盖率 0%」。
    正确的做法是现算它，结果与全现算逐位相同。"""
    ca, cb = [], []
    _register_fn("f1", _counting(_RAW, ca))
    _register_fn("f2", _counting(_RAW2, cb), direction=-1)
    store.build(["f1"], [D1])                     # 只预计算一个
    codes = query.get_universe(D1)
    fresh, wf = base.combine({"f1": 2.0, "f2": 1.0}, D1, codes)
    spent = (len(ca), len(cb))

    out, warns = base.combine({"f1": 2.0, "f2": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(out, fresh, check_exact=True)
    assert wf == []
    assert (len(ca), len(cb)) == (spent[0], spent[1] + 1), "命中的不再算，缺的现算"
    assert any("未命中" in w and "f2" in w for w in warns), warns


def test_the_store_path_still_measures_coverage_on_raw_value(store_env):
    """★ CLAUDE.md 规则 6：库里的 `processed_value` 是 `fillna(0)` 之后的 ——
    3 只票里 2 只没值的因子，raw 覆盖率 33%（该剔），processed 一列满（100%，不剔）。
    量错一列，`min_coverage` 这道闸在缓存路径上永远不响。"""
    thin = dict(_RAW, **{"B00002.SZ": float("nan"), "C00003.SH": float("nan")})
    _register_fn("f1", _counting(_RAW, []))
    _register_fn("thin", _counting(thin, []))
    store.build(["f1", "thin"], [D1])
    conn = _derived.connect_read(_derived.DEFAULT_DERIVED_PATH)
    try:
        assert conn.execute("SELECT count(raw_value), count(processed_value) FROM factor_value "
                            "WHERE factor_name = 'thin'").fetchone() == (1, 3), \
            "前提：库里 thin 的 raw 只有 1/3 非空，processed 三行都满"
    finally:
        conn.close()

    codes = query.get_universe(D1)
    out, warns = base.combine({"f1": 1.0, "thin": 1.0}, D1, codes, use_store=True)
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert any("thin" in w and "33%" in w for w in warns), warns


def test_a_date_with_one_stale_row_is_not_served_from_the_store(store_env):
    """★ 判命中的谓词是 `current_factor_dates` 的 `bool_and`（整天每一行都当前），
    不是 `read_factor_values` 自己那句 `snapshot_id = ?`。

    只靠后者的话，一行陈旧就变成【那只票读回 NaN】—— 覆盖率 2/3 仍然过闸，
    因子留在分母里，而它的 processed 那一列在那只票上是空的：一个缺了一只票的
    横截面，与「这只票今天没值」逐位相同。"""
    calls = []
    _register_fn("f1", _counting(_RAW, calls))
    store.build(["f1"], [D1])
    ph = base.get_factor("f1").param_hash()
    codes = query.get_universe(D1)
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    derived_store.write_factor_values(_rows(          # UPSERT 同主键：把其中一行改陈旧
        ("f1", ph, D1, "B00002.SZ", _RAW["B00002.SZ"], 0.0, "snap_stale")))
    calls.clear()
    out, warns = base.combine({"f1": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert len(calls) == 1, "整天作废 → 现算"
    assert any("未命中" in w for w in warns), warns


def test_a_snapshot_change_sends_the_store_path_back_to_computing(store_env):
    """换了一批数据，因子值就不是这批数据算的了。这一条与
    `test_read_treats_a_snapshot_mismatch_as_a_miss` 是同一件事在 combine 这一层的落点 ——
    不过就是回测拿另一批数据算出的因子跑出一条好看的假净值。"""
    calls = []
    _register_fn("f1", _counting(_RAW, calls))
    store.build(["f1"], [D1])
    codes = query.get_universe(D1)
    calls.clear()
    assert base.combine({"f1": 1.0}, D1, codes, use_store=True)[1] == [] and calls == [], \
        "前提：刚 build 完就该命中"

    _bump_snapshot(store_env)
    out, warns = base.combine({"f1": 1.0}, D1, codes, use_store=True)
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert len(calls) == 2, "缓存作废 → 两次调用各算一遍"
    assert any("未命中" in w for w in warns), warns


def test_the_batch_is_voided_if_the_db_moves_between_judging_and_reading(store_env, monkeypatch):
    """★ `read_current` 打三次库：判命中一次、取 raw 一次、取 processed 一次。
    中间换了库（没钉住快照的调用方撞上一次 nightly promote）→ 判命中说「在」、
    取值说「不在」。而 `read_factor_values` 未命中交回的是**带列名的空表**，
    `frame[n]` 于是是一条【长度为 0】的 Series —— 它进 `num += w·d·z` 就是整列 NaN，
    与「今天没有任何可用因子」逐位相同，而那会被 `build_targets` 判成维持上期持仓。
    整批作废、现算、出声。"""
    calls = []
    _register_fn("f1", _counting(_RAW, calls))
    store.build(["f1"], [D1])
    codes = query.get_universe(D1)
    ph = base.get_factor("f1").param_hash()
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    # 判命中的那一刻还在（替身），取值的那一刻全库已经是另一个快照了
    monkeypatch.setattr(derived_store, "current_factor_dates",
                        lambda hashes, dates: {(n, query.norm_date(d))
                                               for n in hashes for d in dates})
    derived_store.write_factor_values(_rows(
        *[("f1", ph, D1, c, _RAW[c], 0.0, "snap_promoted") for c in codes]))
    calls.clear()

    out, warns = base.combine({"f1": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert len(calls) == 1
    assert any("整批作废" in w for w in warns), warns


def test_the_store_is_skipped_when_the_universe_is_not_the_build_universe(store_env):
    """★ `processed_value` 是**横截面统计量**（去极值取中位数、中性化是横截面 OLS、
    zscore 是横截面均值方差）。同一只票在 3 只的池子里和在 2 只的池子里是两个数，
    而两个都是合理的浮点。`build` 写的是 `get_universe(d)` 那个横截面，
    所以池子对不上就整批不用 —— 否则「命中即等值」这句话是假的。"""
    calls = []
    _register_fn("f1", _counting(_RAW, calls))
    store.build(["f1"], [D1])
    ph = base.get_factor("f1").param_hash()
    sub = query.get_universe(D1)[:2]
    ref, _ = base.combine({"f1": 1.0}, D1, sub)
    stored, _ = derived_store.read_factor_values({"f1": ph}, D1, sub)
    assert not np.allclose(stored["f1"].to_numpy(), ref.to_numpy()), \
        "前提：库里那份（3 只的横截面）与子池现算的确实不同，否则这条用例分辨不出任何东西"

    calls.clear()
    out, warns = base.combine({"f1": 1.0}, D1, sub, use_store=True)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert len(calls) == 1
    assert any("股票池" in w for w in warns), warns


def test_use_store_on_a_cold_store_just_computes_and_says_so(store_env):
    """派生库还不存在 = 第一次跑，不是异常。但「你要的缓存一行都没有」必须出声：
    数字上它与命中【完全一样】，看得见的只有多花的 37 分钟。"""
    _register_fn("f1", _counting(_RAW, []))
    codes = query.get_universe(D1)
    assert not pathlib.Path(_derived.DEFAULT_DERIVED_PATH).exists()

    out, warns = base.combine({"f1": 1.0}, D1, codes, use_store=True)
    ref, _ = base.combine({"f1": 1.0}, D1, codes)

    pd.testing.assert_series_equal(out, ref, check_exact=True)
    assert any("未命中" in w for w in warns), warns


def test_the_store_path_does_not_replay_the_process_warnings(store_env):
    """★ 已知代价，写下来免得下一个人以为它被测住了：`factor_value` **没有一列**
    记着「这一天中性化被跳过了」（`store.build` 的模块头已经点过这件事）。
    于是 `use_store=True` 拿不到 `pipeline.process` 那一层的降级 warning ——
    它们只出现在【当初那次 build】的返回值里。

    值本身仍然逐位相同（跳过中性化的那一步在 build 时就已经跳过了），
    所以这不是结果分叉，是**可观测性**的分叉。要补只能给 `factor_value` 加列，
    那是另一张变更单。"""
    _register_fn("neu", _counting(_RAW, []), neutralize=True)
    codes = query.get_universe(D1)
    _, wb = store.build(["neu"], [D1])
    assert any("中性化跳过" in w for w in wb), "前提：3 只票的横截面走不进 OLS（MIN_OBS=30）"

    fresh, wf = base.combine({"neu": 1.0}, D1, codes)
    cached, wc = base.combine({"neu": 1.0}, D1, codes, use_store=True)

    pd.testing.assert_series_equal(cached, fresh, check_exact=True)
    assert any("中性化跳过" in w for w in wf)
    assert wc == [], "缓存路径拿不到那条 warning —— 它在 build 那次的返回值里"


# ══════════════ 6 · 分层方向 ══════════════

def test_derived_store_does_not_import_the_layers_above_it(store_env):
    """L1 闸只查「谁能 import duckdb」。反向依赖（data 层 import factors / backtest）
    没有静态闸看着，而它一旦成立，`ashare/data` 就不再是能被独立测试的底座。"""
    src = pathlib.Path(derived_store.__file__).read_text(encoding="utf-8")
    mods: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    assert not [m for m in mods if m.startswith(("ashare.factors", "ashare.backtest"))], mods

# 反过来的那一半（`factors/store.py` 不得 import duckdb）不在这里重测：
# `test_layering.py::test_real_codebase_passes` 对真实 ashare/ 跑 L1，已经守着了。


def test_coverage_report_uses_the_same_validity_rule_as_reads(tmp_path, monkeypatch):
    """报表口径必须与 read_factor_values 一致：一个能被正常读出来的日期不该被记成
    n_stale_dates（评审 P3）。这张报表存在的理由就是「别把快照陈旧当灵异事件」，
    它自己撒谎最坏。"""
    from ashare.data import derived_store, query
    seen: list = []
    real = query.valid_factor_snapshots
    monkeypatch.setattr(query, "valid_factor_snapshots",
                        lambda d: seen.append(d) or real(d))
    derived_store.coverage_report()
    assert seen, "coverage_report 没有走 valid_factor_snapshots —— 口径与读取脱节"
