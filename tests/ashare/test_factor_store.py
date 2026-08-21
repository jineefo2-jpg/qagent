"""Task 8：因子落库 —— `ashare/data/derived_store.py`（L1 持 duckdb）
+ `ashare/factors/store.py`（L3 编排「算 → 写」）。

本文件守的不是「能存能取」，是**取回来的因子值确实是当前这批数据算出来的**：

  · `read` 按 `snapshot_id` 校验命中，不等的行【当未命中】—— Task 1 把 snapshot_id 定为
    列而非主键的直接代价。不校验就会把另一批数据算出的因子值静默喂进回测，
    产出一条好看的假净值曲线（架构 B4）。这是本文件最吃重的一条断言。
  · 主键命中 ≠ 缓存有效：`param_hash` 只哈希 `default_params`，`neutralize` / `direction` /
    **因子函数体**都不在里面（函数体本来也没法哈希）。所以判定有效性只能靠
    `snapshot_id` + `overwrite`，不能靠「这个主键下有行」。
  · `read` 未命中返回空，**不静默现算** —— 现算与落库的口径分歧是最难查的一类 bug。
  · `build` 不遍历 `FACTOR_REGISTRY`：`raw_value` 是 DOUBLE，而 `industry` 返回字符串。

fixture 的数值一律取除不尽的值（global-constraints ★）：整齐的数字会让「raw 与 processed
写反了」「zscore 没跑」这类变异与真实现逐位相同。
"""
from __future__ import annotations
import ast
import datetime as dt
import os
import pathlib
import re
import shutil

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, _derived, derived_store, query
from ashare.factors import base, store

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
    def compute(as_of_date, universe):
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

    raw = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"],
                                           processed=False)
    proc = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"])
    assert list(raw.columns) == ["f1"] and raw.index.tolist() == ["A00001.SZ", "B00002.SZ"]
    assert raw["f1"].tolist() == [0.4142135624, -1.7320508076]
    assert proc["f1"].tolist() == [-0.7071067812, 1.4142135624]


def test_read_returns_an_empty_frame_with_the_requested_columns_on_a_miss(store_env):
    """未命中返回空，**不现算** —— 现算与落库的口径分歧是最难查的一类 bug，
    要不要补算由调用方决定。列名必须在（Q3），否则调用方拿到的是一个没有形状的空表。"""
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.0, query.snapshot_id())))

    out = derived_store.read_factor_values({"f1": "ph1"}, D2, ["A00001.SZ", "C00003.SH"])
    assert out.empty, "另一个交易日没落过库，必须返回空而不是就地算一份"
    assert list(out.columns) == ["f1"]


def test_read_treats_a_snapshot_mismatch_as_a_miss(store_env):
    """★ 本文件最吃重的一条：这条不过 = 回测会拿到用另一批数据算出的因子值。

    Task 1 把 snapshot_id 定为列而非主键，代价就是命中判定必须由读取方补上。"""
    stale = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, -0.7071067812, stale),
        ("f1", "ph1", D1, "B00002.SZ", -1.7320508076, 1.4142135624, stale)))
    assert not derived_store.read_factor_values({"f1": "ph1"}, D1,
                                                ["A00001.SZ", "B00002.SZ"]).empty

    _bump_snapshot(store_env)
    assert query.snapshot_id() != stale, "前提没成立：这次改动没有改变数据指纹"

    out = derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ", "B00002.SZ"])
    assert out.empty, "陈旧快照的因子值必须当未命中，而不是照样返回"
    assert list(out.columns) == ["f1"]
    assert len(_table()) == 2, "read 不负责删行 —— 陈旧行留在库里等 build 覆盖"


def test_read_matches_param_hash_not_just_the_factor_name(store_env):
    """闸 5 的参数高原会让同一个因子名下并存多代（window=20 / 26）。
    只按名字读会同时拿到两代 → (ts_code, factor_name) 重复，pivot 要么炸要么静默留一代。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph20", D1, "A00001.SZ", 0.4142135624, -0.7071067812, snap),
        ("f1", "ph26", D1, "A00001.SZ", 2.2360679775, 1.7320508076, snap)))

    out = derived_store.read_factor_values({"f1": "ph26"}, D1, ["A00001.SZ"], processed=False)
    assert out["f1"].tolist() == [2.2360679775]
    assert len(out) == 1


def test_read_reindexes_onto_the_requested_universe(store_env):
    """index 必须逐位等于 universe（compute_factor 的契约）：
    多出来的票要丢掉，少的票补 NaN —— 否则调用方的 score / 权重会与池错位。"""
    snap = query.snapshot_id()
    derived_store.write_factor_values(_rows(
        ("f1", "ph1", D1, "A00001.SZ", 0.4142135624, 0.1, snap),
        ("f1", "ph1", D1, "D00004.SZ", 0.5772156649, 0.2, snap)))

    out = derived_store.read_factor_values({"f1": "ph1"}, D1,
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
    七列一起在入口验，报缺哪一列。"""
    bad = _rows(("f1", "ph1", D1, "A00001.SZ", 1.0, 2.0, "s")).drop(columns=["snapshot_id"])
    with pytest.raises(ValueError, match="snapshot_id"):
        derived_store.write_factor_values(bad)


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
    assert derived_store.read_factor_values({"f1": "ph1"}, D1, ["A00001.SZ"]).empty


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


def test_coverage_report_on_an_empty_store_has_the_columns(store_env):
    rep = derived_store.coverage_report()
    assert rep.empty
    assert list(rep.columns) == ["factor_name", "param_hash", "first_date", "last_date",
                                 "n_dates", "mean_coverage", "n_stale_dates"]


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


def test_a_pk_hit_is_not_a_valid_cache(store_env):
    """★ `param_hash` 只哈希 name + default_params。函数体、`neutralize`、`direction`
    都不在里面（函数体本来也没法哈希）—— 同一个主键下存的可以是另一种语义的值。

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

    assert not derived_store.read_factor_values({"f1": ph}, D1, ["A00001.SZ"]).empty
    assert derived_store.read_factor_values({"f1": ph}, D2, ["A00001.SZ"]).empty


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
    _register("f1")
    seen: list[tuple[int, int]] = []
    store.build(["f1"], [D1, D2, D3], progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


# ══════════════ 5 · 分层方向 ══════════════

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
