from __future__ import annotations
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ashare.data import query
from ashare.factors import base, pipeline


@pytest.fixture(autouse=True)
def clean_registry():
    """FACTOR_REGISTRY 是模块级全局可变状态，测试之间会互相污染（重名注册会 raise）。

    就地 clear/update 而不是重新绑定：别的模块是 `from .base import FACTOR_REGISTRY`
    拿到同一个 dict 对象的引用，rebind 会让它们看着一个孤儿字典。
    开头也清空，这样 Task 4–6 的真因子被导入后本文件的断言仍然只看见自己注册的东西。

    yield 出去的是【真注册表的副本】：绝大多数用例只想要一个干净的表，但少数几条
    （combine 必须拒绝真的三个风险因子）非对着真因子断言不可，否则测的是假因子的假 category。
    """
    saved = dict(base.FACTOR_REGISTRY)
    base.FACTOR_REGISTRY.clear()
    yield saved
    base.FACTOR_REGISTRY.clear()
    base.FACTOR_REGISTRY.update(saved)


def _noop(as_of_date, universe):        # 占位因子体：本层只管注册，不算数
    return None


def _spec(name="f", **default_params):
    return base.FactorSpec(name=name, fn=_noop, direction=1, category="price",
                           lookback_days=30, default_params=default_params)


def _register(name, category="price", **kw):
    return base.factor(name=name, direction=1, category=category, lookback_days=30, **kw)(_noop)


# ── 注册 ──────────────────────────────────────────────────────────────

def test_decorator_returns_the_original_function_and_registers_it():
    """装饰器必须原样返回函数：因子模块要能脱离注册表直接单测 fn(as_of, universe)。"""
    def raw(as_of_date, universe, *, window: int = 20):
        return (as_of_date, universe, window)

    decorated = base.factor(name="rev", direction=1, category="price", lookback_days=30)(raw)

    assert decorated is raw
    assert base.get_factor("rev").fn is raw
    assert decorated(dt.date(2024, 1, 10), ["A00001.SZ"], window=5) \
        == (dt.date(2024, 1, 10), ["A00001.SZ"], 5)


def test_duplicate_name_raises_and_keeps_the_first_registration():
    """静默覆盖 = 两个因子共用一个 factor_value 缓存键，回测拿到的是另一个因子的值。"""
    def first(as_of_date, universe):
        return None

    def second(as_of_date, universe):
        return None

    base.factor(name="dup", direction=1, category="price", lookback_days=30)(first)
    with pytest.raises(ValueError, match="dup"):
        base.factor(name="dup", direction=-1, category="risk", lookback_days=5)(second)

    assert base.get_factor("dup").fn is first


def test_decorator_carries_metadata_and_default_params():
    """Task 3/7/8 读的就是这几个字段，默认值不能漂。"""
    _register("plain")
    s = base.get_factor("plain")
    assert (s.direction, s.category, s.lookback_days) == (1, "price", 30)
    assert (s.neutralize, s.available_from, s.min_coverage) == (True, None, 0.60)

    _register("north", category="flow", neutralize=False,
              available_from=dt.date(2016, 12, 5), min_coverage=0.8, window=20)
    n = base.get_factor("north")
    assert (n.neutralize, n.available_from, n.min_coverage) == (False, dt.date(2016, 12, 5), 0.8)
    assert dict(n.default_params) == {"window": 20}     # 剩余 kwargs 全进 default_params


def test_get_factor_unknown_name_raises():
    with pytest.raises(KeyError):
        base.get_factor("nope")


def test_list_factors_filters_by_category():
    _register("pa", category="price")
    _register("pb", category="price")
    _register("rc", category="risk")

    assert {s.name for s in base.list_factors("price")} == {"pa", "pb"}
    assert {s.name for s in base.list_factors()} == {"pa", "pb", "rc"}


# ── param_hash（进 factor_value 主键，格式一变全部历史因子值失联）───────────

def test_param_hash_matches_the_documented_formula():
    """sha256(name + canonical_json(params))[:12]，canonical = sort_keys + 紧凑分隔符。"""
    payload = "f" + json.dumps({"window": 20}, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    h = _spec(window=20).param_hash()
    assert h == expected
    assert len(h) == 12


def test_param_hash_override_equal_to_default_is_identical():
    """显式传一个恰好等于默认值的参数，不能造出第二代缓存 —— 同样的数据存两份。"""
    s = _spec(window=20)
    assert s.param_hash(window=20) == s.param_hash()
    assert s.param_hash(window=10) != s.param_hash()


def test_param_hash_ignores_key_order():
    """{a:1,b:2} 与 {b:2,a:1} 是同一组参数；哈希不同会把一个因子的缓存劈成两半。"""
    assert _spec(name="f", a=1, b=2).param_hash() == _spec(name="f", b=2, a=1).param_hash()
    s = _spec(name="f", a=1, b=2)
    assert s.param_hash(b=3, a=9) == s.param_hash(a=9, b=3)


def test_param_hash_serializes_date_params():
    """date 不是 JSON 原生类型；不转 isoformat 就直接 TypeError。"""
    s = _spec(start=dt.date(2016, 12, 5))
    assert s.param_hash(start=dt.date(2016, 12, 5)) == s.param_hash()
    assert s.param_hash(start=dt.date(2017, 1, 1)) != s.param_hash()


def test_direction_must_be_plus_or_minus_one():
    """direction 只在装饰器这一个入口进入系统，而每个错值都在钱的路径上静默失败：
    0 = 该因子对合成分数零贡献，2 = 双倍加权，符号反了 = 反向下注。都不报错，只产出假净值。"""
    # 只列【数值上就是错】的：0 / ±2 / None / 字符串。
    # True 与 1.0 在 Python 里 == 1，数值上就是 +1，没有腐蚀 —— 守卫不管类型洁癖，只管符号与幅度。
    for bad in (0, 2, -2, None, "+1"):
        with pytest.raises(ValueError, match="direction"):
            @base.factor(name=f"bad_{bad!r}", direction=bad, category="price", lookback_days=1)
            def _f(as_of_date, universe):
                ...
    assert base.FACTOR_REGISTRY == {}, "校验失败的因子不得留在注册表里"


def test_unserializable_default_param_fails_at_registration_not_at_first_hash():
    """不可确定性序列化的默认参数进了 param_hash 就等于主键随进程变。
    要在【注册时】炸 —— 等到 store.build 才炸的话，traceback 里已经看不出是哪个因子模块。"""
    with pytest.raises(TypeError, match="确定性序列化"):
        @base.factor(name="unserializable", direction=1, category="price", lookback_days=1,
                weird=object())
        def _f(as_of_date, universe):
            ...
    assert "unserializable" not in base.FACTOR_REGISTRY


def test_date_param_serialization_format_is_pinned():
    """只断言"同日期同哈希、异日期异哈希"是不够的：把 isoformat 换成 str() 或 toordinal()
    照样过，却会改掉每一个含日期参数的 param_hash，静默孤立掉对应的 factor_value 行。"""
    @base.factor(name="dated", direction=1, category="flow", lookback_days=1,
            start=dt.date(2016, 12, 5))
    def _f(as_of_date, universe):
        ...
    expect = hashlib.sha256(
        ("dated" + json.dumps({"start": "2016-12-05"}, sort_keys=True, separators=(",", ":")))
        .encode("utf-8")).hexdigest()[:12]
    assert base.FACTOR_REGISTRY["dated"].param_hash() == expect


# ══════════════════════════════════════════════════════════════════════════════
# Task 7：compute_factor / compute_panel / combine
#
# 本任务的失败模式没有一个会抛异常，它们只产出一条画得出来的净值曲线：
#
#   1 combine 用【白名单】排除风险因子，不是黑名单。黑名单（category != 'risk'）
#     失败开放 —— 新增一个类别、或类别名打错一个字母，都会静默变成 alpha。
#     industry 自带保护（category dtype 让 winsorize 的 median() 抛），但 log_mv 与
#     beta_250 会一路顺利穿过 process 产出看起来完全正常的分数。拿 log_mv 当 alpha
#     就是纯粹的规模押注 —— 在 A 股历史回测里【非常好看】，这正是它最危险的地方。
#   2 universe 守卫放在 compute_factor 这个唯一入口。六个量价因子今天对"重复代码"
#     的行为就不一致（2 个在 unstack 处抛、4 个静默返回重复索引），而 pipeline 的
#     横截面回归会把重复项当两只股票加权两次。一处覆盖 18 个因子。
#   3 覆盖率不足的因子从分母【剔除】并重新归一，不是当 0 参与（架构 B5）。
#     而且覆盖率必须量在 process 之【前】：process 末端的 fillna(0) 会把任何因子的
#     非空率抬成 100%，量在之后那道闸永远不会响。
# ══════════════════════════════════════════════════════════════════════════════

AS_OF = "2024-01-05"
U = [f"S{i:05d}.SZ" for i in range(1, 11)]              # 10 只：够算 zscore，又不到 MIN_OBS
V1 = pd.Series(np.arange(1.0, 11.0), index=U)           # 1..10：MAD 截不到，z 值确定
V2 = pd.Series(np.arange(10.0, 0.0, -1.0), index=U)     # 10..1：与 V1 反向


def _fn(values=None, *, calls=None):
    """占位因子体：原样返回给定 Series（默认 V1），并记录每次调用的入参。"""
    def fn(as_of_date, universe, **kw):
        if calls is not None:
            calls.append({"as_of_date": as_of_date, "universe": list(universe), **kw})
        return (V1 if values is None else values).copy()
    return fn


def _reg(name, values=None, *, category="price", direction=1, neutralize=False,
         available_from=None, min_coverage=0.60, calls=None, fn=None, **params):
    """注册一个假因子。默认 neutralize=False —— 让 process 退化成
    winsorize → zscore → fillna(0)，完全不碰 query，本文件只验【接线】，
    中性化本身由 test_factor_pipeline 单独钉死。"""
    base.factor(name=name, direction=direction, category=category, lookback_days=1,
                neutralize=neutralize, available_from=available_from,
                min_coverage=min_coverage, **params)(fn or _fn(values, calls=calls))
    return base.get_factor(name)


def _z(name, values=None):
    """oracle：直接调已被单独测过的 pipeline.process，断言 compute_factor 接的是它。"""
    out, _ = pipeline.process((V1 if values is None else values).copy(), AS_OF, U,
                              spec=base.get_factor(name))
    return out


# ── compute_factor：主路径 ────────────────────────────────────────────

def test_compute_factor_runs_the_processing_chain():
    _reg("a")
    out, warns = base.compute_factor("a", AS_OF, U)

    pd.testing.assert_series_equal(out, _z("a"), check_names=False)
    assert (out.mean(), out.std()) == (pytest.approx(0.0, abs=1e-12), pytest.approx(1.0))
    assert warns == []


def test_compute_factor_unprocessed_returns_the_raw_factor_value():
    """store 要同时落 raw_value 与 processed_value（Task 8），两者不能是同一个东西。"""
    _reg("a")
    out, _ = base.compute_factor("a", AS_OF, U, processed=False)
    pd.testing.assert_series_equal(out, V1, check_names=False)


def test_compute_factor_reindexes_the_result_onto_the_universe():
    """因子契约允许返回 universe 的【子集】（架构 §4.2 第 3 条），补齐必须在唯一入口做：
    横截面回归、覆盖率、合成分数三处都按 universe 的长度算，少几行会静默错位。
    多出来的代码同样要丢 —— 那是池外的股票，进了合成就是给不在池子里的票打分。"""
    _reg("sub", pd.Series([1.0, 2.0, 9.0], index=[U[0], U[1], "X99999.SZ"]))
    out, _ = base.compute_factor("sub", AS_OF, U, processed=False)

    assert list(out.index) == U
    assert out.notna().sum() == 2


def test_compute_factor_names_the_series_after_the_factor():
    _reg("a", fn=lambda as_of_date, universe, **kw: V1.rename("whatever"))
    out, _ = base.compute_factor("a", AS_OF, U, processed=False)
    assert out.name == "a"


# ── compute_factor：available_from 短路 ───────────────────────────────

def test_compute_factor_before_available_from_is_all_nan_and_never_calls_the_fn():
    """北向持股 2016-12-05 才有数据。此前必须是 NaN 而不是 0（0 在那个因子里是合法取值），
    而且不该白取一次数（brief）。全 NaN 本身是【静默】的，所以要有 warning 落地。"""
    calls = []
    _reg("north", available_from=dt.date(2016, 12, 5), calls=calls)
    out, warns = base.compute_factor("north", "2015-06-12", U)

    assert out.isna().all() and list(out.index) == U
    assert calls == []
    assert any("available_from" in w for w in warns)


def test_compute_factor_on_the_available_from_day_itself_computes():
    """边界：available_from 是【可得起始日】，当天就有数据。写成 <= 会白丢一天，
    而丢掉的那天在输出上和"这只票今天没值"长得一模一样。"""
    calls = []
    _reg("north", available_from=dt.date(2016, 12, 5), calls=calls)
    out, warns = base.compute_factor("north", "2016-12-05", U, processed=False)

    assert out.notna().all() and len(calls) == 1 and warns == []


@pytest.mark.parametrize("as_of", ["2015-06-12", dt.date(2015, 6, 12),
                                   dt.datetime(2015, 6, 12, 15, 0)])
def test_available_from_comparison_accepts_str_date_and_datetime(as_of):
    """as_of_date 的三种写法在本仓库都出现过。date 与 datetime 直接比较会 TypeError，
    而那个 TypeError 会在回测跑到 2016 年之前的第一天才炸。"""
    _reg("north", available_from=dt.date(2016, 12, 5))
    out, _ = base.compute_factor("north", as_of, U)
    assert out.isna().all()


# ── compute_factor：参数 ──────────────────────────────────────────────

def test_compute_factor_forwards_param_override_to_the_factor_fn():
    calls = []
    _reg("p", calls=calls, window=20)
    base.compute_factor("p", AS_OF, U, window=5)
    assert calls[0]["window"] == 5


def test_compute_factor_runs_the_declared_default_params_not_the_functions_own():
    """param_hash 哈希的是 `spec.default_params`。装饰器声明 window=5 而函数默认写 99 时，
    不传参算出来的是 99 的值，却按 5 的哈希进 factor_value 主键（Task 8）——
    缓存里于是躺着一份"参数写着 5、内容是 99"的因子值，且永远对不出来。
    声明是唯一真值：唯一入口按声明调用，两者就不可能分家。"""
    seen = {}

    def fn(as_of_date, universe, *, window=99):
        seen["window"] = window
        return V1.copy()

    _reg("mismatch", fn=fn, window=5)
    base.compute_factor("mismatch", AS_OF, U)
    assert seen["window"] == 5


# ── compute_factor：universe 守卫（一处覆盖 18 个因子）─────────────────

@pytest.mark.parametrize("bad,match", [
    ([], "为空"),
    ([U[0], U[1], U[0]], "重复"),
    ([U[0], np.nan], "字符串"),
    ([U[0], None], "字符串"),
    ([U[0], 600000], "字符串"),
])
def test_compute_factor_rejects_a_malformed_universe(bad, match):
    """★ 守卫放在唯一入口。最贵的是重复代码：pipeline 的横截面回归会把它当两只股票
    加权两次，而六个量价因子里四个会静默返回一条重复索引的 Series（另两个在 unstack 抛）。
    get_universe 今天返回 sorted unique 所以不可达 —— 但下一个调用方不一定经过它。"""
    calls = []
    _reg("a", calls=calls)
    with pytest.raises(ValueError, match=match):
        base.compute_factor("a", AS_OF, bad)
    assert calls == [], "守卫必须在调因子函数之前"


def test_universe_is_validated_before_the_available_from_shortcut():
    """两个短路的先后顺序：先验 universe。反过来的话，2016 年之前的每一天都会拿着
    一条重复索引的全 NaN Series 一路往下走，到别的因子那里才炸 —— 那时已经看不出
    是谁把 universe 传坏的。"""
    _reg("north", available_from=dt.date(2016, 12, 5))
    with pytest.raises(ValueError, match="重复"):
        base.compute_factor("north", "2015-06-12", [U[0], U[0]])


def test_compute_factor_on_an_unknown_name_raises_instead_of_returning_nan():
    """名字拼错必须是 KeyError。退化成"这个因子今天没数据"的话，combine 会把它
    自动剔出分母，一个根本不存在的因子就这样在回测里静静地不存在。"""
    with pytest.raises(KeyError):
        base.compute_factor("nope", AS_OF, U)


# ── compute_factor：warning 通道 ──────────────────────────────────────

def test_compute_factor_surfaces_the_pipeline_warnings(monkeypatch):
    """process 会降级（有效样本 < 30 → 跳过中性化并返回原值），那条 warning 必须有
    地方去。返回类型只有 Series 的话，降级就在这一层被吞掉了（global-constraints）。"""
    monkeypatch.setattr(query, "get_daily_basic",
                        lambda *a, **k: pd.DataFrame({"total_mv": pd.Series(1e9, index=U)}))
    monkeypatch.setattr(query, "get_industry", lambda *a, **k: pd.Series("IND", index=U))
    monkeypatch.setattr(query, "industry_source", lambda: "sw")
    _reg("neu", neutralize=True)

    out, warns = base.compute_factor("neu", AS_OF, U)
    assert any("中性化跳过" in w for w in warns)
    assert out.notna().all()


# ── compute_panel ─────────────────────────────────────────────────────

def test_compute_panel_columns_follow_the_names_order_not_sorted():
    _reg("z_last")
    _reg("a_first")
    df, _ = base.compute_panel(["z_last", "a_first"], AS_OF, U)

    assert list(df.columns) == ["z_last", "a_first"]
    assert list(df.index) == U


def test_compute_panel_rejects_duplicate_names():
    """重复列名让 df['a'] 返回 DataFrame 而不是 Series —— 下游每一处取值都换了类型。"""
    _reg("a")
    with pytest.raises(ValueError, match="重复"):
        base.compute_panel(["a", "a"], AS_OF, U)


def test_compute_panel_lets_a_failing_factor_raise_instead_of_dropping_the_column():
    """★ 某个因子抛异常时不能吞成一列 NaN。
    `pipeline.neutralize` 在 industry_source != 'sw' 时【故意】抛（回填的行业标签做
    中性化 = 前视污染）。吞掉就等于把一道阻断项降级成"这个因子今天没数据"，
    而 combine 会自动把没数据的因子剔出分母 —— 整条链路从此看起来完全正常。"""
    def boom(as_of_date, universe, **kw):
        raise RuntimeError("industry_source 不是 sw")

    _reg("ok")
    _reg("bad", fn=boom)
    with pytest.raises(RuntimeError, match="industry_source"):
        base.compute_panel(["ok", "bad"], AS_OF, U)


def test_compute_panel_collects_the_warnings_of_every_column():
    _reg("n1", available_from=dt.date(2016, 12, 5))
    _reg("n2", available_from=dt.date(2016, 12, 5))
    df, warns = base.compute_panel(["n1", "n2"], "2015-06-12", U)

    assert df.isna().all().all()
    assert len(warns) == 2


# ── combine：白名单（★ 失败关闭）──────────────────────────────────────

@pytest.mark.parametrize("name", ["log_mv", "industry", "beta_250"])
def test_combine_refuses_the_real_risk_factors(clean_registry, name):
    """★ 这三个【就是】pipeline.neutralize 的回归元，不是 alpha。
    对着【真的】三个 spec 断言（fixture yield 出真注册表的副本）—— 自己造一个同名
    假因子来测的话，测到的是假 category，等于什么都没测。"""
    base.FACTOR_REGISTRY[name] = clean_registry[name]
    _reg("a")
    with pytest.raises(ValueError, match="白名单"):
        base.combine({"a": 1.0, name: 1.0}, AS_OF, U)


@pytest.mark.parametrize("category", ["risk", "quality", "Price", "pricee", ""])
def test_combine_only_accepts_the_three_alpha_categories(category):
    """★ 白名单，不是黑名单。黑名单（category != 'risk'）对下面这些一路放行：
    将来新增的 'quality'、大小写写错的 'Price'、手滑多一个字母的 'pricee'、空串。
    它们会安安静静穿过 process 变成合成分数的一部分。"""
    _reg("x", category=category)
    with pytest.raises(ValueError, match="白名单"):
        base.combine({"x": 1.0}, AS_OF, U)


@pytest.mark.parametrize("category", ["price", "fundamental", "flow"])
def test_combine_accepts_every_alpha_category(category):
    """反向锚：白名单收紧过头同样是 bug，而且同样静默 —— 比如只剩 'price' 的话，
    2016 年之后的每一天都会拒绝北向因子，而回测只会显示"少了点信号"。"""
    _reg("x", category=category)
    out, _ = base.combine({"x": 1.0}, AS_OF, U)
    assert out.notna().all()


def test_the_alpha_whitelist_lives_in_combine_only():
    """反向锚：白名单**不能**提到 compute_factor / compute_panel 这一层。
    §9 的风格归因就是拿 log_mv / beta_250 去量残差还剩多少规模暴露 —— 那是检验
    「OLS 而非 WLS」那条裁决的唯一手段。把闸提上来等于把检验手段一起关掉。"""
    _reg("size", V1, category="risk")
    out, _ = base.compute_factor("size", AS_OF, U, processed=False)
    df, _ = base.compute_panel(["size"], AS_OF, U, processed=False)
    assert out.notna().all() and list(df.columns) == ["size"]


def test_combine_rejects_the_risk_factor_before_computing_anything():
    """拒绝要发生在取数之前：log_mv 一旦算出来就是一列看起来完全正常的分数。"""
    calls = []
    _reg("a", calls=calls)
    _reg("size", category="risk", calls=calls)
    with pytest.raises(ValueError, match="白名单"):
        base.combine({"a": 1.0, "size": 1.0}, AS_OF, U)
    assert calls == []


# ── combine：合成算术 ─────────────────────────────────────────────────

def test_combine_equal_weight_is_the_plain_mean_of_signed_z():
    _reg("a", V1)
    _reg("b", V2)
    out, warns = base.combine({"a": 1.0, "b": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(out, (_z("a", V1) + _z("b", V2)) / 2, check_names=False)
    assert warns == []


def test_combine_flips_the_sign_of_a_direction_minus_one_factor():
    """direction 是"值越大越好"的符号，样本内确定后冻结（规格 §6）。漏乘就是反向下注 ——
    volatility_60 / turnover_20 / max_ret_20 三个会被当成"越高越好"。"""
    _reg("hi", V1, direction=1)
    _reg("lo", V1, direction=-1)
    hi, _ = base.combine({"hi": 1.0}, AS_OF, U)
    lo, _ = base.combine({"lo": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(lo, -hi, check_names=False)


def test_combine_weights_are_proportional_not_merely_ordered():
    """{a:2, b:1} → a 的贡献恰好是 b 的两倍，分母是 Σw=3 而不是因子个数 2。"""
    _reg("a", V1)
    _reg("b", V2)
    out, _ = base.combine({"a": 2.0, "b": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(out, (2 * _z("a", V1) + _z("b", V2)) / 3, check_names=False)


@pytest.mark.parametrize("w", [0.0, -1.0, float("nan")])
def test_combine_rejects_a_non_positive_weight(w):
    """0 让分母白算一份；负权重是绕过"方向冻结"（规格 §6）的后门 —— 符号翻转要改
    direction、要留痕，不能藏在权重里；NaN 让整条合成分数变成 NaN，看起来像"没信号"。"""
    _reg("a", V1)
    with pytest.raises(ValueError, match="权重"):
        base.combine({"a": w}, AS_OF, U)


def test_combine_rejects_empty_weights():
    with pytest.raises(ValueError, match="权重"):
        base.combine({}, AS_OF, U)


def test_combine_inherits_the_universe_guard():
    _reg("a", V1)
    with pytest.raises(ValueError, match="重复"):
        base.combine({"a": 1.0}, AS_OF, [U[0], U[0]])


# ── combine：剔除 + 重新归一（★ 不是填 0）────────────────────────────

def test_combine_drops_a_low_coverage_factor_and_renormalizes():
    """★ 覆盖率 50% < min_coverage 60% → 从【分母】剔除，剩下的重新归一。
    当 0 参与是静默降权：分子多一个恒等于 0 的项、分母多算一份权重（架构 B5）。"""
    half = V1.copy()
    half.iloc[5:] = np.nan                                   # 5/10 = 50%
    _reg("a", V1)
    _reg("thin", half)
    out, warns = base.combine({"a": 1.0, "thin": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(out, _z("a", V1), check_names=False)
    assert any("thin" in w for w in warns), "剔除必须看得见 —— 少一个因子在结果上看不出来"


def test_combine_keeps_a_factor_exactly_at_min_coverage():
    """边界：min_coverage 是【下限】，等于它就算够。写成 <= 会把恰好达标的因子丢掉。"""
    six = V1.copy()
    six.iloc[6:] = np.nan                                    # 6/10 = 60% == min_coverage
    _reg("a", V1)
    _reg("edge", six)
    out, warns = base.combine({"a": 1.0, "edge": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(out, (_z("a", V1) + _z("edge", six)) / 2, check_names=False)
    assert warns == []


def test_combine_measures_coverage_before_the_pipeline_fills_nan():
    """★ 覆盖率必须量在 process 之【前】。process 的最后一步是 fillna(0)，量在之后的话
    任何因子的非空率都是 100%，min_coverage 这道闸永远不会响 —— 包括 2016 年之前
    那个全 NaN 的北向因子。"""
    one = V1.copy()
    one.iloc[1:] = np.nan                                    # 1/10 = 10%
    _reg("a", V1)
    _reg("thin", one)
    processed, _ = pipeline.process(one.copy(), AS_OF, U, spec=base.get_factor("thin"))
    assert processed.notna().all(), "前提：process 末端 fillna(0) 把非空率抬成 100%"

    out, _ = base.combine({"a": 1.0, "thin": 1.0}, AS_OF, U)
    pd.testing.assert_series_equal(out, _z("a", V1), check_names=False)


def test_combine_drops_a_factor_that_is_not_available_yet_without_rescaling():
    """★ brief 的验收断言：2015 年（北向不可得）的合成分数 == 只用其余因子等权，
    【不是】2/3 缩放。填 0 参与会让 2010–2016 六年被静默降权 —— 净值曲线照样画得
    出来，读起来像"那几年策略比较钝"。"""
    _reg("a", V1)
    _reg("b", V2)
    _reg("north", available_from=dt.date(2016, 12, 5))
    old, warns = base.combine({"a": 1.0, "b": 1.0, "north": 1.0}, "2015-06-12", U)
    ref, _ = base.combine({"a": 1.0, "b": 1.0}, "2015-06-12", U)

    pd.testing.assert_series_equal(old, ref, check_names=False)
    assert any("north" in w for w in warns)


def test_combine_drops_an_empty_factor_even_when_min_coverage_is_zero():
    """min_coverage=0 不等于"空因子也算数"：一列全 NaN 经 process 会变成一列 0，
    进了分子分母就是把其余因子整体缩小 —— 正是 available_from 那条规则要防的形状。"""
    _reg("a", V1)
    _reg("empty", pd.Series(np.nan, index=U), min_coverage=0.0)
    out, _ = base.combine({"a": 1.0, "empty": 1.0}, AS_OF, U)

    pd.testing.assert_series_equal(out, _z("a", V1), check_names=False)


# ── combine：全部不可用 ───────────────────────────────────────────────

def test_combine_returns_all_nan_and_warns_when_nothing_is_usable():
    """不抛 —— 回测该日跳过调仓（brief）。"""
    _reg("n1", available_from=dt.date(2016, 12, 5))
    _reg("n2", available_from=dt.date(2016, 12, 5))
    out, warns = base.combine({"n1": 1.0, "n2": 1.0}, "2015-06-12", U)

    assert out.isna().all() and list(out.index) == U
    assert warns


def test_an_all_nan_composite_reaches_the_build_targets_outage_gate():
    """★ 全 NaN 必须【一路传到】build_targets，由那里的中断闸门判定"维持上期持仓"。
    在 combine 里就地填 0 或返回空 Series 都会绕开它：空 Series 读作【清仓】，
    而数据中断常与极端行情同期，回测里会长成"策略在暴跌前防御性离场"的漂亮假净值。"""
    from ashare.backtest.portfolio import build_targets
    from ashare.backtest.types import PortfolioConstraints

    _reg("n1", available_from=dt.date(2016, 12, 5))
    scores, _ = base.combine({"n1": 1.0}, "2015-06-12", U)
    prev = pd.Series(0.05, index=U[:4])

    w, warns = build_targets(scores, 0.8, prev, pd.Series("IND", index=U),
                             PortfolioConstraints())
    # ★ 2026-08-21：中断日的返回值是 None，不是 prev（计划 Task 10「最终口径」）。
    #   prev 是【T 日收盘】度量的，而 simulate 在 τ 开盘重算漂移 —— 把它当目标传下去
    #   会在一个明确说了「今天不调仓」的日子里，把每只票的隔夜跳空当成交易做掉。
    assert w is None
    assert any("中断" in x for x in warns)


# ══════════════ ★ 收口：注册表装配（子进程 + 两条路径）══════════════
_TOTAL_FACTORS = 18


def test_every_registered_factor_is_reachable_through_both_paths():
    """★ 装配断言必须在【子进程】里，而且要覆盖两条调用路径。

    子进程：本进程里 test_factors_price / test_factor_pipeline 早就 import 过因子模块，
    在这里断言 len == 18 的话，把 `__init__.py` 的四行 import 全删掉照样绿 ——
    把最需要保护的那件事测掉。

    两条路径：Task 6 只验了 `FACTOR_REGISTRY`（`store.build` 与直接调 `spec.fn` 走的
    那条）。`compute_factor` 是 Task 7 新开的入口，它若自己短路（内部懒加载因子模块、
    或对陌生名字返回全 NaN 而不是抛），注册表空掉时它这条路仍然"正常"，
    而 `store.build` 那条已经一个因子都拿不到了。两条一起验，并拿一个不存在的名字
    做非空对照 —— 否则"名字可达"恒真。
    """
    script = (
        "import json\n"
        "import ashare.factors as f\n"
        "from ashare.factors.base import FACTOR_REGISTRY, compute_factor\n"
        "def resolves(n):\n"
        "    try:\n"
        "        compute_factor(n, '2024-01-05', ['S00001.SZ'])\n"
        "    except KeyError:\n"
        "        return False\n"          # 名字没解析到 spec
        "    except Exception:\n"
        "        return True\n"           # 取不到数（子进程里没连库）不影响"名字可达"
        "    return True\n"
        "print(json.dumps({'direct': sorted(FACTOR_REGISTRY),\n"
        "                  'fns_callable': all(callable(s.fn) for s in FACTOR_REGISTRY.values()),\n"
        "                  'via_compute_factor': sorted(n for n in FACTOR_REGISTRY if resolves(n)),\n"
        "                  'unknown_name_is_keyerror': not resolves('__no_such_factor__')}))\n")
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(root),
                          capture_output=True, text=True)

    assert proc.returncode == 0, f"干净解释器里就失败了：\n{proc.stderr}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(got["direct"]) == _TOTAL_FACTORS, "注册表装配（__init__.py 的四行 import）"
    assert got["fns_callable"]
    assert got["via_compute_factor"] == got["direct"], "compute_factor 必须经注册表解析名字"
    assert got["unknown_name_is_keyerror"], "对照项：陌生名字必须抛 KeyError，否则上一条恒真"
