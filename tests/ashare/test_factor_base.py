from __future__ import annotations
import datetime as dt
import hashlib
import json

import pytest

from ashare.factors import base


@pytest.fixture(autouse=True)
def clean_registry():
    """FACTOR_REGISTRY 是模块级全局可变状态，测试之间会互相污染（重名注册会 raise）。

    就地 clear/update 而不是重新绑定：别的模块是 `from .base import FACTOR_REGISTRY`
    拿到同一个 dict 对象的引用，rebind 会让它们看着一个孤儿字典。
    开头也清空，这样 Task 4–6 的真因子被导入后本文件的断言仍然只看见自己注册的东西。
    """
    saved = dict(base.FACTOR_REGISTRY)
    base.FACTOR_REGISTRY.clear()
    yield
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
