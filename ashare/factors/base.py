"""因子注册层 —— `@factor` 装饰器 + `FactorSpec` 元数据 + 全局注册表。

★ 为什么是装饰器而不是抽象基类（架构 §4.2）：D2 要的签名字面量就是
  `f(as_of_date, universe)`，类方法多出的 `self` 会让 `check_ashare_layering` 的 L3 要特判；
  16 个因子没有一个需要继承共享状态。装饰器【原样返回原函数】，因子模块因此
  仍能脱离注册表被直接单测。

★ `param_hash` 是 `factor_value` 的主键列（`derived_schema.sql`），不是调试标签：
  同一组参数换个书写顺序若哈希不同，一个因子的缓存会被劈成两代 —— 回测读到哪一代
  取决于写入顺序，静默且不可复现。所以走 canonical JSON（键排序 + 紧凑分隔符）。
"""
from __future__ import annotations
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


def _json_default(o: Any) -> str:
    if isinstance(o, _dt.date):           # date / datetime 都有 isoformat
        return o.isoformat()
    # 不可确定性序列化的参数进了主键，等于主键随进程变 —— register 会试算一次，注册时就炸
    raise TypeError(f"因子参数类型 {type(o).__name__} 无法确定性序列化，不能进 param_hash")


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: Callable                        # f(as_of_date, universe, *, **params) -> pd.Series
    direction: int                      # +1 值越大越好 / -1 越小越好
    category: str                       # 'price' | 'fundamental' | 'flow' | 'risk'
    lookback_days: int                  # 声明需要多少【交易日】历史 → 驱动 preload 区间
    neutralize: bool = True             # risk 类（log_mv / industry / beta_250）设 False
    available_from: _dt.date | None = None   # 数据可得起始日（north_* = 2016-12-05）
    min_coverage: float = 0.60          # 池内非空占比低于此值 → 该日该因子作废
    default_params: Mapping[str, Any] = field(default_factory=dict)

    def param_hash(self, **override) -> str:
        """`sha256(name + canonical_json(default_params | override))[:12]`。

        override 里恰好等于默认值的参数不会产生新哈希 —— 否则显式写出默认值
        就凭空多一代缓存，同一份数据算两遍存两份。
        """
        params = {**self.default_params, **override}
        payload = self.name + json.dumps(params, sort_keys=True, separators=(",", ":"),
                                         default=_json_default)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


FACTOR_REGISTRY: dict[str, FactorSpec] = {}


def factor(*, name: str, direction: int, category: str, lookback_days: int,
           neutralize: bool = True,
           available_from: _dt.date | None = None,
           min_coverage: float = 0.60,
           **default_params) -> Callable[[Callable], Callable]:
    """注册因子，并【原样返回】被装饰的函数（不包装）。

    重名直接 raise：静默覆盖意味着两个因子共用同一个 (factor_name, param_hash) 缓存键。
    多余的关键字参数全部收进 `default_params`，即因子的可调参数及其默认值。
    """
    def register(fn: Callable) -> Callable:
        # ★ direction 只在这里进入系统，而每个错值都在【钱的路径】上静默失败：
        #   0 让该因子对合成分数毫无贡献；2 双倍加权；符号反了就是反向下注。
        #   三者都不报错，只产出一条看起来合理的净值曲线 —— 正是 CLAUDE.md 铁律要拦的那类。
        #   只校验 direction：category 打错会在 list_factors 里明显缺人，min_coverage 越界会明显输出空，
        #   唯独 direction 是"数字被悄悄改坏而外观正常"。
        if direction not in (1, -1):
            raise ValueError(f"因子 {name!r} 的 direction 必须是 +1 或 -1，实际 {direction!r}")
        old = FACTOR_REGISTRY.get(name)
        if old is not None:
            raise ValueError(
                f"因子重名: {name!r} 已由 {old.fn.__module__}.{old.fn.__qualname__} 注册")
        spec = FactorSpec(
            name=name, fn=fn, direction=direction, category=category,
            lookback_days=lookback_days, neutralize=neutralize,
            available_from=available_from, min_coverage=min_coverage,
            default_params=default_params)
        spec.param_hash()   # 试算一次：不可序列化的默认参数在【注册时】炸，而不是等到 store.build
        FACTOR_REGISTRY[name] = spec
        return fn
    return register


def get_factor(name: str) -> FactorSpec:
    """取因子元数据。名字拼错直接 KeyError，不退化成"这个因子没有数据"。"""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"未注册的因子 {name!r}；已注册 {sorted(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


def list_factors(category: str | None = None) -> list[FactorSpec]:
    """按注册顺序返回；`category=None` 返回全部。"""
    return [s for s in FACTOR_REGISTRY.values() if category is None or s.category == category]
