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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Union

import pandas as pd

DateLike = Union[str, _dt.date]


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


# ══════════════ 计算入口（架构 §4.2）══════════════
#
# 三个函数都返回 `(结果, warnings)`：架构里声明的返回类型只有结果，没有地方放同一份
# 文档自己要求记录的 warning（global-constraints，同一缺陷已在 build_targets / simulate
# 出现过两次）。这里会降级的地方有三处 —— 中性化被跳过、因子未到 available_from、
# 因子被剔出合成分母 —— 每一处都会产出一条看起来完全正常的净值曲线。

# ★ alpha 类别【白名单】。不是黑名单（category != 'risk'）：黑名单失败开放，
#   将来新增一个类别、或者类别名打错一个字母，都会静默变成 alpha。
#   而 FactorSpec 不加 is_alpha 字段 —— category 已经携带完全相同的信息，
#   两个真相来源迟早会打架。
ALPHA_CATEGORIES = ("price", "fundamental", "flow")


def _as_date(x: DateLike) -> _dt.date:
    """str / date / datetime / Timestamp 一律归一成 date。
    date 与 datetime 直接比较会 TypeError，而那一炸只在跑到 available_from 附近才发生。"""
    return pd.Timestamp(x).date()


def _checked_universe(universe: Sequence[str]) -> list[str]:
    """★ universe 的唯一校验点（Task 4 评审转来的裁决）。

    放在 `compute_factor` 而不是每个因子里：一处覆盖 18 个因子。六个量价因子今天
    对"含重复代码"的行为就不一致 —— 两个在 `unstack` 处抛，四个静默返回一条重复
    索引的 Series，而 `pipeline.process` 的横截面回归会把重复项当两只股票加权两次。
    `get_universe` 返回 sorted unique 所以现在不可达，但下一个调用方不一定经过它。
    """
    codes = list(universe)
    if not codes:
        raise ValueError("universe 为空：算不出任何横截面，调用方要么传错了要么该跳过这一天")
    bad = sorted({repr(c) for c in codes if not isinstance(c, str)})
    if bad:
        raise ValueError(f"universe 含非字符串代码 {bad[:5]}（NaN / None 都在此列）："
                         f"它们会在 reindex 后变成一行永远取不到值的 NaN")
    if len(set(codes)) != len(codes):
        dup = sorted(c for c, n in Counter(codes).items() if n > 1)
        raise ValueError(f"universe 含重复代码 {dup[:5]}：横截面回归会把它当两只股票加权两次")
    return codes


def compute_factor(name: str, as_of_date: DateLike, universe: Sequence[str], *,
                   processed: bool = True, **param_override) -> tuple[pd.Series, list[str]]:
    """算一个因子。返回 `(Series, warnings)`，index 恒等于 `universe`（顺序也一样）。

    `processed=True` 走 `pipeline.process`（去极值 → 中性化 → zscore → fillna(0)）；
    `processed=False` 给出原始值，`store` 两列都要落。

    ★ `available_from` 之前直接返回全 NaN，**不调因子函数**（省一次取数），也**不填 0**：
      0 在 `north_hold_chg_20` 里是合法取值（"持股比例没变"），填 0 会让 `combine`
      把它当成可用因子计入分母而分子恒为常数 —— 2010–2016 六年被静默降权。
    ★ 参数按 `spec.default_params` 调用，而不是靠因子函数自己的默认值：
      `param_hash` 哈希的是前者，两者分家就等于缓存里存着一份"参数写着 A、内容是 B"
      的因子值（Task 8 的主键），且永远对不出来。
    ★ 因子抛异常时**原样抛出**，不吞成全 NaN：`pipeline.neutralize` 在
      `industry_source != 'sw'` 时是故意抛的（回填的行业标签做中性化 = 前视污染），
      吞掉就把一道阻断项降级成"这个因子今天没数据"，而那种因子会被 combine 自动
      剔出分母 —— 整条链路从此看起来完全正常。
    """
    # 循环依赖：pipeline 要 base.FactorSpec，只能在调用时取（放模块顶会 ImportError）
    from . import pipeline

    spec = get_factor(name)
    codes = _checked_universe(universe)      # ★ 顺序：先验 universe，再判 available_from
    if spec.available_from is not None and _as_date(as_of_date) < spec.available_from:
        return (pd.Series(float("nan"), index=codes, name=name, dtype=float),
                [f"{as_of_date} {name}: 早于 available_from={spec.available_from}，"
                 f"返回全 NaN（不取数、不填 0）"])

    raw = spec.fn(as_of_date, codes,
                  **{**spec.default_params, **param_override}).reindex(codes).rename(name)
    if not processed:
        return raw, []
    return pipeline.process(raw, as_of_date, codes, spec=spec)


def compute_panel(names: Sequence[str], as_of_date: DateLike, universe: Sequence[str], *,
                  processed: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """多个因子的当日横截面。返回 `(DataFrame, warnings)`；**列顺序 == 传入的 names 顺序**。

    逐个因子顺序算，不并行：DuckDB 的只读连接与 `query._PRELOAD` 缓存都是进程内共享
    状态，为一次能省几秒的调用引入线程是拿"回测结果可复现"换的。
    某个因子抛异常时整个调用一起抛（理由见 `compute_factor`）—— 一列静默变 NaN，
    等于让 combine 把它剔出分母，而报出来的样子是"这个因子今天没数据"。
    """
    cols = list(names)
    if not cols:
        raise ValueError("names 为空：没有因子的面板没有意义")
    dup = sorted({n for n in cols if cols.count(n) > 1})
    if dup:
        raise ValueError(f"names 含重复因子 {dup}：重复列名会让 df[name] 返回 DataFrame 而不是 Series")

    data: dict = {}
    warns: list[str] = []
    for n in cols:
        data[n], w = compute_factor(n, as_of_date, universe, processed=processed)
        warns += w
    return pd.DataFrame(data, columns=cols), warns


def combine(weights: Mapping[str, float], as_of_date: DateLike, universe: Sequence[str]
            ) -> tuple[pd.Series, list[str]]:
    """合成分数 `Σ wᵢ·dᵢ·zᵢ / Σ wᵢ`（规格 §6 默认等权，即全部 w = 1.0）。

    ★ 只接受 `category ∈ ALPHA_CATEGORIES` 的因子，其余一律拒绝（见上面的白名单说明）。
      `industry` 自带保护（category dtype 让 `winsorize_mad` 的 `median()` 抛 TypeError），
      但 `log_mv` 与 `beta_250` 会一路顺利穿过 `process` 产出看起来完全正常的分数 ——
      拿 `log_mv` 当 alpha 是纯粹的规模押注，在 A 股历史回测里【非常好看】。
    ★ 覆盖率不足（非空占比 < `min_coverage`）或 `available_from` 未到的因子，
      从当日**分母中剔除**并按剩余权重重新归一，不是当 0 参与 —— 当 0 参与等于
      静默降权（架构 B5）。覆盖率量在 `process` 之【前】：`process` 末端的 `fillna(0)`
      会把任何因子的非空率抬成 100%，量在之后这道闸永远不会响。
    ★ 全部因子都不可用 → 返回全 NaN + warning，**不抛也不返回空 Series**：
      全 NaN 会被 `build_targets` 的中断闸门判成"维持上期持仓"，而空 Series 读作
      【清仓】。数据中断常与极端行情同期，回测里会长成"策略在暴跌前防御性离场"。
    """
    from . import pipeline

    if not weights:
        raise ValueError("权重表 weights 为空：一个因子都没有就没有合成分数")
    specs: dict = {}
    for n, w in weights.items():
        spec = get_factor(n)
        if spec.category not in ALPHA_CATEGORIES:
            raise ValueError(
                f"因子 {n!r} 的 category={spec.category!r} 不在 alpha 白名单 "
                f"{ALPHA_CATEGORIES} 中，不能进合成。风险因子（log_mv/industry/beta_250）"
                f"是中性化的回归元；类别名写错请改注册，不要放宽白名单")
        if not float(w) > 0:
            raise ValueError(
                f"因子 {n!r} 的权重 {w!r} 不是正数：0 让分母白算一份，负权重是绕过"
                f"「方向样本内冻结」（规格 §6）的后门 —— 要翻符号请改 direction")
        specs[n] = spec

    codes = _checked_universe(universe)
    num = pd.Series(0.0, index=codes)
    den = 0.0
    dropped: list[str] = []
    warns: list[str] = []
    for n, spec in specs.items():
        raw, w1 = compute_factor(n, as_of_date, codes, processed=False)
        warns += w1
        cov = float(raw.notna().sum()) / len(codes)
        # cov == 0 与 min_coverage 无关地剔除：min_coverage=0 不等于"空因子也算数"，
        # 一列全 NaN 经 process 会变成一列 0，进了分子分母就是把其余因子整体缩小。
        if cov == 0.0 or cov < spec.min_coverage:
            dropped.append(f"{n}(覆盖率 {cov:.0%}，min_coverage {spec.min_coverage:.0%})")
            continue
        z, w2 = pipeline.process(raw, as_of_date, codes, spec=spec)
        warns += w2
        num += float(weights[n]) * spec.direction * z
        den += float(weights[n])

    if dropped:
        warns.append(f"{as_of_date} 合成剔除 {len(dropped)}/{len(specs)} 个因子并按剩余权重"
                     f"重新归一：{'; '.join(dropped)}")
    if den == 0.0:
        warns.append(f"{as_of_date} 没有任何可用因子，合成分数全 NaN（该日不调仓）")
        return pd.Series(float("nan"), index=codes, name="score", dtype=float), warns
    return (num / den).rename("score"), warns
