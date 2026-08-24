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
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

DateLike = Union[str, _dt.date]


def _json_default(o: Any) -> str:
    if isinstance(o, _dt.date):           # date / datetime 都有 isoformat
        return o.isoformat()
    # 不可确定性序列化的参数进了主键，等于主键随进程变 —— register 会试算一次，注册时就炸
    raise TypeError(f"因子参数类型 {type(o).__name__} 无法确定性序列化，不能进 param_hash")


def _canon(d: Mapping[str, Any]) -> str:
    """canonical JSON：键排序 + 紧凑分隔符（见模块头 —— 换个书写顺序不能变成两代缓存）。"""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=_json_default)


# ★ 除 default_params 之外还要进 param_hash 的 FactorSpec 字段（2026-08-22 评审 I5）。
#   判据是「它改不改【落库的值】」：
#     · neutralize —— 决定 pipeline 做不做中性化，processed_value 直接不同；
#     · available_from —— 把它之前的整段历史短路成全 NaN，raw/processed 两列都不同。
#   不收 direction / min_coverage：那两个由 `combine` 在读出【之后】施加，库里的值一模一样，
#   进哈希只会凭空多算一代缓存（同一份数据存两份）。
#   函数体仍然哈希不到 —— 那是 `overwrite=True` 存在的理由，见 `store.build`。
_HASHED_SPEC_FIELDS = ("neutralize", "available_from")


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
        """`sha256(name + canonical_json(default_params | override) + canonical_json(非默认的
        _HASHED_SPEC_FIELDS))[:12]`。

        override 里恰好等于默认值的参数不会产生新哈希 —— 否则显式写出默认值
        就凭空多一代缓存，同一份数据算两遍存两份。

        ★ `neutralize` / `available_from` 也在里面（2026-08-22 评审 I5）：两者都**改变
          落库的值**，却曾经不进哈希 —— 于是把 `neutralize` 从 False 改成 True，`build`
          看到「主键命中 + 快照是当前的」直接跳过（返回 0 行），`read` 拿回的是中性化
          **之前**的 z-score。一次静默的假缓存命中，而它长得跟缓存正常工作一模一样。
          进了哈希，同一件事变成一次真正的未命中。
        ★ 只哈希【偏离 dataclass 默认值】的字段：一个从没写过这两个字段的因子，哈希与
          加这段之前**逐位相同**。否则「多哈希一个字段」这个动作本身，就会让全库历史
          因子值集体失联（主键变了），而它们其实一个都没变。
        """
        extra = {f.name: getattr(self, f.name) for f in fields(self)
                 if f.name in _HASHED_SPEC_FIELDS and getattr(self, f.name) != f.default}
        payload = self.name + _canon({**self.default_params, **override}) \
            + (_canon(extra) if extra else "")
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
        if direction not in (1, -1):
            raise ValueError(f"因子 {name!r} 的 direction 必须是 +1 或 -1，实际 {direction!r}")
        # ★ category 自 Task 7 起也在钱的路径上 —— 它是 combine 的 alpha 白名单的钥匙。
        #   枚举闸只拦【拼写】（'pricee' / 'Price' / ''），而且拦在写错的那一行上。
        #   它拦不住【语义】写错（一个风险因子抄成 category="price"）：拦那个的是
        #   test_factors_flow_risk 的 _EXPECTED_COUNTS —— 那条断言按 6/8/1/3 分类计数
        #   而不是只数总数 18，正是为了让"抄串类别"改变某一格的计数。
        #   ★ 不加"neutralize 必须与 category 一致"的交叉校验：一个本来就中性的 alpha
        #   因子合法地可以 neutralize=False，加了等于禁掉它。
        if category not in CATEGORIES:
            raise ValueError(f"因子 {name!r} 的 category={category!r} 不是 {CATEGORIES} 之一")
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

# 注册时的 category 枚举闸（见 `factor`）。写成"ALPHA + risk"而不是第二份字面量：
# 将来新增一类时，加进哪一边就是一次显式的"它算不算 alpha"的决定，漏一边直接注册失败。
CATEGORIES = ALPHA_CATEGORIES + ("risk",)


def _as_date(x: DateLike) -> _dt.date:
    """str / date / datetime / Timestamp 一律归一成 date。
    date 与 datetime 直接比较会 TypeError，而那一炸只在跑到 available_from 附近才发生。"""
    ts = pd.Timestamp(x)
    # ★ None / NaN → NaT，而 NaT 的比较【看对面的类型】：NaT < date 抛 TypeError，
    #   NaT < datetime 或 Timestamp 静默返回 False。available_from 标注是 date，
    #   但 datetime 是 date 的子类，哪个因子这么写都合法 —— 那一天 available_from
    #   这道短路就被静默关掉：2016 年之前照样取数、照样出一列看起来正常的值。
    if pd.isna(ts):
        raise ValueError(f"as_of_date={x!r} 不是日期（NaT）：它会让 available_from 短路时灵时不灵")
    return ts.date()


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


def _not_yet_available(spec: FactorSpec, as_of_date: DateLike) -> bool:
    """`as_of_date` 早于 `spec.available_from` —— 这条短路的**唯一定义**。

    `compute_factor` 与 `combine` 的缓存分流共用它：写成两份的话，缓存那条路会去问
    库里 2016 年之前的北向因子（一整天全 NULL，与「没算过」逐位相同），拿回一列 NaN，
    于是降级从「早于 available_from」变成「覆盖率 0%」—— 同一个剔除，换了个说不清的理由。
    ★ `available_from is None` 时【不】调 `_as_date`：那样会把 NaT 的拒绝提前到
      每一个因子上，而 `compute_factor` 现在只对声明了起始日的因子拒 NaT。
    """
    return spec.available_from is not None and _as_date(as_of_date) < spec.available_from


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
    if _not_yet_available(spec, as_of_date):
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
    """多个因子的当日横截面。返回 `(DataFrame, warnings)`；**列顺序 == 传入的 names 顺序**
    （由下面那个按 `cols` 填的 dict 决定 —— 原来还多传一个 `columns=cols`，是死参数）。

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
    return pd.DataFrame(data), warns


def combine(weights: Mapping[str, float], as_of_date: DateLike, universe: Sequence[str], *,
            use_store: bool = False) -> tuple[pd.Series, list[str]]:
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

    ── `use_store`：因子预计算缓存的快路径（架构 §1.2「净值-only < 8 s」的前提）──

    `True` 时先问 `store.read_current`，命中的因子直接用库里的 `raw_value` /
    `processed_value`，**跳过 `pipeline.process`**（去极值 → 横截面 OLS 中性化 → zscore）。
    省的就是它：架构 §1.1 的规模（16 因子 × 3,100 只 × 780 周）下光中性化内核
    实测 11.0 s/次回测，而闸 3 的 200 次置换只打乱当日的**合成分数** ——
    因子值在 201 次运行里逐位相同，其中 2,496,000 次 `compute_factor` 是纯重算。

    ★ **默认 False，而且应当继续是 False**：命中不等于「与当前实现一致」——
      `param_hash` 覆盖不到因子函数体（见 `store.read_current` 的 ★），
      一份错的缓存比一次慢的现算坏得多，而默认值恰恰是没人会显式写出来的那个选择。
      要走缓存的是闸 3 / 闸 5，它们**写得出来**这个参数。
      另一面同样实在：`read_current` 在冷库上逐次出 warning，默认开等于给每一个
      没预计算过的调用方灌 780 条噪声。
    ★ 快路径**不改变任何结果**，这是它唯一的契约（`test_factor_store.py` 有一条
      逐位相等的差分用例钉着）。所以白名单、覆盖率闸、按剩余权重重新归一
      三件事一步都不少 —— 它们是策略，只住在这里（挪进 engine 就撞架构 A5）。
    ★ 覆盖率照样量 `raw_value`：库里的 `processed_value` 是 `fillna(0)` 之后的，
      一列永远满（CLAUDE.md 规则 6，这个坑咬过三次）。
    ★ 部分命中 → 缺的那几个**现算**，不整批作废（一个因子忘了 build，不该让另外
      15 个也回去重算），但缺了谁必须落进 warning：结果按契约就该与全现算一样，
      未命中在**数字上完全看不出来**，看得见的只有多花的那 37 分钟。
    ★ 已知代价：`factor_value` 没有一列记着「那一天中性化被跳过了」，所以命中的因子
      **不会重放** `pipeline.process` 那层的降级 warning —— 它们只在【当初那次
      `build`】的返回值里。要补只能给表加列，那是另一张变更单。
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

    # universe 守卫【不在这里再调一遍】：weights 非空（上面已拦）→ 下面的循环至少走一趟
    # → compute_factor 一定先验一次，报的是同一条错。原来这里也调 _checked_universe，
    # 理由写的是"全部因子不可用时还要造出正确 index 的 NaN Series"—— 那条路只在至少
    # 成功验过一次之后才到得了，所以那次校验是纯冗余（每个调仓日多验 18 遍）。
    codes = list(universe)
    num = pd.Series(0.0, index=codes)
    den = 0.0
    dropped: list[str] = []
    warns: list[str] = []

    cached: dict = {}
    if use_store:
        from . import store          # 与 pipeline 同理：store 在模块顶层 import base，会成环
        # `available_from` 未到的因子不问库（理由见 `_not_yet_available`）
        want = {n: s.param_hash() for n, s in specs.items()
                if not _not_yet_available(s, as_of_date)}
        cached, wc = store.read_current(want, as_of_date, codes)
        warns += wc
        missed = [n for n in want if n not in cached]
        if missed:
            warns.append(f"{as_of_date} 因子缓存未命中 {len(missed)}/{len(want)} 个，改为现算："
                         f"{', '.join(missed)}（结果不变，代价是时间）")

    for n, spec in specs.items():
        hit = cached.get(n)
        if hit is None:
            raw, w1 = compute_factor(n, as_of_date, codes, processed=False)
            warns += w1
        else:
            raw = hit[0]
        # 用 isfinite 而不是 notna：±inf 也算"没有值"，与 build_targets 的覆盖率闸同一口径
        # （portfolio.py「±inf 与 NaN 同等处理」）。notna 会把 inf 数进覆盖率，而 inf 一旦
        # 撞上 MAD==0（稀疏因子常见，winsorize 原样返回）就是 mean=inf/std=nan → zscore 全 NaN
        # → fillna(0) 变成一列恒 0：留在分母里把其余因子整体缩小，还一条 warning 都不出。
        cov = float(np.isfinite(raw.to_numpy(dtype=float)).sum()) / len(codes)
        # cov == 0 与 min_coverage 无关地剔除：min_coverage=0 不等于"空因子也算数"，
        # 一列全 NaN 经 process 会变成一列 0，进了分子分母就是把其余因子整体缩小。
        if cov == 0.0 or cov < spec.min_coverage:
            dropped.append(f"{n}(覆盖率 {cov:.0%}，min_coverage {spec.min_coverage:.0%})")
            continue
        # ★ 这是全仓第二处调 process（另一处是 compute_factor）。combine 要在 process 之前
        #   量覆盖率，所以复用不了 compute_factor(processed=True) —— 但代价是"processed"
        #   有了两个定义：往 compute_factor 的链上加一步而漏了这里，合成分数就悄悄少一步。
        #   两行必须保持一模一样。
        if hit is None:
            z, w2 = pipeline.process(raw, as_of_date, codes, spec=spec)
            warns += w2
        else:
            z = hit[1]          # 库里的 processed_value 就是同一条链的产物（快路径的全部收益）
        num += float(weights[n]) * spec.direction * z
        den += float(weights[n])

    if dropped:
        warns.append(f"{as_of_date} 合成剔除 {len(dropped)}/{len(specs)} 个因子并按剩余权重"
                     f"重新归一：{'; '.join(dropped)}")
    if den == 0.0:
        warns.append(f"{as_of_date} 没有任何可用因子，合成分数全 NaN（该日不调仓）")
        return pd.Series(float("nan"), index=codes, name="score", dtype=float), warns
    return (num / den).rename("score"), warns
