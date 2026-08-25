"""防自欺五闸（算法说明书 §8 / 架构 §4.3）—— 判断一条净值曲线算不算数。

前面十三个模块负责**产出**一条净值曲线，本文件负责判断那条曲线是不是自欺。
所以这里的默认取向与别处相反：**拿不准就判不过**。
一道该拦没拦的闸比没有这道闸更坏 —— 它把一条烂策略洗成「走过严格流程」的结论，
而报告上看不出任何异常。

下面每一处「看起来像成功的失败」都有对应的守卫和用例：

  · 样本内 Sharpe ≤ 0 时，`SR_oos ≥ 0.6·SR_is` 是个**负**门槛，−0.2 会被判成"守住了 60%"；
  · 真实 Sharpe 是 NaN 时，`SR_b ≥ NaN` 恒为假 → `p = 1/(n+1)` → 噪声被判**显著**；
  · 置换对照里的 NaN 若不计入分子，p 被系统性压小，方向恒指向"显著"；
  · 邻域平均 Sharpe ≤ 0 时 PeakRatio 是负数，`< 1.3` 恒真 → 孤峰满分过闸；
  · θ* 混进自己的邻域会把均值朝 SR* 拉高 → PeakRatio 变小 → 尖峰洗成高原；
  · θ̂ 逐折不动而测试窗净值一路阴跌 —— 「参数稳定」洗白一个只在拟合窗内成立的策略
    （2026-08-24 评审 C1：闸 2 判业绩不判参数离散度）；
  · ⚠ 级告警是引擎**最后**追加的那条，普通告警上百条时一刀切的封顶恰好把它吃掉（C2）；
  · 邻域里算不出的点被剔出均值 → 分母变小 → 尖峰借势把自己洗成高原（C3）。

★ 1【`engine_version` 绝不进 D7 指纹】（§8 闸 1）指纹是 `(param_hash, data_snapshot_id)`。
  哈希在两个方向上都会杀死这道闸：**撞车**（两组参数一个指纹）把新实验读成重放；
  **分家**（一组参数两个指纹）把重放读成新实验，于是**又发一次样本外机会**。
  `engine_version` 塞进哈希正是后者 —— 每次引擎升级白送一次样本外。
  本文件因此只**转述** `BacktestResult.param_hash`，一个字节都不加工。

★ 2【闸 2 / 闸 5 钉死在样本内】滚动选参与 ±30% 网格扫描都是**调参**，在 2020 年之后
  做就是拿样本外调参（D7 要挡的正是这件事）。顺带解决台账污染：`append_oos_run` 只
  跳过 `shuffle_seed` 非 None 的置换对照，闸 2 的 5 折 × g 个候选若跑到样本外，
  会往 `docs/oos-runs.md` 里写 5g 行，把真正的那几行埋掉。
  钉住之后闸 2 / 闸 5 一行台账都不写，`run_all_gates` 全跑一遍只留 3 行
  （闸 1 的样本外、闸 3 的基线、闸 4 的加压），三者互不同指纹，不会误报"重复指纹"。

★ 3【所有闸一律 `compute_diagnostics=False`】架构 A3 只对闸 3 点名（200 × 60 s = 3.3 h），
  但闸 2 的 5 折 × 9 点 = 45 次同样会让这道闸没人跑。合法性来自 §4.3 裁决 ④：
  `compute_diagnostics` 只**新增** ic/layers/attribution，不改动 `metrics` 里的任何一个数，
  也不进 `param_hash`。五个闸读的全是 `metrics` 里的标量，关掉它一个数都不会变。

★ 4【`run_all_gates` 绝不提前退出，未跑的闸必须现身】（U5）操作员要一次看到全貌。
  某个闸抛异常 → `passed=False` + 异常原文进 note，**不是**静默消失；
  没被 `gates=` 选中 → `passed=False, note='未运行'`。悄悄不返回读起来就是"没问题"。

★ 5【闸 3 的置换由引擎做，本文件只发 seed】`engine.run_backtest` 在每个调仓日
  **同日横截面内**打乱合成分数（`shuffle_seed`）。跨时间置换会破坏市场整体涨跌的
  时序结构、对照组分布失真、检验结论无效。本文件因此除了 `shuffle_seed` 之外
  一个字段都不动 —— 改了别的，对照组就不再是同一个策略的零假设样本。

★ 6【±30% 网格必然包含 θ* 自己】`BacktestConfig.param_hash` **不**归一
  「override 恰好等于注册默认值」那一侧（types.py 自己写了理由：本层不查因子注册表）。
  于是 `{"mom":{"window":60}}` 与 `{}` 在 60 就是默认值时是两个指纹、同一次回测。
  本文件的 `_with_param` 在这一层做归一：等于注册默认值的 override **删掉不写**。
  θ* 的识别因此退化成一次 `param_hash` 比较，同时不会凭空造出"分家"的指纹。

已知边界（有意的）：
  · 闸 2 的判据是**拼接样本外的业绩**（`≥ 0.6 × 闸 1 的样本内 Sharpe`，2026-08-24 裁决）；
    θ̂ 的离散度（`theta_spread` / `theta_flips`）只进 detail、**不设阈值** ——
    上一版实现的 `max/min > 3.0` 是编出来的数，规格已裁掉，这个判据不许再长回来。
  · 网格只扫**因子关键字参数**与 `PortfolioConstraints` 字段；扫不到 `cost.*`
    （成本是闸 4 的事）与 `position_cap`（择时属 P3）。
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..factors.base import get_factor
from .engine import run_backtest
from .metrics import compute as _metrics_compute
from .store import OOS_CUTOFF
from .types import BacktestConfig, PortfolioConstraints

_ONE_DAY = _dt.timedelta(days=1)

# §8 的数值判据。都写成常量是为了让"这个阈值从哪来"只有一个答案。
_OOS_SR_RATIO = 0.6            # 闸 1 与闸 2 共用：SR_oos ≥ 0.6 × SR_is（闸 2 复用，不新造数字）
_SHUFFLE_ALPHA = 0.05          # 闸 3：p < 0.05
_PEAK_RATIO_MAX = 1.3          # 闸 5：PeakRatio < 1.3

_MAX_WARNINGS = 20             # detail['warnings'] 的条数上限（闸 3 有 200 次运行）
NOT_RUN = "未运行"
GATE_NAMES = ("gate1", "gate2", "gate3", "gate4", "gate5")
_CONSTRAINT_FIELDS = tuple(f.name for f in _dc.fields(PortfolioConstraints))


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict
    note: str


# ══════════════ 公共小工具 ══════════════

def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bare(cfg: BacktestConfig, **kw) -> BacktestConfig:
    """闸内部跑回测统一用它：强制关诊断（模块头 ★3），再按需覆写字段。"""
    return _dc.replace(cfg, compute_diagnostics=False, **kw)


def _sharpe(res) -> float:
    v = res.metrics.get("sharpe")
    return float("nan") if v is None else float(v)


def _uniq(warns: Sequence[str]) -> list:
    """去重保序 + 封顶。两条规矩：**被砍掉多少条要说出来**（悄悄截断等于降级不可见），
    **⚠ 打头的一条都不许砍**（2026-08-24 评审 C2）—— 引擎把「⚠ D7 重复指纹」追加在
    告警列表**末尾**（`append_oos_run` 是 `run_backtest` 的最后一步），而 `combine` /
    `build_targets` 的普通告警逐日带文本、动辄上百条：一刀切的封顶砍掉的恰好就是那条 ⚠。
    闸判过而它不见了，等于把污染洗掉 —— 正是模块头说的那种最坏形状。"""
    out: list = []
    seen: set = set()
    for w in warns:
        if w not in seen:
            seen.add(w)
            out.append(w)
    rest = [w for w in out if not w.lstrip().startswith("⚠")]
    if len(rest) <= _MAX_WARNINGS:
        return out
    flagged = [w for w in out if w.lstrip().startswith("⚠")]
    return flagged + rest[:_MAX_WARNINGS] + [
        f"…另有 {len(rest) - _MAX_WARNINGS} 条去重后的告警未列出（⚠ 级不参与封顶，已全数保留）"]


def _note(text: str, warns: Sequence[str]) -> str:
    """底层运行喊了 ⚠（比如「D7 重复指纹」）就顶到 note 上。
    闸判过而这条不见了，等于把污染洗掉 —— 而 note 是操作员唯一必看的那一行。
    这里的计数等于截断**前**的 ⚠ 条数：`_uniq` 保证 ⚠ 级永不被封顶砍掉。"""
    n = sum(1 for w in warns if w.lstrip().startswith("⚠"))
    return f"⚠ 底层运行有 {n} 条重大告警（见 detail.warnings） · {text}" if n else text


def _stamp(cfg: BacktestConfig, res) -> dict:
    """一次运行的身份 + 读得到的标量。`engine_version` 与两个指纹**并列**（模块头 ★1）。"""
    return {"start": cfg.start.isoformat(), "end": cfg.end.isoformat(),
            "param_hash": res.param_hash, "data_snapshot_id": res.data_snapshot_id,
            "engine_version": res.engine_version,
            "cost_multiplier": cfg.cost.multiplier,
            "sharpe": _num(res.metrics.get("sharpe")),
            "information_ratio": _num(res.metrics.get("information_ratio"))}


# ── 参数网格（闸 2 选参 / 闸 5 邻域共用）───────────────────────────────────────

def _check_key(cfg: BacktestConfig, key: str) -> None:
    """键写错 = 网格扫的是空气，而闸照样给出一个漂亮的 PeakRatio。必须当场拒绝。"""
    if "." in key:
        name, param = key.split(".", 1)
        if name not in {n for n, _ in cfg.factors}:
            raise ValueError(f"网格键 {key!r} 的因子 {name!r} 不在 config.factors 里 —— "
                             f"扫一个不参与合成的参数，网格里每个点都是同一次回测")
        params = dict(get_factor(name).default_params)
        if param not in params:
            raise ValueError(f"网格键 {key!r}：因子 {name!r} 没有名为 {param!r} 的参数"
                             f"（可扫的是 {sorted(params)}）")
        return
    if key not in _CONSTRAINT_FIELDS:
        raise ValueError(f"网格键 {key!r} 既不是 '<因子名>.<参数名>'，也不是 "
                         f"PortfolioConstraints 的字段 {list(_CONSTRAINT_FIELDS)}")


def _with_param(cfg: BacktestConfig, key: str, value: Any) -> BacktestConfig:
    """把一个网格取值写进 config。等于注册默认值的 override **删掉**（模块头 ★6）。"""
    if "." not in key:
        return _dc.replace(cfg, constraints=_dc.replace(cfg.constraints, **{key: value}))
    name, param = key.split(".", 1)
    ov = {k: dict(v) for k, v in cfg.factor_param_override.items()}
    cur = ov.get(name, {})
    if value == get_factor(name).default_params.get(param):
        cur.pop(param, None)
    else:
        cur[param] = value
    if cur:
        ov[name] = cur
    else:
        ov.pop(name, None)
    return _dc.replace(cfg, factor_param_override=ov)


def _grid_points(cfg: BacktestConfig, grid: Mapping[str, Sequence]) -> list:
    """笛卡尔积展开成 `[(取值字典, config), ...]`，顺序确定（`max` 遇平局取先者）。"""
    keys = list(grid)
    for k in keys:
        _check_key(cfg, k)
        if not len(list(grid[k])):
            raise ValueError(f"网格键 {k!r} 的候选取值为空")
    out: list = []
    for combo in itertools.product(*(list(grid[k]) for k in keys)):
        point = dict(zip(keys, combo))
        c = cfg
        for k, v in point.items():
            c = _with_param(c, k, v)
        out.append((point, c))
    return out


def _label(point: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in point.items())


# ══════════════ 闸 1 · 样本外单次检验（D7 的执行点）══════════════

def gate1_out_of_sample(cfg: BacktestConfig) -> GateResult:
    """样本内 / 样本外各跑一次，判 `SR_oos ≥ 0.6 × SR_is`（§8 闸 1）。

    分界是 `store.OOS_CUTOFF`（2019-12-31）—— 与 `append_oos_run` 用**同一个常量**：
    两处各写一个"样本外"的定义，台账与闸就会各说各话。
    样本外那次运行由 `run_backtest` 自动记进 `docs/oos-runs.md`，本函数不重复记。
    """
    if cfg.end <= OOS_CUTOFF:
        return GateResult("gate1", False, {"end": cfg.end.isoformat()},
                          f"回测终点 {cfg.end} 未越过样本内边界 {OOS_CUTOFF}，"
                          f"样本外区间为空：本闸没有可判的数据（未跑）")
    if cfg.start > OOS_CUTOFF:
        return GateResult("gate1", False, {"start": cfg.start.isoformat()},
                          f"回测起点 {cfg.start} 已在样本内边界 {OOS_CUTOFF} 之后，"
                          f"样本内区间为空：0.6 倍门槛没有基准（未跑）")

    is_cfg = _bare(cfg, end=OOS_CUTOFF)
    oos_cfg = _bare(cfg, start=OOS_CUTOFF + _ONE_DAY)
    is_res = run_backtest(is_cfg)
    oos_res = run_backtest(oos_cfg)

    warns = _uniq(list(is_res.warnings) + list(oos_res.warnings))
    sr_is, sr_oos = _sharpe(is_res), _sharpe(oos_res)
    detail = {"in_sample": _stamp(is_cfg, is_res), "out_of_sample": _stamp(oos_cfg, oos_res),
              "ratio_required": _OOS_SR_RATIO, "warnings": warns}

    if not (_finite(sr_is) and _finite(sr_oos)):
        return GateResult("gate1", False, detail,
                          _note(f"Sharpe 有一侧算不出来（样本内 {sr_is}，样本外 {sr_oos}）："
                                f"算不出不等于达标", warns))
    detail["threshold"] = _OOS_SR_RATIO * sr_is
    detail["ratio"] = sr_oos / sr_is
    if not sr_is > 0:
        # 0.6 × 负数 是个负门槛：SR_oos = −0.20 会"≥ −0.30"地过闸。
        return GateResult("gate1", False, detail,
                          _note(f"样本内 Sharpe {sr_is:.4f} ≤ 0，0.6 倍门槛是个负数 —— "
                                f"样本内就不赚钱，样本外无从「保持」", warns))
    passed = sr_oos >= detail["threshold"]
    verdict = "过" if passed else "不过"
    return GateResult("gate1", passed, detail,
                      _note(f"样本外 Sharpe {sr_oos:.4f} vs 门槛 {detail['threshold']:.4f}"
                            f"（= 0.6 × {sr_is:.4f}）：{verdict}", warns))


# ══════════════ 闸 2 · Walk-forward ══════════════

def _plus_years(d: _dt.date, n: int) -> _dt.date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:                      # 2 月 29 日
        return d.replace(year=d.year + n, day=28)


def _folds(start: _dt.date, end: _dt.date, train_years: int, test_years: int) -> list:
    """`[(训练起, 训练止, 测试起, 测试止), ...]`，按 `test_years` 逐段前滚。"""
    out: list = []
    tr_s = start
    while True:
        te_s = _plus_years(tr_s, train_years)
        te_e = _plus_years(te_s, test_years) - _ONE_DAY
        if te_e > end:
            return out
        out.append((tr_s, te_s - _ONE_DAY, te_s, te_e))
        tr_s = _plus_years(tr_s, test_years)


def _theta_dispersion(grid: Mapping[str, Sequence], chosen: Sequence[Mapping]) -> "tuple[dict, dict]":
    """θ̂ 的两个离散度，只进 detail、**不设阈值**（2026-08-24 裁决）。分开报是因为它们是
    不同现象：`spread = max/min` 把单调漂移与振荡混在一起，而漂移可能是真实的 regime
    变化（A 股持有期随市场成熟而缩短，是信息不是噪声）；`flips` = 逐折差分的**变号次数**，
    才是初稿说的「反复横跳」。非数值/非正数的参数没有「量级」，两个量都报 None
    （逐折取值在 detail.folds 里看得到）。"""
    spread: dict = {}
    flips: dict = {}
    for k in grid:
        vals = [t[k] for t in chosen]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == len(vals) and nums and min(nums) > 0:
            spread[k] = max(nums) / min(nums)
            # 差分为 0 的段没有方向，不算「跳」：[60,54,60,60,54] 的变号是 2 不是 3。
            signs = [1 if b > a else -1 for a, b in zip(nums, nums[1:]) if b != a]
            flips[k] = sum(1 for s, t in zip(signs, signs[1:]) if s != t)
        else:
            spread[k] = flips[k] = None
    return spread, flips


def gate2_walk_forward(cfg: BacktestConfig, train_years: int = 5, test_years: int = 1,
                       *, grid: Optional[Mapping[str, Sequence]] = None) -> GateResult:
    """滚动训练 + 前滚测试，判**拼接样本外**的业绩（§8 闸 2，2026-08-24 裁决）：

        `SR(逐折测试窗收益按时间串联) ≥ 0.6 × SR_is（闸 1 的那一次样本内）`

    判在业绩上，不判在参数离散度上：闸 2 相对闸 1 多出来的是「换多个 regime 还成不成立」，
    参数稳不稳只是失败时的**一种解释**。θ̂ 的离散度经 `_theta_dispersion` 进 detail，
    不设阈值 —— 上一版的 `max/min > 3.0` 判据把「参数几乎不动、样本外业绩死掉」的策略
    认证成「稳定」（评审 C1 的假通过原型），已整个裁掉。

    两侧的口径（都来自裁决原文）：
      · 样本内 = 闸 1 的那一次（`start`…2019-12-31）。「拼起来的样本内」**不存在** ——
        5 年训练窗逐年重叠，串联会把 2011–2018 每年数进去多次。本函数自己再跑这一次：
        config 与闸 1 的 `is_cfg` 逐位相同（`param_hash` 相等，评审实测），结果可互换，
        将来引擎有按指纹的结果缓存时这一次自动免费。
      · 样本外 = 各折测试窗的**逐日收益**按折序串联（测试窗按构造首尾相接），
        Sharpe 用引擎同一个 `metrics.compute` 算 —— 两侧同一把尺子，0.6 才有意义。

    **泄漏就在训练/测试的接缝上**：θ̂_y 只能由**训练窗**的 Sharpe 选出。
    因子的 lookback 回看到测试窗之前是正常的（那是数据窗口），拿测试窗的结果选参数不是。
    折窗一律钉在样本内（模块头 ★2）。任何一折选不出 θ̂、或给不出测试窗净值 → 判不过：
    缺掉的折不会自己变成中性，「N 折的结论」不能由 N−k 折得出（评审 I1）。
    """
    if not grid:
        return GateResult("gate2", False, {},
                          f"{NOT_RUN}：没有参数网格。walk-forward 的产出是逐年的最优参数 θ̂_y，"
                          f"没有候选集就没有可选的东西")
    if train_years < 1 or test_years < 1:
        # test_years=0 会让 `_folds` 的前滚步长为 0、永不推进 —— 一个连 `run_all_gates`
        # 的 try/except 都接不住的挂死。当场拒绝。
        return GateResult("gate2", False,
                          {"train_years": train_years, "test_years": test_years},
                          f"{NOT_RUN}：train_years={train_years} / test_years={test_years}，"
                          f"两个窗宽都至少要 1 年（0 会让前滚永不推进）")
    try:
        points = _grid_points(cfg, grid)
    except ValueError as e:
        return GateResult("gate2", False, {}, f"参数网格无效：{e}")

    end = min(cfg.end, OOS_CUTOFF)
    folds = _folds(cfg.start, end, train_years, test_years)
    if not folds:
        return GateResult("gate2", False, {"window": [cfg.start.isoformat(), end.isoformat()]},
                          f"{NOT_RUN}：{cfg.start}~{end}（已钉在样本内）放不下一个完整的 "
                          f"{train_years}+{test_years} 年折")

    # 样本内那一次 —— 与闸 1 的 is_cfg 逐位相同。先跑它：算不出 / 不为正时，
    # 5 折 × g 个候选的折内运行全部免掉（同闸 3「真实 Sharpe 都没有就不跑对照」的先例）。
    is_cfg = _bare(cfg, end=end)
    is_res = run_backtest(is_cfg)
    runw: list = list(is_res.warnings)
    sr_is = _sharpe(is_res)
    detail: dict = {"in_sample": _stamp(is_cfg, is_res), "ratio_required": _OOS_SR_RATIO,
                    "n_folds": len(folds)}
    if not _finite(sr_is):
        detail["warnings"] = warns = _uniq(runw)
        return GateResult("gate2", False, detail,
                          _note(f"样本内 Sharpe 是 {sr_is}，算不出来：0.6 倍门槛没有基准"
                                f"（算不出 ≠ 达标）", warns))
    if not sr_is > 0:
        detail["warnings"] = warns = _uniq(runw)
        return GateResult("gate2", False, detail,
                          _note(f"样本内 Sharpe {sr_is:.4f} ≤ 0，0.6 倍门槛是个负数 —— "
                                f"样本内就不赚钱，拼接样本外无从「保持」", warns))

    own: list = []
    rows: list = []
    rets: list = []                 # 各折测试窗的逐日收益，按折序串联
    holes: list = []                # 选出了 θ̂ 却给不出测试窗收益的折
    anchor = None                   # 拼接净值的 1.0 锚点：第一段测试窗净值的第一天
    for tr_s, tr_e, te_s, te_e in folds:
        scored: list = []
        for point, pcfg in points:
            r = run_backtest(_bare(pcfg, start=tr_s, end=tr_e))
            runw += r.warnings
            scored.append((point, _sharpe(r), pcfg))
        ok = [s for s in scored if _finite(s[1])]
        if not ok:
            own.append(f"⚠ {tr_s}~{tr_e} 这一折的候选参数全都算不出 Sharpe，选不出 θ̂")
            rows.append({"train": [tr_s.isoformat(), tr_e.isoformat()],
                         "test": [te_s.isoformat(), te_e.isoformat()],
                         "theta": None, "train_sharpe": None, "test_sharpe": None})
            continue
        point, sr, pcfg = max(ok, key=lambda s: s[1])
        tr = run_backtest(_bare(pcfg, start=te_s, end=te_e))
        runw += tr.warnings
        eq = pd.Series(tr.equity, dtype=float)
        if len(eq) >= 2:
            # 逐日收益手写成 shift-除法：`pct_change` 默认 ffill，会把净值缺口桥成一段
            # 编造的收益（metrics.py 为同一个坑写过注释）。缺口在这里保持 NaN，下游有闸。
            rets.append((eq / eq.shift(1) - 1.0).iloc[1:])
            if anchor is None:
                anchor = eq.index[0]
        else:
            holes.append(f"{te_s}~{te_e}")
        rows.append({"train": [tr_s.isoformat(), tr_e.isoformat()],
                     "test": [te_s.isoformat(), te_e.isoformat()],
                     "theta": dict(point), "train_sharpe": sr,
                     "test_sharpe": _num(_sharpe(tr)), "n_test_days": int(len(eq))})

    chosen = [row["theta"] for row in rows if row["theta"] is not None]
    detail["folds"] = rows
    detail["n_chosen"] = len(chosen)
    detail["theta_spread"], detail["theta_flips"] = _theta_dispersion(grid, chosen)
    warns = _uniq(own + runw)
    detail["warnings"] = warns
    if len(chosen) < len(folds):
        # 评审 I1 重放过的假通过：2/5 折选不出参数、被悄悄丢掉，note 却说「5 折全部选参完毕」。
        return GateResult("gate2", False, detail,
                          _note(f"只有 {len(chosen)}/{len(folds)} 折选得出 θ̂：拼接样本外缺了"
                                f"选不出参数的那几折，{len(chosen)} 折说不出"
                                f"「{len(folds)} 折」的结论", warns))
    if holes:
        return GateResult("gate2", False, detail,
                          _note(f"{len(holes)}/{len(folds)} 折的测试窗净值不足两个点"
                                f"（{'; '.join(holes)}）：拼接样本外缺了这些段，判不过", warns))

    rr = pd.concat(rets)
    detail["n_oos_returns"] = int(len(rr))
    n_nan = int(rr.isna().sum())
    if n_nan:
        return GateResult("gate2", False, detail,
                          _note(f"拼接样本外的 {len(rr)} 个逐日收益里有 {n_nan} 个算不出"
                                f"（测试窗净值有缺口）：缺口不会自己变成中性，判不过", warns))
    pooled = pd.concat([pd.Series([1.0], index=[anchor]), (1.0 + rr).cumprod()])
    # 丢掉 compute 的告警是有意的：benchmark=None 是本闸的构造（只判 Sharpe，不判相对指标），
    # 「未提供基准」不是降级；而 pooled 由上面已验无 NaN 的收益构造，缺值告警不可能触发。
    m, _ = _metrics_compute(pooled, pd.DataFrame(), pd.DataFrame(), None, full=False,
                            initial_capital=cfg.initial_capital, periods_per_year=252)
    sr_pool = m.get("sharpe")
    sr_pool = float("nan") if sr_pool is None else float(sr_pool)
    detail["sharpe_oos_pooled"] = _num(sr_pool)
    if not _finite(sr_pool):
        return GateResult("gate2", False, detail,
                          _note(f"拼接样本外的 Sharpe 是 {sr_pool}，算不出来：算不出 ≠ 达标",
                                warns))
    detail["threshold"] = _OOS_SR_RATIO * sr_is
    passed = sr_pool >= detail["threshold"]
    return GateResult("gate2", passed, detail,
                      _note(f"拼接样本外 Sharpe {sr_pool:.4f} vs 门槛 {detail['threshold']:.4f}"
                            f"（= 0.6 × 样本内 {sr_is:.4f}）：{'过' if passed else '不过'}；"
                            f"θ̂ 离散度见 detail（不设阈值）", warns))


# ══════════════ 闸 3 · Shuffle 置换检验 ══════════════

def gate3_shuffle(cfg: BacktestConfig, n: int = 200, seed: int = 0) -> GateResult:
    """零假设「因子分数与未来收益独立」的置换检验（§8 闸 3）。

    `p = (1 + #{b: SR_b ≥ SR_real}) / (n + 1)`，通过标准 `p < 0.05`。
    两个 `+1` 与那个 `≥` 都是承重件：少一个 `+1` 或把 `≥` 写成 `>`，
    都会让这道闸对**每一条**被检验的策略把噪声判得更显著一点，且没有任何症状。

    置换由引擎在**同日横截面内**完成（`shuffle_seed`），本函数除了那颗种子
    一个字段都不改（模块头 ★5）；`compute_diagnostics` 强制关（★3 / 架构 A3）。
    """
    if n < 1:
        return GateResult("gate3", False, {"n": n}, f"{NOT_RUN}：n={n}，置换次数至少是 1")

    base_cfg = _bare(cfg, shuffle_seed=None)
    own: list = []
    if cfg.shuffle_seed is not None:
        # ⚠ 前缀是承重的：_note 只把 ⚠ 级顶到 note 上，闸自己喊的降级不该比引擎的低一级。
        own.append(f"⚠ 传入的 config 自带 shuffle_seed={cfg.shuffle_seed}（那已经是一次置换对照，"
                   f"不是真回测），基线已按 shuffle_seed=None 重跑")
    real = run_backtest(base_cfg)
    runw: list = list(real.warnings)
    sr_real = _sharpe(real)
    detail: dict = {"n": n, "seed": seed, "alpha": _SHUFFLE_ALPHA,
                    "sharpe_real": _num(sr_real), "p_value": None,
                    "real": _stamp(base_cfg, real)}
    if not _finite(sr_real):
        detail["warnings"] = _uniq(own + runw)
        # `SR_b ≥ NaN` 恒为假 → n_ge = 0 → p = 1/(n+1) → 一条算不出 Sharpe 的曲线
        # 会被判成"击败了全部 200 个对照"。这里必须在跑对照之前就停。
        return GateResult("gate3", False, detail,
                          _note(f"真实回测的 Sharpe 是 {sr_real}，不是有限值：置换检验没有可比的"
                                f"参照物（若继续，p 会恒为 1/(n+1) 而判显著）", detail["warnings"]))

    null: list = []
    n_ge = n_bad = 0
    for b in range(n):
        r = run_backtest(_dc.replace(base_cfg, shuffle_seed=seed + b))
        runw += r.warnings
        s = _sharpe(r)
        null.append(_num(s))
        if not _finite(s):
            # 保守方向：算不出的对照【计入】"不低于真实值"那一侧，只会把 p 推大。
            # 反过来（悄悄丢掉）方向恒指向"显著"，而那正是本闸要挡的东西。
            n_bad += 1
            n_ge += 1
        elif s >= sr_real:
            n_ge += 1

    if n_bad:
        own.append(f"⚠ {n_bad}/{n} 次置换对照的 Sharpe 非有限（σ=0 或净值算不出），"
                   f"已按保守口径计入 p 值分子；置换分布的有效样本因此少了这些次")
    p = (1 + n_ge) / (n + 1)
    warns = _uniq(own + runw)
    detail.update(n_ge=n_ge, n_nonfinite=n_bad, p_value=p, sharpe_null=null, warnings=warns)
    passed = p < _SHUFFLE_ALPHA
    return GateResult("gate3", passed, detail,
                      _note(f"p = (1+{n_ge})/({n}+1) = {p:.4f}，"
                            f"{'<' if passed else '≥'} {_SHUFFLE_ALPHA}：真实 Sharpe "
                            f"{sr_real:.4f} {'跑赢' if passed else '没跑赢'}同日置换的零假设分布",
                            warns))


# ══════════════ 闸 4 · 成本敏感性 ══════════════

def gate4_cost_stress(cfg: BacktestConfig, multiplier: float = 2.0) -> GateResult:
    """成本整体翻倍重跑，判「相对基准仍为正超额」（§8 闸 4）。

    `multiplier` 是**乘**在现有 `CostConfig.multiplier` 上而不是覆写它：§8 说的是
    "成本参数整体翻倍"。基线本来就调过成本时（比如 3.0），覆写成 2.0 反而是减压。

    超额的仪器是 `metrics['information_ratio']`：它的符号就是平均超额收益的符号
    （`mean(rp − rb) / sd × √P`）。`metrics` 里没有别的相对指标；拿不到基准时它是 None，
    而"算不出超额"不是"超额为正"。

    冲击成本的**封顶在乘 multiplier 之前**（`cost.charge` 模块头）—— 否则被封顶的
    那批交易在 2.0 下纹丝不动，而它们恰恰是最不流动、最该压力测试的那批。
    """
    if not multiplier >= 1:
        # 评审 I4：multiplier=0.0 会以【零成本】跑一遍并报「闸 4 通过」—— 加压不是减压。
        # `not ≥ 1` 而不是 `< 1`：NaN 两边比较都是 False，得走进这一支而不是溜过去。
        return GateResult("gate4", False, {"multiplier": multiplier},
                          f"multiplier={multiplier} < 1：加压不是减压 —— 乘一个小于 1 的数是在"
                          f"给成本打折，打折跑出来的「通过」证明不了任何抗压性")
    stressed = _bare(cfg, cost=_dc.replace(cfg.cost,
                                           multiplier=cfg.cost.multiplier * multiplier))
    base_hash = cfg.param_hash()
    if stressed.param_hash() == base_hash:
        # 加压之后指纹一模一样 = 这次跑的就是基线本身。结果完全正常，
        # 而"闸 4 跑过了"从此是一句谎话。宁可当场判死。
        return GateResult("gate4", False,
                          {"multiplier": multiplier, "baseline_param_hash": base_hash},
                          f"×{multiplier} 之后 D7 指纹一字未变（param_hash={base_hash}）：成本根本没被加压，"
                          f"这次跑的就是基线本身（CostConfig.multiplier 没进 param_hash？）")

    res = run_backtest(stressed)
    warns = _uniq(res.warnings)
    ir = _num(res.metrics.get("information_ratio"))
    detail = {"multiplier": multiplier, "baseline_param_hash": base_hash,
              "stressed": _stamp(stressed, res), "warnings": warns}
    if not _finite(ir):
        return GateResult("gate4", False, detail,
                          _note(f"加压后的信息比率是 {ir}：没有基准或超额收益标准差为 0，"
                                f"「仍为正超额」这句话无从判定（算不出 ≠ 达标）", warns))
    passed = ir > 0
    return GateResult("gate4", passed, detail,
                      _note(f"成本 ×{multiplier}（multiplier {cfg.cost.multiplier} → "
                            f"{stressed.cost.multiplier}）后信息比率 {ir:.4f}，"
                            f"{'仍为正超额' if passed else '超额已被成本吃掉'}", warns))


# ══════════════ 闸 5 · 参数高原 ══════════════

def gate5_param_plateau(cfg: BacktestConfig,
                        grid: Optional[Mapping[str, Sequence]] = None) -> GateResult:
    """`PeakRatio = SR(θ*) / mean(SR(邻域))`，通过标准 `< 1.3`（§8 闸 5）。

    θ* 就是传进来的 `cfg`；邻域是网格里**除 θ* 之外**的点（±30% 网格必然包含
    θ* 自己，模块头 ★6）。把 θ* 算进自己的邻域会把均值朝 SR* 拉高、PeakRatio 变小 ——
    这个方向只会把尖峰洗成高原。

    邻域取**均值**不是中位数/极值：§8 的公式写的就是均值，而且尖峰的特征正是
    "只有一个点高"，中位数与最大值都会把它抹平。
    网格一律钉在样本内（模块头 ★2）：在样本外扫 ±30% 就是拿样本外调参。
    """
    if not grid:
        return GateResult("gate5", False, {},
                          f"{NOT_RUN}：没有参数网格，「邻域」无从构造")
    end = min(cfg.end, OOS_CUTOFF)
    if cfg.start > end:
        return GateResult("gate5", False, {"window": [cfg.start.isoformat(), end.isoformat()]},
                          f"{NOT_RUN}：{cfg.start} 之后已无样本内区间，网格扫描只能落在样本外")
    star_cfg = _bare(cfg, end=end)
    try:
        points = _grid_points(star_cfg, grid)
    except ValueError as e:
        return GateResult("gate5", False, {}, f"参数网格无效：{e}")

    star_hash = star_cfg.param_hash()
    star_res = run_backtest(star_cfg)
    runw: list = list(star_res.warnings)
    sr_star = _sharpe(star_res)

    nb: dict = {}
    for point, pcfg in points:
        if pcfg.param_hash() == star_hash:      # 网格里的 θ* 自己，不重复跑也不进邻域
            continue
        r = run_backtest(pcfg)
        runw += r.warnings
        nb[_label(point)] = _num(_sharpe(r))

    detail: dict = {"theta_star": _stamp(star_cfg, star_res), "sharpe_star": _num(sr_star),
                    "neighbourhood": nb, "threshold": _PEAK_RATIO_MAX,
                    "peak_ratio": None, "mean_neighbour": None}
    ok = [v for v in nb.values() if _finite(v)]
    if not ok:
        detail["warnings"] = _uniq(runw)
        return GateResult("gate5", False, detail,
                          _note(f"邻域里没有一个点算得出 Sharpe（共 {len(nb)} 个）：无从判高原",
                                detail["warnings"]))
    if len(ok) < len(nb):
        # 评审 C3 重放：3 邻域 1 个 NaN，真比值 1.6941（不过）被剔除后洗成 1.1294（过）。
        # 「剔掉再取均值」的方向取决于剔掉的是谁 —— 拿不准就判不过（模块头）。
        bad = sorted(k for k, v in nb.items() if not _finite(v))
        detail["warnings"] = _uniq(runw)
        return GateResult("gate5", False, detail,
                          _note(f"邻域 {len(nb)} 个点里有 {len(bad)} 个算不出 Sharpe"
                                f"（{'; '.join(bad)}）：把它们剔出均值会让尖峰借小分母把自己"
                                f"洗成高原，判不过", detail["warnings"]))
    mean_nb = sum(ok) / len(ok)
    detail["mean_neighbour"] = mean_nb
    if not _finite(sr_star):
        detail["warnings"] = _uniq(runw)
        return GateResult("gate5", False, detail,
                          _note(f"θ* 自己的 Sharpe 是 {sr_star}，比值无从算起",
                                detail["warnings"]))
    if not mean_nb > 0:
        # 均值 ≤ 0 时 PeakRatio 是负数（或 ±inf），`< 1.3` 恒真 ——
        # 一座周围全是坑的孤峰会拿到满分。这正是本闸要挡的形状。
        detail["warnings"] = _uniq(runw)
        return GateResult("gate5", False, detail,
                          _note(f"邻域平均 Sharpe {mean_nb:.4f} ≤ 0：θ* 周围全是坑，"
                                f"比值会变成负数而「小于 1.3」，这恰恰是最坏的尖峰",
                                detail["warnings"]))
    ratio = sr_star / mean_nb
    warns = _uniq(runw)
    detail.update(peak_ratio=ratio, warnings=warns)
    passed = ratio < _PEAK_RATIO_MAX
    return GateResult("gate5", passed, detail,
                      _note(f"PeakRatio = {sr_star:.4f} / {mean_nb:.4f} = {ratio:.4f}，"
                            f"{'<' if passed else '≥'} {_PEAK_RATIO_MAX}："
                            f"{'高原' if passed else '尖峰（过拟合）'}", warns))


# ══════════════ 编排 ══════════════

GATES = {"gate1": gate1_out_of_sample, "gate2": gate2_walk_forward, "gate3": gate3_shuffle,
         "gate4": gate4_cost_stress, "gate5": gate5_param_plateau}
_NEEDS_GRID = ("gate2", "gate5")


def run_all_gates(cfg: BacktestConfig, *, gates: Optional[Sequence[str]] = None,
                  grid: Optional[Mapping[str, Sequence]] = None) -> dict:
    """跑五个闸，返回 `{闸名: GateResult}`（模块头 ★4）。

    · **绝不因某闸失败提前退出** —— 操作员要一次看到全貌，包括哪几道没过；
    · **没被选中的闸照样出现**，`passed=False, note='未运行'`（U5：悄悄不返回读起来像"没问题"）；
    · **抛异常的闸判失败并把异常原文写进 note**，不是消失。

    `grid` 只发给闸 2 与闸 5（另外三个不吃参数网格）。不给 `grid` 时这两道闸
    返回"未运行"，于是"策略过了五闸"这句话自然说不出口 —— §8：任一闸未过不得进生产。
    """
    wanted = list(GATE_NAMES) if gates is None else list(gates)
    unknown = [g for g in wanted if g not in GATE_NAMES]
    if unknown:
        # 拼错闸名 = 操作员以为跑了而其实没跑。这是调用方的错，当场抛。
        raise ValueError(f"未知的闸名 {unknown}；可选：{list(GATE_NAMES)}")

    out: dict = {}
    for name in GATE_NAMES:
        if name not in wanted:
            out[name] = GateResult(name, False,
                                   {"reason": "本次 run_all_gates 的 gates 参数未选中该闸"},
                                   NOT_RUN)
            continue
        kw = {"grid": grid} if name in _NEEDS_GRID else {}
        try:
            out[name] = GATES[name](cfg, **kw)
        except Exception as e:          # noqa: BLE001 —— 见下
            # 吞掉异常与放它炸穿都不行：前者让一道崩溃的闸看起来"没跑"，
            # 后者让后面四道闸一个都跑不成。判失败 + 原文进 note。
            out[name] = GateResult(name, False,
                                   {"exception": type(e).__name__, "message": str(e)},
                                   f"{type(e).__name__}: {e}")
    return out
