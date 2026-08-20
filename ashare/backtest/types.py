"""回测引擎的输入输出数据结构（架构 §4.3）。

★ 本文件只有数据结构，没有任何 IO —— D1：LLM 层对策略参数只有读权限，这里不能出现写回路径。
  架构 §4.3 在 `BacktestResult` 上还列了 `save(run_id)` / `load(run_id)`（→ derived.duckdb + parquet），
  但落库要 `import duckdb`，而分层规则 L1 只允许 `ashare/data/**` 这么做。所以这两个方法
  **不在本文件**，等 data 层开出公开写入口后由落库任务接；在这里补 `save()` 会当场撞
  `scripts/check_ashare_layering.py`。

★ `param_hash()` 是 D7 的执行机制，不是调试标签。
  「样本外只跑一次」靠的是：每次运行记 `(param_hash, data_snapshot_id)`，同哈希 = 重放，
  异哈希 = 新实验、拒绝。少哈希一个影响结果的字段，两组真不同的参数就撞成一个指纹，
  「已经跑过样本外」的闸门永远不触发 —— 人在样本外数据上调参却以为没有，且**不报错**。
  所以 payload 走 `dataclasses.asdict()` 的**全量字段**再显式剔除，而不是手写字段白名单：
  以后新增字段默认进哈希（安全方向），要排除必须动手写理由。

★ 哈希风格与 `ashare/factors/base.py::FactorSpec.param_hash` 一致（canonical JSON：键排序 +
  紧凑分隔符 + 日期 isoformat + sha256 截断），两处指纹因此可以并排放进 `docs/oos-runs.md`
  而不需要解释"为什么长得不一样"。没有复用那边的私有 helper：本层多了递归定精度与
  元组归一，而跨到 factors 层只为省 6 行会把回测层的导入拴在因子注册表上。
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

# float 舍到 12 位小数再哈希：0.1+0.2 与 0.3 是两个不同的 float，却是同一组参数。
# 不定精度就会为一个 1e-17 的差别造出第二个指纹 ——「什么都没变但指纹变了」与漏哈希
# 一样有害（前者把重放记成新实验，后者把新实验记成重放）。1e-12 以下的差改不动任何一个
# 回测数字：权重量级 1e-1，费率量级 1e-4，股数还要按 100 股取整。
_HASH_FLOAT_NDIGITS = 12

# summary() 走 REST 与 Agent 工具层：3 KB 是【硬预算】，不是建议值。
_SUMMARY_MAX_BYTES = 3072
_SUMMARY_MAX_WARNINGS = 5
_SUMMARY_WARNING_CHARS = 90
_SUMMARY_MAX_FACTORS = 10
_SUMMARY_NAME_CHARS = 32        # 因子名截断：让 factors 这一项的字节数【由构造决定】而不是靠祈祷


def _hash_canon(o):
    """递归规范化成可确定性序列化的结构：float 定精度 / date→isoformat / 序列→list。"""
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return round(o, _HASH_FLOAT_NDIGITS)
    if isinstance(o, Mapping):
        return {str(k): _hash_canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_hash_canon(v) for v in o]
    if isinstance(o, _dt.date):          # date / datetime 都有 isoformat
        return o.isoformat()
    return o                             # 交给 json 的 default 去拒绝


def _hash_reject(o):
    # 吞掉不可序列化的参数 = 指纹随进程变（object() 的 repr 带内存地址），D7 台账全是一次性记录
    raise TypeError(f"参数类型 {type(o).__name__} 无法确定性序列化，不能进 param_hash")


def _jsonable(o):
    """把引擎给的自由结构压成 REST / LLM 收得下的 JSON。

    - float 舍到 6 位：净值指标看到小数点后 6 位已经过分，多出来的全是字节预算；
    - NaN / ±Inf → None：starlette 的 JSONResponse 用 `allow_nan=False`，
      一个 NaN 指标就是一次 500，而不是"数字难看一点"；
    - numpy 标量（`np.int64` 不是 `int` 的子类）先试 float()，再退化成 str —— 宁可难看不可 500。
    """
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):             # np.float64 是 float 的子类，走这里
        return round(o, 6) if math.isfinite(o) else None
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _dt.date):          # pd.Timestamp 是 datetime 的子类，走这里
        return o.isoformat()
    try:
        return _jsonable(float(o))
    except (TypeError, ValueError):
        return str(o)


def _json_size(d: dict) -> int:
    """按最坏情况量：`ensure_ascii=False`（中文 3 字节/字）+ 默认分隔符（比 REST 实际发出的紧凑
    分隔符更宽）。不带 `default=`：真有东西没被 `_jsonable` 收干净，要在这里炸，
    而不是等 starlette 序列化响应时炸 —— 那时 traceback 里已经看不出是哪个字段。"""
    return len(json.dumps(d, ensure_ascii=False).encode("utf-8"))


def _warnings_note(shown: int, total: int) -> str:
    return (f"输出预算 {_SUMMARY_MAX_BYTES} 字节所限，仅列前 {shown}/{total} 条；"
            f"完整列表见 BacktestResult.warnings")


@dataclass(frozen=True)
class CostConfig:                       # 数值来源：规格 §5.3（往返约 0.3%），本文不重复解释
    commission_bps: float = 2.5         # 双边
    stamp_duty_bps: float = 5.0         # 仅卖出
    transfer_bps: float = 0.1           # 沪市
    impact_coef: float = 0.5            # 0.5 × (委托额/当日成交额) × 振幅
    impact_cap_bps: float = 30.0
    multiplier: float = 1.0             # 闸 4（成本敏感）设 2.0


@dataclass(frozen=True)
class PortfolioConstraints:
    top_n: int = 50
    weighting: str = "equal"            # 'equal' | 'risk_parity'
    max_single: float = 0.05
    max_industry: float = 0.20
    max_turnover: float = 0.30          # 单周双边换手上限


@dataclass(frozen=True)
class BacktestConfig:
    start: _dt.date
    end: _dt.date
    factors: tuple[tuple[str, float], ...]      # 有序元组以便 hash；见 param_hash 的顺序说明
    constraints: PortfolioConstraints = PortfolioConstraints()
    cost: CostConfig = CostConfig()
    # ★ 架构 §4.3 的字面量写 True，同一行注释写"P2 阶段默认 False"（自相矛盾）。取 False：
    #   宏观择时层属 P3，默认 True 会让 P2 的每次回测都去调一个还不存在的层。
    macro_timing: bool = False          # False → 恒定满仓
    position_floor: float = 0.20
    position_cap: float = 1.00
    benchmark: str = "000985.CSI"
    initial_capital: float = 1_000_000.0
    # ── 运行模式开关 ──
    compute_diagnostics: bool = True    # False → 跳过 IC/分层/归因，只算净值（8s 档）
    shuffle_seed: int | None = None     # 非 None → 每个调仓日横截面打乱分数（闸 3）
    factor_param_override: Mapping[str, Mapping] = field(default_factory=dict)

    def param_hash(self) -> str:
        """sha256(canonical_json(策略语义字段))[:16] —— D7 的参数指纹。

        与 `query.snapshot_id()` 一起写进 `docs/oos-runs.md`：两者缺一，样本外就没法判定
        「这次是重放还是新实验」。
        """
        payload = _dc.asdict(self)      # 嵌套 dataclass（cost / constraints）由 asdict 递归展开：
                                        # 只哈希顶层标量的话，改一下佣金率指纹不动，闸 4 会把
                                        # 成本翻倍的对照组记成它自己的基线 —— 闸 4 等于没跑
        # ★ 唯一的排除项。合法性条件：compute_diagnostics 只能【新增】ic/layers/attribution
        #   三个诊断字段，不得改动 equity / trades / positions / metrics 里的任何一个数。
        #   哪天诊断开始影响结果路径（比如改用 IC 加权合成），必须把它挪回哈希里，
        #   否则两次结果不同的运行共用一个 D7 指纹。
        payload.pop("compute_diagnostics")
        # 因子顺序不是语义：combine 是 Σ wᵢ·dirᵢ·zᵢ（架构 §4.2），加法可交换，
        # 换个书写顺序是同一个策略。不排序的话 D7 台账里会多出一条根本不存在的"新实验"。
        # （求和次序带来的 ulp 级差异远在 1e-12 定精度之下，不构成"结果不同"。）
        payload["factors"] = sorted(payload["factors"])
        blob = json.dumps(_hash_canon(payload), sort_keys=True, separators=(",", ":"),
                          default=_hash_reject)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class BacktestResult:
    config: BacktestConfig
    param_hash: str
    data_snapshot_id: str               # ★ 补强 D7
    engine_version: str                 # 引擎语义变更时手动 bump
    started_at: _dt.datetime
    elapsed_sec: float

    equity: pd.Series                   # index=trade_date（日频），组合净值（初始 1.0）
    positions: pd.DataFrame             # MultiIndex (rebalance_date, ts_code)
                                        #   cols: score, target_weight, filled_weight,
                                        #         shares, price_hfq, industry
    trades: pd.DataFrame                # cols: exec_date, ts_code, side('BUY'|'SELL'),
                                        #   shares, price_hfq, amount, commission,
                                        #   stamp_duty, transfer_fee, impact, total_cost
    blocked: pd.DataFrame               # ★ 不可交易明细，D6 的证据链
                                        #   cols: exec_date, ts_code, intended_side,
                                        #         intended_weight, reason
    metrics: dict                       # 规格 §5.4 的净值/相对/交易三类指标
    # compute_diagnostics=False 时三者皆 None，故给默认值（字段本身按 §4.3 一个不少）
    ic: pd.DataFrame | None = None      # index=rebalance_date, cols=<factor>__ic/__rank_ic
    layers: pd.DataFrame | None = None  # 10 分层：index=rebalance_date, cols=L1..L10
    attribution: pd.DataFrame | None = None   # Brinson 行业 + 风格回归
    warnings: list[str] = field(default_factory=list)  # 中性化秩亏 / 覆盖率不足 / 日历缺口等

    def summary(self) -> dict:
        """供 REST / Agent 工具返回的精简版，序列化后 < 3 KB。

        明细（净值序列、逐笔成交、分层表）一律不进来：它们随回测年限线性增长，
        而这个返回值要塞进 LLM 上下文。超预算时**继续砍并在输出里写明砍了什么**，
        不静默丢数据，也不返回一个 8 KB 的"精简版"。
        """
        cfg = self.config
        n_warn = len(self.warnings)
        out = {
            # ── D7 身份：param_hash 与 data_snapshot_id 缺一不可 ──
            "param_hash": self.param_hash,
            "data_snapshot_id": self.data_snapshot_id,
            "engine_version": self.engine_version,
            "started_at": _jsonable(self.started_at),
            "elapsed_sec": round(float(self.elapsed_sec), 2),
            # ── 策略口径 ──
            "start": _jsonable(cfg.start),
            "end": _jsonable(cfg.end),
            "benchmark": cfg.benchmark,
            "n_factors": len(cfg.factors),      # 截断可见：n_factors 与 len(factors) 一比就知道
            "factors": [[n[:_SUMMARY_NAME_CHARS], round(float(w), 6)]
                        for n, w in cfg.factors[:_SUMMARY_MAX_FACTORS]],
            "top_n": cfg.constraints.top_n,
            "weighting": cfg.constraints.weighting,
            "cost_multiplier": cfg.cost.multiplier,     # 闸 4 的标记
            "macro_timing": cfg.macro_timing,
            "shuffle_seed": cfg.shuffle_seed,           # ★ 闸 3 的 200 次对照不能长得像真回测
            # ── 结果 ──
            "metrics": _jsonable(self.metrics),
            "n_days": len(self.equity),
            "equity_final": _jsonable(self.equity.iloc[-1]) if len(self.equity) else None,
            "n_rebalances": len(self.positions.index.unique(level=0)) if len(self.positions) else 0,
            "n_trades": len(self.trades),
            "n_blocked": len(self.blocked),             # D6 证据链的规模，0 与 9000 差别很大
            "diagnostics": {"ic": self.ic is not None, "layers": self.layers is not None,
                            "attribution": self.attribution is not None},
            "warnings_total": n_warn,
            "warnings": [w[:_SUMMARY_WARNING_CHARS] for w in self.warnings[:_SUMMARY_MAX_WARNINGS]],
        }
        if len(out["warnings"]) < n_warn:
            out["warnings_note"] = _warnings_note(len(out["warnings"]), n_warn)
        # 上面的静态上限只保证【条数】有界，长度没有：metrics 是 Task 12 给的自由 dict，
        # 告警文本长度也没人管。预算保证不能建立在"它们不会太长"上 —— 真顶到 3 KB 就继续砍，
        # 顺序 = 代价从小到大：告警是诊断线索，metrics 是用户问的那个答案，最后才轮到它。
        while _json_size(out) > _SUMMARY_MAX_BYTES and out["warnings"]:
            out["warnings"].pop()
            out["warnings_note"] = _warnings_note(len(out["warnings"]), n_warn)
        if _json_size(out) > _SUMMARY_MAX_BYTES:
            out["metrics"] = {"_dropped": f"metrics 共 {len(self.metrics)} 项，超出 "
                                          f"{_SUMMARY_MAX_BYTES}B 预算，见 BacktestResult.metrics"}
        return out
