"""绩效指标（§9）、因子有效性检验（§4）与风格归因 —— 报告里每一个能骗人的数字。

本文件是**纯计算**：不碰 DB、不 import query、不 import factors。回归元（log_mv / industry）
由调用方按 §3.2 用过的**同一个定义**喂进来（`risk.log_mv` / `risk.industry`）——
若中性化减掉的量与归因度量的量是两份各自演化的实现，归因可以报告"规模暴露已清零"
而账本实际带着倾斜，§3.2 的裁决就永远无法被证伪。

四件不能算错的事：

★ 1【换手必须扣持仓自然漂移，且归一化分母不可省】（§5.4，2026-08-20 修正过一次）
    Δw = w_t − w_{t−1}(1+R_{t−1}) / (1 + Σ_j w_j R_j)
  漏掉漂移项系统性高估换手 15%–30%；漏掉**分母**会让全市场普涨 10%、一笔不交易的一期
  算出 10% 的换手 —— 与 §9 自己的验收断言「持仓不变、价格 +10% → 换手 0」直接矛盾。
  （分母还有一层意思：未满仓时现金不涨，组合总值从 1 变到 1+Σ w R。所以**留了现金的
  组合在齐涨行情里换手不为 0 是对的** —— 持仓占比被顶上去了，维持目标就得卖。）

★ 2【ICIR 的 t 必须用 Newey-West 标准误】（§4.2 / §10.9）
  IC 序列有显著自相关，朴素标准误把 t 高估 30%–50%，这是把噪声因子判成有效因子的
  头号统计错误。滞后阶数 `floor(4(T/100)^{2/9})`，Bartlett 核。
  本文件用 numpy 直接算（6 行），不把它挂在 `statsmodels` 上：那个依赖在 CI 里可选，
  而"依赖缺席 → NW 钉子测试静默 skip → 朴素 SE 的变异逃逸"正是这条铁律要防的事。
  `statsmodels` 仍列在 requirements 里作**参照实现**，测试用 `cov_type='HAC'` 交叉验证。

★ 3【单调性是秩相关，不是 Pearson】（§4.3）
  真实因子的分层几乎从不线性。单调但凸的一组年化收益 Pearson 只有 0.887、Spearman 是 1.0。

★ 4【残差的规模暴露必须永远出现在归因表里】（§3.2 的裁决 + Task 12 brief）
  §3.2 选 OLS 而非 Barra 的 √MV-WLS，理由是本项目的组合是等权 top-N，相关的度量是
  **无权**正交（OLS 的恒等式 X'e = 0）。接受的代价是回归线被数量占多数的小盘主导、
  大盘股残差系统性偏移。**能推翻这个裁决的唯一证据**就是本文件 `attribution` 的
  `style/size` 行在真实数据上仍显著。所以那一行**不能藏在任何开关后面**，
  样本不足也只报 NaN 不删行。
  并且同时报 `style/size_sq`：裁决说补救是加**非线性规模项**（size² 或秩变换）而不是
  换回 WLS（WLS 同样不解决非线性）。一个纯 size² 的残差在线性项上暴露恰为 0 ——
  只报线性暴露会得出"已经中性了"的错误结论，补救方向就无从验证。

★ 5【多空只是评估口径，不是策略】（§4.3）
  A 股融券成本与券源不支持系统性做空。所以键名叫 `long_short_eval_only_annual` ——
  报告里出现一个叫 `long_short_return` 的数字，下一个人就会把它当成可交易的策略收益。

返回值一律 `(结果, warnings)`（global-constraints：返回类型必须留 warning 通道）。
架构 §4.3 给 `compute` 声明的返回类型只有 dict，没有地方放同一份文档要求记录的降级 ——
与 `build_targets` / `simulate` 是同一个缺陷，按已有惯例补齐。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

try:                                    # 可选依赖（仓库既有惯例）：只用来算 t 检验的 p 值
    from scipy import stats as _sps     # noqa: N812
except ImportError:                     # pragma: no cover - 环境相关
    _sps = None

# §9 的年化口径。equity 是日频（架构 §4.3），IC / 分层是调仓频（周）。
_TRADING_DAYS = 252
_IC_PERIODS = 52

# 秩相关在 n=4 时的最小双侧 p 是 1/12 ≈ 0.083 —— 永远够不到 5%，算出来只是噪声。
# n=5 是"秩相关有可能显著"的最小横截面，故取 5。
_MIN_IC_OBS = 5

# 横截面风格回归的最小样本（架构 B7，与 `pipeline.MIN_OBS` 同值同理由：再少的横截面
# 回归不出可信的规模暴露）。不 import factors —— 本层不该被拴在因子注册表上。
_MIN_STYLE_OBS = 30

ATTRIBUTION_COLS = ["block", "item", "exposure", "contribution", "t_stat"]

_UNKNOWN_IND = "__unknown__"


# ══════════════ Newey-West ══════════════
def newey_west_lag(n: int) -> int:
    """`floor(4·(T/100)^{2/9})`（§4.2）。至少 0，且不超过 T−1。"""
    if n < 2:
        return 0
    return max(0, min(n - 1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def _nw_se(x: np.ndarray, lag: int) -> float:
    """样本均值的 Newey-West（Bartlett 核）标准误。

    S = γ₀ + 2·Σ_{l=1..L} (1 − l/(L+1))·γ_l ,  SE = sqrt(S/T)。
    与 statsmodels 的 `cov_type='HAC', use_correction=False` 逐位一致（见测试）。
    Bartlett 核保证 S ≥ 0；S == 0（常数序列）时返回 0.0，由调用方判成"不可检验"。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    return math.sqrt(s / n) if s > 0 else 0.0


def icir(ic: pd.Series) -> "tuple[dict, list[str]]":
    """§4.2：ICIR 与显著性。返回 `(dict, warnings)`。

    dict: `n / mean / std / icir / icir_ann / nw_lag / t_naive / t_newey_west / p`。
    `t_naive` 一并报出**不是给人用的**，是让 NW 的收缩幅度在报告里看得见 ——
    两个 t 差一倍时，"这个因子显著"这句话该由谁来说是很清楚的。
    """
    s = pd.Series(ic, dtype=float)
    s = s[np.isfinite(s.to_numpy())]
    n = int(s.size)
    out: dict = {"n": n, "mean": float("nan"), "std": float("nan"), "icir": float("nan"),
                 "icir_ann": float("nan"), "nw_lag": 0, "t_naive": float("nan"),
                 "t_newey_west": float("nan"), "p": None}
    if n < 2:
        return out, [f"IC 序列只有 {n} 个有效观测，算不出标准差与显著性"]

    warns: list = []
    arr = s.to_numpy(dtype=float)
    # ★ 恒定序列判在【原始值】上，不判在 std 上：29 个相同的 float 减掉自己的均值仍会剩
    #   ~1e-18 的噪声，std 与 NW 标准误因此都是 1e-18 而不是 0 —— t 值会给出 2e16，
    #   一个"无限显著"的因子。这不是数值毛病，是「没有变化的序列不可检验」被算成了确定性。
    if float(np.ptp(arr)) == 0.0:
        out.update(mean=float(arr[0]), std=0.0)
        return out, [f"IC 序列 {n} 期恒为 {arr[0]:.6g}，无变异 —— ICIR 与 t 值都无定义，"
                     f"不是「无限显著」"]

    mean, std = float(s.mean()), float(s.std(ddof=1))
    lag = newey_west_lag(n)
    se_nw = _nw_se(s.to_numpy(dtype=float), lag)
    out.update(mean=mean, std=std, nw_lag=lag,
               icir=mean / std if std > 0 else float("nan"),
               icir_ann=mean / std * math.sqrt(_IC_PERIODS) if std > 0 else float("nan"),
               t_naive=mean / (std / math.sqrt(n)) if std > 0 else float("nan"))
    if not se_nw > 0:
        warns.append(f"IC 序列的 Newey-West 标准误为 0（{n} 期几乎恒定），"
                     f"t 值无定义 —— 不是「无限显著」")
        return out, warns
    t = mean / se_nw
    out["t_newey_west"] = t
    if _sps is not None:
        out["p"] = float(2.0 * _sps.t.sf(abs(t), n - 1))
    else:
        warns.append("scipy 缺席，ICIR 的 p 值未计算（t 值照常给出）")
    return out, warns


# ══════════════ §4.1 IC / RankIC ══════════════
def ic_series(factor_panel: pd.DataFrame, forward_returns: pd.Series
              ) -> "tuple[pd.DataFrame, list[str]]":
    """逐调仓日的 IC 与 RankIC。返回 `(DataFrame, warnings)`。

    Args:
        factor_panel: MultiIndex `(rebalance_date, ts_code)`，每列一个因子（已处理的 z 值）。
        forward_returns: 同索引的持有期收益 R_{i,t}（§5.1：执行日开盘 → 下一执行日开盘）。

    Returns:
        index = rebalance_date，列 = `<factor>__ic` / `<factor>__rank_ic`
        （即 `BacktestResult.ic` 的形状）。**两列都出**：§4.1 主用 RankIC，
        但 Pearson 与 RankIC 的差距本身就是"这个因子被涨停连板带偏了多少"的度量。
    """
    fp = pd.DataFrame(factor_panel)
    cols: list = []
    for c in fp.columns:
        cols += [f"{c}__ic", f"{c}__rank_ic"]
    if fp.empty:
        return pd.DataFrame(columns=cols).rename_axis("rebalance_date"), []

    fr = pd.Series(forward_returns, dtype=float).reindex(fp.index)
    rows: dict = {}
    warns: list = []
    for d, g in fp.groupby(level=0, sort=True):
        y = fr.loc[g.index].astype(float)
        rec: dict = {}
        for c in fp.columns:
            x = g[c].astype(float)
            ok = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
            xa, ya = x[ok], y[ok]
            if len(xa) < _MIN_IC_OBS:
                rec[f"{c}__ic"] = rec[f"{c}__rank_ic"] = float("nan")
                warns.append(f"{d} {c}: 有效横截面 {len(xa)} 只 < {_MIN_IC_OBS}，IC 置 NaN"
                             f"（n=4 的秩相关最小双侧 p 是 1/12，永远够不到 5%）")
                continue
            if xa.nunique() < 2 or ya.nunique() < 2:
                rec[f"{c}__ic"] = rec[f"{c}__rank_ic"] = float("nan")
                warns.append(f"{d} {c}: 横截面是常数（多半是当日全部因子都被覆盖率闸剔掉、"
                             f"process 末尾 fillna(0) 填出来的一列 0），IC 置 NaN")
                continue
            rec[f"{c}__ic"] = float(xa.corr(ya))
            rec[f"{c}__rank_ic"] = float(xa.corr(ya, method="spearman"))
        rows[d] = rec
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)
    return out.rename_axis("rebalance_date"), warns


# ══════════════ §4.3 分层 ══════════════
def layered_returns(scores: pd.Series, forward_returns: pd.Series,
                    n_layers: int = 10) -> "tuple[pd.DataFrame, list[str]]":
    """按 z 升序等分 `n_layers` 组，组内等权持有一期。返回 `(DataFrame, warnings)`。

    **L1 = 分数最低的一层**（§4.3「升序」）。层序倒过来会让单调性符号整体翻转，
    而报告照样"好看"，所以这个方向由测试钉死。

    ★ 分组前先按 (score, ts_code) 定序：`process` 末尾 `fillna(0)` 与 §3.1 的 MAD 截断
      都会造出**精确的平手**，平手是常态不是边角。不定序的话分层随调用方给的 index
      顺序翻脸，D7 的复现就只是碰巧成立（与 `portfolio.build_targets` 同一处坑）。
    """
    if n_layers < 2:
        raise ValueError(f"n_layers={n_layers} 无意义：至少两层才谈得上单调性")
    sc = pd.Series(scores, dtype=float)
    fr = pd.Series(forward_returns, dtype=float).reindex(sc.index)
    names = [f"L{i}" for i in range(1, n_layers + 1)]
    if sc.empty:
        return pd.DataFrame(columns=names).rename_axis("rebalance_date"), []

    rows: dict = {}
    warns: list = []
    for d, g in sc.groupby(level=0, sort=True):
        y = fr.loc[g.index]
        ok = np.isfinite(g.to_numpy()) & np.isfinite(y.to_numpy())
        xa, ya = g[ok], y[ok]
        if len(xa) < n_layers:
            rows[d] = {k: float("nan") for k in names}
            warns.append(f"{d} 有效横截面 {len(xa)} 只 < {n_layers} 层，该日分层跳过")
            continue
        order = xa.sort_index().sort_values(kind="mergesort").index      # ★ 平手定序
        lab = (np.arange(len(order)) * n_layers) // len(order)
        rows[d] = {names[k]: float(ya.reindex(order).to_numpy()[lab == k].mean())
                   for k in range(n_layers)}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=names).rename_axis("rebalance_date"), warns


def layer_monotonicity(layers: pd.DataFrame, *, periods_per_year: int = _IC_PERIODS
                       ) -> "tuple[dict, list[str]]":
    """§4.3 的单调性判据：组序号与组**年化**收益的 Spearman 秩相关 ρ_mono（判据 |ρ|>0.7）。

    ★ 必须是秩相关。单调但凸的一组年化收益（真实因子的常态）Pearson 只有 0.887 ——
      也能过 0.7 的闸，所以"过闸"这个断言钉不住任何东西，本文件的测试断言 ρ 恰等于 1。

    `long_short_eval_only_annual` = 顶层年化 − 底层年化。键名里的 `eval_only` 不是
    啰嗦：多空**仅为因子评估口径**，A 股融券成本与券源不支持系统性做空（§4.3）。
    """
    lay = pd.DataFrame(layers).astype(float)
    warns: list = []
    if lay.empty or lay.shape[1] < 2:
        return {"rho_mono": float("nan"), "long_short_eval_only_annual": float("nan"),
                "layer_annual": {}}, ["分层表为空或不足两层，单调性无定义"]

    ann: dict = {}
    for c in lay.columns:
        col = lay[c].dropna()
        if col.empty:
            ann[c] = float("nan")
            warns.append(f"分层 {c} 全期为空，年化收益无定义")
            continue
        ann[c] = float((1.0 + col).prod() ** (periods_per_year / len(col)) - 1.0)

    vals = pd.Series([ann[c] for c in lay.columns], dtype=float)
    rho = float(pd.Series(np.arange(1.0, len(vals) + 1.0)).corr(vals, method="spearman"))
    n_dropped = int(lay.isna().all(axis=1).sum())
    if n_dropped:
        warns.append(f"{n_dropped} 期分层全空（横截面不足），已从年化里剔除")
    return {"rho_mono": rho,
            "long_short_eval_only_annual": vals.iloc[-1] - vals.iloc[0],
            "layer_annual": {c: ann[c] for c in lay.columns}}, warns


# ══════════════ §5.4 换手（扣持仓自然漂移）══════════════
def turnover_series(weights: pd.DataFrame, returns: pd.DataFrame
                    ) -> "tuple[pd.Series, list[str]]":
    """逐期双边换手 `Σ_i |Δw_i|`，Δw 按 §5.4 **扣漂移并归一化**。

    Args:
        weights: index = 调仓日（升序），columns = ts_code，值 = 该期【实际成交后】的权重。
            缺席读作 0（不持有）。
        returns: 同形状。第 t 行是**上一期到本期**每只票的收益 R_{i,t−1}，
            首行无意义（没有上一期）。持仓票在这里缺值 = 漂移算不出来，会告警。

    首期（无上期持仓）的换手 = Σ|w_0|，即从现金建仓的双边成交额，这是对的。
    """
    w = pd.DataFrame(weights).astype(float).fillna(0.0)
    if w.empty:
        return pd.Series(dtype=float), []
    r = pd.DataFrame(returns).astype(float).reindex(index=w.index, columns=w.columns)
    wprev = w.shift(1).fillna(0.0)

    warns: list = []
    held_no_ret = (wprev != 0) & r.isna()
    held_no_ret.iloc[0] = False                     # 首期本来就没有上一期
    if held_no_ret.to_numpy().any():
        bad = sorted({c for c in w.columns if held_no_ret[c].any()})
        warns.append(f"{int(held_no_ret.to_numpy().sum())} 处持仓缺上期收益，漂移按 0 处理"
                     f"（换手会被高估）：{bad[:5]}")

    rf = r.fillna(0.0)
    denom = 1.0 + (wprev * rf).sum(axis=1)
    if (denom <= 0).any():
        warns.append(f"组合总值归一化分母 ≤ 0 的期数 {int((denom <= 0).sum())}："
                     f"该期换手无定义（净值已归零）")
        denom = denom.where(denom > 0)
    drift = (wprev * (1.0 + rf)).div(denom, axis=0)
    return (w - drift).abs().sum(axis=1).rename("turnover"), warns


# ══════════════ §9 风格 / 行业归因 ══════════════
def attribution(positions: pd.DataFrame, forward_returns: pd.Series, scores: pd.Series,
                *, size: pd.Series) -> "tuple[pd.DataFrame, list[str]]":
    """§9 归因表：行业暴露/贡献 + 残差的规模暴露 + 换手约束拖累。

    Args:
        positions: MultiIndex `(rebalance_date, ts_code)`；需要 `filled_weight` 与
            `industry`，可选 `intended_weight`（换手裁剪【之前】的目标，Task 13 落列）。
        forward_returns: 同索引的持有期收益。
        scores: **全股票池**的合成分数（中性化后的残差经 §6 合成）。
            不能只传账本里的那几十只 —— §3.2 的裁决问的是"残差在整个横截面上还是不是
            规模的代理"，只看账本量到的是账本的倾斜，不是残差的性质。
        size: 同索引的 `risk.log_mv`。**与 `pipeline.neutralize` 减掉的必须是同一个定义**。

    Returns:
        `(DataFrame[ATTRIBUTION_COLS], warnings)`。`block ∈ {'industry','style','constraint'}`。
        `style/size` 与 `style/size_sq` **恒在表内**（见模块头 ★4），算不出就是 NaN + 告警。
    """
    pos = pd.DataFrame(positions)
    for c in ("filled_weight", "industry"):
        if c not in pos.columns:
            raise ValueError(f"positions 缺列 {c!r}：行业归因算不出来")
    warns: list = []
    rows: list = []
    n_dates = max(1, int(pd.Index(pos.index.get_level_values(0)).nunique()))

    # ── 行业块 ──
    w = pos["filled_weight"].astype(float)
    R = pd.Series(forward_returns, dtype=float).reindex(w.index)
    if R.isna().any():
        warns.append(f"{int(R.isna().sum())} 个持仓没有持有期收益，行业贡献按缺失计")
    # ★ .astype(object) 不可省：category dtype 下 groupby 会为【零观测的类别】发一行
    #   （pandas 2.3.3 实测 `__OTHER__ → NaN`，`value_counts` 同样报 `__OTHER__ → 0`）。
    #   `pipeline.neutralize` 已经被 `get_dummies` 的同一个行为咬过一次。
    #   归因表里多一行不存在的行业暴露，读者会当成真的持仓。
    ind = pos["industry"].astype(object).fillna(_UNKNOWN_IND)
    expo = w.groupby(ind).sum() / n_dates
    contrib = (w * R).groupby(ind).sum() / n_dates
    for k in sorted(expo.index, key=str):
        rows.append(("industry", str(k), float(expo[k]), float(contrib[k]), float("nan")))

    # ── 风格块：逐期横截面回归 score ~ 1 + z + z²，再对系数序列做 Fama-MacBeth ──
    #    z 是**期内标准化**的 log_mv：不标准化的话 size² 与 size 高度共线，
    #    非线性项的系数就没法读；标准化后 z 与 z² 在对称分布上正交。
    df = pd.DataFrame({"y": pd.Series(scores, dtype=float),
                       "x": pd.Series(size, dtype=float)}).dropna()
    df = df[np.isfinite(df.to_numpy()).all(axis=1)]
    b1: list = []
    b2: list = []
    skipped = 0
    for _d, g in df.groupby(level=0, sort=True):
        if len(g) < _MIN_STYLE_OBS:
            skipped += 1
            continue
        sd = float(g["x"].std(ddof=1))
        if not sd > 0:
            skipped += 1
            continue
        z = ((g["x"] - g["x"].mean()) / sd).to_numpy()
        X = np.column_stack([np.ones(len(z)), z, z ** 2])
        beta, _res, rank, _sv = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        if rank < X.shape[1]:
            skipped += 1
            continue
        b1.append(float(beta[1]))
        b2.append(float(beta[2]))
    if skipped:
        warns.append(f"{skipped} 期算不出残差的规模暴露（有效样本 < {_MIN_STYLE_OBS} 或市值恒定）："
                     f"§3.2「OLS 而非 WLS」的裁决靠这一项才能被真实数据检验")
    if not b1:
        warns.append("整段区间都没算出残差的规模暴露 —— §3.2 的裁决在本次运行里"
                     "【未被检验】，不是「已经中性」")
    for item, series in (("size", b1), ("size_sq", b2)):
        if series:
            arr = np.asarray(series, dtype=float)
            se = _nw_se(arr, newey_west_lag(arr.size))
            t = float(arr.mean() / se) if se > 0 else float("nan")
            rows.append(("style", item, float(arr.mean()), float("nan"), t))
        else:
            rows.append(("style", item, float("nan"), float("nan"), float("nan")))

    # ── 约束拖累：意图账本的反事实收益 − 实际收益 ──
    if "intended_weight" in pos.columns:
        gap = pos["intended_weight"].astype(float) - w
        rows.append(("constraint", "turnover_budget",
                     float(gap.abs().sum()) / n_dates,
                     float((gap * R).sum()) / n_dates, float("nan")))
    else:
        warns.append("positions 没有 intended_weight 列：换手预算裁掉的那部分收益无法归因 —— "
                     "分不清「跑输是因为信号不行」还是「因为换手约束让信号表达不出来」，"
                     "对受换手约束的策略这是完全相反的两个结论（Task 13 落这一列）")
        rows.append(("constraint", "turnover_budget", float("nan"), float("nan"), float("nan")))

    return pd.DataFrame(rows, columns=ATTRIBUTION_COLS), warns


# ══════════════ §9 绩效指标 ══════════════
def _drawdown(v: np.ndarray) -> float:
    return float((1.0 - v / np.maximum.accumulate(v)).max())


def compute(equity: pd.Series,
            trades: pd.DataFrame,
            positions: pd.DataFrame,
            benchmark_series: Optional[pd.Series],
            *,
            full: bool,
            periods_per_year: int = _TRADING_DAYS,
            risk_free: float = 0.0,
            factors_used: Optional[pd.Series] = None) -> "tuple[dict, list[str]]":
    """§9 的净值 / 相对 / 交易三类指标。返回 `(metrics, warnings)`。

    Args:
        equity: 日频净值（初始 1.0），index = trade_date。
        trades: `execution.simulate` 的成交表**经 `cost.charge` 补过费用列**。
        positions: MultiIndex `(rebalance_date, ts_code)`，需要 `filled_weight` /
            `target_weight` / `price_hfq`（`full=True` 时）。
            ★ `price_hfq` 请存 **T 日收盘后复权价**：本函数报的换手与 `build_targets`
              的换手上限必须同一个口径（都在 T 收盘度量）。`simulate` 另在 τ 开盘重算
              一遍漂移算真实成本 —— 两个口径**故意不同**（Task 11 交接注），
              差的是一个隔夜跳空，是已知近似不是 bug。混用会让报告里的换手与预算对不上。
        benchmark_series: 基准净值，index 与 `equity` 对齐。`None` → 相对指标缺失 + 告警。
        full: `False` 只算净值类（架构 §4.3 的 8s 档）；`True` 追加换手 / 成本拖累 /
            D6 缺口 / 因子存活数。
        periods_per_year: 年化口径。equity 是日频故默认 252。
            ★ §9 的表格自身不自洽：年化收益写 `252/D`（日频）而年化波动写 `σ_weekly·√52`
              （周频），同一条曲线上混用两个频率算出的 Sharpe 没有定义。本函数一律按
              **入参序列自己的频率**年化，两处用同一个数。
        risk_free: 年化无风险利率（§9：1 年期国债）。默认 0 —— 会让 Sharpe 偏高约 R_f/σ。
        factors_used: index = 调仓日、值 = 该期**实际参与合成的因子个数**。
            ★ 不给就告警：`build_targets` 的 50% 覆盖率闸经 `combine` 喂进来是【二值】的
              （`process` 末尾 fillna(0) 让每只票都有数），探测不到部分降级。
              「12 个因子失效、拿剩下 2 个把整个账本调了一遍」只能靠这一项发现。
    """
    warns: list = []
    eq = pd.Series(equity, dtype=float).dropna()
    out: dict = {"n_days": int(len(eq))}
    if len(eq) < 2:
        return {**out, "annual_return": float("nan"), "annual_vol": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "calmar": float("nan"), "information_ratio": None}, \
               ["净值序列不足两个点，所有绩效指标无定义"]

    v = eq.to_numpy(dtype=float)
    steps = len(v) - 1
    years = steps / float(periods_per_year)
    rp = pd.Series(v[1:] / v[:-1] - 1.0, index=eq.index[1:])

    ann_ret = float((v[-1] / v[0]) ** (1.0 / years) - 1.0)
    ann_vol = float(rp.std(ddof=1) * math.sqrt(periods_per_year))
    mdd = _drawdown(v)
    out.update(years=years, annual_return=ann_ret, annual_vol=ann_vol,
               sharpe=(ann_ret - risk_free) / ann_vol if ann_vol > 0 else float("nan"),
               max_drawdown=mdd,
               calmar=ann_ret / mdd if mdd > 0 else float("nan"),
               equity_final=float(v[-1]))

    # ── 相对指标 ──
    if benchmark_series is None:
        out["information_ratio"] = None
        warns.append("未提供基准净值，信息比率等相对指标缺失（§9 基准：中证全指 000985.CSI）")
    else:
        bm = pd.Series(benchmark_series, dtype=float).reindex(eq.index)
        if bm.isna().any():
            warns.append(f"基准在 {int(bm.isna().sum())} 个交易日缺值，相对指标只用两边都有的那些日子")
        bv = bm.to_numpy(dtype=float)
        rb = pd.Series(bv[1:] / bv[:-1] - 1.0, index=eq.index[1:])
        d = (rp - rb).dropna()
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        out["information_ratio"] = (float(d.mean() / sd * math.sqrt(periods_per_year))
                                    if sd > 0 else None)
        if not sd > 0:
            warns.append("超额收益的标准差为 0，信息比率无定义")

    if not full:
        return out, warns

    # ── 交易类（§9 后三行 + D6 缺口 + 因子存活数）──
    pos = pd.DataFrame(positions)
    out["n_trades"] = int(len(trades))
    if len(pos):
        if "filled_weight" in pos.columns and "price_hfq" in pos.columns:
            w = pos["filled_weight"].astype(float).unstack("ts_code")
            px = pos["price_hfq"].astype(float).unstack("ts_code")
            # ★ `fill_method=None` 不可省：`pct_change` 默认 ffill，一只票在上一期不在
            #   账本里（或那天没有价）就会拿更早的价格补出一个【编造的收益】，
            #   与 D9 说的"把停牌占位行 ffill 成假的 0 收益"是同一个错。缺了就该是 NaN，
            #   由 `turnover_series` 告警。
            to, w_to = turnover_series(w, px.pct_change(fill_method=None))
            warns += w_to
            out["turnover_mean"] = float(to.mean())
            out["turnover_annual"] = float(to.sum() / years) if years > 0 else float("nan")
        else:
            warns.append("positions 缺 filled_weight / price_hfq，换手无法扣持仓漂移（§5.4）")
        if "target_weight" in pos.columns and "filled_weight" in pos.columns:
            # ★ D6 缺口从【实际权重】现算。`simulate` 的 Σ blocked.intended_weight
            #   两个方向都会偏（execution.py 的交接注：实测 0.55 vs 0.80、0.70 vs 0.42），
            #   不能拿它当缺口。
            gap = (pos["target_weight"].astype(float) - pos["filled_weight"].astype(float)) \
                .abs().groupby(level=0).sum()
            out["d6_slippage_mean"] = float(gap.mean())
            out["d6_slippage_max"] = float(gap.max())

    tr = pd.DataFrame(trades)
    if len(tr) and {"exec_date", "total_cost"} <= set(tr.columns):
        by_day = tr.groupby("exec_date")["total_cost"].sum()
        aligned = eq.reindex(by_day.index)
        off = list(by_day.index[aligned.isna()])
        if off:
            warns.append(f"{len(off)} 个成交日不在净值曲线上，其成本未计入拖累：{[str(d) for d in off[:3]]}")
        ct = (by_day / aligned).dropna()
        out["cost_drag_annual"] = float(ct.sum() / years) if years > 0 else float("nan")
        out["cost_total"] = float(by_day.sum())
    elif len(tr):
        warns.append("trades 没有 total_cost 列（未经 cost.charge），成本拖累缺失")

    # ── 每期实际用了几个因子（§9 诊断）──
    if factors_used is None:
        warns.append("未提供每期存活因子数：`combine` 的逐因子剔除是【部分降级】，"
                     "而 `build_targets` 的覆盖率闸经 fillna(0) 之后只剩二值，探测不到它。"
                     "不给这一项，「12 个因子失效、拿剩下 2 个调完整个账本」就只能靠翻 warning 发现")
    else:
        fu = pd.Series(factors_used).dropna().astype(int)
        if len(fu):
            top = int(fu.max())
            low = fu[fu < top / 2.0]
            out.update(factors_used_min=int(fu.min()), factors_used_max=top,
                       factors_used_median=float(fu.median()),
                       n_periods_below_half=int(len(low)))
            if len(low):
                warns.append(f"{len(low)} 期只用了不到一半的因子（最少 {int(fu.min())}/{top}）："
                             f"{[str(d) for d in low.index[:3]]}")
    return out, warns
