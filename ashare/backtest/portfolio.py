"""组合构建（算法说明书 §7.2 / 设计规格 §6.2）—— 等权 top-N + 三条约束，不上 LP。

★ 计划里的五步（取前 N → 等权 → 行业上限 → 单股上限再分配 → 换手裁剪）**不可交换**，
  顺序执行也**不收敛到"全部约束同时满足"**。三处硬冲突，逐条给了结论而不是绕开：

  ① 换手上限与 Σw == π 在数学上可以不相容。||w − w_prev||₁ ≥ |Σw − Σw_prev| 是恒等式：
     仓位从 0.2 调到 1.0 至少要 0.8 的换手，给多少预算都改不了。首期建仓（Σw_prev = 0）
     只是这条恒等式最显眼的一个特例，不是需要单独开洞的边角。
  ② 换手裁剪的部分执行会破坏步 3/4 刚建立的两条上限：没成交的那几笔停在 w_prev 上，
     而 w_prev 可能已经漂移越界。
  ③ 步 4「截断超限的单股、把额度按 scores 比例分给剩余股票」在等权下**不可达**：
     步 1–3 之后所有权重恒等于 w_unit（替补只换名字不换权重），越界要么全体越界、
     要么无人越界，永远没有"剩余股票"可接收再分配。所以这一步在本文件里不存在，
     它的可达形态是「按上限截断 + 差额留现金」，见 `w_unit` 一行。

  裁决 = 一条全局优先级，四处冲突都按它来（不是每处单独拍脑袋）：

      风险上限（max_single / max_industry）  >  Σw == π  >  max_turnover

  为什么换手垫底 —— 不是"成本没风险重要"这种口号，而是**可见性不对称**：
  换手超标的代价会被 §5.4 的成本模型如实扣进净值，回测自己会疼；
  集中度超标却没有任何一个指标会叫，它只在样本里可能压根没出现的那次尾部事件里兑现。
  **优先违反那条"违反后回测会自己算出代价"的约束**，这是本仓库防自欺的一贯方向。
  为什么 Σw == π 压在换手之上：π 是宏观层给的总仓位口径（§7.1），少投的部分没人计价，
  且回测报出来的会是另一个策略的净值。

★ `prev_weights` 是【价格漂移后的实际权重】= 持仓市值 / 权益，**不是上期的目标权重**。
  两者在有现金的组合里连总和都不同（Σw_prev ≠ π 是常态，不是异常）。换手量的是
  真要成交的量，对着上期目标量会系统性错量，而换手直接进成本、成本直接进净额收益。
  → 调用方（Task 13 引擎）每期必须由 prev_holdings × T 日收盘价 / 权益重算这个入参，
    不能把上一轮 build_targets 的返回值直接喂回来。
  ★ 顺带纠一处说明书笔误：§5.4 的漂移项写作 w_{t−1}(1+R_{i,t−1})，少了组合自身收益的
    归一化分母 (1+Σ w_j R_j)。全市场齐涨 10% 而持仓一股没动时，那个式子会算出 10% 的
    换手（正确答案是 0）—— Task 12 的验收断言写的正是"涨 10% → 换手为 0"，两者相抵触。
    本文件的口径（入参就是归一化后的实际权重）与那条验收断言一致。

★ scores 全 NaN → 返回 `prev_weights` **原样**，不是空 Series。空 Series 读作「清仓」，
  prev 读作「维持」。数据中断不是清仓信号，而且中断常与极端行情同期，回测里会长成
  「策略在暴跌前防御性离场」的漂亮假净值。部分 NaN 同理按覆盖率判：低于
  `_MIN_SCORE_COVERAGE` 视为中断（一起冻结），之上就正常在非 NaN 的票里排名建仓。
  → 因此 `scores` 的 index 必须是**当日完整股票池**、算不出的位置填 NaN。
    调用方若先把 NaN 剔掉再传进来，覆盖率恒为 100%，这道闸门直接失明。

本文件是纯计算：不碰 DB、不 import query，可交易性（停牌 / 一字板）属 Task 11 的
`execution.simulate`。返回值是**目标权重**，不是成交结果。
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

from .types import PortfolioConstraints

# 评分覆盖率闸门：低于此值判为取数中断而不是"横截面变瘦"。
# 取 0.5 的理由是两类故障的形态不同 —— 数据源塌掉是断崖（95% → 个位数），
# 而正常的稀疏（次新股不够历史、财报未披露）是渐变，几乎不会把全市场砍掉一半。
_MIN_SCORE_COVERAGE = 0.50

_CAP_EPS = 1e-12        # 上限比较容差；权重量级 1e-2，1e-12 只吞浮点噪声
_SUM_EPS = 1e-9         # Σw == π 的验收容差
_UNKNOWN = "__unknown__"


def _industries(industry, idx: pd.Index) -> pd.Series:
    """行业缺失全部归进同一个桶（保守）。

    让它们"每只自成一个行业"等于对这批股票取消行业上限，而查不到行业通常意味着
    同一批数据出了问题 —— 恰恰是最不该放行的那批。
    """
    return pd.Series(industry, dtype=object).reindex(idx).fillna(_UNKNOWN)


def _side_fill(mag: pd.Series, budget: float, key: pd.Series, buy: bool) -> pd.Series:
    """在 `budget` 内按 |Δw| 降序逐笔成交，最后一笔可部分成交。返回每只的成交额（正数）。

    并列时按分数取舍：买入取分数高者、卖出取分数低者。等权组合里**新建仓的 Δw 全部相等**，
    并列是常态而非边角 —— 不定序的话"先做最重要的调整"在建仓侧等于随机挑。
    """
    if len(mag) == 0 or budget <= 0:
        return pd.Series(0.0, index=mag.index)
    o = pd.DataFrame({"m": mag, "k": key.reindex(mag.index)}).sort_values(
        ["m", "k"], ascending=[False, not buy], na_position="last" if buy else "first")
    prior = o["m"].cumsum() - o["m"]                    # 排在前面的已占用额度
    take = np.minimum((budget - prior).clip(lower=0.0), o["m"])
    return take.reindex(mag.index)


def _execute(prev: pd.Series, delta: pd.Series, budget: float,
             forced: pd.Index, key: pd.Series) -> pd.Series:
    """在换手预算内执行 `delta`；`forced` 里的笔无条件全额执行（预算不够就超支）。

    买卖必须**成对**收缩：净额 ν = Σδ 是仓位口径本身（Σw = Σw_prev + ν = π），
    必须全额执行；只有旋转部分（一买一卖，每单位吃两单位换手）才按剩余预算裁剪。
    照字面"按 |Δw| 降序执行前若干笔、其余保持 w_prev"会只砍单边，Σw 随之偏离 π。

    ν 已超预算时 m 被夹到 0，成交额 = |ν| —— 超支，但只超到"刚好够"，不多做一分。
    """
    rest = delta.index.difference(forced)
    left = max(0.0, budget - float(delta.reindex(forced).abs().sum()))
    d = delta.reindex(rest)
    nu = float(d.sum())
    buys, sells = d[d > 0], -d[d < 0]
    if nu >= 0:                                          # p − m = ν 且 p + m ≤ left
        m = min(float(sells.sum()), max(0.0, (left - nu) / 2.0))
        p = m + nu
    else:
        p = min(float(buys.sum()), max(0.0, (left + nu) / 2.0))
        m = p - nu
    add = pd.Series(0.0, index=delta.index)
    add.loc[buys.index] = _side_fill(buys, p, key, True)
    add.loc[sells.index] = -_side_fill(sells, m, key, False)
    if len(forced):
        add.loc[forced] = delta.reindex(forced)
    return prev + add


def _violators(w: pd.Series, ind: pd.Series, cs: PortfolioConstraints) -> pd.Index:
    """越界的单股 + 越界行业里的**全部**成分股。

    整个行业一起返回是有意的：把该行业的每一笔都执行掉，行业权重就落到 Σw_target ≤ 上限，
    只挑其中几只补足反而可能留下残余。
    """
    over = w.index[w > cs.max_single + _CAP_EPS]
    by = w.groupby(ind).sum()
    bad_inds = by.index[by > cs.max_industry + _CAP_EPS]
    return over.union(w.index[ind.isin(bad_inds)])


def build_targets(scores: pd.Series, target_position: float, prev_weights: pd.Series,
                  industry: pd.Series, constraints: PortfolioConstraints
                  ) -> tuple[pd.Series, list[str]]:
    """按合成分数产出目标权重。返回 `(target_weight, warnings)`。

    Args:
        scores: index = **当日完整股票池**，算不出的位置填 NaN（见模块头：先剔 NaN 会让
            覆盖率闸门失明）。±inf 与 NaN 同等处理。
        target_position: 宏观层给的总仓位 π ∈ [0, 1]。组合允许持现金，Σw = π ≤ 1。
        prev_weights: **价格漂移后的实际权重**（持仓市值 / 权益），不是上期目标权重。
            Σ 一般 ≠ π。首期传空 Series。
        industry: ts_code → 申万一级行业；缺失值归进同一个保守桶。
        constraints: 只支持 `weighting='equal'`；其余字段见 `PortfolioConstraints`。

    Returns:
        `(weights, warnings)`。weights 的 index = 目标持仓 ∪ 上期持仓，**清仓的票显式写 0.0**
        （缺席读作"没意见"，0 读作"卖光"）。warnings 与 `pipeline.process` 同惯例，
        由引擎汇进 `BacktestResult.warnings`：约束被迫让路的那一天必须在报告里看得见。

    约束优先级（模块头有完整论证）：风险上限 > Σw == π > max_turnover。
    因此 **Σw == π 只在账本填得满时成立**；填不满（可投名额不足 / 行业上限拦住）时
    差额留现金并告警 —— 强行按 π/k 放大权重等于往已经顶格的地方继续加仓。
    """
    cs = constraints
    if cs.weighting != "equal":
        raise ValueError(
            f"weighting={cs.weighting!r} 未实现：本签名没有波动率入参，risk_parity 算不出来。"
            f"静默退化成等权 = param_hash 写着 risk_parity 而跑的是 equal（D7 台账失真）")
    if cs.top_n <= 0:
        raise ValueError(f"top_n={cs.top_n} 无意义")
    pi = float(target_position)
    if pi < 0:
        raise ValueError(f"target_position={pi} < 0：A 股纯多头，不存在负总仓位")

    warns: list[str] = []
    prev = pd.Series(prev_weights, dtype=float).fillna(0.0)
    sc = pd.Series(scores, dtype=float)
    valid = sc[np.isfinite(sc.to_numpy())]

    # ── 数据中断：维持上期账本原样，不清仓（模块头 ★3）──
    if len(sc) == 0 or len(valid) < _MIN_SCORE_COVERAGE * len(sc):
        cov = len(valid) / len(sc) if len(sc) else 0.0
        warns.append(f"评分覆盖率 {cov:.0%} < {_MIN_SCORE_COVERAGE:.0%}，判为数据中断："
                     f"维持上期 {int((prev != 0).sum())} 只持仓不调仓")
        return prev.rename("target_weight"), warns

    # ── 选股：等权 + 行业上限。一次贪心扫描 == brief 的「删该行业末位 + 池中下一名替补」
    #    迭代到不动点：终态里每个行业留下的必是它分数最高的那 per_ind 只，替补顺序即分数序。
    #    （所以也不需要那个"最多 10 轮"的轮次上限，单次扫描已是终态。）──
    n_target = min(cs.top_n, len(valid))
    w_unit = min(pi / n_target, cs.max_single)          # ← 步 4 在等权下可达的唯一形态
    per_ind = len(valid) if w_unit <= 0 else int(math.floor(cs.max_industry / w_unit + _CAP_EPS))
    ind_valid = _industries(industry, valid.index)

    book: list = []
    used: dict = {}
    for code in valid.sort_values(ascending=False, kind="mergesort").index:
        if len(book) >= n_target:
            break
        k = ind_valid[code]
        if used.get(k, 0) >= per_ind:
            continue                                     # 该行业已满 → 本只被"删"，继续找替补
        book.append(code)
        used[k] = used.get(k, 0) + 1

    invested = w_unit * len(book)
    if invested < pi - _SUM_EPS:
        warns.append(f"只建了 {len(book)}/{cs.top_n} 只（单股 {cs.max_single:.0%}/"
                     f"行业 {cs.max_industry:.0%} 上限所限），Σw={invested:.4f} < π={pi:.4f}，差额留现金")

    # ── 换手裁剪，并在裁剪后复查两条风险上限 ──
    idx = pd.Index(book, dtype=object).union(prev.index)
    tgt = pd.Series(w_unit, index=book, dtype=float).reindex(idx).fillna(0.0)
    prv = prev.reindex(idx).fillna(0.0)
    delta = tgt - prv
    key = valid.reindex(idx)
    ind_all = _industries(industry, idx)

    # 未成交的残余停在 w_prev 上，可能顶破上限（模块头 ★②）。把越界的那几笔改判为
    # 必须成交后重算 —— forced 单调增长，最坏情形全部入列即 final == tgt（步 1–3 保证其
    # 满足两条上限），故循环必在 ≤ len(idx)+1 轮内收敛，且收敛点一定存在。
    forced = pd.Index([], dtype=object)
    final = prv
    for _ in range(len(idx) + 1):
        final = _execute(prv, delta, cs.max_turnover, forced, key)
        new = _violators(final, ind_all, cs).difference(forced)
        if len(new) == 0:
            break
        forced = forced.union(new)

    turnover = float((final - prv).abs().sum())
    if turnover > cs.max_turnover + _SUM_EPS and float(prv.abs().sum()) > 0:
        why = "补足越界持仓" if len(forced) else "净仓位变动本身即需此换手"
        warns.append(f"换手 {turnover:.1%} > 上限 {cs.max_turnover:.1%}（{why}）："
                     f"风险上限与 Σw==π 优先于换手预算")

    out = final[(final != 0) | (prv != 0)].sort_index()
    return out.rename("target_weight"), warns
