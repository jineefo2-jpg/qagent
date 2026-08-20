"""组合构建（算法说明书 §7.2 / 设计规格 §6.2）。

本文件钉的是**裁决后的优先级**，不是 brief 的五步字面顺序 —— 那五步不可交换：
步 4 的再分配会顶破行业上限，步 5 的部分执行会把步 3/4 刚建立的两条上限一起破掉。
理由与证明见 `portfolio.build_targets` 的文档字符串。测试因此分三层：

  1. 【硬】风险上限 max_single / max_industry —— 任何返回值都必须满足，无例外；
  2. 【硬】Σw == target_position —— 除非账本填不满（名额不够 / 行业上限拦住），此时留现金 + warning；
  3. 【软】max_turnover —— 与前两条冲突时让路，让路必须出现在 warnings 里。

★ 最容易写错、也最值钱的三条：
  - `test_all_nan_scores_holds_the_book`：全 NaN 返回 prev【原样】而不是空 Series。
    空 Series 读作清仓；数据中断常与极端行情同期，回测里会伪装成「暴跌前防御性离场」的假净值。
  - `test_turnover_is_measured_against_drifted_prev_weights`：换手是【实际要成交的量】，
    对着上期目标权重量会系统性错量，而换手直接进交易成本，错量直接落到净额收益上。
  - `test_partial_execution_never_leaves_*`：换手裁剪留下的未成交残余会顶破风险上限，
    此时补足那几笔并超预算告警，而不是"上限破了就破了"。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from ashare.backtest.portfolio import build_targets
from ashare.backtest.types import PortfolioConstraints

TOL = 1e-9


def _s(d) -> pd.Series:
    return pd.Series(d, dtype=float)


def _solo(codes) -> pd.Series:
    """每只股票自成一个行业 —— 把行业约束从这条测试里摘掉。"""
    return pd.Series({c: f"I_{c}" for c in codes})


def _ind(*groups) -> pd.Series:
    """`_ind(("X", "A", "B"), ("Y", "C"))` → {A: X, B: X, C: Y}"""
    return pd.Series({c: g[0] for g in groups for c in g[1:]})


def _wt(w: pd.Series, code: str) -> float:
    """缺席与 0 同义（都不持有）；只有"上期持有、本期清掉"才必须显式写 0。"""
    return float(w.get(code, 0.0))


def _l1(w: pd.Series, prev: pd.Series) -> float:
    idx = w.index.union(prev.index)
    return float((w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)).abs().sum())


def _assert_risk_caps(w: pd.Series, ind: pd.Series, cs: PortfolioConstraints) -> None:
    """两条【硬】上限。返回值不满足这两条，就是把集中度风险藏进了一条好看的净值曲线。"""
    assert (w >= -TOL).all(), "纯多头：不得出现负权重"
    if len(w):
        assert w.max() <= cs.max_single + TOL
        by_ind = w.groupby(ind.reindex(w.index).fillna("?")).sum()
        assert by_ind.max() <= cs.max_industry + TOL


# ── 选股与等权 ────────────────────────────────────────────────────────

def test_takes_the_top_n_by_score_and_weights_them_equally():
    sc = _s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
    cs = PortfolioConstraints(top_n=3, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 0.6, _s({}), _solo(sc.index), cs)

    assert list(w.index) == ["A", "B", "C"]
    assert (w - 0.2).abs().max() < TOL          # 0.6/3 在二进制里不是 0.2，容差不能省
    assert abs(w.sum() - 0.6) < TOL
    assert warns == []


def test_sum_equals_target_position_to_1e_9():
    sc = _s({f"S{i}": float(i) for i in range(7)})
    cs = PortfolioConstraints(top_n=7, max_single=0.5, max_industry=1.0)
    w, _ = build_targets(sc, 0.65, _s({}), _solo(sc.index), cs)
    assert abs(w.sum() - 0.65) < TOL          # 0.65/7 除不尽，1e-9 容差不是摆设


def test_top_n_larger_than_the_pool_scales_the_survivors_up():
    """名额 50 只有 3 只可投：按 π/3 建仓（仍在单股上限内），不是按 π/50 留 94% 现金。"""
    sc = _s({"A": 3.0, "B": 2.0, "C": 1.0})
    cs = PortfolioConstraints(top_n=50, max_single=0.05, max_industry=1.0)
    w, warns = build_targets(sc, 0.12, _s({}), _solo(sc.index), cs)

    assert w.eq(0.04).all()
    assert abs(w.sum() - 0.12) < TOL
    assert warns == []


# ── 行业上限 ──────────────────────────────────────────────────────────

def test_industry_cap_evicts_the_lowest_scoring_name_in_that_industry():
    """前 4 名里 X 占 3 只（0.75 > 0.5）→ 删 X 里分数最低的 C，用池中下一名 E 替补。"""
    sc = _s({"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0, "E": 5.0, "F": 4.0})
    ind = _ind(("X", "A", "B", "C"), ("Y", "D", "E"), ("Z", "F"))
    cs = PortfolioConstraints(top_n=4, max_single=0.5, max_industry=0.5)
    w, warns = build_targets(sc, 1.0, _s({}), ind, cs)

    assert sorted(w.index) == ["A", "B", "D", "E"]     # C 被删（X 里最低），E 替补
    assert w.eq(0.25).all()
    assert abs(w.sum() - 1.0) < TOL
    _assert_risk_caps(w, ind, cs)
    assert warns == []


def test_industry_cap_stops_backfilling_when_every_industry_is_full():
    """12 只 X + 8 只 Y，每行业最多容 3 只（0.3/0.1）→ 只能建 6 只，其余留现金。"""
    sc = _s({f"S{i:02d}": float(20 - i) for i in range(20)})
    ind = pd.Series({c: ("X" if i < 12 else "Y") for i, c in enumerate(sc.index)})
    cs = PortfolioConstraints(top_n=10, max_single=0.5, max_industry=0.3)
    w, warns = build_targets(sc, 1.0, _s({}), ind, cs)

    assert sorted(w.index) == ["S00", "S01", "S02", "S12", "S13", "S14"]
    assert abs(w.sum() - 0.6) < TOL
    _assert_risk_caps(w, ind, cs)
    assert any("现金" in x for x in warns), warns


def test_single_industry_universe_holds_cash_instead_of_breaking_the_cap():
    """全池一个行业 → 怎么替补都填不满。留现金 + warning，不是把行业上限放掉。

    「超出则放宽 N」在这里无解：等权下 Σ_k = n_k·π/N，可行性条件 K·max_industry ≥ π
    与 N 无关（放大 N 是尺度不变的）。所以逃生口只能是【少建几只 + 留现金】。
    """
    sc = _s({"A": 3.0, "B": 2.0, "C": 1.0})
    ind = _ind(("X", "A", "B", "C"))
    cs = PortfolioConstraints(top_n=3, max_single=1.0, max_industry=0.4)
    w, warns = build_targets(sc, 0.9, _s({}), ind, cs)

    assert list(w.index) == ["A"]              # π/N = 0.3，行业只容得下 1 只
    assert abs(w.sum() - 0.3) < TOL
    _assert_risk_caps(w, ind, cs)
    assert any("现金" in x for x in warns), warns


# ── 单股上限 ──────────────────────────────────────────────────────────

def test_single_cap_binds_when_top_n_is_too_small_and_the_rest_is_cash():
    """π/N = 0.10 > max_single = 0.05：等权下【所有】股票同时越界，没有"剩余股票"可供再分配。

    brief 步 4 的「截断后按 scores 比例再分配给剩余股票」在等权下不可达 —— 权重恒等，
    要么全越界要么全不越界。可达的只有这个：按上限截断、差额留现金、告警。
    """
    sc = _s({f"S{i}": float(i) for i in range(10)})
    cs = PortfolioConstraints(top_n=10, max_single=0.05, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)

    assert w.eq(0.05).all()
    assert abs(w.sum() - 0.5) < TOL            # 0.5 现金 —— Σw == π 在填不满时让位于单股上限
    assert any("现金" in x for x in warns), warns


def test_max_weight_never_exceeds_max_single():
    sc = _s({"A": 2.0, "B": 1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.3, max_industry=1.0)
    w, _ = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)
    assert w.max() <= 0.3 + TOL


# ── NaN / 数据中断 ────────────────────────────────────────────────────

def test_all_nan_scores_holds_the_book():
    """全 NaN 是数据中断，不是清仓信号。返回 prev【原样】—— 空 Series 读作清仓。"""
    prev = _s({"A": 0.3, "B": 0.2})
    sc = pd.Series({"A": np.nan, "B": np.nan, "C": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, prev, _solo(sc.index), cs)

    pd.testing.assert_series_equal(w, prev, check_names=False)
    assert len(w) > 0 and abs(w.sum() - prev.sum()) < TOL
    assert any("中断" in x for x in warns), warns


def test_all_nan_scores_with_no_prior_book_returns_empty_without_raising():
    sc = pd.Series({"A": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)
    assert len(w) == 0 and warns


def test_empty_scores_holds_the_book():
    prev = _s({"A": 0.4})
    w, warns = build_targets(_s({}), 1.0, prev, _s({}), PortfolioConstraints())
    pd.testing.assert_series_equal(w, prev, check_names=False)
    assert warns


def test_partial_nan_above_the_coverage_floor_ranks_the_survivors():
    """7/10 有分 → 正常出票，只是从这 7 只里选。稀疏横截面不等于数据中断。"""
    sc = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0, "F": 0.5, "G": 0.1,
                    "H": np.nan, "I": np.nan, "J": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=3, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 0.6, _s({}), _solo(sc.index), cs)

    assert list(w.index) == ["A", "B", "C"]
    assert warns == []


def test_coverage_below_the_floor_is_treated_as_an_outage():
    """4/10 有分：这不是"横截面变瘦"，是取数塌了。冻结账本，而不是拿 4 只票重建组合。"""
    prev = _s({"Z": 0.5})
    sc = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0,
                    "E": np.nan, "F": np.nan, "G": np.nan,
                    "H": np.nan, "I": np.nan, "J": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=3, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, prev, _solo(sc.index), cs)

    pd.testing.assert_series_equal(w, prev, check_names=False)
    assert any("覆盖率" in x for x in warns), warns


def test_infinite_scores_are_not_scores():
    """±inf 是除零 / 空窗口的产物，不是分数。当缺失处理 —— 否则它必然排第一名。"""
    sc = pd.Series({"A": np.inf, "B": 3.0, "C": 2.0, "D": 1.0}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    w, _ = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)

    assert list(w.index) == ["B", "C"]         # 覆盖率 3/4 ≥ 50%，正常出票；A 不在其中
    assert _wt(w, "A") == 0.0


# ── 换手 ──────────────────────────────────────────────────────────────

def test_no_clipping_when_the_move_fits_the_budget():
    prev = _s({"A": 0.5, "B": 0.5})
    sc = _s({"A": 2.0, "B": 1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.6, max_industry=1.0, max_turnover=0.3)
    w, warns = build_targets(sc, 1.0, prev, _solo(sc.index), cs)

    pd.testing.assert_series_equal(w.sort_index(), _s({"A": 0.5, "B": 0.5}), check_names=False)
    assert warns == []


def test_turnover_is_clipped_to_the_budget_and_the_sum_still_lands_on_target():
    """裁剪必须【买卖成对】收缩：只砍单边会让 Σw 偏离 π。这里 ν=0，成交额正好等于 τ。"""
    prev = _s({"A": 0.05, "B": 0.02, "C": 0.03, "D": 0.04})
    sc = _s({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0, max_turnover=0.06)
    w, warns = build_targets(sc, 0.14, prev, _solo(sc.index), cs)

    assert abs(w.sum() - 0.14) < TOL
    assert abs(_l1(w, prev) - 0.06) < TOL      # 预算用满，不是保守地少做
    assert warns == []


def test_the_largest_delta_trades_execute_first():
    """预算只够一半：卖出先做 |Δw| 最大的 D，C 一股没动；买入先做 B，A 一股没动。"""
    prev = _s({"A": 0.05, "B": 0.02, "C": 0.03, "D": 0.04})
    sc = _s({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0, max_turnover=0.06)
    w, _ = build_targets(sc, 0.14, prev, _solo(sc.index), cs)

    assert abs(w["D"] - 0.01) < TOL            # |Δ|=0.04 最大 → 吃掉全部卖出预算 0.03
    assert abs(w["C"] - 0.03) < TOL            # |Δ|=0.03 次之 → 没动
    assert abs(w["B"] - 0.05) < TOL            # |Δ|=0.05 最大 → 吃掉全部买入预算 0.03
    assert abs(w["A"] - 0.05) < TOL            # |Δ|=0.02 → 没动


def test_equal_delta_trades_break_the_tie_by_score():
    """等权组合里【所有新建仓的 Δw 都相等】—— 平手是常态不是边角料。
    预算只够一只时买分数最高的那只，否则"先做最重要的调整"在建仓侧等于随机。"""
    prev = _s({"Z": 0.10})
    sc = _s({"A": 3.0, "B": 2.0, "Z": -1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0, max_turnover=0.10)
    w, _ = build_targets(sc, 0.10, prev, _solo(sc.index), cs)

    assert abs(_wt(w, "A") - 0.05) < TOL       # A、B 的 Δw 都是 +0.05，预算只够一只
    assert abs(_wt(w, "B")) < TOL
    assert abs(_wt(w, "Z") - 0.05) < TOL


def test_equal_delta_sells_break_the_tie_by_selling_the_worse_score_first():
    """卖出侧同额并列时先卖分数低的那只。清仓（分数已算不出）的排在最前面。"""
    prev = _s({"P": 0.05, "Q": 0.05, "R": 0.05})
    sc = _s({"R": 5.0, "P": 2.0, "Q": 1.0})
    cs = PortfolioConstraints(top_n=1, max_single=0.5, max_industry=1.0, max_turnover=0.10)
    w, _ = build_targets(sc, 0.15, prev, _solo(sc.index), cs)

    assert abs(_wt(w, "R") - 0.10) < TOL       # 预算 0.10 拆成买 0.05 / 卖 0.05（成对收缩）
    assert abs(_wt(w, "Q")) < TOL              # P、Q 的 Δw 都是 −0.05，先卖分数低的 Q
    assert abs(_wt(w, "P") - 0.05) < TOL


def test_unscored_holdings_are_sold_before_equally_sized_scored_ones():
    """同额并列时，分数已经算不出的持仓排最前面卖 —— 那正是最该退出、也最没把握的一批。"""
    prev = _s({"GONE": 0.05, "P": 0.05, "R": 0.05})
    sc = pd.Series({"R": 5.0, "P": 2.0, "GONE": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=1, max_single=0.5, max_industry=1.0, max_turnover=0.10)
    w, _ = build_targets(sc, 0.15, prev, _solo(sc.index), cs)

    assert abs(_wt(w, "GONE")) < TOL
    assert abs(_wt(w, "P") - 0.05) < TOL


def test_empty_prev_weights_does_not_block_the_initial_build():
    """首期建仓换手必然是 π（100% > 30%），换手约束不得把建仓拦成 30% 仓位；
    也不该为此刷一条告警 —— 没有上期账本，就没有"超额换手"这回事。"""
    sc = _s({f"S{i}": float(i) for i in range(5)})
    cs = PortfolioConstraints(top_n=5, max_single=0.5, max_industry=1.0, max_turnover=0.3)
    w, warns = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)

    assert abs(w.sum() - 1.0) < TOL
    assert _l1(w, _s({})) > 0.3
    assert warns == []


def test_turnover_may_exceed_the_budget_when_the_mandate_requires_it():
    """||w−w_prev||₁ ≥ |π − Σw_prev| 是恒等式：0.2 加到 1.0 至少要 0.8 换手，两条约束
    在此数学上不相容。Σw == π 优先（多花的成本会被成本模型如实计价，仓位错了却没人计价）。"""
    prev = _s({"A": 0.10, "B": 0.10})
    sc = _s({"A": 3.0, "B": 2.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.6, max_industry=1.0, max_turnover=0.3)
    w, warns = build_targets(sc, 1.0, prev, _solo(sc.index), cs)

    assert abs(w.sum() - 1.0) < TOL
    assert abs(_l1(w, prev) - 0.8) < TOL       # 只超到"刚好够"，不多做一分
    assert any("换手" in x for x in warns), warns


def test_turnover_is_measured_against_drifted_prev_weights():
    """prev_weights 是【价格漂移后的实际权重】，不是上期的目标权重。

    上期目标 A=B=C=0.20（π=0.60，另有 0.40 现金）；本周 A 涨到 3 倍 → 实际权重
    A=0.4286 / B=C=0.1429，Σ=0.7143 ≠ π。换手是"真要成交的量"，只能对着实际权重量；
    对着上期目标量会系统性错量，而换手直接进成本，错量直接落到净额收益上。
    """
    stale = _s({"A": 0.20, "B": 0.20, "C": 0.20})              # 上期【目标】，不是本函数的入参
    drifted = _s({"A": 0.60, "B": 0.20, "C": 0.20}) / 1.4      # 实际持仓市值 / 权益
    sc = _s({"A": 3.0, "B": 2.0, "C": 1.0})
    cs = PortfolioConstraints(top_n=3, max_single=0.5, max_industry=1.0, max_turnover=0.15)
    w, _ = build_targets(sc, 0.60, drifted, _solo(sc.index), cs)

    assert abs(w.sum() - 0.60) < TOL
    assert abs(_l1(w, drifted) - 0.15) < TOL   # 预算恰好用满，且量在【漂移后】的权重上
    assert _l1(w, stale) > 0.15 + TOL          # 对着上期目标量出来的是另一个数（0.193）


# ── 换手裁剪 × 风险上限：五步不可交换的那个交叉点 ────────────────────

def test_partial_execution_never_leaves_a_position_above_max_single():
    """S11 漂移到 30%（上限 10%），换手预算只有 2% —— 未成交的残余会顶破单股上限。

    优先级：风险上限【硬】> 换手预算【软】。多花的换手会被成本模型如实扣掉（可见），
    集中度破了却没有任何指标会叫（不可见）—— 让可见的那个让路。
    S10 是普通换仓卖出（不越界）→ 应当仍被跳过，证明这是"最小补足"而不是"整体全量执行"。
    """
    prev = _s({f"S{i}": 0.10 for i in range(6)} | {"S10": 0.10, "S11": 0.30})
    sc = _s({f"S{i}": float(20 - i) for i in range(12)})
    ind = _solo(sc.index)
    cs = PortfolioConstraints(top_n=10, max_single=0.10, max_industry=1.0, max_turnover=0.02)
    w, warns = build_targets(sc, 1.0, prev, ind, cs)

    _assert_risk_caps(w, ind, cs)
    assert abs(w.sum() - 1.0) < TOL
    assert abs(_wt(w, "S11")) < TOL            # 越界的那只被补足到目标（清掉）
    assert abs(_wt(w, "S10") - 0.10) < TOL     # 不越界的换仓卖出仍被预算拦下
    assert any("换手" in x for x in warns), warns


def test_partial_execution_never_leaves_an_industry_above_max_industry():
    """X 行业漂移到 55%（上限 50%），本期目标 50%，但换手预算只有 1%。"""
    prev = _s({"A": 0.30, "B": 0.25, "C": 0.25, "D": 0.20})
    sc = _s({"A": 1.0, "B": 2.0, "C": 4.0, "D": 3.0})
    ind = _ind(("X", "A", "B"), ("Y", "C", "D"))
    cs = PortfolioConstraints(top_n=4, max_single=0.30, max_industry=0.50, max_turnover=0.01)
    w, warns = build_targets(sc, 1.0, prev, ind, cs)

    _assert_risk_caps(w, ind, cs)
    assert abs(w.sum() - 1.0) < TOL
    assert any("换手" in x for x in warns), warns


@pytest.mark.parametrize("seed", range(30))
def test_all_hard_constraints_hold_jointly_on_random_books(seed):
    """随机盘口的联合断言 —— 顺序执行【不】保证联合满足，所以这条必须随机。"""
    rng = np.random.default_rng(seed)
    codes = [f"S{i:02d}" for i in range(24)]
    sc = pd.Series(rng.normal(size=len(codes)), index=codes)
    sc[rng.random(len(codes)) < 0.15] = np.nan
    ind = pd.Series(rng.choice(list("XYZW"), size=len(codes)), index=codes)
    held = rng.random(len(codes)) < 0.4
    prev = pd.Series(np.where(held, rng.random(len(codes)) * 0.12, 0.0), index=codes)
    pi = float(rng.choice([0.2, 0.6, 1.0]))
    cs = PortfolioConstraints(top_n=int(rng.integers(4, 12)), max_single=0.10,
                              max_industry=0.35, max_turnover=float(rng.choice([0.05, 0.3, 2.0])))
    w, warns = build_targets(sc, pi, prev, ind, cs)

    if np.isfinite(sc).sum() < 0.5 * len(sc):          # 数据中断分支：原样返回 prev，不受约束管辖
        pd.testing.assert_series_equal(w, prev, check_names=False)
        assert warns
        return
    _assert_risk_caps(w, ind, cs)
    assert w.sum() <= pi + TOL
    if abs(w.sum() - pi) > TOL:                        # 没填满 → 必须说出来
        assert warns
    if _l1(w, prev) > cs.max_turnover + TOL:           # 超预算 → 必须说出来
        assert warns


# ── 入参守卫 ──────────────────────────────────────────────────────────

def test_risk_parity_weighting_is_refused_not_silently_equal_weighted():
    """签名里没有 σ，风险平价做不出来。静默退化成等权 = param_hash 写着 risk_parity、
    跑出来却是 equal —— D7 台账里凭空多一条不存在的实验。"""
    cs = PortfolioConstraints(top_n=2, weighting="risk_parity")
    with pytest.raises(ValueError, match="risk_parity"):
        build_targets(_s({"A": 1.0, "B": 2.0}), 1.0, _s({}), _solo(["A", "B"]), cs)


def test_negative_target_position_is_refused():
    with pytest.raises(ValueError):
        build_targets(_s({"A": 1.0}), -0.1, _s({}), _solo(["A"]), PortfolioConstraints(top_n=1))


def test_names_without_an_industry_share_one_conservative_bucket():
    """行业缺失不等于"不受行业约束"。归进同一个 unknown 桶（保守）—— 每只自成一行业的话
    行业上限对它们直接失效，而行业缺失恰恰常常是同一批数据出了问题。"""
    sc = _s({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    ind = pd.Series({"A": np.nan, "B": np.nan, "C": "Y", "D": "Y"}, dtype=object)
    cs = PortfolioConstraints(top_n=4, max_single=0.5, max_industry=0.25)
    w, _ = build_targets(sc, 1.0, _s({}), ind, cs)

    assert w.reindex(["A", "B"]).fillna(0.0).sum() <= 0.25 + TOL


def test_exits_are_returned_as_explicit_zero():
    """被清掉的票必须以 0.0 出现在返回值里 —— 缺席读作"没意见"，0 读作"卖光"。"""
    prev = _s({"OLD": 0.5, "KEEP": 0.5})
    sc = _s({"KEEP": 2.0, "OLD": 1.0})
    cs = PortfolioConstraints(top_n=1, max_single=1.0, max_industry=1.0, max_turnover=5.0)
    w, _ = build_targets(sc, 1.0, prev, _solo(prev.index), cs)

    assert "OLD" in w.index and abs(w["OLD"]) < TOL
    assert abs(w["KEEP"] - 1.0) < TOL
