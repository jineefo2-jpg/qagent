"""组合构建（算法说明书 §7.2 / 设计规格 §6.2）。

本文件钉的是**裁决后的优先级**，不是 brief 的五步字面顺序 —— 那五步不可交换：
步 4 的再分配会顶破行业上限，步 5 的部分执行会把步 3/4 刚建立的两条上限一起破掉。
理由与证明见 `portfolio.build_targets` 的文档字符串。测试因此分三层：

  1. 【硬】风险上限 max_single / max_industry —— 任何返回值都必须满足，无例外；
  2. 【硬】Σw == target_position —— 除非账本填不满（名额不够 / 行业上限拦住），此时留现金 + warning；
  3. 【软】max_turnover —— 与前两条冲突时让路，让路必须出现在 warnings 里。

★ 最容易写错、也最值钱的四条：
  - `test_outage_returns_none_*`：中断日返回 `None`。空 Series 读作清仓（中断常与极端
    行情同期，回测里会伪装成「暴跌前防御性离场」的假净值）；`prev` 原样读作「这是你的
    目标」，而它是 T 收盘度量的、simulate 会在 τ 开盘把整夜跳空当成交易做掉。
  - `test_turnover_is_measured_against_drifted_prev_weights`：换手是【实际要成交的量】，
    对着上期目标权重量会系统性错量，而换手直接进交易成本，错量直接落到净额收益上。
  - `test_partial_execution_never_leaves_*`：换手裁剪留下的未成交残余会顶破风险上限，
    此时补足那几笔并超预算告警，而不是"上限破了就破了"。
  - `test_a_binding_turnover_budget_is_never_silent`：预算【绑定】时被裁掉的调仓不破坏
    任何一条硬约束，所以除了那条 warning 没有第二样东西能发现账本已经不是目标账本。
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


def test_tied_scores_pick_the_same_book_however_the_index_is_ordered():
    """分数打平时选谁，不能取决于调用方 index 的先后 —— 那样 D7 复现只是碰巧成立。

    精确的平手在这里是常态：§3.3 把算不出的 z-score 填成【正好 0】，§3.1 把离群值
    MAD 截到【正好】边界值。`get_universe` 今天返回排好序的 ts_code，但没有任何文档
    承诺这一点，靠它等于把可复现性押在一个上游实现细节上。
    （换手侧天生免疫：那边的 idx 出自 `Index.union`，本来就排过序 —— 正因如此这处
    不对称肉眼看不出来。）
    """
    ind = pd.Series({"AAA": "X", "BBB": "Y", "CCC": "Y"})
    cs = PortfolioConstraints(top_n=2, max_single=1.0, max_industry=0.5)
    books = []
    for order in (["AAA", "BBB", "CCC"], ["AAA", "CCC", "BBB"], ["CCC", "BBB", "AAA"]):
        sc = _s({"AAA": 5.0, "BBB": 3.0, "CCC": 3.0}).reindex(order)
        w, _ = build_targets(sc, 1.0, _s({}), ind, cs)
        books.append(sorted(w.index))
    assert books[0] == books[1] == books[2] == ["AAA", "BBB"]   # 平手取 ts_code 小者


def test_max_weight_never_exceeds_max_single():
    sc = _s({"A": 2.0, "B": 1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.3, max_industry=1.0)
    w, _ = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)
    assert w.max() <= 0.3 + TOL


# ── NaN / 数据中断 ────────────────────────────────────────────────────

def test_outage_returns_none_not_an_empty_book_and_not_prev():
    """中断日交出 `None`。三版口径里只有它读不歪 ——

    空 Series 读作【清仓】，而中断常与极端行情同期，回测里会长成「暴跌前防御性离场」；
    `prev` 原样读作【这是你的目标】，但它是 T 日收盘度量的权重，`simulate` 会在 τ 开盘
    重算漂移，于是在一个说了「今天不调仓」的日子里把整夜跳空当成交易做掉、还真扣成本。
    `None` 不等于引擎什么都不做：`simulate(targets=None)` 是「τ 开盘持平 + 只做强制退出」。
    """
    prev = _s({"A": 0.3, "B": 0.2})
    sc = pd.Series({"A": np.nan, "B": np.nan, "C": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, prev, _solo(sc.index), cs)

    assert w is None                            # 不是空 Series，也不是 prev
    assert any("中断" in x for x in warns), warns


def test_outage_returns_none_with_no_prior_book_too_without_raising():
    """首期就中断：仍然是 None。空 Series 在这里"看起来对"，但它与清仓无法区分。"""
    sc = pd.Series({"A": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    w, warns = build_targets(sc, 1.0, _s({}), _solo(sc.index), cs)
    assert w is None and warns


def test_empty_scores_is_an_outage_too():
    prev = _s({"A": 0.4})
    w, warns = build_targets(_s({}), 1.0, prev, _s({}), PortfolioConstraints())
    assert w is None and warns


def test_the_coverage_floor_sits_at_exactly_50_percent():
    """闸门常数本身要被钉住 —— 它挪到 42% 就是"拿 42% 的覆盖率照常下单"，正是亏钱那侧。

    50/100 建仓、49/100 冻结，把常数夹在 (0.49, 0.50]，同时钉死 `<` 不能写成 `<=`。
    （n=10 的 4 只/5 只钉不住 42%：0.42×10=4.2 与 0.5×10=5 之间没有整数，两者行为全同。）
    """
    cs = PortfolioConstraints(top_n=3, max_single=0.5, max_industry=1.0)

    def _run(n_valid: int):
        d = {f"S{i:03d}": (float(100 - i) if i < n_valid else np.nan) for i in range(100)}
        sc = pd.Series(d, dtype=float)
        return build_targets(sc, 0.6, _s({}), _solo(sc.index), cs)

    w_ok, _ = _run(50)
    assert w_ok is not None and list(w_ok.index) == ["S000", "S001", "S002"]
    w_out, warns = _run(49)
    assert w_out is None and any("覆盖率" in x for x in warns), warns


def test_a_dropna_ing_caller_is_called_out_because_it_blinds_the_coverage_gate():
    """`industry` 按完整股票池索引，`scores` 比它短 = 调用方先剔了 NaN。
    那样覆盖率恒等于 100%，上面那道闸门再也不会响 —— 这条约定不能只写在文档里。"""
    sc = _s({"A": 3.0, "B": 2.0})                    # 已 dropna
    ind = _solo(["A", "B", "C", "D"])                # 完整股票池
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    _, warns = build_targets(sc, 0.4, _s({}), ind, cs)
    assert any("失明" in x for x in warns), warns


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

    assert w is None
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
    # 用满 ≠ 做完：A/B 各差 0.02、C 一股没卖、D 只卖了 3/4。这里原本断言 `warns == []`，
    # 那正是被评审抓到的静默 —— 见下面 test_a_binding_turnover_budget_is_never_silent。
    assert any("拦下 4 笔" in x for x in warns), warns


def test_a_binding_turnover_budget_is_never_silent():
    """预算【绑定】时也必须告警，不只【超出】时 —— 否则整本账都能被换掉而无人察觉。

    评审的最小复现：目标是 {A, B}，实际交出的是 {A, Z} —— 账本留着【最想卖掉的】Z
    （分数最低），也没买进前二的 B。而 Σw = π、换手恰好等于 τ，**每一条硬约束都满足**，
    所以除了这条 warning 之外没有第二样东西能发现账本已经不是目标账本了。
    何况 `BacktestResult.positions.target_weight` 存的是【交出的】权重，
    意图中的账本哪儿都没留 —— 净值曲线因此无法归因。
    """
    prev = _s({"Z": 0.10})
    sc = _s({"A": 3.0, "B": 2.0, "Z": -1.0})
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0, max_turnover=0.10)
    w, warns = build_targets(sc, 0.10, prev, _solo(sc.index), cs)

    assert abs(w.sum() - 0.10) < TOL                     # Σw == π
    assert abs(_l1(w, prev) - 0.10) < TOL                # 换手恰好等于 τ，不超
    assert abs(_wt(w, "Z") - 0.05) < TOL                 # 最想卖的那只留下了一半
    assert abs(_wt(w, "B")) < TOL                        # 前二里的 B 一股没买
    assert any("拦下" in x and "L1=" in x for x in warns), warns


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


def test_a_hair_over_the_industry_cap_still_counts_as_over():
    """行业上限的越界判定容差只吞浮点噪声（1e-12），不吞真实的超配。

    X 漂到 20.05%（上限 20%）—— 0.05% 看着不值一提，但这是**检测阈值**：它一放宽，
    整个不动点循环就当这行业合规，超配会一路留在账上。单股侧已经被 CAP_EPS 的用例钉住，
    行业侧不能只靠"权重都是 w_unit 的整数倍"这个巧合。
    """
    prev = _s({"A": 0.1002, "B": 0.1003, "C": 0.0995, "D": 0.1000})   # Σ = π，ν = 0
    sc = _s({"C": 4.0, "D": 3.0, "A": 2.0, "B": 1.0})
    ind = _ind(("X", "A", "B"), ("Y", "C", "D"))
    cs = PortfolioConstraints(top_n=4, max_single=0.5, max_industry=0.20, max_turnover=0.0002)
    w, _ = build_targets(sc, 0.40, prev, ind, cs)

    _assert_risk_caps(w, ind, cs)
    assert w.groupby(ind.reindex(w.index)).sum()["X"] <= 0.20 + TOL


def test_turnover_a_hair_over_the_budget_still_warns():
    """告警阈值也是个可以被悄悄放宽的常数。ν = 0.32 对着 0.30 的预算只超 7% ——
    现有用例都超到 2 倍以上，把阈值改成 1.5×budget 照样全绿。"""
    prev = _s({"A": 0.10})
    cs = PortfolioConstraints(top_n=1, max_single=0.5, max_industry=1.0, max_turnover=0.30)
    w, warns = build_targets(_s({"A": 5.0}), 0.42, prev, _solo(["A"]), cs)

    assert abs(_l1(w, prev) - 0.32) < TOL
    assert any("> 上限" in x for x in warns), warns


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


def test_a_forced_pin_is_charged_against_the_budget_not_added_on_top():
    """被迫成交的那笔要**从**换手预算里扣，不是额外加在预算之上。

    V 漂到 0.12（上限 0.10），补足它花掉 0.02；剩下的自由调仓只剩 0.08 可用 ——
    买卖各 0.04（成对收缩）。总换手因此恰好 = τ = 0.10。
    把 `left = max(0, budget − forced)` 写成 `left = budget` 的话总换手变成 0.12：
    凭空多出的那 0.02 会被 §5.4 的成本模型如实扣进净值，是真金白银。
    """
    prev = _s({"V": 0.12, "C": 0.10, "D": 0.10} | {f"K{i}": 0.10 for i in range(7)})
    sc = _s({"V": 10.0, "A": 9.0, "B": 8.0, "C": -2.0, "D": -1.0}
            | {f"K{i}": float(7 - i) for i in range(7)})
    ind = _solo(sc.index)
    cs = PortfolioConstraints(top_n=10, max_single=0.10, max_industry=1.0, max_turnover=0.10)
    w, warns = build_targets(sc, 1.0, prev, ind, cs)

    _assert_risk_caps(w, ind, cs)
    assert abs(_wt(w, "V") - 0.10) < TOL       # 越界那只被补足（成交 0.02，无条件）
    assert abs(_l1(w, prev) - 0.10) < TOL      # 总换手 == τ，pin 不是额外配额
    assert abs(_wt(w, "C") - 0.06) < TOL       # 自由额度 0.08 → 卖 0.04
    assert abs(_wt(w, "A") - 0.04) < TOL       #              → 买 0.04
    assert not any("换手 " in x for x in warns), warns   # 没超预算，只是绑定


# 这批数字来自一次定向搜索（16 只票，见提交说明）：不动点循环要走 **13 轮**才收敛。
# 深循环是正常运转不是异常 —— 计划初稿的「最多 10 轮」在这里会静默交出越界的仓位。
_DEEP = {  # code: (score, prev, industry)
    "S00": (1.28, 0.04, "X"), "S01": (-0.66, 0.15, "Y"), "S02": (-0.21, 0.11, "X"),
    "S03": (0.87, 0.14, "X"), "S04": (1.02, 0.13, "Y"), "S05": (1.77, 0.02, "X"),
    "S06": (0.15, 0.13, "X"), "S07": (1.13, 0.12, "Y"), "S08": (2.90, 0.01, "Y"),
    "S09": (-0.43, 0.14, "Y"), "S10": (-1.12, 0.07, "X"), "S11": (-1.64, 0.12, "X"),
    "S12": (1.51, 0.10, "X"), "S13": (0.29, 0.12, "Y"), "S14": (-1.10, 0.13, "X"),
    "S15": (1.25, 0.11, "Y"),
}


def test_the_pin_cascade_runs_well_past_ten_rounds():
    """补足一笔越界会吃掉换手预算，于是下一只原本刚好合规的票掉回 w_prev 又越界 ——
    一轮解一只。实测这条链在 16 只票的账本上走满 13 轮。

    所以轮次上界只能是 `len(idx)+1`：写死 10 轮时本例返回的 S01 = 0.075 对着 0.05 的
    上限（超 50%），而其余用例全绿 —— 没有任何东西会叫。
    """
    sc = _s({k: v[0] for k, v in _DEEP.items()})
    prev = _s({k: v[1] for k, v in _DEEP.items()})
    ind = pd.Series({k: v[2] for k, v in _DEEP.items()})
    cs = PortfolioConstraints(top_n=5, max_single=0.05, max_industry=0.5, max_turnover=0.3)
    w, _ = build_targets(sc, 1.0, prev, ind, cs)

    _assert_risk_caps(w, ind, cs)
    assert _wt(w, "S01") <= 0.05 + TOL         # 10 轮版本在这只上交出 0.075


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

    if np.isfinite(sc).sum() < 0.5 * len(sc):          # 数据中断分支：该日不调仓
        assert w is None and warns
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


def test_nan_prev_weight_is_refused_not_read_as_zero():
    """NaN 的上期权重是「这只票的敞口算不出来」（停牌无价的持仓在
    `holdings × price / equity` 里就是 NaN），不是「没持有」。

    填 0 的后果是三连静默：该票从返回值里整个消失（最后一行按 `prv != 0` 保留）、
    于是永远不会被卖出、它的漂移也永远不计入换手。敞口路径上失败必须响。
    """
    prev = pd.Series({"HELD": 0.2, "SUSPENDED": np.nan}, dtype=float)
    cs = PortfolioConstraints(top_n=2, max_single=0.5, max_industry=1.0)
    with pytest.raises(ValueError, match="SUSPENDED"):
        build_targets(_s({"HELD": 1.0, "SUSPENDED": 2.0}), 0.4, prev,
                      _solo(["HELD", "SUSPENDED"]), cs)


def test_negative_target_position_is_refused():
    with pytest.raises(ValueError):
        build_targets(_s({"A": 1.0}), -0.1, _s({}), _solo(["A"]), PortfolioConstraints(top_n=1))


@pytest.mark.parametrize("top_n", [0, -1])
def test_nonpositive_top_n_is_refused(top_n):
    """没有这条用例时闸门是死的：拿掉它 top_n=0 也只是在 π/0 上抛 ZeroDivisionError。"""
    with pytest.raises(ValueError, match="top_n"):
        build_targets(_s({"A": 1.0}), 0.5, _s({}), _solo(["A"]),
                      PortfolioConstraints(top_n=top_n))


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
