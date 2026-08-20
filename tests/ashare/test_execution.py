"""成交模拟（铁律 D6 / 算法说明书 §5.1–5.3）—— 回测最容易骗自己的那一处。

本文件钉的四件事，每一件单独失守都能产出一条画得出来的假净值：

  1. 【日期】信号日 T 与执行日 τ 是两天。用 τ 的数据【决定】买什么、用 T 的价格【成交】，
     都是前视。所以两个日期是两个入参，`simulate` 只用 τ 取盘口、只用 T 做守卫。
  2. 【方向】不可交易是有方向的：涨停买不进但卖得出，跌停卖不出但买得进，停牌两侧都不行。
     把方向抹平（"不可交易 = 不动"）会让涨停日的减仓凭空消失。
  3. 【拦住 = 不成交】被拦的票停在原仓位上，不清零、不换个价格重试。于是组合与目标之间
     留下一个真实的缺口 —— 它必须出现在返回值里。每期都恰好达成目标权重的回测，
     描述的是一个不存在的市场。
  4. 【价格】成交价是执行日【开盘价】（D6：09:15–09:25 集合竞价），不是均价不是收盘价，
     且必须是后复权价（D8）。

★ 最值钱的三条：
  - `test_rescaling_a_sell_into_a_buy_is_caught`：再分配会把"小幅减仓"放大成"加仓"，
    若那只票正一字涨停，一次天真的实现就在涨停板上买进了。这是本文件唯一需要迭代求
    不动点的地方，也是最容易被写漏的。
  - `test_unknown_limit_is_untradable_end_to_end`：`limit_unknown` 走真库验一遍。
    P1 的 `compute_limits` 在退市整理期 / 新股 / 缺 pre_close 时返回 `(None, None,'unknown')`，
    把 unknown 读成"没涨停所以能买"是把最不该成交的那些日子全部计成成交。
  - `test_held_name_without_a_price_raises`：执行日没有行情的持仓 —— 无论标成上一次的价格、
    标成 0 还是留 NaN，都是在编一个不存在的数。D9 保证占位行存在，缺行就是库坏了，必须炸。
"""
from __future__ import annotations
import datetime as dt

import pandas as pd
import pytest

from ashare.backtest import execution
from ashare.backtest.execution import BLOCKED_COLS, TRADE_COLS, simulate

SIG = dt.date(2024, 1, 4)          # 信号日 T（周四收盘）
EXEC = dt.date(2024, 1, 5)         # 执行日 τ = T+1（周五开盘）
EQ = 1_000_000.0
TOL = 1e-9

MASK_COLS = ["can_buy", "can_sell", "reason", "open_hfq", "close_hfq", "amount", "amplitude"]


# ══════════════ 假掩码（单元层）══════════════
def _row(open_hfq, *, can_buy=True, can_sell=True, reason="", close_hfq=None):
    return {"can_buy": can_buy, "can_sell": can_sell, "reason": reason, "open_hfq": open_hfq,
            "close_hfq": open_hfq if close_hfq is None else close_hfq,
            "amount": 1e7, "amplitude": 0.03}


def _ok(px, close=None):      return _row(px, close_hfq=close)
def _up(px):                  return _row(px, can_buy=False, reason="limit_up_seal")
def _down(px):                return _row(px, can_sell=False, reason="limit_down_seal")
def _susp(px):                return _row(px, can_buy=False, can_sell=False, reason="suspended")
def _unknown(px):             return _row(px, can_buy=False, can_sell=False, reason="limit_unknown")
def _gone():                  return _row(float("nan"), can_buy=False, can_sell=False, reason="no_quote")
def _delisted(open_px, close):
    return _row(open_px, can_buy=False, can_sell=True, reason="delisted", close_hfq=close)


@pytest.fixture
def masked(monkeypatch):
    """装一个假 `get_tradable_mask`，并把它被调用时的 exec_date / ts_codes 记下来。"""
    seen: dict = {}

    def install(rows: dict) -> dict:
        frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=MASK_COLS).rename_axis("ts_code")

        def fake(exec_date, ts_codes):
            seen["exec_date"] = exec_date
            seen["ts_codes"] = list(ts_codes)
            return frame.reindex(list(ts_codes))

        monkeypatch.setattr(execution, "get_tradable_mask", fake)
        return seen

    return install


def _s(d) -> pd.Series:
    return pd.Series(d, dtype=float)


def _w(holdings: pd.Series, prices: dict) -> pd.Series:
    """持仓股数 → 权重，用于断言 Σw。"""
    return pd.Series({c: n * prices[c] / EQ for c, n in holdings.items()}, dtype=float)


# ══════════════ 1. 日期语义（D6 的第一半）══════════════
def test_mask_is_read_at_exec_date_not_signal_date(masked):
    seen = masked({"A.SZ": _ok(10.0)})
    simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), EQ, signal_date=SIG)
    assert seen["exec_date"] == EXEC


def test_signal_date_must_strictly_precede_exec_date(masked):
    """同日 = 用收盘后才算得出的信号在当天成交；倒置更荒谬。两者都必须炸而不是"结果差一点"。"""
    masked({"A.SZ": _ok(10.0)})
    for bad in (EXEC, dt.date(2024, 1, 8)):
        with pytest.raises(ValueError, match="signal_date"):
            simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), EQ, signal_date=bad)


def test_fill_price_is_exec_open_not_exec_close(masked):
    """D6：集合竞价定的是开盘价。收盘价是当天最后才知道的数，用它成交等于当天再前视一次。"""
    masked({"A.SZ": _ok(10.0, close=11.0)})
    trades, holdings, blocked, _ = simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), EQ, signal_date=SIG)
    assert trades.loc[0, "price_hfq"] == 10.0
    assert trades.loc[0, "price_hfq"] != 11.0
    assert holdings["A.SZ"] == pytest.approx(0.3 * EQ / 10.0)


# ══════════════ 2. 方向相关的不可交易 ══════════════
def test_limit_up_blocks_the_buy_and_leaves_the_position_alone(masked):
    """一字涨停：买不进（拦），卖得出（放行）。两只票同为 limit_up_seal，结局必须不同。"""
    masked({"A.SZ": _up(10.0), "B.SZ": _up(20.0)})
    prev = _s({"B.SZ": 1000.0})                       # B 现值 1000×20/1e6 = 0.02
    trades, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.3, "B.SZ": 0.0}), prev, EQ, signal_date=SIG)

    assert list(blocked["ts_code"]) == ["A.SZ"]
    assert blocked.loc[0, "intended_side"] == "BUY"
    assert blocked.loc[0, "reason"] == "limit_up_seal"
    assert blocked.loc[0, "intended_weight"] == pytest.approx(0.3)
    assert "A.SZ" not in set(trades["ts_code"])        # 没有以另一个价格重试
    assert "A.SZ" not in holdings.index                # 拦住 ≠ 建仓
    assert list(trades["ts_code"]) == ["B.SZ"] and trades.loc[0, "side"] == "SELL"


def test_limit_down_blocks_the_sell_but_not_the_buy(masked):
    masked({"A.SZ": _down(10.0), "B.SZ": _down(10.0)})
    prev = _s({"A.SZ": 1000.0})                        # A 现值 0.01，目标清仓 → 想卖
    trades, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.0, "B.SZ": 0.2}), prev, EQ, signal_date=SIG)

    assert list(blocked["ts_code"]) == ["A.SZ"] and blocked.loc[0, "intended_side"] == "SELL"
    assert blocked.loc[0, "intended_weight"] == pytest.approx(0.01)   # 幅值，方向在 side 上
    assert holdings["A.SZ"] == 1000.0                  # 逐位不变，不是"约等于"
    assert list(trades["ts_code"]) == ["B.SZ"] and trades.loc[0, "side"] == "BUY"


@pytest.mark.parametrize("state,reason", [(_susp, "suspended"), (_unknown, "limit_unknown")])
def test_suspended_and_unknown_block_both_directions(masked, state, reason):
    """停牌与涨跌停算不出：两侧都不可交易。unknown 当成"没涨停所以能买"是最贵的一种乐观。"""
    masked({"A.SZ": state(10.0), "B.SZ": state(10.0)})
    prev = _s({"B.SZ": 1000.0})
    trades, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.3, "B.SZ": 0.0}), prev, EQ, signal_date=SIG)

    assert len(trades) == 0
    assert set(blocked["ts_code"]) == {"A.SZ", "B.SZ"}
    assert set(blocked["reason"]) == {reason}
    assert dict(zip(blocked["ts_code"], blocked["intended_side"])) == {"A.SZ": "BUY", "B.SZ": "SELL"}
    assert holdings["B.SZ"] == 1000.0 and "A.SZ" not in holdings.index


# ══════════════ 3. 权重再分配（§5.3）══════════════
def test_locked_weight_is_redistributed_by_relative_target(masked):
    """3 只 1 只锁定：可交易的两只按【目标的相对比例】吃掉 (π − L)，Σw 仍等于 π。"""
    masked({"A.SZ": _susp(10.0), "B.SZ": _ok(10.0), "C.SZ": _ok(10.0)})
    prev = _s({"A.SZ": 2000.0})                        # 锁定 L = 2000×10/1e6 = 0.02
    _, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.2, "B.SZ": 0.2, "C.SZ": 0.4}), prev, EQ, signal_date=SIG)

    w = _w(holdings, {"A.SZ": 10.0, "B.SZ": 10.0, "C.SZ": 10.0})
    assert w["A.SZ"] == pytest.approx(0.02)
    assert w["B.SZ"] == pytest.approx(0.2 / 0.6 * 0.78)
    assert w["C.SZ"] == pytest.approx(0.4 / 0.6 * 0.78)
    assert w["C.SZ"] / w["B.SZ"] == pytest.approx(2.0)  # 相对比例守住
    assert w.sum() == pytest.approx(0.8, abs=TOL)       # == π
    assert list(blocked["ts_code"]) == ["A.SZ"]


def test_sum_of_weights_is_target_position_not_one(masked):
    """不归一化到 1 —— 组合允许持现金。归一化会把 60% 仓位的回测悄悄变成满仓的回测。"""
    masked({"A.SZ": _ok(10.0), "B.SZ": _ok(10.0)})
    _, holdings, _, _ = simulate(EXEC, _s({"A.SZ": 0.2, "B.SZ": 0.4}), _s({}), EQ, signal_date=SIG)
    assert _w(holdings, {"A.SZ": 10.0, "B.SZ": 10.0}).sum() == pytest.approx(0.6, abs=TOL)


def test_locked_over_target_zeroes_the_tradable_side_without_forced_sale(masked):
    """L > π：可交易部分清零，但【不强行卖出】锁定仓位 —— 卖不出就是卖不出。"""
    masked({"A.SZ": _susp(10.0), "B.SZ": _ok(10.0)})
    prev = _s({"A.SZ": 50_000.0})                      # L = 0.5 > π = 0.3
    trades, holdings, blocked, warns = simulate(
        EXEC, _s({"A.SZ": 0.1, "B.SZ": 0.2}), prev, EQ, signal_date=SIG)

    assert len(trades) == 0
    assert holdings["A.SZ"] == 50_000.0 and "B.SZ" not in holdings.index
    assert list(blocked["ts_code"]) == ["A.SZ"]
    assert any("锁定" in w for w in warns)               # 这一天的组合与目标毫无关系，必须看得见


def test_rescaling_a_sell_into_a_buy_is_caught(masked):
    """再分配把 B 的"小幅减仓"放大成"加仓"，而 B 正一字涨停 —— 不能在涨停板上买进。

    A 停牌吃不下 0.5 的目标 → 额度外溢 → 可交易部分 ×2 → B 从 0.30 被推到 0.50。
    正确结果：B 改判为不可交易、停在 0.30，额度再溢给 C。
    """
    masked({"A.SZ": _susp(10.0), "B.SZ": _up(10.0), "C.SZ": _ok(10.0)})
    prev = _s({"B.SZ": 30_000.0})                      # B 现值 0.30，目标 0.25 → 初判"想卖"
    trades, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.5, "B.SZ": 0.25, "C.SZ": 0.25}), prev, EQ, signal_date=SIG)

    assert "B.SZ" not in set(trades["ts_code"])
    assert holdings["B.SZ"] == 30_000.0
    b = blocked.set_index("ts_code")
    assert b.loc["B.SZ", "intended_side"] == "BUY"      # 被拦下的是【再分配后】那笔买入
    assert b.loc["B.SZ", "reason"] == "limit_up_seal"
    w = _w(holdings, {"B.SZ": 10.0, "C.SZ": 10.0})
    assert w["C.SZ"] == pytest.approx(0.70)             # (1.0 − 0.30) 全给 C
    assert w.sum() == pytest.approx(1.0, abs=TOL)


# ══════════════ 4. 退市（B8）══════════════
def test_delisted_is_liquidated_at_half_close_with_a_warning(masked):
    """退市整理期连续跌停、几乎无流动性：按收盘价成交是系统性乐观偏差（架构 B8）。"""
    masked({"A.SZ": _delisted(9.0, 8.0)})
    prev = _s({"A.SZ": 1000.0})
    trades, holdings, _, warns = simulate(EXEC, _s({"A.SZ": 0.0}), prev, EQ, signal_date=SIG)

    assert trades.loc[0, "price_hfq"] == pytest.approx(4.0)     # 8.0 × 0.5
    assert trades.loc[0, "price_hfq"] not in (8.0, 9.0)         # 既不是收盘也不是开盘
    assert trades.loc[0, "shares"] == pytest.approx(1000.0)
    assert trades.loc[0, "amount"] == pytest.approx(4000.0)
    assert "A.SZ" not in holdings.index
    assert any("A.SZ" in w for w in warns)


# ══════════════ 5. 执行日无行情 ══════════════
def test_unheld_name_without_a_quote_is_blocked_not_bought(masked):
    masked({"A.SZ": _gone()})
    trades, holdings, blocked, _ = simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), EQ, signal_date=SIG)
    assert len(trades) == 0 and len(holdings) == 0
    assert list(blocked["reason"]) == ["no_quote"]


def test_tradable_but_unpriced_name_is_still_blocked(masked):
    """掩码说可买、价格却是 NaN（`daily_bar.adj_factor` 可空，validate 显式容忍 NULL）。
    放过去成交股数就是 NaN，持仓 → 权益 → 净值一路染 NaN 且不抛。"""
    masked({"A.SZ": _row(float("nan"))})               # can_buy/can_sell 都是 True
    trades, holdings, blocked, _ = simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), EQ, signal_date=SIG)
    assert len(trades) == 0 and len(holdings) == 0
    assert list(blocked["reason"]) == ["no_price"]


def test_held_name_without_a_price_raises(masked):
    """持仓 + 执行日无价 = 这一天的组合权重根本算不出来。标成 0 是抹掉净值、
    标成上次价格是编造收益、留 NaN 是把 NaN 传染给整条曲线 —— 三条都不行，只能炸。"""
    masked({"A.SZ": _gone()})
    with pytest.raises(ValueError, match="A.SZ"):
        simulate(EXEC, _s({"A.SZ": 0.0}), _s({"A.SZ": 1000.0}), EQ, signal_date=SIG)


def test_held_delisted_name_without_last_close_raises(masked):
    """退市但连最后一根有效 K 线都取不到（close_hfq = NaN）→ 同一条守卫接住。"""
    masked({"A.SZ": _delisted(float("nan"), float("nan"))})
    with pytest.raises(ValueError, match="A.SZ"):
        simulate(EXEC, _s({"A.SZ": 0.0}), _s({"A.SZ": 1000.0}), EQ, signal_date=SIG)


# ══════════════ 6. 形状 / 边界 ══════════════
def test_frame_columns_are_the_contract(masked):
    masked({"A.SZ": _ok(10.0), "B.SZ": _susp(10.0)})
    trades, holdings, blocked, _ = simulate(
        EXEC, _s({"A.SZ": 0.3, "B.SZ": 0.3}), _s({}), EQ, signal_date=SIG)
    assert list(trades.columns) == TRADE_COLS
    assert list(blocked.columns) == BLOCKED_COLS
    assert set(trades["exec_date"]) == {EXEC} and set(blocked["exec_date"]) == {EXEC}
    assert holdings.index.name == "ts_code"


def test_empty_inputs_return_empty_frames_with_columns(masked):
    masked({})
    trades, holdings, blocked, warns = simulate(EXEC, _s({}), _s({}), EQ, signal_date=SIG)
    assert list(trades.columns) == TRADE_COLS and len(trades) == 0
    assert list(blocked.columns) == BLOCKED_COLS and len(blocked) == 0
    assert len(holdings) == 0 and warns == []


def test_fully_sold_name_leaves_holdings(masked):
    masked({"A.SZ": _ok(10.0)})
    _, holdings, _, _ = simulate(EXEC, _s({"A.SZ": 0.0}), _s({"A.SZ": 1000.0}), EQ, signal_date=SIG)
    assert "A.SZ" not in holdings.index


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_equity_must_be_positive(masked, bad):
    masked({"A.SZ": _ok(10.0)})
    with pytest.raises(ValueError, match="equity"):
        simulate(EXEC, _s({"A.SZ": 0.3}), _s({}), bad, signal_date=SIG)


# ══════════════ 7. 走真库的端到端（不打桩）══════════════
pytest.importorskip("duckdb")
from ashare.data import _db, query                                       # noqa: E402


def _set_bar(path, code, d, **cols):
    query.close_db()
    w = _db.connect_write(path)
    sets = ", ".join(f"{k} = ?" for k in cols)
    w.execute(f"UPDATE daily_bar SET {sets} WHERE ts_code = ? AND trade_date = ?", [*cols.values(), code, d])
    w.close()
    query.open_db(path)


@pytest.fixture
def live_db(market_db):
    query.open_db(market_db)
    yield market_db
    query.close_db()


def test_fill_price_is_exec_day_open_end_to_end(live_db):
    """四个候选价全部错开：只有【执行日开盘 × adj_factor】是对的。"""
    _set_bar(live_db, "B00002.SZ", SIG, open=19.0, high=19.6, low=18.9, close=19.5)
    _set_bar(live_db, "B00002.SZ", EXEC, open=21.0, high=23.5, low=20.5, close=23.0)
    adj_sig, adj_exec = 1.07, 1.08                       # fixture：adj = 1 + i×0.01，i(01-04)=7

    trades, _, blocked, _ = simulate(EXEC, _s({"B00002.SZ": 0.5}), _s({}), EQ, signal_date=SIG)

    assert len(blocked) == 0
    px = trades.loc[0, "price_hfq"]
    assert px == pytest.approx(21.0 * adj_exec)          # ← 唯一正确的那个
    for wrong in (23.0 * adj_exec, 19.0 * adj_sig, 19.5 * adj_sig, 21.0):
        assert abs(px - wrong) > 1e-6


def test_unknown_limit_is_untradable_end_to_end(live_db):
    """P1 的 compute_limits 在退市整理期 / 新股 / 缺 pre_close 时给 (None, None, 'unknown')，
    落到 daily_bar 就是 limit_up IS NULL。整条链走下来必须是"不可交易"，不是"没涨停所以能买"。"""
    _set_bar(live_db, "B00002.SZ", EXEC, limit_up=None, limit_down=None, limit_source="unknown")

    trades, holdings, blocked, _ = simulate(EXEC, _s({"B00002.SZ": 0.5}), _s({}), EQ, signal_date=SIG)

    assert len(trades) == 0 and len(holdings) == 0
    assert list(blocked["reason"]) == ["limit_unknown"]
