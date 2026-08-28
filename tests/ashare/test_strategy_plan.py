"""P3 Task 3：§6.3 调仓清单。验收断言按计划原文：JSON 可序列化、双指纹在场且与
config/库一致、剔除 τ 日不可买标的、未校准场景警示语在场；另钉限价带夹逼、
current_weight 的市值口径、SELL 被挡改 blocked、数据中断不出清单。
"""
from __future__ import annotations
import datetime as dt
import json
import pathlib

import pandas as pd
import pytest

pytest.importorskip("duckdb")
from ashare.backtest.types import BacktestConfig, PortfolioConstraints
from ashare.data import query
from ashare.strategy import plan as sp

D = dt.date


# ══════════════ 纯函数 ══════════════
def test_band_math_clamps_into_limits_and_rounds():
    assert sp._band(100.0, 0.04, 91.0, 110.0) == [98.0, 102.0]          # 前收 ± 0.5×4%
    assert sp._band(100.0, 0.30, 91.0, 110.0) == [91.0, 110.0]          # 夹进涨跌停
    assert sp._band(100.0, float("nan"), 91.0, 110.0) is None           # ATR 算不出 → 无带
    assert sp._band(float("nan"), 0.04, 91.0, 110.0) is None
    assert sp._band(3.333, 0.03, float("nan"), float("nan")) == [3.28, 3.38]   # 无涨跌停可依 → 只有公式带


def test_contrib_takes_top3_by_abs_with_sign(monkeypatch):
    class S:                                    # 假 FactorSpec：只要 direction
        def __init__(self, d): self.direction = d
    monkeypatch.setattr(sp, "get_factor", lambda n: S(-1 if n == "rev" else 1))
    panel = pd.DataFrame({"rev": [2.0], "ep": [1.5], "bp": [0.1], "mom": [-1.8]},
                         index=["A.SH"])
    got = sp._contrib(panel, {"rev": 1.0, "ep": 1.0, "bp": 1.0, "mom": 1.0}, "A.SH")
    assert list(got) == ["rev", "mom", "ep"]                            # |w·d·z| 排序
    assert got["rev"] == -2.0 and got["mom"] == -1.8                    # 带符号（rev 方向 −1）


# ══════════════ 编排假世界 ══════════════
CFG = BacktestConfig(start=D(2019, 1, 4), end=D(2019, 12, 31),
                     factors=(("f1", 1.0),),
                     constraints=PortfolioConstraints(top_n=2, weighting="equal",
                                                      max_single=0.6, max_industry=1.0,
                                                      max_turnover=1.0))


@pytest.fixture()
def fake_world(monkeypatch):
    state = {"positions": (None, []), "prior": None, "data_end": D(2026, 8, 31),
             "targets": pd.Series({"A.SH": 0.5, "C.SH": 0.5, "D.SZ": 0.0})}
    monkeypatch.setattr(query, "snapshot_id", lambda *, pin=False: "snap-1")
    monkeypatch.setattr(query, "next_trade_date", lambda d, n=1: D(2026, 8, 31))
    # 默认行情覆盖到 τ → 历史回放路径；实盘用例自己把它调早
    monkeypatch.setattr(query, "last_data_date", lambda: state["data_end"])
    monkeypatch.setattr(query, "get_universe", lambda d, **k: ["A.SH", "B.SZ", "C.SH"])
    monkeypatch.setattr(query, "get_industry", lambda d, **k: pd.Series("行业", index=["A.SH", "B.SZ", "C.SH"]))
    monkeypatch.setattr(query, "get_stock_basic",
                        lambda d, codes=None: pd.DataFrame({"name": ["甲", "丙", "丁"]},
                                                           index=pd.Index(["A.SH", "C.SH", "D.SZ"], name="ts_code")))
    mask = pd.DataFrame({
        "can_buy":  [True, False, True],
        "can_sell": [True, True, False],
        "reason":   ["", "limit_up_seal", "suspended"],
        "pre_close_raw":  [100.0, 50.0, 20.0],
        "limit_up_raw":   [110.0, 55.0, 22.0],
        "limit_down_raw": [90.0, 45.0, 18.0],
        "close_raw":      [101.0, 51.0, 21.0],      # τ 的前收（实盘模式用它）
    }, index=pd.Index(["A.SH", "C.SH", "D.SZ"], name="ts_code"))
    monkeypatch.setattr(query, "get_tradable_mask", lambda d, codes: mask.reindex(codes))
    monkeypatch.setattr(sp, "_combine",
                        lambda w, d, u, use_store=False: (pd.Series(1.0, index=u), ["普通降级", "⚠ 大事一件"]))
    monkeypatch.setattr(sp, "build_targets",
                        lambda *a, **k: (state["targets"], state["targets"], []))
    monkeypatch.setattr(sp, "compute_panel",
                        lambda names, d, u, **k: (pd.DataFrame({"f1": [1.0, 1.0, 1.0]},
                                                               index=["A.SH", "B.SZ", "C.SH"]), []))
    monkeypatch.setattr(sp, "_atr20_ratio",
                        lambda d, codes: pd.Series(0.04, index=pd.Index(codes)))
    from ashare.data import ledger_store
    monkeypatch.setattr(ledger_store, "latest_positions", lambda: state["positions"])
    monkeypatch.setattr(ledger_store, "latest_signal_plan", lambda: state["prior"])

    class S:
        direction = 1
    monkeypatch.setattr(sp, "get_factor", lambda n: S())
    return state


def test_contract_fields_fingerprints_and_warning_filter(fake_world):
    got = sp.build_rebalance_plan("2026-08-28", CFG)
    json.dumps(got, ensure_ascii=False)                                  # 可序列化
    assert got["param_hash"] == CFG.param_hash() and got["data_snapshot_id"] == "snap-1"
    assert got["execute_on"] == "2026-08-31T09:15:00+08:00"
    assert got["as_of_note"] == sp.AS_OF_NOTE and got["execute_note"] == sp.EXECUTE_NOTE
    assert got["target_position"] == CFG.position_cap                    # macro_timing=False
    assert got["position_calibrated"] is True and sp.UNCALIBRATED_WARNING not in got["warnings"]
    assert "⚠ 大事一件" in got["warnings"] and "普通降级" not in got["warnings"]   # 只顶 ⚠ 级


def test_untradable_buy_is_excluded_and_blocked_sell_is_kept(fake_world):
    fake_world["positions"] = (D(2026, 8, 27), [{"ts_code": "D.SZ", "shares": 100, "avg_cost": 20, "source": "manual_confirm"}])
    got = sp.build_rebalance_plan("2026-08-28", CFG)
    by = {o["ts_code"]: o for o in got["orders"]}
    assert "C.SH" not in by                                              # 一字涨停买不进 → 剔除
    assert got["excluded"][0]["ts_code"] == "C.SH" and "limit_up_seal" in got["excluded"][0]["reason"]
    assert by["D.SZ"]["action"] == "SELL" and by["D.SZ"]["urgency"] == "blocked"   # 停牌卖不出 → 留单待顺延
    assert by["A.SH"]["action"] == "BUY"
    assert by["A.SH"]["limit_price_range"] == [98.0, 102.0]
    assert by["D.SZ"]["current_weight"] == pytest.approx(1.0)            # 唯一持仓 → 市值口径 100%


def test_stale_ledger_raises_uncalibrated_flag(fake_world):
    fake_world["prior"] = {"as_of": "2026-08-21", "target_position": 0.8}
    fake_world["positions"] = (D(2026, 8, 14), [{"ts_code": "A.SH", "shares": 10, "avg_cost": 99, "source": "reconcile_csv"}])
    got = sp.build_rebalance_plan("2026-08-28", CFG)
    assert got["position_calibrated"] is False
    assert sp.UNCALIBRATED_WARNING in got["warnings"]
    fake_world["positions"] = (D(2026, 8, 21), fake_world["positions"][1])
    assert sp.build_rebalance_plan("2026-08-28", CFG)["position_calibrated"] is True


def test_data_break_refuses_to_emit_a_plan(fake_world, monkeypatch):
    monkeypatch.setattr(sp, "build_targets", lambda *a, **k: (None, None, ["数据中断"]))
    with pytest.raises(ValueError, match="不出清单"):
        sp.build_rebalance_plan("2026-08-28", CFG)


# ══════════════ 真库冒烟 ══════════════
_MARKET = pathlib.Path("data/ashare_market.duckdb")


@pytest.mark.skipif(not _MARKET.exists(), reason="真实 market 库不存在")
def test_smoke_real_db_full_plan():
    from ashare.factors.base import list_factors, ALPHA_CATEGORIES
    query.close_db(); query.open_db(str(_MARKET))
    try:
        cfg = BacktestConfig(start=D(2019, 1, 4), end=D(2019, 12, 31),
                             factors=tuple((s.name, 1.0) for s in list_factors()
                                           if s.category in ALPHA_CATEGORIES))
        got = sp.build_rebalance_plan("2019-06-28", cfg)
        json.dumps(got, ensure_ascii=False)
        assert got["param_hash"] == cfg.param_hash() and got["data_snapshot_id"] == query.snapshot_id()
        assert len(got["orders"]) > 0
        for o in got["orders"]:
            assert o["action"] in ("BUY", "SELL", "ADJUST")
            if o["limit_price_range"] is not None:
                lo, hi = o["limit_price_range"]
                assert 0 < lo <= hi
            assert len(o["factor_contrib"]) <= 3
        tw = sum(o["target_weight"] for o in got["orders"])
        assert tw <= cfg.position_cap + 1e-6
    finally:
        query.close_db()


# ══════════════ Task 4 · CLI（唯一写库点）══════════════
def test_cli_persists_exports_and_unpins(fake_world, tmp_path, monkeypatch, capsys):
    calls = {"close": 0}
    monkeypatch.setattr(query, "open_db", lambda *a, **k: None)
    monkeypatch.setattr(query, "close_db", lambda: calls.__setitem__("close", calls["close"] + 1))
    saved: list = []
    from ashare.data import ledger_store
    monkeypatch.setattr(ledger_store, "save_signal_plan",
                        lambda p: (saved.append(p), len(saved) > 1)[1])
    rc = sp.main(["--as-of", "2026-08-28", "--out", str(tmp_path / "sig")])
    assert rc == 0 and calls["close"] == 1                     # 收尾必须解钉（引擎 ★9 同契约）
    data = json.loads((tmp_path / "sig" / "2026-08-28.json").read_text(encoding="utf-8"))
    assert data["param_hash"] == saved[0]["param_hash"]
    assert data["data_snapshot_id"] == "snap-1"
    sp.main(["--as-of", "2026-08-28", "--out", str(tmp_path / "sig")])
    assert "幂等覆盖" in capsys.readouterr().out               # 同参重发 = 覆盖并出声，不是新实验


def test_default_config_is_the_validated_arm():
    """生产配置必须【逐位】等于通过样本外闸 1 的那一臂（2026-08-28 仪式判定③）。

    D7 只给一次样本外机会且已用尽：这个哈希对不上 = 每晚的信号来自一个没有背书的
    配置，而报告上看不出任何异常。红了不许改这里的期望值 —— 要么把改动退回去，
    要么明确承认「信号不再有样本外背书」（那是人的决定，不是测试的）。"""
    from ashare.factors.base import list_factors, ALPHA_CATEGORIES
    ref = BacktestConfig(start=D(2010, 1, 1), end=D(2019, 12, 31), position_cap=0.8,
                         factors=tuple((s.name, 1.0) for s in list_factors()
                                       if s.category in ALPHA_CATEGORIES))
    cfg = sp.default_config()
    assert cfg.position_cap == sp.PRODUCTION_POSITION_CAP == 0.8
    assert cfg.macro_timing is False, "宏观层已按 §6.1 关停（样本外 Sharpe 未超恒定仓位）"
    assert cfg.param_hash() == ref.param_hash() == sp.VALIDATED_PARAM_HASH, (
        f"生产配置指纹 {cfg.param_hash()} ≠ 过闸的 {sp.VALIDATED_PARAM_HASH}")
    assert sp.default_config(top_n=30).constraints.top_n == 30
    assert sp.default_config(top_n=30).param_hash() != cfg.param_hash()   # 改参数 = 新指纹


def test_live_mode_does_not_exclude_the_whole_universe(fake_world):
    """实盘出清单时 τ 在未来、那天没有任何行情。掩码若按 τ 求值会全判 no_quote 而剔光
    整个池子 —— 2026-08-28 实测 0 笔订单 / 50 只剔除，报告读起来却像「策略今天什么都
    不该买」。把「未知」当成「不可交易」是错的：掩码改在 T 日求值（§6.2「预期」二字），
    限价带用 T 收盘且不夹涨跌停（τ 的板由 T 收盘推出，按 T 自己的板夹会夹错）。"""
    fake_world["data_end"] = D(2026, 8, 28)          # 行情止于 T，τ=08-31 在未来
    got = sp.build_rebalance_plan("2026-08-28", CFG)
    by = {o["ts_code"]: o for o in got["orders"]}
    assert by, "实盘模式把整个池子剔光了 —— 正是本用例要挡的回归"
    assert by["A.SH"]["limit_price_range"] == [98.98, 103.02]     # 101.0 × (1 ± 0.02)，未夹
    assert "尚未确定" in by["A.SH"]["price_basis"]
    assert any("尚无行情" in w for w in got["warnings"])
    # T 日一字封死仍然剔除 —— 那是「次日预期」的可得预测，不是「未知」
    assert "C.SH" in {e["ts_code"] for e in got["excluded"]}


def test_nightly_refuses_when_todays_bars_are_not_in_yet(monkeypatch, capsys):
    """nightly 必须在入口查「今天的行情入库了没」。不查的后果不是跳过，而是崩在因子
    计算深处（universe 为空），运维看到一句与病因无关的报错（2026-08-28 实测）。
    这里【失败】而不是跳过：定时链里 nightly 紧跟当日增量之后，数据不在 = 增量真出了
    问题，必须出声。绝不拿昨天的数据出今天的清单 —— 执行日会整体错位一天。"""
    today = dt.date.today()
    monkeypatch.setattr(query, "open_db", lambda *a, **k: None)
    monkeypatch.setattr(query, "close_db", lambda: None)
    monkeypatch.setattr(query, "get_trade_dates",
                        lambda d, start=None, freq="D": [today])          # 今天是交易日兼调仓日
    monkeypatch.setattr(query, "last_data_date", lambda: today - dt.timedelta(days=1))
    rc = sp.main(["--nightly"])
    out = capsys.readouterr().out
    assert rc == 1 and "尚未入库" in out and str(today - dt.timedelta(days=1)) in out
