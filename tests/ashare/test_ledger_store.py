"""P3 Task 0：ledger 库（信号/持仓/确认）—— 不可重算的用户资产。

与 derived（缓存，可 rm 重算）的本质区别由两条测试钉住：版本守卫存在、
双指纹缺一不写。其余是读写出口的行为契约。
"""
from __future__ import annotations
import datetime as dt

import pytest

pytest.importorskip("duckdb")
from ashare.data import _ledger, ledger_store

D = dt.date


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(_ledger, "DEFAULT_LEDGER_PATH", str(tmp_path / "ledger.duckdb"))


def _plan(as_of="2026-08-28", ph="ph-1", snap="snap-1"):
    return {"as_of": as_of, "param_hash": ph, "data_snapshot_id": snap,
            "execute_on": "2026-08-31T09:15:00+08:00", "strategy_version": "v1.0.0",
            "orders": [{"ts_code": "600519.SH", "action": "BUY", "target_weight": 0.02}]}


def test_version_guard_rejects_newer_db():
    """本库是用户资产不是缓存：库版本高于代码必须拒开（与 market 同款，derived 没有这道）。"""
    conn = _ledger.connect_write()
    _ledger.init_schema(conn)
    conn.execute("UPDATE _meta SET value='99' WHERE key='schema_version'")
    with pytest.raises(RuntimeError, match="schema_version"):
        _ledger.init_schema(conn)
    conn.close()


def test_plan_without_fingerprints_is_refused():
    """D7：双指纹缺一不写 —— 落一份无法溯源的清单比不落更坏。"""
    for k in ("param_hash", "data_snapshot_id", "execute_on"):
        p = _plan(); p[k] = ""
        with pytest.raises(ValueError, match="缺字段"):
            ledger_store.save_signal_plan(p)


def test_plan_roundtrip_and_idempotent_overwrite():
    assert ledger_store.save_signal_plan(_plan()) is False          # 首次：未覆盖
    assert ledger_store.save_signal_plan(_plan()) is True           # 同 (as_of, ph)：幂等覆盖
    ledger_store.save_signal_plan(_plan(ph="ph-2"))                 # 参数变了是新行（D7 台账连续）
    plans = ledger_store.list_signal_plans()
    assert len(plans) == 2
    latest = ledger_store.latest_signal_plan()
    assert latest["orders"][0]["ts_code"] == "600519.SH"
    assert latest["param_hash"] in ("ph-1", "ph-2")


def test_confirm_states_are_whitelisted():
    with pytest.raises(ValueError, match="state"):
        ledger_store.record_confirms("2026-08-31", [{"ts_code": "600519.SH", "state": "done"}])
    ledger_store.record_confirms("2026-08-31", [
        {"ts_code": "600519.SH", "state": "filled"},
        {"ts_code": "000001.SZ", "state": "partial", "filled_shares": 300, "note": "只成交一半"}])
    got = ledger_store.get_confirms("2026-08-31")
    assert {g["ts_code"]: g["state"] for g in got} == {"600519.SH": "filled", "000001.SZ": "partial"}
    # 改主意 = 同键覆盖
    ledger_store.record_confirms("2026-08-31", [{"ts_code": "600519.SH", "state": "skipped"}])
    assert ledger_store.get_confirms("2026-08-31")[1]["state"] == "skipped"


def test_positions_source_whitelist_and_full_day_snapshot():
    with pytest.raises(ValueError, match="source"):
        ledger_store.write_positions("2026-08-31", [], source="guess")
    ledger_store.write_positions("2026-08-31", [
        {"ts_code": "600519.SH", "shares": 100, "avg_cost": 1580.0}], source="reconcile_csv")
    # 整日整批：重写同一天是全量替换，不是增量叠加
    ledger_store.write_positions("2026-08-31", [
        {"ts_code": "000001.SZ", "shares": 500}], source="manual_confirm")
    d, rows = ledger_store.latest_positions()
    assert d == D(2026, 8, 31) and len(rows) == 1
    assert rows[0]["ts_code"] == "000001.SZ" and rows[0]["source"] == "manual_confirm"


def test_reads_survive_missing_db():
    """库不存在 = 还没生成过信号，读侧一律安静的空值，不炸。"""
    assert ledger_store.latest_signal_plan() is None
    assert ledger_store.list_signal_plans() == []
    assert ledger_store.latest_positions() == (None, [])
    assert ledger_store.get_confirms("2026-08-31") == []
