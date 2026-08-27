"""ledger 库的唯一读写出口 —— 只收发 dict / 基础类型（derived_store 的姊妹）。

写侧只许 strategy CLI 与 server 回写端点调用；LLM 工具只经读侧（D1）。
本层不做业务判断（清单怎么生成是 strategy 的事，未校准怎么警示是读方的事），
只守两条底线：D7 双指纹缺一不写；source / state 走 schema 的 CHECK 白名单。
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Mapping, Optional, Sequence

from . import _ledger, query

POSITION_SOURCES = ("reconcile_csv", "manual_confirm", "signal_assumed")
CONFIRM_STATES = ("filled", "partial", "skipped")


def _norm(d) -> _dt.date:
    return query.norm_date(d, name="as_of_date")


# ══════════════ 写侧（CLI / server 专用）══════════════
def save_signal_plan(plan: Mapping) -> bool:
    """落一份 §6.3 清单。同 (as_of, param_hash) 幂等覆盖，返回是否覆盖了已有行。
    D7：param_hash / data_snapshot_id 缺一即抛 —— 落一份无法溯源的清单比不落更坏。"""
    missing = [k for k in ("as_of", "param_hash", "data_snapshot_id", "execute_on",
                           "strategy_version") if not plan.get(k)]
    if missing:
        raise ValueError(f"signal_plan 缺字段 {missing}：D7 双指纹与执行时点缺一不可")
    d = _norm(plan["as_of"])
    conn = _ledger.connect_write()
    try:
        _ledger.init_schema(conn)
        existed = conn.execute("SELECT 1 FROM signal_plan WHERE as_of_date=? AND param_hash=?",
                               [d, plan["param_hash"]]).fetchone() is not None
        conn.execute(
            "INSERT INTO signal_plan (as_of_date, param_hash, execute_on, data_snapshot_id, "
            "strategy_version, plan_json) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (as_of_date, param_hash) DO UPDATE SET "
            "execute_on=excluded.execute_on, data_snapshot_id=excluded.data_snapshot_id, "
            "strategy_version=excluded.strategy_version, plan_json=excluded.plan_json, "
            "created_at=now()",
            [d, plan["param_hash"], str(plan["execute_on"]), plan["data_snapshot_id"],
             plan["strategy_version"], json.dumps(plan, ensure_ascii=False, default=str)])
        return existed
    finally:
        conn.close()


def record_confirms(as_of, rows: Sequence[Mapping]) -> int:
    """逐单三态确认，同键覆盖（用户改主意是常态）。state 白名单在 schema CHECK，这里先验以给出人话。"""
    d = _norm(as_of)
    bad = [r for r in rows if r.get("state") not in CONFIRM_STATES]
    if bad:
        raise ValueError(f"state 只能是 {CONFIRM_STATES}，收到 {[r.get('state') for r in bad]}")
    conn = _ledger.connect_write()
    try:
        _ledger.init_schema(conn)
        for r in rows:
            conn.execute(
                "INSERT INTO order_confirm (as_of_date, ts_code, state, filled_shares, note) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (as_of_date, ts_code) DO UPDATE SET "
                "state=excluded.state, filled_shares=excluded.filled_shares, "
                "note=excluded.note, created_at=now()",
                [d, r["ts_code"], r["state"], r.get("filled_shares"), r.get("note")])
        return len(rows)
    finally:
        conn.close()


def write_positions(as_of, rows: Sequence[Mapping], *, source: str) -> int:
    """写某日实际持仓（整日整批：对账单/确认折算都是全量快照，不做逐行增量）。
    source 必须显式给 —— 'signal_assumed' 与人工来源的区别是「未校准」警示的判据。"""
    if source not in POSITION_SOURCES:
        raise ValueError(f"source 只能是 {POSITION_SOURCES}，收到 {source!r}")
    d = _norm(as_of)
    conn = _ledger.connect_write()
    try:
        _ledger.init_schema(conn)
        conn.execute("DELETE FROM position_ledger WHERE as_of_date = ?", [d])
        for r in rows:
            conn.execute(
                "INSERT INTO position_ledger (as_of_date, ts_code, shares, avg_cost, source) "
                "VALUES (?, ?, ?, ?, ?)",
                [d, r["ts_code"], float(r["shares"]), r.get("avg_cost"), source])
        return len(rows)
    finally:
        conn.close()


# ══════════════ 读侧（server / agent_tools 共用）══════════════
def latest_signal_plan() -> Optional[dict]:
    """最新一份清单（as_of 最大者；同日多参数取 created_at 最新）。库不存在 → None。"""
    try:
        conn = _ledger.connect_read()
    except FileNotFoundError:
        return None
    try:
        row = conn.execute(
            "SELECT plan_json FROM signal_plan ORDER BY as_of_date DESC, created_at DESC LIMIT 1"
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def list_signal_plans(limit: int = 20) -> list:
    """清单目录（不含 JSON 全文）。库不存在 → []。"""
    try:
        conn = _ledger.connect_read()
    except FileNotFoundError:
        return []
    try:
        rows = conn.execute(
            "SELECT as_of_date, execute_on, param_hash, data_snapshot_id, strategy_version, created_at "
            "FROM signal_plan ORDER BY as_of_date DESC, created_at DESC LIMIT ?",
            [max(1, min(int(limit), 200))]).fetchall()
        cols = ("as_of", "execute_on", "param_hash", "data_snapshot_id", "strategy_version", "created_at")
        return [dict(zip(cols, map(str, r))) for r in rows]
    finally:
        conn.close()


def latest_positions() -> "tuple[Optional[_dt.date], list]":
    """最新一期实际持仓 `(as_of, rows)`。库不存在/空表 → (None, [])。"""
    try:
        conn = _ledger.connect_read()
    except FileNotFoundError:
        return None, []
    try:
        d = conn.execute("SELECT max(as_of_date) FROM position_ledger").fetchone()[0]
        if d is None:
            return None, []
        rows = conn.execute(
            "SELECT ts_code, shares, avg_cost, source FROM position_ledger "
            "WHERE as_of_date = ? ORDER BY ts_code", [d]).fetchall()
        return d, [{"ts_code": r[0], "shares": r[1], "avg_cost": r[2], "source": r[3]} for r in rows]
    finally:
        conn.close()


def get_confirms(as_of) -> list:
    """某日的全部确认。库不存在 → []。"""
    try:
        conn = _ledger.connect_read()
    except FileNotFoundError:
        return []
    try:
        rows = conn.execute(
            "SELECT ts_code, state, filled_shares, note FROM order_confirm "
            "WHERE as_of_date = ? ORDER BY ts_code", [_norm(as_of)]).fetchall()
        return [{"ts_code": r[0], "state": r[1], "filled_shares": r[2], "note": r[3]} for r in rows]
    finally:
        conn.close()
