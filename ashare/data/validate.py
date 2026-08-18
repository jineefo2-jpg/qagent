"""落地校验（规格 §4.4）。每次入库后自动跑；只读；任一【阻断】项失败 → ValidationError。

| 校验               | 判据                                                       | 级别 |
|--------------------|------------------------------------------------------------|------|
| row_completeness   | 每股 daily_bar 行数 == 在市区间内交易日数，误差为 0（D9）  | 阻断 |
| placeholder_rows   | 停牌占位行 vol=0 ∧ amount=0 ∧ open=high=low=close          | 阻断 |
| adj_factor_jumps   | 相邻交易日 adj_factor 比值 ∉ [0.5, 2.0]                     | 告警 |
| financial_ann_date | ann_date 缺失 = 0；ann_date 不晚于入库时间                  | 阻断 |
| macro_publish_date | publish_date/source 缺失 = 0；publish_date >= period（D4）  | 阻断 |
| limit_coverage     | limit_source='unknown' 在非停牌行中的占比（报告）           | 报告 |
| cross_source       | 抽样 BaoStock 后复权收盘价偏差 < 0.5%；不可用 → SKIPPED     | 告警 |
"""
from __future__ import annotations
import datetime as _dt
import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import _db

CROSS_TOL = 0.005          # 双源后复权收盘价相对偏差容忍
ADJ_JUMP_LO, ADJ_JUMP_HI = 0.5, 2.0


class ValidationError(Exception):
    """有阻断级校验未通过。"""


@dataclass
class CheckResult:
    name: str
    passed: bool | None            # None = skipped
    blocking: bool
    detail: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False


def _ro(path: str):
    return _db.connect_read(path)


# ══════════════ 1. 行数完整性（误差为 0）══════════════
def check_row_completeness(path: str) -> CheckResult:
    c = _ro(path)
    try:
        horizon = c.execute("SELECT max(trade_date) FROM daily_bar").fetchone()[0]
        if horizon is None:
            return CheckResult("row_completeness", True, True, {"stocks": [], "note": "daily_bar 为空"})
        rows = c.execute("""
            WITH cal AS (SELECT trade_date FROM calendar WHERE is_open AND trade_date <= ?),
            expect AS (
                SELECT b.ts_code,
                       (SELECT count(*) FROM cal
                         WHERE cal.trade_date >= b.list_date
                           AND cal.trade_date <= coalesce(b.delist_date, ?)) AS expected
                FROM stock_basic b WHERE b.list_date IS NOT NULL
            ),
            actual AS (SELECT ts_code, count(*) AS actual,
                              min(trade_date) AS first_row, max(trade_date) AS last_row
                       FROM daily_bar GROUP BY ts_code)
            SELECT e.ts_code, e.expected, coalesce(a.actual, 0) AS actual, a.first_row, a.last_row
            FROM expect e LEFT JOIN actual a USING (ts_code)
            WHERE e.expected <> coalesce(a.actual, 0)
            ORDER BY e.ts_code
        """, [horizon, horizon]).fetchall()
    finally:
        c.close()
    bad = [{"ts_code": r[0], "expected": r[1], "actual": r[2], "missing": r[1] - r[2],
            "first_row": r[3], "last_row": r[4]} for r in rows]
    return CheckResult("row_completeness", not bad, True, {"stocks": bad, "n_bad": len(bad)})


# ══════════════ 2. 占位行合规 ══════════════
def check_placeholder_rows(path: str) -> CheckResult:
    c = _ro(path)
    try:
        n = c.execute("""
            SELECT count(*) FROM daily_bar WHERE is_suspended AND NOT (
                coalesce(vol, 0) = 0 AND coalesce(amount, 0) = 0
                AND ((open IS NULL AND high IS NULL AND low IS NULL AND close IS NULL)
                     OR (open = high AND high = low AND low = close)))
        """).fetchone()[0]
    finally:
        c.close()
    return CheckResult("placeholder_rows", n == 0, True, {"bad_rows": n})


# ══════════════ 3. 复权因子跳变（告警）══════════════
def check_adj_factor_jumps(path: str) -> CheckResult:
    c = _ro(path)
    try:
        rows = c.execute("""
            SELECT ts_code, trade_date, adj_factor, prev_adj FROM (
                SELECT ts_code, trade_date, adj_factor,
                       lag(adj_factor) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_adj
                FROM daily_bar WHERE adj_factor IS NOT NULL)
            WHERE prev_adj IS NOT NULL AND prev_adj > 0
              AND (adj_factor / prev_adj < ? OR adj_factor / prev_adj > ?)
            ORDER BY ts_code, trade_date
        """, [ADJ_JUMP_LO, ADJ_JUMP_HI]).fetchall()
    finally:
        c.close()
    jumps = [{"ts_code": r[0], "trade_date": r[1], "adj_factor": r[2], "prev_adj": r[3]} for r in rows]
    return CheckResult("adj_factor_jumps", not jumps, False, {"jumps": jumps, "n": len(jumps)})


# ══════════════ 4. 财报 ann_date ══════════════
def check_financial_ann_date(path: str) -> CheckResult:
    c = _ro(path)
    try:
        nulls = c.execute("SELECT count(*) FROM financial_pit WHERE ann_date IS NULL").fetchone()[0]
        future = c.execute("SELECT count(*) FROM financial_pit WHERE ann_date > CAST(_ingested_at AS DATE)").fetchone()[0]
    finally:
        c.close()
    return CheckResult("financial_ann_date", nulls == 0 and future == 0, True,
                       {"null_ann_date": nulls, "future_ann_date": future})


# ══════════════ 5. 宏观 publish_date（D4）══════════════
def check_macro_publish_date(path: str) -> CheckResult:
    c = _ro(path)
    try:
        nulls = c.execute("SELECT count(*) FROM macro_indicator WHERE publish_date IS NULL "
                          "OR publish_date_source IS NULL").fetchone()[0]
        early = c.execute("SELECT count(*) FROM macro_indicator WHERE publish_date < period").fetchone()[0]
    finally:
        c.close()
    return CheckResult("macro_publish_date", nulls == 0 and early == 0, True,
                       {"null_publish": nulls, "publish_before_period": early})


# ══════════════ 6. 涨跌停覆盖率（报告）══════════════
def check_limit_coverage(path: str) -> CheckResult:
    c = _ro(path)
    try:
        by_src = dict(c.execute("SELECT coalesce(limit_source, 'null'), count(*) FROM daily_bar "
                                "WHERE NOT is_suspended GROUP BY 1").fetchall())
    finally:
        c.close()
    total = sum(by_src.values()) or 1
    share = by_src.get("unknown", 0) / total
    return CheckResult("limit_coverage", True, False,
                       {"by_source": by_src, "unknown_share_non_suspended": share})


# ══════════════ 7. 双源交叉（BaoStock，告警；不可用 → SKIPPED）══════════════
def check_cross_source(path: str, bao_src, n_stocks: int = 200, n_days: int = 100,
                       seed: int = 0) -> CheckResult:
    if bao_src is None:
        return CheckResult("cross_source", None, False, {"reason": "no BaoStock source"}, skipped=True)
    c = _ro(path)
    try:
        codes = [r[0] for r in c.execute("SELECT DISTINCT ts_code FROM daily_bar ORDER BY ts_code").fetchall()]
        horizon = c.execute("SELECT max(trade_date) FROM daily_bar").fetchone()[0]
        rng = random.Random(seed)
        sample = rng.sample(codes, min(n_stocks, len(codes)))
        days = [r[0] for r in c.execute("SELECT trade_date FROM calendar WHERE is_open AND trade_date <= ? "
                                        "ORDER BY trade_date DESC LIMIT ?", [horizon, n_days]).fetchall()]
        if not days:
            return CheckResult("cross_source", None, False, {"reason": "no data"}, skipped=True)
        start, end = min(days), max(days)
        diffs: list[dict[str, Any]] = []
        max_diff = 0.0
        compared = 0
        for code in sample:
            try:
                ref = bao_src.hfq_close(code, start, end)
            except Exception as exc:                 # noqa: BLE001 — 源不可用即 SKIPPED，不是 PASS
                return CheckResult("cross_source", None, False,
                                   {"reason": f"BaoStock 不可用: {str(exc)[:200]}"}, skipped=True)
            ours = c.execute("SELECT trade_date, close * adj_factor FROM daily_bar WHERE ts_code=? "
                             "AND trade_date BETWEEN ? AND ? AND NOT is_suspended", [code, start, end]).fetchall()
            ours_map = {r[0]: r[1] for r in ours}
            for _, row in ref.iterrows():
                d, ref_px = row["trade_date"], row["close_hfq"]
                if d in ours_map and ours_map[d] and ref_px:
                    diff = abs(ours_map[d] / ref_px - 1.0)
                    compared += 1
                    max_diff = max(max_diff, diff)
                    if diff > CROSS_TOL:
                        diffs.append({"ts_code": code, "trade_date": d, "ours": ours_map[d], "ref": ref_px, "diff": diff})
    finally:
        c.close()
    return CheckResult("cross_source", not diffs, False,
                       {"compared": compared, "n_bad": len(diffs), "max_abs_pct_diff": max_diff,
                        "worst": diffs[:20]})


# ══════════════ 汇总 ══════════════
def run_all(path: str, bao_src=None) -> list[CheckResult]:
    results = [
        check_row_completeness(path),
        check_placeholder_rows(path),
        check_adj_factor_jumps(path),
        check_financial_ann_date(path),
        check_macro_publish_date(path),
        check_limit_coverage(path),
        check_cross_source(path, bao_src),
    ]
    failed = [r for r in results if r.blocking and not r.skipped and not r.passed]
    if failed:
        raise ValidationError("阻断级校验未通过: " + "; ".join(f"{r.name}={r.detail}" for r in failed)[:2000])
    return results
