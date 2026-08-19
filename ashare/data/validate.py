"""落地校验（规格 §4.4）。每次入库后自动跑；只读；任一【阻断】项失败 → ValidationError。

| 校验               | 判据                                                       | 级别 |
|--------------------|------------------------------------------------------------|------|
| row_completeness   | 每股 daily_bar 行数 == 在市区间内交易日数，误差为 0（D9）  | 阻断 |
| placeholder_rows   | 停牌占位行 vol=0 ∧ amount=0 ∧ open=high=low=close          | 阻断 |
| adj_factor_jumps   | 相邻交易日 adj_factor 比值 ∉ [0.5, 2.0]                     | 告警 |
| financial_ann_date | ann_date 缺失 = 0；ann_date 不晚于入库时间                  | 阻断 |
| macro_publish_date | publish_date/source 缺失 = 0；publish_date >= period（D4）  | 阻断 |
| limit_coverage     | limit_source='unknown' 在非停牌行中的占比（报告）           | 报告 |
| frozen_days        | 某交易日有行但 0 只非停牌（源返回空被写成全市场占位）        | 阻断 |
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
    """每股行数 == 在市区间 ∩ 数据区间 内的交易日数（误差 0）；且不得有上市窗口外的行。
    数据区间下界 = daily_bar 全表最早日期（全量回补从 2010 起，2010 前上市的股票之前本就没有行）。"""
    c = _ro(path)
    try:
        lo, horizon = c.execute("SELECT min(trade_date), max(trade_date) FROM daily_bar").fetchone()
        if horizon is None:
            return CheckResult("row_completeness", True, True, {"stocks": [], "outside_window": 0, "note": "daily_bar 为空"})
        rows = c.execute("""
            WITH cal AS (SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ?),
            expect AS (
                SELECT b.ts_code,
                       (SELECT count(*) FROM cal
                         WHERE cal.trade_date >= greatest(b.list_date, ?)
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
        """, [lo, horizon, lo, horizon]).fetchall()
        outside = c.execute("""
            SELECT count(*) FROM daily_bar d JOIN stock_basic s USING (ts_code)
            WHERE d.trade_date < s.list_date OR (s.delist_date IS NOT NULL AND d.trade_date > s.delist_date)
        """).fetchone()[0]
    finally:
        c.close()
    bad = [{"ts_code": r[0], "expected": r[1], "actual": r[2], "missing": r[1] - r[2],
            "first_row": r[3], "last_row": r[4]} for r in rows]
    return CheckResult("row_completeness", not bad and outside == 0, True,
                       {"stocks": bad, "n_bad": len(bad), "outside_window": outside, "data_start": lo})


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


# ══════════════ 6b. 冻结日（阻断）══════════════
def check_frozen_days(path: str) -> CheckResult:
    """日历开市、daily_bar 有行、但 0 只非停牌 —— 几乎只可能是数据源当天返回空（未发布 / 故障），
    被 normalize 写成了全市场占位行。这种天一旦 promote，增量驱动会从下一天开始、永远不再重拉。
    （架构 §5.3 的 SUSPECT 状态在此落地为阻断校验。）"""
    c = _ro(path)
    try:
        rows = c.execute("""
            SELECT trade_date, count(*) AS n FROM daily_bar
            GROUP BY trade_date HAVING count(*) > 0 AND sum(CASE WHEN NOT is_suspended THEN 1 ELSE 0 END) = 0
            ORDER BY trade_date
        """).fetchall()
    finally:
        c.close()
    days = [{"trade_date": r[0], "rows": r[1]} for r in rows]
    return CheckResult("frozen_days", not days, True, {"days": days, "n": len(days)})


# ══════════════ 7. 双源交叉（BaoStock，告警；不可用 → SKIPPED）══════════════
def check_cross_source(path: str, bao_src, n_stocks: int = 200, n_days: int = 100,
                       seed: int = 0) -> CheckResult:
    if bao_src is None:
        return CheckResult("cross_source", None, False, {"reason": "no BaoStock source"}, skipped=True)
    c = _ro(path)
    try:
        horizon = c.execute("SELECT max(trade_date) FROM daily_bar").fetchone()[0]
        days = [r[0] for r in c.execute("SELECT trade_date FROM calendar WHERE is_open AND trade_date <= ? "
                                        "ORDER BY trade_date DESC LIMIT ?", [horizon, n_days]).fetchall()]
        if not days:
            return CheckResult("cross_source", None, False, {"reason": "no data"}, skipped=True)
        start, end = min(days), max(days)
        # 只抽窗口内有真实成交的沪深股票：BaoStock 不覆盖北交所；已退市 / 全程停牌的股票无可比样本
        codes = [r[0] for r in c.execute(
            "SELECT DISTINCT ts_code FROM daily_bar WHERE trade_date BETWEEN ? AND ? AND NOT is_suspended "
            "AND (ts_code LIKE '%.SH' OR ts_code LIKE '%.SZ') ORDER BY ts_code", [start, end]).fetchall()]
        rng = random.Random(seed)
        sample = rng.sample(codes, min(n_stocks, len(codes)))
        diffs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        max_diff = 0.0
        compared = 0
        for code in sample:
            try:
                ref = bao_src.hfq_close(code, start, end)
            except Exception as exc:                 # noqa: BLE001 — 单只失败记录并继续，不丢掉已比较的结果
                errors.append({"ts_code": code, "error": str(exc)[:200]})
                continue
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
    if compared == 0:                                # 一个都没比上 → 源不可用，SKIPPED 而非 PASS
        return CheckResult("cross_source", None, False,
                           {"reason": "BaoStock 无可比样本", "errors": errors[:5], "sampled": len(sample)}, skipped=True)
    return CheckResult("cross_source", not diffs, False,
                       {"compared": compared, "n_bad": len(diffs), "max_abs_pct_diff": max_diff,
                        "worst": diffs[:20], "errors": errors[:20], "n_errors": len(errors)})


# ══════════════ 汇总 ══════════════
def run_all(path: str, bao_src=None) -> list[CheckResult]:
    results = [
        check_row_completeness(path),
        check_placeholder_rows(path),
        check_adj_factor_jumps(path),
        check_financial_ann_date(path),
        check_macro_publish_date(path),
        check_limit_coverage(path),
        check_frozen_days(path),
        check_cross_source(path, bao_src),
    ]
    failed = [r for r in results if r.blocking and not r.skipped and not r.passed]
    if failed:
        err = ValidationError("阻断级校验未通过: " + "; ".join(f"{r.name}={r.detail}" for r in failed)[:2000])
        err.results = results                        # type: ignore[attr-defined]  # 告警项也一并带出
        raise err
    return results
