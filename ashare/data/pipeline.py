"""ingest 驱动：全量回补 / 每日增量。永远写 staging，校验通过后由调用方 promote。

用法（操作员手动，需 TUSHARE_TOKEN）：
    python -m ashare.data.pipeline full  --start 2010-01-01            # 首次全量（按年分批，可断点续跑）
    python -m ashare.data.pipeline daily                                # 每日增量（end clamp 到最后已发布交易日）

设计要点：
  - 写者只碰 staging；validate 在写连接关闭后跑（同进程读写连接互斥）；promote 由本模块 main 收尾
  - daily 的 end 必须 clamp 到 <= today 的最后一个交易日：否则当天被冻结成停牌占位并标 DONE
  - daily 按时间正序、且 start = 库里最后一根 K 线的下一天：_seed_before 假设批次正序
  - 财报/宏观 daily 时回看一段窗口重拉（幂等 upsert），吃掉迟到公告
"""
from __future__ import annotations
import argparse
import datetime as _dt
import os
import shutil
import sys
from typing import Callable, Sequence

from . import _db, ingest, promote, validate

DEFAULT_INDICES = ("000985.CSI", "000300.SH", "000905.SH", "000852.SH")
MACRO_INDICATORS = ("m1_yoy", "m2_yoy", "cpi_yoy", "ppi_yoy", "pmi_mfg", "tsf_stock_yoy", "shibor_3m", "cn10y")
HK_HOLD_FROM = _dt.date(2016, 12, 5)
STAGING_DIR = "data/ashare_staging"


# ══════════════ 纯函数 helpers ══════════════
def yearly_batches(start: _dt.date, end: _dt.date) -> list[tuple[_dt.date, _dt.date]]:
    out = []
    y = start.year
    while True:
        s = start if y == start.year else _dt.date(y, 1, 1)
        e = min(end, _dt.date(y, 12, 31))
        out.append((s, e))
        if e >= end:
            return out
        y += 1


def clamp_end(conn, today: _dt.date) -> _dt.date:
    """<= today 的最后一个开市日（日历尽头也不超过）。日历为空 → ValueError。"""
    r = conn.execute("SELECT max(trade_date) FROM calendar WHERE is_open AND trade_date <= ?", [today]).fetchone()
    if r is None or r[0] is None:
        raise ValueError("calendar 为空或不含 <= today 的开市日，先 ingest calendar")
    return r[0]


def last_bar_date(conn) -> _dt.date | None:
    r = conn.execute("SELECT max(trade_date) FROM daily_bar").fetchone()
    return r[0] if r and r[0] is not None else None


def _open_days(conn, start: _dt.date, end: _dt.date) -> list[_dt.date]:
    return [r[0] for r in conn.execute("SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ? "
                                       "ORDER BY trade_date", [start, end]).fetchall()]


def _listed_between(conn, start: _dt.date, end: _dt.date) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT ts_code FROM stock_basic WHERE list_date <= ? AND (delist_date IS NULL OR delist_date >= ?) "
        "ORDER BY ts_code", [end, start]).fetchall()]


def _years_back(d: _dt.date, n: int) -> _dt.date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:
        return d.replace(year=d.year - n, day=28)


# ══════════════ 共同的落库步骤 ══════════════
def _ingest_range(conn, src, start: _dt.date, end: _dt.date, *, indices: Sequence[str],
                  fin_start: _dt.date, macro_start: _dt.date, observed_on: _dt.date | None,
                  progress: Callable[[str], None] | None) -> dict:
    log = progress or (lambda msg: None)
    summary: dict = {"start": start, "end": end}

    log("stock_basic / stock_status / industry_member")
    summary["stock_basic"] = ingest.ingest_stock_basic(conn, src)
    summary["stock_status"] = ingest.ingest_stock_status(conn, src)
    summary["industry_member"] = ingest.ingest_industry_member(conn, src)

    codes = _listed_between(conn, start, end)
    n_bar = n_fin = 0
    for i, code in enumerate(codes):
        for bs, be in yearly_batches(start, end):
            n_bar += ingest.ingest_daily_bar(conn, src, code, bs, be)
        n_fin += ingest.ingest_financial(conn, src, code, fin_start, end)
        if (i + 1) % 100 == 0:
            log(f"stocks {i + 1}/{len(codes)}")
    summary["daily_bar"], summary["financial_pit"] = n_bar, n_fin

    days = _open_days(conn, start, end)
    n_db = n_hk = 0
    for d in days:
        n_db += ingest.ingest_daily_basic(conn, src, d)
        if d >= HK_HOLD_FROM:
            n_hk += ingest.ingest_hk_hold(conn, src, d)
    summary["daily_basic"], summary["money_flow"] = n_db, n_hk

    summary["index_daily"] = sum(ingest.ingest_index_daily(conn, src, ix, start, end) for ix in indices)
    summary["macro"] = sum(ingest.ingest_macro(conn, src, ind, macro_start, end, observed_on=observed_on)
                           for ind in MACRO_INDICATORS)
    return summary


def _validate_closed(path: str) -> None:
    """写连接必须已关闭再校验（同进程读写互斥）。阻断项失败 → ValidationError 冒泡，调用方不得 promote。"""
    validate.run_all(path)


# ══════════════ 全量 ══════════════
def run_full(staging_path: str, src, *, start: _dt.date, end: _dt.date,
             indices: Sequence[str] = DEFAULT_INDICES,
             progress: Callable[[str], None] | None = None) -> dict:
    """首次全量回补到 staging。可断点续跑：ingest_log 里 DONE 的 (股票, 年) 批次会被跳过。"""
    os.makedirs(os.path.dirname(staging_path) or ".", exist_ok=True)
    conn = _db.connect_write(staging_path)
    try:
        _db.init_schema(conn)
        # 日历多拉 90 天到未来：next_trade_date / 周期末判定需要看到下一个交易日
        ingest.ingest_calendar(conn, src, _years_back(start, 1), end + _dt.timedelta(days=90))
        summary = _ingest_range(conn, src, start, end, indices=indices,
                                fin_start=_years_back(start, 2), macro_start=_years_back(start, 2),
                                observed_on=None, progress=progress)
    finally:
        conn.close()
    _validate_closed(staging_path)
    summary["validation"] = "passed"
    return summary


# ══════════════ 增量 ══════════════
def run_daily(market_path: str, staging_path: str, src, *, today: _dt.date | None = None,
              indices: Sequence[str] = DEFAULT_INDICES,
              progress: Callable[[str], None] | None = None) -> dict:
    """market → 拷贝为 staging → 增量写 → 校验。start = 库里最后一根 K 线的下一天，end = clamp 到 <= today 的最后交易日。
    observed_on = today：宏观新 period 以今天为 observed 公布日（只能是真实拉取日，不得回填）。"""
    today = today or _dt.date.today()
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"market 不存在: {market_path}，先跑 full")
    os.makedirs(os.path.dirname(staging_path) or ".", exist_ok=True)
    shutil.copyfile(market_path, staging_path)
    conn = _db.connect_write(staging_path)
    try:
        _db.init_schema(conn)
        ingest.ingest_calendar(conn, src, today - _dt.timedelta(days=14), today + _dt.timedelta(days=90))
        end = clamp_end(conn, today)
        last = last_bar_date(conn)
        if last is None:
            start: _dt.date | None = end
        else:
            r = conn.execute("SELECT min(trade_date) FROM calendar WHERE is_open AND trade_date > ?", [last]).fetchone()
            start = r[0] if r else None
        if start is None or start > end:
            return {"start": start, "end": end, "daily_bar": 0, "skipped": True, "validation": "not_run"}
        summary = _ingest_range(conn, src, start, end, indices=indices,
                                fin_start=start - _dt.timedelta(days=120),
                                macro_start=start - _dt.timedelta(days=400),
                                observed_on=today, progress=progress)
        summary["skipped"] = False
    finally:
        conn.close()
    _validate_closed(staging_path)
    summary["validation"] = "passed"
    return summary


# ══════════════ CLI ══════════════
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ashare.data.pipeline")
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("full")
    f.add_argument("--start", default="2010-01-01")
    f.add_argument("--end", default=None)
    d = sub.add_parser("daily")
    d.add_argument("--today", default=None)
    for p in (f, d):
        p.add_argument("--market", default=_db.DEFAULT_MARKET_PATH)
        p.add_argument("--staging", default=os.path.join(STAGING_DIR, "market.duckdb"))
        p.add_argument("--no-promote", action="store_true")
    a = ap.parse_args(argv)

    from .sources.tushare import TushareSource       # 延迟导入：可选依赖
    src = TushareSource()
    log = lambda m: print(f"[pipeline] {m}", file=sys.stderr)   # noqa: E731

    if a.mode == "full":
        start = _dt.date.fromisoformat(a.start)
        end = _dt.date.fromisoformat(a.end) if a.end else _dt.date.today()
        summary = run_full(a.staging, src, start=start, end=end, progress=log)
    else:
        today = _dt.date.fromisoformat(a.today) if a.today else _dt.date.today()
        summary = run_daily(a.market, a.staging, src, today=today, progress=log)
    print(summary)
    if summary.get("skipped"):
        return 0
    if not a.no_promote:
        bak = promote.promote(a.staging, a.market)
        log(f"promoted → {a.market}（快照 {bak or '无'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
