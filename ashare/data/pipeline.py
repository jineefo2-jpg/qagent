"""ingest 驱动：全量回补 / 每日增量。永远写 staging，校验通过后由调用方 promote。

用法（操作员手动，需 TUSHARE_TOKEN）：
    python -m ashare.data.pipeline full  --start 2010-01-01 --end 2026-08-19   # 首次全量（--end 必须锁定，可断点续跑）
    python -m ashare.data.pipeline daily                                        # 每日增量（约 20 次 API 调用）

设计要点：
  - 写者只碰 staging；validate 在写连接关闭后跑（同进程读写连接互斥）；promote 由本模块 main 收尾
  - daily 的 end 必须 clamp 到 <= today 的最后一个交易日：否则当天被冻结成停牌占位并标 DONE
  - daily 按时间正序、且 start = 库里最后一根 K 线的下一天：_seed_before 假设批次正序
  - 财报/宏观 daily 时回看一段窗口重拉（幂等 upsert），吃掉迟到公告
  - 调用量：full 按股票整段拉（≈ 5,300 × 3 + 财报 4 次 ≈ 3.7 万次，多晚跑）；
    daily 走全市场单日形式（bars 3 + daily_basic 1 + hk_hold 1 + 指数 4 + 宏观 8 ≈ 17 次），
    财报扫描按周节流（22,000 次），否则天天跑要 5 小时
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
                  allow_static_industry: bool, per_day_bars: bool, financials: bool,
                  progress: Callable[[str], None] | None) -> dict:
    log = progress or (lambda msg: None)
    summary: dict = {"start": start, "end": end}

    log("stock_basic / stock_status / industry_member")
    summary["stock_basic"] = ingest.ingest_stock_basic(conn, src)
    summary["stock_status"] = ingest.ingest_stock_status(conn, src)
    summary["industry_member"] = ingest.ingest_industry_member(conn, src)
    if allow_static_industry:
        # 显式承认"本库的行业是今天的值回填到上市日" —— validate.check_industry_source 放行，
        # 但键留在库里，P2 的行业中性化必须读它并拒绝。
        conn.execute("INSERT INTO _meta (key, value) VALUES ('industry_source_ack', '1') "
                     "ON CONFLICT (key) DO UPDATE SET value = excluded.value")

    days = _open_days(conn, start, end)
    codes = _listed_between(conn, start, end)
    n_bar = n_fin = 0

    if per_day_bars:
        # 增量：全市场单日形式，每天 3 次调用（5,300 只 × 3 = 15,900 次/天在 120/min 下要 2.2 小时）
        for d in days:
            n_bar += ingest.ingest_daily_bar_by_date(conn, src, d)
    else:
        # 全量：按股票整段拉（Tushare daily 单次 ≤ 6000 行 ≈ 24 年）。跨度 > 20 年才按年切。
        batches = [(start, end)] if (end - start).days <= 365 * 20 else yearly_batches(start, end)
        for i, code in enumerate(codes):
            for bs, be in batches:
                n_bar += ingest.ingest_daily_bar(conn, src, code, bs, be)
            if (i + 1) % 200 == 0:
                log(f"bars {i + 1}/{len(codes)}")

    if financials:
        for i, code in enumerate(codes):
            n_fin += ingest.ingest_financial(conn, src, code, fin_start, end)
            if (i + 1) % 200 == 0:
                log(f"financials {i + 1}/{len(codes)}")
        conn.execute("INSERT INTO _meta (key, value) VALUES ('last_fin_sweep', ?) "
                     "ON CONFLICT (key) DO UPDATE SET value = excluded.value", [end.isoformat()])
    summary["daily_bar"], summary["financial_pit"] = n_bar, n_fin
    summary["financials_swept"] = financials

    n_db = n_hk = 0
    for i, d in enumerate(days):
        n_db += ingest.ingest_daily_basic(conn, src, d)
        if d >= HK_HOLD_FROM:
            n_hk += ingest.ingest_hk_hold(conn, src, d)
        if (i + 1) % 200 == 0:
            log(f"daily_basic/hk_hold {i + 1}/{len(days)}")
    summary["daily_basic"], summary["money_flow"] = n_db, n_hk

    summary["index_daily"] = sum(ingest.ingest_index_daily(conn, src, ix, start, end) for ix in indices)
    summary["macro"] = sum(ingest.ingest_macro(conn, src, ind, macro_start, end, observed_on=observed_on)
                           for ind in MACRO_INDICATORS)
    return summary


def _validate_closed(path: str, *, cross_source: bool = False) -> list:
    """写连接必须已关闭再校验（同进程读写互斥）。阻断项失败 → ValidationError 冒泡，调用方不得 promote。
    cross_source=True 才构造 BaoStock 做双源交叉（200 只 × 100 日，几分钟）；否则该项 SKIPPED。"""
    bao = None
    if cross_source:
        from .sources.baostock import BaoStockSource
        bao = BaoStockSource()
    try:
        return validate.run_all(path, bao)
    finally:
        if bao is not None:
            bao.close()


# ══════════════ 全量 ══════════════
def run_full(staging_path: str, src, *, start: _dt.date, end: _dt.date,
             indices: Sequence[str] = DEFAULT_INDICES,
             allow_static_industry: bool = False, cross_source: bool = False,
             progress: Callable[[str], None] | None = None) -> dict:
    """首次全量回补到 staging。可断点续跑：ingest_log 里 DONE 的 (股票, 年) 批次会被跳过。"""
    os.makedirs(os.path.dirname(staging_path) or ".", exist_ok=True)
    conn = _db.connect_write(staging_path)
    try:
        _db.init_schema(conn)
        # ★ DO NOTHING 不是 DO UPDATE：全量回补支持跨夜续跑，每晚覆盖这个戳的话，
        #   最终 promote 只测得【最后一晚】写入的行 —— 第 1 晚重写了 2010-2015、末晚只碰
        #   几只新股，min_affected 就会记成很晚的日期，于是那批旧因子被判「仍然有效」。
        #   台账要防的正是这件事，方向反了。promote 成功后由 _stamp_snapshot 清除本键，
        #   所以「一轮 ingest」= 从第一次写到发布为止（微秒精度：秒级会与同秒完成的上一轮撞车）。
        conn.execute("INSERT INTO _meta (key, value) VALUES ('ingest_started_at', ?) "
                     "ON CONFLICT (key) DO NOTHING",
                     [_dt.datetime.now().isoformat(sep=" ", timespec="microseconds")])
        # ★ 多晚分批续跑时 end 必须锁定：job key 含 start 不含 end，若第二晚 --end 漂移（默认 today），
        #   第一晚已 DONE 的股票会被跳过、尾部缺行，row_completeness 必挂且无法自愈。首次运行把 end 记进 _meta。
        prev = conn.execute("SELECT value FROM _meta WHERE key='full_end'").fetchone()
        if prev and prev[0] != end.isoformat():
            raise RuntimeError(f"staging 里已有一次 end={prev[0]} 的全量回补；续跑请传 --end {prev[0]}，"
                               f"或删除 staging 重来（本次 end={end.isoformat()}）")
        conn.execute("INSERT INTO _meta (key, value) VALUES ('full_end', ?) "
                     "ON CONFLICT (key) DO UPDATE SET value = excluded.value", [end.isoformat()])
        # 日历多拉 90 天到未来：next_trade_date / 周期末判定需要看到下一个交易日
        ingest.ingest_calendar(conn, src, _years_back(start, 1), end + _dt.timedelta(days=90))
        summary = _ingest_range(conn, src, start, end, indices=indices,
                                fin_start=_years_back(start, 2), macro_start=_years_back(start, 2),
                                observed_on=None, allow_static_industry=allow_static_industry,
                                per_day_bars=False, financials=True, progress=progress)
    finally:
        conn.close()
    results = _validate_closed(staging_path, cross_source=cross_source)
    summary["validation"] = "passed"
    summary["warnings"] = [r.name for r in results if r.passed is False and not r.blocking]
    return summary


# ══════════════ 增量 ══════════════
def run_daily(market_path: str, staging_path: str, src, *, today: _dt.date | None = None,
              until: _dt.date | None = None,
              indices: Sequence[str] = DEFAULT_INDICES,
              allow_static_industry: bool = False,
              financials: bool | None = None, cross_source: bool = False,
              progress: Callable[[str], None] | None = None) -> dict:
    """market → 拷贝为 staging → 增量写 → 校验。start = 库里最后一根 K 线的下一天，end = clamp 到 <= today 的最后交易日。
    observed_on = today：宏观新 period 以今天为 observed 公布日（只能是真实拉取日，不得回填）。
    financials=None：按周节流（距上次扫描 ≥ 7 天才全市场扫财报）；True/False 强制。
    until：**盘中补漏专用** —— K 线终点额外收紧到 <= until（补漏在任意时刻跑都安全：
    昨天为止各数据源必然已落齐），而 observed_on 仍是真实的 today。不能用 today 冒充：
    把 today 伪装成昨天会把宏观 observed 可见日整体回填提前一天，那是 D4 的前视方向。"""
    today = today or _dt.date.today()
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"market 不存在: {market_path}，先跑 full")
    os.makedirs(os.path.dirname(staging_path) or ".", exist_ok=True)
    if os.path.exists(staging_path):
        # ★ full 与 daily 默认同一个 staging 路径。full 是多晚才跑得完的活，
        #   daily 一个 copyfile 就把它抹掉了。有 full_end 标记就拒绝。
        chk = _db.connect_read(staging_path)
        try:
            r = chk.execute("SELECT value FROM _meta WHERE key='full_end'").fetchone()
        except Exception:                                  # noqa: BLE001 — 半成品/损坏文件当作没有标记
            r = None
        finally:
            chk.close()
        if r:
            raise RuntimeError(f"{staging_path} 里有一次未完成的全量回补（full_end={r[0]}）。"
                               f"daily 会覆盖它。请先跑完 full 并 promote，或给 daily 指定 --staging 到别的路径。")
    shutil.copyfile(market_path, staging_path)
    conn = _db.connect_write(staging_path)
    try:
        _db.init_schema(conn)
        # ★ DO NOTHING 不是 DO UPDATE：全量回补支持跨夜续跑，每晚覆盖这个戳的话，
        #   最终 promote 只测得【最后一晚】写入的行 —— 第 1 晚重写了 2010-2015、末晚只碰
        #   几只新股，min_affected 就会记成很晚的日期，于是那批旧因子被判「仍然有效」。
        #   台账要防的正是这件事，方向反了。promote 成功后由 _stamp_snapshot 清除本键，
        #   所以「一轮 ingest」= 从第一次写到发布为止（微秒精度：秒级会与同秒完成的上一轮撞车）。
        conn.execute("INSERT INTO _meta (key, value) VALUES ('ingest_started_at', ?) "
                     "ON CONFLICT (key) DO NOTHING",
                     [_dt.datetime.now().isoformat(sep=" ", timespec="microseconds")])
        ingest.ingest_calendar(conn, src, today - _dt.timedelta(days=14), today + _dt.timedelta(days=90))
        end = clamp_end(conn, min(today, until) if until else today)
        last = last_bar_date(conn)
        if last is None:
            start: _dt.date | None = end
        else:
            r = conn.execute("SELECT min(trade_date) FROM calendar WHERE is_open AND trade_date > ?", [last]).fetchone()
            start = r[0] if r else None
        if start is None or start > end:
            return {"start": start, "end": end, "daily_bar": 0, "skipped": True, "validation": "not_run"}
        # 财报全市场扫描 22,000 次调用，天天跑要 3 小时。按周节流：距上次扫描 ≥ 7 天才扫。
        # 延后只会让财报【晚】可见，不会提前 —— D3 安全方向；要立刻扫用 financials=True。
        if financials is None:
            r = conn.execute("SELECT value FROM _meta WHERE key='last_fin_sweep'").fetchone()
            do_fin = (r is None) or ((end - _dt.date.fromisoformat(r[0])).days >= 7)
        else:
            do_fin = financials
        summary = _ingest_range(conn, src, start, end, indices=indices,
                                fin_start=start - _dt.timedelta(days=120),
                                macro_start=start - _dt.timedelta(days=400),
                                observed_on=today, allow_static_industry=allow_static_industry,
                                per_day_bars=True, financials=do_fin, progress=progress)
        summary["skipped"] = False
    finally:
        conn.close()
    results = _validate_closed(staging_path, cross_source=cross_source)
    summary["validation"] = "passed"
    summary["warnings"] = [r.name for r in results if r.passed is False and not r.blocking]
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
    d.add_argument("--until", default=None,
                   help="盘中补漏：K 线终点收紧到 <= 该日（通常传昨天），observed_on 仍是真实今天")
    d.add_argument("--financials", choices=("auto", "yes", "no"), default="auto",
                   help="财报全市场扫描：auto=按周节流（默认）/ yes=本次强制扫 / no=跳过")
    for p in (f, d):
        p.add_argument("--market", default=_db.DEFAULT_MARKET_PATH)
        p.add_argument("--staging", default=os.path.join(STAGING_DIR, "market.duckdb"))
        p.add_argument("--no-promote", action="store_true")
        p.add_argument("--cross-source", action="store_true",
                       help="跑 BaoStock 双源交叉校验（200 只 × 100 日，需装 baostock，几分钟）")
        p.add_argument("--allow-static-industry", action="store_true",
                       help="没有申万成分接口权限时显式承认：行业将是今天的值回填到上市日，不可做行业中性化")
    a = ap.parse_args(argv)

    from .sources.tushare import TushareSource       # 延迟导入：可选依赖
    src = TushareSource()
    log = lambda m: print(f"[pipeline] {m}", file=sys.stderr)   # noqa: E731

    if a.mode == "full":
        start = _dt.date.fromisoformat(a.start)
        end = _dt.date.fromisoformat(a.end) if a.end else _dt.date.today()
        summary = run_full(a.staging, src, start=start, end=end, cross_source=a.cross_source,
                           allow_static_industry=a.allow_static_industry, progress=log)
    else:
        today = _dt.date.fromisoformat(a.today) if a.today else _dt.date.today()
        until = _dt.date.fromisoformat(a.until) if a.until else None
        fin = {"auto": None, "yes": True, "no": False}[a.financials]
        summary = run_daily(a.market, a.staging, src, today=today, until=until, financials=fin,
                            cross_source=a.cross_source,
                            allow_static_industry=a.allow_static_industry, progress=log)
    print(summary)
    if summary.get("skipped"):
        return 0
    if not a.no_promote:
        bak = promote.promote(a.staging, a.market)
        log(f"promoted → {a.market}（快照 {bak or '无'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
