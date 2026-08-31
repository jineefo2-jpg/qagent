"""全量重建因子库：全部 alpha 因子 × 全区间周频调仓日。

何时用：参考表历史被真实改动、min_affected 追溯很远时（2026-08-31：行业成分分页
修复后覆盖率 50.8%→100%，中性化全历史都变）。日常增量不需要它 —— 有效性判据
按日期局部失效，nightly 只 build 当天。

可断点续跑：store.build 内部按 (因子, 日期) 幂等跳过已是当前快照的组合。
纯本地计算，不碰 API。按批提交并打进度，方便 tail 观察。
"""
from __future__ import annotations
import datetime as dt
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ashare.data import query
from ashare.factors import store
from ashare.factors.base import ALPHA_CATEGORIES, list_factors

BATCH = 20      # 每批 20 个调仓日：批间刷进度，中断最多废一批


def main() -> int:
    fresh = "--fresh" in sys.argv
    query.close_db(); query.open_db()
    if fresh:
        # 全量重建前清空 factor_value：promote 之后旧行全部失效、永远不会再被读到，
        # 留着只是 2.1GB 死重。DROP 而非整库 rm —— 同库的 backtest_run 是【历史】不是
        # 缓存（D7 运行台账），必须保住。★ 中断后续跑【不要】再带 --fresh，否则前功尽弃。
        from ashare.data import _derived
        c = _derived.connect_write()
        try:
            n = c.execute("SELECT count(*) FROM factor_value").fetchone()[0]
            c.execute("DROP TABLE factor_value")
            _derived.init_schema(c)
        finally:
            c.close()
        print(f"--fresh：已清空旧因子 {n:,} 行（backtest_run 保留）", flush=True)
    names = [s.name for s in list_factors() if s.category in ALPHA_CATEGORIES]
    end = query.last_data_date()
    dates = query.get_trade_dates(end, start=dt.date(2010, 1, 1), freq="W")
    print(f"重建 {len(names)} 因子 × {len(dates)} 个调仓日（{dates[0]} ~ {dates[-1]}），快照 {query.snapshot_id()}", flush=True)
    t0, rows = time.monotonic(), 0
    for i in range(0, len(dates), BATCH):
        chunk = dates[i:i + BATCH]
        counts, warns = store.build(names, chunk)
        rows += sum(counts.values())
        el = time.monotonic() - t0
        done = min(i + BATCH, len(dates))
        eta = el / done * (len(dates) - done)
        print(f"  [{done}/{len(dates)}] 累计 {rows:,} 行 | 已用 {el/60:.0f} 分 | 预计还需 {eta/60:.0f} 分"
              + (f" | 告警 {len(warns)}" if warns else ""), flush=True)
    print(f"完成：{rows:,} 行，共 {(time.monotonic()-t0)/60:.0f} 分钟", flush=True)
    query.close_db()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
