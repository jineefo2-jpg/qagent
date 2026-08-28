"""抽样核查：库里存的因子值 与 用【当前数据】重算 是否一致。

用途：`snapshot_log` 的参考表 diff 是个**代理信号** —— 它回答「参考数据变没变」，
而真正要紧的问题是「因子值变没变」。两者之间有缺口：一只从不进股票池的票改了
状态段，diff 会报警，因子值却分毫未动（2026-08-28 实测正是如此）。

代理信号报警时用本脚本取证，再决定要不要重建 —— 重建是十几小时的事，
不该被一个代理信号直接触发。逐位相同 ⇒ 存量可继续使用，把 min_affected_date
按实测收窄；有差异 ⇒ 老老实实重建。

    python3 scripts/verify_factor_store.py [日期数，默认 8]
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from ashare.data import query
from ashare.factors import store as fs
from ashare.factors.base import compute_panel, get_factor, list_factors, ALPHA_CATEGORIES


def main(n: int = 8) -> int:
    query.close_db(); query.open_db()
    names = [s.name for s in list_factors() if s.category in ALPHA_CATEGORIES]
    hashes = {x: get_factor(x).param_hash() for x in names}
    # ★ 只在【因子库真正覆盖的区间】里抽样：日历比数据早一年（full 拉 _years_back(start,1)），
    #   抽到数据窗口之前会撞上空股票池。
    lo, hi = fs.covered_range(hashes)
    if lo is None:
        print("因子库是空的，没什么可核查"); return 0
    wk = [d for d in query.get_trade_dates(hi, freq="W") if lo <= d <= hi]
    step = max(1, len(wk) // n)
    probes = [d for d in wk[::step][:n]]
    print(f"抽样 {len(probes)} 个调仓日（{probes[0]} ~ {probes[-1]}），逐位对比\n")
    bad = 0
    for d in probes:
        uni = query.get_universe(d)
        if not uni:
            print(f"  {d}: 股票池为空（数据窗口之外），跳过"); continue
        stored, snap, _ = fs.read_any_snapshot(hashes, d, uni)
        if not stored:
            print(f"  {d}: 库里没有这天的因子，跳过"); continue
        fresh, _w = compute_panel(names, d, uni, processed=True)
        worst, ndiff = 0.0, 0
        for x in names:
            if x not in stored:
                continue
            a = stored[x][1].reindex(uni).to_numpy(dtype=float)
            b = fresh[x].reindex(uni).to_numpy(dtype=float)
            if (np.isnan(a) != np.isnan(b)).any():
                ndiff += 1; continue
            m = ~np.isnan(a)
            v = float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
            worst = max(worst, v)
            if v > 1e-9:
                ndiff += 1
        bad += bool(ndiff)
        flag = "✅ 一致" if not ndiff else f"❗{ndiff} 个因子有差异"
        print(f"  {d}（池 {len(uni)}，存于 {snap}）：{flag}，最大绝对差 {worst:.3e}")
    print(f"\n结论：{len(probes) - bad}/{len(probes)} 个日期与当前数据一致。"
          + ("存量可继续使用，可把 min_affected_date 按实测收窄。" if not bad
             else "**有差异 → 必须重建**。"))
    query.close_db()
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
