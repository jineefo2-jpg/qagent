"""样本外首跑仪式（P3 Task 9，2026-08-27 用户批准）。口径见 P3 计划「烧前预注册清单」。

两阶段，两次显式调用（中间人工核查，不许一口气烧）：
    python3 scripts/oos_ceremony_2026-08-27.py prelaunch   # 过夜前置：闸 2/4/5 → 闸 3 → 样本外因子落库
    python3 scripts/oos_ceremony_2026-08-27.py burn        # 前置全绿后：gate1(A) + gate1(B)，烧掉唯一一次

结果 JSON 落 out/oos_ceremony/；样本外两行台账由引擎自动写 docs/oos-runs.md。
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # 按文件跑时根目录可导入

from ashare.backtest import guards
from ashare.backtest.types import BacktestConfig
from ashare.data import query
from ashare.factors.base import ALPHA_CATEGORIES, list_factors

SNAPSHOT_EXPECTED = "c2cd172976e1f3fe"          # 预注册快照：对不上就拒跑（口径漂移）
DATA_END = dt.date(2026, 8, 21)
GRID = {"top_n": [30, 50, 80]}
OUT = pathlib.Path("out/oos_ceremony")


def _alphas():
    return tuple((s.name, 1.0) for s in list_factors() if s.category in ALPHA_CATEGORIES)


def cfg_a(end: dt.date) -> BacktestConfig:      # A 臂：恒定 0.8
    return BacktestConfig(start=dt.date(2010, 1, 1), end=end, factors=_alphas(),
                          position_cap=0.8, compute_diagnostics=False)


def cfg_b(end: dt.date) -> BacktestConfig:      # B 臂：宏观择时
    return BacktestConfig(start=dt.date(2010, 1, 1), end=end, factors=_alphas(),
                          macro_timing=True, position_floor=0.2, position_cap=1.0,
                          compute_diagnostics=False)


def _dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _gate_row(r) -> dict:
    return {"passed": r.passed, "note": r.note, "detail": r.detail}


def prelaunch() -> int:
    query.open_db("data/ashare_market.duckdb")
    snap = query.snapshot_id()
    assert snap == SNAPSHOT_EXPECTED, f"快照 {snap} ≠ 预注册 {SNAPSHOT_EXPECTED}：数据变了，仪式口径失效"
    base = cfg_a(dt.date(2019, 12, 31))          # 闸 2-5 钉死样本内（guards ★2）
    t0 = time.monotonic()

    print(f"[{time.strftime('%H:%M:%S')}] 闸 2 walk-forward …", flush=True)
    r2 = guards.gate2_walk_forward(base, grid=GRID)
    print(f"  → {'过' if r2.passed else '不过'}: {r2.note}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] 闸 4 成本加压 …", flush=True)
    r4 = guards.gate4_cost_stress(base)
    print(f"  → {'过' if r4.passed else '不过'}: {r4.note}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] 闸 5 参数高原 …", flush=True)
    r5 = guards.gate5_param_plateau(base, grid=GRID)
    print(f"  → {'过' if r5.passed else '不过'}: {r5.note}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] 闸 3 置换 n=200 …（约 4 小时）", flush=True)
    r3 = guards.gate3_shuffle(base, n=200, seed=0)
    print(f"  → {'过' if r3.passed else '不过'}: {r3.note}", flush=True)
    _dump("gates_prelaunch", {"snapshot": snap, "grid": GRID,
                              "gate2": _gate_row(r2), "gate3": _gate_row(r3),
                              "gate4": _gate_row(r4), "gate5": _gate_row(r5)})

    print(f"[{time.strftime('%H:%M:%S')}] 样本外区间因子落库 2020-01-01 → {DATA_END} …", flush=True)
    from ashare.factors import store as fstore
    dates = query.get_trade_dates(DATA_END, start=dt.date(2020, 1, 1), freq="W")
    last = [0.0]
    def prog(i, n):
        now = time.monotonic()
        if now - last[0] > 120:
            last[0] = now
            print(f"  build {i}/{n}  已用 {(now - t0) / 60:.0f} 分", flush=True)
    counts, bw = fstore.build([n for n, _ in _alphas()], dates, progress=prog)
    _dump("oos_factor_build", {"rows": sum(counts.values()), "dates": len(dates),
                               "warnings_head": bw[:10], "n_warnings": len(bw)})

    ok = all(x.passed for x in (r2, r3, r4, r5))
    print(f"[{time.strftime('%H:%M:%S')}] 前置完成，总耗时 {(time.monotonic() - t0) / 3600:.1f} 小时。"
          f"五闸(2-5) {'全绿 ✅ 可以烧' if ok else '有红 ❌ 停 —— 不烧，先解决'}", flush=True)
    query.close_db()
    return 0 if ok else 2


def burn() -> int:
    pre = json.loads((OUT / "gates_prelaunch.json").read_text(encoding="utf-8"))
    assert all(pre[g]["passed"] for g in ("gate2", "gate3", "gate4", "gate5")), \
        "前置有红闸，按预注册清单不许烧"
    query.open_db("data/ashare_market.duckdb")
    snap = query.snapshot_id()
    assert snap == SNAPSHOT_EXPECTED, f"快照 {snap} ≠ 预注册 {SNAPSHOT_EXPECTED}：不许烧"

    # 无条件连跑两臂（预注册：不许看完一臂再决定另一臂跑不跑）
    print(f"[{time.strftime('%H:%M:%S')}] 🔥 gate1(A 恒定 0.8) …", flush=True)
    ra = guards.gate1_out_of_sample(cfg_a(DATA_END))
    print(f"[{time.strftime('%H:%M:%S')}] 🔥 gate1(B 宏观择时) …", flush=True)
    rb = guards.gate1_out_of_sample(cfg_b(DATA_END))
    _dump("burn_gate1", {"snapshot": snap, "A": _gate_row(ra), "B": _gate_row(rb)})
    print("A:", ra.note)
    print("B:", rb.note)
    print("判定按预注册 ①②机械得出，见 out/oos_ceremony/burn_gate1.json 与 docs/oos-runs.md")
    query.close_db()
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "prelaunch":
        raise SystemExit(prelaunch())
    if mode == "burn":
        raise SystemExit(burn())
    print(__doc__)
    raise SystemExit(2)
