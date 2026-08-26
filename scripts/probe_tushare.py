"""实测 Tushare 各接口的可用性与限频。开工第一天跑一次，30 分钟内出结果。
不写任何业务数据，只写 data/rate_state.json 与探测报告。

用法：
    TUSHARE_TOKEN=<你的token> python3 scripts/probe_tushare.py

结果落两处：
  · data/rate_state.json —— Task 3 的令牌桶读它的 calls_per_min（gitignored）
  · 终端逐行 OK/FAIL —— FAIL 的接口按 P1 计划 Task 0 的替代方案表处置
    （stk_limit → limits.py 规则兜底；hk_hold → 砍北向因子；cn_m/shibor → akshare）
"""
from __future__ import annotations
import json, os, pathlib, time
import tushare as ts

# (接口名, 调用 lambda, 是否 P1 必需)
PROBES = [
    ("trade_cal",     lambda p: p.trade_cal(exchange="SSE", start_date="20240101", end_date="20240131"), True),
    ("stock_basic",   lambda p: p.stock_basic(exchange="", list_status="L", fields="ts_code,name,list_date"), True),
    ("namechange",    lambda p: p.namechange(ts_code="600519.SH"), True),
    ("daily",         lambda p: p.daily(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("adj_factor",    lambda p: p.adj_factor(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("stk_limit",     lambda p: p.stk_limit(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("daily_basic",   lambda p: p.daily_basic(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("fina_indicator",lambda p: p.fina_indicator(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("income",        lambda p: p.income(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("balancesheet",  lambda p: p.balancesheet(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("cashflow",      lambda p: p.cashflow(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("hk_hold",       lambda p: p.hk_hold(trade_date="20240102"), True),
    ("index_daily",   lambda p: p.index_daily(ts_code="000985.CSI", start_date="20240101", end_date="20240131"), True),
    ("cn_m",          lambda p: p.cn_m(start_m="202301", end_m="202312"), True),
    # cn_pmi 输出字段全部「默认显示 N」，必须显式 fields（2026-08-25 回补实测：漏探它导致最后一段才炸）
    ("cn_pmi",        lambda p: p.cn_pmi(start_m="202301", end_m="202312", fields="month,pmi010000"), True),
    ("sf_month",      lambda p: p.sf_month(start_m="202301", end_m="202312"), True),
    ("shibor",        lambda p: p.shibor(start_date="20240101", end_date="20240131"), True),
]

def main() -> int:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("ERROR: 未设置 TUSHARE_TOKEN 环境变量")
        return 1
    pro = ts.pro_api(token)

    results = []
    for name, call, required in PROBES:
        t0 = time.time()
        try:
            df = call(pro)
            results.append({"api": name, "ok": True, "rows": len(df),
                            "elapsed": round(time.time() - t0, 2), "required": required, "error": ""})
            print(f"  OK   {name:<16} rows={len(df):<6} {time.time()-t0:.2f}s")
        except Exception as exc:            # noqa: BLE001 — 探测脚本要看到全部错误类型
            results.append({"api": name, "ok": False, "rows": 0,
                            "elapsed": round(time.time() - t0, 2), "required": required,
                            "error": str(exc)[:200]})
            print(f"  FAIL {name:<16} {str(exc)[:120]}")
        time.sleep(0.5)

    blocked = [r for r in results if not r["ok"] and r["required"]]
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/rate_state.json").write_text(
        json.dumps({"probed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results,
                    "calls_per_min": 120}, ensure_ascii=False, indent=2))

    print(f"\n必需接口不可用: {len(blocked)} 个")
    for r in blocked:
        print(f"  - {r['api']}: {r['error']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
