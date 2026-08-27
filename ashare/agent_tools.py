"""LLM 唯一能触达 A 股本地数据的入口（架构 §6.2，铁律 D1：只读）。

两个工具，全部只读：查动态股票池、查已落库的因子值。P3 的 `get_signal_list`
与 P5 的 `run_stock_report` 落地前【不注册】—— 注册一个必然 not_available 的
工具只会污染 LLM 的工具选择（架构 §6.2 原文）。

三条铁律（T1–T3）：
  · 永远返回 JSON-可序列化 dict，query 层异常在本层捕获转 {"success": False, ...}；
  · 返回体 < 3 KB，列表默认截断并带 truncated 标记；
  · 描述里不出现任何暗示可执行交易的措辞。
本文件不进 TRADING_TOOLS：那道闸的语义是「需要用户身份」，全市场公开数据没有
用户维度（架构 §6.3）。禁止 import brokers.* / ashare.data.ingest / 任何写函数。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ashare.data import query
from ashare.factors import store as factor_store
from ashare.factors.base import ALPHA_CATEGORIES, get_factor, list_factors

_MAX_CODES = 50            # T2：股票列表截断上限


def _err(e: Exception) -> dict:
    return {"success": False, "error": f"{type(e).__name__}: {e}"}


def query_universe(as_of: str, limit: int = 50) -> dict:
    """as_of 当日的动态股票池（D5：含退市股历史、剔 ST/停牌/次新）。只读。"""
    try:
        query.open_db()
        codes = query.get_universe(as_of)
        limit = max(1, min(int(limit), _MAX_CODES))
        head = codes[:limit]
        basic = query.get_stock_basic(as_of, head)
        stocks = [{"ts_code": c,
                   "name": str(basic.loc[c, "name"]) if c in basic.index else "?",
                   "sw_l1": str(basic.loc[c, "sw_l1"]) if c in basic.index else "?"}
                  for c in head]
        return {"success": True, "as_of": str(as_of), "total": len(codes),
                "stocks": stocks, "truncated": len(codes) > limit}
    except Exception as e:                     # noqa: BLE001 — T1：不跨 LLM 边界抛异常
        return _err(e)


def get_factor_exposure(as_of: str, ts_code: Optional[str] = None,
                        factor: Optional[str] = None, top: int = 20) -> dict:
    """已落库的 alpha 因子值（processed z 分，未乘 direction）。只读，不下单、
    不修改任何数据。三种用法：只给 as_of → 各因子覆盖率；给 ts_code → 该股全部
    因子暴露；给 factor → 该因子 top 排名。因子按周频调仓日预计算，非调仓日无值。"""
    try:
        query.open_db()
        alphas = [s.name for s in list_factors() if s.category in ALPHA_CATEGORIES]
        if factor is not None and factor not in alphas:
            return {"success": False, "error": f"未知因子 {factor!r}", "available": alphas}
        names = [factor] if factor else alphas
        universe = query.get_universe(as_of)
        hashes = {n: get_factor(n).param_hash() for n in names}
        got, warns = factor_store.read_current(hashes, as_of, universe)
        if not got:
            wk = query.get_trade_dates(as_of, freq="W")
            hint = str(wk[-1]) if wk else "（无更早调仓日）"
            return {"success": False,
                    "error": f"{as_of} 没有已落库的因子值（按周频调仓日预计算）",
                    "hint": f"最近的周频调仓日是 {hint}，用它再查一次"}
        out: dict = {"success": True, "as_of": str(as_of), "universe_size": len(universe),
                     "note": "processed 为横截面 z 分，未乘 direction；direction=+1 高分看多"}
        if ts_code is not None:
            out["ts_code"] = ts_code
            out["exposures"] = {
                n: {"processed": _num(proc.get(ts_code)),
                    "direction": get_factor(n).direction}
                for n, (_, proc) in sorted(got.items())}
        elif factor is not None:
            _, proc = got[factor]
            top = max(1, min(int(top), _MAX_CODES))
            s = proc.dropna().sort_values(ascending=False)
            out["factor"] = factor
            out["direction"] = get_factor(factor).direction
            out["coverage"] = round(len(s) / max(len(universe), 1), 4)
            out["top"] = [{"ts_code": k, "processed": round(float(v), 4)}
                          for k, v in s.head(top).items()]
            out["truncated"] = len(s) > top
        else:
            out["factors"] = {
                n: {"coverage": round(int(proc.notna().sum()) / max(len(universe), 1), 4),
                    "direction": get_factor(n).direction}
                for n, (_, proc) in sorted(got.items())}
        if warns:
            out["warnings"] = warns[:5]
        return out
    except Exception as e:                     # noqa: BLE001 — T1
        return _err(e)


def _num(v) -> Optional[float]:
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


ASHARE_TOOL_REGISTRY: Dict[str, Callable] = {
    "query_universe": query_universe,
    "get_factor_exposure": get_factor_exposure,
}

ASHARE_TOOL_SCHEMAS: List[dict] = [
    {
        "name": "query_universe",
        "description": """A 股本地数据库：查某个历史/近期日期的动态股票池（只读）。
数据来自本地 PIT 数仓（2010 年至今，含退市股，无幸存者偏差），不联网。
返回池子规模 + 前 N 只的代码/名称/申万一级行业。仅查询，不下单、不修改任何数据。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of": {"type": "string", "description": "交易日，YYYY-MM-DD"},
                "limit": {"type": "integer", "description": "返回股票数上限，默认 50"},
            },
            "required": ["as_of"],
        },
    },
    {
        "name": "get_factor_exposure",
        "description": """A 股本地因子库：查已预计算的 alpha 因子值（只读，不下单、不修改任何数据）。
因子含反转/换手/市值/估值(EP/BP/SP)/盈利质量/北向等，按周频调仓日预计算。
用法：只给 as_of 看各因子覆盖率；加 ts_code 看该股全部因子暴露；加 factor 看该因子 top 排名。
processed 为全市场横截面 z 分（未乘 direction）。非调仓日无值，返回里会提示最近的调仓日。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of": {"type": "string", "description": "交易日，YYYY-MM-DD"},
                "ts_code": {"type": "string", "description": "可选，######.SH/.SZ/.BJ 格式的 ts_code"},
                "factor": {"type": "string", "description": "可选，因子名，如 reversal_20"},
                "top": {"type": "integer", "description": "factor 模式下的排名条数，默认 20"},
            },
            "required": ["as_of"],
        },
    },
]

# ★ 冻结的只读集合，供隔离测试断言（架构 §6.3 M1/M2）
ASHARE_READONLY_TOOLS = frozenset(ASHARE_TOOL_REGISTRY)


# ══════════════ 老工具的 A 股本地路由（非 LLM 工具，供 quant_agent 内部调用）══════════════
# 这三个助手把 quant_agent 里「A 股联网现抓」的路径替换成本地真值（2026-08-26 工具审计）：
# historical_prices → local_history（后复权，D8 口径与回测同源，含退市股）；
# trading_calendar → local_calendar（真实交易日历，替掉「工作日近似」）。
# 失败/未覆盖一律返回 None 或 {"success": False}，由调用方回退网络源 —— 不劫持非 A 股路径。

def to_ts_code(symbol: str) -> Optional[str]:
    """600519 / 600519.SH / sh600519 / 000858.SZ / 300750 → Tushare ts_code；非 A 股返回 None。"""
    s = (symbol or "").strip().upper().replace(".SS", ".SH")
    if s[:2] in ("SH", "SZ", "BJ") and s[2:].isdigit() and len(s) == 8:
        return f"{s[2:]}.{s[:2]}"
    if "." in s:
        code, _, ex = s.partition(".")
        return f"{code}.{ex}" if ex in ("SH", "SZ", "BJ") and code.isdigit() else None
    if s.isdigit() and len(s) == 6:
        if s[0] == "6":
            return f"{s}.SH"
        if s[0] in ("0", "3"):
            return f"{s}.SZ"
        if s[0] in ("4", "8", "9"):
            return f"{s}.BJ"
    return None


def local_history(symbol: str, days: int) -> dict:
    """近 days 个交易日的后复权 K 线，返回结构与 quant_agent._hist_akshare 成功体同构。"""
    try:
        query.open_db()
        ts = to_ts_code(symbol)
        if ts is None:
            return {"success": False, "error": f"无法识别为 A 股代码: {symbol}"}
        tds = query.get_trade_dates(_dt_today())
        if not tds:
            return {"success": False, "error": "本地日历为空"}
        bars = query.get_bars(tds[-1], [ts], lookback=int(days),
                              fields=("open", "high", "low", "close", "vol"))
        if bars.empty or ts not in bars.index.get_level_values(0):
            return {"success": False, "error": f"本地库无 {ts} 行情"}
        df = bars.xs(ts, level=0)
        # 停牌日 get_bars 按设计给 NaN 价（D9 占位行）——网络源的 K 线不含停牌日，滤掉对齐
        df = df[~df["is_suspended"].astype(bool)]
        if df.empty:
            return {"success": False, "error": f"{ts} 窗口内全为停牌日"}
        closes = [float(x) for x in df["close"]]
        if not all(c == c and c > 0 for c in closes):      # NaN != NaN
            return {"success": False, "error": f"{ts} 价格序列含缺值，回退网络源"}
        return {"success": True, "symbol": ts, "market": "A股",
                "days_returned": len(df),
                "dates": [str(d) for d in df.index],
                "open": [float(x) for x in df["open"]],
                "close": closes,
                "high": [float(x) for x in df["high"]],
                "low": [float(x) for x in df["low"]],
                "volume": [int(v) if v == v else 0 for v in df["vol"]],
                "data_source": "本地数仓（后复权，与回测同一口径；停牌日不计入，与网络源 K 线一致）"}
    except Exception as e:                     # noqa: BLE001 — 路由失败即回退，不外抛
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def local_calendar(action: str, date: Optional[str], date2: Optional[str]) -> Optional[dict]:
    """trading_calendar 的本地实现。覆盖不到（日历未及/格式不认）返回 None → 调用方走近似逻辑。"""
    try:
        query.open_db()
        if action == "today":
            today = _dt_today()
            tds = query.get_trade_dates(today)
            return {"success": True, "date": str(today),
                    "is_trading_day": bool(tds and tds[-1] == today),
                    "timezone": "Asia/Shanghai (CST)",
                    "note": "A 股法定交易日历（本地库，含节假日）"}
        if action == "trading_days_between" and date and date2:
            days = query.get_trade_dates(date2, start=date)
            return {"success": True, "from": date, "to": date2,
                    "trading_days": len(days),
                    "note": "A 股法定交易日历（本地库，含节假日）"}
        return None
    except Exception:                          # noqa: BLE001 — 日历未覆盖该日期等，回退近似
        return None


def _dt_today():
    import datetime
    return datetime.date.today()
