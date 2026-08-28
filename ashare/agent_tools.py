"""LLM 唯一能触达 A 股本地数据的入口（架构 §6.2，铁律 D1：只读）。

三个工具，全部只读：查动态股票池、查已落库的因子值、读最新调仓清单。
P5 的 `run_stock_report` 落地前【不注册】—— 注册一个必然 not_available 的
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
        hashes = {n: get_factor(n).param_hash() for n in names}
        # 因子按【周频调仓日】预计算。请求日不是调仓日（或那天还没落库）时，回退到
        # **不晚于**它的最近一个有值的调仓日 —— 只向过去退，PIT 安全（向未来退是前视）。
        # 为什么不像从前那样只回一句 hint 让模型自己重查：2026-08-28 实测，模型拿到
        # error 之后并没有照提示再查一次，而是直接编了一个答案 —— 一次失败的工具调用
        # 换来一个看不出是编的回答，比工具本身出错坏得多。
        # 实际用的日期原样写进 as_of，另附 requested_as_of + note，绝不静默替换。
        req = query.norm_date(as_of)
        # 倒序：先试请求日本身，再依次向【更近的过去】退（顺序写反会退到更早的那天）
        cands = sorted({req, *query.get_trade_dates(req, freq="W")[-2:]}, reverse=True)
        cands = [d for d in cands if d <= req]
        got, warns, used, universe = {}, [], None, []
        for d in cands:
            universe = query.get_universe(d)
            got, warns = factor_store.read_current(hashes, d, universe)
            if got:
                used = d
                break
        # 严格判据一个都没命中时的最后一档：读【陈旧快照】的值并刺眼标注。
        # 为什么需要这一档：snapshot_id 是全库指纹，每日增量往尾部追加新交易日就会
        # 让它变，于是 3000 万行历史因子集体判陈旧 —— 而 2015 年那天的因子值数值上
        # 根本没变。重建全量要十几小时，不可能每晚做（P3 计划 V6）。
        # 决策路径不受影响：回测与信号生成走的是严格的 read_current。
        stale_snap = None
        if not got:
            for d in cands:
                universe = query.get_universe(d)
                got, stale_snap, warns = factor_store.read_any_snapshot(hashes, d, universe)
                if got:
                    used = d
                    break
        if not got:
            return {"success": False,
                    "error": f"{req} 及其之前最近的调仓日都没有算过因子",
                    "hint": "因子按周频调仓日预计算，请确认该区间的因子已 build"}
        out: dict = {"success": True, "as_of": str(used), "universe_size": len(universe),
                     "note": "processed 为横截面 z 分，未乘 direction；direction=+1 高分看多"}
        if stale_snap is not None:
            out["stale"] = True
            out["computed_under_snapshot"] = stale_snap
            out["current_snapshot"] = query.snapshot_id()
            out["note"] += (f"；⚠ 这批因子是在数据快照 {stale_snap} 下算的，当前快照已是 "
                            f"{out['current_snapshot']}（每日增量追加新交易日就会换快照）。"
                            f"历史行情未被改写时数值仍然可用，但**回答时必须说明这一点**；"
                            f"要拿它做决策请先重建因子库")
        if used != req:
            out["requested_as_of"] = str(req)
            out["note"] += (f"；⚠ {req} 不是周频调仓日（或尚未落库），"
                            f"以上是**回退到 {used}** 的值，回答时必须说明这个日期")
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


def get_signal_list(top: int = 10) -> dict:
    """最新调仓清单的只读转述（P3 Task 7，架构 §6.2 预留位兑现）。
    只转述已生成的清单，不评价、不下单；清单由用户人工执行。"""
    try:
        from ashare.data.ledger_store import latest_signal_plan     # 只导读函数（写函数禁触，见测试）
        p = latest_signal_plan()
        if p is None:
            return {"success": False, "error": "还没有生成过任何清单",
                    "hint": "python3 -m ashare.strategy.plan --as-of <交易日> 生成"}
        top = max(1, min(int(top), _MAX_CODES))
        orders = [{k: o.get(k) for k in ("ts_code", "name", "action", "current_weight",
                                          "target_weight", "limit_price_range", "urgency")}
                  for o in p.get("orders", [])[:top]]
        return {"success": True, "as_of": p["as_of"], "execute_on": p["execute_on"],
                "execute_note": p.get("execute_note"),
                "target_position": p.get("target_position"),
                "position_calibrated": p.get("position_calibrated"),
                "n_orders": len(p.get("orders", [])), "n_excluded": len(p.get("excluded", [])),
                "orders": orders, "truncated": len(p.get("orders", [])) > top,
                "warnings": p.get("warnings", [])[:5],
                "param_hash": p.get("param_hash"), "data_snapshot_id": p.get("data_snapshot_id"),
                "note": "清单为只读转述，需人工执行；执行后请在信号看板回写确认/对账"}
    except Exception as e:                     # noqa: BLE001 — T1
        return _err(e)


def get_stock_fundamentals(ts_code: str, as_of: Optional[str] = None) -> dict:
    """个股基本面快照（只读，本地 PIT 数仓）。写「个股详细报告」的主力数据源。

    ★ 补的是第一层的缺口：从前 agent 拿得到因子 z 分却拿不到 PE/ROE，只能转头去联网源
      （2026-08-28 用户实测）。这些数据本地全有，而且比联网源多两样要紧的东西：
      财报带**公告日与滞后天数**（PIT，不是"最新财报"那种前视口径），行情是**后复权**
      （与回测同口径）。
    """
    try:
        query.open_db()
        ts = to_ts_code(ts_code) or ts_code
        d = query.norm_date(as_of) if as_of else query.last_data_date()
        if d is None:
            return {"success": False, "error": "本地库为空"}
        out: dict = {"success": True, "ts_code": ts, "as_of": str(d),
                     "note": "本地 PIT 数仓；财报按公告日可见（非前视），行情为后复权"}

        basic = query.get_stock_basic(d, [ts])
        if ts not in basic.index:
            return {"success": False, "error": f"{ts} 不在 {d} 的本地库里（代码错误或尚未上市）"}
        row = basic.loc[ts]
        out["profile"] = {"name": str(row.get("name", "?")),
                          "sw_l1": str(row.get("sw_l1", "?")), "sw_l2": str(row.get("sw_l2", "?")),
                          "list_date": str(row.get("list_date", ""))}

        db = query.get_daily_basic(d, [ts], fields=("pe_ttm", "pb", "ps_ttm", "dv_ttm",
                                                    "total_mv", "circ_mv", "turnover_rate_f"))
        if ts in db.index:
            v = db.loc[ts]
            out["valuation"] = {k: _num(v.get(k)) for k in ("pe_ttm", "pb", "ps_ttm", "dv_ttm",
                                                            "turnover_rate_f")}
            out["market_cap_yi"] = {k: (None if _num(v.get(k)) is None else round(float(v[k]) / 1e4, 1))
                                    for k in ("total_mv", "circ_mv")}    # 万元 → 亿元

        fin = query.get_financial(d, [ts], ["revenue", "n_income_attr_p", "roe",
                                            "grossprofit_margin", "debt_to_assets"])
        if ts in fin.index:
            f = fin.loc[ts]
            out["financials"] = {
                "revenue_yi": None if _num(f.get("revenue")) is None else round(float(f["revenue"]) / 1e8, 1),
                "net_profit_yi": None if _num(f.get("n_income_attr_p")) is None else round(float(f["n_income_attr_p"]) / 1e8, 1),
                "roe": _num(f.get("roe")), "gross_margin": _num(f.get("grossprofit_margin")),
                "debt_ratio": _num(f.get("debt_to_assets")),
                "report_period": str(f.get("end_date", "")), "announced_on": str(f.get("ann_date", "")),
                "lag_days": int(f["lag_days"]) if f.get("lag_days") == f.get("lag_days") else None}

        bars = query.get_bars(d, [ts], lookback=60, fields=("close",))
        if not bars.empty:
            c = bars["close"].astype(float).dropna()
            if len(c) >= 2:
                out["price_60d"] = {"return_pct": round((float(c.iloc[-1]) / float(c.iloc[0]) - 1) * 100, 2),
                                    "n_bars": int(len(c)),
                                    "basis": "后复权（与回测同口径），非可输入的挂单价"}
        return out
    except Exception as e:                     # noqa: BLE001 — T1
        return _err(e)


ASHARE_TOOL_REGISTRY: Dict[str, Callable] = {
    "query_universe": query_universe,
    "get_factor_exposure": get_factor_exposure,
    "get_signal_list": get_signal_list,
    "get_stock_fundamentals": get_stock_fundamentals,
}

ASHARE_TOOL_SCHEMAS: List[dict] = [
    {
        "name": "query_universe",
        "description": """【A 股首选】查某个日期的动态股票池（只读，本地 PIT 数仓，不联网）。
2010 年至今全市场，**含已退市股** —— 联网数据源只给「今天还活着的」，问历史会有幸存者偏差。
适用：那天有多少票可交易 / 池子里有谁 / 某票当时在不在池子里 / 行业分布。
返回池子规模 + 前 N 只的代码、名称、申万一级行业。仅查询，不下单、不修改任何数据。""",
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
        "description": """【A 股量化分析首选，写个股报告必用】本地因子库（只读，不下单、不修改任何数据）。
15 个 alpha 因子：反转/动量/换手/波动/市值/估值(EP·BP·SP)/盈利质量(ROE·毛利·超预期)/北向等。
**这是联网数据源给不了的东西** —— 全市场横截面 z 分，与回测、与每周信号同一口径。
三种用法：只给 as_of → 各因子覆盖率；加 ts_code → **该股全部因子暴露（写个股报告用这个）**；
加 factor → 该因子全市场 top 排名（问「谁在走强」用这个）。
按周频调仓日预计算；给的日期不是调仓日会自动回退，并在返回里注明实际用的是哪一天。""",
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

ASHARE_TOOL_SCHEMAS.append({
    "name": "get_signal_list",
    "description": """【问「该买卖什么」就用这个】A 股最新一期调仓清单（只读转述，不下单、不修改任何数据）。
本系统的终点产物：策略经样本外验证后，每个周频调仓日自动生成的可执行清单。
返回信号日/执行时点/目标仓位/逐笔订单（含限价带与紧急度）/未校准警示/可追溯的双指纹。
需用户人工执行 —— 本工具只转述已生成的清单，不评价、不建议、不构成任何操作。""",
    "input_schema": {
        "type": "object",
        "properties": {"top": {"type": "integer", "description": "返回订单条数上限，默认 10"}},
        "required": [],
    },
})


ASHARE_TOOL_SCHEMAS.append({
    "name": "get_stock_fundamentals",
    "description": """【写 A 股个股报告的主力工具，优先于任何联网源】本地 PIT 数仓的个股基本面快照（只读）。
一次给全：名称/申万行业/上市日 + 估值(PE_TTM·PB·PS_TTM·股息率·换手) + 市值(亿元)
+ 财务(营收/归母净利/ROE/毛利率/负债率，**带报告期、公告日与滞后天数**) + 近 60 日后复权涨跌。
比联网源多两样要紧的：财报按**公告日**可见（PIT，不是"最新财报"那种前视口径）；
行情是**后复权**，与回测、与因子同一口径。
配合 get_factor_exposure（该股因子暴露）即可写出完整报告；只有盘中实时价才需要 market_quote。""",
    "input_schema": {
        "type": "object",
        "properties": {
            "ts_code": {"type": "string", "description": "股票代码，6 位或 ######.SH/.SZ/.BJ"},
            "as_of": {"type": "string", "description": "可选，交易日 YYYY-MM-DD；不给用库里最新交易日"},
        },
        "required": ["ts_code"],
    },
})

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
