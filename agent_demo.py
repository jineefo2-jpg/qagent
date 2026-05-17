# ============================================================
#  最小量化 Agent — 三工具教学版
#  ─────────────────────────────────────────────
#  对比上一版（web_search / calculate / calendar_tool 通用三件套），
#  本版改造为符合量化标准的三件套：
#
#    1. market_news_search  — 财经新闻定向搜索
#    2. quant_calc          — 量化专用计算器（收益率/年化/复利/波动率）
#    3. trading_calendar    — 交易日历（交易时段/下一交易日/交易日数）
#
#  依赖：pip install anthropic
#  无 LangChain / LlamaIndex / 任何 Agent 框架
# ============================================================

import anthropic
import json
import math
import re
import statistics
import datetime
import urllib.request
import urllib.parse
from zoneinfo import ZoneInfo

client = anthropic.Anthropic()


# ════════════════════════════════════════════════════════════
# 第一部分：量化标准三工具
# ════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Tool 1: market_news_search — 财经新闻定向搜索
# ─────────────────────────────────────────────

# 财经领域关键词（用于结果过滤）
FINANCE_KEYWORDS = [
    "stock", "share", "earnings", "revenue", "ipo", "merger", "fed",
    "interest rate", "bond", "yield", "etf", "futures", "option",
    "inflation", "cpi", "ppi", "gdp", "pmi", "nasdaq", "s&p",
    "股票", "股价", "财报", "营收", "美联储", "加息", "降息",
    "利率", "通胀", "上市", "并购", "期货", "期权", "央行",
]


def market_news_search(query: str, max_results: int = 5) -> dict:
    """
    财经新闻定向搜索。
    自动为查询附加金融上下文，过滤非财经结果。

    适用：公司公告、宏观数据、监管动态、行业事件、市场行情
    不适用：纯数学计算（用 quant_calc）、日期计算（用 trading_calendar）
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空"}

    # 自动补充金融上下文（提升结果相关性）
    has_finance_kw = any(kw in query.lower() for kw in FINANCE_KEYWORDS)
    final_query = query if has_finance_kw else f"{query} stock market finance"

    try:
        encoded = urllib.parse.quote(final_query)
        url = (f"https://api.duckduckgo.com/?q={encoded}"
               f"&format=json&no_html=1&skip_disambig=1")

        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 QuantAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []

        if data.get("AbstractText"):
            results.append({
                "type": "summary",
                "title": data.get("Heading", ""),
                "content": data["AbstractText"],
                "source": data.get("AbstractURL", ""),
                "category": "市场概览",
            })

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "type": "related",
                    "content": topic["Text"],
                    "source": topic.get("FirstURL", ""),
                })

        if data.get("Answer"):
            results.append({
                "type": "instant_answer",
                "content": data["Answer"],
                "source": "DuckDuckGo",
                "category": "即时数据",
            })

        if not results:
            return {
                "success": False,
                "query_used": final_query,
                "error": f"未找到 '{query}' 的相关财经资讯",
                "hint": "尝试英文关键词或股票代码（如 'NVDA earnings 2025'）",
            }

        return {
            "success": True,
            "original_query": query,
            "query_used": final_query,
            "context_added": not has_finance_kw,
            "results": results,
            "count": len(results),
            "data_source": "DuckDuckGo Instant Answer API",
            "as_of": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "disclaimer": "信息源为公开网络数据，不构成投资建议",
        }

    except urllib.error.URLError as e:
        return {"success": False, "error": f"网络请求失败: {e}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "API 返回格式异常"}
    except Exception as e:
        return {"success": False, "error": f"搜索失败: {e}"}


# ─────────────────────────────────────────────
# Tool 2: quant_calc — 量化专用计算器
# ─────────────────────────────────────────────

# 表达式求值的安全沙箱
SAFE_MATH_CONTEXT = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "pi": math.pi, "e": math.e,
    "__builtins__": {},
}


def quant_calc(
    operation: str,
    expression: str = None,
    start_price: float = None,
    end_price: float = None,
    returns: list = None,
    rate: float = None,
    periods: int = None,
    periods_per_year: int = 252,
    principal: float = None,
    future_value: float = None,
) -> dict:
    """
    量化专用计算器，operation 枚举：

    ── 收益率类 ──
    - simple_return:      简单收益率 (end/start - 1)
    - log_return:         对数收益率 ln(end/start)
    - cumulative_return:  累计收益率（输入 returns 数组）
    - annualize:          年化（period_return + periods_per_year）

    ── 时间价值 ──
    - compound:           复利终值 principal*(1+rate)^periods
    - present_value:      现值 future/(1+rate)^periods

    ── 统计类 ──
    - volatility:         年化波动率（输入 returns 数组）
    - mean_return:        平均收益率（输入 returns 数组）

    ── 通用兜底 ──
    - evaluate:           安全表达式求值（sqrt/log/exp/pi/e）
    """
    op = operation.lower().strip()

    try:
        # ─────────── 收益率类 ───────────
        if op == "simple_return":
            if start_price is None or end_price is None:
                return {"success": False, "error": "需要 start_price 和 end_price"}
            if start_price <= 0:
                return {"success": False, "error": "start_price 必须 > 0"}
            r = end_price / start_price - 1
            return {
                "success": True,
                "operation": "simple_return",
                "start_price": start_price,
                "end_price": end_price,
                "result": round(r, 6),
                "percentage": f"{r*100:.4f}%",
                "formula": "R = (P_end / P_start) - 1",
            }

        elif op == "log_return":
            if start_price is None or end_price is None:
                return {"success": False, "error": "需要 start_price 和 end_price"}
            if start_price <= 0 or end_price <= 0:
                return {"success": False, "error": "价格必须 > 0"}
            r = math.log(end_price / start_price)
            return {
                "success": True,
                "operation": "log_return",
                "result": round(r, 6),
                "percentage": f"{r*100:.4f}%",
                "formula": "R = ln(P_end / P_start)",
                "note": "对数收益率可加，适合多期累计与统计建模",
            }

        elif op == "cumulative_return":
            if not returns:
                return {"success": False, "error": "需要 returns 数组"}
            cum = 1.0
            for r in returns:
                cum *= (1 + r)
            total = cum - 1
            return {
                "success": True,
                "operation": "cumulative_return",
                "periods": len(returns),
                "result": round(total, 6),
                "percentage": f"{total*100:.4f}%",
                "ending_multiple": round(cum, 6),
                "formula": "R_cum = Π(1 + r_i) - 1",
            }

        elif op == "annualize":
            if rate is None:
                return {"success": False, "error": "需要 rate（单期收益率）"}
            if periods_per_year <= 0:
                return {"success": False, "error": "periods_per_year 必须 > 0"}
            ann = (1 + rate) ** periods_per_year - 1
            return {
                "success": True,
                "operation": "annualize",
                "period_return": rate,
                "periods_per_year": periods_per_year,
                "annualized_return": round(ann, 6),
                "percentage": f"{ann*100:.4f}%",
                "formula": "R_ann = (1 + R_period)^N - 1",
                "common_N": {"daily": 252, "weekly": 52, "monthly": 12},
            }

        # ─────────── 时间价值 ───────────
        elif op == "compound":
            if principal is None or rate is None or periods is None:
                return {"success": False,
                        "error": "需要 principal / rate / periods"}
            fv = principal * (1 + rate) ** periods
            interest = fv - principal
            return {
                "success": True,
                "operation": "compound",
                "principal": principal,
                "rate": rate,
                "periods": periods,
                "future_value": round(fv, 4),
                "interest_earned": round(interest, 4),
                "formula": "FV = PV * (1 + r)^n",
            }

        elif op == "present_value":
            if future_value is None or rate is None or periods is None:
                return {"success": False,
                        "error": "需要 future_value / rate / periods"}
            pv = future_value / (1 + rate) ** periods
            return {
                "success": True,
                "operation": "present_value",
                "future_value": future_value,
                "rate": rate,
                "periods": periods,
                "present_value": round(pv, 4),
                "discount": round(future_value - pv, 4),
                "formula": "PV = FV / (1 + r)^n",
            }

        # ─────────── 统计类 ───────────
        elif op == "volatility":
            if not returns or len(returns) < 2:
                return {"success": False,
                        "error": "volatility 至少需要 2 个收益率数据点"}
            std = statistics.stdev(returns)
            ann_vol = std * math.sqrt(periods_per_year)
            return {
                "success": True,
                "operation": "volatility",
                "sample_size": len(returns),
                "period_volatility": round(std, 6),
                "annualized_volatility": round(ann_vol, 6),
                "annualized_pct": f"{ann_vol*100:.4f}%",
                "periods_per_year": periods_per_year,
                "formula": "σ_ann = σ_period * √N",
            }

        elif op == "mean_return":
            if not returns:
                return {"success": False, "error": "需要 returns 数组"}
            mean = sum(returns) / len(returns)
            return {
                "success": True,
                "operation": "mean_return",
                "sample_size": len(returns),
                "mean": round(mean, 6),
                "percentage": f"{mean*100:.4f}%",
            }

        # ─────────── 通用表达式（兜底）───────────
        elif op == "evaluate":
            if not expression:
                return {"success": False, "error": "需要 expression"}

            # 安全检查
            allowed = r'^[\d\s\+\-\*\/\%\(\)\.\,\_a-zA-Z]+$'
            if not re.match(allowed, expression.strip()):
                return {"success": False, "error": "表达式含不允许字符"}
            for kw in ["import", "exec", "eval", "open", "os", "__"]:
                if kw in expression.lower():
                    return {"success": False, "error": f"禁止关键字: {kw}"}

            result = eval(expression.strip(), SAFE_MATH_CONTEXT)  # noqa: S307

            if isinstance(result, float):
                if math.isinf(result) or math.isnan(result):
                    return {"success": False, "error": "结果非有限数"}
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 10)

            return {
                "success": True,
                "operation": "evaluate",
                "expression": expression,
                "result": result,
            }

        else:
            return {
                "success": False,
                "error": f"未知 operation: {operation}",
                "valid_operations": [
                    "simple_return", "log_return", "cumulative_return",
                    "annualize", "compound", "present_value",
                    "volatility", "mean_return", "evaluate",
                ],
            }

    except ZeroDivisionError:
        return {"success": False, "operation": op, "error": "除零错误"}
    except (ValueError, OverflowError) as e:
        return {"success": False, "operation": op, "error": f"数值错误: {e}"}
    except Exception as e:
        return {"success": False, "operation": op, "error": str(e)}


# ─────────────────────────────────────────────
# Tool 3: trading_calendar — 交易日历
# ─────────────────────────────────────────────

# 主要市场的交易时段（北京时间）
MARKET_HOURS = {
    "A股":    {"tz": "Asia/Shanghai",  "open": (9, 30),  "close": (15, 0),
               "lunch": ((11, 30), (13, 0))},
    "港股":   {"tz": "Asia/Hong_Kong", "open": (9, 30),  "close": (16, 0),
               "lunch": ((12, 0), (13, 0))},
    "美股":   {"tz": "America/New_York", "open": (9, 30), "close": (16, 0),
               "lunch": None},
}


def _to_minutes(hm):
    return hm[0] * 60 + hm[1]


def _is_in_session(now_dt, market_cfg):
    """判断指定时间是否在某市场交易时段内"""
    if now_dt.weekday() >= 5:
        return False, "非交易日（周末）"

    tz = ZoneInfo(market_cfg["tz"])
    local = now_dt.astimezone(tz)
    cur_min = local.hour * 60 + local.minute

    open_min = _to_minutes(market_cfg["open"])
    close_min = _to_minutes(market_cfg["close"])

    if cur_min < open_min:
        return False, f"未开盘（距开盘还有 {open_min - cur_min} 分钟）"
    if cur_min > close_min:
        return False, "已收盘"

    if market_cfg.get("lunch"):
        lunch_start = _to_minutes(market_cfg["lunch"][0])
        lunch_end = _to_minutes(market_cfg["lunch"][1])
        if lunch_start <= cur_min <= lunch_end:
            return False, "午间休市"

    return True, "盘中交易"


def trading_calendar(
    action: str,
    date: str = None,
    date2: str = None,
    market: str = "A股",
) -> dict:
    """
    交易日历工具，action 枚举：

    - now:                  当前时间 + 各市场开盘状态
    - market_status:        指定市场当前是否开盘
    - next_trading_day:     下一个交易日（按周一到周五近似）
    - trading_days_between: 两日期间交易日数

    ⚠️ 仅按周一到周五近似，未排除法定节假日。
    """
    tz = ZoneInfo("Asia/Shanghai")

    def parse(s):
        if not s or s.lower() in ("today", "now", "今天"):
            return datetime.datetime.now(tz)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.datetime.strptime(s, fmt).replace(tzinfo=tz)
            except ValueError:
                continue
        raise ValueError(f"日期格式无法解析: {s}（支持 YYYY-MM-DD）")

    try:
        # ── now: 当前时间 + 所有市场状态 ──
        if action == "now":
            now = datetime.datetime.now(tz)
            markets = {}
            for name, cfg in MARKET_HOURS.items():
                is_open, status = _is_in_session(now, cfg)
                markets[name] = {"is_open": is_open, "status": status}

            return {
                "success": True,
                "action": "now",
                "beijing_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": now.strftime("%A"),
                "weekday_cn": ["周一","周二","周三","周四",
                               "周五","周六","周日"][now.weekday()],
                "is_weekday": now.weekday() < 5,
                "markets": markets,
            }

        # ── market_status: 单一市场详情 ──
        elif action == "market_status":
            if market not in MARKET_HOURS:
                return {"success": False,
                        "error": f"未知市场: {market}",
                        "supported_markets": list(MARKET_HOURS.keys())}
            now = datetime.datetime.now(tz)
            cfg = MARKET_HOURS[market]
            is_open, status = _is_in_session(now, cfg)
            return {
                "success": True,
                "action": "market_status",
                "market": market,
                "is_open": is_open,
                "status": status,
                "session_config": {
                    "timezone": cfg["tz"],
                    "open": f"{cfg['open'][0]:02d}:{cfg['open'][1]:02d}",
                    "close": f"{cfg['close'][0]:02d}:{cfg['close'][1]:02d}",
                    "lunch_break": (
                        f"{cfg['lunch'][0][0]:02d}:{cfg['lunch'][0][1]:02d}-"
                        f"{cfg['lunch'][1][0]:02d}:{cfg['lunch'][1][1]:02d}"
                        if cfg.get("lunch") else "无"
                    ),
                },
                "as_of": now.strftime("%Y-%m-%d %H:%M:%S"),
            }

        # ── next_trading_day ──
        elif action == "next_trading_day":
            dt = parse(date)
            # 寻找下一个周一到周五
            next_dt = dt + datetime.timedelta(days=1)
            while next_dt.weekday() >= 5:
                next_dt += datetime.timedelta(days=1)
            return {
                "success": True,
                "action": "next_trading_day",
                "from_date": dt.strftime("%Y-%m-%d"),
                "next_trading_day": next_dt.strftime("%Y-%m-%d"),
                "weekday": next_dt.strftime("%A"),
                "days_forward": (next_dt.date() - dt.date()).days,
                "note": "未排除法定节假日",
            }

        # ── trading_days_between ──
        elif action == "trading_days_between":
            d1, d2 = parse(date), parse(date2)
            delta = (d2.date() - d1.date()).days
            sign = 1 if delta > 0 else -1
            trading_days = sum(
                1 for i in range(abs(delta))
                for d in [d1.date() + datetime.timedelta(days=i * sign)]
                if d.weekday() < 5
            )
            return {
                "success": True,
                "action": "trading_days_between",
                "from": d1.strftime("%Y-%m-%d"),
                "to": d2.strftime("%Y-%m-%d"),
                "calendar_days": delta,
                "trading_days_approx": trading_days,
                "weeks": round(abs(delta) / 7, 2),
                "note": "近似值，未排除春节/国庆等法定节假日",
            }

        else:
            return {
                "success": False,
                "error": f"未知 action: {action}",
                "valid_actions": ["now", "market_status",
                                  "next_trading_day", "trading_days_between"],
            }

    except ValueError as e:
        return {"success": False, "action": action, "error": str(e)}
    except Exception as e:
        return {"success": False, "action": action, "error": f"日历工具异常: {e}"}


# ════════════════════════════════════════════════════════════
# 第二部分：工具注册
# ════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "market_news_search": market_news_search,
    "quant_calc":         quant_calc,
    "trading_calendar":   trading_calendar,
}

TOOL_SCHEMAS = [
    {
        "name": "market_news_search",
        "description": """财经新闻定向搜索。
适用：公司公告、宏观数据（CPI/PMI/利率决议）、监管动态、行业事件、市场行情。
不适用：数学计算（用 quant_calc）、日期或交易日（用 trading_calendar）。
建议关键词：英文 + 股票代码效果最佳，如 'NVDA Q3 earnings 2025'。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（中英文均可）",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5, "minimum": 1, "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "quant_calc",
        "description": """量化专用计算器。
operation 枚举：
- simple_return:      简单收益率 (end/start - 1)
- log_return:         对数收益率 ln(end/start)
- cumulative_return:  累计收益率（returns 数组）
- annualize:          年化收益率（单期 + 频率）
- compound:           复利终值
- present_value:      现值贴现
- volatility:         年化波动率（returns 数组）
- mean_return:        平均收益率
- evaluate:           通用表达式求值（sqrt/log/exp/pi）
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["simple_return", "log_return", "cumulative_return",
                             "annualize", "compound", "present_value",
                             "volatility", "mean_return", "evaluate"],
                },
                "expression":       {"type": "string"},
                "start_price":      {"type": "number"},
                "end_price":        {"type": "number"},
                "returns":          {"type": "array",
                                     "items": {"type": "number"}},
                "rate":             {"type": "number",
                                     "description": "利率/收益率（小数，如 0.05）"},
                "periods":          {"type": "integer"},
                "periods_per_year": {"type": "integer", "default": 252,
                                     "description": "日频=252, 周频=52, 月频=12"},
                "principal":        {"type": "number"},
                "future_value":     {"type": "number"},
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trading_calendar",
        "description": """交易日历查询。
action 枚举：
- now:                  当前时间 + 沪深/港股/美股盘中状态
- market_status:        指定市场（A股/港股/美股）的开盘详情
- next_trading_day:     某日期之后的下一个交易日
- trading_days_between: 两个日期间的交易日数
⚠️ 按周一到周五近似，未排除法定节假日。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "market_status",
                             "next_trading_day", "trading_days_between"],
                },
                "date":   {"type": "string",
                           "description": "日期 YYYY-MM-DD 或 'today'"},
                "date2":  {"type": "string"},
                "market": {"type": "string",
                           "enum": ["A股", "港股", "美股"],
                           "default": "A股"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
]


# ════════════════════════════════════════════════════════════
# 第三部分：Agent 核心循环（ReAct while 循环）
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Role / 角色
You are QuantAgent, an elite AI assistant specializing in quantitative 
finance and systematic trading. You operate at the intersection of 
macroeconomic analysis, factor-based investing, and derivatives pricing.
你是 QuantAgent，一位精通量化金融与系统化交易的顶级 AI 助手，配备三种量化标准工具：
1. market_news_search  — 财经新闻定向搜索（公司公告 / 宏观数据 / 监管动态）
2. quant_calc          — 量化专用计算器（收益率 / 年化 / 复利 / 波动率）
3. trading_calendar    — 交易日历（盘中状态 / 下一交易日 / 交易日数）
你的能力横跨宏观经济分析、因子投资框架与衍生品定价三大领域。
专业背景 / Expertise Profile
量化核心：统计套利、多因子模型（Fama-French 五因子）、动量/反转策略
衍生品定价：Black-Scholes、Heston 随机波动率模型、Greeks 计算
宏观覆盖：全球主要市场（A股/港股/美股/欧股）、商品期货、汇率
风险管理：VaR、CVaR、最大回撤、夏普比率、信息比率
数据处理：时间序列分析、协整检验、机器学习信号提取
知识边界 / Knowledge Boundaries
你能做 (CAN DO):
✅ 基于当前接入数据的量化因子分析与打分
✅ 技术指标计算与解读（RSI/MACD/布林带/ATR等）
✅ 期权定价估算与Greeks敏感性分析
✅ 行业轮动逻辑推演与宏观传导路径梳理
✅ 历史回测逻辑验证与策略评估框架
✅ 风险敞口评估与对冲方案建议
你不做 (CANNOT DO):
❌ 不提供任何形式的"保证盈利"承诺
❌ 不对具体持仓金额给出绝对化建议
❌ 不替代持牌投资顾问的法律合规职能
❌ 不在数据缺失时伪造或推测数据

Core Task / 核心任务
任务优先级框架（按用户请求匹配）
Level 1 — 宏观行业扫描
分析全球/中国行业景气度，识别当前处于景气上行周期的板块，
输出行业轮动热力图数据 + 资金流向研判。
Level 2 — 个股量化评分
基于多因子模型对目标股票进行打分：
价值因子（P/E、P/B、EV/EBITDA）
成长因子（营收增速、EPS增速、ROE趋势）
动量因子（1M/3M/6M/12M超额收益）
质量因子（财务杠杆、现金流质量）
技术因子（RSI、MACD背离、成交量异常）
Level 3 — 衍生品策略分析
期权/期货的方向性分析、波动率策略选择、
Greeks敏感度报告、保护性对冲方案设计。
Level 4 — 组合风险评估
多资产组合的相关性分析、VaR计算、
压力测试场景模拟、再平衡建议。

System Context / 系统上下文
当前系统已接入以下数据源（Data Sources Connected）：
📊 实时行情：A股（沪深两市）、港股、美股 Level 1 行情
📰 财经资讯：Bloomberg风格财经新闻流、公告数据
📈 量化数据库：财务报表（TTM）、行业分类（申万三级）
🌐 宏观数据：CPI/PPI/PMI/GDP增速（中国/美国/欧元区）
📉 衍生品：期权链数据（A50/沪深300/主流美股ETF期权）
🔄 资金流向：北向资金、融资融券余额、龙虎榜
数据延迟说明：
实时行情：15分钟延迟（非订阅用户）
财务数据：最近一期季报/年报
宏观数据：最新公布值（注明发布日期）
⚠️ 所有分析结论必须注明数据来源与截止日期。
⚠️ 如数据不可用，明确标注 [DATA UNAVAILABLE] 而非猜测。

Output Format / 输出格式规范
规则一：文字分析 → Markdown 输出
所有报告类、解读类、策略类内容使用 Markdown，结构如下：

📊 [报告标题] | [Report Title]
生成时间 / Generated: YYYY-MM-DD HH:MM (CST)  
数据截止 / Data As Of: YYYY-MM-DD  
风险等级 / Risk Level: 🟢低 / 🟡中 / 🔴高
一、核心结论 / Key Takeaways
3-5条要点，每条不超过50字，先中文后英文
二、量化分析 / Quantitative Analysis
[具体分析内容，含因子得分、技术指标、数据来源]
三、风险提示 / Risk Warnings
[必须包含的风险声明]
四、操作参考 / Actionable Reference
[方向性建议，含入场/止损/止盈参考区间，附计算依据]

⚠️ 免责声明：本报告基于量化模型输出，仅供研究参考，
不构成投资建议。投资有风险，入市需谨慎。

规则二：图表数据 → HTML 页面输出
当用户请求以下内容时，生成完整可运行的 HTML 页面：
K线图 / 技术指标图
因子得分雷达图
行业热力图
期权Greeks曲面图
组合相关性矩阵
HTML规范：
使用 ECharts 或 Plotly.js（CDN引入）
深色主题（背景 #0d1117，主色 #00d4aa）
图表必须包含数据来源注释
支持交互（hover显示详细数据）
响应式布局
规则三：数据表格
使用 Markdown 表格，必须包含来源列：
| 指标 | 数值 | 较上期 | 来源 | 日期 |
|------|------|--------|------|------|

Constraints / 约束规则
专业数据引用规范
每个量化指标必须附带计算公式或定义
示例：「夏普比率 = (年化收益 - 无风险利率) / 年化波动率，
      其中无风险利率取当期10年期国债收益率」
机构预测数据须注明来源（Bloomberg/Wind/Reuters/公司公告）
历史数据须注明时间区间（如「基于过去252个交易日数据」）
结论置信度标注
每项分析结论附置信度标签：
🟢 高置信（数据充分，模型拟合优良）
🟡 中置信（数据部分，存在假设前提）
🔴 低置信（数据不足，仅供参考）
禁止行为
❌ 不使用"一定涨/跌"、"保证收益"等绝对化表述
❌ 不在缺乏数据时编造具体数值
❌ 不忽略风险提示板块（每份报告必须包含）
❌ 不推荐超出用户风险承受能力的杠杆比例
Language / 语言风格
双语规范
专业术语：中文名称 + 英文缩写
示例：「市盈率（P/E Ratio）」「隐含波动率（IV）」「夏普比率（Sharpe Ratio）」
标题：中英双语并列
核心结论：中英双语
详细分析：用户输入语言为准（中文输入→中文为主，英文输入→英文为主）
语气风格
专业严谨但避免堆砌术语
复杂概念必须附简明解释（"即..."/"换句话说..."）
数据说话，避免主观情绪化表达
对不确定性保持诚实：使用"可能"、"历史数据显示"、"模型预测"等措辞
专业词汇标准
量化因子：使用学术界通用命名（Momentum/Value/Quality/Low-Vol）
中国市场：使用监管机构标准表述（如「涨跌停」不写「限制幅度」）
期权术语：标准化（Delta/Gamma/Theta/Vega/Rho + 中文对应）
标准免责声明（每份报告必须附加）

⚠️ 风险提示 / Risk Disclaimer
中文版：
本报告由 QuantAgent AI 系统基于量化模型生成，所有内容仅供学术研究与参考，
不构成任何形式的投资建议或要约。历史收益不代表未来表现。
量化模型存在失效风险，尤其在极端市场环境（黑天鹅事件）下。
投资者应结合自身风险承受能力、资产状况及专业投资顾问意见做出独立判断。
投资有风险，入市需谨慎。
English Version:
This report is generated by QuantAgent AI based on quantitative models for 
research purposes only and does not constitute investment advice or solicitation. 
Past performance is not indicative of future results. Quantitative models carry 
inherent risks, particularly during extreme market conditions (black swan events). 
Investors should make independent decisions based on their own risk tolerance, 
financial situation, and professional advisory opinions.
工作原则：
- 涉及实时财经信息（财报、加息、指数）必须调用 market_news_search，不依赖训练数据
- 任何数值计算必须调用 quant_calc，禁止心算估算
- 涉及交易日、开盘时段必须调用 trading_calendar，不要把工作日当交易日
- 可并行调用多个工具
- 工具失败时读取 error / hint 字段后修正参数重试
- 最终回答附加风险提示，避免「保证盈利」「一定上涨」等绝对化表述
"""


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {
            "success": False,
            "error": f"工具 '{tool_name}' 不存在",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {e}"}
    except Exception as e:
        return {"success": False, "error": f"工具执行异常: {e}"}


def run_agent(user_query: str, max_iterations: int = 15, verbose: bool = True) -> str:
    """
    原生 ReAct Agent 主循环

    Thought  → 模型推理（tool_use 块之前的 text 块）
    Action   → 模型决定调用工具（tool_use 块）
    Observation → 执行工具，结果封装成 tool_result
    循环直到 stop_reason == "end_turn"
    """

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*65}")
    log(f"用户问题：{user_query}")
    log(f"{'='*65}")

    messages = [{"role": "user", "content": user_query}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log(f"\n── 迭代 {iteration:02d} {'─'*50}")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        log(f"   stop_reason: {response.stop_reason}")
        log(f"   content 块数: {len(response.content)}")

        for block in response.content:
            if block.type == "text" and block.text.strip():
                log(f"\nThought:\n{block.text.strip()}")

        if response.stop_reason == "end_turn":
            final = "\n".join(
                b.text for b in response.content
                if b.type == "text" and b.text.strip()
            )
            log(f"\n{'='*65}")
            log(f"最终答案：\n{final}")
            log(f"{'='*65}")
            log(f"\n总计迭代：{iteration} 轮")
            return final

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            log(f"stop_reason={response.stop_reason} 但无 tool_use，意外退出")
            break

        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        tool_results = []

        for tu in tool_use_blocks:
            log(f"\nAction: {tu.name}")
            log(f"   Input:  {json.dumps(tu.input, ensure_ascii=False)}")

            observation = dispatch_tool(tu.name, tu.input)

            obs_str = json.dumps(observation, ensure_ascii=False, default=str)
            if len(obs_str) > 3000:
                if observation.get("results"):
                    observation["results"] = observation["results"][:3]
                    observation["truncated"] = True
                obs_str = json.dumps(observation, ensure_ascii=False, default=str)

            log(f"   Obs:    {obs_str[:300]}{'...' if len(obs_str) > 300 else ''}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": obs_str,
                **({"is_error": True} if not observation.get("success", True) else {}),
            })

        messages.append({
            "role": "user",
            "content": tool_results,
        })

    return f"超过最大迭代次数 ({max_iterations})，请简化问题后重试。"


# ════════════════════════════════════════════════════════════
# 第四部分：测试用例（量化场景）
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█"*65)
    print("  测试 1：单工具 · 收益率计算")
    print("█"*65)
    run_agent("我 100 元买入，120 元卖出，简单收益率和对数收益率各是多少？"
              "如果这段持仓 30 个交易日，年化是多少？")

    print("\n\n" + "█"*65)
    print("  测试 2：单工具 · 交易时段判断")
    print("█"*65)
    run_agent("现在美股是开盘还是收盘？A 股呢？港股呢？")

    print("\n\n" + "█"*65)
    print("  测试 3：搜索 + 计算（多工具串联）")
    print("█"*65)
    run_agent("搜索一下英伟达 NVDA 最新一期财报情况，"
              "并按我手里这组日收益率算一下年化波动率："
              "[0.012, -0.008, 0.015, 0.003, -0.021, 0.008, 0.011, -0.005, "
              "0.018, -0.012, 0.007, 0.022, -0.015, 0.009, 0.013]")

    print("\n\n" + "█"*65)
    print("  测试 4：三工具全部用上（综合任务）")
    print("█"*65)
    run_agent("查一下美联储最近的利率决议消息，"
              "美股现在开盘吗？"
              "如果我现在投 10 万元，按年化 8% 复利 5 年后是多少？"
              "另外 2026 年春节假期假设是 2 月 16 日到 2 月 24 日，"
              "这段期间会损失多少 A 股交易日？")
