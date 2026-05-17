# ============================================================
#  QuantAgent v1.0 — 简易量化分析 Agent
#  ─────────────────────────────────────────────
#  基于 system_prompt.txt 的 QuantAgent 角色设定
#  对标"Agent 工程 + 量化领域知识"复合人才方向
#
#  核心能力：
#    1. 行情快照（mock 数据，标注 [DATA UNAVAILABLE / MOCK]）
#    2. 多因子打分（Value/Growth/Momentum/Quality/Technical）
#    3. 技术指标计算（RSI / MACD / Bollinger Bands）
#    4. 期权 Black-Scholes 定价与 Greeks
#    5. 组合风险指标（Sharpe / Sortino / VaR / MaxDD）
#    6. 财经资讯搜索（DuckDuckGo）
#    7. 交易日历（工作日 ≈ 交易日近似）
#
#  依赖：pip install anthropic
#  注意：实盘数据接入请替换 market_quote 内部实现
# ============================================================

import json
import math
import os
import re
import statistics
import datetime
import urllib.request
import urllib.parse
from zoneinfo import ZoneInfo

# 自动加载项目 .env 文件（如果存在），免去手动 export
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # 没装 dotenv 时静默跳过，回退到系统环境变量

from openai import OpenAI

# ════════════════════════════════════════════════════════════
# LLM 客户端：DeepSeek（OpenAI 协议兼容）
# ════════════════════════════════════════════════════════════
# 申请 API Key: https://platform.deepseek.com/api_keys
# 定价: deepseek-chat 输入 2元/M tokens，输出 8元/M tokens（约为 Claude 1/10）

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 也可用 deepseek-reasoner

if not DEEPSEEK_API_KEY:
    print("⚠️  未设置 DEEPSEEK_API_KEY，Agent 主循环调用 LLM 时会报错。"
          "工具本身不受影响。")
    print("   申请: https://platform.deepseek.com/api_keys")
    print("   设置: export DEEPSEEK_API_KEY=sk-...")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "not-set",
    base_url="https://api.deepseek.com",
)

# ════════════════════════════════════════════════════════════
# 数据源可选依赖检测
# ════════════════════════════════════════════════════════════

# Finnhub：免费 key 申请 https://finnhub.io
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# AKShare：国内 A 股最全数据库（pip install akshare）
try:
    import akshare as _ak
    _AKSHARE_AVAILABLE = True
except ImportError:
    _ak = None
    _AKSHARE_AVAILABLE = False


# ════════════════════════════════════════════════════════════
# 第一部分：量化工具实现
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# 多源数据辅助层（行情 + 新闻）
# ════════════════════════════════════════════════════════════

def _detect_market(symbol: str) -> str:
    """
    识别股票代码所属市场: 'A' / 'HK' / 'US' / 'unknown'
    A 股覆盖：A股、ETF/LOF、可转债、B 股等所有 6 位数字代码
    """
    s = symbol.strip().upper()
    # 带交易所后缀 .SS/.SH/.SZ 或纯 6 位数字
    if re.search(r'\.(SS|SH|SZ)$', s) or re.match(r'^\d{6}$', s):
        return "A"
    if s.endswith('.HK') or (s.isdigit() and 4 <= len(s) <= 5):
        return "HK"
    if re.match(r'^[A-Z]{1,5}$', s) or s.endswith('.US'):
        return "US"
    return "unknown"


def _sina_prefix(symbol: str):
    """
    股票代码 → 新浪行情接口前缀

    A 股交易所路由（按首位数字）：
      上交所 sh：5xx(ETF/基金) / 6xx(A股) / 9xx(B股) / 110/113(可转债)
      深交所 sz：000(主板) / 002(中小板) / 1xx(ETF/基金) / 2xx(B股) / 3xx(创业板)
    """
    s = symbol.strip().upper()
    for suf in (".SS", ".SH", ".SZ", ".HK", ".US"):
        s = s.replace(suf, "")
    market = _detect_market(symbol)

    if market == "A":
        # 首位区分上交所/深交所
        if s.startswith(('5', '6', '9')):
            return f"sh{s}"
        if s.startswith(('0', '1', '2', '3')):
            return f"sz{s}"
        return None
    if market == "HK":
        return f"hk{s.zfill(5)}"
    if market == "US":
        return f"gb_{s.lower()}"
    return None


def _quote_sina(symbol: str):
    """新浪财经实时行情（A/HK/US 通用）"""
    prefix = _sina_prefix(symbol)
    if not prefix:
        return None
    try:
        url = f"https://hq.sinajs.cn/list={prefix}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("gbk", errors="ignore")
        m = re.search(r'"([^"]*)"', text)
        if not m or not m.group(1):
            return None
        f = m.group(1).split(',')
        if len(f) < 5 or not f[0]:
            return None

        if prefix.startswith(('sh', 'sz')):
            return _parse_sina_a(prefix, f)
        if prefix.startswith('gb_'):
            return _parse_sina_us(prefix, f)
        if prefix.startswith('hk'):
            return _parse_sina_hk(prefix, f)
    except Exception:
        return None
    return None


def _parse_sina_a(prefix, f):
    try:
        prev, cur = float(f[2]), float(f[3])
        return {
            "symbol": prefix[2:], "name": f[0], "market": "A股",
            "price": cur, "prev_close": prev, "open": float(f[1]),
            "high": float(f[4]), "low": float(f[5]),
            "change_pct": round((cur - prev) / prev * 100, 4) if prev > 0 else 0,
            "volume": int(f[8]) if f[8] else 0,
            "amount_cny": float(f[9]) if f[9] else 0,
            "as_of": f"{f[30]} {f[31]}" if len(f) > 31 else None,
            "data_source": "新浪财经",
        }
    except (ValueError, IndexError):
        return None


def _parse_sina_us(prefix, f):
    try:
        return {
            "symbol": prefix[3:].upper(), "name": f[0], "market": "美股",
            "price": float(f[1]),
            "change_pct": float(f[2]),
            "change_amount": float(f[4]) if len(f) > 4 and f[4] else 0,
            "open": float(f[5]) if len(f) > 5 and f[5] else 0,
            "high": float(f[6]) if len(f) > 6 and f[6] else 0,
            "low": float(f[7]) if len(f) > 7 and f[7] else 0,
            "volume": int(float(f[10])) if len(f) > 10 and f[10] else 0,
            "as_of": f[3] if len(f) > 3 else None,
            "data_source": "新浪财经",
        }
    except (ValueError, IndexError):
        return None


def _parse_sina_hk(prefix, f):
    try:
        return {
            "symbol": prefix[2:], "name": f[1], "market": "港股",
            "price": float(f[6]), "prev_close": float(f[3]),
            "open": float(f[2]), "high": float(f[4]), "low": float(f[5]),
            "change_pct": float(f[8]) if len(f) > 8 and f[8] else 0,
            "volume": int(float(f[12])) if len(f) > 12 and f[12] else 0,
            "as_of": f"{f[17]} {f[18]}" if len(f) > 18 else None,
            "data_source": "新浪财经",
        }
    except (ValueError, IndexError):
        return None


def _news_eastmoney(keyword: str, max_results: int = 5):
    """东方财富新闻搜索（A股新闻覆盖最佳）"""
    try:
        params = {
            "uid": "", "keyword": keyword,
            "type": ["cmsArticleWebOld"], "client": "web",
            "pageIndex": 1, "pageSize": max_results,
        }
        url = (f"https://search-api-web.eastmoney.com/search/jsonp"
               f"?cb=cb&param={urllib.parse.quote(json.dumps(params))}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        m = re.search(r'cb\((.*)\)\s*$', text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(1))
        items = data.get("result", {}).get("cmsArticleWebOld", [])

        news = []
        for it in items[:max_results]:
            title = re.sub(r'</?em>', '', it.get("title", ""))
            content = re.sub(r'</?em>', '', it.get("content", ""))
            news.append({
                "title": title,
                "publisher": "东方财富",
                "link": it.get("url", ""),
                "published_at": it.get("date", ""),
                "summary": content[:200],
                "source_api": "eastmoney",
            })
        return news
    except Exception:
        return []


def _news_finnhub(symbol: str, max_results: int = 5):
    """Finnhub 全球公司新闻（需 FINNHUB_API_KEY 环境变量）"""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.date.today()
        from_d = today - datetime.timedelta(days=14)
        url = (f"https://finnhub.io/api/v1/company-news"
               f"?symbol={urllib.parse.quote(symbol)}"
               f"&from={from_d}&to={today}&token={FINNHUB_API_KEY}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        if not isinstance(items, list):
            return []

        news = []
        for it in items[:max_results]:
            ts = it.get("datetime", 0)
            news.append({
                "title": it.get("headline", ""),
                "publisher": it.get("source", "Finnhub"),
                "link": it.get("url", ""),
                "published_at": (datetime.datetime.fromtimestamp(ts)
                                 .strftime("%Y-%m-%d %H:%M") if ts else ""),
                "summary": (it.get("summary", "") or "")[:200],
                "related_tickers": [symbol],
                "source_api": "finnhub",
            })
        return news
    except Exception:
        return []


def _news_akshare(symbol: str, max_results: int = 5):
    """AKShare 个股新闻（仅 A 股，需 pip install akshare）"""
    if not _AKSHARE_AVAILABLE:
        return []
    try:
        s = symbol.replace(".SS", "").replace(".SZ", "")
        df = _ak.stock_news_em(symbol=s)
        news = []
        for _, row in df.head(max_results).iterrows():
            news.append({
                "title": row.get("新闻标题", ""),
                "publisher": row.get("文章来源", "AKShare"),
                "link": row.get("新闻链接", ""),
                "published_at": str(row.get("发布时间", "")),
                "summary": (row.get("新闻内容", "") or "")[:200],
                "related_tickers": [s],
                "source_api": "akshare",
            })
        return news
    except Exception:
        return []


# ─────────────────────────────────────────────
# Tool 1: 行情快照（多源：Sina → Yahoo → MOCK）
# ─────────────────────────────────────────────

# MOCK 兜底库（数据源全部失败时使用，明确标注）
MOCK_QUOTES = {
    "AAPL":   {"name": "Apple Inc.",      "price": 213.40, "change_pct": +1.24,
               "volume_m": 52.3,  "pe": 33.5, "market_cap_b": 3250},
    "MSFT":   {"name": "Microsoft Corp.", "price": 428.15, "change_pct": -0.42,
               "volume_m": 21.7,  "pe": 36.1, "market_cap_b": 3180},
    "NVDA":   {"name": "NVIDIA Corp.",    "price": 1185.20,"change_pct": +2.78,
               "volume_m": 38.4,  "pe": 72.4, "market_cap_b": 2920},
    "TSLA":   {"name": "Tesla Inc.",      "price": 248.50, "change_pct": -1.85,
               "volume_m": 95.2,  "pe": 65.8, "market_cap_b": 791},
    "600519": {"name": "贵州茅台",         "price": 1685.0, "change_pct": +0.68,
               "volume_m": 2.4,   "pe": 22.3, "market_cap_b": 2117},
    "000858": {"name": "五粮液",           "price": 142.30, "change_pct": -0.35,
               "volume_m": 8.7,   "pe": 18.6, "market_cap_b": 552},
}


def market_quote(symbol: str) -> dict:
    """
    多源实时行情快照。
    源优先级：新浪财经（实时）→ MOCK 兜底（仅当 Sina 失败且代码在 MOCK 库内）
    """
    symbol_raw = symbol.strip()
    sources_tried = []

    # ── 1. 新浪财经实时行情（A/HK/US 通用）──
    sina = _quote_sina(symbol_raw)
    sources_tried.append({"source": "新浪财经",
                          "ok": sina is not None})
    if sina:
        return {
            "success": True,
            "symbol": sina["symbol"],
            "name": sina["name"],
            "market": sina["market"],
            "price": sina["price"],
            "change_pct": sina["change_pct"],
            "open": sina.get("open"),
            "high": sina.get("high"),
            "low": sina.get("low"),
            "prev_close": sina.get("prev_close"),
            "volume": sina.get("volume"),
            "amount_cny": sina.get("amount_cny"),
            "as_of": sina.get("as_of"),
            "data_source": "新浪财经实时行情",
            "sources_tried": sources_tried,
        }

    # ── 2. MOCK 兜底（仅在 Sina 完全无返回时）──
    s_upper = symbol_raw.upper()
    quote = MOCK_QUOTES.get(s_upper)
    sources_tried.append({"source": "MOCK", "ok": quote is not None})
    if quote:
        return {
            "success": True,
            "symbol": s_upper,
            "name": quote["name"],
            "price": quote["price"],
            "change_pct": quote["change_pct"],
            "pe_ratio": quote["pe"],
            "market_cap_billion_usd": quote["market_cap_b"],
            "data_source": "[MOCK DATA] 演示数据，非实时",
            "sources_tried": sources_tried,
            "warning": "新浪行情接口未返回，已降级至 MOCK 数据",
        }

    return {
        "success": False,
        "error_type": "all_sources_failed",
        "error": f"代码 '{symbol}' 在所有数据源都未找到",
        "sources_tried": sources_tried,
        "hint": "确认代码格式：美股 AAPL / A股 600519 / 港股 0700",
    }


# ─────────────────────────────────────────────
# Tool 2: 多因子打分（Fama-French 思路简化版）
# ─────────────────────────────────────────────

# 预制因子库（演示用，真实场景应从财务数据库计算）
MOCK_FACTOR_RAW = {
    "AAPL":   {"value": 35,  "growth": 78,  "momentum": 82,
               "quality": 92, "technical": 68},
    "MSFT":   {"value": 42,  "growth": 75,  "momentum": 71,
               "quality": 95, "technical": 55},
    "NVDA":   {"value": 18,  "growth": 96,  "momentum": 94,
               "quality": 88, "technical": 85},
    "TSLA":   {"value": 25,  "growth": 65,  "momentum": 48,
               "quality": 72, "technical": 35},
    "600519": {"value": 58,  "growth": 62,  "momentum": 55,
               "quality": 98, "technical": 60},
    "000858": {"value": 65,  "growth": 48,  "momentum": 42,
               "quality": 88, "technical": 50},
}

# ─────────────────────────────────────────────
# 真实因子计算（AKShare，A 股）
# ─────────────────────────────────────────────

import time as _time

_FACTOR_CACHE = {}     # {symbol: (timestamp, result_dict)}
_FACTOR_CACHE_TTL = 3600  # 因子数据缓存 1 小时


def _compute_factors_akshare(symbol: str):
    """
    从 AKShare 拉真实财务/价格数据，计算 5 因子原始值。
    成功返回 dict（含 factors + raw + 计算时间），失败返回 None。
    仅支持 A 股；缓存 1 小时。
    """
    if not _AKSHARE_AVAILABLE:
        return None

    s = symbol.replace(".SS", "").replace(".SH", "").replace(".SZ", "")

    # 命中缓存
    cached = _FACTOR_CACHE.get(s)
    if cached and (_time.time() - cached[0]) < _FACTOR_CACHE_TTL:
        return cached[1]

    factors = {}
    raw = {}

    # 现价（用于算 PE/PB）
    quote = _quote_sina(s)
    price = quote.get("price") if quote else None
    if price:
        raw["price"] = price

    # ── 财务因子：EPS / BPS / ROE / 营收增速 / 净利增速 ──
    try:
        df = _ak.stock_financial_analysis_indicator(symbol=s, start_year="2024")
        if len(df) > 0:
            r = df.iloc[0]
            eps = float(r.get('摊薄每股收益(元)', 0) or 0)
            bps = float(r.get('每股净资产_调整前(元)', 0) or 0)
            roe = float(r.get('净资产收益率(%)', 0) or 0)
            rev_g = float(r.get('主营业务收入增长率(%)', 0) or 0)
            prof_g = float(r.get('净利润增长率(%)', 0) or 0)

            raw["eps"] = eps
            raw["bps"] = bps
            raw["roe_pct"] = roe
            raw["revenue_growth_pct"] = rev_g
            raw["profit_growth_pct"] = prof_g
            raw["report_date"] = str(r.get("日期", ""))

            # Value 评分：用 PE 反向映射（PE 低 → 分高）
            if price and eps > 0:
                pe = price / (eps * 4)   # 季报粗略年化
                raw["pe_estimated"] = round(pe, 2)
                factors["value"] = max(0, min(100, 100 - pe * 2))
            if price and bps > 0:
                raw["pb"] = round(price / bps, 2)

            # Growth 评分：营收 + 净利增速平均
            avg_g = (rev_g + prof_g) / 2
            factors["growth"] = max(0, min(100, 50 + avg_g * 1.5))

            # Quality 评分：ROE 越高越好
            factors["quality"] = max(0, min(100, roe * 4))
    except Exception:
        pass

    # ── 动量因子：近 3 个月收益 + 60 日均线背离 ──
    try:
        end_d = datetime.date.today().strftime('%Y%m%d')
        start_d = (datetime.date.today() - datetime.timedelta(days=120)).strftime('%Y%m%d')
        k = _ak.stock_zh_a_hist(symbol=s, period='daily',
                                 start_date=start_d, end_date=end_d, adjust='qfq')
        if len(k) >= 20:
            cur = float(k['收盘'].iloc[-1])
            ret_3m = (cur / float(k['收盘'].iloc[0]) - 1) * 100
            raw["return_3m_pct"] = round(ret_3m, 2)
            factors["momentum"] = max(0, min(100, 50 + ret_3m * 1.67))

            ma60 = float(k['收盘'].tail(60).mean()) if len(k) >= 60 else float(k['收盘'].mean())
            tech_pct = (cur / ma60 - 1) * 100
            raw["vs_ma60_pct"] = round(tech_pct, 2)
            factors["technical"] = max(0, min(100, 50 + tech_pct * 2))
    except Exception:
        pass

    if not factors:
        return None

    result = {
        "factors": {k: round(v, 1) for k, v in factors.items()},
        "raw": raw,
        "computed_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        "note": "PE 为季报×4 粗略年化估算，与 TTM 口径有差异",
    }
    _FACTOR_CACHE[s] = (_time.time(), result)
    return result


# 因子权重（可调，反映策略风格）
FACTOR_WEIGHTS = {
    "value":     0.20,
    "growth":    0.25,
    "momentum":  0.20,
    "quality":   0.25,
    "technical": 0.10,
}


def factor_score(symbol: str, style: str = "balanced") -> dict:
    """
    多因子综合打分。
    style: balanced(均衡) / value(价值) / growth(成长) / momentum(动量)
    返回 0-100 分综合得分，含各因子贡献度。
    """
    symbol = symbol.upper().strip()

    # 检测是否为概念词/中文/非 ticker 格式
    is_concept = bool(re.search(r'[一-鿿]', symbol)) or \
                 not re.match(r'^[A-Z0-9.]{1,10}$', symbol)
    if is_concept:
        return {
            "success": False,
            "error_type": "not_a_ticker",
            "error": f"'{symbol}' 不是有效股票代码，factor_score 只接受具体 ticker",
            "hint": "若要分析行业/概念：先用 market_news_search 找到相关公司，"
                    "再对每个 ticker 单独调用 factor_score",
            "example_flow": [
                "market_news_search('机器人概念股') → 拿到相关公司列表",
                "factor_score('300024')  # 对每只成份股打分",
            ],
        }

    # ── 优先尝试 AKShare 真因子（A 股可用）──
    market = _detect_market(symbol)
    real = _compute_factors_akshare(symbol) if market == "A" else None

    if real:
        # 真因子覆盖缺失的字段（部分接口失败时用 MOCK 兜底剩余字段）
        raw = {**MOCK_FACTOR_RAW.get(symbol, {
            "value": 50, "growth": 50, "momentum": 50,
            "quality": 50, "technical": 50,
        }), **real["factors"]}
        is_mock = False
        data_source = f"AKShare 实时计算（{len(real['factors'])}/5 因子真值）"
        real_meta = {
            "real_factors_count": len(real["factors"]),
            "real_factor_names": list(real["factors"].keys()),
            "underlying_data": real["raw"],
            "computed_at": real["computed_at"],
            "note": real["note"],
        }
    else:
        # 全 MOCK 兜底（非 A 股 / AKShare 不可用 / 没数据）
        raw = MOCK_FACTOR_RAW.get(symbol)
        if not raw:
            return {
                "success": False,
                "error_type": "no_data",
                "error": f"'{symbol}' 真实因子计算失败且 MOCK 库无此代码",
                "hint": "A 股代码应能拿到 AKShare 数据；美股/港股暂仅 MOCK 库内可用",
                "available_in_mock": list(MOCK_FACTOR_RAW.keys()),
            }
        is_mock = True
        data_source = "[MOCK DATA] 因子值为演示用预设"
        real_meta = None

    # 根据风格调整权重
    weights = dict(FACTOR_WEIGHTS)
    if style == "value":
        weights = {"value": 0.40, "quality": 0.30, "growth": 0.10,
                   "momentum": 0.10, "technical": 0.10}
    elif style == "growth":
        weights = {"growth": 0.40, "momentum": 0.25, "quality": 0.20,
                   "value": 0.05, "technical": 0.10}
    elif style == "momentum":
        weights = {"momentum": 0.45, "technical": 0.25, "growth": 0.15,
                   "quality": 0.10, "value": 0.05}

    composite = sum(raw[f] * w for f, w in weights.items())

    # 评级
    if composite >= 75:
        rating, confidence = "强烈推荐 / Strong Buy", "🟢 高置信"
    elif composite >= 60:
        rating, confidence = "推荐 / Buy", "🟢 高置信"
    elif composite >= 45:
        rating, confidence = "中性 / Hold", "🟡 中置信"
    elif composite >= 30:
        rating, confidence = "减持 / Reduce", "🟡 中置信"
    else:
        rating, confidence = "卖出 / Sell", "🔴 低置信"

    result = {
        "success": True,
        "symbol": symbol,
        "style": style,
        "factor_scores": raw,
        "weights_used": weights,
        "composite_score": round(composite, 2),
        "rating": rating,
        "confidence": confidence,
        "interpretation": {
            "value":     "PE 反向映射（越低分越高）",
            "growth":    "营收增速 + 净利增速 平均",
            "momentum":  "近 3 个月累计收益",
            "quality":   "ROE 净资产收益率",
            "technical": "现价 vs 60 日均线偏离",
        },
        "data_source": data_source,
        "is_mock": is_mock,
        "disclaimer": "因子模型存在失效风险，尤其在风格切换期",
    }
    if real_meta:
        result["real_data"] = real_meta
    return result


# ─────────────────────────────────────────────
# Tool 3: 技术指标计算（真实数学）
# ─────────────────────────────────────────────

def technical_indicator(
    prices: list,
    indicator: str,
    period: int = 14,
) -> dict:
    """
    根据收盘价序列计算技术指标。
    indicator: rsi / macd / bollinger / sma
    prices: 收盘价列表，时间正序（旧→新）
    """
    if not prices or len(prices) < 2:
        return {"success": False, "error": "价格序列至少需要 2 个数据点"}

    if not all(isinstance(p, (int, float)) and p > 0 for p in prices):
        return {"success": False, "error": "价格必须为正数"}

    indicator = indicator.lower()

    try:
        # ── RSI（相对强弱指标）──
        if indicator == "rsi":
            if len(prices) < period + 1:
                return {"success": False,
                        "error": f"RSI({period}) 至少需要 {period+1} 个价格点"}
            gains, losses = [], []
            for i in range(1, len(prices)):
                diff = prices[i] - prices[i-1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # 信号判读
            if rsi >= 70:
                signal = "超买区，警惕回调 / Overbought"
            elif rsi <= 30:
                signal = "超卖区，可能反弹 / Oversold"
            else:
                signal = "中性区间 / Neutral"

            return {
                "success": True,
                "indicator": "RSI",
                "period": period,
                "value": round(rsi, 2),
                "signal": signal,
                "formula": "RSI = 100 - 100/(1+RS)，RS = avg_gain/avg_loss",
                "data_points_used": len(prices),
            }

        # ── MACD ──
        elif indicator == "macd":
            if len(prices) < 26:
                return {"success": False,
                        "error": "MACD 至少需要 26 个价格点"}

            def ema(data, n):
                k = 2 / (n + 1)
                ema_val = data[0]
                for p in data[1:]:
                    ema_val = p * k + ema_val * (1 - k)
                return ema_val

            ema12 = ema(prices, 12)
            ema26 = ema(prices, 26)
            macd_line = ema12 - ema26
            # 简化版 signal：取 macd 序列后段 EMA9
            macd_series = []
            for i in range(26, len(prices) + 1):
                window = prices[:i]
                macd_series.append(ema(window, 12) - ema(window, 26))
            signal_line = ema(macd_series, 9) if len(macd_series) >= 9 else macd_line
            histogram = macd_line - signal_line

            trend = "金叉信号 / Bullish" if histogram > 0 else "死叉信号 / Bearish"

            return {
                "success": True,
                "indicator": "MACD",
                "macd_line": round(macd_line, 4),
                "signal_line": round(signal_line, 4),
                "histogram": round(histogram, 4),
                "interpretation": trend,
                "formula": "MACD = EMA(12) - EMA(26)，Signal = EMA(9, MACD)",
            }

        # ── Bollinger Bands ──
        elif indicator == "bollinger":
            if len(prices) < period:
                return {"success": False,
                        "error": f"布林带至少需要 {period} 个价格点"}
            window = prices[-period:]
            mean = sum(window) / period
            std = statistics.stdev(window) if period > 1 else 0
            upper = mean + 2 * std
            lower = mean - 2 * std
            current = prices[-1]

            if current >= upper:
                pos = "突破上轨，超买 / Above Upper Band"
            elif current <= lower:
                pos = "跌破下轨，超卖 / Below Lower Band"
            else:
                pct = (current - lower) / (upper - lower) * 100
                pos = f"运行于通道内 ({pct:.1f}% 位置)"

            return {
                "success": True,
                "indicator": "Bollinger Bands",
                "period": period,
                "middle": round(mean, 2),
                "upper": round(upper, 2),
                "lower": round(lower, 2),
                "current_price": current,
                "bandwidth": round((upper - lower) / mean * 100, 2),
                "position": pos,
                "formula": "Middle = SMA(N)；Upper/Lower = Middle ± 2σ",
            }

        # ── SMA 简单移动均线 ──
        elif indicator == "sma":
            if len(prices) < period:
                return {"success": False,
                        "error": f"SMA({period}) 至少需要 {period} 个数据点"}
            sma = sum(prices[-period:]) / period
            current = prices[-1]
            return {
                "success": True,
                "indicator": "SMA",
                "period": period,
                "value": round(sma, 4),
                "current_price": current,
                "deviation_pct": round((current - sma) / sma * 100, 2),
                "signal": "价格在均线上方" if current > sma else "价格在均线下方",
            }

        else:
            return {
                "success": False,
                "error": f"未知指标 '{indicator}'",
                "supported": ["rsi", "macd", "bollinger", "sma"],
            }

    except Exception as e:
        return {"success": False, "error": f"指标计算异常: {e}"}


# ─────────────────────────────────────────────
# Tool 4: Black-Scholes 期权定价与 Greeks
# ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """标准正态累积分布函数"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """标准正态密度函数"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> dict:
    """
    欧式期权 Black-Scholes 定价 + Greeks。

    参数：
      spot:           标的现价 S
      strike:         行权价 K
      time_to_expiry: 到期时间（年），如 0.25 = 3个月
      risk_free_rate: 无风险利率（小数，如 0.03 = 3%）
      volatility:     年化波动率（小数，如 0.25 = 25%）
      option_type:    call(认购) / put(认沽)
    """
    if any(x <= 0 for x in [spot, strike, time_to_expiry, volatility]):
        return {"success": False, "error": "spot/strike/T/sigma 必须为正数"}
    if option_type.lower() not in ("call", "put"):
        return {"success": False, "error": "option_type 必须是 call 或 put"}

    S, K, T = spot, strike, time_to_expiry
    r, sigma = risk_free_rate, volatility

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == "call":
            price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
            delta = _norm_cdf(d1)
            rho   = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
        else:
            price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1
            rho   = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100

        gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
        vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100
        theta_per_year = (
            -S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
            - (r * K * math.exp(-r * T) * _norm_cdf(d2)
               if option_type.lower() == "call"
               else -r * K * math.exp(-r * T) * _norm_cdf(-d2))
        )
        theta = theta_per_year / 365

        moneyness = S / K
        if option_type.lower() == "call":
            status = ("实值 ITM" if moneyness > 1.02 else
                      "平值 ATM" if moneyness > 0.98 else "虚值 OTM")
        else:
            status = ("实值 ITM" if moneyness < 0.98 else
                      "平值 ATM" if moneyness < 1.02 else "虚值 OTM")

        return {
            "success": True,
            "option_type": option_type.lower(),
            "inputs": {
                "spot": S, "strike": K, "T_years": T,
                "risk_free": r, "sigma": sigma,
            },
            "theoretical_price": round(price, 4),
            "moneyness": round(moneyness, 4),
            "status": status,
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "vega":  round(vega, 4),
                "theta": round(theta, 4),
                "rho":   round(rho, 4),
            },
            "greeks_interpretation": {
                "delta": "标的价格 ±1 时期权价格变动",
                "gamma": "Delta 对标的价格的二阶敏感度",
                "vega":  "波动率 ±1% 时期权价格变动",
                "theta": "每日时间衰减（每过一天损失多少价值）",
                "rho":   "无风险利率 ±1% 时期权价格变动",
            },
            "formula": "Black-Scholes 1973，假设：欧式、无股息、几何布朗运动",
            "data_source": "理论模型计算",
            "disclaimer": "理论价格不等于市场价格，IV smile 与流动性会造成偏离",
        }

    except (ValueError, ZeroDivisionError) as e:
        return {"success": False, "error": f"BS 计算异常: {e}"}


# ─────────────────────────────────────────────
# Tool 5: 组合风险指标（Sharpe/Sortino/VaR/MaxDD）
# ─────────────────────────────────────────────

def risk_metrics(
    returns: list,
    risk_free_rate: float = 0.03,
    confidence: float = 0.95,
) -> dict:
    """
    根据收益率序列计算关键风险指标。

    returns: 周期收益率列表（小数，如 0.01 = 1%），按日频
    risk_free_rate: 年化无风险利率
    confidence: VaR 置信度
    """
    if not returns or len(returns) < 5:
        return {"success": False, "error": "收益率序列至少需要 5 个数据点"}

    if not all(isinstance(r, (int, float)) for r in returns):
        return {"success": False, "error": "收益率必须为数字"}

    try:
        n = len(returns)
        mean_r = sum(returns) / n
        std_r = statistics.stdev(returns) if n > 1 else 0

        # 年化（假设日频，252 交易日）
        ann_return = mean_r * 252
        ann_vol = std_r * math.sqrt(252)

        # Sharpe
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Sortino（下行风险）
        downside = [r for r in returns if r < 0]
        downside_std = statistics.stdev(downside) if len(downside) > 1 else 0
        downside_ann = downside_std * math.sqrt(252)
        sortino = (ann_return - risk_free_rate) / downside_ann if downside_ann > 0 else 0

        # Max Drawdown
        cumulative = [1.0]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r))
        peak = cumulative[0]
        max_dd = 0
        for v in cumulative:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)

        # 历史 VaR（5% 分位）
        sorted_r = sorted(returns)
        var_idx = int((1 - confidence) * n)
        var = sorted_r[var_idx] if var_idx < n else sorted_r[0]

        # CVaR（VaR 之下的平均损失）
        tail = sorted_r[:max(var_idx, 1)]
        cvar = sum(tail) / len(tail)

        # 胜率
        win_rate = sum(1 for r in returns if r > 0) / n

        # Calmar 比率
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return {
            "success": True,
            "sample_size": n,
            "frequency_assumption": "日频（252 交易日/年）",
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown": round(max_dd, 4),
            "var_95": round(var, 4),
            "cvar_95": round(cvar, 4),
            "win_rate": round(win_rate, 3),
            "interpretation": {
                "sharpe":   "夏普 > 1 优秀, > 2 卓越, < 0 跑输无风险",
                "sortino":  "只惩罚下行波动，更贴近投资者真实感受",
                "calmar":   "年化收益 / 最大回撤，衡量风险调整后的收益",
                "max_dd":   "历史最大回撤，反映极端情况承压能力",
                "var_95":   f"95% 置信下单期最大可能损失",
                "cvar_95":  "VaR 之下的平均损失（尾部风险）",
            },
            "formula": {
                "sharpe":  "(年化收益 - 无风险利率) / 年化波动率",
                "sortino": "(年化收益 - 无风险利率) / 下行波动率",
                "var":     "历史模拟法分位数",
            },
            "disclaimer": "历史数据不代表未来表现；样本量小时统计显著性弱",
        }

    except Exception as e:
        return {"success": False, "error": f"风险指标计算异常: {e}"}


# ─────────────────────────────────────────────
# Tool 6: 财经新闻定向搜索（Yahoo Finance Search API）
# ─────────────────────────────────────────────

# Yahoo Finance 对默认 Python UA 返回 403，需伪装浏览器；UA 越简短越不易被限流
_YAHOO_UA = "Mozilla/5.0"


def _extract_primary_term(query: str) -> str:
    """
    从查询中提取最适合 Yahoo Search 的主词。
    优先级：股票代码（带交易所后缀）> 全大写 ticker > 第一个非停用词。
    Yahoo Search 对长 query 会退化为全站热点，必须收敛到关键词。
    """
    tokens = query.strip().split()
    # 带后缀的交易所代码，如 600519.SS / 0700.HK
    for t in tokens:
        if re.search(r'\.(SS|SZ|HK|US|TO|L)$', t, re.IGNORECASE):
            return t
    # 全大写 1-5 字母的疑似 ticker
    for t in tokens:
        if re.match(r'^[A-Z]{1,5}$', t):
            return t
    # 兜底：第一个 token
    return tokens[0] if tokens else query


def _yahoo_fetch(url: str, retries: int = 1) -> dict:
    """带一次 429 重试的 GET 封装"""
    import time
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": _YAHOO_UA, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt < retries:
                time.sleep(2)   # 限流：等 2 秒重试
                continue
            raise
    raise last_err  # 不会到这


def _news_yahoo(primary: str, max_results: int = 5):
    """Yahoo Finance Search API（quotes 元信息 + news 列表）"""
    try:
        encoded = urllib.parse.quote(primary)
        url = (f"https://query1.finance.yahoo.com/v1/finance/search?"
               f"q={encoded}&newsCount={max_results}&quotesCount=5")
        data = _yahoo_fetch(url, retries=1)
        quotes = [{
            "symbol":   q.get("symbol"),
            "name":     q.get("shortname") or q.get("longname"),
            "exchange": q.get("exchange"),
            "type":     q.get("quoteType"),
        } for q in data.get("quotes", [])[:5]]
        news = []
        for n in data.get("news", [])[:max_results]:
            ts = n.get("providerPublishTime", 0)
            news.append({
                "title": n.get("title", ""),
                "publisher": n.get("publisher", "Yahoo"),
                "link": n.get("link", ""),
                "published_at": (datetime.datetime.fromtimestamp(ts)
                                 .strftime("%Y-%m-%d %H:%M") if ts else ""),
                "related_tickers": n.get("relatedTickers", []),
                "source_api": "yahoo",
            })
        return quotes, news
    except Exception:
        return [], []


def market_news_search(query: str, max_results: int = 5) -> dict:
    """
    多源财经新闻聚合搜索。

    源覆盖：
      - Yahoo Finance      （全球，quotes 元信息 + 美股新闻最佳）
      - 东方财富             （A 股新闻覆盖最佳）
      - Finnhub             （需 FINNHUB_API_KEY env）
      - AKShare             （需 pip install akshare，A 股深度数据）

    自动按市场选源：A 股优先东财/AKShare，美股优先 Yahoo/Finnhub。
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空"}

    max_results = max(1, min(int(max_results), 10))
    primary = _extract_primary_term(query)
    market = _detect_market(primary)

    all_news = []
    quotes = []
    sources_tried = []

    # ── 1. Yahoo（quotes 元信息 + 全球新闻）──
    y_quotes, y_news = _news_yahoo(primary, max_results)
    quotes.extend(y_quotes)
    all_news.extend(y_news)
    sources_tried.append({"source": "Yahoo Finance", "news_count": len(y_news)})

    # ── 2. 东方财富（A 股 + 中文新闻）──
    if market in ("A", "HK", "unknown"):
        em_news = _news_eastmoney(primary, max_results)
        all_news.extend(em_news)
        sources_tried.append({"source": "东方财富", "news_count": len(em_news)})

    # ── 3. Finnhub（可选，全球公司深度新闻）──
    if FINNHUB_API_KEY and market in ("US", "HK"):
        fh_news = _news_finnhub(primary, max_results)
        all_news.extend(fh_news)
        sources_tried.append({"source": "Finnhub",
                              "news_count": len(fh_news)})

    # ── 4. AKShare（可选）──
    # 触发条件：market='A'（A股代码）或 query 包含中文（主题/概念词）
    has_chinese = bool(re.search(r'[一-鿿]', primary))
    if _AKSHARE_AVAILABLE and (market == "A" or has_chinese):
        ak_news = _news_akshare(primary, max_results)
        all_news.extend(ak_news)
        sources_tried.append({"source": "AKShare",
                              "news_count": len(ak_news)})

    # ── 去重（按标题）──
    seen_titles = set()
    deduped = []
    for n in all_news:
        t = (n.get("title") or "").strip()
        if t and t not in seen_titles:
            seen_titles.add(t)
            deduped.append(n)

    # ── 相关性评分：标题/ticker 匹配 + 市场匹配 + 时间次序 ──
    def _relevance(n):
        score = 0
        title = (n.get("title") or "").lower()
        tickers = [t.upper() for t in n.get("related_tickers", [])]
        if primary.upper() in tickers:
            score += 100
        if primary.lower() in title:
            score += 50
        src = n.get("source_api")
        if market == "A" and src in ("eastmoney", "akshare"):
            score += 30
        elif market == "US" and src in ("yahoo", "finnhub"):
            score += 30
        elif market == "HK" and src in ("yahoo", "eastmoney"):
            score += 20
        return score

    deduped.sort(key=lambda x: (_relevance(x), x.get("published_at", "")),
                 reverse=True)
    final_news = deduped[:max_results]

    if not quotes and not final_news:
        return {
            "success": False,
            "query": query,
            "search_term": primary,
            "market_detected": market,
            "sources_tried": sources_tried,
            "error": f"所有数据源都未找到 '{query}' 相关资讯",
            "hint": "确认代码格式：AAPL / 600519 / 0700.HK；中文公司名也可",
        }

    return {
        "success":         True,
        "original_query":  query,
        "search_term":     primary,
        "term_simplified": primary != query.strip(),
        "market_detected": market,
        "quotes":          quotes,
        "news":            final_news,
        "quotes_count":    len(quotes),
        "news_count":      len(final_news),
        "sources_tried":   sources_tried,
        "as_of":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "disclaimer":      "信息源为公开财经数据，不构成投资建议",
    }


# ─────────────────────────────────────────────
# Tool 7: 交易日历（简易版，未考虑节假日）
# ─────────────────────────────────────────────

def trading_calendar(action: str, date: str = None, date2: str = None) -> dict:
    """
    简易交易日历。
    action: today / parse / trading_days_between
    ⚠️ 工作日近似交易日，未排除法定节假日
    """
    tz = ZoneInfo("Asia/Shanghai")

    def parse(s):
        if not s or s.lower() in ("today", "now", "今天"):
            return datetime.datetime.now(tz)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.datetime.strptime(s, fmt).replace(tzinfo=tz)
            except ValueError:
                continue
        raise ValueError(f"日期格式无法解析: {s}")

    try:
        if action == "today":
            now = datetime.datetime.now(tz)
            is_trading = now.weekday() < 5
            return {
                "success": True,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "is_trading_day": is_trading,
                "timezone": "Asia/Shanghai (CST)",
                "note": "未考虑法定节假日",
            }

        elif action == "trading_days_between":
            d1, d2 = parse(date), parse(date2)
            days = (d2.date() - d1.date()).days
            trading_days = sum(
                1 for i in range(abs(days))
                for d in [d1.date() + datetime.timedelta(days=i * (1 if days > 0 else -1))]
                if d.weekday() < 5
            )
            return {
                "success": True,
                "from": d1.strftime("%Y-%m-%d"),
                "to": d2.strftime("%Y-%m-%d"),
                "calendar_days": days,
                "trading_days_approx": trading_days,
                "note": "近似值，未排除春节/国庆等法定节假日",
            }

        else:
            return {"success": False, "error": f"未知 action: {action}",
                    "valid": ["today", "trading_days_between"]}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════
# 第二部分：工具注册
# ════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Tool 8: search_research_docs — RAG 研报检索
# ─────────────────────────────────────────────
# 把 rag/ 模块的检索函数包成工具，惰性 import 避免无 PDF 时启动失败
try:
    from rag.retriever import search_research_docs as _rag_search
    _RAG_AVAILABLE = True
except ImportError as _e:
    _RAG_AVAILABLE = False
    _rag_import_err = str(_e)


def search_research_docs(query: str, top_k: int = 3,
                          doc_filter: str = None) -> dict:
    """在已索引的研报/财报 PDF 库中检索相关段落"""
    if not _RAG_AVAILABLE:
        return {"success": False,
                "error": f"RAG 模块加载失败: {_rag_import_err}",
                "hint": "确认安装 chromadb / pymupdf / sentence-transformers"}
    return _rag_search(query=query, top_k=top_k, doc_filter=doc_filter)


TOOL_REGISTRY = {
    "market_quote":         market_quote,
    "factor_score":         factor_score,
    "technical_indicator":  technical_indicator,
    "black_scholes":        black_scholes,
    "risk_metrics":         risk_metrics,
    "market_news_search":   market_news_search,
    "trading_calendar":     trading_calendar,
    "search_research_docs": search_research_docs,
}

TOOL_SCHEMAS = [
    {
        "name": "market_quote",
        "description": """实时行情快照（多源：新浪财经 → MOCK 兜底）。
返回：现价、涨跌幅、开高低、成交量/额、数据源标注。
支持代码：
  - A 股: 600519（自动转 sh600519）/ 000858（自动 sz）
  - 港股: 0700 / 00700
  - 美股: AAPL / NVDA / TSLA
data_source 字段会标明实际来源，看到 [MOCK DATA] 时必须警示用户。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
            },
            "required": ["symbol"],
        }
    },
    {
        "name": "factor_score",
        "description": """多因子量化打分（Value/Growth/Momentum/Quality/Technical）。
style 可选：balanced(均衡), value(价值偏好), growth(成长偏好), momentum(动量偏好)。
返回 0-100 综合得分 + 评级（Buy/Hold/Sell）。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "style": {
                    "type": "string",
                    "enum": ["balanced", "value", "growth", "momentum"],
                    "default": "balanced",
                },
            },
            "required": ["symbol"],
        }
    },
    {
        "name": "technical_indicator",
        "description": """技术指标计算。
支持 RSI / MACD / Bollinger Bands / SMA。
输入：收盘价数组（时间正序，旧→新）+ 指标名。
RSI 默认周期 14，Bollinger 默认周期 20。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "收盘价序列",
                },
                "indicator": {
                    "type": "string",
                    "enum": ["rsi", "macd", "bollinger", "sma"],
                },
                "period": {"type": "integer", "default": 14},
            },
            "required": ["prices", "indicator"],
        }
    },
    {
        "name": "black_scholes",
        "description": """欧式期权 Black-Scholes 定价与 Greeks 计算。
返回理论价格 + Delta/Gamma/Vega/Theta/Rho。
适用：A50/沪深300/SPY/QQQ 等指数期权、主流个股期权。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "spot":           {"type": "number", "description": "标的现价"},
                "strike":         {"type": "number", "description": "行权价"},
                "time_to_expiry": {"type": "number", "description": "到期时间（年）"},
                "risk_free_rate": {"type": "number", "description": "无风险利率（小数）"},
                "volatility":     {"type": "number", "description": "年化波动率（小数）"},
                "option_type":    {"type": "string", "enum": ["call", "put"]},
            },
            "required": ["spot", "strike", "time_to_expiry",
                         "risk_free_rate", "volatility", "option_type"],
        }
    },
    {
        "name": "risk_metrics",
        "description": """组合/策略风险指标计算。
输入日频收益率序列（小数），输出：
夏普比率、索提诺、Calmar、最大回撤、VaR/CVaR、胜率。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "收益率列表，如 [0.01, -0.005, 0.02, ...]",
                },
                "risk_free_rate": {"type": "number", "default": 0.03},
                "confidence":     {"type": "number", "default": 0.95},
            },
            "required": ["returns"],
        }
    },
    {
        "name": "market_news_search",
        "description": """多源财经新闻聚合搜索。
数据源（自动按市场选源 + 相关性排序）：
  - Yahoo Finance：全球，含股票元信息
  - 东方财富：A 股 / 港股新闻覆盖最佳
  - Finnhub：需 FINNHUB_API_KEY env 时启用
  - AKShare：需 pip install akshare 时启用，A 股深度
⚠️ query 简短最佳，传入「股票代码」或「公司名」。
  ✅ 好：'AAPL'、'600519'、'Tesla'
  ❌ 坏：'AAPL Apple options volatility 2025'
返回：quotes（元信息）+ news（按相关性+时间排序）+ sources_tried（数据源透明度）""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
    },
    {
        "name": "trading_calendar",
        "description": """交易日历查询（北京时区）。
action: today (今天信息) / trading_days_between (两日期间交易日数).
⚠️ 仅按周一到周五近似，未排除法定节假日。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["today", "trading_days_between"]},
                "date":   {"type": "string"},
                "date2":  {"type": "string"},
            },
            "required": ["action"],
        }
    },
    {
        "name": "search_research_docs",
        "description": """在内部研报/财报 PDF 库中检索相关段落（RAG）。
适用场景：
  - 用户问「中信对茅台的最新观点」「某公司财报里写了什么」
  - 引用具体研报或财报内容时
  - 解释行业景气度、机构观点等需要权威来源的问题
不适用：实时行情/价格（用 market_quote）、新闻动态（用 market_news_search）

返回每条结果含：
  - doc_name（文件名）/ page（页码）/ content（段落原文）
  - citation（标准引用格式，如「《中信研报_茅台》第 12 页」）
  - relevance（0-1 相关性分数）

写报告时务必引用 citation 字段。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言查询，可中英文",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回段落数，默认 3，最多 10",
                    "default": 3,
                },
                "doc_filter": {
                    "type": "string",
                    "description": "可选，限定在指定文件名内搜索（精确匹配）",
                },
            },
            "required": ["query"],
        }
    },
]


# ════════════════════════════════════════════════════════════
# 第三部分：QuantAgent System Prompt（基于 system_prompt.txt）
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是 QuantAgent v1.0，精通量化金融与系统化交易的顶级 AI 助手。

【角色定位】
- 量化核心：多因子模型、统计套利、动量/反转策略
- 衍生品定价：Black-Scholes、Greeks 分析
- 风险管理：VaR/CVaR/Sharpe/MaxDD
- 数据严谨：所有结论必须注明数据来源与置信度

【可用工具】
1. market_quote        — 行情快照（MOCK 数据，明确标注）
2. factor_score        — 多因子综合打分（0-100）
3. technical_indicator — RSI/MACD/布林带/SMA
4. black_scholes       — 欧式期权定价与 Greeks
5. risk_metrics        — 组合风险指标（Sharpe/Sortino/VaR/MaxDD）
6. market_news_search  — 财经新闻定向搜索
7. trading_calendar    — 交易日历
8. search_research_docs — 研报/财报 PDF 库检索（RAG），有引用价值的观点必查

【强制工作原则】
✅ 数据驱动：任何数值结论必须来自工具调用，禁止凭记忆估算
✅ 多工具协同：复杂任务并行调用多个工具，再综合分析
✅ 标注数据来源：MOCK 数据必须明示 [MOCK DATA]，实时数据注明截止时间
✅ 置信度标签：🟢高置信 / 🟡中置信 / 🔴低置信
✅ 双语术语：专业词汇中英对照（如「夏普比率 Sharpe Ratio」）

【禁止行为】
❌ 不使用"一定涨/跌"、"保证收益"等绝对化表述
❌ 不在数据缺失时编造具体数值，标注 [DATA UNAVAILABLE]
❌ 不省略风险提示
❌ 不推荐超出常识的杠杆比例

【输出格式（报告类问题）】
使用 Markdown 结构：

# 📊 [报告标题]
**生成时间**：YYYY-MM-DD HH:MM (CST) | **风险等级**：🟢/🟡/🔴

## 一、核心结论 / Key Takeaways
- 3-5 条要点，先中文后英文

## 二、量化分析 / Quantitative Analysis
[因子得分、指标计算、数据引用]

## 三、风险提示 / Risk Warnings
[必须包含]

## 四、操作参考 / Actionable Reference
[方向性建议 + 计算依据]

---
⚠️ **免责声明**：本报告基于量化模型输出，仅供研究参考，不构成投资建议。
历史收益不代表未来表现，量化模型存在失效风险。投资有风险，入市需谨慎。
"""


# ════════════════════════════════════════════════════════════
# 第四部分：Agent 主循环
# ════════════════════════════════════════════════════════════

def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"success": False,
                "error": f"工具 '{tool_name}' 不存在",
                "available_tools": list(TOOL_REGISTRY.keys())}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {e}"}
    except Exception as e:
        return {"success": False, "error": f"工具执行异常: {e}"}


def _to_openai_tools(anthropic_schemas: list) -> list:
    """把 Anthropic 风格 schema 转成 OpenAI/DeepSeek 的 function calling 格式"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_schemas
    ]


# 模块加载时转换一次，避免每轮迭代重算
_OPENAI_TOOLS = _to_openai_tools(TOOL_SCHEMAS)


def _truncate_observation(obs: dict, obs_str: str, max_len: int = 3000) -> str:
    """工具返回过大时按字段截断"""
    if len(obs_str) <= max_len:
        return obs_str
    for k in ("results", "news", "quotes"):
        if obs.get(k):
            obs[k] = obs[k][:3]
            obs["truncated"] = True
    return json.dumps(obs, ensure_ascii=False, default=str)


def stream_quant_agent(messages: list, max_iterations: int = 15):
    """
    生成器版 Agent 主循环（DeepSeek/OpenAI 协议）。
    yield 事件协议保持不变，与前端解耦。
    """
    # 首次调用时注入 system message
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    for iteration in range(1, max_iterations + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=_OPENAI_TOOLS,
                max_tokens=4096,
            )
        except Exception as e:
            yield {"type": "error", "error": f"API 调用失败: {e}"}
            return

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        # ── 模型给出最终答案 ──
        if finish == "stop":
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            yield {"type": "final", "text": text, "iterations": iteration}
            return

        # ── 工具调用 ──
        if finish == "tool_calls" and msg.tool_calls:
            # 调工具前的思考（可能为空）
            if msg.content and msg.content.strip():
                yield {"type": "thought",
                       "text": msg.content.strip(),
                       "iteration": iteration}

            # 把模型这一轮的完整回复入历史
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 逐个工具执行
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = None

                yield {
                    "type": "tool_call",
                    "name": tc.function.name,
                    "input": args if isinstance(args, dict) else {},
                    "id": tc.id,
                }

                if not isinstance(args, dict):
                    obs = {"success": False,
                           "error": "工具参数不是有效 JSON 对象",
                           "raw_arguments": tc.function.arguments[:200]}
                else:
                    obs = dispatch_tool(tc.function.name, args)

                obs_str = json.dumps(obs, ensure_ascii=False, default=str)
                obs_str = _truncate_observation(obs, obs_str)

                is_error = not obs.get("success", True)
                yield {
                    "type": "tool_result",
                    "name": tc.function.name,
                    "result": obs,
                    "is_error": is_error,
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": obs_str,
                })
            continue

        # 兜底：异常 finish_reason（length / content_filter 等）
        yield {"type": "error",
               "error": f"非预期 finish_reason: {finish}"}
        return

    yield {"type": "error", "error": f"超过最大迭代次数 ({max_iterations})"}


def run_quant_agent(user_query: str, max_iterations: int = 15, verbose: bool = True) -> str:
    """CLI 版主循环 —— 内部复用 stream_quant_agent，统一行为"""

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*70}")
    log(f"📊 QuantAgent({DEEPSEEK_MODEL}) 收到请求：{user_query}")
    log(f"{'='*70}")

    messages = [{"role": "user", "content": user_query}]
    final_text = ""

    for event in stream_quant_agent(messages, max_iterations=max_iterations):
        et = event["type"]
        if et == "thought":
            log(f"\n💭 [迭代 {event['iteration']}]: {event['text']}")
        elif et == "tool_call":
            log(f"\n⚡ Tool: {event['name']}")
            log(f"   Input: {json.dumps(event['input'], ensure_ascii=False)}")
        elif et == "tool_result":
            obs_str = json.dumps(event['result'], ensure_ascii=False, default=str)
            tag = "❌" if event['is_error'] else "✅"
            log(f"   {tag} Obs: {obs_str[:250]}{'...' if len(obs_str) > 250 else ''}")
        elif et == "final":
            final_text = event["text"]
            log(f"\n{'='*70}")
            log(f"✅ 最终报告：\n{final_text}")
            log(f"{'='*70}\n📊 迭代轮数：{event['iterations']}")
        elif et == "error":
            log(f"\n⛔ ERROR: {event['error']}")
            return f"错误: {event['error']}"

    return final_text


# ════════════════════════════════════════════════════════════
# 第五部分：测试用例
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█"*70)
    print("  测试 1：个股量化扫描（行情 + 因子）")
    print("█"*70)
    run_quant_agent(
        "帮我分析下英伟达 NVDA，要现价行情和多因子打分，"
        "用成长股风格的权重，给个完整的量化报告"
    )

    print("\n\n" + "█"*70)
    print("  测试 2：技术指标分析")
    print("█"*70)
    run_quant_agent(
        "我有这一段日收盘价：[100, 102, 101, 105, 108, 107, 110, 112, "
        "115, 113, 116, 120, 118, 122, 125, 123, 127, 130, 128, 132, "
        "135, 138, 136, 140, 142, 145, 143, 147, 150]，"
        "帮我算 RSI(14) 和布林带，给出技术面研判"
    )

    print("\n\n" + "█"*70)
    print("  测试 3：期权定价")
    print("█"*70)
    run_quant_agent(
        "苹果股票 213 美元，我想看一下行权价 220、3个月到期的看涨期权，"
        "无风险利率 4.5%，隐含波动率 28%，给我理论价格和全套 Greeks，"
        "并解读这个期权的风险特征"
    )

    print("\n\n" + "█"*70)
    print("  测试 4：组合风险评估")
    print("█"*70)
    run_quant_agent(
        "我策略最近 30 天的日收益率：[0.012, -0.008, 0.015, 0.003, -0.021, "
        "0.008, 0.011, -0.005, 0.018, -0.012, 0.007, 0.022, -0.015, 0.009, "
        "0.013, -0.018, 0.025, 0.004, -0.009, 0.016, 0.011, -0.007, 0.019, "
        "-0.013, 0.008, 0.014, -0.022, 0.017, 0.006, 0.020]，"
        "全套风险指标算一遍，给评价"
    )

    print("\n\n" + "█"*70)
    print("  测试 5：综合任务（多工具协同）")
    print("█"*70)
    run_quant_agent(
        "搜索一下英伟达最近有什么大新闻，"
        "结合 NVDA 当前行情和动量风格的因子打分，"
        "判断现在适不适合用看涨期权策略（标的=现价，行权价=现价+5%，"
        "3个月到期，假设波动率 35%，利率 4.5%），"
        "给出完整研报"
    )
