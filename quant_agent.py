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
import socket
import statistics
import datetime
import urllib.request
import urllib.parse
from zoneinfo import ZoneInfo

# 全局兜底：任何未显式设 timeout 的 socket 操作最多 15 秒
# 避免 AKShare 等三方库内部请求卡死拖累整个 Agent
socket.setdefaulttimeout(15)

# 自动加载项目 .env 文件（如果存在），免去手动 export
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # override=True：.env 优先级高于已有系统环境变量
    # 避免外部 shell 误设空值导致 .env 被无视
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass  # 没装 dotenv 时静默跳过，回退到系统环境变量

from openai import OpenAI
from cache import cache  # 统一缓存层（Redis 或内存）

# 服务端 markdown 渲染（让 QQ/微信浏览器零依赖渲染美观 HTML）
try:
    import markdown as _md
    _MD_AVAILABLE = True
except ImportError:
    _MD_AVAILABLE = False


def _normalize_chart_urls(text: str) -> str:
    """
    把 LLM 误生成的绝对 URL 改成相对路径。
    例：http://localhost:8001/static/charts/xxx.html → /static/charts/xxx.html
        https://任意域/static/charts/xxx.html → /static/charts/xxx.html
    避免老浏览器 / 跨设备访问时跳到错误地址。
    """
    if not text:
        return text
    # 匹配 http(s)://host[:port]/static/... → /static/...
    return re.sub(
        r'https?://[^/\s)]+(/static/[^)\s\]"\']+)',
        r'\1',
        text,
    )


def render_markdown_to_html(text: str) -> str:
    """把 markdown 文本转成 HTML，所有 <a> 自动加 target=_blank。失败回退到 <pre>。"""
    if not text:
        return ""
    # 先把 LLM 可能拼错的绝对 URL 规范化（兜底）
    text = _normalize_chart_urls(text)
    if not _MD_AVAILABLE:
        from html import escape
        return f'<pre style="white-space:pre-wrap;">{escape(text)}</pre>'
    try:
        html = _md.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])
        # 给所有 <a> 加 target=_blank（避免链接跳转覆盖聊天页）
        html = re.sub(r'<a (?![^>]*\btarget=)', '<a target="_blank" rel="noopener" ', html)
        return html
    except Exception:
        from html import escape
        return f'<pre style="white-space:pre-wrap;">{escape(text)}</pre>'

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

# yfinance：自动处理 Yahoo Cookie/Crumb 认证，比直接 HTTP 稳（pip install yfinance）
try:
    import yfinance as _yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _yf = None
    _YFINANCE_AVAILABLE = False


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


def _news_yahoo_rss(symbol: str, max_results: int = 5):
    """
    Yahoo Finance 个股 RSS（全球可用、无需 Key、按 ticker 精准命中）。
    适合美股/港股/欧股，A 股新闻覆盖薄。
    """
    try:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
               f"?s={urllib.parse.quote(symbol)}&region=US&lang=en-US")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_text = resp.read().decode("utf-8", errors="ignore")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        items = root.findall('.//item')
        news = []
        for it in items[:max_results]:
            title = (it.findtext('title') or '').strip()
            link = (it.findtext('link') or '').strip()
            pub = (it.findtext('pubDate') or '').strip()
            desc = (it.findtext('description') or '').strip()
            desc = re.sub(r'<[^>]+>', '', desc)[:200]
            if title:
                news.append({
                    "title": title,
                    "publisher": "Yahoo Finance",
                    "link": link,
                    "published_at": pub,
                    "summary": desc,
                    "source_api": "yahoo_rss",
                    "related_tickers": [symbol.upper()],
                })
        return news
    except Exception:
        return []


def _news_google_rss(query: str, max_results: int = 5):
    """
    Google News RSS 搜索（全球可用、无需 Key、覆盖最广）。
    适合任意关键词搜索，包括公司名/主题词。
    """
    try:
        q = urllib.parse.quote(f"{query} stock")
        url = (f"https://news.google.com/rss/search?q={q}"
               f"&hl=en-US&gl=US&ceid=US:en")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_text = resp.read().decode("utf-8", errors="ignore")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        ns = {"news": "http://news.google.com/"}
        items = root.findall('.//item')
        news = []
        for it in items[:max_results]:
            title = (it.findtext('title') or '').strip()
            link = (it.findtext('link') or '').strip()
            pub = (it.findtext('pubDate') or '').strip()
            # Google News 标题里常带 " - 媒体名" 后缀，提取出来当 publisher
            publisher = "Google News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                if len(parts) == 2 and len(parts[1]) < 40:
                    publisher = parts[1].strip()
                    title = parts[0].strip()
            if title:
                news.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published_at": pub,
                    "summary": "",
                    "source_api": "google_news_rss",
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


_QUOTE_CACHE_TTL = 30        # 盘中价格 30 秒精度足够


def market_quote(symbol: str) -> dict:
    """
    多源实时行情快照（30 秒缓存）。
    源优先级：新浪财经（实时）→ MOCK 兜底
    """
    symbol_raw = symbol.strip()
    # 命中缓存（Redis 或内存）
    cache_key = f"quant:quote:{symbol_raw.upper()}"
    cached = cache.get(cache_key)
    if cached:
        result = dict(cached)
        result["from_cache"] = True
        return result

    sources_tried = []

    # ── 1. 新浪财经实时行情（A/HK/US 通用）──
    sina = _quote_sina(symbol_raw)
    sources_tried.append({"source": "新浪财经",
                          "ok": sina is not None})
    if sina:
        result = {
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
        cache.set(cache_key, result, ttl=_QUOTE_CACHE_TTL)
        return result

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

_FACTOR_CACHE_TTL = 3600  # 因子数据缓存 1 小时


def _fetch_financial_df(s: str):
    """单独抽出来好并发调用"""
    try:
        return _ak.stock_financial_analysis_indicator(symbol=s, start_year="2024")
    except Exception:
        return None


def _fetch_kline_df(s: str, days: int = 120):
    """单独抽出来好并发调用"""
    try:
        end_d = datetime.date.today().strftime('%Y%m%d')
        start_d = (datetime.date.today()
                    - datetime.timedelta(days=days)).strftime('%Y%m%d')
        return _ak.stock_zh_a_hist(symbol=s, period='daily',
                                    start_date=start_d, end_date=end_d,
                                    adjust='qfq')
    except Exception:
        return None


def _compute_factors_akshare(symbol: str):
    """
    从 AKShare 拉真实财务/价格数据，并发 3 个接口后计算 5 因子。
    成功返回 dict（含 factors + raw），失败返回 None。
    """
    if not _AKSHARE_AVAILABLE:
        return None

    s = symbol.replace(".SS", "").replace(".SH", "").replace(".SZ", "")

    cache_key = f"quant:factor:{s}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # ── 并发拉 3 个数据源（核心优化）──
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_price = ex.submit(_quote_sina, s)
        f_fin   = ex.submit(_fetch_financial_df, s)
        f_k     = ex.submit(_fetch_kline_df, s, 120)

        # 各源 6 秒硬超时，慢源不拖累整体（之前 12s 太宽松）
        try:
            quote = f_price.result(timeout=6)
        except Exception:
            quote = None
        try:
            fin_df = f_fin.result(timeout=6)
        except Exception:
            fin_df = None
        try:
            k_df = f_k.result(timeout=6)
        except Exception:
            k_df = None

    # ── 拿到原始数据后串行算因子（CPU 时间忽略不计）──
    factors = {}
    raw = {}

    price = quote.get("price") if quote else None
    if price:
        raw["price"] = price

    # 财务因子
    if fin_df is not None and len(fin_df) > 0:
        try:
            r = fin_df.iloc[0]
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

            if price and eps > 0:
                pe = price / (eps * 4)
                raw["pe_estimated"] = round(pe, 2)
                factors["value"] = max(0, min(100, 100 - pe * 2))
            if price and bps > 0:
                raw["pb"] = round(price / bps, 2)

            avg_g = (rev_g + prof_g) / 2
            factors["growth"] = max(0, min(100, 50 + avg_g * 1.5))
            factors["quality"] = max(0, min(100, roe * 4))
        except Exception:
            pass

    # 动量 + 技术因子
    if k_df is not None and len(k_df) >= 20:
        try:
            cur = float(k_df['收盘'].iloc[-1])
            ret_3m = (cur / float(k_df['收盘'].iloc[0]) - 1) * 100
            raw["return_3m_pct"] = round(ret_3m, 2)
            factors["momentum"] = max(0, min(100, 50 + ret_3m * 1.67))

            ma60 = (float(k_df['收盘'].tail(60).mean())
                    if len(k_df) >= 60
                    else float(k_df['收盘'].mean()))
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
    cache.set(cache_key, result, ttl=_FACTOR_CACHE_TTL)
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

    # 概念词/中文（如「芯片ETF」）→ 直接引导
    is_concept = bool(re.search(r'[一-鿿]', symbol)) or \
                 not re.match(r'^[A-Z0-9.]{1,10}$', symbol)
    if is_concept:
        return {
            "success": False,
            "error_type": "not_a_ticker",
            "error": f"'{symbol}' 不是有效股票代码，factor_score 只接受具体 ticker",
            "hint": "若要分析行业/概念：先用 market_news_search 找到相关公司，"
                    "再对每个 ticker 单独调用 factor_score",
        }

    # ── ETF / 基金 短路：本工具基于股票财务报表，对 ETF 无效 ──
    # A 股 ETF 代码：5xxxxx（上交所） / 159xxx / 15xxxx / 16xxxx（深交所）
    is_etf = bool(re.match(r'^(5\d{5}|15\d{4}|16\d{4})$', symbol))
    if is_etf:
        return {
            "success": False,
            "error_type": "not_applicable_to_etf",
            "error": f"'{symbol}' 是 ETF/基金，无个股财务报表，factor_score 不适用",
            "hint": "ETF 分析应直接走技术面：先 historical_prices(symbol, days=60)，"
                    "再 technical_indicator(close, 'rsi'/'macd'/'bollinger') 即可。"
                    "若想看成份股基本面，可对成份股逐个 factor_score。",
            "recommended_tools": ["historical_prices", "technical_indicator",
                                   "market_news_search"],
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
                        "error_type": "insufficient_data",
                        "error": f"RSI({period}) 至少需要 {period+1} 个价格点，当前 {len(prices)}",
                        "hint": f"先调用 historical_prices(symbol, days={period*2}) 拿数据"}
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
                        "error_type": "insufficient_data",
                        "error": f"MACD 至少需要 26 个价格点，当前只有 {len(prices)} 个",
                        "hint": "先调用 historical_prices(symbol, days=40) 拿到 close 数组再传入"}

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
                        "error_type": "insufficient_data",
                        "error": f"布林带需要 {period} 个价格点，当前 {len(prices)}",
                        "hint": f"先调 historical_prices(symbol, days={period+5})"}
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
                        "error_type": "insufficient_data",
                        "error": f"SMA({period}) 需要 {period} 个数据点，当前 {len(prices)}",
                        "hint": f"先调 historical_prices(symbol, days={period+5})"}
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
    """
    带重试的 Yahoo Finance GET。
    用浏览器级 headers 降低 403/429 概率。
    """
    import time
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com",
        "Connection": "keep-alive",
    }
    last_err = None
    # 失败时切换 query1/query2 重试一次（403 时常常切换子域名就好）
    urls_to_try = [url]
    if "query1.finance.yahoo.com" in url:
        urls_to_try.append(url.replace("query1.", "query2."))

    for attempt_url in urls_to_try:
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(attempt_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429 and attempt < retries:
                    time.sleep(2)
                    continue
                # 403/404 直接换下一个 URL
                break
            except Exception as e:
                last_err = e
                break
    raise last_err


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

    # ── 中文意图检测：基于「原始 query」而不是收敛后的 primary ──
    has_chinese_query = bool(re.search(r'[一-鿿]', query))
    # 提取 query 中的中文关键词喂给国内源（如「ETF 市场 2025」→「ETF 市场」）
    chinese_parts = re.findall(r'[一-鿿]+', query)
    chinese_term = ' '.join(chinese_parts) if chinese_parts else None
    # 国内源用中文关键词，否则回退到 primary（ticker）
    domestic_term = chinese_term if chinese_term else primary

    # ── 多源并发请求 ──
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        # 1. Yahoo Search API（quotes 元信息 + 通用新闻）
        tasks["Yahoo Finance"] = ex.submit(_news_yahoo, primary, max_results)

        # 2. Yahoo Finance per-ticker RSS（无 key，美股最佳，全球可用）
        if market in ("US", "HK"):
            tasks["Yahoo RSS"] = ex.submit(_news_yahoo_rss, primary, max_results)

        # 3. Google News RSS（全球免费，任意关键词都好用）
        if market in ("US", "HK", "unknown"):
            tasks["Google News"] = ex.submit(_news_google_rss, primary, max_results)

        # 4. 东方财富（A 股 / 港股 / 中文 query）—— 不再受 market 局限
        if market in ("A", "HK", "unknown") or has_chinese_query:
            tasks["东方财富"] = ex.submit(_news_eastmoney, domestic_term, max_results)

        # 5. Finnhub（需 key，美股/港股深度）
        if FINNHUB_API_KEY and market in ("US", "HK"):
            tasks["Finnhub"] = ex.submit(_news_finnhub, primary, max_results)

        # 6. AKShare（A 股 OR 中文 query，喂中文关键词）
        if _AKSHARE_AVAILABLE and (market == "A" or has_chinese_query):
            tasks["AKShare"] = ex.submit(_news_akshare, domestic_term, max_results)

        # 收集结果（带超时保护，单源不超过 10 秒）
        for name, fut in tasks.items():
            try:
                result = fut.result(timeout=10)
            except Exception as e:
                sources_tried.append({"source": name, "news_count": 0,
                                       "error": str(e)[:80]})
                continue

            if name == "Yahoo Finance":
                # Yahoo 返回 (quotes, news) 二元组
                y_quotes, y_news = result
                quotes.extend(y_quotes)
                all_news.extend(y_news)
                sources_tried.append({"source": name,
                                       "news_count": len(y_news)})
            else:
                all_news.extend(result)
                sources_tried.append({"source": name,
                                       "news_count": len(result)})

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
        # A 股优先国内源
        if market == "A" and src in ("eastmoney", "akshare"):
            score += 30
        # 美股优先 ticker 精准源（Yahoo RSS / Finnhub）> 通用新闻 > Yahoo Search
        elif market == "US":
            if src in ("yahoo_rss", "finnhub"):
                score += 40
            elif src == "google_news_rss":
                score += 30
            elif src == "yahoo":
                score += 10
        # 港股两套都加权
        elif market == "HK":
            if src in ("yahoo_rss", "eastmoney"):
                score += 30
            elif src in ("google_news_rss", "yahoo"):
                score += 15
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


# ─────────────────────────────────────────────
# Tool 9: correlation_matrix — 多资产相关性
# ─────────────────────────────────────────────

def correlation_matrix(returns_data: dict) -> dict:
    """
    计算多资产收益率相关系数矩阵。

    Args:
        returns_data: 资产 → 收益率序列。
            {"AAPL": [0.01, -0.02, ...], "MSFT": [...], ...}
            所有序列长度必须一致。

    Returns:
        相关系数矩阵 + 高相关性 pair 提醒
    """
    if not returns_data or not isinstance(returns_data, dict):
        return {"success": False, "error": "returns_data 必须为 {asset: [returns]} 字典"}

    assets = list(returns_data.keys())
    if len(assets) < 2:
        return {"success": False, "error": "至少需要 2 个资产"}

    series = [returns_data[a] for a in assets]
    n = len(series[0])
    if not all(len(s) == n for s in series):
        return {"success": False,
                "error": "所有收益率序列长度必须一致",
                "lengths": {a: len(returns_data[a]) for a in assets}}
    if n < 5:
        return {"success": False, "error": "每个序列至少需要 5 个数据点"}

    def _corr(x, y):
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)
        vx = sum((xi - mx) ** 2 for xi in x) / len(x)
        vy = sum((yi - my) ** 2 for yi in y) / len(y)
        denom = math.sqrt(vx * vy)
        return cov / denom if denom > 0 else 0

    matrix = {}
    high_pairs = []
    for i, a1 in enumerate(assets):
        matrix[a1] = {}
        for j, a2 in enumerate(assets):
            if i == j:
                matrix[a1][a2] = 1.0
            elif j < i:
                matrix[a1][a2] = matrix[a2][a1]  # 对称
            else:
                c = round(_corr(series[i], series[j]), 4)
                matrix[a1][a2] = c
                if abs(c) >= 0.7:
                    high_pairs.append({"pair": f"{a1}-{a2}",
                                       "correlation": c,
                                       "interpretation": "高度同向" if c > 0 else "高度反向"})

    return {
        "success": True,
        "assets": assets,
        "sample_size": n,
        "matrix": matrix,
        "high_correlation_pairs": high_pairs,
        "interpretation": {
            ">  0.7": "高正相关（分散效果差）",
            "0.3-0.7": "中正相关",
            "-0.3-0.3": "低相关（适合做组合）",
            "< -0.3": "负相关（天然对冲）",
        },
    }


# ─────────────────────────────────────────────
# Tool 10: portfolio_optimizer — 组合权重优化
# ─────────────────────────────────────────────

def portfolio_optimizer(
    returns_data: dict,
    method: str = "min_variance",
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> dict:
    """
    多资产组合权重优化。

    method:
      - equal_weight:  等权重（基线）
      - inverse_vol:   反向波动率（简易风险平价）
      - min_variance:  最小方差（需 numpy 求协方差逆）
      - max_sharpe:    最大化夏普（数值近似）
    """
    if not returns_data or not isinstance(returns_data, dict):
        return {"success": False, "error": "returns_data 必须为字典"}

    assets = list(returns_data.keys())
    n = len(assets)
    if n < 2:
        return {"success": False, "error": "至少需要 2 个资产"}

    series = [returns_data[a] for a in assets]
    m = len(series[0])
    if not all(len(s) == m for s in series):
        return {"success": False, "error": "收益率序列长度必须一致"}
    if m < 10:
        return {"success": False, "error": "每个序列至少需要 10 个数据点"}

    # 基础统计：均值 + 标准差
    means = [sum(s) / m for s in series]
    stds  = [statistics.stdev(s) for s in series]

    weights = None
    method_used = method.lower().strip()

    if method_used == "equal_weight":
        weights = [1.0 / n] * n

    elif method_used == "inverse_vol":
        # 简易风险平价：反波动率加权
        inv = [1.0 / s if s > 0 else 0 for s in stds]
        total = sum(inv)
        weights = [w / total for w in inv] if total > 0 else [1.0 / n] * n

    elif method_used in ("min_variance", "max_sharpe"):
        try:
            import numpy as np
            R = np.array(series).T  # m x n
            cov = np.cov(R.T, ddof=1)
            ones = np.ones(n)

            if method_used == "min_variance":
                # w = Σ^-1 1 / (1' Σ^-1 1)
                inv_cov = np.linalg.pinv(cov)
                w = inv_cov @ ones
                w = w / w.sum()
            else:
                # max_sharpe: 解析解 w ∝ Σ^-1 (μ - rf/N)
                mu = np.array(means) * periods_per_year
                rf = risk_free_rate
                inv_cov = np.linalg.pinv(cov)
                excess = mu - rf / periods_per_year
                w = inv_cov @ excess
                if w.sum() != 0:
                    w = w / w.sum()
                # 不允许做空：负权重置零再归一化
                w = np.maximum(w, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = np.ones(n) / n

            weights = [round(float(x), 4) for x in w]
        except ImportError:
            return {"success": False,
                    "error": "min_variance/max_sharpe 需要 numpy",
                    "hint": "pip install numpy 或改用 equal_weight / inverse_vol"}
        except Exception as e:
            return {"success": False, "error": f"优化失败: {e}",
                    "hint": "可能数据不足或协方差矩阵奇异，改用 inverse_vol"}
    else:
        return {"success": False,
                "error": f"未知 method: {method}",
                "valid_methods": ["equal_weight", "inverse_vol",
                                   "min_variance", "max_sharpe"]}

    # 组合统计
    port_return = sum(w * mu for w, mu in zip(weights, means)) * periods_per_year
    # 组合波动率（需协方差）
    try:
        import numpy as np
        R = np.array(series).T
        cov = np.cov(R.T, ddof=1)
        w_arr = np.array(weights)
        port_vol = float(np.sqrt(w_arr @ cov @ w_arr)) * math.sqrt(periods_per_year)
    except Exception:
        # 无 numpy 时用近似（忽略相关性）
        port_vol = math.sqrt(sum((w * s) ** 2 for w, s in zip(weights, stds))) \
                   * math.sqrt(periods_per_year)

    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    return {
        "success": True,
        "method": method_used,
        "assets": assets,
        "weights": dict(zip(assets, weights)),
        "portfolio_metrics": {
            "annualized_return": round(port_return, 4),
            "annualized_volatility": round(port_vol, 4),
            "sharpe_ratio": round(sharpe, 3),
        },
        "sample_size": m,
        "disclaimer": "结果基于历史数据，不保证未来表现",
    }


# ─────────────────────────────────────────────
# Tool 11: implied_volatility — 反推 IV
# ─────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-5,
) -> dict:
    """
    从期权市场价反推 Black-Scholes 隐含波动率（Newton-Raphson）。
    """
    if any(x <= 0 for x in [market_price, spot, strike, time_to_expiry]):
        return {"success": False,
                "error": "market_price/spot/strike/T 必须为正数"}
    opt = option_type.lower()
    if opt not in ("call", "put"):
        return {"success": False, "error": "option_type 须为 call 或 put"}

    S, K, T, r = spot, strike, time_to_expiry, risk_free_rate

    def _bs_price(sigma):
        if sigma <= 0:
            return float('inf')
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt == "call":
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    def _vega(sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * _norm_pdf(d1) * math.sqrt(T)

    # 内在价值检查（市场价不能低于内在价值）
    intrinsic = max(S - K, 0) if opt == "call" else max(K - S, 0)
    if market_price < intrinsic * 0.99:
        return {"success": False,
                "error": f"市场价 {market_price} 低于内在价值 {intrinsic:.4f}，无解",
                "hint": "检查输入是否正确，或期权可能已无流动性"}

    # Newton-Raphson 迭代，初值 σ = 0.3
    sigma = 0.3
    converged = False
    for i in range(max_iter):
        price = _bs_price(sigma)
        diff = price - market_price
        if abs(diff) < tol:
            converged = True
            break
        v = _vega(sigma)
        if v < 1e-10:
            break
        sigma = max(0.001, sigma - diff / v)

    if not converged:
        return {"success": False,
                "error": "未收敛，请检查输入参数",
                "last_sigma": round(sigma, 4),
                "iterations": max_iter}

    # 与历史波动率对比的语义化判断
    if sigma < 0.15:
        level = "极低 / 平静期"
    elif sigma < 0.25:
        level = "偏低"
    elif sigma < 0.40:
        level = "正常区间"
    elif sigma < 0.60:
        level = "偏高 / 市场紧张"
    else:
        level = "极高 / 危机或事件驱动"

    return {
        "success": True,
        "implied_volatility": round(sigma, 4),
        "annualized_pct": f"{sigma*100:.2f}%",
        "level": level,
        "iterations_used": i + 1,
        "inputs": {
            "market_price": market_price,
            "spot": spot, "strike": strike,
            "T_years": T, "risk_free": r,
            "option_type": opt,
        },
        "formula": "Newton-Raphson 求解 BS_price(σ) = market_price",
        "disclaimer": "假设欧式期权、无股息、几何布朗运动",
    }


# ─────────────────────────────────────────────
# Tool 12: html_chart_render — ECharts 图表生成
# ─────────────────────────────────────────────

import uuid as _uuid

# 图表输出目录
_CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(_CHARTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Tool 13: historical_prices — 历史 K 线（AKShare）
# ─────────────────────────────────────────────

_PRICE_CACHE_TTL = 1800  # 30 分钟


def _hist_yahoo(symbol: str, days: int) -> dict:
    """
    Yahoo Finance Chart API 拉历史 K 线（全球可用，无需 API Key）。
    返回与 historical_prices 同构的 dict 或 {success: False}。
    """
    market = _detect_market(symbol)
    # symbol → Yahoo 格式
    s_clean = symbol.upper()
    for suf in ('.SS', '.SH', '.SZ', '.HK', '.US'):
        s_clean = s_clean.replace(suf, '')

    if market == "A":
        if s_clean.startswith(('5', '6', '9')):
            ysym = f"{s_clean}.SS"
        else:
            ysym = f"{s_clean}.SZ"
    elif market == "HK":
        ysym = f"{s_clean.zfill(4)}.HK"
    elif market == "US":
        ysym = s_clean
    else:
        return {"success": False, "error": "未识别市场"}

    # 用精确 period1/period2 替代粗糙 range，保证拿够 days 个交易日
    # 多拉 80% 缓冲（含周末/节假日）
    import time as _t
    buffer_days = int(days * 1.8) + 5
    period2 = int(_t.time())
    period1 = period2 - buffer_days * 86400

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(ysym)}"
           f"?period1={period1}&period2={period2}&interval=1d")

    try:
        data = _yahoo_fetch(url, retries=1)
        chart = (data.get("chart", {}).get("result") or [None])[0]
        if not chart:
            return {"success": False, "error": "Yahoo 返回空数据"}

        ts_list = chart.get("timestamp", []) or []
        quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
        opens, closes = quote.get("open", []), quote.get("close", [])
        highs, lows = quote.get("high", []), quote.get("low", [])
        vols = quote.get("volume", [])

        valid = []
        for i, ts in enumerate(ts_list):
            if (i < len(closes) and closes[i] is not None
                    and opens[i] is not None
                    and highs[i] is not None
                    and lows[i] is not None):
                valid.append((
                    datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    opens[i], closes[i], highs[i], lows[i],
                    vols[i] if i < len(vols) and vols[i] else 0,
                ))
        valid = valid[-days:]
        if not valid:
            return {"success": False, "error": "Yahoo 无有效数据点"}

        market_label = {"A": "A股", "HK": "港股", "US": "美股"}[market]
        return {
            "success": True,
            "symbol": s_clean,
            "market": market_label,
            "days_returned": len(valid),
            "dates":  [v[0] for v in valid],
            "open":   [float(v[1]) for v in valid],
            "close":  [float(v[2]) for v in valid],
            "high":   [float(v[3]) for v in valid],
            "low":    [float(v[4]) for v in valid],
            "volume": [int(v[5]) for v in valid],
            "data_source": "Yahoo Finance Chart API（全球可用）",
        }
    except Exception as e:
        return {"success": False, "error": f"Yahoo K 线失败: {e}"}


def _hist_yfinance(symbol: str, days: int) -> dict:
    """
    yfinance 库（pip install yfinance）：
    自动处理 Yahoo Cookie/Crumb 认证，比直接 HTTP 稳。
    """
    if not _YFINANCE_AVAILABLE:
        return {"success": False, "error": "yfinance 未安装"}

    market = _detect_market(symbol)
    s_clean = symbol.upper()
    for suf in ('.SS', '.SH', '.SZ', '.HK', '.US'):
        s_clean = s_clean.replace(suf, '')

    # 转 Yahoo ticker 格式
    if market == "A":
        ysym = f"{s_clean}.SS" if s_clean.startswith(('5','6','9')) else f"{s_clean}.SZ"
    elif market == "HK":
        ysym = f"{s_clean.zfill(4)}.HK"
    elif market == "US":
        ysym = s_clean
    else:
        return {"success": False, "error": "未识别市场"}

    try:
        ticker = _yf.Ticker(ysym)
        # yfinance period 选项: 1d/5d/1mo/3mo/6mo/1y/2y/5y/10y/ytd/max
        if days <= 30:    period = "1mo"
        elif days <= 90:  period = "3mo"
        elif days <= 180: period = "6mo"
        elif days <= 365: period = "1y"
        elif days <= 730: period = "2y"
        else:             period = "5y"

        df = ticker.history(period=period, interval="1d",
                             auto_adjust=True, raise_errors=False)
        if df is None or df.empty:
            return {"success": False, "error": "yfinance 返回空"}

        df = df.tail(days)
        market_label = {"A": "A股", "HK": "港股", "US": "美股"}[market]
        return {
            "success": True,
            "symbol": s_clean,
            "market": market_label,
            "days_returned": len(df),
            "dates":  [d.strftime("%Y-%m-%d") for d in df.index],
            "open":   [float(x) for x in df['Open']],
            "close":  [float(x) for x in df['Close']],
            "high":   [float(x) for x in df['High']],
            "low":    [float(x) for x in df['Low']],
            "volume": [int(x) for x in df['Volume'].fillna(0)],
            "data_source": "yfinance（Yahoo 官方认证）",
        }
    except Exception as e:
        return {"success": False, "error": f"yfinance: {e}"}


def _hist_sina(symbol: str, days: int = 60) -> dict:
    """
    新浪财经 K 线接口（A 股专用 fallback）。
    与 AKShare 内部走的东方财富 endpoint 不同，多一条备路。
    """
    market = _detect_market(symbol)
    if market != "A":
        return {"success": False, "error": "新浪 K 线仅支持 A 股"}

    prefix = _sina_prefix(symbol)
    if not prefix or not prefix.startswith(('sh', 'sz')):
        return {"success": False, "error": "无效 A 股代码"}

    try:
        # scale=240 表示日 K，datalen 多拉 20%
        datalen = min(int(days * 1.2) + 10, 500)
        url = (f"https://quotes.sina.cn/cn/api/json_v2.php/"
               f"CN_MarketDataService.getKLineData"
               f"?symbol={prefix}&scale=240&ma=no&datalen={datalen}")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not isinstance(data, list) or not data:
            return {"success": False, "error": "新浪 K 线返回空"}

        # 返回结构: [{day, open, high, low, close, volume}, ...]
        data = data[-days:]
        return {
            "success": True,
            "symbol": prefix[2:],
            "market": "A股",
            "days_returned": len(data),
            "dates":  [d["day"] for d in data],
            "open":   [float(d["open"]) for d in data],
            "close":  [float(d["close"]) for d in data],
            "high":   [float(d["high"]) for d in data],
            "low":    [float(d["low"]) for d in data],
            "volume": [int(float(d["volume"])) for d in data],
            "data_source": "新浪财经日 K",
        }
    except Exception as e:
        return {"success": False, "error": f"新浪 K 线失败: {e}"}


def _hist_akshare(symbol: str, days: int) -> dict:
    """AKShare 拉历史 K 线（国内源最快）"""
    if not _AKSHARE_AVAILABLE:
        return {"success": False, "error": "AKShare 未安装"}

    market = _detect_market(symbol)
    end_d = datetime.date.today().strftime('%Y%m%d')
    start_d = (datetime.date.today()
               - datetime.timedelta(days=int(days * 1.6))).strftime('%Y%m%d')

    try:
        if market == "A":
            s = symbol.upper().replace(".SS", "").replace(".SH", "").replace(".SZ", "")
            df = _ak.stock_zh_a_hist(symbol=s, period='daily',
                                      start_date=start_d, end_date=end_d,
                                      adjust='qfq')
            label = "A股"
        elif market == "US":
            s = symbol.upper().replace(".US", "")
            df = _ak.stock_us_hist(symbol=s, period='daily',
                                    start_date=start_d, end_date=end_d,
                                    adjust='qfq')
            label = "美股"
        elif market == "HK":
            s = symbol.upper().replace(".HK", "").zfill(5)
            df = _ak.stock_hk_hist(symbol=s, period='daily',
                                    start_date=start_d, end_date=end_d,
                                    adjust='qfq')
            label = "港股"
        else:
            return {"success": False, "error": "未识别市场"}

        if len(df) == 0:
            return {"success": False, "error": "AKShare 返回空"}
        df = df.tail(days)
        return {
            "success": True,
            "symbol": s, "market": label,
            "days_returned": len(df),
            "dates":  df['日期'].astype(str).tolist(),
            "open":   [float(x) for x in df['开盘']],
            "close":  [float(x) for x in df['收盘']],
            "high":   [float(x) for x in df['最高']],
            "low":    [float(x) for x in df['最低']],
            "volume": [int(x) for x in df['成交量']],
            "data_source": "AKShare 东方财富日 K（前复权）",
        }
    except Exception as e:
        return {"success": False, "error": f"AKShare 失败: {e}"}


def historical_prices(symbol: str, days: int = 60) -> dict:
    """
    获取近 N 个交易日的历史 K 线（多源 fallback）。

    数据源优先级：
      1. AKShare（国内源最快，A 股数据最全）
      2. Yahoo Finance（全球可用，无 Key 兜底）

    返回结构统一：dates / open / close / high / low / volume / returns
    """
    if days < 5 or days > 500:
        return {"success": False, "error": "days 需要在 5-500 之间"}

    market = _detect_market(symbol)
    if market == "unknown":
        return {"success": False,
                "error": f"未识别市场代码: {symbol}",
                "hint": "支持 600519(A)/AAPL(US)/0700(HK)"}

    cache_key = f"quant:price:{symbol.upper()}:{days}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    sources_tried = []
    result = None

    # ── 优先级 1: AKShare（数据最全，A 股最快）──
    r = _hist_akshare(symbol, days)
    sources_tried.append({"source": "AKShare", "ok": r.get("success"),
                          "error": r.get("error")})
    if r.get("success"):
        result = r

    # ── 优先级 2: 新浪 K 线（A 股专属 fallback，避免与 AKShare 同时挂）──
    if result is None and market == "A":
        r2 = _hist_sina(symbol, days)
        sources_tried.append({"source": "新浪财经",
                              "ok": r2.get("success"),
                              "error": r2.get("error")})
        if r2.get("success"):
            result = r2

    # ── 优先级 3: yfinance 库（处理 Yahoo 官方认证，比直接 HTTP 稳）──
    if result is None and _YFINANCE_AVAILABLE:
        r3 = _hist_yfinance(symbol, days)
        sources_tried.append({"source": "yfinance",
                              "ok": r3.get("success"),
                              "error": r3.get("error")})
        if r3.get("success"):
            result = r3

    # ── 优先级 4: Yahoo Finance Chart API（裸 HTTP 兜底）──
    if result is None:
        r4 = _hist_yahoo(symbol, days)
        sources_tried.append({"source": "Yahoo Finance",
                              "ok": r4.get("success"),
                              "error": r4.get("error")})
        if r4.get("success"):
            result = r4

    if result is None:
        return {
            "success": False,
            "error": "所有数据源都失败",
            "sources_tried": sources_tried,
            "hint": ("1) 检查 HTTP_PROXY/HTTPS_PROXY 环境变量；"
                     "2) 确认代码格式（A股 600519 / 美股 AAPL / 港股 0700）；"
                     "3) 等几分钟避开 Yahoo 限流"),
        }

    # 顺便算收益率序列，方便后续工具直接用
    closes = result["close"]
    result["returns"] = [round(closes[i] / closes[i-1] - 1, 6)
                          for i in range(1, len(closes))]
    result["sources_tried"] = sources_tried

    cache.set(cache_key, result, ttl=_PRICE_CACHE_TTL)
    return result


def html_chart_render(
    chart_type: str,
    data: dict,
    title: str = "",
) -> dict:
    """
    生成 ECharts 交互式图表 HTML，保存到 static/charts/，返回访问 URL。

    chart_type:
      - "candlestick"  K 线图。data: {dates:[], ohlc:[[open,close,low,high]]}
      - "radar"        因子雷达图。data: {indicators:[...], series:[{name, values}]}
      - "heatmap"      相关性热力图。data: {labels:[...], matrix:[[..]]}
      - "bar"          柱状图。data: {x:[], y:[]} 或 {x:[], series:[{name,y}]}
      - "pie"          饼图。data: {items:[{name, value}]}
      - "line"         折线图。data: {x:[], series:[{name, y}]}
    """
    ct = chart_type.lower().strip()
    chart_id = _uuid.uuid4().hex[:8]
    option = None

    # ── 字段宽容性辅助 ──
    def _first(d, *keys, default=None):
        """从 dict 取第一个存在的字段值"""
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] is not None:
                return d[k]
        return default

    def _list_values(item):
        """从 series 项里提取数值数组，兼容 values/value/data/y"""
        return _first(item, "values", "value", "data", "y")

    try:
        if ct == "candlestick":
            # ⭐ 快捷模式：data={"symbol":"512760","days":60} —— 工具内部自动拉数据
            #    避免 LLM 把 60+ 个日期/OHLC 数组塞进 tool_calls 参数（导致 max_tokens 截断）
            if "symbol" in data and not data.get("dates") and not data.get("ohlc"):
                hp = historical_prices(data["symbol"],
                                        days=int(data.get("days", 60)))
                if not hp.get("success"):
                    return {"success": False,
                            "error": f"K 线快捷模式拉数失败: {hp.get('error')}",
                            "hint": "检查 symbol 格式或网络"}
                dates = hp["dates"]
                # ECharts K 线顺序 [open, close, low, high]
                ohlc = [[hp["open"][i], hp["close"][i],
                         hp["low"][i], hp["high"][i]]
                        for i in range(len(dates))]
            else:
                dates = _first(data, "dates", "x", "labels")
                ohlc = _first(data, "ohlc", "data", "values")
            if not dates or not ohlc:
                return {"success": False,
                        "error": "candlestick 需要 dates+ohlc，或快捷模式 symbol+days",
                        "example": '{"symbol":"512760","days":60}'}
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "xAxis": {"data": dates},
                "yAxis": {"scale": True},
                "series": [{"type": "candlestick", "data": ohlc}],
                "tooltip": {"trigger": "axis"},
            }

        elif ct == "radar":
            inds = _first(data, "indicators", "axes", "labels", "categories")
            series_in = _first(data, "series", "data")
            if not inds or not series_in:
                return {"success": False,
                        "error": "radar 需要 indicators 和 series",
                        "example": ('{"indicators":["Value","Growth"],'
                                     ' "series":[{"name":"AAPL","values":[80,75]}]}')}
            indicator_def = [{"name": x, "max": 100} for x in inds]
            radar_data = []
            for s in series_in:
                if isinstance(s, dict):
                    name = _first(s, "name", "label", default="Series")
                    vals = _list_values(s)
                elif isinstance(s, list):
                    # 兼容直接传数组：series:[[80,75,90,60,70]]
                    name = "Series"
                    vals = s
                else:
                    continue
                if vals:
                    radar_data.append({"name": name, "value": vals})
            if not radar_data:
                return {"success": False,
                        "error": "radar series 中没有有效的 values 数组"}
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "radar": {"indicator": indicator_def},
                "series": [{"type": "radar", "data": radar_data}],
                "tooltip": {},
            }

        elif ct == "heatmap":
            labels = _first(data, "labels", "x", "categories")
            mat = _first(data, "matrix", "data", "values")
            if not labels or not mat:
                return {"success": False,
                        "error": "heatmap 需要 labels 和 matrix",
                        "example": '{"labels":["A","B"], "matrix":[[1,0.7],[0.7,1]]}'}
            cells = []
            for i, _ in enumerate(labels):
                for j, _ in enumerate(labels):
                    if i < len(mat) and j < len(mat[i]):
                        cells.append([j, i, mat[i][j]])
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "xAxis": {"type": "category", "data": labels},
                "yAxis": {"type": "category", "data": labels},
                "visualMap": {"min": -1, "max": 1, "calculable": True,
                              "inRange": {"color": ["#f85149", "#161b22", "#00d4aa"]}},
                "series": [{"type": "heatmap", "data": cells,
                             "label": {"show": True, "formatter": "{c}"}}],
                "tooltip": {"position": "top"},
            }

        elif ct == "bar":
            x = _first(data, "x", "labels", "categories")
            if not x:
                return {"success": False,
                        "error": "bar 需要 x（类别数组）",
                        "example": '{"x":["A","B"], "y":[10,20]}'}
            if "series" in data and isinstance(data["series"], list):
                series = [{"type": "bar",
                           "name": _first(s, "name", "label", default="S"),
                           "data": _list_values(s) or []}
                          for s in data["series"]]
            else:
                ys = _first(data, "y", "values", "data")
                series = [{"type": "bar", "data": ys or []}]
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "xAxis": {"type": "category", "data": x},
                "yAxis": {"type": "value"},
                "series": series,
                "tooltip": {"trigger": "axis"},
                "legend": {"top": "bottom"} if "series" in data else None,
            }

        elif ct == "pie":
            # 兼容 items / data，每条接受 {name|label, value}
            items_in = _first(data, "items", "data", "series")
            if not items_in:
                return {"success": False,
                        "error": "pie 需要 items 数组",
                        "example": '{"items":[{"name":"A","value":40},{"name":"B","value":60}]}'}
            pie_data = []
            for it in items_in:
                if isinstance(it, dict):
                    pie_data.append({
                        "name": _first(it, "name", "label", default="?"),
                        "value": _first(it, "value", "y", default=0),
                    })
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "series": [{"type": "pie", "radius": "55%", "data": pie_data}],
                "tooltip": {"trigger": "item"},
            }

        elif ct == "line":
            x = _first(data, "x", "labels", "dates", "categories")
            series_in = _first(data, "series", "data")
            if not x or not series_in:
                return {"success": False,
                        "error": "line 需要 x 和 series",
                        "example": '{"x":["1","2"], "series":[{"name":"NVDA","y":[100,105]}]}'}
            series = [{"type": "line", "smooth": True,
                       "name": _first(s, "name", "label", default="S"),
                       "data": _list_values(s) or []}
                      for s in series_in if isinstance(s, dict)]
            option = {
                "title": {"text": title, "textStyle": {"color": "#e6edf3"}},
                "xAxis": {"type": "category", "data": x},
                "yAxis": {"type": "value"},
                "series": series,
                "tooltip": {"trigger": "axis"},
                "legend": {"top": "bottom"},
            }

        else:
            return {"success": False,
                    "error": f"未知 chart_type: {chart_type}",
                    "valid_types": ["candlestick", "radar", "heatmap",
                                     "bar", "pie", "line"]}
    except Exception as e:
        return {"success": False,
                "error": f"图表配置异常: {type(e).__name__}: {e}",
                "data_received": str(data)[:200]}

    # 构造 HTML 页面
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title or 'QuantAgent 图表'}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ background: #0d1117; color: #e6edf3; font-family: sans-serif;
         margin: 0; padding: 20px; }}
  #chart {{ width: 100%; height: 90vh; }}
  .footer {{ color: #8b949e; font-size: 12px; margin-top: 10px;
             text-align: center; }}
</style>
</head>
<body>
  <div id="chart"></div>
  <div class="footer">QuantAgent · 数据仅供研究参考，不构成投资建议</div>
  <script>
    const chart = echarts.init(document.getElementById('chart'), 'dark');
    chart.setOption({json.dumps(option, ensure_ascii=False, default=str)});
    window.addEventListener('resize', () => chart.resize());
  </script>
</body>
</html>"""

    filename = f"{ct}_{chart_id}.html"
    filepath = os.path.join(_CHARTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "success": True,
        "chart_type": ct,
        "title": title,
        "url": f"/static/charts/{filename}",
        "filename": filename,
        "render_hint": "前端引用 url 即可在浏览器查看交互式图表",
    }


TOOL_REGISTRY = {
    "market_quote":         market_quote,
    "factor_score":         factor_score,
    "technical_indicator":  technical_indicator,
    "black_scholes":        black_scholes,
    "risk_metrics":         risk_metrics,
    "market_news_search":   market_news_search,
    "trading_calendar":     trading_calendar,
    "search_research_docs": search_research_docs,
    "correlation_matrix":   correlation_matrix,
    "portfolio_optimizer":  portfolio_optimizer,
    "implied_volatility":   implied_volatility,
    "html_chart_render":    html_chart_render,
    "historical_prices":    historical_prices,
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
        "description": """技术指标计算（RSI / MACD / Bollinger Bands / SMA）。
输入：收盘价数组（时间正序，旧→新）+ 指标名。

⚠️ 重要：本工具只做计算，不会自己拉数据。
   必须先调用 `historical_prices(symbol, days=N)` 拿到 close 数组再传入。
   各指标对数据长度要求：
     - RSI(14):    至少 15 个价格点（建议 days=30）
     - MACD:       至少 26 个（建议 days=40+）
     - Bollinger:  至少 period 个（默认 period=20）
     - SMA:        至少 period 个""",
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
        "name": "historical_prices",
        "description": """拉取近 N 个交易日的历史 K 线（OHLCV + 收益率序列）。
⚠️ 这是 technical_indicator / correlation_matrix / portfolio_optimizer 的「上游」工具。
凡需要历史价格数组的场景，先用本工具拿数据。

返回字段：dates / open / close / high / low / volume / returns
  - close 数组直接喂给 technical_indicator
  - returns 数组直接喂给 correlation_matrix 或 portfolio_optimizer
  - dates + ohlc 组合可喂给 html_chart_render(chart_type='candlestick')

支持市场：
  - A 股: 600519 / 000858 / 510300 等
  - 美股: AAPL / NVDA / TSLA
  - 港股: 0700 / 09988

days 建议（自然日，工具内部会换算成交易日）：
  - RSI/SMA: ≥ 45 （保证 ≥30 个交易日）
  - MACD:    ≥ 50 （保证 ≥35 个交易日，覆盖 EMA26+signal9）
  - 布林带:  ≥ 40
  - 因子:    60-120
  - K 线图:  60-90
⚠️ 实际拿到的交易日数 ≈ days × 0.7（去掉周末/节假日）。
   想算 MACD 至少传 days=50，不要传 30。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days":   {"type": "integer",
                           "description": "返回最近 N 个交易日",
                           "default": 60, "minimum": 5, "maximum": 500},
            },
            "required": ["symbol"],
        }
    },
    {
        "name": "correlation_matrix",
        "description": """计算多资产收益率相关性矩阵。
输入：{asset_name: [return_1, return_2, ...]} 字典，至少 2 个资产，每个序列 >=5 点。
返回：对称矩阵 + 高相关性 pair 警示（|corr| >= 0.7）。
适用：评估组合多元化效果、寻找对冲对、识别高度同涨同跌资产。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "returns_data": {
                    "type": "object",
                    "description": "资产→收益率序列的字典",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
            },
            "required": ["returns_data"],
        }
    },
    {
        "name": "portfolio_optimizer",
        "description": """多资产组合权重优化。
method 选项：
  - equal_weight: 等权重（基线参考）
  - inverse_vol:  反波动率加权（简易风险平价）
  - min_variance: 最小方差（需历史协方差）
  - max_sharpe:   最大夏普比率（解析解，不允许做空）

返回：每个资产权重 + 组合年化收益/波动/夏普。
样本量越大结果越稳定，建议日频 60 天以上。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "returns_data": {
                    "type": "object",
                    "description": "资产→收益率序列字典",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "method": {
                    "type": "string",
                    "enum": ["equal_weight", "inverse_vol",
                             "min_variance", "max_sharpe"],
                    "default": "min_variance",
                },
                "risk_free_rate": {"type": "number", "default": 0.03},
                "periods_per_year": {"type": "integer", "default": 252},
            },
            "required": ["returns_data"],
        }
    },
    {
        "name": "implied_volatility",
        "description": """从期权市场价反推 Black-Scholes 隐含波动率（IV）。
适用：评估期权定价是否合理、对比当前 IV 与历史水平判断市场紧张度。
返回的 IV 越高代表市场预期未来波动越大。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_price":   {"type": "number", "description": "期权当前市场价"},
                "spot":           {"type": "number", "description": "标的现价"},
                "strike":         {"type": "number", "description": "行权价"},
                "time_to_expiry": {"type": "number", "description": "到期年数（如 0.25=3月）"},
                "risk_free_rate": {"type": "number"},
                "option_type":    {"type": "string", "enum": ["call", "put"]},
            },
            "required": ["market_price", "spot", "strike",
                         "time_to_expiry", "risk_free_rate", "option_type"],
        }
    },
    {
        "name": "html_chart_render",
        "description": """生成 ECharts 交互式 HTML 图表，返回可访问的 URL。
工具会按 chart_type 解析 data，字段名宽容（values/value/data/y 可互换）。

▶ radar（因子雷达图）—— 最常用，配合 factor_score：
  {
    "indicators": ["Value", "Growth", "Momentum", "Quality", "Technical"],
    "series": [{"name": "AAPL", "values": [80, 75, 60, 90, 70]}]
  }
  注：values 数组顺序必须与 indicators 对齐。

▶ pie（饼图）—— 配合 portfolio_optimizer 画权重：
  {"items": [{"name": "AAPL", "value": 0.4}, {"name": "MSFT", "value": 0.6}]}

▶ heatmap（热力图）—— 配合 correlation_matrix：
  {"labels": ["AAPL", "MSFT", "GLD"],
   "matrix": [[1, 0.8, -0.3], [0.8, 1, -0.2], [-0.3, -0.2, 1]]}

▶ candlestick（K 线图）—— ⭐ 强烈推荐快捷模式（省 token）：
  {"symbol": "512760", "days": 60}
  工具内部自动调 historical_prices 拉数据，无需你传一长串日期/OHLC 数组。
  禁止手动把 60+ 日期数组塞进参数，会触发 max_tokens 截断！

  仅当数据来源不是历史 K 线时才用手动模式：
  {"dates": [...], "ohlc": [[open, close, low, high], ...]}

▶ bar（柱状图）：
  {"x": ["Q1","Q2","Q3"], "y": [100, 120, 95]}

▶ line（折线图）：
  {"x": ["1月","2月"], "series": [{"name": "NVDA", "y": [100, 105]}]}

⚠️ 调用成功后在报告里引用 url 时：
  ✅ 正确：[查看 K 线图](/static/charts/xxx.html)   ← 直接用工具返回的 url 字段，相对路径
  ❌ 错误：[查看](http://localhost:8001/static/...)  ← 千万别加 host/端口！会变死链
  ❌ 错误：[查看](https://example.com/static/...)
  浏览器会自动用当前域名解析相对路径，无需拼接 host。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["candlestick", "radar", "heatmap",
                             "bar", "pie", "line"],
                },
                "data": {
                    "type": "object",
                    "description": "图表数据，结构按 chart_type 不同而异",
                },
                "title": {"type": "string", "default": ""},
            },
            "required": ["chart_type", "data"],
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

SYSTEM_PROMPT = """你是 QuantAgent v1.0，量化金融分析助手（多因子/期权/风险/组合/RAG）。

【数据流要求】
- 凡需要历史价格的下游计算（技术指标/相关性/组合优化/K 线图），先调 historical_prices
- 凡需要实时价格，调 market_quote
- 凡需要分析公司财报/研报观点，先调 search_research_docs

【代码类型识别 - 避免无效调用】
- 个股代码（6 位数字开头 6/0/3）: 全套工具可用（factor_score / technical_indicator / news 等）
- ETF/LOF/基金代码（5xxxxx / 15xxxx / 159xxx / 16xxxx）:
    ❌ 不要调 factor_score（无财务报表，会快速报错但浪费一轮）
    ✅ 直接 historical_prices + technical_indicator + market_news_search
- 中文主题词（"芯片ETF" 等非代码）:
    ✅ 先 market_news_search 找具体代码，再走代码分析路径

【工具协同建议】
- 复杂任务**并行调用多工具**（一次决策里多个 tool_calls），减少轮次
- 数据源标识：data_source 字段写"实时计算"/"实时行情"是真数据，直接引用；
  写 [MOCK DATA] 才需明示"演示数据"
- 工具失败读 hint 字段后纠正参数重试

【禁止】
- 不用"一定涨/跌"、"保证收益"等绝对化表述
- 不编造数据，缺数据标注 [DATA UNAVAILABLE]
- 不省略风险提示

【报告输出格式（仅"分析"类问题）】
```
# 📊 [标题]
**风险等级**：🟢低 / 🟡中 / 🔴高

## 核心结论 / Key Takeaways
- 3-5 条要点

## 量化分析 / Quantitative Analysis
[数据 + 计算 + 数据源]

## 风险提示 / Risk Warnings
[必含]

---
⚠️ 仅供研究参考，不构成投资建议。
```

简短问题（如"现在几点"、"AAPL 多少钱"）直接回答即可，不必套报告格式。"""


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


def _truncate_observation(obs: dict, obs_str: str, max_len: int = 2500) -> str:
    """
    工具返回过大时按字段裁剪。
    ⚠️ 注意：historical_prices 的 close/dates 等数组不裁剪 ——
       它们是下游工具的必需输入，裁了就无法调用 technical_indicator。
    """
    if len(obs_str) <= max_len:
        return obs_str

    # 新闻类（多源聚合容易超长）：限 top-3
    for k in ("results", "news", "quotes"):
        if isinstance(obs.get(k), list) and len(obs[k]) > 3:
            obs[k] = obs[k][:3]
            obs["truncated"] = True

    # 新闻 content/summary 字段过长再截
    for list_field in ("results", "news"):
        items = obs.get(list_field)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for tk in ("content", "summary", "title"):
                        v = item.get(tk)
                        if isinstance(v, str) and len(v) > 200:
                            item[tk] = v[:200] + "..."

    return json.dumps(obs, ensure_ascii=False, default=str)


def _generate_followups(messages: list) -> list:
    """
    基于近几轮对话生成 3 个用户可能接着想问的问题。
    单独再调用一次 LLM（轻量 prompt + max_tokens 限制），成本约 0.002 元/次。
    失败时静默返回 []，不影响主流程。
    """
    # 抽取最近的 user + assistant 对话片段（控制 prompt 长度）
    recent = []
    for m in reversed(messages):
        if m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        # 截断过长的 assistant 报告（只保留前 600 字符给上下文）
        snippet = content if len(content) < 600 else content[:600] + "..."
        recent.insert(0, {"role": m["role"], "content": snippet})
        if len(recent) >= 4:
            break

    if not recent:
        return []

    sys_prompt = (
        "你是量化分析对话的「问题推荐器」。基于以下用户与 Agent 的对话，"
        "推断用户接下来可能想问的 3 个紧密相关的问题。\n"
        "要求：\n"
        "1. 每个问题 ≤ 25 个字\n"
        "2. 紧扣前文上下文，可深入挖掘或横向扩展\n"
        "3. 不要重复用户已问过的问题\n"
        "4. 偏量化分析视角（涉及具体股票/指标/对比/风险等）\n"
        "5. 只输出 JSON 数组，格式：[\"问题1\",\"问题2\",\"问题3\"]，"
        "不要任何前后缀解释。"
    )

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                *recent,
                {"role": "user",
                 "content": "请基于以上对话生成 3 个 follow-up 问题（JSON 数组）"},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        # 提取 JSON 数组（容忍模型偶尔加解释）
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            return []
        items = json.loads(match.group(0))
        if not isinstance(items, list):
            return []
        cleaned = [str(x).strip() for x in items if x and isinstance(x, str)]
        return cleaned[:3]
    except Exception:
        return []


def _sanitize_messages(messages: list) -> list:
    """
    清洗历史：移除「悬空」的 assistant.tool_calls（没对应 tool 响应的）。
    通常由用户中途取消请求导致。
    原地修改并返回 messages。
    """
    if not messages:
        return messages

    cleaned = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            expected = {tc["id"] for tc in m["tool_calls"]}
            # 看后面紧跟着的 tool 消息
            j = i + 1
            following = []
            while j < len(messages) and messages[j].get("role") == "tool":
                following.append(messages[j])
                j += 1
            covered = {t.get("tool_call_id") for t in following}

            if expected <= covered:
                # 完整闭环，原样保留
                cleaned.append(m)
                cleaned.extend(following)
            else:
                # 不完整 —— 跳过这个 assistant 和后面的 tool 残片
                # （保留之前的 user message，让 LLM 当从未发生）
                pass
            i = j
        else:
            cleaned.append(m)
            i += 1

    # 原地替换
    messages[:] = cleaned
    return messages


def stream_quant_agent(messages: list, max_iterations: int = 15):
    """
    生成器版 Agent 主循环（DeepSeek/OpenAI 协议）。
    yield 事件协议保持不变，与前端解耦。
    """
    # 首次调用时注入 system message
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # 防御：清洗历史中可能存在的悬空 tool_calls（来自上次取消请求）
    _sanitize_messages(messages)

    for iteration in range(1, max_iterations + 1):
        # ── 流式调用 DeepSeek，边接收边推送给前端 ──
        try:
            stream = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=_OPENAI_TOOLS,
                max_tokens=8000,
                stream=True,
            )
        except Exception as e:
            yield {"type": "error", "error": f"API 调用失败: {e}"}
            return

        # 流式累积器
        content_buf = ""
        finish = None
        tool_calls_buf = []   # [{id, name, arguments}, ...] 按 index 累积

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish = choice.finish_reason

                if delta and delta.content:
                    content_buf += delta.content
                    yield {"type": "content_delta",
                           "text": delta.content,
                           "iteration": iteration}

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(tool_calls_buf) <= idx:
                            tool_calls_buf.append({"id": "", "name": "",
                                                    "arguments": ""})
                        if tc_delta.id:
                            tool_calls_buf[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_buf[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_buf[idx]["arguments"] += tc_delta.function.arguments
        except Exception as e:
            yield {"type": "error", "error": f"流式接收失败: {e}"}
            return

        # ── finish_reason 分支 ──
        if finish == "stop":
            messages.append({"role": "assistant", "content": content_buf})
            yield {
                "type": "final",
                "text": content_buf,
                "text_html": render_markdown_to_html(content_buf),  # 服务端渲染好
                "iterations": iteration,
            }
            followups = _generate_followups(messages)
            if followups:
                yield {"type": "suggestions", "items": followups}
            return

        if finish == "length":
            warning = ("\n\n---\n\n⚠️ **本回复因长度限制被截断**，"
                       "建议把问题拆分成几个小问题分别提问。")
            full_text = content_buf + warning
            messages.append({"role": "assistant", "content": content_buf})
            yield {
                "type": "final",
                "text": full_text,
                "text_html": render_markdown_to_html(full_text),
                "iterations": iteration,
                "truncated": True,
            }
            followups = _generate_followups(messages)
            if followups:
                yield {"type": "suggestions", "items": followups}
            return

        if finish == "tool_calls" and tool_calls_buf:
            # 流式过程中如果有思考文本，已通过 content_delta 推过了
            # 这里发个 thought 总结（兼容老前端）
            if content_buf.strip():
                yield {"type": "thought",
                       "text": content_buf.strip(),
                       "iteration": iteration}

            # 预构造 assistant 消息
            assistant_msg = {
                "role": "assistant",
                "content": content_buf,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                   "arguments": tc["arguments"]}}
                    for tc in tool_calls_buf
                ],
            }
            tool_msgs = []

            for tc in tool_calls_buf:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = None

                yield {"type": "tool_call",
                       "name": tc["name"],
                       "input": args if isinstance(args, dict) else {},
                       "id": tc["id"]}

                if not isinstance(args, dict):
                    # 通常是 max_tokens 截断导致 JSON 不完整
                    truncated_hint = ""
                    if tc["name"] == "html_chart_render":
                        truncated_hint = (
                            " 推荐改用快捷模式："
                            "html_chart_render(chart_type='candlestick', "
                            "data={'symbol':'XXXXXX','days':60})，"
                            "不要手动传 dates/ohlc 数组，会被截断。"
                        )
                    obs = {
                        "success": False,
                        "error_type": "args_json_parse_failed",
                        "error": "工具参数 JSON 不完整（可能被 max_tokens 截断）",
                        "raw_arguments_preview": tc["arguments"][:200] + "...",
                        "hint": "请简化参数（少传大数组）或重新尝试。" + truncated_hint,
                    }
                else:
                    obs = dispatch_tool(tc["name"], args)

                obs_str = json.dumps(obs, ensure_ascii=False, default=str)
                obs_str = _truncate_observation(obs, obs_str)

                yield {"type": "tool_result",
                       "name": tc["name"],
                       "result": obs,
                       "is_error": not obs.get("success", True)}

                tool_msgs.append({"role": "tool",
                                   "tool_call_id": tc["id"],
                                   "content": obs_str})

            # 原子提交
            messages.append(assistant_msg)
            messages.extend(tool_msgs)
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
