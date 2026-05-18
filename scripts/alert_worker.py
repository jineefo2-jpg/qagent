"""
告警后台执行器
─────────────
独立进程，每 60 秒扫一次所有用户的告警，触发的发送通知。

启动：
    python -m scripts.alert_worker

部署（systemd 单独 unit）：
    [Service]
    ExecStart=/home/ubuntu/quant-agent/venv/bin/python -m scripts.alert_worker
    Restart=on-failure

支持的 channel：
    log              仅打印到 stdout（默认/演示）
    webhook:<url>    POST JSON 到指定 URL
    serverchan:<sckey>  Server酱微信推送
"""
import time
import json
import sys
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache import cache
from quant_agent import market_quote

SCAN_INTERVAL = 60   # 秒
ALERT_INDEX_PATTERN = "quant:alert:*"


def _eval_condition(condition: str, quote: dict) -> bool:
    """
    简易条件求值器（白名单变量 + 比较运算）。
    支持：price / change_pct / volume + 比较符 < > <= >= == !=
    """
    safe_vars = {
        "price": quote.get("price", 0),
        "change_pct": quote.get("change_pct", 0),
        "volume": quote.get("volume", 0),
    }
    # 安全检查：只允许变量名 + 数字 + 比较符
    import re
    if not re.match(r'^[a-z_]+\s*[<>=!]+\s*-?[\d.]+\s*$', condition.strip()):
        return False
    try:
        # 提取变量名
        m = re.match(r'^([a-z_]+)\s*([<>=!]+)\s*(-?[\d.]+)\s*$', condition.strip())
        if not m:
            return False
        var, op, val = m.group(1), m.group(2), float(m.group(3))
        if var not in safe_vars:
            return False
        cur = float(safe_vars[var])
        return {
            ">":  cur >  val,
            "<":  cur <  val,
            ">=": cur >= val,
            "<=": cur <= val,
            "==": cur == val,
            "!=": cur != val,
        }.get(op, False)
    except Exception:
        return False


def _notify(alert: dict, quote: dict):
    """根据 channel 发送通知"""
    channel = alert.get("channel", "log")
    msg = (f"⚠️ {alert['symbol']} 触发告警「{alert['condition']}」"
           f"\n现价: {quote.get('price')}  涨跌: {quote.get('change_pct')}%"
           f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if channel == "log":
        print(f"[ALERT] {msg}")
        return True

    if channel.startswith("webhook:"):
        url = channel[len("webhook:"):]
        try:
            data = json.dumps({"alert": alert, "quote": quote,
                                "message": msg}).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=8).read()
            return True
        except Exception as e:
            print(f"[ALERT] webhook 发送失败: {e}")
            return False

    if channel.startswith("serverchan:"):
        sckey = channel[len("serverchan:"):]
        try:
            url = f"https://sctapi.ftqq.com/{sckey}.send"
            data = urllib.parse.urlencode({
                "title": f"QuantAgent 告警: {alert['symbol']}",
                "desp": msg,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=8).read()
            return True
        except Exception as e:
            print(f"[ALERT] Server酱发送失败: {e}")
            return False

    print(f"[ALERT] 未知 channel: {channel}")
    return False


def scan_once():
    """扫描一轮所有用户的告警"""
    keys = cache.keys(ALERT_INDEX_PATTERN)
    triggered_count = 0
    checked_count = 0

    for key in keys:
        alerts = cache.get(key) or []
        if not alerts:
            continue
        changed = False
        for a in alerts:
            if a.get("triggered"):
                continue   # 已触发的不再重复
            checked_count += 1
            quote = market_quote(a["symbol"])
            if not quote.get("success"):
                continue
            if _eval_condition(a["condition"], quote):
                if _notify(a, quote):
                    a["triggered"] = True
                    a["triggered_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
                    a["triggered_price"] = quote.get("price")
                    changed = True
                    triggered_count += 1
        if changed:
            cache.set(key, alerts, ttl=86400 * 30)

    return checked_count, triggered_count


def main():
    print(f"🔔 Alert worker 启动，扫描间隔 {SCAN_INTERVAL}s")
    while True:
        try:
            checked, triggered = scan_once()
            now = time.strftime('%H:%M:%S')
            if triggered > 0:
                print(f"[{now}] 扫描 {checked} 个告警，触发 {triggered} 条")
            time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            print("\n👋 退出")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
