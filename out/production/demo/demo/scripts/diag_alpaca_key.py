"""
Alpaca 凭证脱敏诊断脚本。
只打印前缀/长度/是否含空格 等元信息，不打印实际 key 值。
也尝试一次 curl 调用看返回。

用法：
    cd demo
    python scripts/diag_alpaca_key.py
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    print(f"✅ .env loaded from {ROOT / '.env'}")
except ImportError:
    print("⚠️  python-dotenv 未安装")


def mask(s: str, head: int = 2, tail: int = 0) -> str:
    if not s:
        return "(empty)"
    if len(s) <= head + tail:
        return "*" * len(s)
    return s[:head] + "*" * (len(s) - head - tail) + (s[-tail:] if tail else "")


def diagnose(name: str, value: str, expected_prefix=None):
    raw = value or ""
    stripped = raw.strip()
    print(f"\n── {name} ──")
    if not raw:
        print(f"  ❌ 未设置")
        return None
    print(f"  长度（含空白）:    {len(raw)}")
    print(f"  长度（strip 后）:  {len(stripped)}")
    if len(raw) != len(stripped):
        print(f"  ⚠️  含前导/尾随空白！这会导致 401。请去 .env 删掉多余空格/换行")
    print(f"  前缀（脱敏）:     {mask(stripped, head=2)}")
    print(f"  是否含空格:       {' ' in stripped}")
    print(f"  是否含换行:       {chr(10) in stripped or chr(13) in stripped}")
    if expected_prefix:
        ok = stripped.startswith(expected_prefix)
        flag = "✅" if ok else "❌"
        print(f"  期望前缀 {expected_prefix}: {flag}")
        if not ok and stripped[:2].upper() in ("CK", "AK"):
            actual = stripped[:2].upper()
            if actual == "CK":
                print("     → 这是 Broker API 凭证，不是 Trading API。")
                print("     → 去 https://app.alpaca.markets/ 重新生成 PK 开头的 key")
            elif actual == "AK":
                print("     → 这是 Live Trading key（真金白银）。")
                print("     → 我们要的是 Paper Trading 的 PK key。")
                print("     → 或者把 ALPACA_BASE_URL 改成 https://api.alpaca.markets")
    return stripped


key = diagnose("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""),
                expected_prefix="PK")
secret = diagnose("ALPACA_API_SECRET", os.getenv("ALPACA_API_SECRET", ""))
base = diagnose("ALPACA_BASE_URL", os.getenv("ALPACA_BASE_URL", ""))

if not (key and secret and base):
    print("\n❌ 凭证不完整，先补齐再继续")
    sys.exit(1)

# 一致性检查
print("\n── 一致性检查 ──")
is_paper_url = "paper-api.alpaca.markets" in base
is_paper_key = key.startswith("PK")
is_live_key = key.startswith("AK")
if is_paper_url and is_live_key:
    print("  ❌ Live key (AK...) 配了 paper URL → 401")
    print("     修正：把 ALPACA_BASE_URL 改成 https://api.alpaca.markets")
elif (not is_paper_url) and is_paper_key:
    print("  ❌ Paper key (PK...) 配了 Live URL → 401")
    print("     修正：把 ALPACA_BASE_URL 改成 https://paper-api.alpaca.markets")
elif is_paper_url and is_paper_key:
    print("  ✅ Paper URL + Paper key，配置匹配")
elif (not is_paper_url) and is_live_key:
    print("  ⚠️  Live URL + Live key（这是实盘，确认你真的想用真钱测试）")
else:
    print(f"  ⚠️  无法判断（key 前缀: {key[:2]}, url: {base}）")

# 直接 curl /v2/account
print("\n── 直连测试 /v2/account ──")
try:
    import urllib.request
    req = urllib.request.Request(
        base.rstrip("/") + "/v2/account",
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "User-Agent": "alpaca-diag/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        print(f"  ✅ 200 OK")
        print(f"  account_number: {mask(data.get('account_number',''), 2, 2)}")
        print(f"  status:         {data.get('status')}")
        print(f"  cash:           ${float(data.get('cash', 0)):,.2f}")
        print(f"  equity:         ${float(data.get('equity', 0)):,.2f}")
        print(f"  buying_power:   ${float(data.get('buying_power', 0)):,.2f}")
        print("\n  → 凭证 OK，再跑 test_alpaca_connect.py 应该也通了")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")[:300]
    print(f"  ❌ HTTP {e.code}: {body}")
    if e.code == 401:
        print("\n  排查清单：")
        print("    1. Key/Secret 是否复制完整（看上面长度对不对）")
        print("    2. Key/Secret 是否互换填错")
        print("    3. 是不是从 Trading API（不是 Broker / OAuth）页面生成的")
        print("    4. 该 Key 是否在 Dashboard 被 revoke 了")
        print("    5. PK key 配了 paper URL（已验证一致性 OK 也可能 dashboard 端禁用了）")
    elif e.code == 403:
        print("  → 403：账户可能未激活或被限制")
except urllib.error.URLError as e:
    print(f"  ❌ 网络错误：{e}")
    print("  → 国内直连 alpaca.markets 可能不稳定，试试挂代理")
except Exception as e:
    print(f"  ❌ 其他错误：{type(e).__name__}: {e}")
