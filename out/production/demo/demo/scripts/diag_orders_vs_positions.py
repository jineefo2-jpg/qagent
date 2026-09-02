"""
对账诊断：列出 Alpaca 那边真实的订单 + 持仓，看是否对得上。

用法：
    cd demo
    python3 scripts/diag_orders_vs_positions.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from brokers.registry import get_current_broker

b = get_current_broker(broker_type="alpaca")
if not b.is_configured():
    print("❌ Alpaca 凭证未配置")
    sys.exit(1)

print("=" * 70)
print("📋 最近 500 笔订单（所有状态，按时间倒序）")
print("=" * 70)
orders = b.list_orders(status="all", limit=500)
print(f"共拉到 {len(orders)} 笔订单")
if not orders:
    print("  （无订单）")
# 只完整展示前 30 笔，太多刷屏；NVDA/TSLA 相关的全列
target_syms = {"NVDA", "TSLA"}
shown = 0
for o in orders:
    is_target = o.symbol in target_syms
    is_filled = o.status.value in ("filled", "partially_filled")
    if not (is_target or is_filled or shown < 10):
        continue
    submitted = (o.submitted_at or "")[:19].replace("T", " ")
    price = f"@${o.limit_price:.2f}" if o.limit_price else "@MKT"
    tag = " ⭐" if is_target else ""
    print(f"  id={o.broker_order_id[:8]}  {submitted}  "
          f"{o.side.value:4}  {o.symbol:6}  qty={int(o.qty):>3} {price}  "
          f"→ status=[{o.status.value:<18}]  filled={o.filled_qty}/{o.qty}{tag}")
    shown += 1

print()
print("=" * 70)
print("📦 当前持仓")
print("=" * 70)
positions = b.list_positions()
if not positions:
    print("  （无持仓）")
for p in positions:
    print(f"  {p.symbol:6}  qty={p.qty}  avg=${p.avg_entry_price:.2f}  "
          f"市值=${p.market_value:.2f}  P&L={p.unrealized_pl_pct:+.2f}%")

print()
print("=" * 70)
print("🔍 一致性检查")
print("=" * 70)
# 按 symbol 累计 filled_qty (买 - 卖)
fills = {}
for o in orders:
    if o.status.value not in ("filled", "partially_filled"):
        continue
    sign = 1 if o.side.value == "buy" else -1
    fills[o.symbol] = fills.get(o.symbol, 0) + sign * float(o.filled_qty)

pos_map = {p.symbol: float(p.qty) for p in positions}
all_syms = sorted(set(list(fills.keys()) + list(pos_map.keys())))
for s in all_syms:
    fill_qty = fills.get(s, 0)
    pos_qty = pos_map.get(s, 0)
    flag = "✅" if abs(fill_qty - pos_qty) < 0.01 else "⚠️"
    print(f"  {flag} {s:6}  累计成交={fill_qty:>6}  当前持仓={pos_qty:>6}")
print()

# 提示常见误解
new_orders = [o for o in orders if o.status.value in ("new", "accepted", "pending_new")]
if new_orders:
    print("ℹ️  以下订单处于「待成交」状态（display: ⏳ 待成交，有撤单按钮，属正常）:")
    for o in new_orders:
        price = f"@${o.limit_price:.2f}" if o.limit_price else "@市价"
        print(f"     {o.symbol} {o.side.value} {int(o.qty)}股 {price} - {o.broker_order_id[:8]}")
    print("     → 限价单需要市价触达限价才会成交")
    print("     → 美股盘外时间（北京时间 04:00-21:30）也会一直待成交")
