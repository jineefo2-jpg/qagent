"""
检查 mock 模式下的订单为何没成交。

用法：
    python3 scripts/diag_mock_orders.py YOUR_USER_NS
    # YOUR_USER_NS 例如 "u:email:you@gmail.com" 或 "u:google:1234..."
    # 不传则用 default
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError: pass

import os
os.environ['BROKER_MODE'] = 'mock'

from brokers import get_broker
from quant_agent import _set_request_device_id, market_quote

ns = sys.argv[1] if len(sys.argv) > 1 else "default"
print(f"User namespace: {ns}\n")

_set_request_device_id(ns)
b = get_broker()

# 账户
acc = b.get_account()
print(f"账户:  cash=${acc.cash:,.2f}  equity=${acc.equity:,.2f}")

# 持仓
positions = b.list_positions()
print(f"持仓 ({len(positions)} 笔):")
for p in positions:
    print(f"  {p.symbol}  qty={p.qty}  avg=${p.avg_entry_price:.2f}  "
          f"mv=${p.market_value:.2f}  P&L={p.unrealized_pl_pct:+.2f}%")

# 全部订单 + 详细成交判断
orders = b.list_orders(status='all', limit=50)
print(f"\n订单 ({len(orders)} 笔):")
for o in orders:
    print(f"  {o.broker_order_id}  {o.side.value:4}  {o.symbol:6}  "
          f"qty={o.qty}  limit=${o.limit_price}  → status={o.status.value}")

    # 对 new 状态的单做诊断
    if o.status.value == 'new':
        print(f"     ─ 为什么没成交？")
        q = market_quote(o.symbol)
        if not (q and q.get('success')):
            print(f"     ⚠️  market_quote 失败: {q.get('error', 'unknown')}")
            continue
        cur = q.get('price')
        if cur is None:
            print(f"     ⚠️  market_quote 没返回 price 字段")
            continue
        print(f"     当前市价 ${cur:.2f}  vs  限价 ${o.limit_price:.2f}")
        if o.side.value == 'buy':
            if cur <= o.limit_price:
                print(f"     ✅ 限价 ≥ 市价 → 应该成交！下次查询会触发撮合")
                print(f"        （或者跑：python3 -c \"...\" 强制刷新）")
            else:
                print(f"     ❌ 限价 < 市价 → 不会成交（你出价低于市价，没人卖你）")
                print(f"        要成交：limit_price ≥ {cur:.2f}")
        else:  # sell
            if cur >= o.limit_price:
                print(f"     ✅ 市价 ≥ 限价 → 应该成交！")
            else:
                print(f"     ❌ 市价 < 限价 → 不会成交（你开价高于市价，没人愿意按你价买）")
                print(f"        要成交：limit_price ≤ {cur:.2f}")
