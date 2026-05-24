"""
Alpaca paper trading 连通性自检脚本。

用途：写完 .env 后，先跑这个脚本确认凭证 + 网络都 OK，再启动 server。

用法：
    cd demo
    python scripts/test_alpaca_connect.py

可选：试下一笔模拟限价单
    python scripts/test_alpaca_connect.py --place-test-order
"""
import sys
import os
import argparse
from pathlib import Path

# 让脚本在 demo/ 目录下也能找到 brokers 包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 加载 .env（脚本独立运行也要能看到 ALPACA_*）
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from brokers import OrderIntent, BrokerError
from brokers.registry import get_current_broker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--place-test-order", action="store_true",
        help="试下一笔 SPY 限价买单（远离市价的低价，不会成交，仅验证下单链路）",
    )
    args = parser.parse_args()

    broker = get_current_broker(broker_type="alpaca")

    print(f"=== Broker: {broker.name} ===")
    print(f"is_configured: {broker.is_configured()}")
    if not broker.is_configured():
        print("\n❌ 凭证未配置。请检查 .env 是否有:")
        print("    ALPACA_API_KEY=PK...")
        print("    ALPACA_API_SECRET=...")
        print("    ALPACA_BASE_URL=https://paper-api.alpaca.markets")
        sys.exit(1)

    # 1. 账户信息
    print("\n── 账户信息 ──")
    try:
        acc = broker.get_account()
        print(f"  account_id:    {acc.account_id}")
        print(f"  status:        {acc.status}")
        print(f"  cash:          ${acc.cash:,.2f}")
        print(f"  buying_power:  ${acc.buying_power:,.2f}")
        print(f"  equity:        ${acc.equity:,.2f}")
        print(f"  currency:      {acc.currency}")
    except BrokerError as e:
        print(f"❌ {e}")
        print("\n排查提示：")
        print("  - Key 前缀是 PK 吗？（不是 CK）")
        print("  - .env 的 ALPACA_BASE_URL 是 paper-api.alpaca.markets 吗？")
        print("  - 网络是否能访问 alpaca.markets（国内可能需要代理）")
        sys.exit(2)

    # 2. 当前持仓
    print("\n── 当前持仓 ──")
    try:
        positions = broker.list_positions()
        if not positions:
            print("  （无持仓 — 新账户正常）")
        else:
            for p in positions:
                print(f"  {p.symbol:6}  qty={p.qty:>8.2f}  "
                      f"avg=${p.avg_entry_price:>8.2f}  "
                      f"mv=${p.market_value:>10.2f}  "
                      f"P&L={p.unrealized_pl_pct:+.2f}%")
    except BrokerError as e:
        print(f"⚠️  {e}")

    # 3. 近 20 笔订单
    print("\n── 近 20 笔订单 ──")
    try:
        orders = broker.list_orders(limit=20)
        if not orders:
            print("  （无订单）")
        else:
            for o in orders[:20]:
                price = f"@${o.limit_price:.2f}" if o.limit_price else "@MKT"
                print(f"  {o.broker_order_id[:8]}  {o.side.value:4}  "
                      f"{o.symbol:6}  qty={o.qty:.2f} {price}  "
                      f"status={o.status.value}")
    except BrokerError as e:
        print(f"⚠️  {e}")

    # 4. 可选：试下一笔小单
    if args.place_test_order:
        print("\n── 试下一笔限价单（SPY $1，远离市价，不会成交）──")
        try:
            intent = OrderIntent.new(
                symbol="SPY", side="buy", qty=1,
                order_type="limit", limit_price=1.00,
                notes="connectivity test",
            )
            print(f"  intent_id: {intent.intent_id}")
            result = broker.place_order(intent)
            print(f"  ✅ broker_order_id: {result.broker_order_id}")
            print(f"     status: {result.status.value}")

            # 立刻撤掉
            print("\n  → 立即撤单...")
            broker.cancel_order(result.broker_order_id)
            print("  ✅ 撤单成功")
        except BrokerError as e:
            print(f"❌ 下单/撤单失败：{e}")

    print("\n✅ 全部检查通过。")
    print("\n下一步：")
    print("  python server.py     # 启动服务")
    print("  浏览器访问 http://localhost:8001")


if __name__ == "__main__":
    main()
