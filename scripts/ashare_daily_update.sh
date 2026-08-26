#!/bin/zsh
# A 股每日增量更新（launchd 每个工作日 21:30 调用，见 scripts/com.jineefo.ashare-daily.plist）。
# 21:30 的依据：日线/复权 ~15:30、daily_basic ~17-18 点、北向（港交所披露）~19-21 点，
# 全部落齐后再拉，避免半截数据触发校验拦截。非交易日 run_daily 自动 skip。
# TUSHARE_TOKEN 从 .env 读（pipeline 只认环境变量，这里 source 进来）。
set -euo pipefail
cd /Users/jineefo/Documents/AI-Agent/demo
set -a; source .env; set +a

if /usr/bin/python3 -m ashare.data.pipeline daily >> logs/ashare_daily.log 2>&1; then
  echo "$(date '+%F %T') daily OK" >> logs/ashare_daily.log
else
  echo "$(date '+%F %T') daily FAILED" >> logs/ashare_daily.log
  # 夜间静默失败是数据陈旧的头号来源 —— 桌面通知一声
  /usr/bin/osascript -e 'display notification "查看 logs/ashare_daily.log" with title "A 股每日更新失败"' || true
  exit 1
fi
