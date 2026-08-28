#!/bin/zsh
# A 股每日增量更新（launchd：工作日 21:30 定时 + 登录/开机即跑，见 com.jineefo.ashare-daily.plist）。
#
# 两个 pass，共同保证「怎么漏都补得回来」：
#   ① 补漏 pass（任意时刻安全）：K 线只补到昨天（--until）——昨天为止各源必然已落齐；
#      宏观 observed 可见日仍记真实今天（D4：不得回填）。电脑在 21:30 关机导致定时没跑，
#      下次开机/登录这一步就把欠账补齐；没有欠账时它是秒级 no-op。
#   ② 当日 pass（仅 21 点后）：21:30 的依据 —— 日线/复权 ~15:30、daily_basic ~17-18 点、
#      北向披露 ~19-21 点，全部落齐后再拉当日。
# TUSHARE_TOKEN 从 .env 单独取（不 source 全文，见下方注释）。
set -uo pipefail
cd /Users/jineefo/Documents/AI-Agent/demo
# 只取 TUSHARE_TOKEN，不 source 整个 .env（2026-08-28 实测：某行的值带空格时 source 会把
# 后半截当命令执行，脚本从此走偏；且每日更新没有理由持有券商/OAuth/邮箱密钥 —— 最小权限）
mkdir -p logs
TUSHARE_TOKEN=$(sed -n 's/^TUSHARE_TOKEN=//p' .env | tail -1 | tr -d '\r' | tr -d '"' | tr -d "'" | tr -d ' ')
export TUSHARE_TOKEN
if [ -z "${TUSHARE_TOKEN:-}" ]; then
  echo "$(date '+%F %T') .env 里读不到 TUSHARE_TOKEN，本次不跑" >> logs/ashare_daily.log
  exit 1
fi
LOG=logs/ashare_daily.log

notify_fail() {
  echo "$(date '+%F %T') $1 FAILED" >> $LOG
  /usr/bin/osascript -e 'display notification "查看 logs/ashare_daily.log" with title "A 股每日更新失败"' || true
}

YESTERDAY=$(date -v-1d +%F)
if /usr/bin/python3 -m ashare.data.pipeline daily --until $YESTERDAY >> $LOG 2>&1; then
  echo "$(date '+%F %T') catch-up(≤$YESTERDAY) OK" >> $LOG
else
  notify_fail "catch-up"; exit 1
fi

if [ "$(date +%H)" -ge 21 ]; then
  if /usr/bin/python3 -m ashare.data.pipeline daily >> $LOG 2>&1; then
    echo "$(date '+%F %T') daily OK" >> $LOG
  else
    notify_fail "daily"; exit 1
  fi
  # ③ 信号链（P3 Task 6）：非交易日/非周频调仓日在 python 侧安静跳过；
  #    调仓日 = 先 build 当日因子（V6 只 build 这一天）再出清单落 ledger + 导出 JSON。
  #    信号失败不影响已 promote 的数据，但要出声。
  if /usr/bin/python3 -m ashare.strategy.plan --nightly >> $LOG 2>&1; then
    echo "$(date '+%F %T') nightly-signal OK" >> $LOG
  else
    notify_fail "nightly-signal"; exit 1
  fi
fi
