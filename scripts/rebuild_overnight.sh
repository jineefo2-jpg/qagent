#!/bin/zsh
# 过夜链（2026-08-31 行业成分修复后的一次性全量重建）：
#   ① 每日增量（拉当天 + 重拉参考表 → promote，行业覆盖 50.8%→100%）
#   ② 全量重建因子（首跑 --fresh 清旧 ~4.5h；标记文件在 = 上次中断过 → 续跑不清）
#   ③ 抽样取证（8 个日期逐位对比）
# 中断后重跑本脚本即续传。日志: logs/rebuild_overnight.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
LOG=logs/rebuild_overnight.log
MARK=data/.rebuild_in_progress
{
  echo "═══ $(date '+%F %T') 过夜链启动 ═══"
  zsh scripts/ashare_daily_update.sh || { echo "增量失败，停（IP/代理？）"; exit 1; }
  if [ -f "$MARK" ]; then
    echo "检测到中断标记 → 续跑（不清旧）"
    python3 scripts/rebuild_factors.py || { echo "重建续跑失败，可再次重跑本脚本"; exit 1; }
  else
    touch "$MARK"
    python3 scripts/rebuild_factors.py --fresh || { echo "重建失败，可重跑本脚本续传"; exit 1; }
  fi
  rm -f "$MARK"
  python3 scripts/verify_factor_store.py 8
  echo "═══ $(date '+%F %T') 过夜链完成 ═══"
} >> $LOG 2>&1
