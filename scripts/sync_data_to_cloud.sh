#!/bin/zsh
# 把数据库与 RAG 库同步到云服务器（代码走 git，数据走 rsync —— 镜像里不带数据）。
# 用法: scripts/sync_data_to_cloud.sh user@云服务器IP [目标目录，默认 ~/quantagent]
# 幂等：rsync 只传差异；重建后重跑一次即可把新 derived 库补上去。
set -euo pipefail
HOST="${1:?用法: $0 user@host [远端目录]}"
DEST="${2:-quantagent}"
cd "$(cd "$(dirname "$0")/.." && pwd)"
ssh "$HOST" "mkdir -p $DEST/data $DEST/rag_db $DEST/out $DEST/logs"
# brokers.db 三件套必须整组同拷（SQLite 主库/-shm/-wal 是一体的，缺 wal 会丢未合并写入）；
# 里面是加密的券商绑定，云端还需 .env 里同一把 BROKER_KEK 才解得开
rsync -avz --progress \
  data/ashare_market.duckdb data/ashare_derived.duckdb data/ashare_ledger.duckdb \
  data/brokers.db data/brokers.db-shm data/brokers.db-wal data/rate_state.json \
  "$HOST:$DEST/data/"
rsync -avz --progress rag_db/ "$HOST:$DEST/rag_db/"
[ -d out/signals ] && rsync -avz out/signals "$HOST:$DEST/out/"
echo "✅ 数据同步完成。.env 请单独手工传（含密钥，不进脚本）："
echo "   scp .env $HOST:$DEST/.env"
