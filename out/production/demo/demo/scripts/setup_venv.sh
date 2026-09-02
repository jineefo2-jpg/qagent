#!/usr/bin/env bash
# 本地开发：创建 .venv 并安装依赖（IntelliJ / PyCharm 选解释器时指向 .venv/bin/python）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
echo ""
echo "完成。解释器路径："
echo "  $ROOT/.venv/bin/python"
