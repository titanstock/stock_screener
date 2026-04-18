#!/bin/bash
# 日本株スクリーナー 実行ラッパー
# launchd / cron から呼び出されるスクリプト

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 多重起動防止
LOCKFILE="$SCRIPT_DIR/.screener.lock"
if [ -f "$LOCKFILE" ]; then
    echo "既に実行中のプロセスがあります。スキップします。(lockfile: $LOCKFILE)"
    exit 0
fi
touch "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# 仮想環境を有効化
source "$SCRIPT_DIR/venv/bin/activate"

# スクリーニング実行（即時モード）
# caffeinate -i: 実行中はアイドルスリープを抑制する
caffeinate -i python stock_screener.py --now
