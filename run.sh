#!/bin/bash
# 日本株スクリーナー 実行ラッパー
# launchd / cron から呼び出されるスクリプト

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 多重起動防止
# ロックに PID を書き、そのプロセスが既に存在しなければ（強制終了・スリープ中の電源断等で
# trap が走らず残ったロック）削除して続行する。以前は残ったロックで以降の実行が全てスキップされた。
LOCKFILE="$SCRIPT_DIR/.screener.lock"
if [ -f "$LOCKFILE" ]; then
    OLD_PID="$(cat "$LOCKFILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') 既に実行中のプロセスがあります (PID=$OLD_PID)。スキップします。"
        exit 0
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') 古いロックファイルを削除します (PID=${OLD_PID:-不明})"
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# 仮想環境を有効化
source "$SCRIPT_DIR/venv/bin/activate"

# スクリーニング実行（即時モード）
# caffeinate -i: 実行中はアイドルスリープを抑制する
caffeinate -i python stock_screener.py --now
