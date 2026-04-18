#!/bin/bash
# 日本株スクリーナー セットアップスクリプト
# 初回のみ実行してください

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_NAME="com.user.stock-screener.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "========================================"
echo "  日本株スクリーナー セットアップ"
echo "========================================"
echo ""

# ── 1. 仮想環境の作成 ──────────────────────────────────────────────────────────
echo "[1/4] 仮想環境を作成中..."
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo "      → venv が既に存在します。スキップします。"
else
    python3 -m venv venv
    echo "      → venv を作成しました。"
fi

# ── 2. 依存パッケージのインストール ────────────────────────────────────────────
echo "[2/4] 依存パッケージをインストール中..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "      → インストール完了。"

# ── 3. .env ファイルの準備 ──────────────────────────────────────────────────────
echo "[3/4] 環境変数ファイルを確認中..."

if [ -f ".env" ]; then
    echo "      → .env が既に存在します。スキップします。"
else
    cp .env.example .env
    echo ""
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  .env.example を .env にコピーしました。"
    echo ""
    echo "  LINE Notify トークンを設定してください:"
    echo "    1. https://notify-bot.line.me/my/ にアクセス"
    echo "    2. 「トークンを発行する」をクリック"
    echo "    3. トークンをコピーして .env に貼り付け:"
    echo "       LINE_NOTIFY_TOKEN=コピーしたトークン"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -rp "  .env を今すぐ編集しますか？ [y/N]: " EDIT_ENV
    if [[ "$EDIT_ENV" =~ ^[Yy]$ ]]; then
        "${EDITOR:-nano}" .env
    fi
fi

# ── 4. launchd への登録（毎日 16:00 自動実行）─────────────────────────────────
echo "[4/4] LaunchAgent を登録中..."

chmod +x run.sh

mkdir -p "$LAUNCH_AGENTS_DIR"

PLIST_DEST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

if launchctl list | grep -q "com.user.stock-screener" 2>/dev/null; then
    echo "      → 既に登録済みです。一度アンロードして再登録します..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

cp "$SCRIPT_DIR/$PLIST_NAME" "$PLIST_DEST"
launchctl load "$PLIST_DEST"
echo "      → LaunchAgent を登録しました（毎日 16:00 に自動実行）。"

# ── 5. 自動ウェイクアップ設定（スリープ中でも 16:00 に実行できるよう 15:55 にウェイク）──
echo "[5/5] 自動ウェイクアップを設定中..."
echo "      → 毎日 15:55 に Mac を自動起動するため、管理者パスワードが必要です。"
if sudo pmset repeat wakeorpoweron MTWRFSU 15:55:00; then
    echo "      → 毎日 15:55 に自動ウェイクを設定しました。"
else
    echo ""
    echo "  ⚠ 自動ウェイクの設定に失敗しました。手動で以下を実行してください:"
    echo "    sudo pmset repeat wakeorpoweron MTWRFSU 15:55:00"
    echo ""
fi

# ── 完了 ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  セットアップ完了！"
echo "========================================"
echo ""
echo "  【動作確認】今すぐスクリーニングを実行:"
echo "    cd $SCRIPT_DIR"
echo "    source venv/bin/activate"
echo "    python stock_screener.py --now"
echo ""
echo "  【LaunchAgent 操作】"
echo "    停止 : launchctl unload $PLIST_DEST"
echo "    再開 : launchctl load   $PLIST_DEST"
echo "    確認 : launchctl list | grep stock-screener"
echo ""
echo "  【ログ確認】"
echo "    tail -f $SCRIPT_DIR/screener.log"
echo ""
