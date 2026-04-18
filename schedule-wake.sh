#!/bin/bash
# 毎朝 8:00 に実行し、16:10 まで Mac をアイドルスリープさせない
# caffeinate -i : アイドルスリープを抑制（ディスプレイを閉じた場合は別途 -d 不要）
# 8:00 → 16:10 = 487 分 = 29220 秒

# pmset schedule wake は root 権限が必要なため使用しない
# 代わりに caffeinate でアイドルスリープを一日通じて防止する

NOW=$(date "+%s")
TARGET=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) 16:10:00" "+%s")
WAIT=$(( TARGET - NOW ))

if [ "$WAIT" -gt 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') caffeinate 開始 (${WAIT}秒 = 16:10 まで)"
    # -s : システムスリープ防止（AC接続時に Sleep Service Back to Sleep も防ぐ）
    # -i : アイドルスリープ防止（バッテリー時のフォールバック）
    nohup /usr/bin/caffeinate -s -i -t "${WAIT}" > /dev/null 2>&1 &
    disown $!
    echo "$(date '+%Y-%m-%d %H:%M:%S') caffeinate PID=$! をバックグラウンドで起動"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') 16:10 を過ぎているため caffeinate スキップ"
fi
