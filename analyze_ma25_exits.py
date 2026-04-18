#!/usr/bin/env python3
import pickle, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS
from gridsearch_ma25recovery import preprocess, MAX_HOLD, RR, MIN_HISTORY

CACHE_PATH = Path(__file__).parent / "backtest_cache.pkl"

def classify_exits(df: pd.DataFrame):
    rows = []
    closes = df["Close"].values.astype(float)
    mask = df["rec_ret"].notna() & df["recovery"]
    for i in df.index[mask]:
        pos = df.index.get_loc(i)
        entry = df.at[i, "Open"] if pos + 1 < len(df) else np.nan  # next open
        # entry is already stored as next-open in preprocess
        # rec_ret is calculated but we need to classify why it exited
        # Re-derive entry/stop/take from preprocess columns
        atr = df.at[i, "ATR"]
        if pos + 1 >= len(df):
            continue
        e = df["Open"].iloc[pos + 1]
        if np.isnan(e) or e <= 0:
            continue
        s = max(e - atr * 2.0, e * 0.90)
        t = e + (e - s) * RR

        exit_type = "forced"
        exit_ret = np.nan
        for d in range(1, MAX_HOLD + 1):
            fi = pos + 1 + d
            if fi >= len(df):
                break
            c = closes[fi]
            if c <= s:
                exit_type = "stop"
                exit_ret = (c - e) / e * 100
                break
            elif c >= t:
                exit_type = "take"
                exit_ret = (c - e) / e * 100
                break
        else:
            last = pos + 1 + MAX_HOLD
            if last < len(df):
                exit_ret = (closes[last] - e) / e * 100

        rows.append({"type": exit_type, "ret": exit_ret})
    return rows

print("キャッシュ読み込み中...")
with open(CACHE_PATH, "rb") as f:
    raw_data = pickle.load(f)["data"]

print("前処理中...")
all_dfs = []
lock = threading.Lock()
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(preprocess, df): t for t, df in raw_data.items()}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            all_dfs.append(res)

print(f"有効銘柄: {len(all_dfs)}")
print("決済分類中...")
all_rows = []
for df in all_dfs:
    all_rows.extend(classify_exits(df))

df_exits = pd.DataFrame(all_rows).dropna(subset=["ret"])
print(f"\n総トレード数: {len(df_exits):,}")
print("=" * 55)
for etype, label in [("stop", "①損切り"), ("take", "②利確"), ("forced", "③強制終了(20日)")]:
    sub = df_exits[df_exits["type"] == etype]["ret"]
    if len(sub) == 0:
        print(f"{label}: 0件")
        continue
    print(f"{label}到達: {len(sub):,}件 ({len(sub)/len(df_exits)*100:.1f}%)")
    print(f"  平均損益: {sub.mean():.2f}%  中央値: {sub.median():.2f}%")
    print(f"  最小: {sub.min():.2f}%  最大: {sub.max():.2f}%")
print("=" * 55)
