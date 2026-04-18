#!/usr/bin/env python3
import pickle, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore")

from gridsearch_atr_squeeze import (
    preprocess_with_windows, run_window_grid, fetch_all_shares
)

CACHE_PATH = Path(__file__).parent / "backtest_cache.pkl"
WINDOWS = [3, 5, 7, 10, 15]

print("キャッシュ読み込み中...")
with open(CACHE_PATH, "rb") as f:
    raw_data = pickle.load(f)["data"]
print(f"  {len(raw_data)} 銘柄")

print("\n株数取得中...")
shares_dict = fetch_all_shares(list(raw_data.keys()))

print("\n前処理中（複数ATR窓）...")
all_dfs = []
done = 0
lock = threading.Lock()

def _proc(item):
    t, df = item
    return preprocess_with_windows(df, shares_dict.get(t), WINDOWS)

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(_proc, item): item[0] for item in raw_data.items()}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            all_dfs.append(res)
        with lock:
            done += 1
        if done % 500 == 0 or done == len(raw_data):
            print(f"  {done}/{len(raw_data)} 完了  有効: {len(all_dfs)}")

trading_days = max((len(df) - 200 for df in all_dfs), default=289)
print(f"推定取引日数: {trading_days} 日")

run_window_grid(all_dfs, trading_days)
