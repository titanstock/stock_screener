#!/usr/bin/env python3
import pickle, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS
from backtest_atr_squeeze import (
    preprocess, fetch_all_shares,
    _exit_returns_vec, _exit_types_vec,
    MAX_HOLD, ATR_MULT, ATR_FLOOR,
    VOL_MULT, RSI_LO, RSI_HI, MKTCAP_MAX,
)

CACHE_PATH = Path(__file__).parent / "backtest_cache.pkl"
RR = 2.0

print("キャッシュ読み込み中...")
with open(CACHE_PATH, "rb") as f:
    raw_data = pickle.load(f)["data"]

print("株数取得中...")
shares_dict = fetch_all_shares(list(raw_data.keys()))

print("前処理中...")
all_dfs = []
lock = threading.Lock()

def _process(item):
    ticker, df = item
    return preprocess(df, shares_dict.get(ticker))

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(_process, item): item[0] for item in raw_data.items()}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            all_dfs.append(res)

print(f"有効銘柄: {len(all_dfs)}")

# RR1:2で決済計算
all_rets, all_types = [], []
for df in all_dfs:
    e_arr = df["_entry"].values.astype(float)
    s_arr = df["_stop"].values.astype(float)
    c_arr = df["Close"].values.astype(float)
    t_arr = np.where(
        ~np.isnan(e_arr) & ~np.isnan(s_arr),
        e_arr + (e_arr - s_arr) * RR,
        np.nan,
    )
    rets  = _exit_returns_vec(c_arr, e_arr, s_arr, t_arr)
    types = _exit_types_vec(c_arr, e_arr, s_arr, t_arr)
    mask  = ~np.isnan(rets)
    all_rets.append(pd.Series(rets[mask]))
    all_types.append(pd.Series(types[mask]))

rets_s  = pd.concat(all_rets,  ignore_index=True)
types_s = pd.concat(all_types, ignore_index=True)

stop_rets  = rets_s[types_s == 1]
take_rets  = rets_s[types_s == 2]
force_rets = rets_s[types_s == 0]

print(f"\n{'='*55}")
print(f"【ATRスクイーズ型 RR1:2 決済内訳】  総トレード: {len(rets_s):,}件")
print(f"{'='*55}")
for label, sub in [("①損切り", stop_rets), ("②利確", take_rets), ("③強制終了(20日)", force_rets)]:
    n = len(sub)
    pct = n / len(rets_s) * 100
    print(f"\n{label}: {n:,}件 ({pct:.1f}%)")
    if n > 0:
        print(f"  平均損益: {sub.mean():+.2f}%  中央値: {sub.median():+.2f}%")
        print(f"  最小: {sub.min():.2f}%  最大: {sub.max():.2f}%")

# 強制終了の損益分布
print(f"\n{'='*55}")
print("【強制終了 損益分布】")
bins = [-50, -10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10, 50]
labels = ["<-10%", "-10〜-7%", "-7〜-5%", "-5〜-3%", "-3〜-1%",
          "-1〜0%", "0〜+1%", "+1〜+3%", "+3〜+5%", "+5〜+7%", "+7〜+10%", ">+10%"]
cut = pd.cut(force_rets, bins=bins, labels=labels)
dist = cut.value_counts().sort_index()
total_force = len(force_rets)
print(f"  {'損益帯':<12} {'件数':>6}  {'割合':>7}  {'累積':>7}")
print("  " + "-" * 38)
cumsum = 0
for label_i, cnt in dist.items():
    cumsum += cnt
    bar = "█" * int(cnt / total_force * 40)
    print(f"  {label_i:<12} {cnt:>6,}  {cnt/total_force*100:>6.1f}%  {cumsum/total_force*100:>6.1f}%  {bar}")
print(f"\n  損益プラス: {(force_rets > 0).sum():,}件 ({(force_rets > 0).mean()*100:.1f}%)")
print(f"  損益マイナス: {(force_rets <= 0).sum():,}件 ({(force_rets <= 0).mean()*100:.1f}%)")
print(f"  ±1%以内: {((force_rets > -1) & (force_rets < 1)).sum():,}件 ({((force_rets > -1) & (force_rets < 1)).mean()*100:.1f}%)")
print("="*55)
