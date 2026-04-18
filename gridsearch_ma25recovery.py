#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA25回復型グリッドサーチ
固定条件: 先週終値≤MA25W → 今週終値≥MA25W / 売買代金3000万 / 翌日始値 / ATR×2.0(-10%) / RR1:1.5
グリッド: RSI下限・上限 / 出来高倍率 / 週足ダウ理論あり・なし
"""

import itertools
import operator
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stock_screener import (
    calc_rsi, calc_macd, MIN_AVG_TURNOVER, BREAKOUT_DAYS, DOW_N_SWINGS,
    fetch_jpx_stock_list,
)

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 200
MAX_HOLD    = 20
MAX_WORKERS = 8
RR          = 1.5

# ── グリッド定義 ─────────────────────────────────────────────────────────────
GRID = {
    "rsi_lo":   [40.0, 45.0, 50.0, 55.0],
    "rsi_hi":   [55.0, 60.0, 65.0, 70.0],
    "vol_mult": [1.2, 1.5, 2.0],
    "use_dow":  [True, False],
}

# 評価基準
CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# 週足ダウ理論（日次 Series）
# ──────────────────────────────────────────────────────────────────────────────
def _weekly_uptrend_series(df: pd.DataFrame, n: int = DOW_N_SWINGS) -> pd.Series:
    weekly = df.resample("W").agg({"High": "max", "Low": "min"}).dropna()
    result = pd.Series(False, index=weekly.index, dtype=bool)
    arr_h, arr_l = weekly["High"].values, weekly["Low"].values

    def _sw(arr, cmp):
        return [arr[i] for i in range(2, len(arr) - 2)
                if cmp(arr[i], arr[i-1]) and cmp(arr[i], arr[i-2])
                and cmp(arr[i], arr[i+1]) and cmp(arr[i], arr[i+2])]

    for i in range(4, len(weekly)):
        hs = _sw(arr_h[:i+1], operator.ge)
        ls = _sw(arr_l[:i+1], operator.le)
        if len(hs) >= n and len(ls) >= n:
            h, l = hs[-n:], ls[-n:]
            if (all(h[j] < h[j+1] for j in range(n-1)) and
                    all(l[j] < l[j+1] for j in range(n-1))):
                result.iloc[i] = True

    return result.reindex(df.index, method="ffill").fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# 出口リターン（ベクトル化）
# ──────────────────────────────────────────────────────────────────────────────
def _exit_returns_vec(closes: np.ndarray, entries: np.ndarray,
                      stops: np.ndarray, takes: np.ndarray) -> np.ndarray:
    n    = len(closes)
    rets = np.full(n, np.nan)
    valid = (~np.isnan(entries)) & (entries > 0)
    vidx  = np.where(valid)[0]
    if len(vidx) == 0:
        return rets

    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)

    hit      = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + fhp + 1, 0, n - 1)]
    rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の事前計算
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 1:
        return None

    df = df_raw.copy()
    df["MA25"] = df["Close"].rolling(25).mean()
    ms, ss = calc_macd(df["Close"])
    df["MACD"], df["SIG"] = ms, ss
    df["RSI"] = calc_rsi(df["Close"])
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = df["Volume"].rolling(n_p).mean().shift(1)
    df["avg_to"]  = (df["Close"] * df["Volume"]).rolling(n_p).mean().shift(1)

    # 週足MA25
    wc    = df["Close"].resample("W").last()
    wma25 = wc.rolling(25).mean()
    df["MA25W"] = wma25.reindex(df.index, method="ffill")

    # MA25回復フラグ（先週終値 ≤ 先週MA25W → 今日終値 ≥ 今日MA25W）
    prev_wc    = wc.shift(1).reindex(df.index, method="ffill")
    prev_wma25 = wma25.shift(1).reindex(df.index, method="ffill")
    df["recovery"] = (prev_wc <= prev_wma25) & (df["Close"] >= df["MA25W"])

    # 週足ダウ理論
    try:
        df["dow_up"] = _weekly_uptrend_series(df)
    except Exception:
        df["dow_up"] = False

    # 出口リターン事前計算（固定条件のもとでの翌日始値エントリー）
    n_rows  = len(df)
    closes  = df["Close"].values.astype(float)
    opens   = df["Open"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    avg_tos = df["avg_to"].values.astype(float)
    recovs  = df["recovery"].values.astype(bool)

    valid = (
        recovs &
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(valid)[0]

    entries = np.full(n_rows, np.nan)
    stops   = np.full(n_rows, np.nan)
    takes   = np.full(n_rows, np.nan)

    e = opens[vidx + 1]          # 翌日始値
    a = atrs[vidx]
    valid_e = (e > 0) & (~np.isnan(e))
    vidx2   = vidx[valid_e]
    e2      = e[valid_e]
    a2      = a[valid_e]
    s2      = np.maximum(e2 - a2 * 2.0, e2 * 0.90)
    t2      = e2 + (e2 - s2) * RR

    entries[vidx2] = e2
    stops[vidx2]   = s2
    takes[vidx2]   = t2

    df["rec_ret"] = _exit_returns_vec(closes, entries, stops, takes)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# メトリクス
# ──────────────────────────────────────────────────────────────────────────────
def _metrics(rets: pd.Series, trading_days: int) -> dict:
    rets = rets.dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "spd": 0.0}
    w   = rets[rets > 0]
    l   = rets[rets <= 0]
    wr  = len(w) / n * 100
    aw  = w.mean() if len(w) > 0 else 0.0
    al  = l.mean() if len(l) > 0 else 0.0
    pf  = abs(aw / al) if al != 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "spd": round(n / trading_days, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# グリッドサーチ
# ──────────────────────────────────────────────────────────────────────────────
def run_grid(all_dfs: list[pd.DataFrame], trading_days: int) -> list[dict]:
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    # RSI下限 < RSI上限 のみ有効
    combos = [c for c in combos if c[0] < c[1]]  # rsi_lo < rsi_hi
    print(f"有効な組み合わせ: {len(combos)} 通り")

    results = []
    for ci, combo in enumerate(combos, 1):
        params  = dict(zip(keys, combo))
        lo      = params["rsi_lo"]
        hi      = params["rsi_hi"]
        vm      = params["vol_mult"]
        use_dow = params["use_dow"]

        parts = []
        for df in all_dfs:
            mask = (
                df["recovery"] &
                df["rec_ret"].notna() &
                (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                (df["avg_vol"] > 0) & (df["Volume"] >= df["avg_vol"] * vm) &
                (df["avg_to"] >= MIN_AVG_TURNOVER)
            )
            if use_dow:
                mask = mask & df["dow_up"]
            parts.append(df.loc[mask, "rec_ret"])

        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**params, **m})

        if ci % 20 == 0 or ci == len(combos):
            print(f"  {ci:3d}/{len(combos)}: rsi={lo:.0f}-{hi:.0f} vol={vm}x "
                  f"dow={'✓' if use_dow else '✗'} "
                  f"→ n={m['n']:4d} WR={m['wr']:.1f}% PF={m['pf']:.2f} {m['spd']:.2f}/日")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュからデータ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print(f"\n特徴量を事前計算中...")
    all_dfs: list[pd.DataFrame] = []
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(preprocess, df): t for t, df in raw_data.items()}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                all_dfs.append(res)
            with lock:
                done += 1
            if done % 500 == 0 or done == len(raw_data):
                print(f"  {done}/{len(raw_data)} 完了  有効: {len(all_dfs)} 銘柄")

    trading_days = max(
        (len(df) - MIN_HISTORY for df in all_dfs if df is not None),
        default=250,
    )
    print(f"\n推定取引日数: {trading_days} 日")

    print("\nグリッドサーチ開始...")
    results = run_grid(all_dfs, trading_days)

    # 評価基準でフィルタ
    qualified = [
        r for r in results
        if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD
    ]

    print("\n" + "=" * 70)
    print("【MA25回復型 グリッドサーチ結果】")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")
    print(f"  合格: {len(qualified)} / {len(results)} 通り")

    top_src = sorted(qualified, key=lambda r: r["pf"], reverse=True) if qualified else \
              sorted(results,   key=lambda r: r["pf"], reverse=True)

    label = "【合格のみ・PF順】" if qualified else "【合格なし・PF順上位】"
    print(f"\n  上位5件 {label}")
    print(f"  {'RSI範囲':<12} {'出来高':<8} {'ダウ':<6} {'勝率':>7} {'PF':>6} {'件数':>6} {'件/日':>6}")
    print("  " + "-" * 56)
    for r in top_src[:5]:
        dow_mark = "あり" if r["use_dow"] else "なし"
        print(f"  {r['rsi_lo']:.0f}〜{r['rsi_hi']:.0f}{'':6} {r['vol_mult']}x{'':4} {dow_mark:<6} "
              f"{r['wr']:>6.1f}% {r['pf']:>6.2f} {r['n']:>6,} {r['spd']:>6.2f}")

    if qualified:
        best = top_src[0]
        print(f"\n  ★ 最良パラメータ:")
        print(f"    RSI        : {best['rsi_lo']:.0f} 〜 {best['rsi_hi']:.0f}")
        print(f"    出来高倍率 : {best['vol_mult']}x")
        print(f"    週足ダウ   : {'あり' if best['use_dow'] else 'なし'}")
        print(f"    勝率       : {best['wr']:.1f}%")
        print(f"    PF         : {best['pf']:.2f}")
        print(f"    シグナル   : {best['n']:,} 件 ({best['spd']:.2f}/日)")
    print("=" * 70)
