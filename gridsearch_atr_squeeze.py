#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATRスクイーズ型グリッドサーチ
固定: ATR収縮 / 売買代金3000万 / 翌日始値 / ATR×2.0(-10%) / RR1:1.5
グリッド: 出来高倍率 / RSI帯 / 時価総額上限
"""

import itertools
import pickle
import threading
import warnings
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS, DOW_N_SWINGS

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 200
MAX_HOLD    = 20
MAX_WORKERS = 8
RR          = 1.5
ATR_MULT    = 2.0
ATR_FLOOR   = 0.10

CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.5

GRID = {
    "vol_mult":   [2.0, 2.5, 3.0, 3.5],
    "rsi_lo":     [40.0, 45.0, 50.0, 55.0],
    "rsi_hi":     [55.0, 60.0, 65.0, 70.0],
    "mktcap_max": [10_000_000_000, 20_000_000_000, 30_000_000_000],
}


# ──────────────────────────────────────────────────────────────────────────────
# 株数取得
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        fi = yf.Ticker(ticker).fast_info
        sh = getattr(fi, "shares", None)
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t, sh = fut.result()
            if sh:
                result[t] = sh
            done += 1
            if done % 500 == 0 or done == len(tickers):
                print(f"  株数取得: {done}/{len(tickers)}  成功: {len(result)}")
    return result


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

    hit     = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + fhp + 1, 0, n - 1)]
    rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の前処理
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, shares: float | None) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 1:
        return None

    df = df_raw.copy()
    closes = df["Close"]

    df["RSI"] = calc_rsi(closes)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - closes.shift(1)).abs(),
        (df["Low"]  - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = df["Volume"].rolling(n_p).mean().shift(1)
    df["avg_to"]  = (closes * df["Volume"]).rolling(n_p).mean().shift(1)

    # ATR収縮（直近5日平均 < 前5日平均）
    atr5_now  = df["ATR"].rolling(5).mean()
    atr5_prev = df["ATR"].rolling(5).mean().shift(5)
    df["atr_shrink"] = (atr5_now < atr5_prev).fillna(False)

    # 時価総額
    df["mktcap"] = (closes * shares) if shares is not None else np.nan

    # ── 出口リターン事前計算（売買代金フィルタのみ）──────────────────────────
    n_rows  = len(df)
    c_arr   = closes.values.astype(float)
    o_arr   = df["Open"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    avg_tos = df["avg_to"].values.astype(float)

    base_valid = (
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(base_valid)[0]

    entries = np.full(n_rows, np.nan)
    stops   = np.full(n_rows, np.nan)
    takes   = np.full(n_rows, np.nan)

    e  = o_arr[vidx + 1]
    a  = atrs[vidx]
    ve = (e > 0) & (~np.isnan(e))
    vi2 = vidx[ve]; e2 = e[ve]; a2 = a[ve]
    s2  = np.maximum(e2 - a2 * ATR_MULT, e2 * (1 - ATR_FLOOR))
    t2  = e2 + (e2 - s2) * RR

    entries[vi2] = e2
    stops[vi2]   = s2
    takes[vi2]   = t2

    df["_ret"] = _exit_returns_vec(c_arr, entries, stops, takes)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# メトリクス
# ──────────────────────────────────────────────────────────────────────────────
def _metrics(rets: pd.Series, trading_days: int) -> dict:
    rets = rets.dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "spd": 0.0}
    w  = rets[rets > 0]
    l  = rets[rets <= 0]
    wr = len(w) / n * 100
    aw = w.mean() if len(w) > 0 else 0.0
    al = l.mean() if len(l) > 0 else 0.0
    pf = abs(aw / al) if al != 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "spd": round(n / trading_days, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# グリッドサーチ
# ──────────────────────────────────────────────────────────────────────────────
def run_grid(all_dfs: list[pd.DataFrame], trading_days: int) -> list[dict]:
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    combos = [c for c in combos if c[1] < c[2]]  # rsi_lo < rsi_hi
    print(f"有効な組み合わせ: {len(combos)} 通り")

    results = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        vm  = params["vol_mult"]
        lo  = params["rsi_lo"]
        hi  = params["rsi_hi"]
        cap = params["mktcap_max"]

        parts = []
        for df in all_dfs:
            mktcap_ok = df["mktcap"].isna() | (df["mktcap"] <= cap)
            mask = (
                df["atr_shrink"] &
                df["_ret"].notna() &
                (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                (df["avg_vol"] > 0) & (df["Volume"] >= df["avg_vol"] * vm) &
                (df["avg_to"] >= MIN_AVG_TURNOVER) &
                mktcap_ok
            )
            parts.append(df.loc[mask, "_ret"])

        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**params, **m})

        if ci % 30 == 0 or ci == len(combos):
            passed = sum(1 for r in results
                         if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                         and r["spd"] >= CRITERIA_SPD)
            print(f"  {ci:3d}/{len(combos)}: 合格 {passed} 件")

    return results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n株数取得中（yfinance）...")
    shares_dict = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中...")
    all_dfs: list[pd.DataFrame] = []
    done = 0
    lock = threading.Lock()

    def _proc(item):
        t, df = item
        return preprocess(df, shares_dict.get(t))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_proc, item): item[0] for item in raw_data.items()}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                all_dfs.append(res)
            with lock:
                done += 1
            if done % 500 == 0 or done == len(raw_data):
                print(f"  {done}/{len(raw_data)} 完了  有効: {len(all_dfs)}")

    trading_days = max(
        (len(df) - MIN_HISTORY for df in all_dfs if df is not None),
        default=289,
    )
    print(f"\n推定取引日数: {trading_days} 日")

    print("\nグリッドサーチ開始...")
    results = run_grid(all_dfs, trading_days)

    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]

    print("\n" + "=" * 75)
    print("【ATRスクイーズ型 グリッドサーチ結果】")
    print(f"  固定: ATR収縮 / 売買代金≥3000万 / 翌日始値 / ATR×{ATR_MULT}(-10%) / RR1:{RR}")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")
    print(f"  合格: {len(qualified)} / {len(results)} 通り")

    top_src = sorted(qualified, key=lambda r: r["pf"], reverse=True) if qualified else \
              sorted(results,   key=lambda r: r["pf"], reverse=True)
    label = "【合格・PF順】" if qualified else "【合格なし・PF順上位】"

    cap_label = {10_000_000_000: "100億", 20_000_000_000: "200億", 30_000_000_000: "300億"}

    print(f"\n  上位10件 {label}")
    print(f"  {'出来高':>6} {'RSI帯':<9} {'時価総額':>8} {'勝率':>7} {'PF':>6} {'件数':>6} {'件/日':>6}")
    print("  " + "-" * 60)
    for r in top_src[:10]:
        mark = "★" if (r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                        and r["spd"] >= CRITERIA_SPD) else "  "
        print(f"{mark} {r['vol_mult']:.1f}x  "
              f"RSI{r['rsi_lo']:.0f}〜{r['rsi_hi']:.0f}  "
              f"{cap_label[r['mktcap_max']]:>8}  "
              f"{r['wr']:>6.1f}%  "
              f"{r['pf']:>6.2f}  "
              f"{r['n']:>6,}  "
              f"{r['spd']:>6.2f}")

    if qualified:
        best = top_src[0]
        print(f"\n  ★ 最良パラメータ（PF最大）:")
        print(f"    出来高倍率  : {best['vol_mult']}x")
        print(f"    RSI帯      : {best['rsi_lo']:.0f} 〜 {best['rsi_hi']:.0f}")
        print(f"    時価総額上限: {cap_label[best['mktcap_max']]}")
        print(f"    勝率        : {best['wr']:.1f}%")
        print(f"    PF          : {best['pf']:.2f}")
        print(f"    シグナル    : {best['n']:,}件 ({best['spd']:.2f}/日)")
    print("=" * 75)


# ──────────────────────────────────────────────────────────────────────────────
# 追加: ATR収縮ウィンドウをグリッドに加えた拡張サーチ
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_with_windows(df_raw: pd.DataFrame, shares: float | None,
                             windows: list[int]) -> pd.DataFrame | None:
    """複数ATRウィンドウを一度に計算して列として保持する。"""
    if df_raw is None or len(df_raw) < MIN_HISTORY + 1:
        return None

    df = df_raw.copy()
    closes = df["Close"]

    df["RSI"] = calc_rsi(closes)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - closes.shift(1)).abs(),
        (df["Low"]  - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = df["Volume"].rolling(n_p).mean().shift(1)
    df["avg_to"]  = (closes * df["Volume"]).rolling(n_p).mean().shift(1)
    df["mktcap"]  = (closes * shares) if shares is not None else np.nan

    # 各ウィンドウのATR収縮フラグ
    for w in windows:
        now  = df["ATR"].rolling(w).mean()
        prev = df["ATR"].rolling(w).mean().shift(w)
        df[f"atr_shrink_{w}"] = (now < prev).fillna(False)

    # 出口リターン（売買代金フィルタのみ）
    n_rows  = len(df)
    c_arr   = closes.values.astype(float)
    o_arr   = df["Open"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    avg_tos = df["avg_to"].values.astype(float)

    base_valid = (
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(base_valid)[0]

    entries = np.full(n_rows, np.nan)
    stops   = np.full(n_rows, np.nan)
    takes   = np.full(n_rows, np.nan)

    e  = o_arr[vidx + 1]
    a  = atrs[vidx]
    ve = (e > 0) & (~np.isnan(e))
    vi2 = vidx[ve]; e2 = e[ve]; a2 = a[ve]
    s2  = np.maximum(e2 - a2 * ATR_MULT, e2 * (1 - ATR_FLOOR))
    t2  = e2 + (e2 - s2) * RR

    entries[vi2] = e2
    stops[vi2]   = s2
    takes[vi2]   = t2

    df["_ret"] = _exit_returns_vec(c_arr, entries, stops, takes)
    return df


def run_window_grid(all_dfs, trading_days):
    GRID2 = {
        "vol_mult":   [3.0, 3.5],
        "rsi_lo":     [45.0],
        "rsi_hi":     [55.0],
        "mktcap_max": [10_000_000_000, 20_000_000_000, 30_000_000_000],
        "atr_win":    [3, 5, 7, 10, 15],
    }
    cap_label = {10_000_000_000: "100億", 20_000_000_000: "200億", 30_000_000_000: "300億"}
    combos = list(itertools.product(*GRID2.values()))
    keys   = list(GRID2.keys())
    print(f"\n【ウィンドウ拡張サーチ】{len(combos)} 通り")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        vm, lo, hi, cap, w = (params["vol_mult"], params["rsi_lo"],
                               params["rsi_hi"], params["mktcap_max"], params["atr_win"])
        col = f"atr_shrink_{w}"
        parts = []
        for df in all_dfs:
            if col not in df.columns:
                continue
            mktcap_ok = df["mktcap"].isna() | (df["mktcap"] <= cap)
            mask = (
                df[col] &
                df["_ret"].notna() &
                (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                (df["avg_vol"] > 0) & (df["Volume"] >= df["avg_vol"] * vm) &
                (df["avg_to"] >= MIN_AVG_TURNOVER) &
                mktcap_ok
            )
            parts.append(df.loc[mask, "_ret"])
        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**params, **m})

    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]
    top = sorted(qualified, key=lambda r: r["pf"], reverse=True) if qualified else \
          sorted(results,   key=lambda r: r["pf"], reverse=True)

    print(f"  合格: {len(qualified)} / {len(results)} 通り\n")
    print(f"  {'出来高':>6} {'RSI帯':<9} {'時価総額':>8} {'ATR窓':>6} {'勝率':>7} {'PF':>6} {'件数':>6} {'件/日':>6}")
    print("  " + "-" * 65)
    for r in top[:15]:
        mark = "★" if (r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                        and r["spd"] >= CRITERIA_SPD) else "  "
        print(f"{mark} {r['vol_mult']:.1f}x  "
              f"RSI{r['rsi_lo']:.0f}〜{r['rsi_hi']:.0f}  "
              f"{cap_label[r['mktcap_max']]:>8}  "
              f"  {r['atr_win']:>2}日  "
              f"{r['wr']:>6.1f}%  "
              f"{r['pf']:>6.2f}  "
              f"{r['n']:>6,}  "
              f"{r['spd']:>6.2f}")
