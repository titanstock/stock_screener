#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フルスキャン型グリッドサーチ（新型探索）
22種の指標フラグを総当たり（出来高必須 + 追加1〜2条件）
固定決済: 翌日始値 / ATR×2.0(-10%) / RR1:1.5 / 最大20日
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
)

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 200
MAX_HOLD    = 20
MAX_WORKERS = 8
RR          = 1.5
ATR_MULT    = 2.0
ATR_FLOOR   = 0.10   # 損切り上限 -10%

CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.3   # 件/日（新型なので緩め）


# ──────────────────────────────────────────────────────────────────────────────
# 週足ダウ理論
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

    hit     = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + fhp + 1, 0, n - 1)]
    rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の前処理（全フラグ生成）
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 1:
        return None

    df = df_raw.copy()
    closes = df["Close"]

    # ── 基本指標 ──────────────────────────────────────────────────────────────
    df["MA5"]  = closes.rolling(5).mean()
    df["MA25"] = closes.rolling(25).mean()
    df["MA75"] = closes.rolling(75).mean()
    ms, ss = calc_macd(closes)
    df["MACD"], df["SIG"] = ms, ss
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

    # ── ボリンジャーバンド ────────────────────────────────────────────────────
    mid = closes.rolling(20).mean()
    std = closes.rolling(20).std()
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std
    df["BB_width"]     = (df["BB_upper"] - df["BB_lower"]) / mid
    df["BB_width_avg"] = df["BB_width"].rolling(20).mean().shift(1)

    # ── ATR ──────────────────────────────────────────────────────────────────
    df["ATR_avg"] = df["ATR"].rolling(20).mean().shift(1)

    # ── 週足 ──────────────────────────────────────────────────────────────────
    wc    = closes.resample("W").last()
    wma25 = wc.rolling(25).mean()
    df["MA25W"]    = wma25.reindex(df.index, method="ffill")
    df["MA25W_4w"] = wma25.shift(4).reindex(df.index, method="ffill")
    prev_wc    = wc.shift(1).reindex(df.index, method="ffill")
    prev_wma25 = wma25.shift(1).reindex(df.index, method="ffill")

    # ── 高値 ──────────────────────────────────────────────────────────────────
    df["high5"]   = df["High"].rolling(5).max().shift(1)
    df["high10"]  = df["High"].rolling(10).max().shift(1)
    df["high20"]  = df["High"].rolling(20).max().shift(1)
    df["high52w"] = df["High"].rolling(250).max().shift(1)

    # ── 週足ダウ理論 ──────────────────────────────────────────────────────────
    try:
        df["dow_up"] = _weekly_uptrend_series(df)
    except Exception:
        df["dow_up"] = False

    # ══════════════════════════════════════════════════════════════════════════
    # フラグ定義（全て当日の情報のみ使用、先読みなし）
    # ══════════════════════════════════════════════════════════════════════════
    rsi  = df["RSI"]
    avol = df["avg_vol"]

    # ── RSIゾーン（排他的範囲）────────────────────────────────────────────────
    df["f_rsi_over"]   = rsi <= 30                    # 過売われ（RSI≤30）
    df["f_rsi_lo"]     = (rsi > 30) & (rsi <= 45)    # 売られ気味（30〜45）
    df["f_rsi_mid"]    = (rsi > 45) & (rsi <= 55)    # 中立（45〜55）
    df["f_rsi_hi"]     = (rsi > 55) & (rsi <= 70)    # 勢い（55〜70）
    df["f_rsi_hot"]    = rsi > 70                     # 過熱（RSI>70）

    # ── MA関連 ────────────────────────────────────────────────────────────────
    df["f_above_ma25"]  = closes >= df["MA25"]
    df["f_below_ma25"]  = closes < df["MA25"]
    df["f_ma25_cross"]  = (closes >= df["MA25"]) & (closes.shift(1) < df["MA25"].shift(1))   # MA25上抜け
    df["f_ma_align"]    = (df["MA5"] > df["MA25"]) & (df["MA25"] > df["MA75"])               # パーフェクトオーダー
    df["f_ma25_rising"] = df["MA25"] > df["MA25"].shift(5)                                   # MA25が5日前より上昇
    df["f_ma25w_up"]    = df["MA25W"] > df["MA25W_4w"]                                       # 週足MA25上昇
    df["f_ma25w_recov"] = (prev_wc <= prev_wma25) & (closes >= df["MA25W"])                  # 週足MA25回復

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["f_macd_cross"] = (df["MACD"] >= df["SIG"]) & (df["MACD"].shift(1) < df["SIG"].shift(1))  # ゴールデンクロス
    df["f_macd_plus"]  = df["MACD"] >= df["SIG"]                                                  # MACD＞シグナル

    # ── ブレイクアウト ────────────────────────────────────────────────────────
    df["f_bo5"]      = closes > df["high5"]           # 5日高値更新
    df["f_bo10"]     = closes > df["high10"]          # 10日高値更新
    df["f_bo20"]     = closes > df["high20"]          # 20日高値更新
    df["f_near52w"]  = closes >= df["high52w"] * 0.95 # 52週高値の95%以上（高値圏）

    # ── ボリンジャー ──────────────────────────────────────────────────────────
    df["f_bb_squeeze"] = df["BB_width"] < df["BB_width_avg"]  # バンド収縮（スクイーズ）
    df["f_bb_break"]   = closes > df["BB_upper"]              # 上方バンドブレイク
    df["f_bb_inside"]  = (closes <= df["BB_upper"]) & (closes >= df["BB_lower"])  # バンド内

    # ── ボラティリティ収縮 ────────────────────────────────────────────────────
    df["f_atr_shrink"] = df["ATR"] < df["ATR_avg"]    # ATR縮小（低ボラ）

    # ── ダウ理論 ──────────────────────────────────────────────────────────────
    df["f_dow_up"] = df["dow_up"].astype(bool)

    # ── 出来高倍率（必須候補）────────────────────────────────────────────────
    df["f_vol12x"] = (df["Volume"] >= avol * 1.2) & (avol > 0)
    df["f_vol15x"] = (df["Volume"] >= avol * 1.5) & (avol > 0)
    df["f_vol20x"] = (df["Volume"] >= avol * 2.0) & (avol > 0)
    df["f_vol30x"] = (df["Volume"] >= avol * 3.0) & (avol > 0)

    # ── 全フラグをboolに統一 ──────────────────────────────────────────────────
    flag_cols = [c for c in df.columns if c.startswith("f_")]
    for c in flag_cols:
        df[c] = df[c].fillna(False).astype(bool)

    # ══════════════════════════════════════════════════════════════════════════
    # 出口リターン事前計算（売買代金フィルタのみ）
    # ══════════════════════════════════════════════════════════════════════════
    n_rows  = len(df)
    c_arr   = df["Close"].values.astype(float)
    o_arr   = df["Open"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    avg_tos = df["avg_to"].values.astype(float)

    valid = (
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(valid)[0]

    entries = np.full(n_rows, np.nan)
    stops   = np.full(n_rows, np.nan)
    takes   = np.full(n_rows, np.nan)

    e  = o_arr[vidx + 1]
    a  = atrs[vidx]
    ve = (e > 0) & (~np.isnan(e))
    vi2 = vidx[ve]
    e2  = e[ve];  a2 = a[ve]
    s2  = np.maximum(e2 - a2 * ATR_MULT, e2 * (1 - ATR_FLOOR))
    t2  = e2 + (e2 - s2) * RR

    entries[vi2] = e2
    stops[vi2]   = s2
    takes[vi2]   = t2

    df["_ret"]   = _exit_returns_vec(c_arr, entries, stops, takes)
    df["_valid"] = ~np.isnan(df["_ret"])

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
# 出来高フラグ（必須 1 つ選択）
VOL_FLAGS = ["f_vol12x", "f_vol15x", "f_vol20x", "f_vol30x"]

# シグナルフラグ（追加条件 1〜2 個の組み合わせ）
SIGNAL_FLAGS = [
    "f_rsi_over", "f_rsi_lo", "f_rsi_mid", "f_rsi_hi", "f_rsi_hot",
    "f_above_ma25", "f_below_ma25", "f_ma25_cross", "f_ma_align",
    "f_ma25_rising", "f_ma25w_up", "f_ma25w_recov",
    "f_macd_cross", "f_macd_plus",
    "f_bo5", "f_bo10", "f_bo20", "f_near52w",
    "f_bb_squeeze", "f_bb_break", "f_bb_inside", "f_atr_shrink",
    "f_dow_up",
]


def _build_combos():
    # (vol) × (1シグナル)
    c2 = [(v, s) for v in VOL_FLAGS for s in SIGNAL_FLAGS]
    # (vol) × (2シグナル)
    c3 = [(v, a, b)
          for v in VOL_FLAGS
          for a, b in itertools.combinations(SIGNAL_FLAGS, 2)]
    return c2 + c3


def run_fullscan(all_dfs: list[pd.DataFrame], trading_days: int) -> list[dict]:
    combos = _build_combos()
    print(f"総組み合わせ: {len(combos):,} 通り "
          f"(2条件: {len(VOL_FLAGS)*len(SIGNAL_FLAGS)}, "
          f"3条件: {len(VOL_FLAGS)*len(list(itertools.combinations(SIGNAL_FLAGS,2)))})")

    results = []
    qualified_count = 0

    for ci, combo in enumerate(combos, 1):
        parts = []
        for df in all_dfs:
            mask = df["_valid"].copy()
            for flag in combo:
                mask = mask & df[flag]
            parts.append(df.loc[mask, "_ret"])

        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)

        passed = (m["wr"] >= CRITERIA_WR and m["pf"] >= CRITERIA_PF
                  and m["spd"] >= CRITERIA_SPD)
        if passed:
            qualified_count += 1

        results.append({"combo": combo, "passed": passed, **m})

        if ci % 500 == 0 or ci == len(combos):
            print(f"  {ci:5d}/{len(combos)}: 合格 {qualified_count} 件")

    return results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n前処理中（全フラグ計算）...")
    all_dfs: list[pd.DataFrame] = []
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(preprocess, df): t for t, df in raw_data.items()}
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
        default=250,
    )
    print(f"\n推定取引日数: {trading_days} 日")

    print("\nグリッドサーチ開始...")
    results = run_fullscan(all_dfs, trading_days)

    qualified = [r for r in results if r["passed"]]

    print("\n" + "=" * 80)
    print("【フルスキャン グリッドサーチ結果】")
    print(f"  固定決済: 翌日始値 / ATR×{ATR_MULT}(-{ATR_FLOOR*100:.0f}%) / RR1:{RR} / 最大{MAX_HOLD}日")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")
    print(f"  合格: {len(qualified)} / {len(results):,} 通り")

    # 総合スコア: WR × PF（高い方がいい）
    def score(r):
        return r["wr"] / 100 * r["pf"] * min(r["spd"] / 1.0, 1.0)

    top_src = sorted(qualified, key=score, reverse=True) if qualified else \
              sorted(results, key=lambda r: r["pf"], reverse=True)
    label = "【合格・総合スコア順】" if qualified else "【合格なし・PF順上位】"

    print(f"\n  上位30件 {label}")
    print(f"  {'条件（出来高 & シグナル）':<55} {'勝率':>7} {'PF':>6} {'件数':>6} {'件/日':>6}")
    print("  " + "-" * 82)
    for r in top_src[:30]:
        cname = " & ".join(c.replace("f_", "") for c in r["combo"])
        mark  = "★" if r["passed"] else "  "
        print(f"{mark} {cname:<55} {r['wr']:>6.1f}% {r['pf']:>6.2f} {r['n']:>6,} {r['spd']:>6.2f}")
    print("=" * 80)

    # 合格品を条件別に集計
    if qualified:
        print("\n【合格条件の傾向分析】")
        from collections import Counter
        flag_counter: Counter = Counter()
        for r in qualified:
            for f in r["combo"]:
                flag_counter[f] += 1
        print("  よく登場するフラグ（上位10）:")
        for flag, cnt in flag_counter.most_common(10):
            print(f"    {flag.replace('f_',''):<25} {cnt:3d} 件")
