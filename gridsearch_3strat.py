#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存3型グリッドサーチ（正しいバックテスト版）
baseline : 翌日安値≤終値×0.98 → 指値約定
breakout : 翌日始値 成行
pullback : MA25 指値 最大3日待ち
"""

import itertools, operator, pickle, threading, warnings
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

RR = {"baseline": 1.5, "breakout": 2.0, "pullback": 1.5}

CRITERIA_WR  = 50.0
CRITERIA_PF  = 1.4
CRITERIA_SPD = 0.5

GRID = {
    "baseline": {
        "vol_mult":   [1.5, 2.0, 2.5],
        "rsi_lo":     [45.0, 50.0, 55.0],
        "rsi_hi":     [65.0, 70.0, 75.0],
        "weekly_dev": [15, 20, 25, 30],
    },
    "breakout": {
        "vol_mult": [2.5, 3.0, 3.5],
        "rsi_lo":   [55.0, 60.0, 65.0],
    },
    "pullback": {
        "touch_pct": [1.01, 1.02, 1.03, 1.04],
        "rsi_lo":    [50.0, 55.0, 60.0],
        "rsi_hi":    [60.0, 65.0, 70.0],
    },
}


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
# 出口リターン（始値から：baseline / breakout）
# ──────────────────────────────────────────────────────────────────────────────
def _exit_vec(closes: np.ndarray, entries: np.ndarray,
              stops: np.ndarray, takes: np.ndarray,
              start_offset: int = 1) -> np.ndarray:
    """entry[i] が有効な各日について、close[i+start_offset] から出口を探す。"""
    n    = len(closes)
    rets = np.full(n, np.nan)
    valid = (~np.isnan(entries)) & (entries > 0)
    vidx  = np.where(valid)[0]
    if len(vidx) == 0:
        return rets

    raw_idx  = vidx[:, np.newaxis] + np.arange(start_offset, start_offset + MAX_HOLD)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)

    hit     = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + start_offset + fhp, 0, n - 1)]
    rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 出口リターン（pullback 用：可変スタートオフセット）
# ──────────────────────────────────────────────────────────────────────────────
def _exit_vec_pullback(closes: np.ndarray, entries: np.ndarray,
                       stops: np.ndarray, takes: np.ndarray,
                       fill_offsets: np.ndarray) -> np.ndarray:
    """fill_offsets[i] が 1〜3 の日について、close[i+fill_offsets[i]] から出口を探す。"""
    n    = len(closes)
    rets = np.full(n, np.nan)

    for k in [1, 2, 3]:
        vidx = np.where((fill_offsets == k) & (~np.isnan(entries)) & (entries > 0))[0]
        if len(vidx) == 0:
            continue

        raw_idx  = vidx[:, np.newaxis] + np.arange(k, k + MAX_HOLD)
        in_range = raw_idx < n
        safe_idx = np.where(in_range, raw_idx, n - 1)
        fut      = np.where(in_range, closes[safe_idx], np.nan)

        hit     = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
        has_hit = hit.any(axis=1)
        has_fut = in_range.any(axis=1)
        last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
        fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
        ep      = closes[np.clip(vidx + k + fhp, 0, n - 1)]
        rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)

    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の前処理
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 5:
        return None

    df = df_raw.copy()
    closes = df["Close"]
    opens  = df["Open"]
    lows   = df["Low"]

    # ── 基本指標 ──────────────────────────────────────────────────────────────
    df["MA25"] = closes.rolling(25).mean()
    ms, ss = calc_macd(closes)
    df["MACD"], df["SIG"] = ms, ss
    df["RSI"] = calc_rsi(closes)

    tr = pd.concat([
        df["High"] - lows,
        (df["High"] - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"]    = df["Volume"].rolling(n_p).mean().shift(1)
    df["avg_to"]     = (closes * df["Volume"]).rolling(n_p).mean().shift(1)
    df["past_high"]  = df["High"].rolling(n_p).max().shift(1)

    # ── 週足 ──────────────────────────────────────────────────────────────────
    wc    = closes.resample("W").last()
    wma25 = wc.rolling(25).mean()
    df["MA25W"] = wma25.reindex(df.index, method="ffill")

    try:
        df["dow_up"] = _weekly_uptrend_series(df)
    except Exception:
        df["dow_up"] = False

    # ── 固定フラグ（全型共通）─────────────────────────────────────────────────
    df["above_ma25"]  = closes > df["MA25"]
    df["above_ma25w"] = closes > df["MA25W"]
    df["macd_pos"]    = df["MACD"] > df["SIG"]
    df["to_ok"]       = df["avg_to"] >= MIN_AVG_TURNOVER

    # 週足MA25 乖離率列（全閾値）
    for wd in GRID["baseline"]["weekly_dev"]:
        df[f"within_w{wd}"] = df["MA25W"].isna() | (closes <= df["MA25W"] * (1 + wd / 100))

    # 出来高倍率列
    all_vms = sorted(set(GRID["baseline"]["vol_mult"]) | set(GRID["breakout"]["vol_mult"]))
    for vm in all_vms:
        df[f"vol_{vm}x"] = (df["avg_vol"] > 0) & (df["Volume"] >= df["avg_vol"] * vm)

    # MA25タッチ列（直近2日以内 close が MA25〜MA25×tp の範囲）
    for tp in GRID["pullback"]["touch_pct"]:
        col = f"touch_{int(tp*100)}"
        t   = (closes >= df["MA25"]) & (closes <= df["MA25"] * tp)
        p   = (closes.shift(1) >= df["MA25"].shift(1)) & (closes.shift(1) <= df["MA25"].shift(1) * tp)
        df[col] = t | p

    # ── 値固め（breakout 固定条件）────────────────────────────────────────────
    atr_rec   = df["ATR"].shift(1).rolling(5).mean()
    atr_prev5 = df["ATR"].shift(6).rolling(5).mean()
    df["atr_shrink"] = (atr_rec < atr_prev5) & (atr_prev5 > 0)

    near_hi = sum(
        ((closes.shift(k) >= df["past_high"] * 0.96) &
         (closes.shift(k) <= df["past_high"] * 1.02)).astype(int)
        for k in range(1, 6)
    )
    df["consol"] = df["atr_shrink"] & (near_hi >= 2)

    # ブレイクアウト固定フラグ
    df["bo_ok"]      = (closes > df["past_high"]) & ((closes - df["past_high"]) / df["past_high"] * 100 >= 1.0)
    df["change_ok"]  = closes.pct_change() * 100 >= 2.0
    df["recent_bo"]  = closes.shift(2) <= df["past_high"]   # 2日前終値がまだ高値以下

    # ── numpy 配列 ────────────────────────────────────────────────────────────
    n_rows  = len(df)
    c_arr   = closes.values.astype(float)
    o_arr   = opens.values.astype(float)
    l_arr   = lows.values.astype(float)
    ma25s   = df["MA25"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    avg_tos = df["avg_to"].values.astype(float)

    base_ok = (
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # baseline: 翌日安値 ≤ close×0.98 → 指値約定
    # ══════════════════════════════════════════════════════════════════════════
    limit_arr  = c_arr * 0.98
    next_low   = np.empty(n_rows); next_low[:] = np.nan
    next_low[:-1] = l_arr[1:]
    bl_filled = base_ok & (next_low <= limit_arr)
    vidx = np.where(bl_filled)[0]

    bl_entries = np.full(n_rows, np.nan)
    bl_stops   = np.full(n_rows, np.nan)
    bl_takes   = np.full(n_rows, np.nan)

    e = limit_arr[vidx]; a = atrs[vidx]
    s = np.maximum(e - a * 2.0, e * 0.90)
    t = e + (e - s) * RR["baseline"]
    bl_entries[vidx] = e; bl_stops[vidx] = s; bl_takes[vidx] = t

    df["baseline_ret"] = _exit_vec(c_arr, bl_entries, bl_stops, bl_takes, start_offset=1)

    # ══════════════════════════════════════════════════════════════════════════
    # breakout: 翌日始値 成行
    # ══════════════════════════════════════════════════════════════════════════
    vidx = np.where(base_ok)[0]
    e    = o_arr[vidx + 1]
    ve   = (e > 0) & (~np.isnan(e))
    vi2  = vidx[ve]; e2 = e[ve]; a2 = atrs[vi2]

    bo_entries = np.full(n_rows, np.nan)
    bo_stops   = np.full(n_rows, np.nan)
    bo_takes   = np.full(n_rows, np.nan)
    s2 = np.maximum(e2 - a2 * 2.0, e2 * 0.90)
    t2 = e2 + (e2 - s2) * RR["breakout"]
    bo_entries[vi2] = e2; bo_stops[vi2] = s2; bo_takes[vi2] = t2

    df["breakout_ret"] = _exit_vec(c_arr, bo_entries, bo_stops, bo_takes, start_offset=1)

    # ══════════════════════════════════════════════════════════════════════════
    # pullback: MA25 指値 最大3日待ち
    # ══════════════════════════════════════════════════════════════════════════
    pb_fill    = np.zeros(n_rows, dtype=int)   # 0=未約定, 1/2/3=約定オフセット
    pb_entries = np.full(n_rows, np.nan)
    pb_stops   = np.full(n_rows, np.nan)
    pb_takes   = np.full(n_rows, np.nan)

    for k in [1, 2, 3]:
        can_k  = base_ok & (np.arange(n_rows) + k < n_rows)
        unfilled = pb_fill == 0
        next_lo_k = np.empty(n_rows); next_lo_k[:] = np.nan
        next_lo_k[:-k] = l_arr[k:]
        fills_k = can_k & unfilled & (next_lo_k <= ma25s)
        vidx_k  = np.where(fills_k)[0]
        pb_fill[vidx_k] = k

        e_k = ma25s[vidx_k]
        a_k = atrs[vidx_k]
        s_k = np.maximum(e_k - a_k * 2.0, e_k * 0.90)
        t_k = e_k + (e_k - s_k) * RR["pullback"]
        pb_entries[vidx_k] = e_k
        pb_stops[vidx_k]   = s_k
        pb_takes[vidx_k]   = t_k

    df["pullback_ret"]    = _exit_vec_pullback(c_arr, pb_entries, pb_stops, pb_takes, pb_fill)
    df["pb_fill_offset"]  = pb_fill

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
def run_baseline(all_dfs, trading_days):
    g = GRID["baseline"]
    combos = [c for c in itertools.product(*g.values()) if c[1] < c[2]]   # rsi_lo < rsi_hi
    keys   = list(g.keys())
    print(f"\n【ベースライン型】 {len(combos)} 通り")
    results = []
    for ci, combo in enumerate(combos, 1):
        p  = dict(zip(keys, combo))
        vm, lo, hi, wd = p["vol_mult"], p["rsi_lo"], p["rsi_hi"], p["weekly_dev"]
        parts = []
        for df in all_dfs:
            mask = (
                df["above_ma25"] & df["above_ma25w"] & df["dow_up"].astype(bool) &
                df["macd_pos"] & df["to_ok"] &
                df[f"vol_{vm}x"] & df[f"within_w{wd}"] &
                (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                df["baseline_ret"].notna()
            )
            parts.append(df.loc[mask, "baseline_ret"])
        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**p, **m})
        if ci % 30 == 0 or ci == len(combos):
            passed = sum(1 for r in results if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD)
            print(f"  {ci:3d}/{len(combos)}: 合格 {passed} 件")
    return results


def run_breakout(all_dfs, trading_days):
    g = GRID["breakout"]
    combos = list(itertools.product(*g.values()))
    keys   = list(g.keys())
    print(f"\n【ブレイクアウト型】 {len(combos)} 通り")
    results = []
    for ci, combo in enumerate(combos, 1):
        p  = dict(zip(keys, combo))
        vm, lo = p["vol_mult"], p["rsi_lo"]
        parts = []
        for df in all_dfs:
            mask = (
                df["above_ma25"] & df["above_ma25w"] & df["dow_up"].astype(bool) &
                df["bo_ok"] & df["change_ok"] & df["recent_bo"] & df["consol"] &
                df["to_ok"] &
                df[f"vol_{vm}x"] & (df["RSI"] >= lo) &
                df["breakout_ret"].notna()
            )
            parts.append(df.loc[mask, "breakout_ret"])
        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**p, **m})
    print(f"  完了: 合格 {sum(1 for r in results if r['wr'] >= CRITERIA_WR and r['pf'] >= CRITERIA_PF and r['spd'] >= CRITERIA_SPD)} 件")
    return results


def run_pullback(all_dfs, trading_days):
    g = GRID["pullback"]
    combos = [c for c in itertools.product(*g.values()) if c[1] < c[2]]   # rsi_lo < rsi_hi
    keys   = list(g.keys())
    print(f"\n【押し目買い型】 {len(combos)} 通り")
    results = []
    for ci, combo in enumerate(combos, 1):
        p  = dict(zip(keys, combo))
        tp, lo, hi = p["touch_pct"], p["rsi_lo"], p["rsi_hi"]
        col = f"touch_{int(tp*100)}"
        parts = []
        for df in all_dfs:
            mask = (
                df["above_ma25"] & df["above_ma25w"] & df["dow_up"].astype(bool) &
                df["macd_pos"] & df["to_ok"] &
                df[col] & (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                df["pullback_ret"].notna()
            )
            parts.append(df.loc[mask, "pullback_ret"])
        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = _metrics(combined, trading_days)
        results.append({**p, **m})
        if ci % 10 == 0 or ci == len(combos):
            passed = sum(1 for r in results if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD)
            print(f"  {ci:2d}/{len(combos)}: 合格 {passed} 件")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────────────────────────────────────
def _print_results(name: str, results: list[dict], top_n: int = 5):
    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD]
    top = sorted(qualified, key=lambda r: r["pf"], reverse=True) if qualified else \
          sorted(results,   key=lambda r: r["pf"], reverse=True)

    print(f"\n{'='*68}")
    print(f"【{name}】  合格: {len(qualified)} / {len(results)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")
    label = "合格・PF順" if qualified else "合格なし・PF順上位"

    # ヘッダー（戦略ごとにパラメータが異なる）
    sample = top[0] if top else {}
    param_keys = [k for k in sample if k not in {"n", "wr", "pf", "spd"}]

    print(f"\n  上位{top_n}件 【{label}】")
    param_hdr = "  ".join(f"{k}" for k in param_keys)
    print(f"  {param_hdr:<42} {'勝率':>7} {'PF':>6} {'件数':>6} {'件/日':>6}")
    print("  " + "-" * 70)
    for r in top[:top_n]:
        param_str = "  ".join(
            f"{r[k]:.0f}%" if k == "weekly_dev" else
            f"{r[k]:.2f}x" if k in ("vol_mult",) else
            f"{r[k]:.0f}" if isinstance(r[k], float) else
            f"{r[k]}"
            for k in param_keys
        )
        mark = "★" if (r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD) else "  "
        print(f"{mark} {param_str:<42} {r['wr']:>6.1f}%  {r['pf']:>6.2f}  {r['n']:>6,}  {r['spd']:>6.2f}")

    if qualified:
        best = top[0]
        print(f"\n  ★ 最良パラメータ:")
        for k in param_keys:
            v = best[k]
            print(f"    {k:<15}: {v}")
        print(f"    {'勝率':<15}: {best['wr']:.1f}%")
        print(f"    {'PF':<15}: {best['pf']:.2f}")
        print(f"    {'シグナル':<15}: {best['n']:,}件 ({best['spd']:.2f}/日)")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n前処理中（全指標計算）...")
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
        default=289,
    )
    print(f"\n推定取引日数: {trading_days} 日")

    bl_res = run_baseline(all_dfs, trading_days)
    bo_res = run_breakout(all_dfs, trading_days)
    pb_res = run_pullback(all_dfs, trading_days)

    _print_results("ベースライン型（指値 close×0.98 / RR1:1.5）", bl_res)
    _print_results("ブレイクアウト型（翌日始値 / RR1:2）",         bo_res)
    _print_results("押し目買い型（MA25指値 最大3日 / RR1:1.5）",   pb_res)
