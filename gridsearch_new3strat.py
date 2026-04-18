#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新3型グリッドサーチ（翌日始値 成行 / ATR×2.0損切り）
①モメンタム加速型
②ボラティリティ収縮ブレイク型（ATRスクイーズ改良版）
③出来高プロファイル型
"""

import itertools, pickle, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 60
MAX_HOLD    = 20

CRITERIA_WR  = 52.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.5

# ── グリッド定義 ──────────────────────────────────────────────────────────────
GRID = {
    "momentum": {
        "rsi_lo":  [45.0, 50.0, 55.0],
        "rsi_hi":  [60.0, 65.0, 70.0],
        "vol_acc": [1.0, 1.1, 1.2],   # vol_5d_now > vol_5d_prev * vol_acc
        "rr":      [1.0, 1.5, 2.0],
    },
    "vol_squeeze": {
        "rsi_lo":  [35.0, 40.0, 45.0],
        "rsi_hi":  [55.0, 60.0, 65.0],
        "vol_mult":[1.5, 2.0, 2.5],
        "rr":      [1.0, 1.5, 2.0],
    },
    "vol_profile": {
        "rsi_lo":   [40.0, 45.0, 50.0],
        "rsi_hi":   [60.0, 65.0, 70.0],
        "vol_mult": [2.5, 3.0, 3.5],
        "close_pct":[0.03, 0.05, 0.07],
        "rr":       [1.5, 2.0],
    },
}


# ── 出口計算（翌日始値エントリー → close[i+1]から探索）────────────────────────
def _calc_rets(closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """signal index vidx の各トレードのリターン（%）を返す。"""
    n = len(closes)
    if len(vidx) == 0:
        return np.array([])

    raw_idx  = vidx[:, np.newaxis] + np.arange(1, 1 + MAX_HOLD)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)

    hit     = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets    = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


# ── 1銘柄前処理 ────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 20:
        return None

    df  = df_raw.copy()
    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]
    o   = df["Open"]

    # ATR 14日
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # RSI 14日
    df["RSI"] = calc_rsi(c)

    # MA25
    df["MA25"] = c.rolling(25).mean()

    # 売買代金（21日平均, 1日前シフト）
    df["avg_to"] = (c * v).rolling(21).mean().shift(1)

    # 出来高 5日平均（現在） / 前5日平均（vol_acc 用）
    df["vol_5d_now"]  = v.rolling(5).mean()
    df["vol_5d_prev"] = v.shift(5).rolling(5).mean()

    # 出来高 20日平均（vol_mult 用、1日シフト）
    df["avg_vol_20"] = v.rolling(21).mean().shift(1)

    # ①モメンタム加速: 5日騰落率の加速
    df["mom_5d_now"]  = c / c.shift(5) - 1
    df["mom_5d_prev"] = c.shift(5) / c.shift(10) - 1

    # ②BB幅（正規化）と ATR 収縮
    bb_std = c.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    df["bb_width"]     = np.where(bb_mid > 0, 2 * bb_std / bb_mid, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(20).mean()

    df["atr_5d"]       = df["ATR"].rolling(5).mean()
    df["atr_10d_prev"] = df["ATR"].shift(5).rolling(10).mean()

    # ③終値が高値から何%下か（0=高値引け, 0.05=5%下）
    df["close_gap_from_high"] = np.where(h > 0, (h - c) / h, np.nan)

    return df


# ── 結果集計・表示ユーティリティ ─────────────────────────────────────────────
def _aggregate(combo_rets: dict, trading_days: int) -> list:
    results = []
    for key, rets_list in combo_rets.items():
        r = np.array(rets_list)
        if len(r) < 10:
            continue
        wins   = r[r > 0]
        losses = r[r <= 0]
        wr     = len(wins) / len(r) * 100
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        spd    = len(r) / trading_days
        passed = (wr >= CRITERIA_WR) and (pf >= CRITERIA_PF) and (spd >= CRITERIA_SPD)
        results.append({
            "key": key, "wr": wr, "pf": pf, "n": len(r),
            "spd": spd, "avg_w": avg_w, "avg_l": avg_l, "passed": passed,
        })
    results.sort(key=lambda x: x["pf"], reverse=True)
    return results


def _print_results(strat_name: str, results: list, n_combos: int,
                   col_headers: list, row_fmt):
    qualified = [r for r in results if r["passed"]]
    print(f"\n{'=' * 70}")
    print(f"【{strat_name}】  合格: {len(qualified)} / {n_combos} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")
    top = (qualified if qualified else results)[:5]
    hdr = "  " + "  ".join(col_headers)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in top:
        mark = "★" if r["passed"] else " "
        print(f"{mark} {row_fmt(r)}")
    if qualified:
        best = qualified[0]
        print(f"\n  ★ 最良パラメータ:")
        for k, v in zip(best["key"]._fields if hasattr(best["key"], "_fields") else range(len(best["key"])), best["key"]):
            print(f"    param{k:<10}: {v}")
        print(f"    勝率           : {best['wr']:.1f}%")
        print(f"    PF             : {best['pf']:.2f}")
        print(f"    シグナル         : {best['n']}件 ({best['spd']:.2f}/日)")


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n前処理中（全指標計算）...")
    processed = {}
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(df_raw)
        if r is not None:
            processed[tk] = r
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)} 完了  有効: {len(processed)}")
    print(f"  {len(raw_data)}/{len(raw_data)} 完了  有効: {len(processed)}")

    # 取引日数推定
    dates = set()
    for df in list(processed.values())[:30]:
        dates.update(df.index.tolist())
    trading_days = len(dates)
    print(f"\n推定取引日数: {trading_days} 日")

    # ══════════════════════════════════════════════════════════════════════════
    # ①モメンタム加速型
    # 直近5日騰落率 > 前5日騰落率 & 出来高増加 & MA25上 & RSI範囲
    # ══════════════════════════════════════════════════════════════════════════
    g = GRID["momentum"]
    combos_m = [
        (rl, rh, va, rr)
        for rl, rh, va, rr in itertools.product(
            g["rsi_lo"], g["rsi_hi"], g["vol_acc"], g["rr"]
        )
        if rl < rh
    ]
    print(f"\n【①モメンタム加速型】 {len(combos_m)} 通り")
    combo_rets_m: dict = defaultdict(list)

    for i, (tk, df) in enumerate(processed.items(), 1):
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        ma25  = df["MA25"].values.astype(float)
        v5n   = df["vol_5d_now"].values.astype(float)
        v5p   = df["vol_5d_prev"].values.astype(float)
        mom_n = df["mom_5d_now"].values.astype(float)
        mom_p = df["mom_5d_prev"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above    = (c_a > ma25) & (~np.isnan(ma25))
        mom_acc  = (mom_n > mom_p) & (~np.isnan(mom_n)) & (~np.isnan(mom_p))
        fixed_ok = base_ok & above & mom_acc

        for (rl, rh, va, rr) in combos_m:
            rsi_ok = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok = (v5n > v5p * va) & (v5p > 0) & (~np.isnan(v5n))
            vidx   = np.where(fixed_ok & rsi_ok & vol_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_m[(rl, rh, va, rr)].extend(rets.tolist())

        if i % 500 == 0:
            print(f"   {i}/{len(processed)} 完了")

    res_m = _aggregate(combo_rets_m, trading_days)

    # 表示
    qual_m = [r for r in res_m if r["passed"]]
    print(f"\n{'=' * 70}")
    print(f"【①モメンタム加速型】  合格: {len(qual_m)} / {len(combos_m)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")
    top_m = (qual_m if qual_m else res_m)[:5]
    print(f"  {'rsi_lo':>6}  {'rsi_hi':>6}  {'vol_acc':>8}  {'rr':>4}  {'勝率':>8}  {'PF':>6}  {'件数':>6}  {'件/日':>5}")
    print("  " + "-" * 65)
    for r in top_m:
        rl, rh, va, rr = r["key"]
        m = "★" if r["passed"] else " "
        print(f"{m} {rl:>5.0f}  {rh:>5.0f}  {va:>7.1f}x  {rr:>4.1f}  "
              f"{r['wr']:>7.1f}%  {r['pf']:>6.2f}  {r['n']:>6,}  {r['spd']:>5.2f}")
    if qual_m:
        best = qual_m[0]
        rl, rh, va, rr = best["key"]
        print(f"\n  ★ 最良パラメータ:")
        print(f"    rsi_lo  : {rl}")
        print(f"    rsi_hi  : {rh}")
        print(f"    vol_acc : {va}x（出来高5日平均 > 前5日平均×{va}）")
        print(f"    rr      : 1:{rr}")
        print(f"    勝率    : {best['wr']:.1f}%")
        print(f"    PF      : {best['pf']:.2f}")
        print(f"    シグナル: {best['n']}件 ({best['spd']:.2f}/日)")

    # ══════════════════════════════════════════════════════════════════════════
    # ②ボラティリティ収縮ブレイク型
    # ATR収縮（5日<前10日）& BB幅収縮（<20日平均）& 出来高急増 & RSI範囲
    # ══════════════════════════════════════════════════════════════════════════
    g = GRID["vol_squeeze"]
    combos_v = [
        (rl, rh, vm, rr)
        for rl, rh, vm, rr in itertools.product(
            g["rsi_lo"], g["rsi_hi"], g["vol_mult"], g["rr"]
        )
        if rl < rh
    ]
    print(f"\n\n【②ボラティリティ収縮ブレイク型】 {len(combos_v)} 通り")
    combo_rets_v: dict = defaultdict(list)

    for i, (tk, df) in enumerate(processed.items(), 1):
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        avg_v = df["avg_vol_20"].values.astype(float)
        atr5d = df["atr_5d"].values.astype(float)
        atr10p= df["atr_10d_prev"].values.astype(float)
        bb_w  = df["bb_width"].values.astype(float)
        bb_wa = df["bb_width_avg"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        atr_shrink = (atr5d < atr10p) & (~np.isnan(atr5d)) & (~np.isnan(atr10p)) & (atr10p > 0)
        bb_shrink  = (bb_w <= bb_wa) & (~np.isnan(bb_w)) & (~np.isnan(bb_wa))
        fixed_ok   = base_ok & atr_shrink & bb_shrink

        for (rl, rh, vm, rr) in combos_v:
            rsi_ok = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok = (vol_a >= avg_v * vm) & (avg_v > 0) & (~np.isnan(avg_v))
            vidx   = np.where(fixed_ok & rsi_ok & vol_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_v[(rl, rh, vm, rr)].extend(rets.tolist())

        if i % 500 == 0:
            print(f"   {i}/{len(processed)} 完了")

    res_v = _aggregate(combo_rets_v, trading_days)

    qual_v = [r for r in res_v if r["passed"]]
    print(f"\n{'=' * 70}")
    print(f"【②ボラティリティ収縮ブレイク型】  合格: {len(qual_v)} / {len(combos_v)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")
    top_v = (qual_v if qual_v else res_v)[:5]
    print(f"  {'rsi_lo':>6}  {'rsi_hi':>6}  {'vol':>5}  {'rr':>4}  {'勝率':>8}  {'PF':>6}  {'件数':>6}  {'件/日':>5}")
    print("  " + "-" * 60)
    for r in top_v:
        rl, rh, vm, rr = r["key"]
        m = "★" if r["passed"] else " "
        print(f"{m} {rl:>5.0f}  {rh:>5.0f}  {vm:>4.1f}x  {rr:>4.1f}  "
              f"{r['wr']:>7.1f}%  {r['pf']:>6.2f}  {r['n']:>6,}  {r['spd']:>5.2f}")
    if qual_v:
        best = qual_v[0]
        rl, rh, vm, rr = best["key"]
        print(f"\n  ★ 最良パラメータ:")
        print(f"    rsi_lo  : {rl}")
        print(f"    rsi_hi  : {rh}")
        print(f"    vol_mult: {vm}x")
        print(f"    rr      : 1:{rr}")
        print(f"    勝率    : {best['wr']:.1f}%")
        print(f"    PF      : {best['pf']:.2f}")
        print(f"    シグナル: {best['n']}件 ({best['spd']:.2f}/日)")

    # ══════════════════════════════════════════════════════════════════════════
    # ③出来高プロファイル型
    # 出来高急増（vol_mult倍） & 終値が高値からclose_pct以内 & MA25上 & RSI範囲
    # ══════════════════════════════════════════════════════════════════════════
    g = GRID["vol_profile"]
    combos_p = [
        (rl, rh, vm, cp, rr)
        for rl, rh, vm, cp, rr in itertools.product(
            g["rsi_lo"], g["rsi_hi"], g["vol_mult"], g["close_pct"], g["rr"]
        )
        if rl < rh
    ]
    print(f"\n\n【③出来高プロファイル型】 {len(combos_p)} 通り")
    combo_rets_p: dict = defaultdict(list)

    for i, (tk, df) in enumerate(processed.items(), 1):
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        ma25  = df["MA25"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        avg_v = df["avg_vol_20"].values.astype(float)
        gap_h = df["close_gap_from_high"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above    = (c_a > ma25) & (~np.isnan(ma25))
        fixed_ok = base_ok & above

        for (rl, rh, vm, cp, rr) in combos_p:
            rsi_ok   = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok   = (vol_a >= avg_v * vm) & (avg_v > 0) & (~np.isnan(avg_v))
            close_ok = (gap_h <= cp) & (~np.isnan(gap_h))
            vidx     = np.where(fixed_ok & rsi_ok & vol_ok & close_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_p[(rl, rh, vm, cp, rr)].extend(rets.tolist())

        if i % 500 == 0:
            print(f"   {i}/{len(processed)} 完了")

    res_p = _aggregate(combo_rets_p, trading_days)

    qual_p = [r for r in res_p if r["passed"]]
    print(f"\n{'=' * 70}")
    print(f"【③出来高プロファイル型】  合格: {len(qual_p)} / {len(combos_p)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")
    top_p = (qual_p if qual_p else res_p)[:5]
    print(f"  {'rsi_lo':>6}  {'rsi_hi':>6}  {'vol':>5}  {'close%':>7}  {'rr':>4}  {'勝率':>8}  {'PF':>6}  {'件数':>6}  {'件/日':>5}")
    print("  " + "-" * 72)
    for r in top_p:
        rl, rh, vm, cp, rr = r["key"]
        m = "★" if r["passed"] else " "
        print(f"{m} {rl:>5.0f}  {rh:>5.0f}  {vm:>4.1f}x  {cp*100:>5.0f}%  {rr:>4.1f}  "
              f"{r['wr']:>7.1f}%  {r['pf']:>6.2f}  {r['n']:>6,}  {r['spd']:>5.2f}")
    if qual_p:
        best = qual_p[0]
        rl, rh, vm, cp, rr = best["key"]
        print(f"\n  ★ 最良パラメータ:")
        print(f"    rsi_lo   : {rl}")
        print(f"    rsi_hi   : {rh}")
        print(f"    vol_mult : {vm}x")
        print(f"    close_pct: {cp*100:.0f}%以内（高値から）")
        print(f"    rr       : 1:{rr}")
        print(f"    勝率     : {best['wr']:.1f}%")
        print(f"    PF       : {best['pf']:.2f}")
        print(f"    シグナル : {best['n']}件 ({best['spd']:.2f}/日)")

    # ── 全体サマリー ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("【全体サマリー】")
    print(f"  ①モメンタム加速型            : 合格 {len(qual_m)}/{len(combos_m)}")
    print(f"  ②ボラティリティ収縮ブレイク型 : 合格 {len(qual_v)}/{len(combos_v)}")
    print(f"  ③出来高プロファイル型         : 合格 {len(qual_p)}/{len(combos_p)}")

    all_qualified = []
    for strat, results in [("①モメンタム加速", qual_m), ("②Vol収縮ブレイク", qual_v), ("③出来高プロファイル", qual_p)]:
        for r in results:
            all_qualified.append((strat, r))
    all_qualified.sort(key=lambda x: x[1]["pf"], reverse=True)
    if all_qualified:
        print("\n  ★ 合格条件 PF順:")
        print(f"  {'型':>18}  {'勝率':>8}  {'PF':>6}  {'件/日':>5}")
        for strat, r in all_qualified[:10]:
            print(f"  {strat:>18}  {r['wr']:>7.1f}%  {r['pf']:>6.2f}  {r['spd']:>5.2f}")


if __name__ == "__main__":
    main()
