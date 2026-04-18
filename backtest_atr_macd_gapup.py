#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
①ATRスクイーズ + MACDゴールデンクロス バックテスト（RR1:1.5）
②ギャップアップ型 グリッドサーチ（RR1:1.5 / 1:2）
"""

import itertools, pickle, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 100
MAX_HOLD    = 20
MAX_WORKERS = 20

# ① ATRスクイーズ設定（グリッドサーチ最良値）
ATR_VOL_MULT   = 3.0
ATR_RSI_LO     = 45.0
ATR_RSI_HI     = 55.0
ATR_MKTCAP_MAX = 10_000_000_000   # 100億
ATR_RR         = 1.5

# ② ギャップアップ グリッド
GRID_GAP = {
    "gap_pct":  [1.0, 2.0, 3.0],
    "vol_mult": [2.0, 2.5, 3.0],
    "rsi_range": [(40.0, 60.0), (45.0, 65.0), (50.0, 70.0)],
    "mktcap":   [10_000_000_000, 20_000_000_000, 30_000_000_000],
    "rr":       [1.5, 2.0],
}

CRITERIA_WR  = 52.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.5


# ── 株数取得（並列）──────────────────────────────────────────────────────────
def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        fi = yf.Ticker(ticker).fast_info
        sh = getattr(fi, "shares", None)
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float]:
    result = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t, sh = fut.result()
            if sh:
                result[t] = sh
            done += 1
            if done % 300 == 0 or done == len(tickers):
                print(f"  株数取得: {done}/{len(tickers)}  成功: {len(result)}")
    return result


# ── 出口計算（offset=1: 翌日close始まり / offset=0: 当日close始まり）──────
def _calc_rets(closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray,
               offset: int = 1) -> np.ndarray:
    n = len(closes)
    if len(vidx) == 0:
        return np.array([])
    raw_idx  = vidx[:, np.newaxis] + np.arange(offset, offset + MAX_HOLD)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    hit      = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + offset + fhp, 0, n - 1)]
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


def _calc_types(closes: np.ndarray, vidx: np.ndarray,
                e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray,
                offset: int = 1) -> np.ndarray:
    """0=強制, 1=損切り, 2=利確"""
    n = len(closes)
    if len(vidx) == 0:
        return np.array([], dtype=int)
    raw_idx  = vidx[:, np.newaxis] + np.arange(offset, offset + MAX_HOLD)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    is_stop  = (fut <= s_arr[:, np.newaxis]) & in_range
    is_take  = (fut >= t_arr[:, np.newaxis]) & in_range
    hit      = is_stop | is_take
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    hit_stop = is_stop[np.arange(len(vidx)), fhp] & has_hit
    hit_take = is_take[np.arange(len(vidx)), fhp] & has_hit & ~hit_stop
    types    = np.where(hit_stop, 1, np.where(hit_take, 2, 0))
    return types[has_fut]


# ── 前処理（①②共通）──────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, shares: float | None) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 30:
        return None

    df = df_raw.copy()
    c  = df["Close"]
    h  = df["High"]
    l  = df["Low"]
    v  = df["Volume"]
    o  = df["Open"]

    # ATR 14日
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # RSI
    df["RSI"] = calc_rsi(c)

    # MA25
    df["MA25"] = c.rolling(25).mean()

    # 売買代金・出来高
    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = v.rolling(n_p).mean().shift(1)
    df["avg_to"]  = (c * v).rolling(n_p).mean().shift(1)

    # ① ATR収縮（3日ウィンドウ: グリッドサーチ最良）
    df["atr_3d"]      = df["ATR"].rolling(3).mean()
    df["atr_3d_prev"] = df["ATR"].shift(3).rolling(3).mean()

    # ① MACD (12/26/9)
    ema12    = c.ewm(span=12, adjust=False).mean()
    ema26    = c.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    sig_line = macd.ewm(span=9, adjust=False).mean()
    # ゴールデンクロス：直近3日以内にMACD>SIGNALに転換
    gc0 = (macd > sig_line) & (macd.shift(1) <= sig_line.shift(1))
    gc1 = (macd.shift(1) > sig_line.shift(1)) & (macd.shift(2) <= sig_line.shift(2))
    gc2 = (macd.shift(2) > sig_line.shift(2)) & (macd.shift(3) <= sig_line.shift(3))
    df["macd_gc_3d"] = gc0 | gc1 | gc2

    # 時価総額
    df["mktcap"] = (c * shares) if shares is not None else np.nan

    # ② ギャップアップ用（前日値）
    df["prev_close"]     = c.shift(1)
    df["prev_ATR"]       = df["ATR"].shift(1)
    df["prev_RSI"]       = df["RSI"].shift(1)
    df["prev_above_ma25"] = (c.shift(1) > df["MA25"].shift(1)).astype(float)  # NaN-safe

    return df


# ── 統計ヘルパー ──────────────────────────────────────────────────────────────
def _print_stat(label: str, r_list: list, t_list: list, trading_days: int):
    r = np.array(r_list)
    t = np.array(t_list, dtype=int) if t_list else np.array([], dtype=int)
    if len(r) == 0:
        print(f"\n  {label}: シグナルなし")
        return
    wins   = r[r > 0]; losses = r[r <= 0]
    wr     = len(wins) / len(r) * 100
    avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_l  = losses.mean() if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = len(r) / trading_days
    print(f"\n  {label}")
    print(f"  総トレード : {len(r):,}件  ({spd:.2f}/日)")
    print(f"  勝率       : {wr:.1f}%")
    print(f"  PF         : {pf:.2f}")
    print(f"  期待値     : {ev:+.2f}%/トレード")
    print(f"  平均利益   : +{avg_w:.2f}%  平均損失: {avg_l:.2f}%")
    if len(t) > 0 and len(t) == len(r):
        n_tot = len(t)
        for code, name in [(1, "損切り"), (2, "利確"), (0, "強制終了")]:
            mask  = (t == code)
            n_c   = int(mask.sum())
            avg_c = r[mask].mean() if n_c > 0 else 0.0
            sign  = "+" if avg_c >= 0 else ""
            print(f"  {name:5s}     : {n_c:4d}件 ({n_c/n_tot*100:.1f}%)  avg {sign}{avg_c:.2f}%")
    passed = (wr >= CRITERIA_WR) and (pf >= CRITERIA_PF) and (spd >= CRITERIA_SPD)
    print(f"  評価       : {'★ 合格' if passed else '× 不合格'} (基準: 勝率{CRITERIA_WR}%/PF{CRITERIA_PF}/{CRITERIA_SPD}件/日)")
    return wr, pf, spd, ev


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n時価総額データ取得中（yfinance）...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中...")
    processed = {}
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(df_raw, shares_map.get(tk))
        if r is not None:
            processed[tk] = r
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)} 完了  有効: {len(processed)}")
    print(f"  完了  有効: {len(processed)}")

    dates = set()
    for df in list(processed.values())[:30]:
        dates.update(df.index.tolist())
    trading_days = len(dates)
    print(f"\n推定取引日数: {trading_days} 日")

    # ══════════════════════════════════════════════════════════════════════════
    # ① ATRスクイーズ + MACDゴールデンクロス
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("【① ATRスクイーズ + MACDゴールデンクロス】")
    print(f"  ATR収縮(3日) + 出来高{ATR_VOL_MULT}倍+ RSI{ATR_RSI_LO:.0f}-{ATR_RSI_HI:.0f}")
    print(f"  + MACDゴールデンクロス（直近3日以内）")
    print(f"  時価総額100億以下 / エントリー翌日始値 / RR1:{ATR_RR}")
    print(f"{'=' * 65}")

    rets_base  = []          # ATRスクイーズのみ（比較用）
    rets_macd  = []          # + MACD GC
    types_macd = []

    for tk, df in processed.items():
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        avg_v = df["avg_vol"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        atr3d = df["atr_3d"].values.astype(float)
        atr3p = df["atr_3d_prev"].values.astype(float)
        mktcap= df["mktcap"].values.astype(float)
        gc_3d = df["macd_gc_3d"].values.astype(bool)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        atr_shrink = (atr3d < atr3p) & (~np.isnan(atr3d)) & (~np.isnan(atr3p)) & (atr3p > 0)
        rsi_ok     = (rsi >= ATR_RSI_LO) & (rsi <= ATR_RSI_HI) & (~np.isnan(rsi))
        vol_ok     = (vol_a >= avg_v * ATR_VOL_MULT) & (avg_v > 0)
        to_ok      = (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER)
        mktcap_ok  = np.isnan(mktcap) | (mktcap <= ATR_MKTCAP_MAX)

        base_sig = (
            (~np.isnan(atr)) & (atr > 0) &
            to_ok & vol_ok & rsi_ok & atr_shrink & mktcap_ok &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )

        # ATRスクイーズのみ（比較）
        vidx_b = np.where(base_sig)[0]
        if len(vidx_b) > 0:
            e = next_o[vidx_b]; a = atr[vidx_b]
            s = np.maximum(e - a * 2.0, e * 0.90)
            t = e + (e - s) * ATR_RR
            rets_base.extend(_calc_rets(c_a, vidx_b, e, s, t).tolist())

        # + MACDゴールデンクロス
        vidx = np.where(base_sig & gc_3d)[0]
        if len(vidx) == 0:
            continue
        e = next_o[vidx]; a = atr[vidx]
        s = np.maximum(e - a * 2.0, e * 0.90)
        t = e + (e - s) * ATR_RR
        rets_macd.extend(_calc_rets(c_a, vidx, e, s, t).tolist())
        types_macd.extend(_calc_types(c_a, vidx, e, s, t).tolist())

    print("\n  ── 比較: ATRスクイーズのみ ──────────────────────────────")
    stat_base = _print_stat("ATRスクイーズのみ (RR1:1.5)", rets_base, [], trading_days)

    print("\n  ── ATRスクイーズ + MACDゴールデンクロス ──────────────────")
    stat_macd = _print_stat("ATR + MACD GC (RR1:1.5)", rets_macd, types_macd, trading_days)

    # ══════════════════════════════════════════════════════════════════════════
    # ② ギャップアップ型 グリッドサーチ
    # ══════════════════════════════════════════════════════════════════════════
    combos_gap = list(itertools.product(
        GRID_GAP["gap_pct"],
        GRID_GAP["vol_mult"],
        GRID_GAP["rsi_range"],
        GRID_GAP["mktcap"],
        GRID_GAP["rr"],
    ))

    print(f"\n\n{'=' * 65}")
    print("【② ギャップアップ型 グリッドサーチ】")
    print("  条件: 始値>前日終値×(1+gap%) + 出来高倍率 + MA25上")
    print("  エントリー: 当日始値 / 損切り: ATR×2.0（上限-10%）")
    print(f"  グリッド: {len(combos_gap)} 通り")
    print(f"{'=' * 65}")

    combo_rets_gap: dict = defaultdict(list)

    for tk, df in processed.items():
        n       = len(df)
        c_a     = df["Close"].values.astype(float)
        o_a     = df["Open"].values.astype(float)
        to_a    = df["avg_to"].values.astype(float)
        avg_v   = df["avg_vol"].values.astype(float)
        vol_a   = df["Volume"].values.astype(float)
        mktcap  = df["mktcap"].values.astype(float)
        prev_c  = df["prev_close"].values.astype(float)
        prev_a  = df["prev_ATR"].values.astype(float)
        prev_r  = df["prev_RSI"].values.astype(float)
        prev_ab = df["prev_above_ma25"].values  # float (NaN-safe)

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(prev_a)) & (prev_a > 0) &
            (~np.isnan(o_a)) & (o_a > 0) &
            (~np.isnan(prev_c)) & (prev_c > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (prev_ab == 1.0) &
            (idx >= MIN_HISTORY) & (idx < n)
        )

        for (gp, vm, (rl, rh), mkt_max, rr) in combos_gap:
            gap_ok    = o_a > prev_c * (1.0 + gp / 100.0)
            vol_ok    = (vol_a >= avg_v * vm) & (avg_v > 0)
            rsi_ok    = (prev_r >= rl) & (prev_r <= rh) & (~np.isnan(prev_r))
            mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt_max)

            vidx = np.where(base_ok & gap_ok & vol_ok & rsi_ok & mktcap_ok)[0]
            if len(vidx) == 0:
                continue

            e_arr = o_a[vidx]        # 当日始値でエントリー
            a_arr = prev_a[vidx]     # 前日ATRで損切り幅計算
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr

            # offset=0: 当日close[i]から決済探索
            rets = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr, offset=0)
            if len(rets) > 0:
                combo_rets_gap[(gp, vm, rl, rh, mkt_max, rr)].extend(rets.tolist())

    # 集計
    results_gap = []
    for key, rets_list in combo_rets_gap.items():
        r = np.array(rets_list)
        if len(r) < 10:
            continue
        wins   = r[r > 0]; losses = r[r <= 0]
        wr     = len(wins) / len(r) * 100
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd    = len(r) / trading_days
        passed = (wr >= CRITERIA_WR) and (pf >= CRITERIA_PF) and (spd >= CRITERIA_SPD)
        results_gap.append({
            "key": key, "wr": wr, "pf": pf, "n": len(r),
            "spd": spd, "ev": ev, "passed": passed,
        })
    results_gap.sort(key=lambda x: x["pf"], reverse=True)
    qualified_gap = [r for r in results_gap if r["passed"]]

    print(f"\n  合格: {len(qualified_gap)} / {len(combos_gap)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")

    top = (qualified_gap if qualified_gap else results_gap)[:5]
    print(f"  {'gap%':>5}  {'vol':>5}  {'RSI範囲':>9}  {'時価総額':>7}  {'RR':>4}  "
          f"{'勝率':>7}  {'PF':>5}  {'期待値':>7}  {'件数':>5}  {'件/日':>5}")
    print("  " + "-" * 78)
    for r in top:
        gp, vm, rl, rh, mkt, rr = r["key"]
        m = "★" if r["passed"] else " "
        print(f"{m} {gp:>4.0f}%  {vm:>4.1f}x  {rl:.0f}-{rh:.0f}  "
              f"{int(mkt/1e8):>5}億  {rr:>4.1f}  "
              f"{r['wr']:>6.1f}%  {r['pf']:>5.2f}  {r['ev']:>+6.2f}%  "
              f"{r['n']:>5,}  {r['spd']:>5.2f}")

    if qualified_gap:
        best = qualified_gap[0]
        gp, vm, rl, rh, mkt, rr = best["key"]
        print(f"\n  ★ 最良パラメータ:")
        print(f"    ギャップ幅   : {gp:.0f}%以上")
        print(f"    出来高倍率   : {vm}倍以上")
        print(f"    RSI範囲      : {rl:.0f}〜{rh:.0f}")
        print(f"    時価総額上限 : {int(mkt/1e8)}億円以下")
        print(f"    RR           : 1:{rr}")
        print(f"    勝率         : {best['wr']:.1f}%")
        print(f"    PF           : {best['pf']:.2f}")
        print(f"    期待値       : {best['ev']:+.2f}%/トレード")
        print(f"    シグナル     : {best['n']}件 ({best['spd']:.2f}/日)")

    # ── 最終サマリー ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("【全体まとめ】")
    print(f"\n  {'戦略':30}  {'勝率':>7}  {'PF':>5}  {'件/日':>5}  評価")
    print("  " + "-" * 60)

    def _row(label, r_list, trading_days):
        r = np.array(r_list)
        if len(r) == 0:
            print(f"  {label:30}  データなし")
            return
        wins = r[r > 0]; losses = r[r <= 0]
        wr = len(wins)/len(r)*100
        aw = wins.mean() if len(wins) > 0 else 0.0
        al = losses.mean() if len(losses) > 0 else 0.0
        pf = abs(aw/al) if al != 0 else 0.0
        spd = len(r)/trading_days
        ok = wr >= CRITERIA_WR and pf >= CRITERIA_PF and spd >= CRITERIA_SPD
        mark = "★合格" if ok else "×"
        print(f"  {label:30}  {wr:>6.1f}%  {pf:>5.2f}  {spd:>5.2f}  {mark}")

    _row("①ATRスクイーズのみ (RR1:1.5)", rets_base, trading_days)
    _row("①ATR + MACD GC (RR1:1.5)", rets_macd, trading_days)

    if qualified_gap:
        best = qualified_gap[0]
        print(f"  {'②ギャップアップ型 (最良)':30}  {best['wr']:>6.1f}%  "
              f"{best['pf']:>5.2f}  {best['spd']:>5.2f}  ★合格")
    else:
        # 合格なしでも最良を表示
        if results_gap:
            best = results_gap[0]
            print(f"  {'②ギャップアップ型 (最良)':30}  {best['wr']:>6.1f}%  "
                  f"{best['pf']:>5.2f}  {best['spd']:>5.2f}  ×不合格")


if __name__ == "__main__":
    main()
