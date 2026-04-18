#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新3型グリッドサーチ
①週足転換型      ②価格帯出来高型（VOC突破）      ③新高値接近型
共通: 翌日始値 / ATR×2.0（-10%上限）/ 売買代金3000万以上
"""

import itertools, pickle, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 100
MAX_HOLD    = 20
MAX_WORKERS = 20

CRITERIA_WR  = 52.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.5

MKTCAP_LIST = [10e9, 20e9, 30e9]                              # 100/200/300億
RSI_STD     = [(40.0, 60.0), (45.0, 65.0), (50.0, 70.0)]
RSI_HIGH    = [(45.0, 65.0), (50.0, 70.0), (55.0, 75.0)]

GRID = {
    "weekly_turn": {
        "vol_mult":  [1.2, 1.5, 2.0],
        "rsi_range": RSI_STD,
        "mktcap":    MKTCAP_LIST,
        "rr":        [1.5, 2.0],
    },
    "vol_poc": {
        "vol_mult":  [2.0, 2.5, 3.0],
        "rsi_range": RSI_STD,
        "mktcap":    MKTCAP_LIST,
        "rr":        [1.5, 2.0],
    },
    "near_high": {
        "lo_pct":    [0.88, 0.90, 0.92],
        "vol_mult":  [1.5, 2.0, 2.5],
        "rsi_range": RSI_HIGH,
        "mktcap":    MKTCAP_LIST,
        "rr":        [1.5, 2.0],
    },
}


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


# ── 出口計算（翌日始値エントリー → close[i+1]から探索）────────────────────
def _calc_rets(closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    n = len(closes)
    if len(vidx) == 0:
        return np.array([])
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    hit      = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


# ── 前処理（3型共通）──────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, shares: float | None) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 30:
        return None

    df = df_raw.copy()
    c  = df["Close"]
    h  = df["High"]
    l  = df["Low"]
    v  = df["Volume"]
    o  = df["Open"]
    n  = len(df)

    # ── 共通指標 ──────────────────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(14).mean()
    df["RSI"]     = calc_rsi(c)
    df["MA25"]    = c.rolling(25).mean()
    n_p           = BREAKOUT_DAYS + 1
    df["avg_vol"] = v.rolling(n_p).mean().shift(1)
    df["avg_to"]  = (c * v).rolling(n_p).mean().shift(1)
    df["mktcap"]  = (c * shares) if shares is not None else np.nan

    # ── ①週足転換型：週足データ ───────────────────────────────────────────────
    try:
        # 今週の最初の始値（月曜日始値、祝日スキップあり）
        df["week_open"]   = o.resample("W-MON").transform("first")

        # 最後に完了した週のデータ（ffillで各日に伝播）
        weekly = df.resample("W").agg({"Open": "first", "Close": "last", "Volume": "sum"})
        df["last_w_open"]  = weekly["Open"].reindex(df.index, method="ffill")
        df["last_w_close"] = weekly["Close"].reindex(df.index, method="ffill")
        # 1日あたり平均（週総出来高 / 5）
        df["last_w_vol_d"] = weekly["Volume"].reindex(df.index, method="ffill") / 5.0
    except Exception:
        for col in ["week_open", "last_w_open", "last_w_close", "last_w_vol_d"]:
            df[col] = np.nan

    # ── ②価格帯出来高型：POC（過去20日で最大出来高の価格）────────────────────
    vol_arr = v.values.astype(float)
    c_arr   = c.values.astype(float)
    poc_arr = np.full(n, np.nan)
    if n > 20:
        # sliding_window_view[k] = vol_arr[k:k+20]
        # 日 i (i>=20) の「直近20日（今日を含まない）」= row k = i-20
        vol_w   = sliding_window_view(vol_arr, 20)  # shape (n-19, 20)
        c_w     = sliding_window_view(c_arr,   20)
        n_rows  = n - 20                             # rows for days 20..n-1
        max_j   = np.argmax(vol_w[:n_rows], axis=1) # shape (n-20,)
        row_b   = np.arange(n_rows)
        poc_arr[20:] = c_w[row_b, max_j]
    df["poc_price"] = poc_arr

    # ── ④STH用：EMA25・MACD・Signal ──────────────────────────────────────────
    ema12            = c.ewm(span=12, adjust=False).mean()
    ema26            = c.ewm(span=26, adjust=False).mean()
    df["EMA25"]      = c.ewm(span=25, adjust=False).mean()
    df["MACD"]       = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── ③新高値接近型：52週高値・週足MA25 ────────────────────────────────────
    df["high_52w"] = c.shift(1).rolling(252, min_periods=100).max()
    try:
        wc          = c.resample("W").last()
        df["MA25W"] = wc.rolling(25).mean().reindex(df.index, method="ffill")
    except Exception:
        df["MA25W"] = np.nan

    return df


# ── 結果集計・表示 ────────────────────────────────────────────────────────────
def _aggregate(combo_rets: dict, trading_days: int) -> list:
    results = []
    for key, r_list in combo_rets.items():
        r = np.array(r_list)
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
        results.append({"key": key, "wr": wr, "pf": pf, "n": len(r),
                        "spd": spd, "ev": ev, "passed": passed})
    results.sort(key=lambda x: x["pf"], reverse=True)
    return results


def _display(strat_name: str, results: list, n_combos: int,
             header: str, row_fn, key_labels: list):
    qualified = [r for r in results if r["passed"]]
    print(f"\n{'=' * 70}")
    print(f"【{strat_name}】  合格: {len(qualified)} / {n_combos} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上\n")
    print(f"  {header}")
    print("  " + "-" * (len(header) - 2))
    top = (qualified if qualified else results)[:5]
    for r in top:
        mark = "★" if r["passed"] else " "
        print(f"{mark} {row_fn(r)}")
    if qualified:
        best = qualified[0]
        print(f"\n  ★ 最良パラメータ:")
        for lbl, val in zip(key_labels, best["key"]):
            print(f"    {lbl:<14}: {val}")
        print(f"    {'勝率':<14}: {best['wr']:.1f}%")
        print(f"    {'PF':<14}: {best['pf']:.2f}")
        print(f"    {'期待値':<14}: {best['ev']:+.2f}%/トレード")
        print(f"    {'シグナル':<14}: {best['n']}件 ({best['spd']:.2f}/日)")


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

    print("\n前処理中（全指標計算）...")
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

    # ── グリッドコンボ生成 ────────────────────────────────────────────────────
    combos_wt = list(itertools.product(
        GRID["weekly_turn"]["vol_mult"], GRID["weekly_turn"]["rsi_range"],
        GRID["weekly_turn"]["mktcap"],  GRID["weekly_turn"]["rr"],
    ))
    combos_vp = list(itertools.product(
        GRID["vol_poc"]["vol_mult"], GRID["vol_poc"]["rsi_range"],
        GRID["vol_poc"]["mktcap"],  GRID["vol_poc"]["rr"],
    ))
    combos_nh = list(itertools.product(
        GRID["near_high"]["lo_pct"],   GRID["near_high"]["vol_mult"],
        GRID["near_high"]["rsi_range"],GRID["near_high"]["mktcap"],
        GRID["near_high"]["rr"],
    ))
    print(f"\nグリッド: ①{len(combos_wt)}通り / ②{len(combos_vp)}通り / ③{len(combos_nh)}通り")

    # ══════════════════════════════════════════════════════════════════════════
    # ①週足転換型
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n【①週足転換型】 処理中...")
    combo_rets_wt: dict = defaultdict(list)

    for tk, df in processed.items():
        n      = len(df)
        c_a    = df["Close"].values.astype(float)
        o_a    = df["Open"].values.astype(float)
        atr    = df["ATR"].values.astype(float)
        rsi    = df["RSI"].values.astype(float)
        ma25   = df["MA25"].values.astype(float)
        avg_v  = df["avg_vol"].values.astype(float)
        to_a   = df["avg_to"].values.astype(float)
        mktcap = df["mktcap"].values.astype(float)
        vol_a  = df["Volume"].values.astype(float)
        wk_op  = df["week_open"].values.astype(float)
        lw_op  = df["last_w_open"].values.astype(float)
        lw_cl  = df["last_w_close"].values.astype(float)
        lw_vd  = df["last_w_vol_d"].values.astype(float)   # prev week daily avg

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above_ma25  = (c_a > ma25) & (~np.isnan(ma25))
        curr_bull   = (c_a > wk_op) & (~np.isnan(wk_op))
        prev_bear   = (lw_cl < lw_op) & (~np.isnan(lw_cl)) & (~np.isnan(lw_op))
        fixed_ok    = base_ok & above_ma25 & curr_bull & prev_bear

        for (vm, (rl, rh), mkt, rr) in combos_wt:
            rsi_ok    = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok    = (vol_a >= lw_vd * vm) & (lw_vd > 0) & (~np.isnan(lw_vd))
            mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt)
            vidx      = np.where(fixed_ok & rsi_ok & vol_ok & mktcap_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_wt[(vm, rl, rh, int(mkt/1e8), rr)].extend(rets.tolist())

    res_wt = _aggregate(combo_rets_wt, trading_days)
    _display(
        "①週足転換型（今週陽線+先週陰線+出来高増加+MA25上）",
        res_wt, len(combos_wt),
        f"{'vol':>5}  {'RSI範囲':>9}  {'時価総額':>7}  {'RR':>4}  "
        f"{'勝率':>7}  {'PF':>5}  {'期待値':>7}  {'件数':>5}  {'件/日':>5}",
        lambda r: (
            f"{r['key'][0]:>4.1f}x  {r['key'][1]:.0f}-{r['key'][2]:.0f}  "
            f"{r['key'][3]:>5}億  {r['key'][4]:>4.1f}  "
            f"{r['wr']:>6.1f}%  {r['pf']:>5.2f}  {r['ev']:>+6.2f}%  "
            f"{r['n']:>5,}  {r['spd']:>5.2f}"
        ),
        ["vol_mult", "rsi_lo", "rsi_hi", "mktcap(億)", "rr"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ②価格帯出来高型（POC突破）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n【②価格帯出来高型】 処理中...")
    combo_rets_vp: dict = defaultdict(list)

    for tk, df in processed.items():
        n       = len(df)
        c_a     = df["Close"].values.astype(float)
        o_a     = df["Open"].values.astype(float)
        atr     = df["ATR"].values.astype(float)
        rsi     = df["RSI"].values.astype(float)
        ma25    = df["MA25"].values.astype(float)
        avg_v   = df["avg_vol"].values.astype(float)
        to_a    = df["avg_to"].values.astype(float)
        mktcap  = df["mktcap"].values.astype(float)
        vol_a   = df["Volume"].values.astype(float)
        poc_p   = df["poc_price"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above_ma25 = (c_a > ma25) & (~np.isnan(ma25))
        above_poc  = (c_a > poc_p) & (~np.isnan(poc_p))
        fixed_ok   = base_ok & above_ma25 & above_poc

        for (vm, (rl, rh), mkt, rr) in combos_vp:
            rsi_ok    = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok    = (vol_a >= avg_v * vm) & (avg_v > 0)
            mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt)
            vidx      = np.where(fixed_ok & rsi_ok & vol_ok & mktcap_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_vp[(vm, rl, rh, int(mkt/1e8), rr)].extend(rets.tolist())

    res_vp = _aggregate(combo_rets_vp, trading_days)
    _display(
        "②価格帯出来高型（POC上抜け+出来高急増+MA25上）",
        res_vp, len(combos_vp),
        f"{'vol':>5}  {'RSI範囲':>9}  {'時価総額':>7}  {'RR':>4}  "
        f"{'勝率':>7}  {'PF':>5}  {'期待値':>7}  {'件数':>5}  {'件/日':>5}",
        lambda r: (
            f"{r['key'][0]:>4.1f}x  {r['key'][1]:.0f}-{r['key'][2]:.0f}  "
            f"{r['key'][3]:>5}億  {r['key'][4]:>4.1f}  "
            f"{r['wr']:>6.1f}%  {r['pf']:>5.2f}  {r['ev']:>+6.2f}%  "
            f"{r['n']:>5,}  {r['spd']:>5.2f}"
        ),
        ["vol_mult", "rsi_lo", "rsi_hi", "mktcap(億)", "rr"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ③新高値接近型
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n【③新高値接近型】 処理中...")
    combo_rets_nh: dict = defaultdict(list)

    for tk, df in processed.items():
        n       = len(df)
        c_a     = df["Close"].values.astype(float)
        o_a     = df["Open"].values.astype(float)
        atr     = df["ATR"].values.astype(float)
        rsi     = df["RSI"].values.astype(float)
        ma25    = df["MA25"].values.astype(float)
        ma25w   = df["MA25W"].values.astype(float)
        avg_v   = df["avg_vol"].values.astype(float)
        to_a    = df["avg_to"].values.astype(float)
        mktcap  = df["mktcap"].values.astype(float)
        vol_a   = df["Volume"].values.astype(float)
        hi52    = df["high_52w"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (~np.isnan(hi52)) & (hi52 > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above_ma25  = (c_a > ma25)  & (~np.isnan(ma25))
        above_ma25w = (c_a > ma25w) & (~np.isnan(ma25w))
        near_hi_hi  = c_a <= hi52 * 0.98   # 上限 98%
        fixed_base  = base_ok & above_ma25 & above_ma25w & near_hi_hi

        for (lo_p, vm, (rl, rh), mkt, rr) in combos_nh:
            near_lo   = c_a >= hi52 * lo_p
            rsi_ok    = (rsi >= rl) & (rsi <= rh) & (~np.isnan(rsi))
            vol_ok    = (vol_a >= avg_v * vm) & (avg_v > 0)
            mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt)
            vidx      = np.where(fixed_base & near_lo & rsi_ok & vol_ok & mktcap_ok)[0]
            if len(vidx) == 0:
                continue
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) > 0:
                combo_rets_nh[(int(lo_p*100), vm, rl, rh, int(mkt/1e8), rr)].extend(rets.tolist())

    res_nh = _aggregate(combo_rets_nh, trading_days)
    _display(
        "③新高値接近型（52週高値の88〜98%圏内+出来高+MA25日週上）",
        res_nh, len(combos_nh),
        f"{'lo%':>5}  {'vol':>5}  {'RSI範囲':>9}  {'時価総額':>7}  {'RR':>4}  "
        f"{'勝率':>7}  {'PF':>5}  {'期待値':>7}  {'件数':>5}  {'件/日':>5}",
        lambda r: (
            f"{r['key'][0]:>4d}%  {r['key'][1]:>4.1f}x  {r['key'][2]:.0f}-{r['key'][3]:.0f}  "
            f"{r['key'][4]:>5}億  {r['key'][5]:>4.1f}  "
            f"{r['wr']:>6.1f}%  {r['pf']:>5.2f}  {r['ev']:>+6.2f}%  "
            f"{r['n']:>5,}  {r['spd']:>5.2f}"
        ),
        ["lo_pct(%)", "vol_mult", "rsi_lo", "rsi_hi", "mktcap(億)", "rr"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ④STH（MACD+EMA25クロス・利確+10%・損切り-5%）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n【④STH（MACD+EMA25クロス）】 処理中...")
    sth_rets: list = []

    for tk, df in processed.items():
        n      = len(df)
        c_a    = df["Close"].values.astype(float)
        o_a    = df["Open"].values.astype(float)
        to_a   = df["avg_to"].values.astype(float)
        ema25  = df["EMA25"].values.astype(float)
        macd   = df["MACD"].values.astype(float)
        sig    = df["MACDSignal"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        # MACDゴールデンクロス: 今日 MACD>Signal かつ 昨日 MACD<=Signal
        macd_p = np.roll(macd, 1); macd_p[0] = np.nan
        sig_p  = np.roll(sig,  1); sig_p[0]  = np.nan
        cross  = (
            (~np.isnan(macd))   & (~np.isnan(sig)) &
            (~np.isnan(macd_p)) & (~np.isnan(sig_p)) &
            (macd > sig) & (macd_p <= sig_p)
        )

        idx     = np.arange(n)
        base_ok = (
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        above_ema25 = (c_a > ema25) & (~np.isnan(ema25))

        vidx = np.where(base_ok & cross & above_ema25)[0]
        if len(vidx) == 0:
            continue

        e_arr = next_o[vidx]
        s_arr = e_arr * 0.95   # 損切り -5%
        t_arr = e_arr * 1.10   # 利確  +10%
        rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
        if len(rets) > 0:
            sth_rets.extend(rets.tolist())

    print(f"\n{'=' * 70}")
    print(f"【④STH（MACD+EMA25クロス・利確+10%・損切り-5%）】")
    print(f"  固定パラメータ: TP=+10%  SL=-5%  保有上限={MAX_HOLD}日  エントリー=翌日始値")
    print(f"  条件: MACDゴールデンクロス + 終値>EMA25 + 売買代金≥{MIN_AVG_TURNOVER/1e6:.0f}百万\n")

    r_sth = np.array(sth_rets)
    sth_passed = False
    if len(r_sth) >= 10:
        wins_s   = r_sth[r_sth > 0]; losses_s = r_sth[r_sth <= 0]
        wr_s     = len(wins_s) / len(r_sth) * 100
        avg_w_s  = wins_s.mean()   if len(wins_s)   > 0 else 0.0
        avg_l_s  = losses_s.mean() if len(losses_s) > 0 else 0.0
        pf_s     = abs(avg_w_s / avg_l_s) if avg_l_s != 0 else 0.0
        ev_s     = wr_s / 100 * avg_w_s + (1 - wr_s / 100) * avg_l_s
        spd_s    = len(r_sth) / trading_days
        sth_passed = (wr_s >= CRITERIA_WR) and (pf_s >= CRITERIA_PF) and (spd_s >= CRITERIA_SPD)
        mark_s   = "★" if sth_passed else " "
        print(f"{mark_s} 勝率: {wr_s:.1f}%  PF: {pf_s:.2f}  期待値: {ev_s:+.2f}%  "
              f"件数: {len(r_sth):,}件  {spd_s:.2f}件/日")
        print(f"  平均利益: {avg_w_s:+.2f}%  平均損失: {avg_l_s:+.2f}%")
        print(f"  評価: {'★ 合格' if sth_passed else '不合格'}")
    else:
        print(f"  シグナル不足（{len(r_sth)}件）")

    # ── 全体サマリー ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("【全体サマリー】")
    qual_wt = [r for r in res_wt if r["passed"]]
    qual_vp = [r for r in res_vp if r["passed"]]
    qual_nh = [r for r in res_nh if r["passed"]]
    print(f"  ①週足転換型        : 合格 {len(qual_wt)}/{len(combos_wt)}")
    print(f"  ②価格帯出来高型    : 合格 {len(qual_vp)}/{len(combos_vp)}")
    print(f"  ③新高値接近型      : 合格 {len(qual_nh)}/{len(combos_nh)}")
    print(f"  ④STH              : {'★ 合格' if sth_passed else '不合格'}")

    all_q = (
        [("①週足転換", r) for r in qual_wt] +
        [("②価格帯出来高", r) for r in qual_vp] +
        [("③新高値接近", r) for r in qual_nh]
    )
    if all_q:
        all_q.sort(key=lambda x: x[1]["pf"], reverse=True)
        print(f"\n  PF順上位 (合格のみ):")
        print(f"  {'型':>15}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件/日':>5}")
        print("  " + "-" * 50)
        for strat, r in all_q[:10]:
            print(f"  {strat:>15}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
                  f"{r['ev']:>+7.2f}%  {r['spd']:>5.2f}")


if __name__ == "__main__":
    main()
