#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
売られすぎ反発型 条件グリッドサーチ
====================================
目標: 2023年でもWR≥55% を達成する条件を探す
変数:
  rsi_hi      : RSI上限 [20, 25, 30]
  vol_mult    : 出来高倍率 [1.5, 2.0, 2.5, 3.0]
  close_pct   : シグナル日の値位置下限 (close-low)/(high-low) [0, 0.3, 0.5]
               = 0 なら条件なし / 0.5なら日中値幅の上半分で終値

固定:
  ATR拡大フィルター: あり（現状維持）
  損切り = シグナル日安値 / エントリー = 翌日始値
  RR=2.0 / 最大保有20日
"""

import itertools, pickle, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_cache_name = sys.argv[1] if len(sys.argv) > 1 else "backtest_cache.pkl"
CACHE_PATH    = Path(__file__).parent / _cache_name
MIN_HISTORY   = 100
MIN_TURNOVER  = 30_000_000
STOP_CAP_PCT  = 0.10
NO_LIMIT_DAYS = 1000

RR       = 2.0
MAX_HOLD = 20

# グリッド
RSI_HI_LIST     = [20, 25, 30]
VOL_MULT_LIST   = [1.5, 2.0, 2.5, 3.0]
CLOSE_PCT_LIST  = [0.0, 0.3, 0.5]   # 0=条件なし

# 合格基準
OK_PF_YEAR  = 1.0    # 各年の最低PF
OK_COUNT    = 0.3    # 件/日（全体）


def _rsi(c: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if len(c) <= period:
        return out
    d = np.diff(c.astype(float))
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def _signals(df: pd.DataFrame, rsi_hi: float, vol_mult: float,
             close_pct_min: float) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    rsi14  = _rsi(c, 14)
    rng    = h - lo
    atr_s  = pd.Series(pd.Series(
        np.maximum.reduce([rng,
                           np.abs(h - np.roll(c, 1)),
                           np.abs(lo - np.roll(c, 1))])
    ).rolling(14).mean())
    atr3   = atr_s.rolling(3).mean().values
    atr3p  = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (atr3 > atr3p) & ~np.isnan(atr3) & ~np.isnan(atr3p)

    # 終値の値位置 (close-low)/(high-low)
    cp = np.where(rng > 0, (c - lo) / rng, np.nan)

    mask = (liquid &
            (rsi14 <= rsi_hi) &
            (vol_r >= vol_mult) &
            atr_exp &
            ~np.isnan(rsi14) & ~np.isnan(vol_r))

    if close_pct_min > 0:
        mask = mask & (cp >= close_pct_min) & ~np.isnan(cp)

    mask[-50:] = False
    return np.where(mask)[0]


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray) -> list[tuple]:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    dates = df.index
    n  = len(c)
    results = []

    for si in sig_idx:
        stop  = lo[si]
        limit = stop / (1 - STOP_CAP_PCT)
        if c[si] > limit:
            continue
        j0 = si + 1
        if j0 >= n:
            continue
        entry = o[j0]
        if entry <= 0 or entry <= stop:
            continue

        take = entry + (entry - stop) * RR
        exit_price = None
        end = min(si + 1 + MAX_HOLD, n)

        for j in range(j0, end):
            op = o[j]
            if op <= stop:   exit_price = op;   break
            if op >= take:   exit_price = take;  break
            if h[j] >= take: exit_price = take;  break
            if lo[j] <= stop: exit_price = stop; break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]

        results.append((dates[si].year, (exit_price - entry) / entry * 100))

    return results


def _pf(rets: np.ndarray) -> float | None:
    if len(rets) < 5:
        return None
    wins   = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    return wins / losses if losses > 0 else (float("inf") if wins > 0 else None)


def run_grid(all_data: dict, trading_days: int):
    combos = list(itertools.product(RSI_HI_LIST, VOL_MULT_LIST, CLOSE_PCT_LIST))
    YEARS  = [2022, 2023, 2024, 2025, 2026]

    print(f"\n売られすぎ反発型 グリッドサーチ  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"合格基準: 全年PF≥{OK_PF_YEAR} / 件数≥{OK_COUNT}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'RSI':>4} {'Vol':>4} {'値位':>4} │ "
           f"{'2022':>6} {'2023':>6} {'2024':>6} {'2025':>6} {'2026':>5} │ "
           f"{'全体PF':>7}  {'件/日':>5}  {'判定':>4}")
    print(hdr)
    print("  " + "─" * 94)

    passed = []

    for rsi_hi, vol_mult, cp_min in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="first")]
            sig_idx = _signals(df, rsi_hi, vol_mult, cp_min)
            if len(sig_idx) == 0:
                continue
            trades = _backtest_one(df, sig_idx)
            for year, ret in trades:
                year_rets[year].append(ret)
                filled += 1

        if filled < 5:
            continue

        year_pf = {yr: _pf(np.array(year_rets.get(yr, []))) for yr in YEARS}

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        total_pf = _pf(all_rets)
        spd = filled / trading_days

        eval_yrs = [2022, 2023, 2024, 2025]
        ok_years = all(year_pf.get(yr) is not None and year_pf[yr] >= OK_PF_YEAR
                       for yr in eval_yrs)
        ok_count = spd >= OK_COUNT
        mark = "✅" if ok_years and ok_count else "  "

        def fmt_pf(pf):
            if pf is None: return "  N/A"
            if pf == float("inf"): return "  ∞  "
            flag = "⚠" if pf < OK_PF_YEAR else " "
            return f"{pf:5.2f}{flag}"

        cp_label = f"{cp_min:.1f}" if cp_min > 0 else " off"
        total_str = f"{total_pf:.2f}" if total_pf is not None else " N/A"
        print(f"  {rsi_hi:>4} {vol_mult:>4.1f} {cp_label:>4} │ "
              f"{fmt_pf(year_pf.get(2022))} {fmt_pf(year_pf.get(2023))} "
              f"{fmt_pf(year_pf.get(2024))} {fmt_pf(year_pf.get(2025))} "
              f"{fmt_pf(year_pf.get(2026))} │ "
              f"{total_str:>7}  {spd:>4.2f}/日  {mark}")

        if ok_years and ok_count:
            passed.append({
                "rsi_hi": rsi_hi, "vol_mult": vol_mult, "close_pct": cp_min,
                "pf_2022": year_pf.get(2022), "pf_2023": year_pf.get(2023),
                "pf_2024": year_pf.get(2024), "pf_2025": year_pf.get(2025),
                "total_pf": total_pf, "spd": spd
            })

    print("\n" + "=" * 96)
    if passed:
        print(f"✅ 合格: {len(passed)}通り")
        for p in passed:
            print(f"  RSI≤{p['rsi_hi']}  Vol≥{p['vol_mult']}x  値位≥{p['close_pct']:.1f}  "
                  f"│ 2022:{p['pf_2022']:.2f}  2023:{p['pf_2023']:.2f}  "
                  f"2024:{p['pf_2024']:.2f}  2025:{p['pf_2025']:.2f}  "
                  f"全体PF:{p['total_pf']:.2f}  {p['spd']:.2f}件/日")
    else:
        print("❌ 合格なし")


def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    trading_days   = len(next(iter(all_data.values())))
    print(f"銘柄数: {len(all_data)}  取引日数: {trading_days}日  データ期間〜{cache['date']}")

    run_grid(all_data, trading_days)
    print("\n完了")


if __name__ == "__main__":
    main()
