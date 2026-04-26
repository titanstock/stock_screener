#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOA条件分解：RSI(30) vs MACD vs 両方
どちらの条件がWRに貢献しているか切り分ける。
"""

import pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH   = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY  = 100
MIN_TURNOVER = 30_000_000
STOP_CAP_PCT = 0.10
RR           = 2.0
MAX_HOLD     = 20
NO_LIMIT     = 1000

NOA_RSI_PERIOD = 30
NOA_RSI_HI     = 30.0

VARIANTS = [
    ("RSI(30)≤30 のみ",         "rsi_only"),
    ("MACD<Signal のみ",        "macd_only"),
    ("RSI(30)≤30 + MACD<Signal（現NOA）", "both"),
]


def _rsi(c: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if len(c) <= period:
        return out
    d  = np.diff(c.astype(float))
    g  = np.where(d > 0, d, 0.0)
    l  = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def _macd(c: np.ndarray, fast=12, slow=26, sig=9):
    s = pd.Series(c.astype(float))
    m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return m.values, m.ewm(span=sig, adjust=False).mean().values


def _signals(df: pd.DataFrame, variant: str) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    rsi30      = _rsi(c, NOA_RSI_PERIOD)
    macd, msig = _macd(c)

    rsi_cond  = (rsi30 <= NOA_RSI_HI) & ~np.isnan(rsi30)
    macd_cond = (macd < msig) & ~np.isnan(macd)

    if variant == "rsi_only":
        mask = liquid & rsi_cond
    elif variant == "macd_only":
        mask = liquid & macd_cond & ~np.isnan(rsi30)   # RSI計算済みであることを保証
    else:  # both
        mask = liquid & rsi_cond & macd_cond

    mask[-50:] = False
    return np.where(mask)[0]


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray) -> list[tuple]:
    c     = df["Close"].values.astype(float)
    h     = df["High"].values.astype(float)
    lo    = df["Low"].values.astype(float)
    o     = df["Open"].values.astype(float)
    dates = df.index
    n     = len(c)
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
            if op <= stop:    exit_price = op;   break
            if op >= take:    exit_price = take;  break
            if h[j] >= take:  exit_price = take;  break
            if lo[j] <= stop: exit_price = stop;  break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]

        results.append((dates[si].year, (exit_price - entry) / entry * 100))

    return results


def run(all_data: dict, trading_days: int):
    print(f"\nRR={RR}  最大保有={MAX_HOLD}日")
    print(f"{'='*72}")

    for label, variant in VARIANTS:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0
        total_sig = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, variant)
            total_sig += len(sig_idx)
            for yr, ret in _backtest_one(df, sig_idx):
                year_rets[yr].append(ret)
                filled += 1

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        if len(all_rets) < 5:
            print(f"\n【{label}】サンプル不足")
            continue

        wins   = all_rets[all_rets > 0]
        losses = all_rets[all_rets <= 0]
        wr     = len(wins) / len(all_rets) * 100
        avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd    = filled / trading_days
        skip_r = (total_sig - filled) / total_sig * 100 if total_sig > 0 else 0

        print(f"\n【{label}】")
        print(f"  シグナル数: {total_sig}件  約定: {filled}件  "
              f"スキップ率: {skip_r:.1f}%  件/日: {spd:.2f}")
        print(f"  全体  WR: {wr:.1f}%  PF: {pf:.2f}  EV: {ev:+.2f}%  "
              f"平均利: {avg_w:+.2f}%  平均損: {avg_l:+.2f}%")

        print(f"\n  {'年':>5}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件数':>5}")
        print("  " + "─" * 38)
        for yr in [2024, 2025, 2026]:
            r = np.array(year_rets.get(yr, []))
            if len(r) < 5:
                print(f"  {yr:>5}  (少 {len(r)}件)")
                continue
            w  = r[r > 0]; ls = r[r <= 0]
            y_wr = len(w) / len(r) * 100
            y_pf = abs(w.mean() / ls.mean()) if len(ls) > 0 else 0.0
            y_ev = y_wr / 100 * (w.mean() if len(w) > 0 else 0) + \
                   (1 - y_wr / 100) * (ls.mean() if len(ls) > 0 else 0)
            flag = "✅" if y_wr >= 55 else "⚠️"
            print(f"  {yr:>5}  {y_wr:>5.1f}%  {y_pf:>5.2f}  {y_ev:>+6.2f}%  "
                  f"{len(r):>5}件  {flag}")


def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    trading_days   = len(next(iter(all_data.values())))
    print(f"銘柄数: {len(all_data)}  取引日数: {trading_days}日  データ期間〜{cache['date']}")

    run(all_data, trading_days)
    print(f"\n{'='*72}")
    print("完了")


if __name__ == "__main__":
    main()
