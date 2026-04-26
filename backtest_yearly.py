#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年別バックテスト検証 - サバイバーシップバイアス確認用
======================================================
売られすぎ反発型 / NOA を年ごとに分解してWR推移を確認する。
全体WRが高くても特定年に偏っていればバイアスの可能性が高い。
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
NO_LIMIT_DAYS = 1000

# 検証パラメータ（new_designの代表値）
RR       = 2.0
MAX_HOLD = 20

# 戦略パラメータ
OB_RSI_HI   = 30.0
OB_VOL_MULT = 1.5
NOA_RSI_PERIOD = 30
NOA_RSI_HI     = 30.0


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


def _macd(c: np.ndarray, fast=12, slow=26, sig=9):
    s = pd.Series(c.astype(float))
    m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return m.values, m.ewm(span=sig, adjust=False).mean().values


def _signals(df: pd.DataFrame, strategy: str) -> np.ndarray:
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    if strategy == "oversold_bounce":
        rsi14 = _rsi(c, 14)
        atr_s = pd.Series(pd.Series(
            np.maximum.reduce([h - lo,
                               np.abs(h - np.roll(c, 1)),
                               np.abs(lo - np.roll(c, 1))])
        ).rolling(14).mean())
        atr3  = atr_s.rolling(3).mean().values
        atr3p = atr_s.shift(3).rolling(3).mean().values
        atr_exp = (atr3 > atr3p) & ~np.isnan(atr3) & ~np.isnan(atr3p)
        mask = (liquid & (rsi14 <= OB_RSI_HI) & (vol_r >= OB_VOL_MULT) &
                atr_exp & ~np.isnan(rsi14) & ~np.isnan(vol_r))

    elif strategy == "noa":
        rsi30      = _rsi(c, NOA_RSI_PERIOD)
        macd, msig = _macd(c)
        mask = (liquid & (rsi30 <= NOA_RSI_HI) & (macd < msig) &
                ~np.isnan(rsi30) & ~np.isnan(macd))
    else:
        return np.array([], dtype=int)

    mask[-50:] = False
    return np.where(mask)[0]


def _backtest_one_with_dates(df: pd.DataFrame, sig_idx: np.ndarray,
                              rr: float, max_hold: int) -> list[tuple]:
    """(year, ret) のリストを返す"""
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    dates = df.index
    n  = len(c)
    results = []
    hold = max_hold if max_hold > 0 else NO_LIMIT_DAYS

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

        take = entry + (entry - stop) * rr
        exit_price = None
        end = min(si + 1 + hold, n)

        for j in range(j0, end):
            op = o[j]
            if op <= stop:
                exit_price = op; break
            if op >= take:
                exit_price = take; break
            if h[j] >= take:
                exit_price = take; break
            if lo[j] <= stop:
                exit_price = stop; break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]

        ret  = (exit_price - entry) / entry * 100
        year = dates[si].year
        results.append((year, ret))

    return results


def run_yearly(all_data: dict, strategy: str):
    label = {"oversold_bounce": "②売られすぎ反発型", "noa": "NOA"}[strategy]
    print(f"\n{'='*70}")
    print(f"【{label}】年別バックテスト  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"{'='*70}")

    # シグナル生成
    year_trades: dict[int, list[float]] = defaultdict(list)
    for ticker, df in all_data.items():
        sig_idx = _signals(df, strategy)
        if len(sig_idx) == 0:
            continue
        trades = _backtest_one_with_dates(df, sig_idx, RR, MAX_HOLD)
        for year, ret in trades:
            year_trades[year].append(ret)

    # 年別集計
    print(f"\n  {'年':>6}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件数':>6}  {'平均利':>7}  {'平均損':>7}")
    print("  " + "-" * 60)

    all_rets = []
    for year in sorted(year_trades.keys()):
        rets = np.array(year_trades[year])
        all_rets.extend(rets)
        if len(rets) < 5:
            print(f"  {year:>6}  (サンプル不足: {len(rets)}件)")
            continue
        wins   = rets[rets > 0]
        losses = rets[rets <= 0]
        wr     = len(wins) / len(rets) * 100
        avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        flag   = "⚠️" if wr < 55 else ""
        print(f"  {year:>6}  {wr:>5.1f}%  {pf:>5.2f}  {ev:>+6.2f}%  {len(rets):>6}件  "
              f"{avg_w:>+6.2f}%  {avg_l:>+6.2f}%  {flag}")

    # 全体
    if all_rets:
        rets = np.array(all_rets)
        wins   = rets[rets > 0]
        losses = rets[rets <= 0]
        wr     = len(wins) / len(rets) * 100
        avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        print("  " + "-" * 60)
        print(f"  {'全体':>6}  {wr:>5.1f}%  {pf:>5.2f}  {ev:>+6.2f}%  {len(rets):>6}件  "
              f"{avg_w:>+6.2f}%  {avg_l:>+6.2f}%")


def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    print(f"銘柄数: {len(all_data)}  データ期間〜{cache['date']}")

    for strat in ["oversold_bounce", "noa"]:
        run_yearly(all_data, strat)

    print(f"\n完了")


if __name__ == "__main__":
    main()
