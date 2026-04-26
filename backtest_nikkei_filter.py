#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B案: 日経225フィルター付き売られすぎ反発型
============================================
日経225がMA75以上の時だけ売られすぎ反発シグナルを採用。
相場環境が悪い時のトレードを排除する。

日経データ: yfinance ^N225 を別途取得
"""

import pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY   = 100
MIN_TURNOVER  = 30_000_000
STOP_CAP_PCT  = 0.10
RR       = 2.0
MAX_HOLD = 20

OB_RSI_HI   = 30.0
OB_VOL_MULT = 1.5

OK_WR  = 55.0
OK_SPD = 0.3

# 日経フィルター種類
NIKKEI_FILTERS = [
    ("MA75以上", 75),
    ("MA25以上", 25),
    ("MA200以上", 200),
]


def fetch_nikkei(start_date: str) -> pd.Series:
    print("日経225データ取得中...")
    df = yf.download("^N225", start=start_date, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("日経225データ取得失敗")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    print(f"  取得: {len(close)}日分  {close.index[0].date()} 〜 {close.index[-1].date()}")
    return close


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


def _signals(df: pd.DataFrame, nikkei_up: pd.Series,
             ma_period: int) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    dates = df.index
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    rsi14  = _rsi(c, 14)
    atr_s  = pd.Series(pd.Series(
        np.maximum.reduce([h - lo,
                           np.abs(h - np.roll(c, 1)),
                           np.abs(lo - np.roll(c, 1))])
    ).rolling(14).mean())
    atr3   = atr_s.rolling(3).mean().values
    atr3p  = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (atr3 > atr3p) & ~np.isnan(atr3) & ~np.isnan(atr3p)

    ob_mask = (liquid & (rsi14 <= OB_RSI_HI) & (vol_r >= OB_VOL_MULT) &
               atr_exp & ~np.isnan(rsi14) & ~np.isnan(vol_r))

    # 日経フィルター適用
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not ob_mask[i]:
            continue
        d = dates[i]
        # 日付がnikkei_upに存在するか確認
        if d not in nikkei_up.index:
            # 直近の有効日を探す
            try:
                pos = nikkei_up.index.searchsorted(d) - 1
                if pos < 0:
                    continue
                nk_up = nikkei_up.iloc[pos]
            except Exception:
                continue
        else:
            nk_up = nikkei_up[d]

        if nk_up:
            mask[i] = True

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


def run_filter(all_data: dict, trading_days: int,
               nikkei_close: pd.Series, label: str, ma_period: int):
    # 日経MA計算
    nk_ma = nikkei_close.rolling(ma_period).mean()
    nk_up = (nikkei_close > nk_ma)   # bool Series（日付インデックス）

    print(f"\n{'='*70}")
    print(f"【B案】売られすぎ反発型 + 日経{label}フィルター  RR={RR}  最大保有={MAX_HOLD}日")

    # 日経がフィルター通過する日の割合
    pct_pass = nk_up.sum() / len(nk_up) * 100
    print(f"  日経フィルター通過率: {pct_pass:.1f}%（全{len(nk_up)}日中{nk_up.sum()}日）")
    print(f"{'='*70}")

    year_rets: dict[int, list] = defaultdict(list)
    filled = 0

    for ticker, df in all_data.items():
        sig_idx = _signals(df, nk_up, ma_period)
        if len(sig_idx) == 0:
            continue
        for year, ret in _backtest_one(df, sig_idx):
            year_rets[year].append(ret)
            filled += 1

    def wr_of(yr):
        r = np.array(year_rets.get(yr, []))
        if len(r) < 10:
            return None, len(r)
        return len(r[r > 0]) / len(r) * 100, len(r)

    print(f"\n  {'年':>6}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件数':>6}")
    print("  " + "─" * 45)

    all_rets_list = []
    for yr in [2023, 2024, 2025, 2026]:
        r = np.array(year_rets.get(yr, []))
        all_rets_list.extend(r)
        wr, cnt = wr_of(yr)
        if wr is None:
            print(f"  {yr:>6}  (サンプル不足: {cnt}件)")
            continue
        wins   = r[r > 0]
        losses = r[r <= 0]
        avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        flag   = "⚠️" if wr < OK_WR else "✅"
        print(f"  {yr:>6}  {wr:>5.1f}%  {pf:>5.2f}  {ev:>+6.2f}%  {cnt:>6}件  {flag}")

    if all_rets_list:
        rets = np.array(all_rets_list)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        wr  = len(wins) / len(rets) * 100
        avg_w = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses.mean()) if len(losses) > 0 else 0.0
        pf  = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        ev  = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd = filled / trading_days
        print("  " + "─" * 45)
        print(f"  {'全体':>6}  {wr:>5.1f}%  {pf:>5.2f}  {ev:>+6.2f}%  {filled:>6}件  {spd:.2f}件/日")


def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    trading_days   = len(next(iter(all_data.values())))
    sample_df = next(iter(all_data.values()))
    start_date = str(sample_df.index[0].date())
    print(f"銘柄数: {len(all_data)}  取引日数: {trading_days}日  データ期間〜{cache['date']}")

    nikkei_close = fetch_nikkei(start_date)

    for label, ma_period in NIKKEI_FILTERS:
        run_filter(all_data, trading_days, nikkei_close, label, ma_period)

    print(f"\n{'='*70}")
    print("完了")

if __name__ == "__main__":
    main()
