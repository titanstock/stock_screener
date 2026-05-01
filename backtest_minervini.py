#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ミネルヴィニ SEPA 手法バックテスト
=====================================
【トレンドテンプレート】
  ① 株価 > MA50 > MA150 > MA200（パーフェクトオーダー）
  ② MA200が上昇中（slope_days日前より上）
  ③ 株価が52週安値より MIN_RISE_FROM_LOW % 以上高い
  ④ 株価が52週高値の NEAR_HIGH_PCT % 以内

【エントリー（ブレイクアウト）】
  ⑤ 直近 breakout_days 日高値を終値が上回る
  ⑥ 出来高が20日平均の vol_mult 倍以上
  ⑦ 直近 consol_days 日で価格レンジが収縮（値固め）

損切り = エントリー価格の STOP_PCT % 下
エントリー = 翌日始値
利確 = RR 倍
最大保有 = MAX_HOLD 日
"""

import itertools, pickle, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_cache_name = sys.argv[1] if len(sys.argv) > 1 else "backtest_cache.pkl"
CACHE_PATH  = Path(__file__).parent / _cache_name

MIN_HISTORY      = 250   # MA200に最低250日必要
MIN_TURNOVER     = 30_000_000

# トレンドテンプレート（固定）
MIN_RISE_FROM_LOW = 30.0   # 52週安値より30%以上
NEAR_HIGH_PCT     = 25.0   # 52週高値の25%以内

# グリッド
SLOPE_DAYS_LIST   = [20, 40]         # MA200の傾き確認期間
BREAKOUT_DAYS_LIST = [10, 20, 30]    # ブレイクアウトの高値期間
VOL_MULT_LIST     = [1.5, 2.0, 3.0] # 出来高倍率
STOP_PCT_LIST     = [0.05, 0.08, 0.10]  # 損切り幅
CONSOL_DAYS_LIST  = [10]             # 値固め確認期間

RR       = 2.0
MAX_HOLD = 20

OK_WR  = 55.0
OK_SPD = 0.1   # 中型株は銘柄数が少ないので件数基準を緩める


def _atr(hi, lo, cl, period=14):
    tr = np.maximum(hi[1:] - lo[1:],
         np.maximum(np.abs(hi[1:] - cl[:-1]),
                    np.abs(lo[1:] - cl[:-1])))
    atr = np.full(len(cl), np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(cl)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    return atr


def _signals(df: pd.DataFrame, slope_days: int, breakout_days: int,
             vol_mult: float, stop_pct: float, consol_days: int) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    hi = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_to  = pd.Series(c * v).rolling(20).mean().values
    avg_v   = pd.Series(v).rolling(20).mean().values
    ma50    = pd.Series(c).rolling(50).mean().values
    ma150   = pd.Series(c).rolling(150).mean().values
    ma200   = pd.Series(c).rolling(200).mean().values

    liquid  = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    sig = np.zeros(n, dtype=bool)

    for i in range(MIN_HISTORY, n - 1):
        if not liquid[i]:
            continue
        if np.isnan(ma50[i]) or np.isnan(ma150[i]) or np.isnan(ma200[i]):
            continue
        if np.isnan(ma200[i - slope_days]):
            continue

        # ① パーフェクトオーダー
        if not (c[i] > ma50[i] > ma150[i] > ma200[i]):
            continue

        # ② MA200が上昇中
        if ma200[i] <= ma200[i - slope_days]:
            continue

        # ③ 52週安値より30%以上高い
        wk52_lo = np.min(lo[max(0, i - 252): i + 1])
        if c[i] < wk52_lo * (1 + MIN_RISE_FROM_LOW / 100):
            continue

        # ④ 52週高値の25%以内
        wk52_hi = np.max(hi[max(0, i - 252): i + 1])
        if c[i] < wk52_hi * (1 - NEAR_HIGH_PCT / 100):
            continue

        # ⑤ 直近N日高値のブレイクアウト
        window_hi = np.max(hi[max(0, i - breakout_days): i])
        if c[i] <= window_hi:
            continue

        # ⑥ 出来高急増
        if np.isnan(avg_v[i]) or avg_v[i] <= 0:
            continue
        if v[i] < avg_v[i] * vol_mult:
            continue

        # ⑦ 値固め：直近consol_days日の価格レンジが縮小していること
        if i >= consol_days + breakout_days:
            pre_range  = np.max(hi[i - breakout_days - consol_days: i - breakout_days]) - \
                         np.min(lo[i - breakout_days - consol_days: i - breakout_days])
            cons_range = np.max(hi[i - consol_days: i]) - \
                         np.min(lo[i - consol_days: i])
            if pre_range > 0 and cons_range >= pre_range:
                continue  # 収縮していない

        sig[i] = True

    sig[-(MAX_HOLD + 1):] = False
    return np.where(sig)[0]


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray,
                  stop_pct: float) -> list[tuple]:
    c     = df["Close"].values.astype(float)
    hi    = df["High"].values.astype(float)
    lo    = df["Low"].values.astype(float)
    o     = df["Open"].values.astype(float)
    dates = df.index
    n     = len(c)
    results = []

    for si in sig_idx:
        j0 = si + 1
        if j0 >= n:
            continue
        entry = o[j0]
        if entry <= 0:
            continue

        stop = entry * (1 - stop_pct)
        take = entry + (entry - stop) * RR

        exit_price = None
        end = min(si + 1 + MAX_HOLD, n)

        for j in range(j0, end):
            op = o[j]
            if op <= stop:    exit_price = op;    break
            if op >= take:    exit_price = take;   break
            if hi[j] >= take: exit_price = take;   break
            if lo[j] <= stop: exit_price = stop;   break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]

        results.append((dates[si].year, (exit_price - entry) / entry * 100))

    return results


def run_grid(all_data: dict, trading_days: int):
    combos = list(itertools.product(
        SLOPE_DAYS_LIST, BREAKOUT_DAYS_LIST, VOL_MULT_LIST,
        STOP_PCT_LIST, CONSOL_DAYS_LIST
    ))
    print(f"\nミネルヴィニ SEPA型  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"合格基準: 2023/2024/2025 全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'MA傾き':>5} {'HV期間':>6} {'出来高':>6} {'損切':>5} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7} │ "
           f"{'全体':>6}  {'件/日':>6}  判定")
    print(hdr)
    print("  " + "─" * 88)

    passed = []

    for slope_d, bo_days, vol_m, stop_p, consol_d in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, slope_d, bo_days, vol_m, stop_p, consol_d)
            if len(sig_idx) == 0:
                continue
            for year, ret in _backtest_one(df, sig_idx, stop_p):
                year_rets[year].append(ret)
                filled += 1

        if filled < 5:
            continue

        def wr_of(yr):
            r = np.array(year_rets.get(yr, []))
            return len(r[r > 0]) / len(r) * 100 if len(r) >= 5 else None

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        total_wr = len(all_rets[all_rets > 0]) / len(all_rets) * 100 if len(all_rets) > 0 else 0
        spd = filled / trading_days

        wr23, wr24, wr25, wr26 = wr_of(2023), wr_of(2024), wr_of(2025), wr_of(2026)

        ok_y = all(w is not None and w >= OK_WR for w in [wr23, wr24, wr25])
        ok_n = spd >= OK_SPD
        mark = "✅" if ok_y and ok_n else "  "

        def fmt(w):
            if w is None: return "    N/A"
            f = "⚠" if w < OK_WR else " "
            return f"{w:6.1f}%{f}"

        print(f"  {slope_d:>4}日 {bo_days:>5}日 {vol_m:>4.1f}x {stop_p*100:>4.0f}% │ "
              f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)} {fmt(wr26)} │ "
              f"{total_wr:>5.1f}%  {spd:>5.2f}/日  {mark}")

        if ok_y and ok_n:
            passed.append(dict(
                slope_days=slope_d, breakout_days=bo_days, vol_mult=vol_m,
                stop_pct=stop_p, wr23=wr23, wr24=wr24, wr25=wr25,
                total_wr=total_wr, spd=spd, n=filled
            ))

    print("\n" + "=" * 90)
    if passed:
        print(f"✅ 合格 {len(passed)}通り（最小勝率の高い順）:")
        passed.sort(key=lambda x: min(x['wr23'], x['wr24'], x['wr25']), reverse=True)
        for p in passed[:5]:
            print(f"  MA傾き{p['slope_days']}日  HV{p['breakout_days']}日  "
                  f"Vol{p['vol_mult']}x  損切{p['stop_pct']*100:.0f}%  │  "
                  f"2023:{p['wr23']:.1f}%  2024:{p['wr24']:.1f}%  2025:{p['wr25']:.1f}%  "
                  f"全体:{p['total_wr']:.1f}%  {p['spd']:.2f}件/日")
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
