#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ミネルヴィニ SEPA + トレーリングストップ バックテスト
========================================================
【トレンドテンプレート】
  ① 株価 > MA50 > MA150 > MA200（パーフェクトオーダー）
  ② MA200が上昇中（slope_days日前より上）
  ③ 株価が52週安値より30%以上高い
  ④ 株価が52週高値の25%以内

【エントリー】
  ⑤ 直近 breakout_days 日高値を終値が上回る
  ⑥ 出来高が20日平均の vol_mult 倍以上
  ⑦ 値固め（出来高枯れ + レンジ収縮）
     - ベース前半(40〜20日前)平均出来高 vs 収縮フェーズ(直近20日)平均出来高
     - 収縮フェーズ出来高 < ベース前半 × 0.7（30%以上減少）
     - 収縮フェーズのレンジ < ベース前半のレンジ

【決済】
  初期損切り = 収縮フェーズ（直近20日）の最安値 × 0.99（最大-15%キャップ）
  トレーリングストップ = 保有中の最高値から trail_pct % 下
  → 利確ラインなし。株価が上がるほどストップが上がる
  最大保有 = MAX_HOLD 日
"""

import itertools, pickle, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_cache_name = sys.argv[1] if len(sys.argv) > 1 else "backtest_midcap_5y_cache.pkl"
CACHE_PATH  = Path(__file__).parent / _cache_name

MIN_HISTORY       = 250
MIN_TURNOVER      = 30_000_000
MIN_RISE_FROM_LOW = 30.0
NEAR_HIGH_PCT     = 25.0
MAX_STOP_PCT      = 0.15   # 初期損切り上限（収縮安値が遠すぎる場合のキャップ）

# グリッド（損切りは動的になったのでSTOP_PCT_LISTを廃止）
SLOPE_DAYS_LIST    = [20]
BREAKOUT_DAYS_LIST = [20, 30]
VOL_MULT_LIST      = [1.5, 2.0]
TRAIL_PCT_LIST     = [0.10, 0.15, 0.20]

MAX_HOLD = 60

OK_PF  = 1.5
OK_SPD = 0.1


def _signals(df: pd.DataFrame, slope_days: int, breakout_days: int,
             vol_mult: float) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    hi = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_to = pd.Series(c * v).rolling(20).mean().values
    avg_v  = pd.Series(v).rolling(20).mean().values
    ma50   = pd.Series(c).rolling(50).mean().values
    ma150  = pd.Series(c).rolling(150).mean().values
    ma200  = pd.Series(c).rolling(200).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    sig = np.zeros(n, dtype=bool)

    for i in range(MIN_HISTORY, n - 1):
        if not liquid[i]: continue
        if np.isnan(ma50[i]) or np.isnan(ma150[i]) or np.isnan(ma200[i]): continue
        if np.isnan(ma200[i - slope_days]): continue

        if not (c[i] > ma50[i] > ma150[i] > ma200[i]): continue
        if ma200[i] <= ma200[i - slope_days]: continue

        wk52_lo = np.min(lo[max(0, i - 252): i + 1])
        if c[i] < wk52_lo * (1 + MIN_RISE_FROM_LOW / 100): continue

        wk52_hi = np.max(hi[max(0, i - 252): i + 1])
        if c[i] < wk52_hi * (1 - NEAR_HIGH_PCT / 100): continue

        window_hi = np.max(hi[max(0, i - breakout_days): i])
        if c[i] <= window_hi: continue

        if np.isnan(avg_v[i]) or avg_v[i] <= 0: continue
        if v[i] < avg_v[i] * vol_mult: continue

        # 値固め確認（ベース前半40〜20日前 vs 収縮フェーズ直近20日）
        if i < 40:
            continue
        base_vol   = np.mean(v[i - 39: i - 19])  # 40〜20日前（20本）
        shrink_vol = np.mean(v[i - 19: i + 1])   # 直近20日（今日含む）
        volume_drying = base_vol > 0 and shrink_vol < base_vol * 0.8

        pre_hi  = np.max(hi[i - 39: i - 19])
        pre_lo  = np.min(lo[i - 39: i - 19])
        cons_hi = np.max(hi[i - 19: i])
        cons_lo = np.min(lo[i - 19: i])
        range_contracting = (pre_hi - pre_lo) > 0 and (cons_hi - cons_lo) < (pre_hi - pre_lo)

        if not (volume_drying and range_contracting):
            continue

        sig[i] = True

    sig[-(MAX_HOLD + 1):] = False
    return np.where(sig)[0]


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray,
                  trail_pct: float) -> list[tuple]:
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

        # 初期損切り = 収縮フェーズ（直近20日）の最安値 × 0.99、最大-15%キャップ
        consol_lo    = np.min(lo[max(0, si - 19): si])
        initial_stop = consol_lo * 0.99
        initial_stop = max(initial_stop, entry * (1 - MAX_STOP_PCT))

        trailing_stop = initial_stop
        highest       = entry
        exit_price    = None
        hold_days     = 0
        end           = min(si + 1 + MAX_HOLD, n)

        for j in range(j0, end):
            op = o[j]

            if op <= trailing_stop:
                exit_price = op
                hold_days  = j - j0
                break

            if hi[j] > highest:
                highest = hi[j]
                trailing_stop = max(trailing_stop, highest * (1 - trail_pct))

            if lo[j] <= trailing_stop:
                exit_price = trailing_stop
                hold_days  = j - j0
                break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]
            hold_days  = end - 1 - j0

        ret = (exit_price - entry) / entry * 100
        results.append((dates[si].year, ret, hold_days))

    return results


def run_grid(all_data: dict, trading_days: int):
    combos = list(itertools.product(
        SLOPE_DAYS_LIST, BREAKOUT_DAYS_LIST, VOL_MULT_LIST, TRAIL_PCT_LIST
    ))
    print(f"\nミネルヴィニ SEPA + トレーリングストップ  最大保有={MAX_HOLD}日")
    print(f"損切り: 収縮フェーズ最安値×0.99（上限-{MAX_STOP_PCT*100:.0f}%）")
    print(f"値固め: 出来高20%枯れ + レンジ収縮（40日ベース比較）")
    print(f"合格基準: 2022〜2025 全年PF≥{OK_PF} / 件数≥{OK_SPD}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'HV':>5} {'Vol':>5} {'Trail':>5} │ "
           f"{'──────────────── PF ────────────────':^42} │ 全体PF   EV  /日  保有")
    hdr2 = (f"  {'':>5} {'':>5} {'':>5} │ "
            f"{'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7}  │")
    print(hdr)
    print(hdr2)
    print("  " + "─" * 110)

    passed = []

    for slope_d, bo_days, vol_m, trail_p in combos:
        year_rets:  dict[int, list] = defaultdict(list)
        hold_days_all: list[int]    = []
        filled = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, slope_d, bo_days, vol_m)
            if len(sig_idx) == 0:
                continue
            for year, ret, hd in _backtest_one(df, sig_idx, trail_p):
                year_rets[year].append(ret)
                hold_days_all.append(hd)
                filled += 1

        if filled < 5:
            continue

        def stats_of(yr):
            r = np.array(year_rets.get(yr, []))
            if len(r) < 5:
                return None
            w = r[r > 0]; l = r[r <= 0]
            pf_yr = (len(w) * w.mean()) / (len(l) * abs(l.mean())) \
                if len(l) > 0 and len(w) > 0 else 0
            return pf_yr

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        wins     = all_rets[all_rets > 0]
        losses   = all_rets[all_rets <= 0]
        avg_win  = wins.mean()        if len(wins)   > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses)  > 0 else 1
        pf       = (len(wins) * avg_win) / (len(losses) * avg_loss) \
                   if len(losses) > 0 and avg_loss > 0 else 0
        ev       = all_rets.mean()
        spd      = filled / trading_days
        avg_hold = np.mean(hold_days_all) if hold_days_all else 0

        pf22 = stats_of(2022)
        pf23 = stats_of(2023)
        pf24 = stats_of(2024)
        pf25 = stats_of(2025)
        pf26 = stats_of(2026)

        ok_pf_yr = all(p is not None and p >= OK_PF for p in [pf22, pf23, pf24, pf25])
        ok_n     = spd >= OK_SPD
        mark     = "✅" if ok_pf_yr and ok_n else "  "

        n22 = len(year_rets.get(2022, []))
        n23 = len(year_rets.get(2023, []))
        n24 = len(year_rets.get(2024, []))
        n25 = len(year_rets.get(2025, []))
        n26 = len(year_rets.get(2026, []))

        def fmt_pf(p):
            if p is None: return "    N/A"
            f = "⚠" if p < OK_PF else " "
            return f"{p:5.2f}{f} "

        print(f"  {bo_days:>5}日 {vol_m:>4.1f}x {trail_p*100:>5.0f}% │ "
              f"PF {fmt_pf(pf22)} {fmt_pf(pf23)} {fmt_pf(pf24)} {fmt_pf(pf25)} {fmt_pf(pf26)} │ "
              f"全体PF:{pf:>5.2f} EV:{ev:>+5.2f}% {spd:>4.2f}/日 {avg_hold:>4.1f}日 {mark}")
        print(f"  {'':>5}    {'':>4}  {'':>5} │ "
              f"N  {n22:>6}  {n23:>6}  {n24:>6}  {n25:>6}  {n26:>6}   │")

        if ok_pf_yr and ok_n:
            passed.append(dict(
                breakout_days=bo_days, vol_mult=vol_m, trail_pct=trail_p,
                pf22=pf22, pf23=pf23, pf24=pf24, pf25=pf25,
                pf=pf, ev=ev, spd=spd, avg_hold=avg_hold
            ))

    print("\n" + "=" * 102)
    if passed:
        print(f"✅ 合格 {len(passed)}通り（PF順）:")
        passed.sort(key=lambda x: x['pf'], reverse=True)
        for p in passed[:5]:
            print(f"  HV{p['breakout_days']}日  Vol{p['vol_mult']}x  Trail{p['trail_pct']*100:.0f}%  │  "
                  f"PF: 2022:{p['pf22']:.2f} 2023:{p['pf23']:.2f} 2024:{p['pf24']:.2f} 2025:{p['pf25']:.2f}  "
                  f"全体:{p['pf']:.2f}  EV:{p['ev']:+.2f}%  {p['spd']:.2f}件/日  平均{p['avg_hold']:.1f}日保有")
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
