#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ブレイクアウト型 バックテスト
==============================
新高値を出来高急増で更新した銘柄に翌日乗る。
押し目を待たず、勢いそのものに乗る設計。

シグナル条件:
  ① 終値 > 過去N日の終値最高値（N日高値更新）
  ② 出来高 > 20日平均 × vol_mult（出来高急増）
  ③ 終値 > MA75（長期上昇トレンド内）
  ④ 流動性: 20日平均売買代金 ≥ 3,000万円

エントリー: 翌日始値
損切り: シグナル日安値（entry > stop でないとスキップ）
エントリー上限: 損切り ÷ 0.90（リスク上限10%）
"""

import itertools, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY   = 120
MIN_TURNOVER  = 30_000_000
STOP_CAP_PCT  = 0.10
NO_LIMIT_DAYS = 1000

# グリッド
BREAKOUT_DAYS_LIST = [20, 40, 60, 100]   # 何日間の高値更新か
VOL_MULT_LIST      = [1.5, 2.0, 2.5, 3.0]
RR_LIST            = [1.5, 2.0, 2.5, 3.0]
MAX_HOLD_LIST      = [10, 20, 40]

OK_WR  = 55.0
OK_SPD = 0.3


def _signals(df: pd.DataFrame, breakout_days: int,
             vol_mult: float) -> tuple[np.ndarray, np.ndarray]:
    """(signal_indices, past_high_array) を返す"""
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int), np.array([])

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)
    ma75   = pd.Series(c).rolling(75).mean().values

    # 過去N日の終値最高値（当日は含まない）→ 損切りラインになる
    past_high = pd.Series(c).shift(1).rolling(breakout_days).max().values

    mask = (
        liquid &
        ~np.isnan(past_high) &
        (c > past_high) &                          # N日終値高値更新
        ~np.isnan(avg_v) & (avg_v > 0) &
        (v >= avg_v * vol_mult) &                  # 出来高急増
        ~np.isnan(ma75) & (c > ma75)               # MA75上
    )
    mask[-50:] = False
    return np.where(mask)[0], past_high


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray,
                  rr: float, max_hold: int,
                  past_high_arr: np.ndarray) -> list[tuple]:
    c     = df["Close"].values.astype(float)
    h     = df["High"].values.astype(float)
    lo    = df["Low"].values.astype(float)
    o     = df["Open"].values.astype(float)
    dates = df.index
    n     = len(c)
    hold  = max_hold if max_hold > 0 else NO_LIMIT_DAYS
    results = []

    for si in sig_idx:
        # 損切り = ブレイクアウトライン（過去N日終値最高値）
        stop = past_high_arr[si]
        if np.isnan(stop) or stop <= 0:
            continue

        j0 = si + 1
        if j0 >= n:
            continue
        entry = o[j0]
        # 始値がブレイクアウトライン以下 = 失敗ブレイクアウト → スキップ
        if entry <= 0 or entry <= stop:
            continue
        # リスク上限チェック: エントリーが損切りから10%超上 → スキップ
        if entry > stop * (1 + STOP_CAP_PCT):
            continue

        take = entry + (entry - stop) * rr
        exit_price = None
        end = min(si + 1 + hold, n)

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


def run_yearly_grid(all_data: dict, trading_days: int):
    """フェーズ1: breakout_days × vol_mult の年別WRサマリー（RR=2.0, hold=20日固定）"""
    RR_FIX   = 2.0
    HOLD_FIX = 20
    combos = list(itertools.product(BREAKOUT_DAYS_LIST, VOL_MULT_LIST))

    print(f"\n{'='*80}")
    print(f"【フェーズ1】年別WRサーチ  RR={RR_FIX}  最大保有={HOLD_FIX}日")
    print(f"  合格基準: 2023/2024/2025 全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/日")
    print(f"  グリッド: {len(combos)}通り")
    print(f"{'='*80}\n")

    hdr = (f"  {'高値日':>6} {'Vol':>4} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7} │ "
           f"{'全体':>6}  {'件/日':>5}  判定")
    print(hdr)
    print("  " + "─" * 78)

    passed_phase1 = []

    for bd, vm in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            sig_idx, past_high = _signals(df, bd, vm)
            if len(sig_idx) == 0:
                continue
            for year, ret in _backtest_one(df, sig_idx, RR_FIX, HOLD_FIX, past_high):
                year_rets[year].append(ret)
                filled += 1

        if filled < 10:
            continue

        def wr_of(yr):
            r = np.array(year_rets.get(yr, []))
            return len(r[r > 0]) / len(r) * 100 if len(r) >= 10 else None

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        wins = all_rets[all_rets > 0]
        total_wr = len(wins) / len(all_rets) * 100
        spd = filled / trading_days
        wr23, wr24, wr25, wr26 = wr_of(2023), wr_of(2024), wr_of(2025), wr_of(2026)

        ok_y = all(w is not None and w >= OK_WR for w in [wr23, wr24, wr25])
        ok_n = spd >= OK_SPD
        mark = "✅" if ok_y and ok_n else "  "

        def fmt(w):
            if w is None: return "    N/A"
            return f"{w:6.1f}%{'⚠' if w < OK_WR else ' '}"

        print(f"  {bd:>5}日  {vm:>3.1f}x │ "
              f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)} {fmt(wr26)} │ "
              f"{total_wr:>5.1f}%  {spd:>4.2f}/日  {mark}")

        if ok_y and ok_n:
            passed_phase1.append((bd, vm, wr23, wr24, wr25, total_wr, spd))

    return passed_phase1


def run_rr_hold_grid(all_data: dict, trading_days: int, passed: list):
    """フェーズ2: 合格したbreakout_days×vol_mult に対して RR×hold グリッド"""
    if not passed:
        print("\n【フェーズ2】フェーズ1合格なし → スキップ")
        return

    print(f"\n{'='*80}")
    print(f"【フェーズ2】RR × 最大保有 グリッド（合格パラメータのみ）")
    print(f"{'='*80}")

    for bd, vm, *_ in passed:
        # シグナルを事前計算
        sig_map = {}
        for ticker, df in all_data.items():
            idx, past_high = _signals(df, bd, vm)
            if len(idx) > 0:
                sig_map[ticker] = (df, idx, past_high)

        total_sig = sum(len(v[1]) for v in sig_map.values())
        print(f"\n  高値{bd}日  Vol≥{vm}x  シグナル総数: {total_sig}件")
        print(f"  {'RR':>4}  {'保有':>5}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  "
              f"{'件/日':>5}  {'2023':>6}  {'2024':>6}  {'2025':>6}  判定")
        print("  " + "─" * 78)

        for rr, mh in itertools.product(RR_LIST, MAX_HOLD_LIST):
            year_rets: dict[int, list] = defaultdict(list)
            filled = 0

            for ticker, (df, sig_idx, past_high) in sig_map.items():
                for year, ret in _backtest_one(df, sig_idx, rr, mh, past_high):
                    year_rets[year].append(ret)
                    filled += 1

            if filled < 10:
                continue

            def wr_of(yr):
                r = np.array(year_rets.get(yr, []))
                return len(r[r > 0]) / len(r) * 100 if len(r) >= 10 else None

            all_rets = np.array([r for rs in year_rets.values() for r in rs])
            wins   = all_rets[all_rets > 0]
            losses = all_rets[all_rets <= 0]
            wr  = len(wins) / len(all_rets) * 100
            avg_w = float(wins.mean())   if len(wins)   > 0 else 0.0
            avg_l = float(losses.mean()) if len(losses) > 0 else 0.0
            pf  = abs(avg_w / avg_l)     if avg_l != 0  else 0.0
            ev  = wr / 100 * avg_w + (1 - wr / 100) * avg_l
            spd = filled / trading_days
            wr23, wr24, wr25 = wr_of(2023), wr_of(2024), wr_of(2025)

            ok = (wr >= OK_WR and spd >= OK_SPD and
                  all(w is not None and w >= OK_WR for w in [wr23, wr24, wr25]))
            mark = "✅" if ok else "  "
            mh_s = f"{mh}日"

            def fmt(w):
                if w is None: return "  N/A"
                return f"{w:5.1f}%"

            print(f"  {rr:>4.1f}  {mh_s:>5}  {wr:>5.1f}%  {pf:>5.2f}  "
                  f"{ev:>+6.2f}%  {spd:>4.2f}/日  "
                  f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)}  {mark}")


def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    trading_days   = len(next(iter(all_data.values())))
    print(f"銘柄数: {len(all_data)}  取引日数: {trading_days}日  データ期間〜{cache['date']}")

    passed = run_yearly_grid(all_data, trading_days)

    print(f"\n{'='*80}")
    if passed:
        print(f"✅ フェーズ1合格: {len(passed)}通り")
        for bd, vm, w23, w24, w25, total, spd in passed:
            print(f"  高値{bd}日  Vol≥{vm}x │ "
                  f"2023:{w23:.1f}%  2024:{w24:.1f}%  2025:{w25:.1f}%  "
                  f"全体:{total:.1f}%  {spd:.2f}件/日")
    else:
        print("❌ フェーズ1合格なし")

    run_rr_hold_grid(all_data, trading_days, passed)

    print(f"\n{'='*80}")
    print("完了")


if __name__ == "__main__":
    main()
