#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A案: 52週高値更新後の押し目型
================================
新高値を取った勢いある銘柄が静かに押したところを狙う。
モメンタムが確認済みなので、押しが一時的になりやすい。

シグナル条件（グリッドで変化）:
  ① 過去252日の最高値が直近N日以内に記録された（勢いがある）
  ② 現在の終値がその高値からX%以上下落している（押しが入っている）
  ③ 終値 > MA75（トレンド崩壊ではない）
  ④ 出来高 < avg * vol_cap（静かな押し目）

固定:
  損切り = シグナル日安値 / エントリー上限 = 安値÷0.90
  エントリー = 翌日始値 / RR = 2.0 / 最大保有 = 20日
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
RR       = 2.0
MAX_HOLD = 20

# グリッド
HIGH_DAYS_LIST   = [20, 30, 40]       # 高値更新から何日以内か
PULLBACK_LIST    = [3.0, 5.0, 8.0]    # 高値からの下落率（%）以上
VOL_CAP_LIST     = [1.0, 1.5]         # 出来高 < cap * 平均

OK_WR  = 55.0
OK_SPD = 0.3


def _signals(df: pd.DataFrame, high_days: int,
             pullback_pct: float, vol_cap: float) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)
    ma75   = pd.Series(c).rolling(75).mean().values

    mask = np.zeros(n, dtype=bool)
    year_high = pd.Series(c).rolling(252, min_periods=60).max().values

    for i in range(MIN_HISTORY, n - 1):
        if not liquid[i]:
            continue
        if np.isnan(ma75[i]) or c[i] <= ma75[i]:
            continue
        if np.isnan(avg_v[i]) or avg_v[i] <= 0:
            continue
        # 出来高チェック
        if v[i] >= avg_v[i] * vol_cap:
            continue
        # 52週高値が直近 high_days 以内に記録されたか
        yh = year_high[i]
        if np.isnan(yh) or yh <= 0:
            continue
        window = c[max(0, i - high_days + 1):i + 1]
        if np.max(window) < yh * 0.995:   # 直近windowに最高値がない
            continue
        # 現在の終値が高値からX%以上下落しているか
        drop = (yh - c[i]) / yh * 100
        if drop < pullback_pct:
            continue
        # 下がりすぎていないか（MA75より上を確認済み）
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


def run_grid(all_data: dict, trading_days: int):
    combos = list(itertools.product(HIGH_DAYS_LIST, PULLBACK_LIST, VOL_CAP_LIST))
    print(f"\n【A案】52週高値更新後の押し目型  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"合格基準: 2023/2024/2025 全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'高値N日':>6} {'下落%':>5} {'Vol':>4} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7} │ "
           f"{'全体':>6}  {'件/日':>5}  判定")
    print(hdr)
    print("  " + "─" * 82)

    passed = []
    for hd, pb, vc in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, hd, pb, vc)
            if len(sig_idx) == 0:
                continue
            for year, ret in _backtest_one(df, sig_idx):
                year_rets[year].append(ret)
                filled += 1

        if filled < 5:
            continue

        def wr_of(yr):
            r = np.array(year_rets.get(yr, []))
            return len(r[r > 0]) / len(r) * 100 if len(r) >= 10 else None

        all_rets = np.array([r for rs in year_rets.values() for r in rs])
        wins = all_rets[all_rets > 0]
        total_wr = len(wins) / len(all_rets) * 100 if len(all_rets) > 0 else 0
        spd = filled / trading_days
        wr23, wr24, wr25, wr26 = wr_of(2023), wr_of(2024), wr_of(2025), wr_of(2026)

        ok_y = all(w is not None and w >= OK_WR for w in [wr23, wr24, wr25])
        ok_n = spd >= OK_SPD
        mark = "✅" if ok_y and ok_n else "  "

        def fmt(w):
            if w is None: return "    N/A"
            return f"{w:6.1f}%{'⚠' if w < OK_WR else ' '}"

        print(f"  {hd:>5}日  {pb:>4.0f}%↓  {vc:>3.1f}x │ "
              f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)} {fmt(wr26)} │ "
              f"{total_wr:>5.1f}%  {spd:>4.2f}/日  {mark}")

        if ok_y and ok_n:
            passed.append(dict(high_days=hd, pullback=pb, vol_cap=vc,
                               wr23=wr23, wr24=wr24, wr25=wr25,
                               total_wr=total_wr, spd=spd))

    print("\n" + "=" * 84)
    if passed:
        print(f"✅ 合格 {len(passed)}通り:")
        for p in passed:
            print(f"  高値{p['high_days']}日以内  下落≥{p['pullback']:.0f}%  Vol<{p['vol_cap']}x  │  "
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
