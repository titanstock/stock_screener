#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上昇トレンド押し目型 バックテスト
===================================
設計: MA25 > MA75 の上昇トレンド中、MA25割れの静かな押し目で仕込む
     → MA25回復を待たない（回復後に入ると高値掴み）

シグナル条件（グリッドで変化させる）:
  ① MA25 > MA75（中期上昇）
  ② MA75*1.0 < close < MA25*threshold  ← MA25割れ中でMA75は維持
  ③ 出来高 < vol_cap * avg_vol（静かな調整）
  ④ close > open（陽線）

固定:
  損切り = シグナル日安値
  エントリー上限 = 安値 ÷ 0.90
  エントリー = 翌日始値
  RR = 2.0 / 最大保有 = 20日
"""

import itertools, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY   = 100
MIN_TURNOVER  = 30_000_000
STOP_CAP_PCT  = 0.10

RR       = 2.0
MAX_HOLD = 20

# グリッド
MA25_THRESH_LIST = [1.00, 0.99, 0.97]   # close < MA25 * thresh（0.99=1%割れまで）
VOL_CAP_LIST     = [0.8, 1.0, 1.5]      # 出来高 < cap * 平均
REQUIRE_YANG     = [True, False]         # 陽線必須か

OK_WR   = 55.0
OK_SPD  = 0.3


def _signals(df: pd.DataFrame, ma25_thresh: float,
             vol_cap: float, yang: bool) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    ma25 = pd.Series(c).rolling(25).mean().values
    ma75 = pd.Series(c).rolling(75).mean().values

    # ① 中期上昇
    uptrend = (ma25 > ma75) & ~np.isnan(ma25) & ~np.isnan(ma75)

    # ② MA25割れ中 かつ MA75は維持
    below_ma25 = (c < ma25 * ma25_thresh) & (c > ma75)

    # ③ 出来高静か
    quiet_vol = (avg_v > 0) & (v < avg_v * vol_cap) & ~np.isnan(avg_v)

    # ④ 陽線
    bullish = (c > o) if yang else np.ones(n, dtype=bool)

    mask = liquid & uptrend & below_ma25 & quiet_vol & bullish
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
    combos = list(itertools.product(MA25_THRESH_LIST, VOL_CAP_LIST, REQUIRE_YANG))
    print(f"\n上昇トレンド押し目型 グリッドサーチ  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"合格基準: 2023/2024/2025 全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'MA25割':>6} {'Vol上限':>6} {'陽線':>4} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7} │ "
           f"{'全体':>6}  {'件/日':>5}  {'判定':>4}")
    print(hdr)
    print("  " + "─" * 82)

    passed = []

    for ma25_t, vol_c, yang in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, ma25_t, vol_c, yang)
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
            f = "⚠" if w < OK_WR else " "
            return f"{w:6.1f}%{f}"

        yang_s = "あり" if yang else "なし"
        thresh_s = f"{(1-ma25_t)*100:.0f}%↓" if ma25_t < 1 else "直下"
        print(f"  {thresh_s:>6} {vol_c:>4.1f}x  {yang_s:>4} │ "
              f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)} {fmt(wr26)} │ "
              f"{total_wr:>5.1f}%  {spd:>4.2f}/日  {mark}")

        if ok_y and ok_n:
            passed.append(dict(ma25_thresh=ma25_t, vol_cap=vol_c, yang=yang,
                               wr23=wr23, wr24=wr24, wr25=wr25,
                               total_wr=total_wr, spd=spd, n=filled))

    print("\n" + "=" * 84)
    if passed:
        print(f"✅ 合格 {len(passed)}通り（上位5件）:")
        passed.sort(key=lambda x: x['wr23'])
        for p in passed[:5]:
            thresh_s = f"{(1-p['ma25_thresh'])*100:.0f}%割れ" if p['ma25_thresh'] < 1 else "直下割れ"
            yang_s = "陽線あり" if p['yang'] else "陽線なし"
            print(f"  MA25{thresh_s}  Vol<{p['vol_cap']}x  {yang_s}  │  "
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
