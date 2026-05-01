#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安値切り上がり + MA25押し目型 バックテスト
==========================================
設計:
  ① MA25が右肩上がり（直近N日でMA25が上昇）
  ② 直近スイング安値が2回以上切り上がっている
  ③ 株価がMA25の上からMA25に接近（MA25の100〜X%以内）
  ④ 陽線 or 出来高条件（グリッドで変化）

損切り = シグナル日安値（上限 = 安値÷0.90）
エントリー = 翌日始値
RR = 2.0 / 最大保有 = 20日
"""

import itertools, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
_cache_name  = sys.argv[1] if len(sys.argv) > 1 else "backtest_cache.pkl"
CACHE_PATH   = Path(__file__).parent / _cache_name
MIN_HISTORY  = 100
MIN_TURNOVER = 30_000_000
STOP_CAP_PCT = 0.10

RR       = 2.0
MAX_HOLD = 20

# グリッド
MA25_NEAR_LIST    = [1.03, 1.05, 1.08]   # close ≤ MA25 × X（上からの接近度）
MA_SLOPE_DAYS_LIST = [5, 10]              # MA25の傾き確認期間（N日前より上か）
SWING_LOOKBACK_LIST = [3, 5]             # スイング安値の前後N本
REQUIRE_YANG      = [True, False]        # 陽線必須か

OK_WR  = 55.0
OK_SPD = 0.3


def _find_swing_low_values(lo: np.ndarray, lookback: int) -> list[float]:
    """スイング安値の価格リストを返す（古い順）"""
    result = []
    for i in range(lookback, len(lo) - lookback):
        if (lo[i] == np.min(lo[i - lookback: i + lookback + 1])):
            result.append(lo[i])
    return result


def _signals(df: pd.DataFrame, ma25_near: float, slope_days: int,
             swing_lb: int, yang: bool) -> np.ndarray:
    c  = df["Close"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    ma25 = pd.Series(c).rolling(25).mean().values

    sig = np.zeros(n, dtype=bool)

    for i in range(MIN_HISTORY, n - 1):
        if not liquid[i]:
            continue
        if np.isnan(ma25[i]) or np.isnan(ma25[i - slope_days]):
            continue

        # ① MA25が右肩上がり
        if ma25[i] <= ma25[i - slope_days]:
            continue

        # ② 株価がMA25の上から接近中（MA25 ≤ close ≤ MA25 × near）
        if not (ma25[i] <= c[i] <= ma25[i] * ma25_near):
            continue

        # ③ 直近スイング安値が切り上がっている（直近window内で検索）
        window = lo[max(0, i - 60): i - swing_lb + 1]
        swing_lows = _find_swing_low_values(window, swing_lb)
        if len(swing_lows) < 2:
            continue
        # 直近2点が切り上がっているか
        if swing_lows[-1] <= swing_lows[-2]:
            continue

        # ④ 陽線
        if yang and c[i] <= o[i]:
            continue

        sig[i] = True

    sig[-(MAX_HOLD + 1):] = False
    return np.where(sig)[0]


def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray) -> list[tuple]:
    c     = df["Close"].values.astype(float)
    hi    = df["High"].values.astype(float)
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
        MA25_NEAR_LIST, MA_SLOPE_DAYS_LIST, SWING_LOOKBACK_LIST, REQUIRE_YANG
    ))
    print(f"\n安値切り上がり + MA25押し目型  RR={RR}  最大保有={MAX_HOLD}日")
    print(f"合格基準: 2023/2024/2025 全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/日")
    print(f"グリッド: {len(combos)}通り\n")

    hdr = (f"  {'MA25接近':>8} {'傾き確認':>6} {'SW前後':>6} {'陽線':>4} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7} │ "
           f"{'全体':>6}  {'件/日':>6}  判定")
    print(hdr)
    print("  " + "─" * 90)

    passed = []

    for ma25_near, slope_days, swing_lb, yang in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0

        for ticker, df in all_data.items():
            sig_idx = _signals(df, ma25_near, slope_days, swing_lb, yang)
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

        yang_s   = "あり" if yang else "なし"
        near_s   = f"+{(ma25_near - 1)*100:.0f}%"
        print(f"  {near_s:>8} {slope_days:>4}日 {swing_lb:>5}本 {yang_s:>4} │ "
              f"{fmt(wr23)} {fmt(wr24)} {fmt(wr25)} {fmt(wr26)} │ "
              f"{total_wr:>5.1f}%  {spd:>5.2f}/日  {mark}")

        if ok_y and ok_n:
            passed.append(dict(
                ma25_near=ma25_near, slope_days=slope_days, swing_lb=swing_lb, yang=yang,
                wr23=wr23, wr24=wr24, wr25=wr25, total_wr=total_wr, spd=spd, n=filled
            ))

    print("\n" + "=" * 92)
    if passed:
        print(f"✅ 合格 {len(passed)}通り（上位5件）:")
        passed.sort(key=lambda x: min(x['wr23'], x['wr24'], x['wr25']), reverse=True)
        for p in passed[:5]:
            yang_s = "陽線あり" if p['yang'] else "陽線なし"
            print(f"  MA25+{(p['ma25_near']-1)*100:.0f}%以内  傾き{p['slope_days']}日  SW{p['swing_lb']}本  {yang_s}  │  "
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
