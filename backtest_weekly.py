#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
週足バックテスト
================
日足データを週足に変換して戦略を検証する。
翌日始値の高値掴み問題 → 翌週始値問題に緩和（週次ノイズが平滑化）

【検証戦略】
  A: 週足ブレイクアウト型  - N週高値更新 + 出来高急増
  B: 週足押し目型         - 上昇トレンド中の週足MA割れ → 翌週始値

エントリー: 翌週月曜始値（= 翌週Open）
損切り:
  A: シグナル週の安値 / B: シグナル週の安値
エントリー上限: 損切り ÷ 0.90（リスク10%以内）
"""

import itertools, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
MIN_WEEKS     = 30        # 最低週数
MIN_TURNOVER  = 30_000_000
STOP_CAP_PCT  = 0.10
NO_LIMIT_WEEKS = 200

OK_WR  = 55.0
OK_SPD = 0.05   # 件/週（週足なので基準を下げる）


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日足 → 週足（金曜基準）"""
    w = df.resample("W-FRI").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Close"])
    # 出来高ゼロ週除外
    return w[w["Volume"] > 0]


# ══════════════════════════════════════════════════════════════════════════════
# シグナル生成
# ══════════════════════════════════════════════════════════════════════════════

def signals_breakout(wdf: pd.DataFrame, bo_weeks: int,
                     vol_mult: float) -> tuple[np.ndarray, np.ndarray]:
    """週足ブレイクアウト: N週終値高値更新 + 出来高急増 + MA26上"""
    c  = wdf["Close"].values.astype(float)
    v  = wdf["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_WEEKS:
        return np.array([], dtype=int), np.array([])

    avg_v   = pd.Series(v).rolling(20).mean().values
    avg_to  = pd.Series(c * v).rolling(20).mean().values
    liquid  = (avg_to >= MIN_TURNOVER * 5) & ~np.isnan(avg_to)  # 週足なので×5
    ma26    = pd.Series(c).rolling(26).mean().values

    past_high = pd.Series(c).shift(1).rolling(bo_weeks).max().values

    mask = (
        liquid &
        ~np.isnan(past_high) &
        (c > past_high) &
        ~np.isnan(avg_v) & (avg_v > 0) &
        (v >= avg_v * vol_mult) &
        ~np.isnan(ma26) & (c > ma26)
    )
    mask[-10:] = False
    return np.where(mask)[0], past_high


def signals_pullback(wdf: pd.DataFrame, ma_slow: int,
                     vol_cap: float) -> np.ndarray:
    """週足押し目: MA13>MA26 上昇中、終値がMA13割れ + 静か出来高"""
    c  = wdf["Close"].values.astype(float)
    lo = wdf["Low"].values.astype(float)
    v  = wdf["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_WEEKS:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    liquid = (avg_to >= MIN_TURNOVER * 5) & ~np.isnan(avg_to)

    ma13 = pd.Series(c).rolling(13).mean().values
    ma26 = pd.Series(c).rolling(ma_slow).mean().values

    mask = (
        liquid &
        ~np.isnan(ma13) & ~np.isnan(ma26) &
        (ma13 > ma26) &          # 上昇トレンド
        (c < ma13) &             # MA13割れ（押し目）
        (c > ma26) &             # MA26は維持
        ~np.isnan(avg_v) & (avg_v > 0) &
        (v < avg_v * vol_cap)    # 静かな調整
    )
    mask[-10:] = False
    return np.where(mask)[0]


# ══════════════════════════════════════════════════════════════════════════════
# バックテスト（共通）
# ══════════════════════════════════════════════════════════════════════════════

def backtest_one(wdf: pd.DataFrame, sig_idx: np.ndarray,
                 rr: float, max_hold: int,
                 stop_arr: np.ndarray = None) -> list[tuple]:
    """
    stop_arr: ブレイクアウト型はpast_high、押し目型はNone（→シグナル週安値を使う）
    returns: [(year, ret%), ...]
    """
    c     = wdf["Close"].values.astype(float)
    h     = wdf["High"].values.astype(float)
    lo    = wdf["Low"].values.astype(float)
    o     = wdf["Open"].values.astype(float)
    dates = wdf.index
    n     = len(c)
    hold  = max_hold if max_hold > 0 else NO_LIMIT_WEEKS
    results = []

    for si in sig_idx:
        # 損切りライン
        stop = stop_arr[si] if stop_arr is not None else lo[si]
        if np.isnan(stop) or stop <= 0:
            continue

        j0 = si + 1
        if j0 >= n:
            continue
        entry = o[j0]
        if entry <= 0 or entry <= stop:
            continue
        if entry > stop / (1 - STOP_CAP_PCT):   # リスク上限10%
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


# ══════════════════════════════════════════════════════════════════════════════
# グリッドサーチ共通ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def print_yearly_table(label: str, rr: float, hold: int,
                       year_rets: dict, total_filled: int, total_weeks: int):
    spd = total_filled / total_weeks if total_weeks > 0 else 0

    def wr_stats(yr):
        r = np.array(year_rets.get(yr, []))
        if len(r) < 5:
            return None, None, None, len(r)
        wins = r[r > 0]; losses = r[r <= 0]
        wr = len(wins) / len(r) * 100
        avg_w = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses.mean()) if len(losses) > 0 else 0.0
        pf = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        ev = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        return wr, pf, ev, len(r)

    all_r = np.array([r for rs in year_rets.values() for r in rs])
    if len(all_r) == 0:
        print(f"  {label}: サンプルなし")
        return False

    wins = all_r[all_r > 0]; losses = all_r[all_r <= 0]
    t_wr = len(wins) / len(all_r) * 100
    t_pf = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 0.0
    t_ev = t_wr / 100 * (wins.mean() if len(wins) > 0 else 0) + \
           (1 - t_wr / 100) * (losses.mean() if len(losses) > 0 else 0)

    print(f"\n  {label}  RR={rr}  保有≤{hold}週")
    print(f"  {'年':>5}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件数':>5}")
    print("  " + "─" * 42)

    all_ok = True
    for yr in [2023, 2024, 2025, 2026]:
        wr, pf, ev, cnt = wr_stats(yr)
        if wr is None:
            print(f"  {yr:>5}  (少 {cnt}件)")
            if yr in [2023, 2024, 2025]:
                all_ok = False
            continue
        flag = "✅" if wr >= OK_WR else "⚠️"
        print(f"  {yr:>5}  {wr:>5.1f}%  {pf:>5.2f}  {ev:>+6.2f}%  {cnt:>5}件  {flag}")
        if yr in [2023, 2024, 2025] and wr < OK_WR:
            all_ok = False

    print("  " + "─" * 42)
    ok_n = spd >= OK_SPD
    mark = "✅" if all_ok and ok_n else "  "
    print(f"  {'全体':>5}  {t_wr:>5.1f}%  {t_pf:>5.2f}  {t_ev:>+6.2f}%  "
          f"{total_filled:>5}件  {spd:.3f}件/週  {mark}")
    return all_ok and ok_n


# ══════════════════════════════════════════════════════════════════════════════
# A: 週足ブレイクアウト
# ══════════════════════════════════════════════════════════════════════════════

def run_breakout(weekly_data: dict, total_weeks: int):
    BO_WEEKS_LIST = [13, 26, 52]      # 13週=3ヶ月, 26週=半年, 52週=1年
    VOL_MULT_LIST = [1.5, 2.0, 2.5]
    RR_LIST       = [1.5, 2.0, 2.5]
    HOLD_LIST     = [4, 8, 13]       # 週数

    print(f"\n{'='*70}")
    print("【A】週足ブレイクアウト型")
    print(f"  合格基準: 2023/2024/2025全年WR≥{OK_WR}% / 件数≥{OK_SPD}件/週")
    print(f"{'='*70}")

    combos_p1 = list(itertools.product(BO_WEEKS_LIST, VOL_MULT_LIST))
    RR_FIX, HOLD_FIX = 2.0, 8

    print(f"\nフェーズ1: 高値期間×出来高  (RR={RR_FIX}, 保有{HOLD_FIX}週固定)\n")
    hdr = (f"  {'高値週':>5} {'Vol':>4} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>6} │ "
           f"{'全体':>6}  {'件/週':>6}  判定")
    print(hdr); print("  " + "─" * 72)

    passed = []
    for bw, vm in combos_p1:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0
        for ticker, wdf in weekly_data.items():
            idx, past_high = signals_breakout(wdf, bw, vm)
            if len(idx) == 0:
                continue
            for yr, ret in backtest_one(wdf, idx, RR_FIX, HOLD_FIX, past_high):
                year_rets[yr].append(ret)
                filled += 1

        def wr(yr):
            r = np.array(year_rets.get(yr, []))
            return f"{len(r[r>0])/len(r)*100:6.1f}%{'⚠' if len(r[r>0])/len(r)*100<OK_WR else ' '}" \
                   if len(r) >= 5 else "   N/A "

        all_r = np.array([r for rs in year_rets.values() for r in rs])
        if len(all_r) == 0:
            continue
        t_wr = len(all_r[all_r > 0]) / len(all_r) * 100
        spd  = filled / total_weeks
        ok_y = all(len(np.array(year_rets.get(y,[]))) >= 5 and
                   len(np.array(year_rets.get(y,[]))[np.array(year_rets.get(y,[])) > 0]) /
                   len(np.array(year_rets.get(y,[]))) * 100 >= OK_WR
                   for y in [2023, 2024, 2025])
        mark = "✅" if ok_y and spd >= OK_SPD else "  "
        print(f"  {bw:>4}週  {vm:>3.1f}x │ "
              f"{wr(2023)} {wr(2024)} {wr(2025)} {wr(2026)} │ "
              f"{t_wr:>5.1f}%  {spd:>5.3f}/週  {mark}")
        if ok_y and spd >= OK_SPD:
            passed.append((bw, vm))

    if not passed:
        print("\n  ❌ フェーズ1合格なし")
        return

    print(f"\nフェーズ2: RR×保有週グリッド（合格パラメータのみ）")
    for bw, vm in passed:
        sig_cache = {}
        for ticker, wdf in weekly_data.items():
            idx, ph = signals_breakout(wdf, bw, vm)
            if len(idx) > 0:
                sig_cache[ticker] = (wdf, idx, ph)

        print(f"\n  高値{bw}週  Vol≥{vm}x")
        print(f"  {'RR':>4}  {'保有':>5}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  "
              f"{'件/週':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  判定")
        print("  " + "─" * 72)

        for rr, mh in itertools.product(RR_LIST, HOLD_LIST):
            yr2 = defaultdict(list); f2 = 0
            for ticker, (wdf, idx, ph) in sig_cache.items():
                for yr, ret in backtest_one(wdf, idx, rr, mh, ph):
                    yr2[yr].append(ret); f2 += 1
            if f2 < 5:
                continue
            all_r = np.array([r for rs in yr2.values() for r in rs])
            wins = all_r[all_r > 0]; losses = all_r[all_r <= 0]
            t_wr = len(wins)/len(all_r)*100
            t_pf = abs(wins.mean()/losses.mean()) if len(losses)>0 else 0
            t_ev = t_wr/100*(wins.mean() if len(wins)>0 else 0) + \
                   (1-t_wr/100)*(losses.mean() if len(losses)>0 else 0)
            spd  = f2/total_weeks

            def fw(y):
                r = np.array(yr2.get(y, []))
                return f"{len(r[r>0])/len(r)*100:5.1f}%" if len(r)>=5 else "  N/A"

            ok_y = all(len(np.array(yr2.get(y,[])))>=5 and
                       len(np.array(yr2.get(y,[]))[np.array(yr2.get(y,[]))>0])/
                       len(np.array(yr2.get(y,[])))*100>=OK_WR
                       for y in [2023,2024,2025])
            mark = "✅" if ok_y and spd>=OK_SPD else "  "
            print(f"  {rr:>4.1f}  {mh:>4}週  {t_wr:>5.1f}%  {t_pf:>5.2f}  "
                  f"{t_ev:>+6.2f}%  {spd:>5.3f}/週  "
                  f"{fw(2023)} {fw(2024)} {fw(2025)}  {mark}")


# ══════════════════════════════════════════════════════════════════════════════
# B: 週足押し目
# ══════════════════════════════════════════════════════════════════════════════

def run_pullback(weekly_data: dict, total_weeks: int):
    MA_SLOW_LIST  = [26, 52]
    VOL_CAP_LIST  = [0.8, 1.0, 1.5]
    RR_LIST       = [1.5, 2.0, 2.5]
    HOLD_LIST     = [4, 8, 13]

    print(f"\n{'='*70}")
    print("【B】週足押し目型（MA13>MA26 上昇中のMA13割れ）")
    print(f"{'='*70}\n")

    combos = list(itertools.product(MA_SLOW_LIST, VOL_CAP_LIST))
    RR_FIX, HOLD_FIX = 2.0, 8

    hdr = (f"  {'MAスロー':>6} {'Vol上限':>6} │ "
           f"{'2023':>7} {'2024':>7} {'2025':>7} {'2026':>6} │ "
           f"{'全体':>6}  {'件/週':>6}  判定")
    print(hdr); print("  " + "─" * 72)

    passed = []
    for ms, vc in combos:
        year_rets: dict[int, list] = defaultdict(list)
        filled = 0
        for ticker, wdf in weekly_data.items():
            idx = signals_pullback(wdf, ms, vc)
            if len(idx) == 0:
                continue
            for yr, ret in backtest_one(wdf, idx, RR_FIX, HOLD_FIX):
                year_rets[yr].append(ret); filled += 1

        def wr(yr):
            r = np.array(year_rets.get(yr, []))
            return f"{len(r[r>0])/len(r)*100:6.1f}%{'⚠' if len(r[r>0])/len(r)*100<OK_WR else ' '}" \
                   if len(r) >= 5 else "   N/A "

        all_r = np.array([r for rs in year_rets.values() for r in rs])
        if len(all_r) == 0:
            continue
        t_wr = len(all_r[all_r > 0]) / len(all_r) * 100
        spd  = filled / total_weeks
        ok_y = all(len(np.array(year_rets.get(y,[]))) >= 5 and
                   len(np.array(year_rets.get(y,[]))[np.array(year_rets.get(y,[])) > 0]) /
                   len(np.array(year_rets.get(y,[]))) * 100 >= OK_WR
                   for y in [2023, 2024, 2025])
        mark = "✅" if ok_y and spd >= OK_SPD else "  "
        print(f"  {ms:>5}週  {vc:>5.1f}x  │ "
              f"{wr(2023)} {wr(2024)} {wr(2025)} {wr(2026)} │ "
              f"{t_wr:>5.1f}%  {spd:>5.3f}/週  {mark}")
        if ok_y and spd >= OK_SPD:
            passed.append((ms, vc))

    if not passed:
        print("\n  ❌ 合格なし")
        return

    print(f"\nフェーズ2: RR×保有週グリッド")
    for ms, vc in passed:
        sig_cache = {}
        for ticker, wdf in weekly_data.items():
            idx = signals_pullback(wdf, ms, vc)
            if len(idx) > 0:
                sig_cache[ticker] = (wdf, idx)

        print(f"\n  MAスロー{ms}週  Vol<{vc}x")
        print(f"  {'RR':>4}  {'保有':>5}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  "
              f"{'件/週':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  判定")
        print("  " + "─" * 72)

        for rr, mh in itertools.product(RR_LIST, HOLD_LIST):
            yr2 = defaultdict(list); f2 = 0
            for ticker, (wdf, idx) in sig_cache.items():
                for yr, ret in backtest_one(wdf, idx, rr, mh):
                    yr2[yr].append(ret); f2 += 1
            if f2 < 5:
                continue
            all_r = np.array([r for rs in yr2.values() for r in rs])
            wins = all_r[all_r > 0]; losses = all_r[all_r <= 0]
            t_wr = len(wins)/len(all_r)*100
            t_pf = abs(wins.mean()/losses.mean()) if len(losses)>0 else 0
            t_ev = t_wr/100*(wins.mean() if len(wins)>0 else 0) + \
                   (1-t_wr/100)*(losses.mean() if len(losses)>0 else 0)
            spd  = f2/total_weeks

            def fw(y):
                r = np.array(yr2.get(y, []))
                return f"{len(r[r>0])/len(r)*100:5.1f}%" if len(r)>=5 else "  N/A"

            ok_y = all(len(np.array(yr2.get(y,[])))>=5 and
                       len(np.array(yr2.get(y,[]))[np.array(yr2.get(y,[]))>0])/
                       len(np.array(yr2.get(y,[])))*100>=OK_WR
                       for y in [2023,2024,2025])
            mark = "✅" if ok_y and spd>=OK_SPD else "  "
            print(f"  {rr:>4.1f}  {mh:>4}週  {t_wr:>5.1f}%  {t_pf:>5.2f}  "
                  f"{t_ev:>+6.2f}%  {spd:>5.3f}/週  "
                  f"{fw(2023)} {fw(2024)} {fw(2025)}  {mark}")


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    print(f"銘柄数: {len(all_data)}  データ期間〜{cache['date']}")

    print("週足変換中...")
    weekly_data = {}
    for ticker, df in all_data.items():
        wdf = to_weekly(df)
        if len(wdf) >= MIN_WEEKS:
            weekly_data[ticker] = wdf

    sample_w = next(iter(weekly_data.values()))
    total_weeks = len(sample_w)
    print(f"週足変換完了: {len(weekly_data)}銘柄  {total_weeks}週分")
    print(f"  期間: {sample_w.index[0].date()} 〜 {sample_w.index[-1].date()}")

    run_breakout(weekly_data, total_weeks)
    run_pullback(weekly_data, total_weeks)

    print(f"\n{'='*70}")
    print("完了")


if __name__ == "__main__":
    main()
