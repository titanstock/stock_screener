#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新設計バックテスト
==================
【設計方針】
  1. 損切り  = シグナル日安値
  2. 上限チェック = シグナル日安値 / 0.90 > 終値 でなければスキップ
  3. エントリー  = 翌日始値（後日変更予定）
  4. 利確    = エントリー + (エントリー - 損切り) × RR
  5. 最大保有 = グリッドパラメータ（0=制限なし）

【対象戦略】
  ② 売られすぎ反発型
  ③ 出来高枯渇反発型（2026-04-22更新済みパラメータ）
  ④ 連続陰線下ヒゲ反発型（2026-04-22更新済みパラメータ）
  NOA
"""

import itertools, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH   = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY  = 100
MIN_TURNOVER = 30_000_000
STOP_CAP_PCT = 0.10   # エントリー上限チェック用（シグナル安値/0.90 > 終値）

CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.3

# グリッド（全戦略共通）
RR_LIST       = [1.5, 2.0, 2.5, 3.0]
MAX_HOLD_LIST = [10, 20, 40, 0]   # 0=制限なし（1000日で代替）
NO_LIMIT_DAYS = 1000

# ── 戦略別シグナル条件 ──────────────────────────────────────────────────────
# ② 売られすぎ反発型
OB_RSI_HI    = 30.0
OB_VOL_MULT  = 1.5

# ③ 出来高枯渇反発型（2026-04-22更新）
VD_DRY_DAYS  = 5
VD_SPIKE     = 2.0
VD_RSI_HI    = 55.0
VD_MA25_DEV  = 10.0

# ④ 連続陰線下ヒゲ反発型（2026-04-22更新）
CB_DAYS      = 5
CB_SHADOW    = 30.0
CB_VOL_MULT  = 2.0
CB_RSI_HI    = 45.0

# NOA
NOA_RSI_PERIOD = 30
NOA_RSI_HI     = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# 指標計算
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# シグナル生成（戦略別）
# ══════════════════════════════════════════════════════════════════════════════

def _signals(df: pd.DataFrame, strategy: str) -> np.ndarray:
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o = df["Open"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    if n < MIN_HISTORY:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    if strategy == "oversold_bounce":
        rsi14   = _rsi(c, 14)
        atr_s   = pd.Series(pd.Series(
            np.maximum.reduce([h - lo,
                               np.abs(h - np.roll(c, 1)),
                               np.abs(lo - np.roll(c, 1))])
        ).rolling(14).mean())
        atr3    = atr_s.rolling(3).mean().values
        atr3p   = atr_s.shift(3).rolling(3).mean().values
        atr_exp = (atr3 > atr3p) & ~np.isnan(atr3) & ~np.isnan(atr3p)
        mask = (liquid & (rsi14 <= OB_RSI_HI) & (vol_r >= OB_VOL_MULT) &
                atr_exp & ~np.isnan(rsi14) & ~np.isnan(vol_r))

    elif strategy == "vol_dry_bounce":
        rsi14  = _rsi(c, 14)
        ma25   = pd.Series(c).rolling(25).mean().values
        ma25d  = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)
        mask   = np.zeros(n, dtype=bool)
        for i in range(VD_DRY_DAYS + 2, n):
            dry = (avg_v[i] > 0 and
                   float(v[i - VD_DRY_DAYS - 1:i - 1].max()) < avg_v[i])
            if (liquid[i] and dry and
                    vol_r[i] >= VD_SPIKE and
                    not np.isnan(rsi14[i]) and rsi14[i] <= VD_RSI_HI and
                    not np.isnan(ma25d[i]) and ma25d[i] <= VD_MA25_DEV):
                mask[i] = True

    elif strategy == "consec_bear_shadow":
        rsi14 = _rsi(c, 14)
        rng   = h - lo
        # 下ヒゲ比率: (min(open,close) - low) / (high - low)
        shadow_pct = np.where(rng > 0,
                               (np.minimum(o, c) - lo) / rng * 100, np.nan)
        mask = np.zeros(n, dtype=bool)
        for i in range(CB_DAYS, n):
            consec = all(c[i - CB_DAYS + j] < o[i - CB_DAYS + j]
                         for j in range(CB_DAYS))
            if (liquid[i] and consec and
                    not np.isnan(shadow_pct[i]) and shadow_pct[i] >= CB_SHADOW and
                    vol_r[i] >= CB_VOL_MULT and
                    not np.isnan(rsi14[i]) and rsi14[i] <= CB_RSI_HI):
                mask[i] = True

    elif strategy == "noa":
        rsi30      = _rsi(c, NOA_RSI_PERIOD)
        macd, msig = _macd(c)
        mask = (liquid & (rsi30 <= NOA_RSI_HI) & (macd < msig) &
                ~np.isnan(rsi30) & ~np.isnan(macd))

    else:
        return np.array([], dtype=int)

    # 末尾は出口データ不足のため除外
    mask[-50:] = False
    return np.where(mask)[0]


# ══════════════════════════════════════════════════════════════════════════════
# バックテスト（1銘柄）
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_one(df: pd.DataFrame, sig_idx: np.ndarray,
                  rr: float, max_hold: int) -> list[float]:
    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    o  = df["Open"].values.astype(float)
    n  = len(c)
    rets = []
    hold = max_hold if max_hold > 0 else NO_LIMIT_DAYS

    for si in sig_idx:
        stop   = lo[si]             # シグナル日安値
        limit  = stop / (1 - STOP_CAP_PCT)  # = stop / 0.90

        # エントリー上限チェック：終値 > 上限ならスキップ
        if c[si] > limit:
            continue

        # エントリー = 翌日始値
        j0 = si + 1
        if j0 >= n:
            continue
        entry = o[j0]
        if entry <= 0 or entry <= stop:
            continue

        take = entry + (entry - stop) * rr

        # 損益追跡
        exit_price = None
        end = min(si + 1 + hold, n)

        for j in range(j0, end):
            op = o[j]
            # ギャップダウンで始値が損切り以下
            if op <= stop:
                exit_price = op
                break
            # ギャップアップで始値が利確以上
            if op >= take:
                exit_price = take
                break
            # 日中に利確
            if h[j] >= take:
                exit_price = take
                break
            # 日中に損切り
            if lo[j] <= stop:
                exit_price = stop
                break

        if exit_price is None:
            exit_price = c[min(end - 1, n - 1)]

        rets.append((exit_price - entry) / entry * 100)

    return rets


# ══════════════════════════════════════════════════════════════════════════════
# グリッドサーチ
# ══════════════════════════════════════════════════════════════════════════════

def run(all_data: dict, strategy: str, trading_days: int) -> list[dict]:
    label = {
        "oversold_bounce":    "②売られすぎ反発型",
        "vol_dry_bounce":     "③出来高枯渇反発型",
        "consec_bear_shadow": "④連続陰線下ヒゲ反発型",
        "noa":                "NOA",
    }[strategy]

    print(f"\n{'='*80}")
    print(f"【{label}】シグナル計算中...")

    # シグナルを銘柄ごとに事前計算
    sig_map: dict[str, np.ndarray] = {}
    for ticker, df in all_data.items():
        idx = _signals(df, strategy)
        if len(idx) > 0:
            sig_map[ticker] = idx

    total_signals = sum(len(v) for v in sig_map.values())
    print(f"  シグナル銘柄: {len(sig_map)}社  総シグナル数: {total_signals}件")

    results = []
    combos  = list(itertools.product(RR_LIST, MAX_HOLD_LIST))
    print(f"  グリッド: {len(combos)} 通り")

    for rr, mh in combos:
        all_rets: list[float] = []
        filled = 0

        for ticker, sig_idx in sig_map.items():
            df = all_data[ticker]
            r  = _backtest_one(df, sig_idx, rr, mh)
            filled   += len(r)
            all_rets += r

        rets = np.array(all_rets)
        mh_label = f"{mh}日" if mh > 0 else "無制限"

        if len(rets) < 5:
            results.append({"rr": rr, "max_hold": mh_label, "n": len(rets),
                             "wr": 0, "pf": 0, "ev": 0, "spd": 0,
                             "skip_rate": 0, "avg_w": 0, "avg_l": 0})
            continue

        wins   = rets[rets > 0]
        losses = rets[rets <= 0]
        wr     = len(wins) / len(rets) * 100
        avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd    = filled / trading_days
        skip_r = (total_signals - filled) / total_signals * 100 if total_signals > 0 else 0

        results.append({"rr": rr, "max_hold": mh_label, "n": filled,
                         "wr": wr, "pf": pf, "ev": ev, "spd": spd,
                         "skip_rate": skip_r, "avg_w": avg_w, "avg_l": avg_l})

    # 表示
    print(f"\n  設計: 損切=シグナル日安値 / 上限=安値÷0.90 / エントリー=翌日始値")
    print(f"  合格基準: WR≥{CRITERIA_WR}% / PF≥{CRITERIA_PF} / 件数≥{CRITERIA_SPD}件/日")
    hdr = (f"  {'RR':>4}  {'保有':>5}  {'勝率':>6}  {'PF':>5}  "
           f"{'EV':>7}  {'件/日':>6}  {'スキップ率':>8}  {'サンプル':>7}")
    sep = "  " + "-" * 72
    print(hdr); print(sep)

    for r in results:
        ok = ("✅" if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                      and r["spd"] >= CRITERIA_SPD else "  ")
        print(f"  {r['rr']:>4.1f}  {r['max_hold']:>5}  {r['wr']:>5.1f}%  "
              f"{r['pf']:>5.2f}  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  "
              f"{r['skip_rate']:>7.1f}%  {r['n']:>7}件 {ok}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    trading_days   = len(next(iter(all_data.values())))
    print(f"銘柄数: {len(all_data)}  取引日数: {trading_days}日  データ期間〜{cache['date']}")

    for strat in ["oversold_bounce", "vol_dry_bounce", "consec_bear_shadow", "noa"]:
        run(all_data, strat, trading_days)

    print(f"\n{'='*80}")
    print("完了")


if __name__ == "__main__":
    main()
