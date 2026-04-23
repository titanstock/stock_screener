#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指値エントリー検証 - 終値マイナスX%の効果をバックテスト
==========================================================
対象戦略 : 売られすぎ反発型 / NOA
エントリー: 終値 × (1 - discount_pct / 100) の指値
約定条件 : 翌日〜MAX_FILL_DAYS 日以内に安値が指値以下
"""

import pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
STOP_COEF     = 2.0
STOP_CAP      = 0.10   # 損切り上限 -10%
MAX_HOLD      = 10     # 最大保有日数
MAX_FILL_DAYS = 3      # 指値約定待ち最大日数
MIN_TURNOVER  = 30_000_000  # 最低売買代金（円）

# グリッド
DISCOUNT_PCTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# 戦略別RR
RR = {
    "oversold_bounce": 2.5,
    "noa":             2.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 指標計算
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    rsi = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return rsi
    deltas = np.diff(prices.astype(float))
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


def _calc_atr(h, l, c, period=14) -> np.ndarray:
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return pd.Series(tr).rolling(period).mean().values


def _calc_macd(prices: np.ndarray, fast=12, slow=26, sig=9):
    s = pd.Series(prices.astype(float))
    macd   = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd.values, signal.values


# ══════════════════════════════════════════════════════════════════════════════
# シグナル生成
# ══════════════════════════════════════════════════════════════════════════════

def get_signal_idx(df: pd.DataFrame, strategy: str) -> np.ndarray:
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    if n < 60:
        return np.array([], dtype=int)

    avg_v  = pd.Series(v).rolling(20).mean().values
    avg_to = pd.Series(c * v).rolling(20).mean().values
    atr    = _calc_atr(h, l, c)

    # 流動性フィルター
    liquid = (avg_to >= MIN_TURNOVER) & ~np.isnan(avg_to)

    if strategy == "oversold_bounce":
        rsi14 = _calc_rsi(c, 14)
        vol_r = np.where(avg_v > 0, v / avg_v, np.nan)
        # ATR拡大: 直近3日平均 > 前3日平均
        atr_s      = pd.Series(atr)
        atr3d      = atr_s.rolling(3).mean().values
        atr3d_prev = atr_s.shift(3).rolling(3).mean().values
        atr_exp    = (atr3d > atr3d_prev) & ~np.isnan(atr3d) & ~np.isnan(atr3d_prev)
        mask = (
            liquid &
            (rsi14 <= 30.0) &
            (vol_r >= 1.5) &
            atr_exp &
            ~np.isnan(rsi14) & ~np.isnan(vol_r) & ~np.isnan(atr)
        )

    elif strategy == "noa":
        rsi30          = _calc_rsi(c, 30)
        macd, macd_sig = _calc_macd(c)
        mask = (
            liquid &
            (rsi30 <= 30.0) &
            (macd < macd_sig) &
            ~np.isnan(rsi30) & ~np.isnan(macd) & ~np.isnan(macd_sig) & ~np.isnan(atr)
        )
    else:
        return np.array([], dtype=int)

    # 最後の MAX_HOLD + MAX_FILL_DAYS 日はシグナル除外（出口データ不足）
    mask[-(MAX_HOLD + MAX_FILL_DAYS):] = False
    return np.where(mask)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 指値バックテスト
# ══════════════════════════════════════════════════════════════════════════════

def backtest_limit(all_data: dict, strategy: str, discount_pct: float):
    rr = RR[strategy]
    all_rets      = []
    signal_total  = 0
    filled_total  = 0

    for ticker, df in all_data.items():
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        o = df["Open"].values.astype(float)
        n = len(c)
        atr = _calc_atr(h, l, c)

        sig_idx = get_signal_idx(df, strategy)
        signal_total += len(sig_idx)

        for si in sig_idx:
            entry = c[si] * (1 - discount_pct / 100)
            a     = atr[si]
            if np.isnan(a) or a <= 0 or entry <= 0:
                continue
            stop = max(entry - a * STOP_COEF, entry * (1 - STOP_CAP))
            take = entry + (entry - stop) * rr
            if stop >= entry or take <= entry:
                continue

            # 指値約定チェック
            fill_day = -1
            for d in range(1, MAX_FILL_DAYS + 1):
                j = si + d
                if j >= n:
                    break
                if l[j] <= entry:
                    fill_day = j
                    break

            if fill_day < 0:
                continue
            filled_total += 1

            # 損益追跡（fill_day から MAX_HOLD 日）
            exit_price = None
            hold_end   = min(si + MAX_HOLD + 1, n)

            for j in range(fill_day, hold_end):
                if j == fill_day:
                    # 約定当日: 始値確認
                    if o[j] <= stop:
                        exit_price = min(o[j], stop)
                        break
                    if o[j] >= take:
                        exit_price = take
                        break
                    if h[j] >= take:
                        exit_price = take
                        break
                    if l[j] <= stop:
                        exit_price = stop
                        break
                else:
                    if o[j] <= stop:
                        exit_price = min(o[j], stop)
                        break
                    if o[j] >= take:
                        exit_price = take
                        break
                    if h[j] >= take:
                        exit_price = take
                        break
                    if l[j] <= stop:
                        exit_price = stop
                        break

            if exit_price is None:
                exit_price = c[min(hold_end - 1, n - 1)]

            ret = (exit_price - entry) / entry * 100
            all_rets.append(ret)

    return np.array(all_rets), signal_total, filled_total


# ══════════════════════════════════════════════════════════════════════════════
# メトリクス
# ══════════════════════════════════════════════════════════════════════════════

def metrics(rets: np.ndarray, trading_days: int, signal_total: int, filled_total: int) -> dict:
    if len(rets) < 5:
        return {"n": len(rets), "wr": 0, "pf": 0, "ev": 0,
                "spd": 0, "fill_rate": 0, "avg_w": 0, "avg_l": 0}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets) * 100
    avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = filled_total / trading_days if trading_days > 0 else 0.0
    fill_r = filled_total / signal_total * 100 if signal_total > 0 else 0.0
    return {"n": len(rets), "wr": wr, "pf": pf, "ev": ev,
            "spd": spd, "fill_rate": fill_r, "avg_w": avg_w, "avg_l": avg_l}


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_data: dict = cache["data"]
    print(f"銘柄数: {len(all_data)}  データ期間: 〜{cache['date']}")

    # 取引日数推定
    sample_df = next(iter(all_data.values()))
    trading_days = len(sample_df)

    for strategy in ["oversold_bounce", "noa"]:
        strat_label = "売られすぎ反発型" if strategy == "oversold_bounce" else "NOA"
        print(f"\n{'='*80}")
        print(f"【{strat_label}】指値エントリー検証")
        print(f"  約定条件: 終値 × (1 - X%) の指値 / {MAX_FILL_DAYS}日以内約定")
        print(f"  最大保有: {MAX_HOLD}日 / 損切: ATR×{STOP_COEF}（上限-{int(STOP_CAP*100)}%）/ RR: {RR[strategy]}")
        print(f"{'='*80}")

        hdr = f"  {'割引率':>6}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件/日':>6}  {'約定率':>6}  {'サンプル':>7}  {'平均利':>7}  {'平均損':>7}"
        sep = "  " + "-" * 76
        print(hdr); print(sep)

        for pct in DISCOUNT_PCTS:
            rets, sig_n, fill_n = backtest_limit(all_data, strategy, pct)
            m = metrics(rets, trading_days, sig_n, fill_n)
            marker = " ◀" if pct == 0.0 else ""
            print(f"  -{pct:.1f}%    "
                  f"{m['wr']:>5.1f}%  {m['pf']:>5.2f}  {m['ev']:>+6.2f}%  "
                  f"{m['spd']:>5.2f}/日  {m['fill_rate']:>5.1f}%  "
                  f"{m['n']:>7}件  {m['avg_w']:>+6.2f}%  {m['avg_l']:>+6.2f}%{marker}")

    print("\n完了")


if __name__ == "__main__":
    main()
