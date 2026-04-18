#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バックテスト比較スクリプト
① 週足ダウ理論 vs MA25W上昇（3戦略の勝率・PF・件数を並列比較）
② 52週高値ブレイクアウト型（新規）のバックテスト

高速化: 週足指標（ダウ判定・MA25W・52w高値）をすべてループ外で事前計算
"""

import operator
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stock_screener import (
    BREAKOUT_DAYS,
    MIN_AVG_TURNOVER,
    VOL_SPIKE_MULT,
    PULLBACK_TOUCH_PCT,
    PULLBACK_LOOKBACK,
    DOW_N_SWINGS,
    calc_rsi,
    calc_macd,
    check_consolidation,
)

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
LOOKBACK    = 252   # 52週分 + 余裕
MAX_HOLD    = 20
MAX_WORKERS = 8

# ──────────────────────────────────────────────────────────────────────────────
# 週足ダウ理論（事前計算版）
# ──────────────────────────────────────────────────────────────────────────────
def _weekly_uptrend_series(df: pd.DataFrame, n_swings: int = DOW_N_SWINGS) -> pd.Series:
    """週足ダウ理論の上昇トレンド判定を日次 Series として返す（ffill）。"""
    weekly = df.resample("W").agg({"High": "max", "Low": "min"}).dropna()
    result = pd.Series(False, index=weekly.index, dtype=bool)

    arr_h = weekly["High"].values
    arr_l = weekly["Low"].values

    def _swings(arr, cmp):
        return [arr[i] for i in range(2, len(arr) - 2)
                if cmp(arr[i], arr[i-1]) and cmp(arr[i], arr[i-2])
                and cmp(arr[i], arr[i+1]) and cmp(arr[i], arr[i+2])]

    for i in range(4, len(weekly)):
        hs = _swings(arr_h[:i+1], operator.ge)
        ls = _swings(arr_l[:i+1], operator.le)
        if len(hs) >= n_swings and len(ls) >= n_swings:
            h, l = hs[-n_swings:], ls[-n_swings:]
            if (all(h[j] < h[j+1] for j in range(n_swings - 1)) and
                    all(l[j] < l[j+1] for j in range(n_swings - 1))):
                result.iloc[i] = True

    return result.reindex(df.index, method="ffill").fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# 出口シミュレーション
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_exit(closes: np.ndarray, signal_idx: int,
                   entry: float, stop: float, take: float,
                   next_open: bool = False) -> tuple[float | None, str]:
    start  = 0 if next_open else 1
    exit_p, reason = None, "max"
    n = len(closes)
    for offset in range(start, MAX_HOLD + start):
        idx = signal_idx + 1 + offset
        if idx >= n:
            break
        fc = closes[idx]
        if fc <= stop:
            exit_p, reason = fc, "stop"; break
        if fc >= take:
            exit_p, reason = fc, "take"; break
        exit_p = fc
    return exit_p, reason


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄バックテスト
# ──────────────────────────────────────────────────────────────────────────────
def backtest_ticker_compare(ticker: str, df: pd.DataFrame) -> dict[str, list]:
    if df is None or len(df) < LOOKBACK + 1:
        return {}

    df = df.copy()

    # ── 全指標を事前計算 ──────────────────────────────────────────────────────
    df["MA25"] = df["Close"].rolling(25).mean()
    ms, ss     = calc_macd(df["Close"])
    df["MACD"] = ms
    df["SIG"]  = ss
    df["RSI"]  = calc_rsi(df["Close"])
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_period        = BREAKOUT_DAYS + 1
    df["avg_vol"]   = df["Volume"].rolling(n_period).mean().shift(1)
    df["avg_to"]    = (df["Close"] * df["Volume"]).rolling(n_period).mean().shift(1)
    df["past_high"] = df["High"].rolling(n_period).max().shift(1)
    df["high_52w"]  = df["High"].rolling(250).max().shift(1)

    # 週足MA25（日次フォワードフィル）
    wc             = df["Close"].resample("W").last()
    wma25          = wc.rolling(25).mean()
    wma25_4w       = wma25.shift(4)
    df["MA25W"]    = wma25.reindex(df.index, method="ffill")
    df["MA25W_4w"] = wma25_4w.reindex(df.index, method="ffill")
    df["ma25w_up"] = df["MA25W"] > df["MA25W_4w"]

    # 週足ダウ理論（現行条件用）
    try:
        df["dow_uptrend"] = _weekly_uptrend_series(df)
    except Exception:
        df["dow_uptrend"] = False

    # MA25タッチ（直近PULLBACK_LOOKBACK日以内）
    tp = 1.0 + PULLBACK_TOUCH_PCT
    touch_today = (df["Close"] >= df["MA25"]) & (df["Close"] <= df["MA25"] * tp)
    touch_prev  = (df["Close"].shift(1) >= df["MA25"].shift(1)) & \
                  (df["Close"].shift(1) <= df["MA25"].shift(1) * tp)
    df["touched_ma25"] = touch_today | touch_prev   # PULLBACK_LOOKBACK=2

    # 値固め（20日高値版・check_consolidation 相当）を事前計算
    atr_r = df["ATR"].shift(1).rolling(5).mean()
    atr_p = df["ATR"].shift(6).rolling(5).mean()
    atr_shrink = (atr_r < atr_p) & (atr_p > 0)
    near_20d = sum(
        ((df["Close"].shift(k) >= df["past_high"] * 0.96) &
         (df["Close"].shift(k) <= df["past_high"] * 1.02)).astype(int)
        for k in range(1, 6)
    )
    df["consolidation"] = atr_shrink & (near_20d >= 2)

    # 値固め（52週高値版）
    near_52w = sum(
        ((df["Close"].shift(k) >= df["high_52w"] * 0.96) &
         (df["Close"].shift(k) <= df["high_52w"] * 1.02)).astype(int)
        for k in range(1, 6)
    )
    df["consol_52w"]    = atr_shrink & (near_52w >= 2)
    df["recent_bo_52w"] = df["Close"].shift(2) <= df["high_52w"]

    # MA25回復型: 先週終値がMA25W以下 → 今週終値がMA25W以上
    wc_last = df["Close"].resample("W").last()
    wma25_w = wma25.reindex(df.index, method="ffill")   # 既に df["MA25W"] と同じ
    # 先週終値（前週の週次終値）
    prev_week_close = wc_last.shift(1).reindex(df.index, method="ffill")
    prev_week_ma25w = wma25.shift(1).reindex(df.index, method="ffill")
    df["ma25w_recovery"] = (
        (prev_week_close <= prev_week_ma25w) &   # 先週終値 ≤ 先週MA25W
        (df["Close"] >= df["MA25W"])              # 今週（当日）終値 ≥ MA25W
    )

    # numpy 配列化（ループ高速化）
    closes       = df["Close"].values.astype(float)
    opens        = df["Open"].values.astype(float)
    ma25s        = df["MA25"].values.astype(float)
    atrs         = df["ATR"].values.astype(float)
    rsis         = df["RSI"].values.astype(float)
    macds        = df["MACD"].values.astype(float)
    sigs         = df["SIG"].values.astype(float)
    vols         = df["Volume"].values.astype(float)
    avg_vols     = df["avg_vol"].values.astype(float)
    avg_tos      = df["avg_to"].values.astype(float)
    past_highs   = df["past_high"].values.astype(float)
    high_52ws    = df["high_52w"].values.astype(float)
    ma25ws       = df["MA25W"].values.astype(float)
    ma25w_ups    = df["ma25w_up"].values.astype(bool)
    dow_ups      = df["dow_uptrend"].values.astype(bool)
    toucheds     = df["touched_ma25"].values.astype(bool)
    consols      = df["consolidation"].values.astype(bool)
    consols_52w  = df["consol_52w"].values.astype(bool)
    recent_52ws  = df["recent_bo_52w"].values.astype(bool)
    recoveries   = df["ma25w_recovery"].values.astype(bool)
    n_rows       = len(df)

    out = {"current": [], "new": [], "breakout_52w": [], "breakout_52w_v2": [], "ma25w_recovery": []}
    rr_map = {"baseline": 1.5, "breakout": 2.0, "pullback": 1.5}

    for i in range(LOOKBACK, n_rows - 1):
        close   = closes[i]
        prev    = closes[i-1]
        ma25    = ma25s[i]
        rsi     = rsis[i]
        macd    = macds[i]
        sig     = sigs[i]
        atr     = atrs[i]
        vol     = vols[i]
        avg_vol = avg_vols[i]
        avg_to  = avg_tos[i]
        past_h  = past_highs[i]
        ma25_w  = None if np.isnan(ma25ws[i]) else ma25ws[i]
        ma25w_up= ma25w_ups[i]
        dow_up  = dow_ups[i]
        touched = toucheds[i]
        consol  = consols[i]
        h52w    = None if np.isnan(high_52ws[i]) else high_52ws[i]

        if (np.isnan(ma25) or np.isnan(rsi) or np.isnan(macd) or np.isnan(sig)
                or np.isnan(atr) or np.isnan(avg_vol) or np.isnan(avg_to)
                or np.isnan(past_h) or atr <= 0 or avg_vol <= 0):
            continue
        if avg_to < MIN_AVG_TURNOVER:
            continue

        bo_pct     = (close - past_h) / past_h * 100 if past_h > 0 else 0.0
        chg        = (close - prev) / prev * 100 if prev > 0 else 0.0
        above_d    = close > ma25
        above_w    = ma25_w is not None and close > ma25_w
        within_w   = ma25_w is None or close <= ma25_w * 1.2
        vol_15x    = vol >= avg_vol * 1.5
        vol_3x     = vol >= avg_vol * VOL_SPIKE_MULT
        recent_bo  = i >= 2 and closes[i-2] <= past_h
        consol_52w = consols_52w[i]
        recent_52w = recent_52ws[i]
        recovery   = recoveries[i]
        h52w       = None if np.isnan(high_52ws[i]) else high_52ws[i]

        def _add(label, strat, entry, rr, next_open=False):
            s  = max(entry - atr * 2.0, entry * 0.90)
            t  = entry + (entry - s) * rr
            ep, rsn = _simulate_exit(closes, i, entry, s, t, next_open)
            if ep is not None:
                ret = (ep - entry) / entry * 100
                out[label].append({"strategy": strat, "return": ret,
                                   "win": ret > 0, "exit_reason": rsn})

        # ── ① 現行条件（ダウ理論） ───────────────────────────────────────────
        if above_d and above_w and dow_up and within_w and 50.0 <= rsi <= 70.0 and macd > sig and vol_15x:
            _add("current", "baseline", close * 0.98, 1.5)
        if (close > past_h and bo_pct >= 1.0 and vol_3x and chg >= 2.0
                and above_d and above_w and dow_up and rsi >= 60.0
                and recent_bo and consol):
            _add("current", "breakout", close, 2.0)
        if above_d and above_w and dow_up and touched and 60.0 <= rsi <= 70.0 and macd > sig:
            _add("current", "pullback", ma25, 1.5)

        # ── ① 新条件（MA25W上昇） ────────────────────────────────────────────
        if above_d and above_w and ma25w_up and within_w and 50.0 <= rsi <= 70.0 and macd > sig and vol_15x:
            _add("new", "baseline", close * 0.98, 1.5)
        if (close > past_h and bo_pct >= 1.0 and vol_3x and chg >= 2.0
                and above_d and above_w and ma25w_up and rsi >= 60.0
                and recent_bo and consol):
            _add("new", "breakout", close, 2.0)
        if above_d and above_w and ma25w_up and touched and 60.0 <= rsi <= 70.0 and macd > sig:
            _add("new", "pullback", ma25, 1.5)

        # ── ② 52週高値ブレイクアウト（基本版）────────────────────────────────
        if above_w and ma25w_up and h52w is not None and close > h52w and vol >= avg_vol * 2.0:
            entry_52w = opens[i + 1]
            if entry_52w > 0 and not np.isnan(entry_52w):
                s52 = max(entry_52w - atr * 2.0, entry_52w * 0.90)
                t52 = entry_52w + (entry_52w - s52) * 2.0
                ep52, rsn52 = _simulate_exit(closes, i, entry_52w, s52, t52, next_open=True)
                if ep52 is not None:
                    ret52 = (ep52 - entry_52w) / entry_52w * 100
                    out["breakout_52w"].append({"return": ret52,
                                                "win": ret52 > 0,
                                                "exit_reason": rsn52})

        # ── ② 52週高値ブレイクアウト（強化版：値固め・RSI・前日比・ダウ）─────
        if (h52w is not None and close > h52w
                and above_w and dow_up
                and vol >= avg_vol * 2.0 and chg >= 2.0
                and rsi >= 60.0 and recent_52w and consol_52w):
            entry_52v2 = opens[i + 1]
            if entry_52v2 > 0 and not np.isnan(entry_52v2):
                s52v2 = max(entry_52v2 - atr * 2.0, entry_52v2 * 0.90)
                t52v2 = entry_52v2 + (entry_52v2 - s52v2) * 2.0
                ep52v2, rsn52v2 = _simulate_exit(closes, i, entry_52v2, s52v2, t52v2, next_open=True)
                if ep52v2 is not None:
                    ret52v2 = (ep52v2 - entry_52v2) / entry_52v2 * 100
                    out["breakout_52w_v2"].append({"return": ret52v2,
                                                   "win": ret52v2 > 0,
                                                   "exit_reason": rsn52v2})

        # ── ③ MA25回復型（先週MA25W以下→今週MA25W以上、翌日始値）────────────
        if (recovery and above_d
                and vol_15x and avg_to >= MIN_AVG_TURNOVER
                and 45.0 <= rsi <= 60.0):
            entry_rec = opens[i + 1]
            if entry_rec > 0 and not np.isnan(entry_rec):
                s_rec = max(entry_rec - atr * 2.0, entry_rec * 0.90)
                t_rec = entry_rec + (entry_rec - s_rec) * 1.5
                ep_rec, rsn_rec = _simulate_exit(closes, i, entry_rec, s_rec, t_rec, next_open=True)
                if ep_rec is not None:
                    ret_rec = (ep_rec - entry_rec) / entry_rec * 100
                    out["ma25w_recovery"].append({"return": ret_rec,
                                                  "win": ret_rec > 0,
                                                  "exit_reason": rsn_rec})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# キャッシュ読み込み
# ──────────────────────────────────────────────────────────────────────────────
def load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)["data"]
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# メトリクス計算
# ──────────────────────────────────────────────────────────────────────────────
def _metrics(signals: list[dict], trading_days: int) -> dict:
    df = pd.DataFrame(signals)
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "spd": 0.0,
                "stop_rate": 0.0, "take_rate": 0.0, "avg_ret": 0.0}
    n    = len(df)
    w    = df["win"].astype(bool)
    aw   = df[w]["return"].mean()   if w.any()   else 0.0
    al   = df[~w]["return"].mean()  if (~w).any() else 0.0
    pf   = abs(aw / al) if al != 0 else float("inf")
    return {
        "n":         n,
        "wr":        w.mean() * 100,
        "pf":        pf,
        "spd":       n / trading_days,
        "stop_rate": df["exit_reason"].eq("stop").mean() * 100,
        "take_rate": df["exit_reason"].eq("take").mean() * 100,
        "avg_ret":   df["return"].mean(),
    }


def _metrics_by_strategy(signals: list[dict], trading_days: int) -> dict[str, dict]:
    df = pd.DataFrame(signals) if signals else pd.DataFrame()
    result = {}
    for strat in ["baseline", "breakout", "pullback"]:
        sub = df[df["strategy"] == strat].to_dict("records") if not df.empty else []
        result[strat] = _metrics(sub, trading_days)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュからデータを読み込み中...")
    all_data = load_cache()
    if all_data is None:
        raise RuntimeError("backtest_cache.pkl が見つかりません。backtest.py を先に実行してください。")
    print(f"  {len(all_data)} 銘柄を読み込み完了")

    all_current:  list[dict] = []
    all_new:      list[dict] = []
    all_52w:      list[dict] = []
    all_52w_v2:   list[dict] = []
    all_recovery: list[dict] = []
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(backtest_ticker_compare, t, df): t
                   for t, df in all_data.items()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_current.extend(res.get("current", []))
                all_new.extend(res.get("new", []))
                all_52w.extend(res.get("breakout_52w", []))
                all_52w_v2.extend(res.get("breakout_52w_v2", []))
                all_recovery.extend(res.get("ma25w_recovery", []))
            with lock:
                done += 1
            if done % 300 == 0 or done == len(all_data):
                print(f"  進捗: {done}/{len(all_data)}  "
                      f"20d={len(all_current)} 52w={len(all_52w)} "
                      f"52w_v2={len(all_52w_v2)} rec={len(all_recovery)}")

    trading_days = max(
        (len(df) - LOOKBACK for df in all_data.values()
         if df is not None and len(df) > LOOKBACK),
        default=250,
    )
    print(f"\n推定取引日数: {trading_days} 日 / 対象: {len(all_data)} 銘柄")

    # ── ① 比較表示 ────────────────────────────────────────────────────────────
    strats_jp = {
        "baseline": "ベースライン型",
        "breakout": "ブレイクアウト型",
        "pullback": "押し目買い型",
    }
    cur_by = _metrics_by_strategy(all_current, trading_days)
    new_by = _metrics_by_strategy(all_new,     trading_days)

    print("\n" + "=" * 82)
    print("【① 週足条件比較: ダウ理論 vs MA25W上昇（4週前比較）】")
    print(f"    期間: 約{trading_days}営業日  銘柄数: {len(all_data)}")
    W = 9
    print()
    print(f"  {'戦略':<16}  {'現行（ダウ理論）':^{W*4+3}}    {'新条件（MA25W上昇）':^{W*4+3}}")
    print(f"  {'':16}  {'勝率':>{W}} {'PF':>{W}} {'件数':>{W}} {'件/日':>{W}}    "
          f"{'勝率':>{W}} {'PF':>{W}} {'件数':>{W}} {'件/日':>{W}}")
    print("  " + "-" * 84)

    for sk, jp in strats_jp.items():
        c = cur_by[sk]
        n = new_by[sk]
        print(f"  {jp:<16}  "
              f"{c['wr']:>{W}.1f}% {c['pf']:>{W}.2f} {c['n']:>{W},} {c['spd']:>{W}.1f}    "
              f"{n['wr']:>{W}.1f}% {n['pf']:>{W}.2f} {n['n']:>{W},} {n['spd']:>{W}.1f}")

    print("  " + "-" * 84)
    ct = _metrics(all_current, trading_days)
    nt = _metrics(all_new,     trading_days)
    print(f"  {'合計（3戦略）':<16}  "
          f"{ct['wr']:>{W}.1f}% {ct['pf']:>{W}.2f} {ct['n']:>{W},} {ct['spd']:>{W}.1f}    "
          f"{nt['wr']:>{W}.1f}% {nt['pf']:>{W}.2f} {nt['n']:>{W},} {nt['spd']:>{W}.1f}")

    print("\n  ▼ 変化量（新 − 現行）")
    for sk, jp in strats_jp.items():
        c, n = cur_by[sk], new_by[sk]
        print(f"  {jp:<16}  "
              f"勝率: {n['wr']-c['wr']:+.1f}%  "
              f"PF: {n['pf']-c['pf']:+.2f}  "
              f"件数: {n['n']-c['n']:+,}  "
              f"件/日: {n['spd']-c['spd']:+.1f}")

    # ── ② 52週高値ブレイクアウト比較 ────────────────────────────────────────
    # 現行20日版 breakout の結果を取り出す
    cur_bo = [s for s in all_current if s["strategy"] == "breakout"]

    print("\n" + "=" * 82)
    print("【② ブレイクアウト型比較（20日高値 vs 52週高値）】")
    print(f"    期間: 約{trading_days}営業日  銘柄数: {len(all_data)}")

    def _print_bo(label, signals):
        m = _metrics(signals, trading_days)
        if m["n"] == 0:
            print(f"  {label}: シグナルなし")
            return
        mx = 100 - m["stop_rate"] - m["take_rate"]
        print(f"  {label}")
        print(f"    シグナル数  : {m['n']:,} 件  ({m['spd']:.2f} 件/日)")
        print(f"    勝率        : {m['wr']:.1f}%")
        print(f"    PF          : {m['pf']:.2f}")
        print(f"    平均リターン: {m['avg_ret']:+.2f}%")
        print(f"    損切率      : {m['stop_rate']:.1f}%  利確率: {m['take_rate']:.1f}%  期間満了: {mx:.1f}%")

    _print_bo("現行ブレイクアウト型（20日高値・引値エントリー）", cur_bo)
    print()
    _print_bo("52週高値ブレイクアウト（基本版・翌日始値）", all_52w)
    print()
    _print_bo("52週高値ブレイクアウト（強化版：値固め+RSI60+前日比+ダウ）", all_52w_v2)

    # ── ③ MA25回復型 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("【③ MA25回復型（先週MA25W以下→今週MA25W以上・翌日始値）】")
    _print_bo("MA25回復型", all_recovery)

    print("=" * 82)
