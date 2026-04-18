#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATRスクイーズ型バックテスト
条件: 出来高3倍 / RSI45-55 / ATR収縮 / 売買代金3000万 / 時価総額300億以下
決済: 翌日始値 / ATR×2.0(-10%) / RR 1:1 / 1:1.5 / 1:2
"""

import operator
import pickle, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS, DOW_N_SWINGS

CACHE_PATH   = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY  = 200
MAX_HOLD     = 20
MAX_WORKERS  = 8
ATR_MULT     = 2.0
ATR_FLOOR    = 0.10        # 最大損切り -10%
RR_LIST      = [1.0, 1.5]
MKTCAP_MAX   = 30_000_000_000   # 300億円
VOL_MULT     = 3.0
RSI_LO, RSI_HI = 45.0, 55.0


# ──────────────────────────────────────────────────────────────────────────────
# 週足ダウ理論
# ──────────────────────────────────────────────────────────────────────────────
def _weekly_uptrend_series(df: pd.DataFrame, n: int = DOW_N_SWINGS) -> pd.Series:
    weekly = df.resample("W").agg({"High": "max", "Low": "min"}).dropna()
    result = pd.Series(False, index=weekly.index, dtype=bool)
    arr_h, arr_l = weekly["High"].values, weekly["Low"].values

    def _sw(arr, cmp):
        return [arr[i] for i in range(2, len(arr) - 2)
                if cmp(arr[i], arr[i-1]) and cmp(arr[i], arr[i-2])
                and cmp(arr[i], arr[i+1]) and cmp(arr[i], arr[i+2])]

    for i in range(4, len(weekly)):
        hs = _sw(arr_h[:i+1], operator.ge)
        ls = _sw(arr_l[:i+1], operator.le)
        if len(hs) >= n and len(ls) >= n:
            h, l = hs[-n:], ls[-n:]
            if (all(h[j] < h[j+1] for j in range(n-1)) and
                    all(l[j] < l[j+1] for j in range(n-1))):
                result.iloc[i] = True

    return result.reindex(df.index, method="ffill").fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# 時価総額取得（現在の発行済株数を使用、過去近似）
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        fi = yf.Ticker(ticker).fast_info
        sh = getattr(fi, "shares", None)
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t, sh = fut.result()
            if sh:
                result[t] = sh
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  株数取得: {done}/{len(tickers)}  取得成功: {len(result)}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 出口リターン（ベクトル化、RR可変）
# ──────────────────────────────────────────────────────────────────────────────
def _exit_returns_vec(closes: np.ndarray, entries: np.ndarray,
                      stops: np.ndarray, takes: np.ndarray) -> np.ndarray:
    n    = len(closes)
    rets = np.full(n, np.nan)
    valid = (~np.isnan(entries)) & (entries > 0)
    vidx  = np.where(valid)[0]
    if len(vidx) == 0:
        return rets

    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)

    hit     = ((fut <= stops[vidx, np.newaxis]) | (fut >= takes[vidx, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + fhp + 1, 0, n - 1)]
    rets[vidx] = np.where(has_fut, (ep - entries[vidx]) / entries[vidx] * 100, np.nan)
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 決済種別（損切り / 利確 / 強制）
# ──────────────────────────────────────────────────────────────────────────────
def _exit_types_vec(closes: np.ndarray, entries: np.ndarray,
                    stops: np.ndarray, takes: np.ndarray) -> np.ndarray:
    """0=強制終了, 1=損切り, 2=利確"""
    n    = len(closes)
    types = np.full(n, -1, dtype=int)
    valid = (~np.isnan(entries)) & (entries > 0)
    vidx  = np.where(valid)[0]
    if len(vidx) == 0:
        return types

    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)

    is_stop = (fut <= stops[vidx, np.newaxis]) & in_range
    is_take = (fut >= takes[vidx, np.newaxis]) & in_range
    hit     = (is_stop | is_take) & in_range

    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)

    hit_stop = is_stop[np.arange(len(vidx)), fhp] & has_hit
    hit_take = is_take[np.arange(len(vidx)), fhp] & has_hit & ~hit_stop

    exit_type = np.where(hit_stop, 1, np.where(hit_take, 2, 0))
    types[vidx] = np.where(has_fut, exit_type, -1)
    return types


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の前処理
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, shares: float | None) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 1:
        return None

    df = df_raw.copy()
    closes = df["Close"]

    df["RSI"] = calc_rsi(closes)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - closes.shift(1)).abs(),
        (df["Low"]  - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = df["Volume"].rolling(n_p).mean().shift(1)
    df["avg_to"]  = (closes * df["Volume"]).rolling(n_p).mean().shift(1)

    # ATR収縮（直近5日平均 < 前5日平均）
    atr5_now  = df["ATR"].rolling(5).mean()
    atr5_prev = df["ATR"].rolling(5).mean().shift(5)
    df["atr_shrink"] = atr5_now < atr5_prev

    # 週足MA25
    wc          = closes.resample("W").last()
    wma25       = wc.rolling(25).mean()
    df["MA25W"] = wma25.reindex(df.index, method="ffill")

    # 週足ダウ理論
    try:
        df["dow_up"] = _weekly_uptrend_series(df)
    except Exception:
        df["dow_up"] = False

    # 時価総額（株数 × 終値、近似）
    if shares is not None:
        df["mktcap"] = closes * shares
    else:
        df["mktcap"] = np.nan  # 不明→フィルタ外す

    # ── エントリーフラグ ──────────────────────────────────────────────────────
    avol = df["avg_vol"]
    rsi  = df["RSI"]
    mktcap_ok  = df["mktcap"].isna() | (df["mktcap"] <= MKTCAP_MAX)
    ma25w_ok   = df["MA25W"].isna() | (closes >= df["MA25W"])

    df["signal"] = (
        (df["Volume"] >= avol * VOL_MULT) & (avol > 0) &
        (rsi >= RSI_LO) & (rsi <= RSI_HI) &
        df["atr_shrink"] &
        (df["avg_to"] >= MIN_AVG_TURNOVER) &
        mktcap_ok &
        ma25w_ok &
        df["dow_up"].astype(bool)
    )

    # ── 翌日始値エントリー・損切りライン ─────────────────────────────────────
    n_rows  = len(df)
    c_arr   = df["Close"].values.astype(float)
    o_arr   = df["Open"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)
    sigs    = df["signal"].values.astype(bool)
    avg_tos = df["avg_to"].values.astype(float)

    valid = (
        sigs &
        (~np.isnan(atrs)) & (atrs > 0) &
        (~np.isnan(avg_tos)) & (avg_tos >= MIN_AVG_TURNOVER) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(valid)[0]

    entries = np.full(n_rows, np.nan)
    stops   = np.full(n_rows, np.nan)

    e  = o_arr[vidx + 1]
    a  = atrs[vidx]
    ve = (e > 0) & (~np.isnan(e))
    vi2 = vidx[ve]; e2 = e[ve]; a2 = a[ve]
    s2  = np.maximum(e2 - a2 * ATR_MULT, e2 * (1 - ATR_FLOOR))

    entries[vi2] = e2
    stops[vi2]   = s2
    df["_entry"] = entries
    df["_stop"]  = stops

    return df


# ──────────────────────────────────────────────────────────────────────────────
# メトリクス計算
# ──────────────────────────────────────────────────────────────────────────────
def _metrics(rets: pd.Series, types: pd.Series, trading_days: int) -> dict:
    rets  = rets.dropna()
    types = types.loc[rets.index]
    n = len(rets)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "stop_r": 0.0, "take_r": 0.0, "force_r": 0.0, "spd": 0.0}
    w  = rets[rets > 0]
    l  = rets[rets <= 0]
    wr = len(w) / n * 100
    aw = w.mean() if len(w) > 0 else 0.0
    al = l.mean() if len(l) > 0 else 0.0
    pf = abs(aw / al) if al != 0 else float("inf")
    stop_r  = (types == 1).sum() / n * 100
    take_r  = (types == 2).sum() / n * 100
    force_r = (types == 0).sum() / n * 100
    return {
        "n": n, "wr": round(wr, 1), "pf": round(pf, 2),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "stop_r": round(stop_r, 1),
        "take_r": round(take_r, 1),
        "force_r": round(force_r, 1),
        "spd": round(n / trading_days, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data: dict = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n時価総額データ取得中（yfinance）...")
    shares_dict = fetch_all_shares(list(raw_data.keys()))
    covered = len(shares_dict)
    print(f"  株数取得成功: {covered}/{len(raw_data)} 銘柄")
    if covered < len(raw_data) * 0.5:
        print("  ※ 株数取得が半数未満 → 時価総額フィルタを全銘柄に適用できない場合あり")

    print("\n前処理中...")
    all_dfs: list[pd.DataFrame] = []
    done = 0
    lock = threading.Lock()

    def _process(item):
        ticker, df = item
        sh = shares_dict.get(ticker)
        return preprocess(df, sh)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process, item): item[0] for item in raw_data.items()}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                all_dfs.append(res)
            with lock:
                done += 1
            if done % 500 == 0 or done == len(raw_data):
                print(f"  {done}/{len(raw_data)} 完了  有効: {len(all_dfs)}")

    # シグナル件数確認
    total_signals = sum(df["signal"].sum() for df in all_dfs)
    print(f"\n総シグナル数（フィルタ後）: {total_signals:,} 件")

    trading_days = max(
        (len(df) - MIN_HISTORY for df in all_dfs if df is not None),
        default=250,
    )
    print(f"推定取引日数: {trading_days} 日")

    # ── RRごとに計算 ──────────────────────────────────────────────────────────
    print("\nRR別バックテスト計算中...")
    rr_results = {}
    for rr in RR_LIST:
        all_rets  = []
        all_types = []
        for df in all_dfs:
            e_arr  = df["_entry"].values.astype(float)
            s_arr  = df["_stop"].values.astype(float)
            c_arr  = df["Close"].values.astype(float)
            # take = entry + (entry - stop) × RR
            t_arr = np.where(
                ~np.isnan(e_arr) & ~np.isnan(s_arr),
                e_arr + (e_arr - s_arr) * rr,
                np.nan,
            )
            rets  = _exit_returns_vec(c_arr, e_arr, s_arr, t_arr)
            types = _exit_types_vec(c_arr, e_arr, s_arr, t_arr)

            mask = ~np.isnan(rets)
            all_rets.append(pd.Series(rets[mask]))
            all_types.append(pd.Series(types[mask]))

        combined_rets  = pd.concat(all_rets,  ignore_index=True)
        combined_types = pd.concat(all_types, ignore_index=True)
        rr_results[rr] = _metrics(combined_rets, combined_types, trading_days)

    # ── 結果表示 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("【ATRスクイーズ型 バックテスト結果】")
    print(f"  条件: 出来高{VOL_MULT:.0f}倍 / RSI{RSI_LO:.0f}〜{RSI_HI:.0f} / ATR収縮(5日) / 売買代金≥3000万")
    print(f"        時価総額≤300億 / 週足MA25上 / 週足ダウ理論 / エントリー:翌日始値 / 損切り:ATR×{ATR_MULT}(-10%上限)")
    print()
    hdr = f"  {'RR':<8} {'勝率':>7} {'PF':>6} {'平均益':>8} {'平均損':>8} {'損切率':>7} {'利確率':>7} {'強制率':>7} {'件数':>6} {'件/日':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rr, m in rr_results.items():
        print(
            f"  RR1:{rr:<4.1f}  "
            f"{m['wr']:>6.1f}%  "
            f"{m['pf']:>6.2f}  "
            f"{m['avg_win']:>+7.2f}%  "
            f"{m['avg_loss']:>+7.2f}%  "
            f"{m['stop_r']:>6.1f}%  "
            f"{m['take_r']:>6.1f}%  "
            f"{m['force_r']:>6.1f}%  "
            f"{m['n']:>6,}  "
            f"{m['spd']:>6.2f}"
        )
    print("=" * 70)

    # 時価総額フィルタの効果確認
    print("\n【参考: 時価総額フィルタなし（株数未取得銘柄を除外せず）の件数】")
    no_mktcap_filter = sum(
        ((df["_entry"].notna()) & (df.index.isin(df.index[df["signal"]]))).sum()
        for df in all_dfs
    )
    uncovered = len(raw_data) - covered
    if uncovered > 0:
        print(f"  株数未取得: {uncovered} 銘柄（これらの時価総額フィルタはスキップ）")
