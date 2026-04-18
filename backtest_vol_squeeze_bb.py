#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ボラティリティ収縮ブレイク型バックテスト（ATRスクイーズ知見適用版）
条件:
  ATR収縮（直近5日ATR < 前10日ATR）
  BB幅収縮（BB幅 ≤ 20日平均BB幅）
  出来高2倍以上
  RSI 45〜55
  時価総額100億円以下
  売買代金3000万円以上
エントリー: 翌日始値（成行）
損切り: ATR×2.0（上限-10%）
RR: 1:1 / 1:1.5 / 1:2
"""

import pickle, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 60
MAX_HOLD    = 20
MAX_WORKERS = 20

MKTCAP_MAX  = 10_000_000_000   # 100億円
VOL_MULT    = 2.0
RSI_LO, RSI_HI = 45.0, 55.0
RR_LIST     = [1.0, 1.5, 2.0]


# ── 株数取得（並列）──────────────────────────────────────────────────────────
def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        fi = yf.Ticker(ticker).fast_info
        sh = getattr(fi, "shares", None)
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t, sh = fut.result()
            if sh:
                result[t] = sh
            done += 1
            if done % 300 == 0 or done == len(tickers):
                print(f"  株数取得: {done}/{len(tickers)}  成功: {len(result)}")
    return result


# ── 出口計算（ベクトル化）────────────────────────────────────────────────────
def _calc_rets(closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    n = len(closes)
    if len(vidx) == 0:
        return np.array([])
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    hit     = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep      = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets    = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


# ── 決済種別（損切り/利確/強制）─────────────────────────────────────────────
def _exit_types(closes: np.ndarray, vidx: np.ndarray,
                e_arr: np.ndarray, s_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """0=強制, 1=損切り, 2=利確"""
    n = len(closes)
    if len(vidx) == 0:
        return np.array([], dtype=int)
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    is_stop = (fut <= s_arr[:, np.newaxis]) & in_range
    is_take = (fut >= t_arr[:, np.newaxis]) & in_range
    hit     = (is_stop | is_take)
    has_hit = hit.any(axis=1)
    has_fut = in_range.any(axis=1)
    last_v  = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp     = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    hit_stop = is_stop[np.arange(len(vidx)), fhp] & has_hit
    hit_take = is_take[np.arange(len(vidx)), fhp] & has_hit & ~hit_stop
    types    = np.where(hit_stop, 1, np.where(hit_take, 2, 0))
    return types[has_fut]


# ── 1銘柄前処理 ────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, shares: float | None) -> pd.DataFrame | None:
    if df_raw is None or len(df_raw) < MIN_HISTORY + 20:
        return None

    df = df_raw.copy()
    c  = df["Close"]
    h  = df["High"]
    l  = df["Low"]
    v  = df["Volume"]
    o  = df["Open"]

    # ATR 14日
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # RSI
    df["RSI"] = calc_rsi(c)

    # 売買代金・出来高
    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = v.rolling(n_p).mean().shift(1)
    df["avg_to"]  = (c * v).rolling(n_p).mean().shift(1)

    # ATR収縮: 直近5日平均 < 前10日平均
    df["atr_5d"]       = df["ATR"].rolling(5).mean()
    df["atr_10d_prev"] = df["ATR"].shift(5).rolling(10).mean()

    # BB幅（正規化）と20日平均
    bb_std = c.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    df["bb_width"]     = np.where(bb_mid > 0, 2 * bb_std / bb_mid, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(20).mean()

    # 時価総額（株数 × 終値、近似）
    df["mktcap"] = (c * shares) if shares is not None else np.nan

    return df


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n時価総額データ取得中（yfinance）...")
    shares_map = fetch_all_shares(list(raw_data.keys()))
    print(f"  取得完了: {len(shares_map)}/{len(raw_data)} 銘柄")

    print("\n前処理中...")
    processed = {}
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        sh = shares_map.get(tk)
        r  = preprocess(df_raw, sh)
        if r is not None:
            processed[tk] = r
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)} 完了  有効: {len(processed)}")
    print(f"  完了  有効: {len(processed)}")

    # 取引日数
    dates = set()
    for df in list(processed.values())[:30]:
        dates.update(df.index.tolist())
    trading_days = len(dates)
    print(f"\n推定取引日数: {trading_days} 日")

    # ── 全RRのリターン収集 ─────────────────────────────────────────────────
    all_rets   = {rr: [] for rr in RR_LIST}
    all_types  = {rr: [] for rr in RR_LIST}  # 決済種別
    total_sig  = 0

    for tk, df in processed.items():
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        avg_v = df["avg_vol"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        atr5d = df["atr_5d"].values.astype(float)
        atr10p= df["atr_10d_prev"].values.astype(float)
        bb_w  = df["bb_width"].values.astype(float)
        bb_wa = df["bb_width_avg"].values.astype(float)
        mktcap= df["mktcap"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        mktcap_ok  = np.isnan(mktcap) | (mktcap <= MKTCAP_MAX)
        atr_shrink = (atr5d < atr10p) & (~np.isnan(atr5d)) & (~np.isnan(atr10p)) & (atr10p > 0)
        bb_shrink  = (bb_w <= bb_wa) & (~np.isnan(bb_w)) & (~np.isnan(bb_wa))
        rsi_ok     = (rsi >= RSI_LO) & (rsi <= RSI_HI) & (~np.isnan(rsi))
        vol_ok     = (vol_a >= avg_v * VOL_MULT) & (avg_v > 0)
        to_ok      = (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER)

        sig = (
            (~np.isnan(atr)) & (atr > 0) &
            to_ok & vol_ok & rsi_ok &
            atr_shrink & bb_shrink &
            mktcap_ok &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )

        vidx = np.where(sig)[0]
        if len(vidx) == 0:
            continue

        total_sig += len(vidx)
        e_arr = next_o[vidx]
        a_arr = atr[vidx]
        s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)

        for rr in RR_LIST:
            t_arr = e_arr + (e_arr - s_arr) * rr
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)
            types = _exit_types(c_a, vidx, e_arr, s_arr, t_arr)
            all_rets[rr].extend(rets.tolist())
            all_types[rr].extend(types.tolist())

    # ── 結果表示 ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"【ボラティリティ収縮ブレイク型（ATRスクイーズ知見適用版）】")
    print(f"  ATR収縮（5日<前10日）+ BB幅収縮 + 出来高{VOL_MULT}倍+ RSI{RSI_LO:.0f}-{RSI_HI:.0f}")
    print(f"  時価総額100億以下 / 売買代金3000万以上 / エントリー翌日始値")
    print(f"  総シグナル数: {total_sig} 件 ({total_sig/trading_days:.2f}件/日)")
    print(f"{'=' * 65}")

    for rr in RR_LIST:
        r = np.array(all_rets[rr])
        t = np.array(all_types[rr], dtype=int)
        if len(r) == 0:
            print(f"\n  RR 1:{rr}  — データなし")
            continue

        wins   = r[r > 0]
        losses = r[r <= 0]
        wr     = len(wins) / len(r) * 100
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd    = len(r) / trading_days

        # 決済種別
        n_stop  = int((t == 1).sum())
        n_take  = int((t == 2).sum())
        n_force = int((t == 0).sum())
        n_total = len(t)

        print(f"\n  ── RR 1:{rr} ──────────────────────────────────────────────")
        print(f"  総トレード: {len(r):,}件  ({spd:.2f}/日)")
        print(f"  勝率      : {wr:.1f}%")
        print(f"  PF        : {pf:.2f}")
        print(f"  期待値    : {ev:+.2f}%/トレード")
        print(f"  平均利益  : +{avg_w:.2f}%  平均損失: {avg_l:.2f}%")
        print(f"  決済内訳  :")
        if n_total > 0:
            print(f"    損切り  : {n_stop:4d}件 ({n_stop/n_total*100:.1f}%)  "
                  f"avg {r[t==1].mean():.2f}%" if n_stop > 0 else
                  f"    損切り  : {n_stop:4d}件 ({n_stop/n_total*100:.1f}%)")
            print(f"    利確    : {n_take:4d}件 ({n_take/n_total*100:.1f}%)  "
                  f"avg +{r[t==2].mean():.2f}%" if n_take > 0 else
                  f"    利確    : {n_take:4d}件 ({n_take/n_total*100:.1f}%)")
            print(f"    強制終了: {n_force:4d}件 ({n_force/n_total*100:.1f}%)  "
                  f"avg {r[t==0].mean():+.2f}%" if n_force > 0 else
                  f"    強制終了: {n_force:4d}件 ({n_force/n_total*100:.1f}%)")

        passed = wr >= 52.0 and pf >= 1.5 and spd >= 0.5
        print(f"  評価      : {'★ 合格' if passed else '× 不合格'} (基準: 勝率52%/PF1.5/0.5件/日)")

    # ── 比較サマリー ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"【RR比較サマリー】")
    print(f"  {'RR':>6}  {'勝率':>8}  {'PF':>6}  {'期待値':>8}  {'件数':>6}  {'件/日':>5}")
    print("  " + "-" * 50)
    for rr in RR_LIST:
        r = np.array(all_rets[rr])
        if len(r) == 0:
            continue
        wins   = r[r > 0]
        losses = r[r <= 0]
        wr     = len(wins) / len(r) * 100
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
        ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
        spd    = len(r) / trading_days
        mark   = "★" if (wr >= 52.0 and pf >= 1.5 and spd >= 0.5) else " "
        print(f"{mark} 1:{rr:<4.1f}  {wr:>7.1f}%  {pf:>6.2f}  {ev:>+8.2f}%  {len(r):>6,}  {spd:>5.2f}")


if __name__ == "__main__":
    main()
