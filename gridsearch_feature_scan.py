#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最高WR・PF条件スキャン
~25の特徴量フラグを単独・ペア・トリプレットで総当たり
→ WR/PFが高い条件の組み合わせを発見する

エントリー: 翌日始値 / RR1:1.5 / ATR×2.0（-10%上限）
フィルタ:   売買代金3000万以上 / ATR有効 / 最低シグナル20件
"""

import itertools, pickle, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import calc_rsi, MIN_AVG_TURNOVER, BREAKOUT_DAYS

CACHE_PATH  = Path(__file__).parent / "backtest_cache.pkl"
MIN_HISTORY = 80
MAX_HOLD    = 20
MAX_WORKERS = 20
RR          = 1.5
MIN_TRADES  = 30   # 合格最低件数

# 表示設定
TOP_N       = 30   # 上位表示件数
MIN_WR_SHOW = 45.0 # この勝率以上のみ表示（単独）
MIN_WR_PAIR = 48.0 # ペアの表示閾値


# ── 株数取得 ──────────────────────────────────────────────────────────────────
def _fetch_shares(t):
    try:
        fi = yf.Ticker(t).fast_info
        sh = getattr(fi, "shares", None)
        return t, float(sh) if sh else None
    except Exception:
        return t, None


def fetch_all_shares(tickers):
    res = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            t, sh = fut.result()
            if sh: res[t] = sh
            done += 1
            if done % 300 == 0 or done == len(tickers):
                print(f"  {done}/{len(tickers)} 完了  取得: {len(res)}")
    return res


# ── 出口計算 ──────────────────────────────────────────────────────────────────
def _calc_rets(closes, vidx, e_arr, s_arr, t_arr):
    n = len(closes)
    if len(vidx) == 0:
        return np.array([])
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    hit      = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


# ── 前処理：全特徴量を一括計算 ────────────────────────────────────────────────
def preprocess(df_raw, shares):
    if df_raw is None or len(df_raw) < MIN_HISTORY + 20:
        return None

    df = df_raw.copy()
    c  = df["Close"]
    h  = df["High"]
    l  = df["Low"]
    v  = df["Volume"]
    o  = df["Open"]

    # ── 基本指標 ──────────────────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"]  = tr.rolling(14).mean()
    df["RSI"]  = calc_rsi(c)
    df["MA25"] = c.rolling(25).mean()
    df["MA75"] = c.rolling(75).mean()
    n_p = BREAKOUT_DAYS + 1
    df["avg_vol"] = v.rolling(n_p).mean().shift(1)
    df["avg_to"]  = (c * v).rolling(n_p).mean().shift(1)

    # ── 時価総額 ──────────────────────────────────────────────────────────────
    df["mktcap"] = (c * shares) if shares is not None else np.nan

    # ── ATR 収縮（3日・5日ウィンドウ）────────────────────────────────────────
    df["atr3"]    = df["ATR"].rolling(3).mean()
    df["atr3p"]   = df["ATR"].shift(3).rolling(3).mean()
    df["atr5"]    = df["ATR"].rolling(5).mean()
    df["atr5p"]   = df["ATR"].shift(5).rolling(5).mean()
    df["atr10p"]  = df["ATR"].shift(5).rolling(10).mean()   # 前10日
    df["atr20avg"]= df["ATR"].rolling(20).mean()            # 20日平均

    # ── 出来高 ────────────────────────────────────────────────────────────────
    df["vol5d"]   = v.rolling(5).mean()
    df["vol5dp"]  = v.shift(5).rolling(5).mean()

    # ── MACD (12/26/9) + GC ───────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    msig  = macd.ewm(span=9, adjust=False).mean()
    gc0   = (macd > msig) & (macd.shift(1) <= msig.shift(1))
    gc1   = (macd.shift(1) > msig.shift(1)) & (macd.shift(2) <= msig.shift(2))
    gc2   = (macd.shift(2) > msig.shift(2)) & (macd.shift(3) <= msig.shift(3))
    df["macd_gc"] = gc0 | gc1 | gc2
    df["macd_pos"]= macd > msig

    # ── モメンタム加速 ────────────────────────────────────────────────────────
    df["mom5"]    = c / c.shift(5) - 1
    df["mom5p"]   = c.shift(5) / c.shift(10) - 1

    # ── BB幅 ──────────────────────────────────────────────────────────────────
    bbs = c.rolling(20).std()
    bbm = c.rolling(20).mean()
    df["bb_width"] = np.where(bbm > 0, 2 * bbs / bbm, np.nan)
    df["bb_avg"]   = df["bb_width"].rolling(20).mean()

    # ── 価格帯 ────────────────────────────────────────────────────────────────
    df["high20"]   = h.shift(1).rolling(20).max()           # 前日まで20日高値
    df["pct_chg"]  = c.pct_change() * 100                   # 当日騰落率
    df["cl_hi_gap"]= np.where(h > 0, (h - c) / h, np.nan)  # 高値からの距離
    df["above_ma25_pct"] = (c - df["MA25"]) / df["MA25"] * 100  # MA25乖離率

    # ── 週足MA25 ──────────────────────────────────────────────────────────────
    try:
        wc = c.resample("W").last()
        df["MA25W"] = wc.rolling(25).mean().reindex(df.index, method="ffill")
    except Exception:
        df["MA25W"] = np.nan

    # ── 52週高値 ──────────────────────────────────────────────────────────────
    df["high52w"] = c.shift(1).rolling(252, min_periods=80).max()

    # ── 連続陽線 ──────────────────────────────────────────────────────────────
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumsum()

    return df


# ── リターン集計ユーティリティ ────────────────────────────────────────────────
def _stat(r_list):
    r = np.array(r_list)
    if len(r) < MIN_TRADES:
        return None
    wins   = r[r > 0]; losses = r[r <= 0]
    wr     = len(wins) / len(r) * 100
    avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_l  = losses.mean() if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    return {"wr": wr, "pf": pf, "ev": ev, "n": len(r)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("キャッシュ読み込み中...")
    with open(CACHE_PATH, "rb") as f:
        raw_data = pickle.load(f)["data"]
    print(f"  {len(raw_data)} 銘柄")

    print("\n時価総額データ取得中...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中（全指標計算）...")
    processed = {}
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(df_raw, shares_map.get(tk))
        if r is not None:
            processed[tk] = r
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)} 完了  有効: {len(processed)}")
    print(f"  完了  有効: {len(processed)}")

    dates = set()
    for df in list(processed.values())[:30]:
        dates.update(df.index.tolist())
    trading_days = len(dates)
    print(f"\n推定取引日数: {trading_days} 日")

    # ── 全銘柄の特徴量フラグと出口リターンを事前計算 ─────────────────────────
    # 各特徴量: {feat_name: list of (base_vidx, all_rets)}
    # より効率的: 全銘柄について base_ok での returns を計算しておき
    # 各特徴量フラグで idx を絞り込む

    # 特徴量定義（名前 → ラムダ）
    # ラムダは preprocessed df の各配列を受け取り bool array を返す
    feat_defs = {
        # RSI帯
        "RSI20-40":    lambda d: (d["rsi"] >= 20) & (d["rsi"] <= 40),
        "RSI30-50":    lambda d: (d["rsi"] >= 30) & (d["rsi"] <= 50),
        "RSI40-60":    lambda d: (d["rsi"] >= 40) & (d["rsi"] <= 60),
        "RSI45-55":    lambda d: (d["rsi"] >= 45) & (d["rsi"] <= 55),
        "RSI50-70":    lambda d: (d["rsi"] >= 50) & (d["rsi"] <= 70),
        "RSI55-70":    lambda d: (d["rsi"] >= 55) & (d["rsi"] <= 70),
        "RSI60-80":    lambda d: (d["rsi"] >= 60) & (d["rsi"] <= 80),
        # ATR収縮
        "ATR収縮3日":  lambda d: (d["atr3"] < d["atr3p"]) & (d["atr3p"] > 0),
        "ATR収縮5日":  lambda d: (d["atr5"] < d["atr5p"]) & (d["atr5p"] > 0),
        "ATR収縮10日": lambda d: (d["atr5"] < d["atr10p"]) & (d["atr10p"] > 0),
        "ATR低ボラ":   lambda d: (d["atr"] < d["atr20avg"] * 0.8) & (d["atr20avg"] > 0),
        "ATR拡大":     lambda d: (d["atr5"] > d["atr5p"]) & (d["atr5p"] > 0),
        # 出来高
        "出来高1.5倍": lambda d: (d["vol"] >= d["avgvol"] * 1.5) & (d["avgvol"] > 0),
        "出来高2倍":   lambda d: (d["vol"] >= d["avgvol"] * 2.0) & (d["avgvol"] > 0),
        "出来高3倍":   lambda d: (d["vol"] >= d["avgvol"] * 3.0) & (d["avgvol"] > 0),
        "出来高5倍":   lambda d: (d["vol"] >= d["avgvol"] * 5.0) & (d["avgvol"] > 0),
        "出来高減少":  lambda d: (d["vol5d"] < d["vol5dp"]) & (d["vol5dp"] > 0),
        # トレンド
        "MA25上":      lambda d: d["close"] > d["ma25"],
        "MA75上":      lambda d: (d["close"] > d["ma75"]) & (~np.isnan(d["ma75"])),
        "MA25W上":     lambda d: (d["close"] > d["ma25w"]) & (~np.isnan(d["ma25w"])),
        "MA25>MA75":   lambda d: (d["ma25"] > d["ma75"]) & (~np.isnan(d["ma75"])),
        # パターン
        "高値引け3%":  lambda d: d["cl_hi_gap"] < 0.03,
        "高値引け5%":  lambda d: d["cl_hi_gap"] < 0.05,
        "20日高値突破": lambda d: (d["close"] > d["high20"]) & (~np.isnan(d["high20"])),
        "20日高値95%": lambda d: (d["close"] >= d["high20"] * 0.95) & (~np.isnan(d["high20"])),
        "52週高値90%": lambda d: (d["close"] >= d["hi52w"] * 0.90) & (~np.isnan(d["hi52w"])),
        "52週高値95%": lambda d: (d["close"] >= d["hi52w"] * 0.95) & (~np.isnan(d["hi52w"])),
        "モメンタム加速": lambda d: (d["mom5"] > d["mom5p"]) & (~np.isnan(d["mom5p"])),
        "MACD GC3日":  lambda d: d["macd_gc"],
        "MACD正":      lambda d: d["macd_pos"],
        "当日+1%":     lambda d: d["pct_chg"] >= 1.0,
        "当日+2%":     lambda d: d["pct_chg"] >= 2.0,
        "3連続陽線":   lambda d: d["consec_up"] >= 3,
        "BB収縮":      lambda d: (d["bb_w"] <= d["bb_avg"]) & (~np.isnan(d["bb_avg"])),
        "MA25乖離+5%": lambda d: d["ma25_dev"] >= 5.0,
        "MA25乖離-5%": lambda d: d["ma25_dev"] <= -5.0,
        # 時価総額
        "時価総額100億": lambda d: (~np.isnan(d["mktcap"])) & (d["mktcap"] <= 10e9),
        "時価総額200億": lambda d: (~np.isnan(d["mktcap"])) & (d["mktcap"] <= 20e9),
        "時価総額300億": lambda d: np.isnan(d["mktcap"]) | (d["mktcap"] <= 30e9),
    }
    feat_names = list(feat_defs.keys())
    n_feat = len(feat_names)
    print(f"\n特徴量フラグ数: {n_feat}")
    print(f"単独: {n_feat}  ペア: {n_feat*(n_feat-1)//2}  3組: {n_feat*(n_feat-1)*(n_feat-2)//6}")

    # ── 全銘柄でのフラグ×リターン計算 ─────────────────────────────────────────
    # feat_idx_rets[feat_name] = list of (vidx_in_signal, ret) pairs
    # 効率化: base_ok での全シグナルとリターンを計算し、
    # フラグで絞り込んでWR/PFを計算する

    # 銘柄ごとに: base_ok なシグナル idx と その リターン を計算
    # フラグ値も並行保存
    print("\nシグナル計算中...")

    # feat_name → {signal_day_flags} + returns の対応を集約
    # 実装: 全銘柄について (flags_array, rets_array) を計算して連結
    # flags_array shape: (n_signals, n_feat)
    # rets_array  shape: (n_signals,)

    all_flags = []   # list of (n_signals_in_ticker, n_feat) arrays
    all_rets  = []   # list of (n_signals_in_ticker,) arrays

    for i, (tk, df) in enumerate(processed.items(), 1):
        n      = len(df)
        c_a    = df["Close"].values.astype(float)
        o_a    = df["Open"].values.astype(float)
        atr    = df["ATR"].values.astype(float)
        rsi    = df["RSI"].values.astype(float)
        ma25   = df["MA25"].values.astype(float)
        ma75   = df["MA75"].values.astype(float)
        ma25w  = df["MA25W"].values.astype(float)
        avg_v  = df["avg_vol"].values.astype(float)
        to_a   = df["avg_to"].values.astype(float)
        mktcap = df["mktcap"].values.astype(float)
        vol_a  = df["Volume"].values.astype(float)
        atr3   = df["atr3"].values.astype(float)
        atr3p  = df["atr3p"].values.astype(float)
        atr5   = df["atr5"].values.astype(float)
        atr5p  = df["atr5p"].values.astype(float)
        atr10p = df["atr10p"].values.astype(float)
        atr20a = df["atr20avg"].values.astype(float)
        vol5d  = df["vol5d"].values.astype(float)
        vol5dp = df["vol5dp"].values.astype(float)
        macd_gc= df["macd_gc"].values.astype(bool)
        macd_p = df["macd_pos"].values.astype(bool)
        mom5   = df["mom5"].values.astype(float)
        mom5p  = df["mom5p"].values.astype(float)
        bb_w   = df["bb_width"].values.astype(float)
        bb_avg = df["bb_avg"].values.astype(float)
        high20 = df["high20"].values.astype(float)
        hi52w  = df["high52w"].values.astype(float)
        pct_ch = df["pct_chg"].values.astype(float)
        cl_hig = df["cl_hi_gap"].values.astype(float)
        ma25d  = df["above_ma25_pct"].values.astype(float)
        consec = df["consec_up"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )
        vidx = np.where(base_ok)[0]
        if len(vidx) == 0:
            continue

        # 各シグナルのリターンを計算（全シグナル、フラグなし）
        e_arr = next_o[vidx]
        a_arr = atr[vidx]
        s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
        t_arr = e_arr + (e_arr - s_arr) * RR
        rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)

        if len(rets) != len(vidx):
            # 末尾のシグナルでデータ不足 → 対応する vidx を絞る
            # _calc_rets は has_fut=True のもののみ返す
            # 簡易的に: 全vidxのunicode rets を取得して has_fut を確認
            raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
            in_range = raw_idx < n
            has_fut  = in_range.any(axis=1)
            vidx = vidx[has_fut]
            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * RR
            rets  = _calc_rets(c_a, vidx, e_arr, s_arr, t_arr)

        if len(vidx) == 0 or len(rets) != len(vidx):
            continue

        # 各シグナル日の特徴量フラグを計算
        d = {
            "rsi": rsi[vidx],    "close": c_a[vidx],
            "ma25": ma25[vidx],  "ma75": ma75[vidx],
            "ma25w": ma25w[vidx],"avgvol": avg_v[vidx],
            "vol": vol_a[vidx],  "mktcap": mktcap[vidx],
            "atr": atr[vidx],    "atr3": atr3[vidx],
            "atr3p": atr3p[vidx],"atr5": atr5[vidx],
            "atr5p": atr5p[vidx],"atr10p": atr10p[vidx],
            "atr20avg": atr20a[vidx],
            "vol5d": vol5d[vidx],"vol5dp": vol5dp[vidx],
            "macd_gc": macd_gc[vidx], "macd_pos": macd_p[vidx],
            "mom5": mom5[vidx],  "mom5p": mom5p[vidx],
            "bb_w": bb_w[vidx],  "bb_avg": bb_avg[vidx],
            "high20": high20[vidx],"hi52w": hi52w[vidx],
            "pct_chg": pct_ch[vidx],"cl_hi_gap": cl_hig[vidx],
            "ma25_dev": ma25d[vidx],"consec_up": consec[vidx],
        }

        # (n_sig, n_feat) のフラグ行列
        flags = np.zeros((len(vidx), n_feat), dtype=bool)
        for j, fname in enumerate(feat_names):
            try:
                flags[:, j] = feat_defs[fname](d)
            except Exception:
                flags[:, j] = False

        all_flags.append(flags)
        all_rets.append(rets)

        if i % 500 == 0:
            print(f"  {i}/{len(processed)} 完了")

    if not all_flags:
        print("データなし")
        return

    # 全銘柄を連結
    FLAGS = np.vstack(all_flags)   # (N_total, n_feat)
    RETS  = np.concatenate(all_rets)  # (N_total,)
    print(f"\n総シグナル数（ベース）: {len(RETS):,}件 ({len(RETS)/trading_days:.1f}/日)")
    print(f"全体WR: {(RETS>0).mean()*100:.1f}%  PF: {abs(RETS[RETS>0].mean()/RETS[RETS<=0].mean()):.2f}")

    # ── 単独特徴量スキャン ────────────────────────────────────────────────────
    print("\n単独特徴量スキャン中...")
    single_results = []
    for j, fname in enumerate(feat_names):
        mask = FLAGS[:, j]
        r = RETS[mask]
        s = _stat(r.tolist())
        if s:
            s["name"] = fname
            s["spd"]  = s["n"] / trading_days
            single_results.append(s)

    single_results.sort(key=lambda x: x["wr"], reverse=True)

    print(f"\n{'='*70}")
    print("【単独特徴量スキャン】 WR降順 TOP30")
    print(f"{'条件':20}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件数':>6}  {'件/日':>5}")
    print("-" * 62)
    for r in single_results[:TOP_N]:
        mark = "★" if r["wr"] >= 52 and r["pf"] >= 1.5 and r["spd"] >= 0.5 else " "
        print(f"{mark}{r['name']:20}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
              f"{r['ev']:>+7.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}")

    # ── ペア特徴量スキャン ────────────────────────────────────────────────────
    print(f"\nペアスキャン中（{n_feat*(n_feat-1)//2}通り）...")
    pair_results = []
    for j1, j2 in itertools.combinations(range(n_feat), 2):
        mask = FLAGS[:, j1] & FLAGS[:, j2]
        r = RETS[mask]
        s = _stat(r.tolist())
        if s:
            s["name"] = f"{feat_names[j1]} ＋ {feat_names[j2]}"
            s["spd"]  = s["n"] / trading_days
            pair_results.append(s)

    pair_results.sort(key=lambda x: x["wr"], reverse=True)

    print(f"\n{'='*70}")
    print("【ペアスキャン】 WR降順 TOP30")
    print(f"{'条件':42}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件数':>6}  {'件/日':>5}")
    print("-" * 80)
    shown = 0
    for r in pair_results:
        if shown >= TOP_N:
            break
        mark = "★" if r["wr"] >= 52 and r["pf"] >= 1.5 and r["spd"] >= 0.5 else " "
        print(f"{mark}{r['name']:42}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
              f"{r['ev']:>+7.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}")
        shown += 1

    # ── 3組スキャン（上位15特徴量の組み合わせ）────────────────────────────────
    # 上位15をWRベースで選択
    top_idx = [feat_names.index(r["name"]) for r in single_results[:15]
               if r["name"] in feat_names][:15]
    n3 = len(list(itertools.combinations(top_idx, 3)))
    print(f"\n3組スキャン中（上位15特徴量から {n3} 通り）...")
    triple_results = []
    for j1, j2, j3 in itertools.combinations(top_idx, 3):
        mask = FLAGS[:, j1] & FLAGS[:, j2] & FLAGS[:, j3]
        r = RETS[mask]
        s = _stat(r.tolist())
        if s:
            s["name"] = f"{feat_names[j1]} ＋ {feat_names[j2]} ＋ {feat_names[j3]}"
            s["spd"]  = s["n"] / trading_days
            triple_results.append(s)

    triple_results.sort(key=lambda x: x["wr"], reverse=True)

    print(f"\n{'='*70}")
    print("【3組スキャン】 WR降順 TOP20")
    print(f"{'条件':55}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件数':>5}  {'件/日':>5}")
    print("-" * 90)
    shown = 0
    for r in triple_results:
        if shown >= 20:
            break
        mark = "★" if r["wr"] >= 52 and r["pf"] >= 1.5 and r["spd"] >= 0.5 else " "
        print(f"{mark}{r['name']:55}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
              f"{r['ev']:>+7.2f}%  {r['n']:>5,}  {r['spd']:>5.2f}")
        shown += 1

    # ── PF最高ランキング ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("【PF最高ランキング（全スキャン合算）】 件/日≥0.5 のみ")
    all_results = single_results + pair_results + triple_results
    pf_rank = sorted(
        [r for r in all_results if r["spd"] >= 0.5 and r["n"] >= MIN_TRADES],
        key=lambda x: x["pf"], reverse=True
    )
    print(f"{'条件':55}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件数':>5}  {'件/日':>5}")
    print("-" * 90)
    for r in pf_rank[:20]:
        mark = "★" if r["wr"] >= 52 and r["pf"] >= 1.5 and r["spd"] >= 0.5 else " "
        print(f"{mark}{r['name']:55}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
              f"{r['ev']:>+7.2f}%  {r['n']:>5,}  {r['spd']:>5.2f}")

    # ── 総合サマリー ──────────────────────────────────────────────────────────
    all_pass = [r for r in all_results
                if r["wr"] >= 52 and r["pf"] >= 1.5 and r["spd"] >= 0.5]
    all_pass.sort(key=lambda x: x["pf"], reverse=True)
    print(f"\n{'='*70}")
    print(f"【合格条件（WR≥52% かつ PF≥1.5 かつ 件/日≥0.5）】 {len(all_pass)} 件")
    if all_pass:
        print(f"{'条件':55}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件/日':>5}")
        print("-" * 85)
        for r in all_pass[:20]:
            print(f"★{r['name']:55}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}  "
                  f"{r['ev']:>+7.2f}%  {r['spd']:>5.2f}")
    else:
        print("  なし — WR最高の条件:")
        best = single_results[0] if single_results else None
        if best:
            print(f"  {best['name']}: WR={best['wr']:.1f}% / PF={best['pf']:.2f} / {best['spd']:.2f}件/日")


if __name__ == "__main__":
    main()
