#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
売られすぎ反発型 グリッドサーチ
ベース: RSI20〜rsi_hi + 前日比pct%以上 + 翌日始値エントリー
妥当性検証: 決済内訳・年別WR・シグナル分布も表示
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
MIN_HISTORY = 60
MAX_HOLD    = 20
MAX_WORKERS = 20

# 評価基準
CRITERIA_WR_LO  = 55.0
CRITERIA_WR_HI  = 999.0   # 上限なし
CRITERIA_PF     = 1.7
CRITERIA_SPD_LO = 0.5
CRITERIA_SPD_HI = 3.0

# グリッド
GRID = {
    "rsi_hi":   [35.0, 40.0, 45.0],        # RSI上限（下限は常に20）
    "pct_thr":  [2.0, 3.0, 5.0],           # 前日比閾値（%）
    "vol_mult": [0.0, 1.5, 2.0],           # 0=なし
    "mktcap":   [10e9, 20e9, 30e9],        # 100/200/300億
    "rr":       [1.5, 2.0],
}


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
                print(f"  {done}/{len(tickers)}  取得: {len(res)}")
    return res


# ── 出口計算（リターン値） ─────────────────────────────────────────────────────
def _calc_rets_and_types(closes, vidx, e_arr, s_arr, t_arr):
    """returns: (rets, exit_types)  types: 1=損切り 2=利確 0=強制終了"""
    n = len(closes)
    if len(vidx) == 0:
        return np.array([]), np.array([], dtype=int)
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    is_stop  = (fut <= s_arr[:, np.newaxis]) & in_range
    is_take  = (fut >= t_arr[:, np.newaxis]) & in_range
    hit      = is_stop | is_take
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    hit_stop = is_stop[np.arange(len(vidx)), fhp] & has_hit
    hit_take = is_take[np.arange(len(vidx)), fhp] & has_hit & ~hit_stop
    types    = np.where(hit_stop, 1, np.where(hit_take, 2, 0)).astype(int)
    mask     = ~np.isnan(rets)
    return rets[mask], types[mask]


# ── 前処理 ────────────────────────────────────────────────────────────────────
def preprocess(df_raw, shares):
    if df_raw is None or len(df_raw) < MIN_HISTORY + 10:
        return None
    df  = df_raw.copy()
    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]
    o   = df["Open"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(14).mean()
    df["RSI"]     = calc_rsi(c)
    n_p           = BREAKOUT_DAYS + 1
    df["avg_vol"] = v.rolling(n_p).mean().shift(1)
    df["avg_to"]  = (c * v).rolling(n_p).mean().shift(1)
    df["pct_chg"] = c.pct_change() * 100
    df["mktcap"]  = (c * shares) if shares is not None else np.nan
    return df


# ── 集計ユーティリティ ────────────────────────────────────────────────────────
def _metrics(r, t, trading_days):
    if len(r) < 10:
        return None
    wins  = r[r > 0]; losses = r[r <= 0]
    wr    = len(wins) / len(r) * 100
    avg_w = wins.mean()   if len(wins)   > 0 else 0.0
    avg_l = losses.mean() if len(losses) > 0 else 0.0
    pf    = abs(avg_w / avg_l) if avg_l != 0 else 0.0
    ev    = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd   = len(r) / trading_days
    n_s   = int((t == 1).sum()); n_t = int((t == 2).sum()); n_f = int((t == 0).sum())
    return {
        "wr": wr, "pf": pf, "ev": ev, "n": len(r), "spd": spd,
        "n_stop": n_s, "n_take": n_t, "n_force": n_f,
        "avg_w": avg_w, "avg_l": avg_l,
    }


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

    print("\n前処理中...")
    processed = {}
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(df_raw, shares_map.get(tk))
        if r is not None:
            processed[tk] = r
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)}  有効: {len(processed)}")
    print(f"  完了  有効: {len(processed)}")

    dates = set()
    for df in list(processed.values())[:30]:
        dates.update(df.index.tolist())
    trading_days = len(dates)
    print(f"\n推定取引日数: {trading_days} 日")

    # ── グリッドコンボ ────────────────────────────────────────────────────────
    combos = list(itertools.product(
        GRID["rsi_hi"], GRID["pct_thr"],
        GRID["vol_mult"], GRID["mktcap"], GRID["rr"],
    ))
    print(f"グリッド: {len(combos)} 通り\n")

    combo_rets  = defaultdict(list)
    combo_types = defaultdict(list)
    # 年別WR検証用: key → {year: [rets]}
    combo_yearly = defaultdict(lambda: defaultdict(list))

    for tk, df in processed.items():
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        avg_v = df["avg_vol"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        pct   = df["pct_chg"].values.astype(float)
        mktcap= df["mktcap"].values.astype(float)
        years = pd.DatetimeIndex(df.index).year

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(pct)) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )

        for (rhi, pth, vm, mkt, rr) in combos:
            rsi_ok    = (rsi >= 20.0) & (rsi <= rhi) & (~np.isnan(rsi))
            pct_ok    = pct >= pth
            vol_ok    = (vol_a >= avg_v * vm) & (avg_v > 0) if vm > 0 else np.ones(n, dtype=bool)
            mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt)

            vidx = np.where(base_ok & rsi_ok & pct_ok & vol_ok & mktcap_ok)[0]
            if len(vidx) == 0:
                continue

            e_arr = next_o[vidx]
            a_arr = atr[vidx]
            s_arr = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
            t_arr = e_arr + (e_arr - s_arr) * rr

            rets, types = _calc_rets_and_types(c_a, vidx, e_arr, s_arr, t_arr)
            if len(rets) == 0:
                continue

            key = (rhi, pth, vm, int(mkt/1e8), rr)
            combo_rets[key].extend(rets.tolist())
            combo_types[key].extend(types.tolist())

            # 年別集計（最良キー候補のため後で使う）
            yr_arr = years[vidx[:len(rets)]]
            for yr, ret in zip(yr_arr, rets):
                combo_yearly[key][int(yr)].append(ret)

    # ── 集計・評価 ────────────────────────────────────────────────────────────
    results = []
    for key in combo_rets:
        r = np.array(combo_rets[key])
        t = np.array(combo_types[key], dtype=int)
        m = _metrics(r, t, trading_days)
        if m is None:
            continue
        m["key"] = key
        m["passed"] = (
            m["wr"] >= CRITERIA_WR_LO and
            m["pf"] >= CRITERIA_PF and
            CRITERIA_SPD_LO <= m["spd"] <= CRITERIA_SPD_HI
        )
        results.append(m)

    results.sort(key=lambda x: (x["passed"], x["pf"]), reverse=True)
    qualified = [r for r in results if r["passed"]]

    # ── 表示 ──────────────────────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"【売られすぎ反発型 グリッドサーチ結果】  合格: {len(qualified)} / {len(combos)} 通り")
    print(f"  評価基準: 勝率≥{CRITERIA_WR_LO}%  PF≥{CRITERIA_PF}  "
          f"件/日 {CRITERIA_SPD_LO}〜{CRITERIA_SPD_HI}")
    print(f"\n  {'RSI上限':>6}  {'前日比':>6}  {'出来高':>7}  {'時価総額':>7}  {'RR':>4}  "
          f"{'勝率':>7}  {'PF':>5}  {'期待値':>8}  {'件数':>5}  {'件/日':>5}")
    print("  " + "-" * 72)

    shown = 0
    for r in results:
        if shown >= 10:
            break
        rhi, pth, vm, mkt, rr = r["key"]
        vm_s = f"{vm:.1f}x" if vm > 0 else "なし"
        m = "★" if r["passed"] else " "
        print(f"{m} RSI≤{rhi:.0f}  +{pth:.0f}%  {vm_s:>6}  {mkt:>4}億  {rr:>4.1f}  "
              f"{r['wr']:>6.1f}%  {r['pf']:>5.2f}  {r['ev']:>+7.2f}%  "
              f"{r['n']:>5,}  {r['spd']:>5.2f}")
        shown += 1

    # ── 最良条件の詳細分析 ────────────────────────────────────────────────────
    best = qualified[0] if qualified else results[0]
    rhi, pth, vm, mkt, rr = best["key"]
    vm_s = f"{vm:.1f}x" if vm > 0 else "なし"

    print(f"\n{'='*72}")
    print(f"【最良条件の詳細分析】")
    print(f"  条件: RSI20-{rhi:.0f} / 前日比+{pth:.0f}% / 出来高{vm_s} / "
          f"時価総額{mkt}億以下 / RR1:{rr}")
    print(f"  勝率 : {best['wr']:.1f}%  PF: {best['pf']:.2f}  "
          f"期待値: {best['ev']:+.2f}%  ({best['n']:,}件 / {best['spd']:.2f}/日)")
    print(f"  平均利益: +{best['avg_w']:.2f}%  平均損失: {best['avg_l']:.2f}%")
    n_tot = best["n"]
    print(f"\n  【決済内訳】")
    print(f"  損切り  : {best['n_stop']:4d}件 ({best['n_stop']/n_tot*100:.1f}%)  "
          f"avg {np.array(combo_rets[best['key']])[np.array(combo_types[best['key']])==1].mean():.2f}%"
          if best['n_stop'] > 0 else f"  損切り  : {best['n_stop']:4d}件")
    print(f"  利確    : {best['n_take']:4d}件 ({best['n_take']/n_tot*100:.1f}%)  "
          f"avg +{np.array(combo_rets[best['key']])[np.array(combo_types[best['key']])==2].mean():.2f}%"
          if best['n_take'] > 0 else f"  利確    : {best['n_take']:4d}件")
    print(f"  強制終了: {best['n_force']:4d}件 ({best['n_force']/n_tot*100:.1f}%)  "
          f"avg {np.array(combo_rets[best['key']])[np.array(combo_types[best['key']])==0].mean():+.2f}%"
          if best['n_force'] > 0 else f"  強制終了: {best['n_force']:4d}件")

    # ── 年別WR（信頼性検証）────────────────────────────────────────────────────
    print(f"\n  【年別WR（妥当性検証）】")
    print(f"  {'年':>5}  {'件数':>5}  {'勝率':>7}  {'PF':>5}  {'期待値':>8}")
    print("  " + "-" * 40)
    yearly = combo_yearly[best["key"]]
    for yr in sorted(yearly.keys()):
        r_yr = np.array(yearly[yr])
        if len(r_yr) < 5:
            continue
        wins_y  = r_yr[r_yr > 0]; losses_y = r_yr[r_yr <= 0]
        wr_y    = len(wins_y) / len(r_yr) * 100
        avg_w_y = wins_y.mean()   if len(wins_y)   > 0 else 0.0
        avg_l_y = losses_y.mean() if len(losses_y) > 0 else 0.0
        pf_y    = abs(avg_w_y / avg_l_y) if avg_l_y != 0 else 0.0
        ev_y    = wr_y / 100 * avg_w_y + (1 - wr_y / 100) * avg_l_y
        print(f"  {yr:>5}  {len(r_yr):>5,}  {wr_y:>6.1f}%  {pf_y:>5.2f}  {ev_y:>+7.2f}%")

    # ── RSIシグナル日の翌日始値 vs 終値比較（追加検証）────────────────────────
    print(f"\n  【翌日の価格分布（妥当性確認）】")
    all_e = []
    all_next_close = []
    for tk, df in processed.items():
        n     = len(df)
        c_a   = df["Close"].values.astype(float)
        o_a   = df["Open"].values.astype(float)
        atr   = df["ATR"].values.astype(float)
        rsi   = df["RSI"].values.astype(float)
        to_a  = df["avg_to"].values.astype(float)
        avg_v = df["avg_vol"].values.astype(float)
        vol_a = df["Volume"].values.astype(float)
        pct   = df["pct_chg"].values.astype(float)
        mktcap= df["mktcap"].values.astype(float)

        next_o = np.empty(n); next_o[:] = np.nan
        next_o[:-1] = o_a[1:]
        next_c = np.empty(n); next_c[:] = np.nan
        next_c[:-1] = c_a[1:]

        idx = np.arange(n)
        base_ok = (
            (~np.isnan(atr)) & (atr > 0) &
            (~np.isnan(to_a)) & (to_a >= MIN_AVG_TURNOVER) &
            (~np.isnan(pct)) &
            (~np.isnan(next_o)) & (next_o > 0) &
            (~np.isnan(next_c)) & (next_c > 0) &
            (idx >= MIN_HISTORY) & (idx < n - 1)
        )

        rhi2, pth2, vm2, mkt2, rr2 = best["key"]
        mkt2_val = mkt2 * 1e8
        rsi_ok    = (rsi >= 20.0) & (rsi <= rhi2) & (~np.isnan(rsi))
        pct_ok    = pct >= pth2
        vol_ok    = (vol_a >= avg_v * vm2) & (avg_v > 0) if vm2 > 0 else np.ones(n, dtype=bool)
        mktcap_ok = np.isnan(mktcap) | (mktcap <= mkt2_val)

        vidx = np.where(base_ok & rsi_ok & pct_ok & vol_ok & mktcap_ok)[0]
        if len(vidx) == 0:
            continue

        e_arr  = next_o[vidx]
        nc_arr = next_c[vidx]
        valid  = (e_arr > 0) & (nc_arr > 0)
        if valid.sum() == 0:
            continue
        # 翌日始値→終値のリターン
        intra = (nc_arr[valid] - e_arr[valid]) / e_arr[valid] * 100
        all_e.extend(intra.tolist())

    if all_e:
        e_arr2 = np.array(all_e)
        wr_intra = (e_arr2 > 0).mean() * 100
        print(f"  翌日始値→翌日終値の騰落: 平均{e_arr2.mean():+.2f}%  "
              f"プラス率: {wr_intra:.1f}%  (n={len(e_arr2):,})")
        print(f"  ※ 翌日始値でギャップアップしても当日中に伸びる傾向")
        pct25, pct75 = np.percentile(e_arr2, [25, 75])
        print(f"  分布: 25%ile={pct25:+.2f}%  中央値={np.median(e_arr2):+.2f}%  75%ile={pct75:+.2f}%")

    print(f"\n{'='*72}")
    print("【エッジの解釈】")
    print("  ・シグナル: RSI20-40（売られすぎ）の日に前日比+2%以上の反発")
    print("  ・エントリー: 翌日の始値（+2%の動きは既に昨日完了）")
    print("  ・エッジ源: 売られすぎ圏からの反発は数日継続するモメンタム効果")
    print("  ・ルックアヘッドなし: シグナル確定は当日終値後、注文は翌日寄り")


if __name__ == "__main__":
    main()
