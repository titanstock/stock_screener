#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新戦略候補 グリッドサーチ
============================
6候補からバックテスト上位2を選定する。

候補:
  A. 順張りモメンタム型    : 出来高急増 + 前日比上昇 + RSI中間 + MACD上
  B. 押し目買い型          : MA25タッチ + RSI中間 + MACD上
  C. 出来高枯渇反発型      : 出来高枯渇後の急増 + RSI売られすぎ圏
  D. 連続陰線後下ヒゲ反発型: N日連続陰線 + 下ヒゲ大 + 出来高増
  E. 連続安値更新後急反発型: N日連続安値更新 + 急騰 + 出来高増
  F. 値幅縮小後ブレイク型  : レンジ縮小 + 終値ブレイクアウト + RSI中間
"""

import itertools, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_PATH   = Path(__file__).parent / "backtest_cache_long.pkl"
MIN_HISTORY  = 100
MAX_HOLD     = 20
STOP_COEF    = 2.0
STOP_CAP     = 0.10
MIN_TURNOVER = 30_000_000
MAX_WORKERS  = 8

# 合格基準
CRITERIA_WR  = 50.0
CRITERIA_PF  = 1.4
CRITERIA_SPD = 0.3   # 件/日

# ── グリッド ────────────────────────────────────────────────────────────────
GRID_A = {  # 順張りモメンタム型
    "vol_mult":   [1.5, 2.0, 2.5, 3.0],
    "change_lo":  [1.0, 2.0, 3.0],
    "rsi_lo":     [45.0, 50.0, 55.0],
    "rsi_hi":     [65.0, 70.0, 75.0],
    "weekly_dev": [15, 20, 25],
    "rr":         [1.5, 2.0],
}

GRID_B = {  # 押し目買い型
    "touch_pct":  [1.01, 1.02, 1.03, 1.04],
    "rsi_lo":     [45.0, 50.0, 55.0],
    "rsi_hi":     [60.0, 65.0, 70.0],
    "rr":         [1.5, 2.0],
}

GRID_C = {  # 出来高枯渇反発型
    "dry_days":    [3, 5],
    "vol_spike":   [1.5, 2.0, 2.5],
    "rsi_lo":      [20.0, 25.0, 30.0],
    "rsi_hi":      [50.0, 55.0],
    "ma25_dev_hi": [0.0, 5.0, 10.0],
    "rr":          [1.5, 2.0, 2.5],
}

GRID_D = {  # 連続陰線後下ヒゲ反発型
    "consec_bear": [3, 4, 5],
    "shadow_pct":  [30.0, 40.0, 50.0],
    "vol_mult":    [1.5, 2.0],
    "rsi_hi":      [45.0, 50.0, 55.0],
    "rr":          [1.5, 2.0, 2.5],
}  # 3*3*2*3*3 = 162通り

GRID_E = {  # 連続安値更新後急反発型
    "consec_low":  [3, 5],
    "change_lo":   [1.0, 2.0, 3.0],
    "vol_mult":    [1.5, 2.0, 2.5],
    "rsi_hi":      [40.0, 45.0, 50.0],
    "rr":          [1.5, 2.0, 2.5],
}  # 2*3*3*3*3 = 162通り

GRID_F = {  # 値幅縮小後ブレイク型
    "range_days":   [5, 7],
    "contract_pct": [0.7, 0.8, 0.9],
    "rsi_lo":       [45.0, 50.0, 55.0],
    "rsi_hi":       [65.0, 70.0, 75.0],
    "rr":           [1.5, 2.0],
}  # 2*3*3*3*2 = 108通り


# ── RSI計算 ─────────────────────────────────────────────────────────────────
def _calc_rsi14(prices):
    period = 14
    n = len(prices)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    diff  = np.diff(prices)
    gains = np.where(diff > 0, diff, 0.)
    losses= np.where(diff < 0, -diff, 0.)
    ag = gains[:period].mean()
    al = losses[:period].mean()
    rsi[period] = 100. if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(period, n - 1):
        ag = ag * 13/14 + gains[i] / 14
        al = al * 13/14 + losses[i] / 14
        rsi[i+1] = 100. if al == 0 else 100 - 100 / (1 + ag / al)
    return rsi


# ── 銘柄前処理 ───────────────────────────────────────────────────────────────
def preprocess(df_raw):
    df = df_raw.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.iloc[(df["Close"].to_numpy() > 0)].dropna(
        subset=["Open","High","Low","Close","Volume"])
    if len(df) < MIN_HISTORY + 5:
        return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)

    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr  = np.maximum.reduce([h-l, np.abs(h-prev_c), np.abs(l-prev_c)])
    atr = pd.Series(tr).rolling(14).mean().values

    rsi14    = _calc_rsi14(c)
    pct      = np.full(n, np.nan); pct[1:] = (c[1:]-c[:-1])/c[:-1]*100
    ma25     = pd.Series(c).rolling(25).mean().values
    ma25_dev = np.where(ma25>0, (c-ma25)/ma25*100, np.nan)

    avg_v  = pd.Series(v).rolling(21).mean().shift(1).values
    vol_r  = np.where(avg_v>0, v/avg_v, np.nan)
    avg_to = pd.Series(c*v).rolling(21).mean().shift(1).values

    # MA25タッチ（当日or前日）
    above_daily = c > ma25
    ma25_touch  = {}
    for tp_int in [101, 102, 103, 104]:
        tp = tp_int / 100.0
        today = (~np.isnan(ma25)) & (c >= ma25) & (c <= ma25 * tp)
        prev  = np.roll(today, 1); prev[0] = False
        ma25_touch[tp_int] = today | prev

    # MACD上向き
    e12  = pd.Series(c).ewm(span=12).mean().values
    e26  = pd.Series(c).ewm(span=26).mean().values
    macd = e12 - e26
    sig  = pd.Series(macd).ewm(span=9).mean().values
    macd_up = macd > sig

    # 出来高枯渇フラグ（直近N日の出来高が平均以下）
    dry = {}
    for nd in [3, 5]:
        max_v_in_window = np.array([
            v[max(0,i-nd):i].max() if i >= nd else np.nan
            for i in range(n)
        ])
        dry[nd] = (~np.isnan(max_v_in_window)) & (max_v_in_window < avg_v)

    # ── D用: 連続陰線カウント ──
    consec_bear = np.zeros(n, dtype=int)
    for i in range(1, n):
        consec_bear[i] = consec_bear[i-1] + 1 if c[i] < o[i] else 0

    # 下ヒゲ比率（下ヒゲ / レンジ * 100）
    range_ = h - l
    lower_shadow = np.minimum(o, c) - l
    lower_shadow_ratio = np.where(range_ > 0, lower_shadow / range_ * 100, 0.0)

    # ── E用: 連続安値更新カウント ──
    consec_low = np.zeros(n, dtype=int)
    for i in range(1, n):
        consec_low[i] = consec_low[i-1] + 1 if c[i] < c[i-1] else 0

    # ── F用: レンジ縮小 + ブレイクアウト ──
    range_ratio = {}
    prev_high   = {}
    for rd in [5, 7]:
        recent_range = np.array([
            (h[max(0,i-rd+1):i+1].max() - l[max(0,i-rd+1):i+1].min())
            if i >= rd else np.nan
            for i in range(n)
        ])
        prior_range = np.array([
            (h[max(0,i-2*rd+1):i-rd+1].max() - l[max(0,i-2*rd+1):i-rd+1].min())
            if i >= 2*rd else np.nan
            for i in range(n)
        ])
        range_ratio[rd] = np.where(prior_range > 0, recent_range / prior_range, np.nan)
        prev_high[rd]   = np.array([
            h[max(0,i-rd):i].max() if i >= rd else np.nan
            for i in range(n)
        ])

    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi14)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    return {
        "c": c, "o": o, "h": h, "l": l,
        "next_o": next_o, "atr": atr, "n": n,
        "rsi14": rsi14, "pct": pct, "vol_r": vol_r,
        "ma25_dev": ma25_dev, "above_daily": above_daily,
        "ma25_touch": ma25_touch, "macd_up": macd_up,
        "dry": dry, "base": base,
        "consec_bear": consec_bear, "lower_shadow_ratio": lower_shadow_ratio,
        "consec_low": consec_low,
        "range_ratio": range_ratio, "prev_high": prev_high,
    }


# ── 1トレードのリターン計算 ──────────────────────────────────────────────────
def _calc_returns(td, mask, rr):
    c, next_o, atr, n = td["c"], td["next_o"], td["atr"], td["n"]
    idxs = np.where(mask)[0]
    rets = []
    for i in idxs:
        ep   = next_o[i]
        a    = atr[i]
        stop = max(ep - a * STOP_COEF, ep * (1 - STOP_CAP))
        take = ep + (ep - stop) * rr
        if not (ep > 0 and stop > 0 and stop < ep < take):
            continue
        for j in range(i+1, min(i+MAX_HOLD+1, n)):
            if c[j] <= stop:
                rets.append((stop - ep) / ep * 100)
                break
            if c[j] >= take:
                rets.append((take - ep) / ep * 100)
                break
        else:
            rets.append((c[min(i+MAX_HOLD, n-1)] - ep) / ep * 100)
    return np.array(rets)


# ── メトリクス計算 ───────────────────────────────────────────────────────────
def _metrics(rets, trading_days):
    n = len(rets)
    if n < 5:
        return {"n":0,"wr":0.,"pf":0.,"spd":0.,"ev":0.,"score":0.}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr  = len(wins) / n * 100
    aw  = wins.mean()   if len(wins)   > 0 else 0.
    al  = losses.mean() if len(losses) > 0 else 0.
    pf  = abs(aw / al)  if al != 0 else 0.
    ev  = wr/100*aw + (1-wr/100)*al
    spd = n / trading_days
    score = wr/100 * pf * np.sqrt(spd)
    return {"n":n, "wr":round(wr,1), "pf":round(pf,2),
            "spd":round(spd,2), "ev":round(ev,2), "score":round(score,3)}


# ── グリッドサーチ（各戦略） ─────────────────────────────────────────────────
def run_grid_A(all_proc, trading_days):
    """順張りモメンタム型"""
    keys   = list(GRID_A.keys())
    combos = list(itertools.product(*GRID_A.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        rets = []
        for td in all_proc:
            m = (
                td["base"] &
                td["above_daily"] &
                (~np.isnan(td["vol_r"])) & (td["vol_r"] >= p["vol_mult"]) &
                (~np.isnan(td["pct"]))   & (td["pct"]   >= p["change_lo"]) &
                (~np.isnan(td["rsi14"])) &
                (td["rsi14"] >= p["rsi_lo"]) & (td["rsi14"] <= p["rsi_hi"]) &
                td["macd_up"]
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def run_grid_B(all_proc, trading_days):
    """押し目買い型"""
    keys   = list(GRID_B.keys())
    combos = list(itertools.product(*GRID_B.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        tp_int = int(p["touch_pct"] * 100)
        rets = []
        for td in all_proc:
            if tp_int not in td["ma25_touch"]:
                continue
            m = (
                td["base"] &
                td["ma25_touch"][tp_int] &
                (~np.isnan(td["rsi14"])) &
                (td["rsi14"] >= p["rsi_lo"]) & (td["rsi14"] <= p["rsi_hi"]) &
                td["macd_up"]
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def run_grid_C(all_proc, trading_days):
    """出来高枯渇反発型"""
    keys   = list(GRID_C.keys())
    combos = list(itertools.product(*GRID_C.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        rets = []
        for td in all_proc:
            if p["dry_days"] not in td["dry"]:
                continue
            m = (
                td["base"] &
                td["dry"][p["dry_days"]] &
                (~np.isnan(td["vol_r"])) & (td["vol_r"] >= p["vol_spike"]) &
                (~np.isnan(td["rsi14"])) &
                (td["rsi14"] >= p["rsi_lo"]) & (td["rsi14"] <= p["rsi_hi"]) &
                (~np.isnan(td["ma25_dev"])) & (td["ma25_dev"] <= p["ma25_dev_hi"])
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def run_grid_D(all_proc, trading_days):
    """連続陰線後下ヒゲ反発型"""
    keys   = list(GRID_D.keys())
    combos = list(itertools.product(*GRID_D.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        rets = []
        for td in all_proc:
            m = (
                td["base"] &
                (td["consec_bear"] >= p["consec_bear"]) &
                (td["lower_shadow_ratio"] >= p["shadow_pct"]) &
                (~np.isnan(td["vol_r"])) & (td["vol_r"] >= p["vol_mult"]) &
                (~np.isnan(td["rsi14"])) & (td["rsi14"] <= p["rsi_hi"])
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def run_grid_E(all_proc, trading_days):
    """連続安値更新後急反発型"""
    keys   = list(GRID_E.keys())
    combos = list(itertools.product(*GRID_E.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        rets = []
        for td in all_proc:
            m = (
                td["base"] &
                (td["consec_low"] >= p["consec_low"]) &
                (~np.isnan(td["pct"])) & (td["pct"] >= p["change_lo"]) &
                (~np.isnan(td["vol_r"])) & (td["vol_r"] >= p["vol_mult"]) &
                (~np.isnan(td["rsi14"])) & (td["rsi14"] <= p["rsi_hi"])
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def run_grid_F(all_proc, trading_days):
    """値幅縮小後ブレイク型"""
    keys   = list(GRID_F.keys())
    combos = list(itertools.product(*GRID_F.values()))
    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        rd = p["range_days"]
        rets = []
        for td in all_proc:
            rr_arr = td["range_ratio"].get(rd)
            ph_arr = td["prev_high"].get(rd)
            if rr_arr is None or ph_arr is None:
                continue
            m = (
                td["base"] &
                (~np.isnan(rr_arr)) & (rr_arr <= p["contract_pct"]) &
                (~np.isnan(ph_arr)) & (td["c"] > ph_arr) &
                (~np.isnan(td["rsi14"])) &
                (td["rsi14"] >= p["rsi_lo"]) & (td["rsi14"] <= p["rsi_hi"])
            )
            r = _calc_returns(td, m, p["rr"])
            if len(r): rets.append(r)
        combined = np.concatenate(rets) if rets else np.array([])
        results.append({**p, **_metrics(combined, trading_days)})
    return results


def _print_top(name, results, top_n=5):
    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]
    src   = qualified if qualified else results
    label = "合格" if qualified else "合格なし・スコア順"
    top   = sorted(src, key=lambda r: r["score"], reverse=True)[:top_n]
    print(f"\n{'='*65}")
    print(f"【{name}】 合格: {len(qualified)}/{len(results)}通り  ({label})")
    print(f"{'='*65}")
    for r in top:
        mark = "★" if (r["wr"]>=CRITERIA_WR and r["pf"]>=CRITERIA_PF
                       and r["spd"]>=CRITERIA_SPD) else "  "
        params = {k:v for k,v in r.items()
                  if k not in ("n","wr","pf","spd","ev","score")}
        print(f"  {mark} WR{r['wr']:5.1f}%  PF{r['pf']:.2f}  {r['spd']:.2f}/日"
              f"  EV{r['ev']:+.2f}%  score{r['score']:.3f}")
        print(f"       {params}")
    return qualified if qualified else top[:1]


def main():
    print("="*65)
    print("新戦略候補 グリッドサーチ（5年データ）")
    print("="*65)

    if not CACHE_PATH.exists():
        print("backtest_cache_long.pkl が見つかりません")
        return

    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    raw_data = cached["data"]
    print(f"キャッシュ: {cached['date']} / {len(raw_data)} 銘柄")

    print("\n前処理中...")
    all_proc = []
    for i, (ticker, df_raw) in enumerate(raw_data.items(), 1):
        td = preprocess(df_raw)
        if td is not None:
            all_proc.append(td)
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)}")
    print(f"  完了: {len(all_proc)} 銘柄")

    trading_days = int(np.median([td["n"] for td in all_proc]))
    print(f"  取引日数（中央値）: {trading_days} 日")

    strategies = [
        ("A", "順張りモメンタム型",    GRID_A, run_grid_A),
        ("B", "押し目買い型",          GRID_B, run_grid_B),
        ("C", "出来高枯渇反発型",      GRID_C, run_grid_C),
        ("D", "連続陰線後下ヒゲ反発型",GRID_D, run_grid_D),
        ("E", "連続安値更新後急反発型",GRID_E, run_grid_E),
        ("F", "値幅縮小後ブレイク型",  GRID_F, run_grid_F),
    ]

    all_results = {}
    for key, name, grid, func in strategies:
        n_combos = len(list(itertools.product(*grid.values())))
        print(f"\n{key}: {name} {n_combos}通り...")
        res = func(all_proc, trading_days)
        _print_top(name, res)
        all_results[name] = res

    # ── 総合ランキング ──
    print(f"\n{'='*65}")
    print("【総合ランキング（スコア順）】")
    print(f"{'='*65}")
    all_best = []
    for _, name, _, _ in strategies:
        results = all_results[name]
        qualified = [r for r in results
                     if r["wr"]>=CRITERIA_WR and r["pf"]>=CRITERIA_PF
                     and r["spd"]>=CRITERIA_SPD]
        if qualified:
            best = max(qualified, key=lambda r: r["score"])
            all_best.append((name, best))

    all_best.sort(key=lambda x: x[1]["score"], reverse=True)
    for rank, (name, r) in enumerate(all_best, 1):
        params = {k:v for k,v in r.items()
                  if k not in ("n","wr","pf","spd","ev","score")}
        print(f"\n  {rank}位 【{name}】")
        print(f"      WR{r['wr']:.1f}%  PF{r['pf']:.2f}  {r['spd']:.2f}/日"
              f"  EV{r['ev']:+.2f}%  score{r['score']:.3f}")
        print(f"      {params}")

    if len(all_best) >= 2:
        print(f"\n→ 採用推奨: {all_best[0][0]} / {all_best[1][0]}")
    elif len(all_best) == 1:
        print(f"\n→ 合格1件のみ: {all_best[0][0]}")
    else:
        print("\n→ 合格なし")

    print(f"\n{'='*65}")
    print("完了")


if __name__ == "__main__":
    main()
