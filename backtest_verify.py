#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファクトチェック用 詳細検証スクリプト
======================================
backtest_final.py で見つかった2条件を単体で詳細検証する。

【精度型】RSI14≤40 + 前日比+5% + ATR拡大 + cd1日  → WR80%超 / 1.32件/日
【件数型】RSI14≤30 + 出来高1.5倍 + ATR拡大         → WR55%  / 4.41件/日

確認項目:
  ① サンプル数（十分か？）
  ② 年別WR（時系列で安定しているか？）
  ③ 月別WR（特定の季節に偏っていないか？）
  ④ リターン分布（外れ値依存していないか？）
  ⑤ 中央値ベースの指標（平均が歪んでいないか？）
"""

import pickle, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_PATH   = Path(__file__).parent / "backtest_cache_long.pkl"
MIN_HISTORY  = 100
MAX_HOLD     = 20
STOP_COEF    = 2.0
STOP_CAP     = 0.10
MIN_TURNOVER = 30_000_000
MAX_WORKERS  = 20

# ── 検証対象2条件 ─────────────────────────────────────────────────────────────
CONDITIONS = {
    "精度型": {
        "desc": "RSI14≤40 + 前日比+5% + ATR拡大 + cd1日（前日陰線後反発）",
        "rsi_hi":     40.0,
        "pct_thr":    5.0,     # 前日比 ≥ +5%
        "vol_mult":   0.0,     # 出来高制限なし
        "atr_expand": True,    # ATR拡大あり
        "cons_down":  1,       # 1日連続下落後
        "ma25_dev":   None,    # MA25制限なし
        "mktcap_max": None,    # 時価総額制限なし
        "rr":         1.5,
    },
    "件数型": {
        "desc": "RSI14≤30 + 出来高1.5倍 + ATR拡大（MA75下）",
        "rsi_hi":     30.0,
        "pct_thr":    None,    # 前日比制限なし
        "vol_mult":   1.5,     # 出来高 ≥ 1.5倍
        "atr_expand": True,
        "cons_down":  0,
        "ma25_dev":   None,
        "mktcap_max": None,
        "rr":         2.5,
    },
    "旧最強": {
        "desc": "RSI14≤35 + 前日比+5% + MA25-10% + ATR拡大（vol制限なし）",
        "rsi_hi":     35.0,
        "pct_thr":    5.0,
        "vol_mult":   0.0,
        "atr_expand": True,
        "cons_down":  0,
        "ma25_dev":   -10.0,
        "mktcap_max": None,
        "rr":         2.5,
    },
}


# ── データ読込 ─────────────────────────────────────────────────────────────────

def load_cache():
    if not CACHE_PATH.exists():
        raise FileNotFoundError("backtest_cache.pkl が見つかりません")
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    print(f"キャッシュ利用（{cached['date']}）: {len(cached['data'])} 銘柄")
    return cached["data"]


def fetch_all_shares(tickers):
    def _get(t):
        try:
            sh = yf.Ticker(t).fast_info.shares
            return t, float(sh) if sh else None
        except Exception:
            return t, None
    result = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_get, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            t, sh = fut.result()
            result[t] = sh
            done += 1
            if done % 500 == 0 or done == len(tickers):
                print(f"  {done}/{len(tickers)}")
    return result


# ── RSI計算 ────────────────────────────────────────────────────────────────────

def _calc_rsi14(prices):
    period = 14
    n = len(prices)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    diff   = np.diff(prices)
    gains  = np.where(diff > 0, diff, 0.)
    losses = np.where(diff < 0, -diff, 0.)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    rsi[period] = 100. if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = avg_g * 13/14 + gains[i]/14
        avg_l = avg_l * 13/14 + losses[i]/14
        rsi[i+1] = 100. if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


# ── 銘柄前処理 ─────────────────────────────────────────────────────────────────

def preprocess(df_raw, shares):
    df = df_raw.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.iloc[(df["Close"].to_numpy() > 0)].dropna(subset=["Open","High","Low","Close","Volume"])
    if len(df) < MIN_HISTORY:
        return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    dates = df.index
    n = len(c)

    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr  = np.maximum.reduce([h-l, np.abs(h-prev_c), np.abs(l-prev_c)])
    atr = pd.Series(tr).rolling(14).mean().values

    atr_s      = pd.Series(atr)
    atr3d      = atr_s.rolling(3).mean().values
    atr3d_prev = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (
        (atr3d > atr3d_prev)
        & ~np.isnan(atr3d) & ~np.isnan(atr3d_prev) & (atr3d_prev > 0)
    )

    rsi14 = _calc_rsi14(c)

    pct = np.full(n, np.nan)
    pct[1:] = (c[1:] - c[:-1]) / c[:-1] * 100

    ma25 = pd.Series(c).rolling(25).mean().values
    ma25_dev = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)

    avg_v  = pd.Series(v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    avg_to = pd.Series(c * v).rolling(20).mean().values
    mktcap = c * shares if shares else np.full(n, np.nan)

    down = (pct < 0) & ~np.isnan(pct)
    up   = (pct > 0) & ~np.isnan(pct)
    prev_down1 = np.roll(down, 1); prev_down1[0] = False
    prev_down2 = np.roll(down, 2); prev_down2[:2] = False
    cd1 = up & prev_down1
    cd2 = up & prev_down1 & prev_down2

    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi14)) & (~np.isnan(pct)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    return {
        "close": c, "next_o": next_o, "atr": atr, "dates": dates,
        "rsi14": rsi14, "pct": pct, "vol_r": vol_r,
        "ma25_dev": ma25_dev, "mktcap": mktcap,
        "atr_exp": atr_exp, "cd1": cd1, "cd2": cd2,
        "base": base, "n": n,
    }


# ── マスク取得 ─────────────────────────────────────────────────────────────────

def get_mask(td, cond):
    m = td["base"] & (td["rsi14"] <= cond["rsi_hi"]) & ~np.isnan(td["rsi14"])
    if cond["pct_thr"] is not None:
        m = m & (td["pct"] >= cond["pct_thr"]) & ~np.isnan(td["pct"])
    if cond["vol_mult"] > 0:
        m = m & (td["vol_r"] >= cond["vol_mult"]) & ~np.isnan(td["vol_r"])
    if cond["atr_expand"]:
        m = m & td["atr_exp"]
    if cond["cons_down"] == 1:
        m = m & td["cd1"]
    elif cond["cons_down"] == 2:
        m = m & td["cd2"]
    if cond["ma25_dev"] is not None:
        m = m & (td["ma25_dev"] <= cond["ma25_dev"]) & ~np.isnan(td["ma25_dev"])
    if cond["mktcap_max"] is not None:
        m = m & (np.isnan(td["mktcap"]) | (td["mktcap"] <= cond["mktcap_max"]))
    return m


# ── リターン計算 ───────────────────────────────────────────────────────────────

def calc_trade(td, idx, rr):
    """1トレードのリターンと日付を返す"""
    c    = td["close"]
    ep   = td["next_o"][idx]
    a    = td["atr"][idx]
    stop = max(ep - a * STOP_COEF, ep * (1 - STOP_CAP))
    take = ep + (ep - stop) * rr
    if not (ep > 0 and stop > 0 and take > 0 and stop < ep < take):
        return None, None

    entry_date = td["dates"][idx] if idx < len(td["dates"]) else None

    for j in range(idx + 1, min(idx + MAX_HOLD + 1, td["n"])):
        lo = c[j]; hi = c[j]
        # 終値ベースで判定（日足では高値・安値のみでなく終値使用）
        if lo <= stop:
            ret = (stop - ep) / ep * 100
            return ret, entry_date
        if hi >= take:
            ret = (take - ep) / ep * 100
            return ret, entry_date

    # タイムアウト
    last_idx = min(idx + MAX_HOLD, td["n"] - 1)
    ret = (c[last_idx] - ep) / ep * 100
    return ret, entry_date


def run_condition(all_proc, cond):
    """全銘柄に対して条件を適用し、全トレードのリターンと日付を収集"""
    all_rets  = []
    all_dates = []

    for td in all_proc:
        m    = get_mask(td, cond)
        idxs = np.where(m)[0]
        for idx in idxs:
            ret, dt = calc_trade(td, idx, cond["rr"])
            if ret is not None and dt is not None:
                all_rets.append(ret)
                all_dates.append(dt)

    return np.array(all_rets), pd.DatetimeIndex(all_dates)


# ── 詳細統計表示 ───────────────────────────────────────────────────────────────

def analyze(name, desc, rets, dates, trading_days):
    print(f"\n{'='*70}")
    print(f"【{name}】 {desc}")
    print(f"{'='*70}")

    n = len(rets)
    if n < 5:
        print(f"  サンプル数が少なすぎます: {n}件")
        return

    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / n * 100
    avg_w  = wins.mean()   if len(wins)   > 0 else 0.
    avg_l  = losses.mean() if len(losses) > 0 else 0.
    pf     = abs(avg_w / avg_l) if avg_l != 0 else 0.
    ev     = wr/100 * avg_w + (1 - wr/100) * avg_l
    spd    = n / trading_days
    med_ret = np.median(rets)

    print(f"\n  ■ 基本統計")
    print(f"    サンプル数   : {n:,}件")
    print(f"    件数/日      : {spd:.2f}件/日  ({trading_days}取引日)")
    print(f"    勝率         : {wr:.1f}%  ({len(wins)}勝 / {len(losses)}敗)")
    print(f"    PF           : {pf:.2f}")
    print(f"    期待値(平均) : {ev:+.2f}%/トレード")
    print(f"    期待値(中央) : {med_ret:+.2f}%/トレード  ← 外れ値の影響を除外")
    print(f"    平均利益     : {avg_w:+.2f}%")
    print(f"    平均損失     : {avg_l:+.2f}%")
    print(f"    最大利益     : {rets.max():+.2f}%")
    print(f"    最大損失     : {rets.min():+.2f}%")
    print(f"    標準偏差     : {rets.std():.2f}%")

    # ── 外れ値の影響確認 ──
    p95 = np.percentile(rets, 95)
    p05 = np.percentile(rets, 5)
    trimmed = rets[(rets >= p05) & (rets <= p95)]
    trim_wr = (trimmed > 0).mean() * 100
    trim_ev = trimmed.mean()
    print(f"\n  ■ 外れ値除去後（上下5%カット）")
    print(f"    勝率         : {trim_wr:.1f}%")
    print(f"    期待値       : {trim_ev:+.2f}%")
    print(f"    サンプル     : {len(trimmed):,}件")

    # ── 年別WR ──
    print(f"\n  ■ 年別WR（時系列安定性）")
    years = sorted(set(d.year for d in dates))
    year_ok = True
    for yr in years:
        mask  = np.array([d.year == yr for d in dates])
        yr_r  = rets[mask]
        if len(yr_r) < 3:
            continue
        yr_wr  = (yr_r > 0).mean() * 100
        yr_n   = len(yr_r)
        yr_ev  = yr_r.mean()
        marker = " ◀ 低WR年" if yr_wr < 45 else ""
        if yr_wr < 45:
            year_ok = False
        print(f"    {yr}年: WR {yr_wr:5.1f}%  {yr_n:4d}件  EV{yr_ev:+5.2f}%{marker}")

    if year_ok:
        print(f"    → 全年でWR45%以上を維持（時系列安定）")
    else:
        print(f"    → ⚠ 一部の年でWR45%を下回っている")

    # ── 月別WR ──
    print(f"\n  ■ 月別WR（季節性）")
    month_wrs = []
    for mo in range(1, 13):
        mask = np.array([d.month == mo for d in dates])
        mo_r = rets[mask]
        if len(mo_r) < 3:
            month_wrs.append(None)
            continue
        mo_wr = (mo_r > 0).mean() * 100
        month_wrs.append(mo_wr)
    months = "  ".join(
        f"{m:>2}月:{w:.0f}%" if w is not None else f"{m:>2}月: --"
        for m, w in enumerate(month_wrs, 1)
    )
    print(f"    {months}")

    # ── リターン分布 ──
    print(f"\n  ■ リターン分布")
    buckets = [(-100,-20),(-20,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,100)]
    for lo, hi in buckets:
        cnt = ((rets >= lo) & (rets < hi)).sum()
        pct = cnt / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {lo:>5}〜{hi:>4}%: {cnt:4d}件 ({pct:4.1f}%) {bar}")

    # ── 総合判定 ──
    print(f"\n  ■ ファクトチェック判定")
    issues = []
    if n < 50:
        issues.append(f"サンプル数が少ない ({n}件 < 50件)")
    if abs(med_ret - ev) > 3:
        issues.append(f"平均({ev:+.2f}%)と中央値({med_ret:+.2f}%)の乖離が大きい → 外れ値依存の疑い")
    if trim_wr < wr - 5:
        issues.append(f"外れ値除去後にWRが{wr:.1f}%→{trim_wr:.1f}%に低下")
    if not year_ok:
        issues.append("特定年でWR45%未満の期間あり → 時系列非安定")
    if rets.max() > 50 and rets.max() > avg_w * 5:
        issues.append(f"最大利益({rets.max():.1f}%)が平均の5倍以上 → 外れ値に依存")

    if issues:
        print(f"    ⚠ 懸念事項:")
        for issue in issues:
            print(f"      - {issue}")
        print(f"    → 実績数値は参考値として扱うべき")
    else:
        print(f"    ✓ 懸念事項なし。バックテスト結果は信頼できる水準")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ファクトチェック 詳細検証")
    print("=" * 70)

    raw_data = load_cache()

    print(f"\n株数取得中...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中...")
    all_proc = []
    for i, (ticker, df_raw) in enumerate(raw_data.items(), 1):
        td = preprocess(df_raw, shares_map.get(ticker))
        if td is not None:
            all_proc.append(td)
    print(f"  完了: {len(all_proc)} 銘柄")

    trading_days = int(np.mean([td["n"] for td in all_proc]))
    print(f"  推定取引日数: {trading_days} 日")

    for name, cond in CONDITIONS.items():
        print(f"\n{name} を計算中...")
        rets, dates = run_condition(all_proc, cond)
        analyze(name, cond["desc"], rets, dates, trading_days)

    print(f"\n{'='*70}")
    print("検証完了")


if __name__ == "__main__":
    main()
