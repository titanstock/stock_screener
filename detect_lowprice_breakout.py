#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低位株ブレイクアウト 検知スクリプト
================================================================
売買ロジック・リターン集計なし。条件ヒット銘柄の一覧と検知統計のみ。

【検知条件】（仕様値のまま）
  1. 株価 ≤ 500円
  2. MA200 右肩下がり(過去20日比) + 株価 < MA200
  3. 直近30日レンジ (高値-安値)/終値 ≤ 15%
  4. 直近20日平均出来高 < 前30日平均 × 50%（出来高枯渇）
  5. ブレイクトリガー:
       - 出来高 ≥ 直近20日平均 × 5倍
       - 当日騰落率 ≥ +10%
       - 終値または高値が直近30日レンジ高値を上抜け
  6. 参考: 売買代金倍率 / 値幅制限節目突破フラグ（100/300/500円）

使い方:
  python detect_lowprice_breakout.py [キャッシュ名] [--relax]
  例: python detect_lowprice_breakout.py backtest_cache.pkl
      python detect_lowprice_breakout.py backtest_cache.pkl --relax
"""

import pickle, sys, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_args       = sys.argv[1:]
_relax      = "--relax" in _args
_cache_args = [a for a in _args if not a.startswith("--")]
_cache_name = _cache_args[0] if _cache_args else "backtest_cache.pkl"
CACHE_PATH  = Path(__file__).parent / _cache_name

# ── 仕様値パラメータ ─────────────────────────────────────────
PRICE_MAX        = 500.0
MA200_SLOPE_DAYS = 20
CONSOL_DAYS      = 30
CONSOL_RANGE_PCT = 0.15    # 15%
VOL_DRY_RECENT   = 20
VOL_DRY_RATIO    = 0.50    # 50%
BREAK_VOL_MULT   = 5.0     # 5倍
BREAK_RETURN_PCT = 10.0    # +10%

# 緩和パラメータ（--relax 時に使用 / 比較出力用）
RELAX = dict(
    range_pct = 0.25,
    dry_ratio = 0.60,
    vol_mult  = 3.0,
    ret_pct   = 5.0,
)

PRICE_BANDS   = [100.0, 300.0, 500.0]
BAND_LOOKBACK = 5
COOLDOWN_BARS = 20
MIN_HISTORY   = 240  # MA200(200) + CONSOL_DAYS(30) + バッファ


# ── 検知ロジック ─────────────────────────────────────────────

def _detect_one(ticker: str, df: pd.DataFrame,
                range_pct: float, dry_ratio: float,
                vol_mult: float,  ret_pct: float) -> list[dict]:
    """1銘柄の全検知日を返す。"""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)
    if n < MIN_HISTORY:
        return []

    ma200  = pd.Series(c).rolling(200).mean().values
    dates  = df.index
    hits   = []
    last_i = -COOLDOWN_BARS - 1

    for i in range(MIN_HISTORY, n):
        if i - last_i <= COOLDOWN_BARS:
            continue
        ci = c[i]

        # 1) 低位株
        if ci > PRICE_MAX:
            continue

        # 2) MA200 右肩下がり + 株価 < MA200
        m200 = ma200[i]
        if np.isnan(m200) or i < MA200_SLOPE_DAYS:
            continue
        if np.isnan(ma200[i - MA200_SLOPE_DAYS]):
            continue
        if m200 >= ma200[i - MA200_SLOPE_DAYS] or ci >= m200:
            continue

        # 3) 横ばい（直近 CONSOL_DAYS 日、当日除く）
        ws = i - CONSOL_DAYS
        if ws < 0 or len(h[ws:i]) < CONSOL_DAYS:
            continue
        wh = h[ws:i]
        wl = lo[ws:i]
        range_ratio = (wh.max() - wl.min()) / ci
        if range_ratio > range_pct:
            continue

        # 4) 出来高枯渇
        v_rec = v[i - VOL_DRY_RECENT:i]
        v_pri = v[ws:i - VOL_DRY_RECENT]
        if len(v_pri) < 10:
            continue
        avg_rec = v_rec.mean()
        avg_pri = v_pri.mean()
        if avg_pri <= 0 or avg_rec >= avg_pri * dry_ratio:
            continue

        # 5) ブレイクトリガー
        avg_v20 = v[i - 20:i].mean() if i >= 20 else 0.0
        if avg_v20 <= 0 or v[i] / avg_v20 < vol_mult:
            continue
        if c[i - 1] <= 0:
            continue
        chg = (ci - c[i - 1]) / c[i - 1] * 100
        if chg < ret_pct:
            continue
        if ci <= wh.max() and h[i] <= wh.max():
            continue

        # 売買代金倍率
        to_now   = ci * v[i]
        avg_to20 = float((c[i - 20:i] * v[i - 20:i]).mean()) if i >= 20 else 0.0
        to_mult  = to_now / avg_to20 if avg_to20 > 0 else float("nan")

        # 値幅制限節目突破フラグ
        band_parts = []
        for b in PRICE_BANDS:
            if ci >= b:
                prev_c = c[max(0, i - BAND_LOOKBACK):i]
                if len(prev_c) > 0 and np.any(prev_c < b):
                    band_parts.append(f"{int(b)}円")
        band_flag = "・".join(band_parts)

        hits.append({
            "ticker":       ticker,
            "date":         dates[i].date().isoformat(),
            "close":        round(ci, 1),
            "change_pct":   round(chg, 2),
            "vol_20x":      round(v[i] / avg_v20, 1),
            "to_man":       int(to_now / 1e4),
            "to_mult":      round(to_mult, 1) if not np.isnan(to_mult) else None,
            "range_pct":    round(range_ratio * 100, 1),
            "dry_pct":      round(avg_rec / avg_pri * 100, 1),
            "band_flag":    band_flag,
        })
        last_i = i

    return hits


def _scan(data: dict, range_pct: float, dry_ratio: float,
          vol_mult: float, ret_pct: float) -> list[dict]:
    all_hits: list[dict] = []
    for ticker, df in data.items():
        all_hits.extend(_detect_one(ticker, df, range_pct, dry_ratio, vol_mult, ret_pct))
    return all_hits


def _funnel(data: dict) -> dict[str, int]:
    """仕様値パラメータで各条件の通過数を返す（ファネル分析用）。"""
    cnt = {k: 0 for k in ["total", "p1", "p2", "p3", "p4", "p5"]}

    for ticker, df in data.items():
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        c  = df["Close"].values.astype(float)
        h  = df["High"].values.astype(float)
        lo = df["Low"].values.astype(float)
        v  = df["Volume"].values.astype(float)
        n  = len(c)
        if n < MIN_HISTORY:
            continue
        ma200 = pd.Series(c).rolling(200).mean().values

        for i in range(MIN_HISTORY, n):
            ci = c[i]
            cnt["total"] += 1

            if ci > PRICE_MAX:
                continue
            cnt["p1"] += 1

            m200 = ma200[i]
            if np.isnan(m200) or i < MA200_SLOPE_DAYS or np.isnan(ma200[i - MA200_SLOPE_DAYS]):
                continue
            if m200 >= ma200[i - MA200_SLOPE_DAYS] or ci >= m200:
                continue
            cnt["p2"] += 1

            ws = i - CONSOL_DAYS
            if ws < 0 or len(h[ws:i]) < CONSOL_DAYS:
                continue
            if (h[ws:i].max() - lo[ws:i].min()) / ci > CONSOL_RANGE_PCT:
                continue
            cnt["p3"] += 1

            v_rec = v[i - VOL_DRY_RECENT:i]
            v_pri = v[ws:i - VOL_DRY_RECENT]
            if len(v_pri) < 10:
                continue
            ar = v_rec.mean(); ap = v_pri.mean()
            if ap <= 0 or ar >= ap * VOL_DRY_RATIO:
                continue
            cnt["p4"] += 1

            avg_v20 = v[i - 20:i].mean() if i >= 20 else 0.0
            if avg_v20 <= 0 or v[i] / avg_v20 < BREAK_VOL_MULT:
                continue
            if c[i - 1] <= 0:
                continue
            if (ci - c[i - 1]) / c[i - 1] * 100 < BREAK_RETURN_PCT:
                continue
            wh = h[ws:i]
            if ci <= wh.max() and h[i] <= wh.max():
                continue
            cnt["p5"] += 1

    return cnt


# ── 出力 ────────────────────────────────────────────────────

def _print_summary(hits: list[dict], data: dict,
                   start_str: str, end_str: str,
                   relaxed: bool) -> None:
    label = "（緩和パラメータ）" if relaxed else "（仕様値パラメータ）"

    # ── 検知一覧 ──────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"【検知一覧】{label}")

    if hits:
        print(f"  {'検知日':12} {'コード':8} {'株価':>7}  {'騰落率':>7}  "
              f"{'出来高倍':>8}  {'売買代金倍':>10}  節目突破")
        print(f"  {'─'*64}")
        for h in sorted(hits, key=lambda x: x["date"]):
            to_s  = f"{h['to_mult']:.1f}x" if h['to_mult'] is not None else "─"
            band  = h['band_flag'] if h['band_flag'] else "─"
            print(f"  {h['date']:12} {h['ticker']:8} {h['close']:>7.1f}円"
                  f"  {h['change_pct']:>+6.2f}%  {h['vol_20x']:>6.1f}倍"
                  f"  {to_s:>8}  {band}")
    else:
        print("  (検知なし — 指定パラメータを満たす事例が見当たりませんでした)")

    # ── サマリー ──────────────────────────────────────────
    n_hits    = len(hits)
    n_tickers = len({h["ticker"] for h in hits})

    from datetime import datetime
    start_d = datetime.fromisoformat(start_str).date()
    end_d   = datetime.fromisoformat(end_str).date()
    total_years = (end_d - start_d).days / 365.25
    biz_days = len(pd.bdate_range(start_str, end_str))

    print(f"\n{'─' * 68}")
    print("【サマリー】")
    print(f"  検知件数          : {n_hits} 件")
    print(f"  ユニーク銘柄数    : {n_tickers} 銘柄")
    print(f"  対象期間          : {start_str} 〜 {end_str}  ({total_years:.1f}年 / 約{biz_days}営業日)")

    if n_hits > 0:
        avg_biz  = biz_days / n_hits
        avg_mon  = total_years * 12 / n_hits
        print(f"  検知頻度          : 平均 {avg_mon:.1f}ヶ月に1回（営業日 {avg_biz:.0f}日に1回）")
    else:
        print(f"  検知頻度          : 0件 / {total_years:.1f}年間 → 対象期間に検知なし")

    # ── 年別推移 ──────────────────────────────────────────
    from_yr = int(start_str[:4])
    to_yr   = int(end_str[:4])
    by_year  = Counter(h["date"][:4] for h in hits)
    by_month = Counter(h["date"][:7] for h in hits)

    print(f"\n【年別検知件数】")
    for yr in range(from_yr, to_yr + 1):
        bar = "█" * by_year.get(str(yr), 0)
        print(f"  {yr}: {by_year.get(str(yr), 0):>3}件  {bar}")

    if hits:
        print(f"\n【月別検知件数（検知月のみ）】")
        for mo in sorted(by_month):
            bar = "█" * by_month[mo]
            print(f"  {mo}: {by_month[mo]:>2}件  {bar}")


def main() -> None:
    print(f"キャッシュ読み込み: {CACHE_PATH.name} ...", end=" ", flush=True)
    if not CACHE_PATH.exists():
        print(f"\nERROR: ファイルが見つかりません → {CACHE_PATH}")
        return
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    data: dict = cache["data"]
    cache_date = cache.get("date", "不明")
    print(f"{len(data)}銘柄  キャッシュ日付: {cache_date}")

    # 検知期間（キャッシュ内の最古〜最新日）
    all_dates_min, all_dates_max = [], []
    for df in data.values():
        if not df.empty:
            all_dates_min.append(df.index[0])
            all_dates_max.append(df.index[-1])
    start_str = min(all_dates_min).date().isoformat()
    end_str   = max(all_dates_max).date().isoformat()

    # 使用パラメータ
    if _relax:
        rng, dry, vol, ret = RELAX["range_pct"], RELAX["dry_ratio"], RELAX["vol_mult"], RELAX["ret_pct"]
    else:
        rng, dry, vol, ret = CONSOL_RANGE_PCT, VOL_DRY_RATIO, BREAK_VOL_MULT, BREAK_RETURN_PCT

    label = "（緩和モード --relax）" if _relax else "（仕様値）"
    print(f"\n【パラメータ】{label}")
    print(f"  1. 株価上限     : ≤{PRICE_MAX:.0f}円")
    print(f"  2. MA200スロープ: 過去{MA200_SLOPE_DAYS}日で右肩下がり + 株価 < MA200")
    print(f"  3. 横ばい       : {CONSOL_DAYS}日レンジ ≤{rng*100:.0f}%")
    print(f"  4. 出来高枯渇   : 直近{VOL_DRY_RECENT}日 < 前期間比{dry*100:.0f}%")
    print(f"  5. ブレイク     : 出来高{vol:.0f}倍以上 + 騰落+{ret:.0f}%以上 + レンジ突破")

    print(f"\n走査中 ({len(data)}銘柄)...", flush=True)
    hits = _scan(data, rng, dry, vol, ret)

    _print_summary(hits, data, start_str, end_str, _relax)

    # 仕様値で0件のとき: 緩和パラメータの件数と条件ファネルを表示
    if not _relax and len(hits) == 0:
        print(f"\n【参考: 緩和パラメータでの検知件数】")
        r_hits = _scan(data, **RELAX)
        print(f"  レンジ≤{RELAX['range_pct']*100:.0f}% / 枯渇≤{RELAX['dry_ratio']*100:.0f}% / "
              f"出来高{RELAX['vol_mult']:.0f}倍 / 騰落+{RELAX['ret_pct']:.0f}%: {len(r_hits)}件")
        print(f"  → '--relax' オプションで詳細を表示できます")

    # 条件ファネル（仕様値パラメータ固定で表示）
    print(f"\n【条件ファネル（仕様値パラメータ / 絞り込み状況）】")
    print("走査中...", flush=True)
    f = _funnel(data)
    rows = [
        ("全日数",                         "total"),
        (f"1. 株価≤{PRICE_MAX:.0f}円",    "p1"),
        ("2. MA200右肩下がり+割れ",        "p2"),
        (f"3. 横ばい≤{CONSOL_RANGE_PCT*100:.0f}%",  "p3"),
        (f"4. 出来高枯渇≤{VOL_DRY_RATIO*100:.0f}%", "p4"),
        (f"5. ブレイクトリガー",            "p5"),
    ]
    prev = f["total"]
    for label, key in rows:
        val = f[key]
        pct_tot  = val / f["total"] * 100 if f["total"] > 0 else 0
        pct_prev = val / prev * 100        if prev > 0        else 0
        print(f"  {label:30s}: {val:>8,}  ({pct_tot:5.2f}% of total"
              + (f", {pct_prev:5.1f}% of prev)" if key != "total" else ")"))
        if key != "total":
            prev = val

    # CSV 保存（ヒットがある場合のみ）
    if hits:
        out = Path(__file__).parent / "results" / "lowprice_breakout_detections.csv"
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(hits).sort_values("date", ascending=False).to_csv(
            out, index=False, encoding="utf-8-sig"
        )
        print(f"\nCSV保存: {out}")


if __name__ == "__main__":
    main()
