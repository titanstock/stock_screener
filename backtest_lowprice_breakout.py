#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横ばい・出来高枯渇からの初動ブレイク バックテスト
================================================================
【検知パターン】
  1. 低位株フィルター  : 終値 ≤ 500円
  2. 長期下落         : MA200 右肩下がり(過去20日比) かつ 終値 < MA200
  3. 横ばい判定       : 直近30日の (高値-安値)/終値 ≤ 15%
  4. 出来高枯渇       : 直近20日平均出来高 < 前30日平均出来高 × 50%
  5. ブレイクトリガー : 出来高5倍以上 + 騰落+10%以上 + レンジ高値突破
  6. 売買代金倍率     : 参考値として出力（足切りなし）
  7. 値幅制限節目突破 : 100/300/500円の節目を直近で超えたかフラグ出力

  ※ MA収束条件（5）は横ばいが成立した時点で自然に充足されるため除外
  ※ 仕様値: CONSOL_DAYS=30 / レンジ15% / 枯渇50% / 出来高5倍 / 騰落+10%

【出力】
  ヒット銘柄・日付、5日/10日/20日後リターン、勝率・平均リターン

使い方:
  python backtest_lowprice_breakout.py [キャッシュ名]
  例: python backtest_lowprice_breakout.py backtest_cache.pkl
"""

import pickle, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_cache_name = sys.argv[1] if len(sys.argv) > 1 else "backtest_cache.pkl"
CACHE_PATH  = Path(__file__).parent / _cache_name

# ── パラメータ（上部で変更可） ────────────────────────────
PRICE_MAX        = 500.0   # 低位株: 終値上限（円）

MA200_SLOPE_DAYS = 20      # MA200右肩下がり確認期間（日）

CONSOL_DAYS      = 30      # 横ばい・枯渇判定の窓（日）
CONSOL_RANGE_PCT = 0.15    # 横ばい: (高値-安値)/終値 ≤ この値（15%）

VOL_DRY_RECENT   = 20      # 枯渇判定: 直近N日
VOL_DRY_RATIO    = 0.50    # 枯渇判定: 直近平均 < 前期間平均 × この比率（50%）

BREAK_VOL_MULT   = 5.0     # ブレイク: 出来高が直近20日平均の何倍以上
BREAK_RETURN_PCT = 10.0    # ブレイク: 当日騰落率の下限(%)

MIN_HISTORY      = 260     # 最低必要日数（MA200 + CONSOL_DAYS(30) + バッファ）
MIN_TURNOVER_BRK = 3_000_000  # ブレイク当日の最低売買代金（円）

PRICE_BANDS  = [100.0, 300.0, 500.0]  # 値幅制限節目
BAND_LOOKBACK = 5  # 節目突破確認: 直近N日以内

HOLD_DAYS = [5, 10, 20]  # 保有日数（営業日）

# 同一銘柄の再ヒット抑制（ヒット後Nバー以内の再ヒットは無視）
COOLDOWN_BARS = 20
# ────────────────────────────────────────────────────────


def _band_flag(close_arr: np.ndarray, i: int) -> str:
    """直近 BAND_LOOKBACK 日以内に PRICE_BANDS の節目を上抜けたかチェック。"""
    crossed = []
    for band in PRICE_BANDS:
        window_start = max(0, i - BAND_LOOKBACK)
        prev_closes = close_arr[window_start:i]          # 直近窓（当日除く）
        if len(prev_closes) == 0:
            continue
        # 節目を今日以前に割れていて、今日以上で引けている
        if close_arr[i] >= band and np.any(prev_closes < band):
            crossed.append(f"{int(band)}円")
    return "・".join(crossed) if crossed else ""


def detect_signals(ticker: str, df: pd.DataFrame) -> list[dict]:
    """1銘柄の全シグナル日を検出して返す。"""
    # yfinance バッチ取得で重複カラムが生じることがあるため先頭列だけ取る
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    c  = df["Close"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    v  = df["Volume"].values.astype(float)
    n  = len(c)

    if n < MIN_HISTORY:
        return []

    # 事前計算（pandas rolling で NaN 込み）
    s      = pd.Series(c)
    ma200  = s.rolling(200).mean().values

    dates = df.index
    hits  = []
    last_hit_i = -COOLDOWN_BARS - 1   # クールダウン管理

    # 走査開始インデックス = 最低限必要な履歴を確保できる位置
    start = MIN_HISTORY
    for i in range(start, n - max(HOLD_DAYS)):
        # ── クールダウン ──
        if i - last_hit_i <= COOLDOWN_BARS:
            continue

        ci = c[i]

        # ① 低位株フィルター
        if ci > PRICE_MAX:
            continue

        # ② MA200: 計算可能か + 右肩下がり + 株価がMA200未満
        m200 = ma200[i]
        if np.isnan(m200):
            continue
        m200_prev = ma200[i - MA200_SLOPE_DAYS] if i >= MA200_SLOPE_DAYS else np.nan
        if np.isnan(m200_prev):
            continue
        if m200 >= m200_prev:   # 上昇または横ばい → 除外
            continue
        if ci >= m200:          # 株価がMA200以上 → 除外
            continue

        # ── 横ばい窓: 直近 CONSOL_DAYS 日（当日除く）──
        w_start = i - CONSOL_DAYS
        if w_start < 0:
            continue
        wh = h[w_start:i]
        wl = lo[w_start:i]
        if len(wh) < CONSOL_DAYS:
            continue

        # ③ 横ばい判定
        range_ratio = (wh.max() - wl.min()) / ci
        if range_ratio > CONSOL_RANGE_PCT:
            continue

        # ④ 出来高枯渇
        # 直近 VOL_DRY_RECENT 日（当日除く）
        v_recent = v[i - VOL_DRY_RECENT:i]
        # 前期間: CONSOL_DAYS 日（直近20日より前）
        v_prior  = v[w_start:i - VOL_DRY_RECENT]
        if len(v_prior) < 10:   # 前期間データが少ない場合はスキップ
            continue
        avg_recent = v_recent.mean()
        avg_prior  = v_prior.mean()
        if avg_prior <= 0:
            continue
        if avg_recent >= avg_prior * VOL_DRY_RATIO:
            continue

        # ⑤ ブレイクトリガー（当日）
        # 直近20日平均出来高（当日除く）
        avg_vol_20 = v[i - 20:i].mean() if i >= 20 else 0.0
        if avg_vol_20 <= 0:
            continue

        # 出来高5倍以上
        if v[i] < avg_vol_20 * BREAK_VOL_MULT:
            continue

        # 騰落率+10%以上
        if c[i - 1] <= 0:
            continue
        change_pct = (ci - c[i - 1]) / c[i - 1] * 100
        if change_pct < BREAK_RETURN_PCT:
            continue

        # 終値または高値が横ばいレンジの高値を上抜け
        range_hi = wh.max()
        if ci <= range_hi and h[i] <= range_hi:
            continue

        # 売買代金フィルター（最低限）
        turnover_today = ci * v[i]
        if turnover_today < MIN_TURNOVER_BRK:
            continue

        # ⑦ 売買代金倍率（参考値）
        avg_to_20 = (c[i - 20:i] * v[i - 20:i]).mean() if i >= 20 else 0.0
        to_mult = turnover_today / avg_to_20 if avg_to_20 > 0 else float("nan")

        # ⑧ 値幅制限節目突破フラグ
        band_flag = _band_flag(c, i)

        # ── 先後リターン計算 ──
        fwd = {}
        for nd in HOLD_DAYS:
            j = i + nd
            if j < n and c[j] > 0:
                fwd[nd] = (c[j] - ci) / ci * 100
            else:
                fwd[nd] = float("nan")

        hits.append({
            "ticker":       ticker,
            "date":         dates[i].date().isoformat(),
            "close":        round(ci, 1),
            "change_pct":   round(change_pct, 2),
            "vol_20x":      round(v[i] / avg_vol_20, 1),
            "to_mult":      round(to_mult, 1) if not np.isnan(to_mult) else None,
            "range_ratio":  round(range_ratio * 100, 1),   # %表示
            "vol_dry_ratio": round(avg_recent / avg_prior * 100, 1),  # %
            "band_flag":    band_flag,
            "ret_5d":       round(fwd[5],  2) if not np.isnan(fwd[5])  else None,
            "ret_10d":      round(fwd[10], 2) if not np.isnan(fwd[10]) else None,
            "ret_20d":      round(fwd[20], 2) if not np.isnan(fwd[20]) else None,
        })
        last_hit_i = i

    return hits


def summarize(hits: list[dict]) -> None:
    if not hits:
        print("\nヒットなし")
        return

    df = pd.DataFrame(hits)

    print(f"\n{'=' * 72}")
    print(f"ヒット件数: {len(df)} 件  /  銘柄数: {df['ticker'].nunique()} 銘柄")
    print(f"{'=' * 72}")

    # ── 全ヒット一覧（上位50件） ──
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if pd.notna(x) else "  N/A")

    show_cols = [
        "ticker", "date", "close", "change_pct",
        "vol_20x", "to_mult", "range_ratio", "vol_dry_ratio",
        "band_flag", "ret_5d", "ret_10d", "ret_20d",
    ]
    print("\n── ヒット一覧（新しい順, 最大50件）──")
    print(df.sort_values("date", ascending=False)[show_cols].head(50).to_string(index=False))

    # ── 勝率・平均リターン ──
    print(f"\n{'─' * 72}")
    print("【勝率・平均リターン】")
    for nd, col in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
        sub = df[col].dropna()
        if len(sub) == 0:
            print(f"  {nd:>2}日後: データなし")
            continue
        wr  = (sub > 0).mean() * 100
        avg = sub.mean()
        med = sub.median()
        pf_pos = sub[sub > 0].sum()
        pf_neg = sub[sub < 0].abs().sum()
        pf  = pf_pos / pf_neg if pf_neg > 0 else float("inf")
        print(
            f"  {nd:>2}日後:  勝率 {wr:5.1f}%  "
            f"平均 {avg:+6.2f}%  中央値 {med:+6.2f}%  "
            f"PF {pf:.2f}  n={len(sub)}"
        )

    # ── 値幅節目突破あり vs なし ──
    with_flag    = df[df["band_flag"] != ""]
    without_flag = df[df["band_flag"] == ""]
    if len(with_flag) > 0:
        print(f"\n【節目突破フラグあり: {len(with_flag)}件】")
        for nd, col in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
            sub = with_flag[col].dropna()
            if len(sub) == 0: continue
            print(f"  {nd:>2}日後:  勝率 {(sub>0).mean()*100:5.1f}%  平均 {sub.mean():+6.2f}%  n={len(sub)}")
    if len(without_flag) > 0:
        print(f"\n【節目突破フラグなし: {len(without_flag)}件】")
        for nd, col in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
            sub = without_flag[col].dropna()
            if len(sub) == 0: continue
            print(f"  {nd:>2}日後:  勝率 {(sub>0).mean()*100:5.1f}%  平均 {sub.mean():+6.2f}%  n={len(sub)}")

    # ── 売買代金倍率分位 ──
    to_valid = df["to_mult"].dropna()
    if len(to_valid) > 0:
        print(f"\n【売買代金倍率 (to_mult) 分布】")
        for q in [0.25, 0.50, 0.75, 0.90]:
            print(f"  {int(q*100)}%ile: {to_valid.quantile(q):.1f}x")


def main() -> None:
    print(f"キャッシュ読み込み: {CACHE_PATH.name} ...", end=" ", flush=True)
    if not CACHE_PATH.exists():
        print(f"\nERROR: キャッシュファイルが見つかりません: {CACHE_PATH}")
        return
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    data: dict[str, pd.DataFrame] = cache["data"]
    print(f"{len(data)}銘柄  キャッシュ日付: {cache.get('date', '不明')}")

    print(f"\nパラメータ:")
    print(f"  株価上限          : ≤{PRICE_MAX:.0f}円")
    print(f"  MA200スロープ     : 過去{MA200_SLOPE_DAYS}日で右肩下がり")
    print(f"  横ばい窓          : {CONSOL_DAYS}日  値幅≤{CONSOL_RANGE_PCT*100:.0f}%")
    print(f"  出来高枯渇        : 直近{VOL_DRY_RECENT}日が前期間比≤{VOL_DRY_RATIO*100:.0f}%")
    print(f"  ブレイクトリガー  : 出来高{BREAK_VOL_MULT:.0f}倍以上 + 騰落+{BREAK_RETURN_PCT:.0f}%以上 + レンジ突破")
    print(f"  クールダウン      : {COOLDOWN_BARS}バー")

    all_hits: list[dict] = []
    total = len(data)
    done  = 0

    for ticker, df in data.items():
        hits = detect_signals(ticker, df)
        all_hits.extend(hits)
        done += 1
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total} 完了  ヒット累計: {len(all_hits)}件", flush=True)

    summarize(all_hits)

    # CSV 保存
    if all_hits:
        out_path = Path(__file__).parent / "results" / "lowprice_breakout_hits.csv"
        out_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(all_hits).sort_values("date", ascending=False).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"\nCSV保存: {out_path}")


if __name__ == "__main__":
    main()
