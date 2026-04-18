#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グリッドサーチスクリプト
- 2年分データを1回ダウンロードしてキャッシュ → 全パラメータ組み合わせに再利用
- 各戦略の最適パラメータを探索（PF≥1.5、勝率≥55%、1日≥1件）
- backtest_cache.pkl を共用（backtest.py と同じキャッシュファイル）
"""

import itertools
import json
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import (
    fetch_jpx_stock_list,
    calc_rsi,
    calc_macd,
    MIN_AVG_TURNOVER,
    BREAKOUT_DAYS,
    DOW_N_SWINGS,
)

# ──────────────────────────────────────────────────────────────────────────────
CACHE_PATH   = Path(__file__).parent / "backtest_cache.pkl"
RESULTS_PATH = Path(__file__).parent / "gridsearch_results.json"
PERIOD       = "2y"
MIN_HISTORY  = 200
MAX_HOLD     = 20
MAX_WORKERS  = 8
BATCH_SIZE   = 100

# 評価基準
CRITERIA_PF  = 1.5
CRITERIA_WR  = 55.0
CRITERIA_SPD = 1.0   # シグナル数/日

# ──────────────────────────────────────────────────────────────────────────────
# パラメータグリッド
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_GRID = {
    "vol_mult":   [1.2, 1.5, 2.0, 2.5],   # 出来高倍率
    "rsi_lo":     [45.0, 50.0, 55.0],      # RSI下限
    "rsi_hi":     [65.0, 70.0, 75.0],      # RSI上限
    "weekly_dev": [15, 20, 25, 30],        # 週足MA25乖離上限（%）
}

PULLBACK_GRID = {
    "touch_pct": [1.01, 1.02, 1.03, 1.04], # MA25タッチ許容幅
    "rsi_lo":    [50.0, 55.0, 60.0],
    "rsi_hi":    [60.0, 65.0, 70.0],
}

BREAKOUT_GRID = {
    "bo_pct":   [1.0, 2.0, 3.0],          # 高値突破率（%）
    "vol_mult": [2.5, 3.0, 3.5],           # 出来高倍率
    "rsi_lo":   [55.0, 60.0, 65.0],        # RSI下限
    "consol":   [0, 1, 2],                 # 値固めパターン
}

# 値固めパターン: (下限比, 上限比, 最小日数)
CONSOL_PATTERNS = {
    0: (0.97, 1.00, 2),   # タイト: 97-100% × 2日
    1: (0.96, 1.02, 2),   # デフォルト: 96-102% × 2日
    2: (0.95, 1.03, 1),   # ゆるい: 95-103% × 1日
}

# preprocess で作る出来高倍率列（両グリッドの全値をカバー）
_VOL_MULTS_ALL = sorted(
    set(BASELINE_GRID["vol_mult"]) | set(BREAKOUT_GRID["vol_mult"])
)

# ──────────────────────────────────────────────────────────────────────────────
# 週足上昇トレンド（日次フォワードフィル）
# ──────────────────────────────────────────────────────────────────────────────
def _compute_weekly_uptrend_daily(df: pd.DataFrame, n_swings: int = DOW_N_SWINGS) -> pd.Series:
    import operator

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
# ベクトル化された出口リターン計算
# ──────────────────────────────────────────────────────────────────────────────
def _exit_returns_vec(closes: np.ndarray, entries: np.ndarray,
                      stops: np.ndarray, takes: np.ndarray) -> np.ndarray:
    """
    entries[i] が NaN でない各日についてシグナルが出たと仮定して出口リターンを計算。
    ストップ/テイクプロフィットに先にヒットした方、またはMAX_HOLD日後の終値で決済。
    """
    n    = len(closes)
    rets = np.full(n, np.nan)

    valid_mask = (~np.isnan(entries)) & (entries > 0)
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return rets

    # (n_valid, MAX_HOLD) の未来インデックス行列
    raw_idx  = valid_idx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)

    fut      = closes[safe_idx]
    fut      = np.where(in_range, fut, np.nan)

    stop_2d  = stops[valid_idx, np.newaxis]
    take_2d  = takes[valid_idx, np.newaxis]

    hit      = ((fut <= stop_2d) | (fut >= take_2d)) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)

    last_valid    = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    first_hit_pos = np.where(has_hit, np.argmax(hit, axis=1), last_valid)
    first_hit_pos = np.clip(first_hit_pos, 0, MAX_HOLD - 1)

    exit_idx    = np.clip(valid_idx + first_hit_pos + 1, 0, n - 1)
    exit_prices = closes[exit_idx]
    ents        = entries[valid_idx]

    ret_vals = np.where(has_fut, (exit_prices - ents) / ents * 100, np.nan)
    rets[valid_idx] = ret_vals
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# 1銘柄の事前計算
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_stock(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    全特徴量 + 各戦略の出口リターンを事前計算。
    グリッドサーチでパラメータを変えても、この計算は再実行不要。
    """
    if df_raw is None or len(df_raw) < MIN_HISTORY:
        return None

    df = df_raw.copy()

    # ── 基本指標 ──────────────────────────────────────────────────────────────
    df["MA25"] = df["Close"].rolling(25).mean()
    macd_s, sig_s = calc_macd(df["Close"])
    df["MACD"] = macd_s
    df["SIG"]  = sig_s
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

    # 週足MA25（日次フォワードフィル）
    wc          = df["Close"].resample("W").last()
    wma25       = wc.rolling(25).mean()
    df["MA25W"] = wma25.reindex(df.index, method="ffill")

    # 週足上昇トレンド
    try:
        df["weekly_uptrend"] = _compute_weekly_uptrend_daily(df)
    except Exception:
        df["weekly_uptrend"] = False

    # ── 派生フラグ ────────────────────────────────────────────────────────────
    df["above_daily"]   = df["Close"] > df["MA25"]
    df["above_weekly"]  = df["Close"] > df["MA25W"]
    df["breakout_pct"]  = (df["Close"] - df["past_high"]) / df["past_high"] * 100
    df["change_pct"]    = df["Close"].pct_change() * 100
    df["recent_bo"]     = df["Close"].shift(2) <= df["past_high"]
    df["macd_rising"]   = df["MACD"] > df["SIG"]
    df["turnover_ok"]   = df["avg_to"] >= MIN_AVG_TURNOVER
    df["bo_base"]       = df["Close"] > df["past_high"]

    # 出来高倍率（全閾値を一括生成）
    for vm in _VOL_MULTS_ALL:
        df[f"vol_{vm}x"] = (df["avg_vol"] > 0) & (df["Volume"] >= df["avg_vol"] * vm)

    # 週足MA25乖離上限（全閾値）
    for wd in BASELINE_GRID["weekly_dev"]:
        df[f"within_w{wd}"] = df["MA25W"].isna() | (
            df["Close"] <= df["MA25W"] * (1 + wd / 100)
        )

    # MA25タッチ（全閾値、直近2日以内）
    for tp in PULLBACK_GRID["touch_pct"]:
        col   = f"touch_{int(tp * 100)}"
        today = (df["Close"] >= df["MA25"]) & (df["Close"] <= df["MA25"] * tp)
        prev  = (df["Close"].shift(1) >= df["MA25"].shift(1)) & (
            df["Close"].shift(1) <= df["MA25"].shift(1) * tp
        )
        df[col] = today | prev

    # 値固めパターン（ATR縮小 + 高値近傍 × 各パターン）
    atr_recent      = df["ATR"].shift(1).rolling(5).mean()
    atr_prev5       = df["ATR"].shift(6).rolling(5).mean()
    df["atr_shrink"] = (atr_recent < atr_prev5) & (atr_prev5 > 0)

    for pat, (lo, hi, min_days) in CONSOL_PATTERNS.items():
        near_count = sum(
            ((df["Close"].shift(k) >= df["past_high"] * lo) &
             (df["Close"].shift(k) <= df["past_high"] * hi)).astype(int)
            for k in range(1, 6)
        )
        df[f"consol_{pat}"] = df["atr_shrink"] & (near_count >= min_days)

    # ── 出口リターン（3戦略分、ベクトル化）────────────────────────────────────
    n_rows  = len(df)
    closes  = df["Close"].values.astype(float)
    opens   = df["Open"].values.astype(float)
    ma25s   = df["MA25"].values.astype(float)
    atrs    = df["ATR"].values.astype(float)

    base_valid = (
        ~np.isnan(ma25s) & ~np.isnan(atrs) & (atrs > 0) &
        (np.arange(n_rows) >= MIN_HISTORY) &
        (np.arange(n_rows) < n_rows - 1)
    )
    vidx = np.where(base_valid)[0]

    # エントリー: 全戦略とも翌日始値
    next_open = opens[vidx + 1]
    valid_e   = (next_open > 0) & (~np.isnan(next_open))
    vidx2     = vidx[valid_e]
    e_arr     = next_open[valid_e]
    a_arr     = atrs[vidx2]

    for strat, rr in [
        ("baseline", 1.5),
        ("pullback",  1.5),
        ("breakout",  2.0),
    ]:
        entries = np.full(n_rows, np.nan)
        stops   = np.full(n_rows, np.nan)
        takes   = np.full(n_rows, np.nan)

        s = np.maximum(e_arr - a_arr * 2.0, e_arr * 0.90)
        t = e_arr + (e_arr - s) * rr

        entries[vidx2] = e_arr
        stops[vidx2]   = s
        takes[vidx2]   = t

        df[f"{strat}_ret"] = _exit_returns_vec(closes, entries, stops, takes)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# メトリクス計算
# ──────────────────────────────────────────────────────────────────────────────
def _calc_metrics(rets: pd.Series, trading_days: int) -> dict:
    rets = rets.dropna()
    n    = len(rets)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "spd": 0.0}

    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / n * 100
    avg_w  = wins.mean()  if len(wins)   > 0 else 0.0
    avg_l  = losses.mean() if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l) if avg_l != 0 else float("inf")

    return {
        "n":   n,
        "wr":  round(wr, 1),
        "pf":  round(pf, 2),
        "spd": round(n / trading_days, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# グリッドサーチ（1戦略）
# ──────────────────────────────────────────────────────────────────────────────
def grid_search(strategy: str, grid: dict,
                all_dfs: list[pd.DataFrame], trading_days: int) -> list[dict]:
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"\n【{strategy}】 {len(combos)} 通り × {len(all_dfs)} 銘柄")

    results = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        parts  = []

        for df in all_dfs:
            if df is None:
                continue

            if strategy == "baseline":
                vm  = params["vol_mult"]
                lo  = params["rsi_lo"]
                hi  = params["rsi_hi"]
                wd  = params["weekly_dev"]
                mask = (
                    df["above_daily"] & df["above_weekly"] & df["weekly_uptrend"] &
                    df[f"vol_{vm}x"] & df[f"within_w{wd}"] &
                    (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                    df["macd_rising"] & df["turnover_ok"] &
                    df["baseline_ret"].notna()
                )
                parts.append(df.loc[mask, "baseline_ret"])

            elif strategy == "pullback":
                tp  = params["touch_pct"]
                lo  = params["rsi_lo"]
                hi  = params["rsi_hi"]
                col = f"touch_{int(tp * 100)}"
                mask = (
                    df["above_daily"] & df["above_weekly"] & df["weekly_uptrend"] &
                    df[col] &
                    (df["RSI"] >= lo) & (df["RSI"] <= hi) &
                    df["macd_rising"] & df["turnover_ok"] &
                    df["pullback_ret"].notna()
                )
                parts.append(df.loc[mask, "pullback_ret"])

            elif strategy == "breakout":
                bp  = params["bo_pct"]
                vm  = params["vol_mult"]
                lo  = params["rsi_lo"]
                pat = params["consol"]
                mask = (
                    df["bo_base"] & (df["breakout_pct"] >= bp) &
                    df[f"vol_{vm}x"] & (df["change_pct"] >= 2.0) &
                    df["above_daily"] & df["above_weekly"] & df["weekly_uptrend"] &
                    (df["RSI"] >= lo) & df["recent_bo"] &
                    df[f"consol_{pat}"] & df["turnover_ok"] &
                    df["breakout_ret"].notna()
                )
                parts.append(df.loc[mask, "breakout_ret"])

        combined = pd.concat(parts) if parts else pd.Series(dtype=float)
        m        = _calc_metrics(combined, trading_days)
        results.append({**params, **m})

        if ci % 20 == 0 or ci == len(combos):
            print(f"  {ci:4d}/{len(combos)}: {params} "
                  f"→ n={m['n']:5d}, PF={m['pf']:.2f}, WR={m['wr']:.1f}%, {m['spd']:.2f}/日")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────────────────────────────────────
_STRAT_JP = {
    "baseline": "ベースライン型",
    "pullback": "押し目買い型",
    "breakout": "ブレイクアウト型",
}
_METRIC_KEYS = {"n", "wr", "pf", "spd"}


def print_results(strategy: str, results: list[dict]) -> None:
    qualified = [
        r for r in results
        if r["pf"] >= CRITERIA_PF and r["wr"] >= CRITERIA_WR and r["spd"] >= CRITERIA_SPD
    ]
    print(f"\n{'=' * 70}")
    print(f"【{_STRAT_JP.get(strategy, strategy)}】グリッドサーチ結果")
    print(f"  評価基準: PF≥{CRITERIA_PF}, 勝率≥{CRITERIA_WR}%, 1日≥{CRITERIA_SPD}件")
    print(f"  合格: {len(qualified)} / {len(results)} 通り")

    top_src = sorted(qualified, key=lambda r: r["pf"], reverse=True) if qualified else \
              sorted(results,   key=lambda r: r["pf"], reverse=True)

    if qualified:
        best = top_src[0]
        print(f"\n  ★ 最良パラメータ（PF最大）:")
        for k, v in best.items():
            if k in _METRIC_KEYS:
                continue
            # 値固めパターンはわかりやすく表示
            if k == "consol":
                lo, hi, md = CONSOL_PATTERNS[v]
                print(f"    {k:15s}: {v}  ({lo*100:.0f}%-{hi*100:.0f}% × {md}日以上)")
            else:
                print(f"    {k:15s}: {v}")
        print(f"    {'--- 結果 ---':15s}")
        print(f"    {'シグナル':15s}: {best['n']} 件 ({best['spd']:.2f}/日)")
        print(f"    {'勝率':15s}: {best['wr']:.1f}%")
        print(f"    {'PF':15s}: {best['pf']:.2f}")

    print(f"\n  上位5件（PF順）{'【合格のみ】' if qualified else '【条件未達】'}:")
    for r in top_src[:5]:
        param_str = "  ".join(
            f"{k}={v}" for k, v in r.items() if k not in _METRIC_KEYS
        )
        print(f"    PF={r['pf']:.2f}  WR={r['wr']:.1f}%  {r['spd']:.2f}/日  n={r['n']:5d}  | {param_str}")


# ──────────────────────────────────────────────────────────────────────────────
# キャッシュ操作（backtest.py と同形式）
# ──────────────────────────────────────────────────────────────────────────────
def _load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("date") == date.today().isoformat():
            return cached["data"]
        print(f"キャッシュが古い ({cached.get('date')})。再取得します。")
    except Exception as e:
        print(f"キャッシュ読み込みエラー: {e}")
    return None


def _save_cache(data: dict) -> None:
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"date": date.today().isoformat(), "data": data}, f)
        print(f"キャッシュ保存: {len(data)} 銘柄 → {CACHE_PATH.name}")
    except Exception as e:
        print(f"キャッシュ保存失敗: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("JPX銘柄リストを取得中...")
    tickers = fetch_jpx_stock_list()
    print(f"対象: {len(tickers)} 銘柄")

    # ── データ取得（当日キャッシュ優先） ──────────────────────────────────────
    raw_data = _load_cache()
    if raw_data:
        print(f"キャッシュから {len(raw_data)} 銘柄を読み込み（yfinanceスキップ）")
    else:
        raw_data = {}
        print(f"yfinanceから {PERIOD} 分データを一括取得中...")
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            try:
                raw = yf.download(
                    batch, period=PERIOD, auto_adjust=True,
                    group_by="ticker", progress=False, threads=True,
                )
                for t in batch:
                    try:
                        df = raw if len(batch) == 1 else (
                            raw[t] if t in raw.columns.get_level_values(0) else None
                        )
                        if df is not None and len(df) >= MIN_HISTORY:
                            raw_data[t] = df.dropna(how="all")
                    except Exception:
                        pass
            except Exception as e:
                print(f"  バッチ {i // BATCH_SIZE + 1} エラー: {e}")
            print(f"  {min(i + BATCH_SIZE, len(tickers))}/{len(tickers)} 完了  "
                  f"取得: {len(raw_data)} 銘柄")
        _save_cache(raw_data)

    # ── 事前計算（並列処理） ──────────────────────────────────────────────────
    print(f"\n特徴量を事前計算中... ({len(raw_data)} 銘柄, 並列{MAX_WORKERS}スレッド)")
    all_dfs: list[pd.DataFrame] = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(preprocess_stock, df): t for t, df in raw_data.items()}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)
            done += 1
            if done % 500 == 0 or done == len(raw_data):
                print(f"  {done}/{len(raw_data)} 完了  有効: {len(all_dfs)} 銘柄")

    # 実際の取引日数を推定（MIN_HISTORY 以降のバー数の最大値）
    trading_days = max(
        (len(df) - MIN_HISTORY for df in all_dfs if df is not None),
        default=500,
    )
    print(f"\n推定取引日数: {trading_days} 日")

    # ── グリッドサーチ ────────────────────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}
    for strat, grid in [
        ("baseline", BASELINE_GRID),
        ("pullback", PULLBACK_GRID),
        ("breakout", BREAKOUT_GRID),
    ]:
        results = grid_search(strat, grid, all_dfs, trading_days)
        all_results[strat] = results
        print_results(strat, results)

    # ── 結果保存 ──────────────────────────────────────────────────────────────
    RESULTS_PATH.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n結果を保存: {RESULTS_PATH.name}")
