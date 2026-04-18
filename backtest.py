#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バックテストスクリプト
- yfinanceから2年分の株価データを一括取得（J-Quants Freeプラン制限回避）
- 3戦略の条件を各営業日に適用してシグナルを生成
- 翌営業日の終値でパフォーマンスを計測
- breakout_v2: 値固め検出 + 翌日始値エントリー（breakoutとの比較用）
- 注意: 時価総額フィルターは現在値で代用（サバイバーシップバイアスあり）
"""

import json
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from stock_screener import (
    BREAKOUT_DAYS, MIN_AVG_TURNOVER, VOL_SPIKE_MULT,
    fetch_jpx_stock_list,
    calc_rsi, calc_macd,
    evaluate_strategies,
    check_consolidation,
    has_weekly_uptrend,
)

# ──────────────────────────────────────────────────────────────────────────────
MIN_HISTORY  = 200   # バックテストに必要な最低データ数
MAX_HOLD     = 20    # 最大保有営業日数（約1ヶ月）
MAX_WORKERS  = 6
RESULTS_PATH = Path(__file__).parent / "backtest_results.json"
CACHE_PATH   = Path(__file__).parent / "backtest_cache.pkl"
PERIOD       = "2y"
# ──────────────────────────────────────────────────────────────────────────────

# 前回v2結果（比較用）
PREV_V2 = {"n": 145, "wr": 50.3, "ret": 2.08, "stop_rate": 39.3, "pf": 1.61, "days": 12.0}


def check_consolidation_v3(sl: pd.DataFrame, past_high: float) -> bool:
    """
    値固め検出 v3
    ① ATR縮小: 直近5日ATR平均 < 前5日ATR平均
    ② 高値95%以内に1日以上停滞（直前5日間）
    """
    if len(sl) < 12 or "ATR" not in sl.columns:
        return False

    # ① ATR縮小
    atr_window = sl["ATR"].iloc[-11:-1]
    if len(atr_window) < 10 or atr_window.isna().any():
        return False
    atr_recent = atr_window.iloc[5:].mean()
    atr_prev   = atr_window.iloc[:5].mean()
    if atr_prev <= 0:
        return False
    atr_shrinking = atr_recent < atr_prev

    # ② 高値95%以内に1日以上（close が past_high の 95〜100% の範囲）
    close_window = sl["Close"].iloc[-6:-1]
    near_high = (
        (close_window >= past_high * 0.95) &
        (close_window <= past_high)
    ).sum()

    return atr_shrinking and near_high >= 1



def backtest_ticker(ticker: str, df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) < MIN_HISTORY:
        return []

    df = df.copy()
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

    signals  = []
    lookback = 200
    rr_ratio = {"baseline": 1.5, "breakout": 2.0, "pullback": 1.5}

    for i in range(lookback, len(df) - 1):
        sl    = df.iloc[:i+1]
        close = float(sl["Close"].iloc[-1])
        prev  = float(sl["Close"].iloc[-2])
        ma25  = float(sl["MA25"].iloc[-1])
        rsi   = float(sl["RSI"].iloc[-1])
        macd  = float(sl["MACD"].iloc[-1])
        sig   = float(sl["SIG"].iloc[-1])
        atr   = float(sl["ATR"].iloc[-1])
        vol   = float(sl["Volume"].iloc[-1])

        if any(pd.isna(v) for v in [ma25, rsi, macd, sig, atr]) or atr <= 0:
            continue

        n         = BREAKOUT_DAYS + 1
        avg_vol   = float(sl["Volume"].iloc[-(n+1):-1].mean())
        avg_to    = float((sl["Close"].iloc[-(n+1):-1] * sl["Volume"].iloc[-(n+1):-1]).mean())
        past_high = float(sl["High"].iloc[-(n+1):-1].max())

        if avg_to < MIN_AVG_TURNOVER:
            continue

        wc     = sl["Close"].resample("W").last().dropna()
        ma25_w = float(wc.rolling(25).mean().iloc[-1]) if len(wc) >= 25 else None

        breakout_pct = (close - past_high) / past_high * 100 if past_high > 0 else 0.0
        change_pct   = (close - prev) / prev * 100 if prev > 0 else 0.0

        # ── 既存3戦略（翌日始値エントリー）──────────────────────────────────
        matched_keys = evaluate_strategies(
            df          = sl,
            close       = close,
            prev_close  = prev,
            ma25_daily  = ma25,
            ma25_weekly = ma25_w,
            avg_vol_20  = avg_vol,
            vol_now     = vol,
            past_high_20= past_high,
            rsi         = rsi,
            macd_now    = macd,
            sig_now     = sig,
            breakout_pct= breakout_pct,
            change_pct  = change_pct,
        )

        for strategy in matched_keys:
            # ── 約定ロジック（戦略別） ────────────────────────────────────────
            if strategy == "breakout":
                # 翌日始値で成行
                entry = float(df["Open"].iloc[i + 1])
                if entry <= 0 or pd.isna(entry):
                    continue
                entry_offset = 1   # 翌日（i+1）から出口チェック開始

            elif strategy == "baseline":
                # 翌日に終値×0.98の指値。翌日安値 ≤ 指値なら約定
                limit = close * 0.98
                next_low = float(df["Low"].iloc[i + 1])
                if pd.isna(next_low) or next_low > limit:
                    continue   # 翌日約定せずスキップ
                entry = limit
                entry_offset = 1   # 翌日約定 → 翌日Closeから出口チェック

            elif strategy == "pullback":
                # 翌日〜3日間、安値 ≤ MA25 なら指値約定
                entry = None
                entry_offset = None
                for wait in range(1, 4):
                    wi = i + wait
                    if wi >= len(df):
                        break
                    wl = float(df["Low"].iloc[wi])
                    if not pd.isna(wl) and wl <= ma25:
                        entry = ma25
                        entry_offset = wait   # 約定した日から出口チェック
                        break
                if entry is None:
                    continue   # 3日以内に約定せずスキップ

            else:
                continue

            stop = max(entry - atr * 2.0, entry * 0.90)
            take = entry + (entry - stop) * rr_ratio[strategy]

            exit_price  = None
            exit_day    = None
            exit_reason = "max"

            for offset in range(entry_offset, entry_offset + MAX_HOLD):
                idx = i + offset
                if idx >= len(df):
                    break
                fc = float(df["Close"].iloc[idx])
                if fc <= stop:
                    exit_price  = fc
                    exit_day    = offset - entry_offset + 1
                    exit_reason = "stop"
                    break
                if fc >= take:
                    exit_price  = fc
                    exit_day    = offset - entry_offset + 1
                    exit_reason = "take"
                    break
                exit_price = fc
                exit_day   = offset - entry_offset + 1

            if exit_price is None:
                continue

            ret = (exit_price - entry) / entry * 100
            signals.append({
                "date":        sl.index[-1].strftime("%Y-%m-%d"),
                "code":        ticker.replace(".T", ""),
                "strategy":    strategy,
                "entry":       round(entry, 2),
                "stop":        round(stop, 2),
                "take":        round(take, 2),
                "exit_price":  round(exit_price, 2),
                "exit_day":    exit_day,
                "exit_reason": exit_reason,
                "return":      round(ret, 3),
                "win":         ret > 0,
            })

        # ── breakout_v3: 新条件（前日比・RSI除外、高値95%以内1日+） ────────────
        if i + 1 < len(df):
            above_weekly_v3 = ma25_w is not None and close > ma25_w
            vol_3x_v3       = avg_vol > 0 and vol >= avg_vol * VOL_SPIKE_MULT
            recent_bo_v3    = len(sl) >= 3 and float(sl["Close"].iloc[-3]) <= past_high
            breakout_ok_v3  = close > past_high and breakout_pct >= 1.0

            if (close > ma25 and above_weekly_v3 and has_weekly_uptrend(sl)
                    and breakout_ok_v3 and vol_3x_v3 and recent_bo_v3
                    and check_consolidation_v3(sl, past_high)):
                entry_v3 = float(df["Open"].iloc[i + 1])
                if entry_v3 > 0 and not pd.isna(entry_v3):
                    stop_v3 = max(entry_v3 - atr * 2.0, entry_v3 * 0.90)
                    take_v3 = entry_v3 + (entry_v3 - stop_v3) * 2.0

                    exit_price_v3  = None
                    exit_day_v3    = None
                    exit_reason_v3 = "max"

                    for offset in range(0, MAX_HOLD):
                        idx = i + 1 + offset
                        if idx >= len(df):
                            break
                        fc = float(df["Close"].iloc[idx])
                        if fc <= stop_v3:
                            exit_price_v3, exit_day_v3, exit_reason_v3 = fc, offset + 1, "stop"
                            break
                        if fc >= take_v3:
                            exit_price_v3, exit_day_v3, exit_reason_v3 = fc, offset + 1, "take"
                            break
                        exit_price_v3, exit_day_v3 = fc, offset + 1

                    if exit_price_v3 is not None:
                        ret_v3 = (exit_price_v3 - entry_v3) / entry_v3 * 100
                        signals.append({
                            "date":        sl.index[-1].strftime("%Y-%m-%d"),
                            "code":        ticker.replace(".T", ""),
                            "strategy":    "breakout_v3",
                            "entry":       round(entry_v3, 2),
                            "stop":        round(stop_v3, 2),
                            "take":        round(take_v3, 2),
                            "exit_price":  round(exit_price_v3, 2),
                            "exit_day":    exit_day_v3,
                            "exit_reason": exit_reason_v3,
                            "return":      round(ret_v3, 3),
                            "win":         ret_v3 > 0,
                        })

        # ── breakout_v2: 値固め検出 + 翌日始値エントリー ─────────────────────
        if "breakout" in matched_keys and i + 1 < len(df):
            if check_consolidation(sl, past_high):
                entry_v2 = float(df["Open"].iloc[i + 1])
                if entry_v2 > 0 and not pd.isna(entry_v2):
                    stop_v2 = max(entry_v2 - atr * 2.0, entry_v2 * 0.90)
                    take_v2 = entry_v2 + (entry_v2 - stop_v2) * 2.0

                    exit_price_v2  = None
                    exit_day_v2    = None
                    exit_reason_v2 = "max"

                    for offset in range(0, MAX_HOLD):
                        idx = i + 1 + offset
                        if idx >= len(df):
                            break
                        fc = float(df["Close"].iloc[idx])
                        if fc <= stop_v2:
                            exit_price_v2, exit_day_v2, exit_reason_v2 = fc, offset + 1, "stop"
                            break
                        if fc >= take_v2:
                            exit_price_v2, exit_day_v2, exit_reason_v2 = fc, offset + 1, "take"
                            break
                        exit_price_v2, exit_day_v2 = fc, offset + 1

                    if exit_price_v2 is not None:
                        ret_v2 = (exit_price_v2 - entry_v2) / entry_v2 * 100
                        signals.append({
                            "date":        sl.index[-1].strftime("%Y-%m-%d"),
                            "code":        ticker.replace(".T", ""),
                            "strategy":    "breakout_v2",
                            "entry":       round(entry_v2, 2),
                            "stop":        round(stop_v2, 2),
                            "take":        round(take_v2, 2),
                            "exit_price":  round(exit_price_v2, 2),
                            "exit_day":    exit_day_v2,
                            "exit_reason": exit_reason_v2,
                            "return":      round(ret_v2, 3),
                            "win":         ret_v2 > 0,
                        })

    return signals


def summarize(signals: list[dict]) -> None:
    df = pd.DataFrame(signals)
    if df.empty:
        print("シグナルなし")
        return

    strategies = {
        "baseline":    "ベースライン型",
        "breakout":    "ブレイクアウト（現行・引値）",
        "breakout_v2": "ブレイクアウトv2（値固め+始値・旧条件）",
        "breakout_v3": "ブレイクアウトv3（前日比・RSI除外・高値95%）",
        "pullback":    "押し目買い型",
    }

    print("\n" + "=" * 75)
    print(f"バックテスト結果（条件付き出口 / 最大保有{MAX_HOLD}営業日）")
    print("=" * 75)

    for sk, label in strategies.items():
        s = df[df["strategy"] == sk].copy()
        if s.empty:
            print(f"\n【{label}】データなし")
            continue

        n        = len(s)
        win_mask = s["win"].astype(bool)
        wins     = int(win_mask.sum())
        wr       = wins / n * 100
        avg_ret  = s["return"].mean()
        avg_win  = s[win_mask]["return"].mean() if wins > 0 else 0.0
        avg_loss = s[~win_mask]["return"].mean() if (n - wins) > 0 else 0.0
        pf       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        by_reason = s["exit_reason"].value_counts()
        take_n = int(by_reason.get("take", 0))
        stop_n = int(by_reason.get("stop", 0))
        max_n  = int(by_reason.get("max",  0))
        avg_days = s["exit_day"].mean()

        # 1日あたりシグナル数（約500営業日）
        sig_per_day = n / 500

        print(f"\n【{label}】  サンプル数: {n} 件  ({sig_per_day:.1f}件/日)")
        print(f"  勝率          : {wr:.1f}%  （勝: {wins} / 負: {n-wins}）")
        print(f"  平均リターン  : {avg_ret:+.2f}%")
        print(f"  平均利益      : {avg_win:+.2f}%  平均損失: {avg_loss:+.2f}%")
        print(f"  プロフィット  : {pf:.2f}")
        print(f"  平均保有日数  : {avg_days:.1f}日")
        print(f"  決済内訳      : 利確={take_n}件({take_n/n*100:.1f}%)  損切={stop_n}件({stop_n/n*100:.1f}%)  期間満了={max_n}件({max_n/n*100:.1f}%)")

    # ── breakout_v2 vs breakout_v3 比較サマリー ─────────────────────────────
    b2 = df[df["strategy"] == "breakout_v2"]
    b3 = df[df["strategy"] == "breakout_v3"]
    if not b3.empty:
        print("\n" + "=" * 75)
        print("【ブレイクアウト v2 vs v3 比較】")
        print(f"  v2: 旧条件（前日比+2%・RSI60+・高値96-102%×2日）翌日始値")
        print(f"  v3: 新条件（前日比・RSI除外・高値95%以内×1日）翌日始値")
        print(f"{'項目':<20} {'v2（前回）':>15} {'v3（今回）':>15}")
        print("-" * 52)

        def _row(label, v2val, v3val):
            return f"{label:<20} {v2val:>15} {v3val:>15}"

        n3   = len(b3)
        wr3  = b3["win"].astype(bool).mean() * 100
        ar3  = b3["return"].mean()
        st3  = b3["exit_reason"].eq("stop").mean() * 100
        d3   = b3["exit_day"].mean()
        w3   = b3[b3["win"].astype(bool)]["return"].mean() if n3 > 0 else 0
        l3   = b3[~b3["win"].astype(bool)]["return"].mean() if n3 > 0 else -1
        pf3  = abs(w3 / l3) if l3 != 0 else float("inf")

        p = PREV_V2
        print(_row("サンプル数", f"{p['n']}件({p['n']/500:.1f}/日)", f"{n3}件({n3/500:.1f}/日)"))
        print(_row("勝率",       f"{p['wr']:.1f}%",  f"{wr3:.1f}%"))
        print(_row("平均リターン", f"{p['ret']:+.2f}%", f"{ar3:+.2f}%"))
        print(_row("損切り率",   f"{p['stop_rate']:.1f}%", f"{st3:.1f}%"))
        print(_row("平均保有日数", f"{p['days']:.1f}日", f"{d3:.1f}日"))
        print(_row("プロフィット", f"{p['pf']:.2f}", f"{pf3:.2f}"))

    # ── breakout vs breakout_v2 比較サマリー ────────────────────────────────
    b1 = df[df["strategy"] == "breakout"]
    if not b1.empty and not b2.empty:
        print("\n" + "=" * 75)
        print("【ブレイクアウト 比較サマリー】")
        print(f"{'項目':<20} {'現行（引値）':>15} {'v2（値固め+始値）':>18}")
        print("-" * 55)

        def row(label, v1, v2):
            return f"{label:<20} {v1:>15} {v2:>18}"

        n1, n2 = len(b1), len(b2)
        wr1 = b1["win"].astype(bool).mean() * 100
        wr2 = b2["win"].astype(bool).mean() * 100
        ar1, ar2 = b1["return"].mean(), b2["return"].mean()
        stop1 = b1["exit_reason"].eq("stop").mean() * 100
        stop2 = b2["exit_reason"].eq("stop").mean() * 100
        days1, days2 = b1["exit_day"].mean(), b2["exit_day"].mean()
        pf1_w = b1[b1["win"].astype(bool)]["return"].mean() if n1 > 0 else 0
        pf1_l = b1[~b1["win"].astype(bool)]["return"].mean() if n1 > 0 else -1
        pf2_w = b2[b2["win"].astype(bool)]["return"].mean() if n2 > 0 else 0
        pf2_l = b2[~b2["win"].astype(bool)]["return"].mean() if n2 > 0 else -1
        pf1 = abs(pf1_w / pf1_l) if pf1_l != 0 else float("inf")
        pf2 = abs(pf2_w / pf2_l) if pf2_l != 0 else float("inf")

        print(row("サンプル数", f"{n1}件({n1/500:.1f}/日)", f"{n2}件({n2/500:.1f}/日)"))
        print(row("勝率", f"{wr1:.1f}%", f"{wr2:.1f}%"))
        print(row("平均リターン", f"{ar1:+.2f}%", f"{ar2:+.2f}%"))
        print(row("損切り率", f"{stop1:.1f}%", f"{stop2:.1f}%"))
        print(row("平均保有日数", f"{days1:.1f}日", f"{days2:.1f}日"))
        print(row("プロフィット", f"{pf1:.2f}", f"{pf2:.2f}"))

    print("\n" + "=" * 75)
    total = len(df[df["strategy"] != "breakout_v2"])  # v2は比較用なのでtotalから除外
    wins_all = int(df[df["strategy"] != "breakout_v2"]["win"].astype(bool).sum())
    if total > 0:
        print(f"総シグナル数（3戦略）: {total} 件  総合勝率: {wins_all/total*100:.1f}%  平均リターン: {df[df['strategy'] != 'breakout_v2']['return'].mean():+.2f}%")


def load_cache() -> dict | None:
    """当日キャッシュがあれば返す。"""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("date") == date.today().isoformat():
            return cached["data"]
    except Exception:
        pass
    return None


def save_cache(data: dict) -> None:
    """yfinanceデータを当日キャッシュとして保存する。"""
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"date": date.today().isoformat(), "data": data}, f)
        print(f"キャッシュ保存完了: {len(data)} 銘柄 → {CACHE_PATH.name}")
    except Exception as e:
        print(f"キャッシュ保存失敗: {e}")


if __name__ == "__main__":
    print("JPX銘柄リストを取得中...")
    tickers = fetch_jpx_stock_list()
    print(f"対象: {len(tickers)} 銘柄")

    # キャッシュ確認 → なければyfinanceから取得
    all_data = load_cache()
    if all_data:
        print(f"キャッシュからデータを読み込み: {len(all_data)} 銘柄（yfinanceスキップ）")
    else:
        BATCH = 100
        all_data = {}
        print(f"yfinanceから{PERIOD}分データを一括取得中（{len(tickers)//BATCH+1}バッチ）...")
        for i in range(0, len(tickers), BATCH):
            batch = tickers[i:i+BATCH]
            try:
                raw = yf.download(batch, period=PERIOD, auto_adjust=True,
                                  group_by="ticker", progress=False, threads=True)
                for t in batch:
                    try:
                        if len(batch) == 1:
                            df = raw
                        else:
                            df = raw[t] if t in raw.columns.get_level_values(0) else None
                        if df is not None and len(df) >= MIN_HISTORY:
                            all_data[t] = df.dropna(how="all")
                    except Exception:
                        pass
            except Exception as e:
                print(f"  バッチ{i//BATCH+1} エラー: {e}")
            print(f"  {min(i+BATCH, len(tickers))}/{len(tickers)} 完了  取得済み: {len(all_data)} 銘柄")
        save_cache(all_data)

    print(f"\nデータ取得完了: {len(all_data)} 銘柄  バックテスト開始...\n")

    all_signals: list[dict] = []
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(backtest_ticker, t, df): t for t, df in all_data.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_signals.extend(result)
            with lock:
                completed += 1
            if completed % 200 == 0 or completed == len(all_data):
                print(f"  進捗: {completed}/{len(all_data)} 銘柄  シグナル累計: {len(all_signals)} 件")

    print(f"\n完了。総シグナル数: {len(all_signals)} 件")
    RESULTS_PATH.write_text(
        json.dumps(all_signals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"結果保存: {RESULTS_PATH.name}")
    summarize(all_signals)
