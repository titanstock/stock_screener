#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バックテスト用キャッシュ再構築
================================
時価総額フィルターを適用してバックテストキャッシュを作成する。

フロー:
  1. JPX銘柄リスト取得（プライム・スタンダード・グロース）
  2. yfinance で OHLCV 2年分を一括取得
  3. 各銘柄の時価総額を並列取得（yfinance fast_info）
  4. 時価総額 ≤ MAX_MARKET_CAP でフィルタリング
  5. backtest_cache.pkl として保存（上書き）

注意: 時価総額は現在の値を使用（過去の時価総額は取得不可）
     → 現在500億以下の銘柄を対象にバックテストする近似
"""

import os, pickle, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

CACHE_PATH      = Path(__file__).parent / "backtest_cache.pkl"
MAX_MARKET_CAP  = 500 * 10**8        # 500億円
MIN_ROWS        = 100                 # 最低日数
PERIOD          = "2y"
BATCH_SIZE      = 100                 # yfinance一括取得のバッチサイズ
MAX_WORKERS     = 30                  # 時価総額取得の並列数

JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)


def fetch_jpx_tickers() -> list[str]:
    print("JPX銘柄リスト取得中...")
    resp = requests.get(JPX_LIST_URL, timeout=30)
    resp.raise_for_status()
    df   = pd.read_excel(BytesIO(resp.content), dtype=str)
    mkt  = next((c for c in df.columns if "市場" in str(c)), None)
    code = next((c for c in df.columns if "コード" in str(c)), None)
    if not mkt or not code:
        raise RuntimeError("JPXリストのカラムが見つかりません")
    targets = ["プライム", "スタンダード", "グロース"]
    df = df[df[mkt].str.contains("|".join(targets), na=False)]
    codes = df[code].str.strip().tolist()
    tickers = [f"{c}.T" for c in codes if c.isdigit() and len(c) == 4]
    print(f"  JPX上場銘柄: {len(tickers)}件")
    return tickers


def fetch_ohlcv_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yfinanceで一括OHLCV取得"""
    result = {}
    total  = len(tickers)
    done   = 0

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            raw = yf.download(
                batch, period=PERIOD, auto_adjust=True,
                group_by="ticker", progress=False, threads=True,
            )
            for t in batch:
                try:
                    df = raw[t] if len(batch) > 1 else raw
                    if df is not None and len(df) >= MIN_ROWS:
                        df = df.dropna(how="all")
                        cols = [c for c in ["Open","High","Low","Close","Volume"]
                                if c in df.columns]
                        if len(cols) == 5:
                            result[t] = df[cols]
                except Exception:
                    pass
        except Exception as e:
            print(f"  バッチ{i//BATCH_SIZE+1} エラー: {e}")

        done += len(batch)
        if done % 500 == 0 or done == total:
            print(f"  OHLCV: {done}/{total}  取得済み: {len(result)}銘柄")

    return result


def fetch_market_cap(ticker: str) -> tuple[str, float]:
    """1銘柄の時価総額を取得"""
    try:
        mc = yf.Ticker(ticker).fast_info.market_cap
        if mc and mc > 0:
            return ticker, float(mc)
    except Exception:
        pass
    return ticker, 0.0


def fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    """時価総額を並列取得"""
    caps = {}
    total = len(tickers)
    done  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_market_cap, t): t for t in tickers}
        for fut in as_completed(futs):
            t, mc = fut.result()
            if mc > 0:
                caps[t] = mc
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  時価総額: {done}/{total}  取得済み: {len(caps)}銘柄")

    return caps


def main():
    today_str = date.today().isoformat()
    cap_oku   = MAX_MARKET_CAP / 10**8

    print("=" * 60)
    print(f"バックテストキャッシュ再構築")
    print(f"  時価総額上限 : {cap_oku:.0f}億円")
    print(f"  取得期間     : {PERIOD}")
    print(f"  最低日数     : {MIN_ROWS}日")
    print(f"  保存先       : {CACHE_PATH.name}")
    print("=" * 60)

    # ── Step 1: JPX銘柄リスト ──
    tickers = fetch_jpx_tickers()

    # ── Step 2: OHLCV一括取得 ──
    print(f"\nOHLCV取得中（{PERIOD}分 / バッチ{BATCH_SIZE}）...")
    ohlcv_data = fetch_ohlcv_batch(tickers)
    print(f"OHLCV取得完了: {len(ohlcv_data)}銘柄")

    # ── Step 3: 時価総額取得 ──
    print(f"\n時価総額取得中（並列{MAX_WORKERS}スレッド）...")
    valid_tickers = list(ohlcv_data.keys())
    market_caps   = fetch_market_caps(valid_tickers)
    print(f"時価総額取得完了: {len(market_caps)}銘柄")

    # ── Step 4: 時価総額フィルタリング ──
    print(f"\nフィルタリング（≤{cap_oku:.0f}億円）...")
    filtered = {}
    no_cap   = 0
    too_big  = 0

    for t, df in ohlcv_data.items():
        mc = market_caps.get(t, 0.0)
        if mc <= 0:
            no_cap += 1
            continue
        if mc > MAX_MARKET_CAP:
            too_big += 1
            continue
        filtered[t] = df

    print(f"  対象外（時価総額取得不可）: {no_cap}銘柄")
    print(f"  対象外（{cap_oku:.0f}億円超）: {too_big}銘柄")
    print(f"  フィルター後              : {len(filtered)}銘柄")

    # 時価総額分布
    filtered_caps = [market_caps[t]/10**8 for t in filtered if t in market_caps]
    if filtered_caps:
        print(f"\n  時価総額分布（億円）:")
        print(f"    平均: {np.mean(filtered_caps):.0f}億")
        print(f"    中央: {np.median(filtered_caps):.0f}億")
        print(f"    最小: {np.min(filtered_caps):.0f}億")
        print(f"    最大: {np.max(filtered_caps):.0f}億")
        for lo, hi in [(0,50),(50,100),(100,200),(200,300),(300,400),(400,500)]:
            cnt = sum(1 for x in filtered_caps if lo <= x < hi)
            print(f"    {lo:>3}〜{hi}億: {cnt}銘柄")

    # 日数分布
    lengths = [len(df) for df in filtered.values()]
    if lengths:
        print(f"\n  データ日数:")
        print(f"    平均: {np.mean(lengths):.0f}日  最小: {min(lengths)}日  最大: {max(lengths)}日")

    # ── Step 5: 保存 ──
    print(f"\nキャッシュ保存中...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": filtered}, f)
    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"保存完了: {CACHE_PATH.name}  ({size_mb:.1f} MB)  {len(filtered)}銘柄")
    print(f"\n完了 → バックテストスクリプトを再実行してください")


if __name__ == "__main__":
    main()
