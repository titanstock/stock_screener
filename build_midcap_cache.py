#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中型株バックテスト用キャッシュ構築
=====================================
時価総額 500億〜5000億円の銘柄を対象にキャッシュを作成する。
"""

import pickle, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_PATH     = Path(__file__).parent / "backtest_500to1000_5y_cache.pkl"
MIN_MARKET_CAP = 500  * 10**8   # 500億円
MAX_MARKET_CAP = 1000 * 10**8   # 1000億円
MIN_ROWS       = 100
PERIOD         = "5y"
BATCH_SIZE     = 100
MAX_WORKERS    = 30

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
                        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
                        if len(cols) == 5:
                            result[t] = df[cols]
                except Exception:
                    pass
        except Exception as e:
            print(f"  バッチエラー: {e}")
        done += len(batch)
        if done % 500 == 0 or done == total:
            print(f"  OHLCV: {done}/{total}  取得済み: {len(result)}銘柄")
    return result


def fetch_market_cap(ticker: str) -> tuple[str, float]:
    try:
        mc = yf.Ticker(ticker).fast_info.market_cap
        if mc and mc > 0:
            return ticker, float(mc)
    except Exception:
        pass
    return ticker, 0.0


def fetch_market_caps(tickers: list[str]) -> dict[str, float]:
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
    min_oku = MIN_MARKET_CAP / 10**8
    max_oku = MAX_MARKET_CAP / 10**8
    print("=" * 60)
    print(f"中型株キャッシュ構築")
    print(f"  時価総額範囲 : {min_oku:.0f}億〜{max_oku:.0f}億円")
    print(f"  取得期間     : {PERIOD}")
    print(f"  保存先       : {CACHE_PATH.name}")
    print("=" * 60)

    tickers    = fetch_jpx_tickers()
    print(f"\nOHLCV取得中...")
    ohlcv_data = fetch_ohlcv_batch(tickers)
    print(f"OHLCV取得完了: {len(ohlcv_data)}銘柄")

    print(f"\n時価総額取得中...")
    market_caps = fetch_market_caps(list(ohlcv_data.keys()))
    print(f"時価総額取得完了: {len(market_caps)}銘柄")

    print(f"\nフィルタリング（{min_oku:.0f}億〜{max_oku:.0f}億円）...")
    filtered = {}
    for t, df in ohlcv_data.items():
        mc = market_caps.get(t, 0.0)
        if MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP:
            filtered[t] = df

    caps_list = [market_caps[t]/10**8 for t in filtered if t in market_caps]
    print(f"  フィルター後 : {len(filtered)}銘柄")
    if caps_list:
        print(f"  時価総額 平均: {np.mean(caps_list):.0f}億  中央: {np.median(caps_list):.0f}億")

    cache = {"date": date.today().isoformat(), "data": filtered}
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"\nキャッシュ保存完了: {CACHE_PATH}")


if __name__ == "__main__":
    main()
