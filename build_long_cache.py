#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
長期キャッシュ構築（5年分）
============================
J-Quants API（優先）+ yfinance（5年）で日足データを取得。
出力: backtest_cache_long.pkl
"""

import os, pickle, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from stock_screener import _get_jquants_id_token

CACHE_PATH    = Path(__file__).parent / "backtest_cache_long.pkl"
YEARS         = 5
JPX_LIST_URL  = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JQUANTS_BASE  = "https://api.jquants.com/v1"
JQUANTS_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN", "")
MAX_WORKERS   = 20
MIN_ROWS      = 200   # 最低200日必要（5年分なら約1250日）


def fetch_jpx_tickers() -> list[str]:
    resp = requests.get(JPX_LIST_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content), dtype=str)
    mkt  = next((c for c in df.columns if "市場" in str(c)), None)
    code = next((c for c in df.columns if "コード" in str(c)), None)
    targets = ["プライム", "スタンダード", "グロース"]
    df = df[df[mkt].str.contains("|".join(targets), na=False)]
    codes = df[code].str.strip().tolist()
    return [f"{c}.T" for c in codes if c.isdigit() and len(c) == 4]


def fetch_jquants(ticker: str, token: str, from_date: str) -> pd.DataFrame | None:
    code = ticker.replace(".T", "")
    try:
        r = requests.get(
            f"{JQUANTS_BASE}/prices/daily_quotes",
            params={"code": code, "from": from_date},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("daily_quotes", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        if len(cols) < 5:
            return None
        return df[cols].astype(float)
    except Exception:
        return None


def fetch_yfinance(ticker: str, from_date: str) -> pd.DataFrame | None:
    try:
        yf_df = yf.download(
            ticker, start=from_date, interval="1d",
            auto_adjust=True, progress=False, timeout=20,
        )
        if yf_df is None or len(yf_df) < MIN_ROWS:
            return None
        yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
        return yf_df[["Open","High","Low","Close","Volume"]]
    except Exception:
        return None


def fetch_one(ticker: str, token: str, from_date: str) -> tuple[str, pd.DataFrame | None]:
    # J-Quants 優先
    if token:
        df = fetch_jquants(ticker, token, from_date)
        if df is not None and len(df) >= MIN_ROWS:
            return ticker, df
    # yfinance フォールバック
    df = fetch_yfinance(ticker, from_date)
    return ticker, df


def main():
    today     = date.today()
    from_date = (today - timedelta(days=365 * YEARS + 30)).strftime("%Y-%m-%d")
    today_str = today.isoformat()

    print("=" * 60)
    print(f"長期キャッシュ構築（{YEARS}年分 / {from_date} 〜）")
    print("=" * 60)

    # 既存キャッシュ確認
    raw_data: dict[str, pd.DataFrame] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        cache_date = cached.get("date", "")
        if cache_date == today_str:
            print(f"当日キャッシュ利用: {len(cached['data'])} 銘柄")
            return
        print(f"既存キャッシュ（{cache_date}）を更新します")
        raw_data = cached.get("data", {})

    tickers = fetch_jpx_tickers()
    print(f"JPX上場銘柄: {len(tickers)} 件")

    # J-Quantsトークン取得
    token = ""
    if JQUANTS_TOKEN:
        try:
            token = _get_jquants_id_token(JQUANTS_TOKEN)
            print("J-Quants: トークン取得成功")
        except Exception as e:
            print(f"J-Quants: トークン取得失敗 → yfinance のみ使用 ({e})")

    # 取得実行
    to_fetch = tickers  # 全件再取得（5年分）
    print(f"\n取得開始: {len(to_fetch)} 銘柄（並列 {MAX_WORKERS} スレッド）")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, t, token, from_date): t for t in to_fetch}
        done = ok = 0
        for fut in as_completed(futs):
            t, df = fut.result()
            done += 1
            if df is not None and len(df) >= MIN_ROWS:
                raw_data[t] = df
                ok += 1
            if done % 200 == 0 or done == len(to_fetch):
                print(f"  {done}/{len(to_fetch)}  成功: {ok}  "
                      f"平均日数: {np.mean([len(v) for v in raw_data.values()]):.0f}日")

    # 日数分布確認
    lengths = [len(v) for v in raw_data.values()]
    print(f"\n取得完了: {len(raw_data)} 銘柄")
    print(f"  平均日数: {np.mean(lengths):.0f}日")
    print(f"  中央値  : {np.median(lengths):.0f}日")
    print(f"  最小    : {min(lengths)}日")
    print(f"  最大    : {max(lengths)}日")
    yr_bins = [(0,250),(250,500),(500,750),(750,1000),(1000,9999)]
    for lo, hi in yr_bins:
        cnt = sum(1 for l in lengths if lo <= l < hi)
        label = f"~{hi//250}年" if hi < 9999 else f"{lo//250}年+"
        print(f"  {label}: {cnt}銘柄")

    print(f"\nキャッシュ保存中...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print(f"保存完了: {CACHE_PATH.name}  ({CACHE_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
