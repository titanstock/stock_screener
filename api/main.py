#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株スクリーナー FastAPI バックエンド
"""

import io
import json
import os
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests as _requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# stock_screener.py を親ディレクトリからインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
import stock_screener as ss

# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

app = FastAPI(title="株スクリーナー API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# バックグラウンドタスクの状態管理
_task_status: dict = {"screening": "idle", "backtest": "idle"}
_task_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# スキーマ
# ──────────────────────────────────────────────────────────────────────────────

class ScreeningResult(BaseModel):
    code: str
    name: Optional[str]
    close: float
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_capped: bool
    rsi: float
    macd: float
    macd_signal: float
    macd_dir: str


class DayResults(BaseModel):
    date: str
    baseline: list[ScreeningResult]
    breakout: list[ScreeningResult]
    pullback: list[ScreeningResult]
    performance: Optional[list[dict]] = None


class ParamsUpdate(BaseModel):
    breakout_days: Optional[int] = None
    vol_mult: Optional[float] = None
    vol_spike_mult: Optional[float] = None
    rsi_period: Optional[int] = None
    pullback_touch_pct: Optional[float] = None
    pullback_lookback: Optional[int] = None
    max_market_cap_oku: Optional[float] = None
    min_avg_turnover_man: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────────────────────

def _load_results(date_str: str) -> dict:
    path = RESULTS_DIR / f"{date_str}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{date_str} の結果が見つかりません")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_date_str() -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="スクリーニング結果がありません")
    return files[0].stem


# ──────────────────────────────────────────────────────────────────────────────
# エンドポイント: スクリーニング結果
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/results/latest", response_model=DayResults, tags=["results"])
def get_latest_results():
    """最新のスクリーニング結果を取得"""
    date_str = _latest_date_str()
    data = _load_results(date_str)
    return _format_results(data)


@app.get("/results/{date_str}", response_model=DayResults, tags=["results"])
def get_results_by_date(date_str: str):
    """指定日のスクリーニング結果を取得（例: 2026-03-15）"""
    data = _load_results(date_str)
    return _format_results(data)


@app.get("/results", tags=["results"])
def list_results():
    """結果が存在する日付一覧を返す"""
    RESULTS_DIR.mkdir(exist_ok=True)
    dates = sorted([f.stem for f in RESULTS_DIR.glob("*.json")], reverse=True)
    return {"dates": dates}


def _format_results(data: dict) -> dict:
    strategies = data.get("strategies", {})
    return {
        "date":        data.get("date", ""),
        "baseline":    strategies.get("baseline", []),
        "breakout":    strategies.get("breakout", []),
        "pullback":    strategies.get("pullback", []),
        "performance": data.get("performance"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 信用取引データ（JPX 日次公開データ）
# ──────────────────────────────────────────────────────────────────────────────

_margin_cache: dict = {"date": None, "data": None}
_JPX_MARGIN_BASE = "https://www.jpx.co.jp"
_JPX_MARGIN_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_margin_df() -> pd.DataFrame | None:
    """JPXの最新信用残高XLSを取得してDataFrameに変換。当日キャッシュ使用。"""
    today = date.today().isoformat()
    if _margin_cache["date"] == today and _margin_cache["data"] is not None:
        return _margin_cache["data"]

    try:
        r = _requests.get(_JPX_MARGIN_PAGE, timeout=10, headers=_HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if ".xls" in a["href"]]
        if not links:
            return None
        xls_url = _JPX_MARGIN_BASE + links[0]

        r2 = _requests.get(xls_url, timeout=20, headers=_HEADERS)
        df = pd.read_excel(io.BytesIO(r2.content), header=None)
        data = df.iloc[7:].copy()
        data.columns = range(len(data.columns))
        data = data[data[6].notna()]
        data[6] = data[6].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        # 数値列をキャスト
        for col in [8, 11, 14]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        _margin_cache["date"] = today
        _margin_cache["data"] = data
        return data
    except Exception:
        return None


def _get_margin_for_code(code: str) -> dict | None:
    """4桁コードから信用取引データを返す。"""
    df = _fetch_margin_df()
    if df is None:
        return None
    jpx_code = code + "0"  # JPXは5桁コード（4桁+'0'）
    row = df[df[6] == jpx_code]
    if row.empty:
        return None
    r = row.iloc[0]
    sell = r[8]   # 信用売り残（株数）
    buy  = r[11]  # 信用買い残（株数）
    ratio = r[14] # 信用倍率
    return {
        "short_balance": int(sell) if pd.notna(sell) else None,
        "long_balance":  int(buy)  if pd.notna(buy)  else None,
        "margin_ratio":  float(ratio) if pd.notna(ratio) else None,
        "gyaku_hibu":    None,  # 逆日歩: 公開APIなし
        "source":        "JPX日次",
    }


# ──────────────────────────────────────────────────────────────────────────────
# エンドポイント: 銘柄詳細
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/stock/{code}", tags=["stock"])
def get_stock_detail(code: str):
    """銘柄の最新スクリーニング詳細を取得"""
    date_str = _latest_date_str()
    data = _load_results(date_str)
    strategies = data.get("strategies", {})

    found = []
    for strategy_key, results in strategies.items():
        for r in results:
            if r.get("code") == code:
                found.append({"strategy": strategy_key, **r})

    if not found:
        raise HTTPException(status_code=404, detail=f"{code} は本日の結果に含まれていません")

    margin = _get_margin_for_code(code)
    return {"code": code, "date": date_str, "matches": found, "margin": margin}


# ──────────────────────────────────────────────────────────────────────────────
# エンドポイント: パラメータ
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/params", tags=["params"])
def get_params():
    """現在のスクリーニングパラメータを取得"""
    return {
        "breakout_days":        ss.BREAKOUT_DAYS,
        "vol_mult":             ss.VOL_MULT,
        "vol_spike_mult":       ss.VOL_SPIKE_MULT,
        "rsi_period":           ss.RSI_PERIOD,
        "pullback_touch_pct":   ss.PULLBACK_TOUCH_PCT,
        "pullback_lookback":    ss.PULLBACK_LOOKBACK,
        "max_market_cap_oku":   ss.MAX_MARKET_CAP_YEN / 1e8,
        "min_avg_turnover_man": ss.MIN_AVG_TURNOVER / 1e4,
        "strategies": {
            "baseline": {
                "entry": "終値×0.98",
                "rr": 1.5,
                "rsi_min": 50, "rsi_max": 70,
                "vol_mult": ss.VOL_MULT,
                "weekly_ma25_deviation_max": 1.2,
            },
            "breakout": {
                "entry": "翌日始値",
                "rr": 2.0,
                "rsi_min": 60,
                "vol_spike_mult": ss.VOL_SPIKE_MULT,
                "breakout_pct_min": 1.0,
                "change_pct_min": 2.0,
                "consolidation_atr_shrink": True,
                "consolidation_range": "96-102%",
                "consolidation_days": 2,
            },
            "pullback": {
                "entry": "日足MA25",
                "rr": 1.5,
                "rsi_min": 55, "rsi_max": 65,
                "touch_pct": ss.PULLBACK_TOUCH_PCT,
                "lookback_days": ss.PULLBACK_LOOKBACK,
            },
        },
    }


@app.patch("/params", tags=["params"])
def update_params(body: ParamsUpdate):
    """スクリーニングパラメータを変更（実行中のプロセスに即時反映）"""
    changed = {}
    if body.breakout_days is not None:
        ss.BREAKOUT_DAYS = body.breakout_days
        changed["breakout_days"] = body.breakout_days
    if body.vol_mult is not None:
        ss.VOL_MULT = body.vol_mult
        changed["vol_mult"] = body.vol_mult
    if body.vol_spike_mult is not None:
        ss.VOL_SPIKE_MULT = body.vol_spike_mult
        changed["vol_spike_mult"] = body.vol_spike_mult
    if body.pullback_touch_pct is not None:
        ss.PULLBACK_TOUCH_PCT = body.pullback_touch_pct
        changed["pullback_touch_pct"] = body.pullback_touch_pct
    if body.pullback_lookback is not None:
        ss.PULLBACK_LOOKBACK = body.pullback_lookback
        changed["pullback_lookback"] = body.pullback_lookback
    if body.max_market_cap_oku is not None:
        ss.MAX_MARKET_CAP_YEN = int(body.max_market_cap_oku * 1e8)
        changed["max_market_cap_yen"] = ss.MAX_MARKET_CAP_YEN
    if body.min_avg_turnover_man is not None:
        ss.MIN_AVG_TURNOVER = int(body.min_avg_turnover_man * 1e4)
        changed["min_avg_turnover"] = ss.MIN_AVG_TURNOVER

    if not changed:
        return {"message": "変更なし"}
    return {"message": "パラメータを更新しました", "changed": changed}


# ──────────────────────────────────────────────────────────────────────────────
# エンドポイント: スクリーニング実行
# ──────────────────────────────────────────────────────────────────────────────

def _run_screening_task():
    with _task_lock:
        _task_status["screening"] = "running"
    try:
        ss.run_screening()
    except Exception as e:
        _task_status["screening"] = f"error: {e}"
        return
    with _task_lock:
        _task_status["screening"] = "done"


@app.post("/screening/run", tags=["screening"])
def run_screening(background_tasks: BackgroundTasks):
    """スクリーニングをバックグラウンドで実行"""
    if _task_status["screening"] == "running":
        raise HTTPException(status_code=409, detail="スクリーニングは既に実行中です")
    background_tasks.add_task(_run_screening_task)
    return {"message": "スクリーニングを開始しました"}


@app.get("/screening/status", tags=["screening"])
def get_screening_status():
    """スクリーニングの実行状態を取得"""
    return {"status": _task_status["screening"]}


# ──────────────────────────────────────────────────────────────────────────────
# エンドポイント: バックテスト
# ──────────────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    period: str = "2y"


_backtest_result: dict = {}


def _run_backtest_task(period: str):
    global _backtest_result
    with _task_lock:
        _task_status["backtest"] = "running"
        _backtest_result = {}
    try:
        import backtest as bt
        import yfinance as yf
        import pandas as pd

        tickers = ss.fetch_jpx_stock_list()
        BATCH = 100
        all_data: dict[str, pd.DataFrame] = {}

        for i in range(0, len(tickers), BATCH):
            batch = tickers[i:i+BATCH]
            try:
                raw = yf.download(batch, period=period, auto_adjust=True,
                                  group_by="ticker", progress=False, threads=True)
                for t in batch:
                    try:
                        df = raw if len(batch) == 1 else (
                            raw[t] if t in raw.columns.get_level_values(0) else None)
                        if df is not None and len(df) >= bt.MIN_HISTORY:
                            all_data[t] = df.dropna(how="all")
                    except Exception:
                        pass
            except Exception:
                pass

        all_signals = []
        for t, df in all_data.items():
            all_signals.extend(bt.backtest_ticker(t, df))

        # 集計
        import pandas as pd
        df_s = pd.DataFrame(all_signals)
        summary = {}
        for sk in ["baseline", "breakout_v2", "pullback"]:
            s = df_s[df_s["strategy"] == sk] if not df_s.empty else pd.DataFrame()
            if s.empty:
                summary[sk] = {"n": 0}
                continue
            n = len(s)
            wins = int(s["win"].astype(bool).sum())
            avg_ret = float(s["return"].mean())
            stop_n = int(s["exit_reason"].eq("stop").sum())
            win_ret = float(s[s["win"].astype(bool)]["return"].mean()) if wins > 0 else 0
            loss_ret = float(s[~s["win"].astype(bool)]["return"].mean()) if (n-wins) > 0 else -1
            pf = abs(win_ret / loss_ret) if loss_ret != 0 else 999
            summary[sk] = {
                "n": n,
                "win_rate": round(wins / n * 100, 1),
                "avg_return": round(avg_ret, 2),
                "stop_rate": round(stop_n / n * 100, 1),
                "profit_factor": round(pf, 2),
                "avg_days": round(float(s["exit_day"].mean()), 1),
            }

        _backtest_result = {"period": period, "summary": summary, "signals": all_signals[:200]}
        _task_status["backtest"] = "done"
    except Exception as e:
        _task_status["backtest"] = f"error: {e}"


@app.post("/backtest/run", tags=["backtest"])
def run_backtest(body: BacktestRequest, background_tasks: BackgroundTasks):
    """バックテストをバックグラウンドで実行"""
    if _task_status["backtest"] == "running":
        raise HTTPException(status_code=409, detail="バックテストは既に実行中です")
    background_tasks.add_task(_run_backtest_task, body.period)
    return {"message": f"バックテスト（{body.period}）を開始しました"}


@app.get("/backtest/status", tags=["backtest"])
def get_backtest_status():
    """バックテストの状態と結果を取得"""
    return {
        "status": _task_status["backtest"],
        "result": _backtest_result if _task_status["backtest"] == "done" else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ヘルスチェック
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
