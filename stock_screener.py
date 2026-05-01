#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本株スクリーニングツール
================================
対象市場  : スタンダード・グロース（内国株式）
時価総額  : 500 億円以下
戦略      : ②売られすぎ反発型 / NOA（ニッポン・オプティマライザー）
実行タイミング: 毎日 16:00（JST）自動実行 / --now オプションで即時実行
"""

import json
import os
import pickle
import re
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import requests
import schedule
import yfinance as yf
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
JQUANTS_REFRESH_TOKEN: str     = os.getenv("JQUANTS_REFRESH_TOKEN", "")
JQUANTS_API_BASE: str          = "https://api.jquants.com/v1"
LINE_USER_ID: str              = os.getenv("LINE_USER_ID", "")
LINE_PUSH_URL: str             = "https://api.line.me/v2/bot/message/push"
DISCORD_WEBHOOK_URL: str       = os.getenv("DISCORD_WEBHOOK_URL", "")
MAX_MARKET_CAP_YEN: int        = 500 * 10**8
LINE_MAX_CHARS: int            = 5_000   # LINE テキストメッセージ上限
TIMESTAMP_FMT: str             = "%Y/%m/%d %H:%M"

# スクリーニングパラメータ
BREAKOUT_DAYS: int      = 20    # ブレイクアウト・出来高判定期間
VOL_MULT: float         = 2.0   # 出来高条件（①②③）：20日平均の何倍
VOL_SPIKE_MULT: float   = 3.0   # 出来高条件（④）：20日平均の何倍
RSI_PERIOD: int         = 14
PULLBACK_TOUCH_PCT: float = 0.03  # MA25 タッチ判定の許容レンジ（終値がMA25の103%以内）
PULLBACK_LOOKBACK: int  = 2       # MA25 タッチを遡る日数
DOW_N_SWINGS: int       = 2       # 週足ダウ理論：切り上がりを確認する連続スイング数

# 流動性フィルター（全型共通・OR 条件）
MIN_AVG_TURNOVER: int = 30_000_000  # 20日平均売買代金の下限（円）

# 並列処理
MAX_WORKERS: int = 8

# 戦略定義（表示ラベル）
STRATEGIES: dict[str, str] = {
    "oversold_bounce":    "売られすぎ反発型",
    "noa":                "ニッポン・オプティマライザー（NOA）",
    "minervini":          "ミネルヴィニ SEPA型",
}

# LINE/Discord 通知対象戦略
NOTIFY_STRATEGIES: set[str] = {
    "oversold_bounce",
    "noa",
    "minervini",
}


# 売られすぎ反発型（件数型）専用パラメータ
# 条件: RSI14≤30 + 出来高1.5倍 + ATR拡大
# バックテスト実績: WR55.5% / PF1.62 / EV+2.5% / 9.88件/日（5年間）
OVERSOLD_BOUNCE_RSI_HI: float  = 30.0   # RSI14 上限
OVERSOLD_BOUNCE_VOL_MULT: float = 1.5   # 出来高 ≥ 20日平均の1.5倍
OVERSOLD_BOUNCE_RR: float      = 2.5    # リスクリワード比


# ニッポン・オプティマライザー（NOA）パラメータ
# 条件: RSI(30)≤30 + MACDがシグナル以下（下向き局面）
# バックテスト実績: WR57.2% / PF1.50 / EV+2.17% / 3.85件/日（5年間）
NOA_RSI_PERIOD: int   = 30
NOA_RSI_HI: float     = 30.0
NOA_RR: float         = 2.0
NOA_MAX_HOLD: int     = 10   # 最大保有日数（パフォーマンス追跡用）

# ミネルヴィニ SEPA型パラメータ
# 条件: パーフェクトオーダー + MA200上昇 + 52週安値+30% + 52週高値-25%以内 + ブレイクアウト + 出来高
# バックテスト実績: PF2.04 / EV+4.33% / 2.74件/日（5年間, 500〜5000億）
MINERVINI_BREAKOUT_DAYS: int  = 20     # ブレイクアウト判定期間（日）
MINERVINI_VOL_MULT: float     = 1.5    # 出来高 ≥ 20日平均の1.5倍
MINERVINI_STOP_PCT: float     = 0.10   # 初期損切り幅（エントリー比-10%）
MINERVINI_SLOPE_DAYS: int     = 20     # MA200上昇確認期間（日）
MINERVINI_MIN_RISE_FROM_LOW: float = 30.0  # 52週安値からの上昇率下限（%）
MINERVINI_NEAR_HIGH_PCT: float = 25.0  # 52週高値からの最大乖離（%）
MINERVINI_MIN_CAP: int        = 500  * 10**8   # 500億円
MINERVINI_MAX_CAP: int        = 5000 * 10**8   # 5000億円

# 結果保存ディレクトリ
RESULTS_DIR = Path(__file__).parent / "results"

# ポートフォリオ
PORTFOLIO_FILE     = Path(__file__).parent / "portfolio.json"
POSITION_MAX_DAYS  = 20   # 最大保有営業日数（超過で期間終了アラート）

# JPX 上場銘柄一覧 URL
JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

# ──────────────────────────────────────────────────────────────────────────────
# ロガー
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# JPX 銘柄リスト取得
# ──────────────────────────────────────────────────────────────────────────────
def fetch_jpx_stock_list() -> list[str]:
    logger.info("JPX 銘柄リストを取得中...")
    try:
        resp = requests.get(JPX_LIST_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(BytesIO(resp.content), dtype=str)
    except Exception as e:
        raise RuntimeError(f"JPX 銘柄リスト取得失敗: {e}") from e

    mkt_col  = _find_col(df, ["市場・商品区分", "市場区分"])
    code_col = _find_col(df, ["コード", "証券コード"])
    if not mkt_col or not code_col:
        raise RuntimeError(f"必要な列が見つかりません。取得列: {df.columns.tolist()}")

    mask = df[mkt_col].str.contains(
        r"(?=.*(?:スタンダード|グロース))(?=.*内国株式)", na=False, regex=True
    )
    tickers = (df[mask][code_col].str.strip().str.zfill(4) + ".T").tolist()
    logger.info(f"対象銘柄数: {len(tickers)} 件（スタンダード・グロース 内国株式）")
    return tickers


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    for name in candidates:
        matches = [c for c in df.columns if name in str(c)]
        if matches:
            return matches[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 時価総額取得
# ──────────────────────────────────────────────────────────────────────────────
def get_market_cap(stock: yf.Ticker) -> float:
    ticker = stock.ticker

    # ── キャッシュ確認 ──
    with _ohlcv_cache_lock:
        cached = _ohlcv_cache.get(ticker)
        if cached is not None and "market_cap" in cached:
            return cached["market_cap"]

    mc = 0.0
    try:
        val = stock.fast_info.market_cap
        if val and val > 0:
            mc = float(val)
    except Exception:
        pass

    # ── キャッシュ保存 ──
    if mc > 0:
        with _ohlcv_cache_lock:
            entry = _ohlcv_cache.setdefault(ticker, {})
            entry["market_cap"] = mc

    return mc


# ──────────────────────────────────────────────────────────────────────────────
# J-Quants API / データ取得
# ──────────────────────────────────────────────────────────────────────────────
_jq_id_token: str   = ""
_jq_token_expiry: float = 0.0
_jq_token_lock  = threading.Lock()
_JQ_TOKEN_CACHE = Path(__file__).parent / ".jq_token_cache.json"


def _get_jquants_id_token() -> str:
    """IDトークンを取得（23時間キャッシュ・ファイル永続化）"""
    global _jq_id_token, _jq_token_expiry
    with _jq_token_lock:
        # メモリキャッシュが有効なら即返す
        if _jq_id_token and time.time() < _jq_token_expiry:
            return _jq_id_token
        # ファイルキャッシュを確認
        try:
            cache = json.loads(_JQ_TOKEN_CACHE.read_text())
            if cache.get("token") and time.time() < cache.get("expiry", 0):
                _jq_id_token    = cache["token"]
                _jq_token_expiry = cache["expiry"]
                logger.debug("J-Quants IDトークン: ファイルキャッシュから再利用")
                return _jq_id_token
        except Exception:
            pass
        # 新規取得
        resp = requests.post(
            f"{JQUANTS_API_BASE}/token/auth_refresh",
            params={"refreshtoken": JQUANTS_REFRESH_TOKEN},
            timeout=10,
        )
        resp.raise_for_status()
        _jq_id_token    = resp.json()["idToken"]
        _jq_token_expiry = time.time() + 23 * 3600
        # ファイルに保存
        try:
            _JQ_TOKEN_CACHE.write_text(
                json.dumps({"token": _jq_id_token, "expiry": _jq_token_expiry})
            )
        except Exception as e:
            logger.warning(f"J-Quants トークンキャッシュ保存失敗: {e}")
        logger.debug("J-Quants IDトークン: 新規取得")
        return _jq_id_token


def _fetch_history_jquants(code4: str, days: int = 400) -> pd.DataFrame | None:
    """J-Quants APIから調整済みOHLCVを取得してyfinance互換DataFrameを返す"""
    try:
        id_token  = _get_jquants_id_token()
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        to_date   = datetime.now().strftime("%Y%m%d")
        resp = requests.get(
            f"{JQUANTS_API_BASE}/prices/daily_quotes",
            params={"code": code4, "from": from_date, "to": to_date},
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        records = resp.json().get("daily_quotes", [])
        if not records:
            return None
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize("Asia/Tokyo")
        df = df.set_index("Date").sort_index()
        df = df.rename(columns={
            "AdjustmentOpen":   "Open",
            "AdjustmentHigh":   "High",
            "AdjustmentLow":    "Low",
            "AdjustmentClose":  "Close",
            "AdjustmentVolume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df
    except Exception as e:
        logger.debug(f"J-Quants fetch failed ({code4}): {e}")
        return None


# ── OHLCV 日次キャッシュ ──────────────────────────────────────────────────────
# ── OHLCV + 時価総額 日次キャッシュ ──────────────────────────────────────────
# キャッシュ構造: ticker → {"df": DataFrame, "market_cap": float}
_OHLCV_CACHE_PATH = Path(__file__).parent / "ohlcv_cache.pkl"
_ohlcv_cache: dict[str, dict] = {}
_ohlcv_cache_date: str = ""
_ohlcv_cache_lock = threading.Lock()


def _load_ohlcv_cache(skip: bool = False) -> None:
    """当日付のキャッシュファイルがあればメモリに読み込む。skip=True のときはキャッシュを使わず常に最新を取得。"""
    global _ohlcv_cache, _ohlcv_cache_date
    today = date.today().isoformat()
    if skip:
        _ohlcv_cache      = {}
        _ohlcv_cache_date = today
        logger.info("OHLCVキャッシュをスキップ（最新データを取得）")
        return
    try:
        if _OHLCV_CACHE_PATH.exists():
            with open(_OHLCV_CACHE_PATH, "rb") as f:
                stored = pickle.load(f)
            if stored.get("date") == today:
                _ohlcv_cache      = stored.get("data", {})
                _ohlcv_cache_date = today
                logger.info(f"OHLCVキャッシュ読み込み: {len(_ohlcv_cache)} 銘柄")
                return
    except Exception as e:
        logger.warning(f"OHLCVキャッシュ読み込み失敗: {e}")
    _ohlcv_cache      = {}
    _ohlcv_cache_date = today


def _save_ohlcv_cache() -> None:
    """メモリキャッシュをファイルに書き出す。"""
    try:
        with open(_OHLCV_CACHE_PATH, "wb") as f:
            pickle.dump({"date": _ohlcv_cache_date, "data": _ohlcv_cache}, f)
        logger.info(f"OHLCVキャッシュ保存: {len(_ohlcv_cache)} 銘柄")
    except Exception as e:
        logger.warning(f"OHLCVキャッシュ保存失敗: {e}")


def fetch_history(ticker: str, days: int = 400) -> pd.DataFrame | None:
    """yfinanceメイン → J-QuantsフォールバックでOHLCVを取得する
    NOTE: J-Quantsのリフレッシュトークンが復旧次第、順序を元に戻す。
    """
    code4 = ticker.replace(".T", "")

    # ── キャッシュ確認（"df" キーがある場合のみヒット）──
    with _ohlcv_cache_lock:
        entry = _ohlcv_cache.get(ticker)
        if entry is not None and "df" in entry:
            return entry["df"]

    df: pd.DataFrame | None = None

    # ── メイン: yfinance ──
    try:
        period = "2y" if days >= 300 else "3mo"
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is not None and len(df) < 30:
            df = None
    except Exception as e:
        logger.debug(f"yfinance fetch failed ({ticker}): {e}")

    # ── フォールバック: J-Quants ──
    if df is None and JQUANTS_REFRESH_TOKEN:
        jq = _fetch_history_jquants(code4, days=days)
        if jq is not None and len(jq) >= 200:
            df = jq

    # ── キャッシュ保存（df のみ先に保存・market_cap は screen_ticker で追記）──
    if df is not None:
        with _ohlcv_cache_lock:
            entry = _ohlcv_cache.setdefault(ticker, {})
            entry["df"] = df

    return df


# ──────────────────────────────────────────────────────────────────────────────
# テクニカル指標
# ──────────────────────────────────────────────────────────────────────────────
def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    return 100 - 100 / (1 + avg_gain / avg_loss)


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def get_weekly_ma25(df: pd.DataFrame) -> float | None:
    try:
        weekly_close = df["Close"].resample("W").last().dropna()
        if len(weekly_close) < 25:
            return None
        val = float(weekly_close.rolling(25).mean().iloc[-1])
        return None if pd.isna(val) else val
    except Exception:
        return None


def _find_swings(series: pd.Series, cmp) -> list[float]:
    """前後 2 本との比較でスイングポイントを検出し、値のリストを返す。"""
    arr = series.to_numpy()
    return [
        float(arr[i])
        for i in range(2, len(arr) - 2)
        if (cmp(arr[i], arr[i-1]) and cmp(arr[i], arr[i-2])
            and cmp(arr[i], arr[i+1]) and cmp(arr[i], arr[i+2]))
    ]


def has_weekly_uptrend(df: pd.DataFrame) -> bool:
    """週足の高値・安値が直近 DOW_N_SWINGS 個連続して切り上がっているか。"""
    try:
        import operator
        weekly = df.resample("W").agg({"High": "max", "Low": "min"}).dropna()
        if len(weekly) < 5:
            return False
        highs = _find_swings(weekly["High"], operator.ge)
        lows  = _find_swings(weekly["Low"],  operator.le)
        if len(highs) < DOW_N_SWINGS or len(lows) < DOW_N_SWINGS:
            return False
        h = highs[-DOW_N_SWINGS:]
        l = lows[-DOW_N_SWINGS:]
        return (all(h[i] < h[i + 1] for i in range(DOW_N_SWINGS - 1)) and
                all(l[i] < l[i + 1] for i in range(DOW_N_SWINGS - 1)))
    except Exception:
        return False


def check_consolidation(df: pd.DataFrame, past_high: float) -> bool:
    """
    値固め検出（ブレイクアウト直前の状態を確認）
    ① ATR縮小: 直近5日のATR平均 < 前5日のATR平均
    ② 高値付近停滞: 終値が20日高値の96〜102%の範囲に2日以上（直前5日間）
    df["ATR"] が事前に計算されている必要がある。
    """
    if len(df) < 12 or "ATR" not in df.columns:
        return False

    # ① ATR縮小
    atr_window = df["ATR"].iloc[-11:-1]
    if len(atr_window) < 10 or atr_window.isna().any():
        return False
    atr_recent = atr_window.iloc[5:].mean()
    atr_prev   = atr_window.iloc[:5].mean()
    if atr_prev <= 0:
        return False
    atr_shrinking = atr_recent < atr_prev

    # ② 高値付近停滞（直前5日間で2日以上）
    close_window = df["Close"].iloc[-6:-1]
    near_high = (
        (close_window >= past_high * 0.96) &
        (close_window <= past_high * 1.02)
    ).sum()
    price_consolidating = near_high >= 2

    return atr_shrinking and price_consolidating


def evaluate_strategies(
    df: pd.DataFrame,
    close: float,
    prev_close: float,
    ma25_daily: float,
    ma25_weekly: float | None,
    avg_vol_20: float,
    vol_now: float,
    past_high_20: float,
    rsi: float,
    macd_now: float,
    sig_now: float,
    breakout_pct: float,
    change_pct: float,
) -> list[str]:
    """
    各戦略の条件を評価してマッチした戦略キーのリストを返す。
    stock_screener.py と backtest.py の両方から呼び出される唯一の条件定義。
    breakout は df["ATR"] が事前に計算されている必要がある。
    """
    above_daily        = close > ma25_daily
    above_weekly       = ma25_weekly is not None and close > ma25_weekly
    vol_15x            = avg_vol_20 > 0 and vol_now >= avg_vol_20 * 1.5
    vol_3x             = avg_vol_20 > 0 and vol_now >= avg_vol_20 * VOL_SPIKE_MULT
    weekly_uptrend     = has_weekly_uptrend(df)
    within_weekly_ma25 = ma25_weekly is None or close <= ma25_weekly * 1.2

    # 押し目買い：直近 PULLBACK_LOOKBACK 日以内に終値が MA25 の 100〜103% 以内
    touched_ma25 = any(
        not pd.isna(df["MA25"].iloc[i]) and
        float(df["MA25"].iloc[i]) <= float(df["Close"].iloc[i]) <= float(df["MA25"].iloc[i]) * 1.03
        for i in range(-PULLBACK_LOOKBACK, 0)
    )

    # ブレイクアウト：直近2日以内（3日前終値がまだ20日高値以下）
    recent_breakout = len(df) >= 3 and float(df["Close"].iloc[-3]) <= past_high_20

    matched: list[str] = []

    # ① ブレイクアウト/出来高急増型（値固め検出を含む）
    if (close > past_high_20 and breakout_pct >= 1.0 and vol_3x
            and change_pct >= 2.0 and above_daily and above_weekly
            and weekly_uptrend and rsi >= 60.0 and recent_breakout
            and check_consolidation(df, past_high_20)):
        matched.append("breakout")

    return matched


# ──────────────────────────────────────────────────────────────────────────────
# 1 銘柄スクリーニング（4戦略を一括チェック）
# ──────────────────────────────────────────────────────────────────────────────
def screen_ticker(ticker: str) -> dict[str, dict] | None:
    """
    データを 1 回取得して 4 戦略すべてをチェック。
    マッチした戦略のみ {strategy_key: result_dict} で返す。
    何もマッチしなければ None。
    """
    try:
        stock = yf.Ticker(ticker)

        market_cap = get_market_cap(stock)
        if market_cap <= 0:
            return None
        # 既存戦略は500億以下、ミネルヴィニは500〜5000億を対象
        in_small   = market_cap <= MAX_MARKET_CAP_YEN
        in_minerv  = MINERVINI_MIN_CAP <= market_cap <= MINERVINI_MAX_CAP
        if not in_small and not in_minerv:
            return None

        df = fetch_history(ticker, days=500)
        if df is None or len(df) < BREAKOUT_DAYS + 5:
            return None

        # ── 共通指標を計算 ──
        close      = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        vol_now    = float(df["Volume"].iloc[-1])
        vol_prev   = float(df["Volume"].iloc[-2])

        df["MA25"] = df["Close"].rolling(25).mean()
        ma25_daily = float(df["MA25"].iloc[-1])
        if pd.isna(ma25_daily):
            return None

        # ATR 系列（値固め検出に使用）
        _tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"]  - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = _tr.rolling(14).mean()

        ma25_weekly  = get_weekly_ma25(df)
        n            = BREAKOUT_DAYS + 1
        avg_vol_20   = float(df["Volume"].iloc[-n:-1].mean())
        past_high_20 = float(df["High"].iloc[-n:-1].max())

        # ── 流動性フィルター（全型共通）：平均出来高 OR 平均売買代金 ──
        avg_turnover_20 = float((df["Close"].iloc[-n:-1] * df["Volume"].iloc[-n:-1]).mean())
        if avg_turnover_20 < MIN_AVG_TURNOVER:
            return None

        atr = calc_atr(df)
        if pd.isna(atr) or atr <= 0:
            return None

        rsi = float(calc_rsi(df["Close"]).iloc[-1])
        if pd.isna(rsi):
            return None

        rsi30 = float(calc_rsi(df["Close"], period=NOA_RSI_PERIOD).iloc[-1])

        macd_s, sig_s = calc_macd(df["Close"])
        macd_now = float(macd_s.iloc[-1])
        sig_now  = float(sig_s.iloc[-1])
        macd_dir = "up" if macd_now > sig_now else "down"

        vol_ratio    = vol_now / vol_prev   if vol_prev   > 0 else 0.0
        vol_20x      = vol_now / avg_vol_20 if avg_vol_20 > 0 else 0.0
        breakout_pct = (close - past_high_20) / past_high_20 * 100 if past_high_20 > 0 else 0.0
        change_pct   = (close - prev_close)   / prev_close   * 100 if prev_close   > 0 else 0.0

        # ── 新戦略用追加計算 ──
        open_now = float(df["Open"].iloc[-1])
        high_now = float(df["High"].iloc[-1])
        low_now  = float(df["Low"].iloc[-1])

        # 連続陰線カウント（当日含む）
        consec_bear_count = 0
        for _i in range(-1, -len(df)-1, -1):
            if float(df["Close"].iloc[_i]) < float(df["Open"].iloc[_i]):
                consec_bear_count += 1
            else:
                break

        # 下ヒゲ比率
        _range = high_now - low_now
        _lower_shadow = min(open_now, close) - low_now
        lower_shadow_pct = (_lower_shadow / _range * 100) if _range > 0 else 0.0

        # MA25乖離率
        ma25_dev_pct = ((close - ma25_daily) / ma25_daily * 100) if ma25_daily > 0 else 0.0

        # ── ミネルヴィニ用 MA50/150/200 計算 ──
        ma50  = float(df["Close"].rolling(50).mean().iloc[-1])
        ma150 = float(df["Close"].rolling(150).mean().iloc[-1])
        ma200 = float(df["Close"].rolling(200).mean().iloc[-1])
        ma200_prev = float(df["Close"].rolling(200).mean().iloc[-1 - MINERVINI_SLOPE_DAYS]) \
            if len(df) > 200 + MINERVINI_SLOPE_DAYS else float("nan")
        wk52_lo = float(df["Low"].iloc[-252:].min())
        wk52_hi = float(df["High"].iloc[-252:].max())
        minerv_breakout_hi = float(df["High"].iloc[-MINERVINI_BREAKOUT_DAYS - 1:-1].max())

        matched_keys = evaluate_strategies(
            df          = df,
            close       = close,
            prev_close  = prev_close,
            ma25_daily  = ma25_daily,
            ma25_weekly = ma25_weekly,
            avg_vol_20  = avg_vol_20,
            vol_now     = vol_now,
            past_high_20= past_high_20,
            rsi         = rsi,
            macd_now    = macd_now,
            sig_now     = sig_now,
            breakout_pct= breakout_pct,
            change_pct  = change_pct,
        )

        base = {
            "code":               ticker.replace(".T", ""),
            "name":               None,
            "close":              close,
            "prev_close":         prev_close,
            "market_cap":         market_cap,
            "ma25_daily":         ma25_daily,
            "ma25_weekly":        ma25_weekly,
            "atr":                atr,
            "rsi":                rsi,
            "macd":               macd_now,
            "macd_signal":        sig_now,
            "macd_dir":           macd_dir,
            "vol_ratio":          vol_ratio,
            "vol_20x":            vol_20x,
            "past_high_20":       past_high_20,
            "breakout_pct":       breakout_pct,
            "change_pct":         change_pct,
            "consec_bear_count":  consec_bear_count,
            "lower_shadow_pct":   lower_shadow_pct,
            "ma25_dev_pct":       ma25_dev_pct,
            "rsi30":              rsi30,
        }

        matched: dict[str, dict] = {sk: base.copy() for sk in matched_keys}

        # ── 売られすぎ反発型（件数型）: RSI14≤30 + 出来高1.5倍 + ATR拡大 ──
        if in_small:
            atr3d_now  = float(df["ATR"].iloc[-3:].mean())
            atr3d_prev = float(df["ATR"].iloc[-6:-3].mean())
            atr_expand = (
                not pd.isna(atr3d_now) and not pd.isna(atr3d_prev)
                and atr3d_prev > 0 and atr3d_now > atr3d_prev
            )
            if rsi <= OVERSOLD_BOUNCE_RSI_HI and vol_20x >= OVERSOLD_BOUNCE_VOL_MULT and atr_expand:
                matched["oversold_bounce"] = base.copy()

        # ── ニッポン・オプティマライザー（NOA）: RSI(30)≤30 + MACD下向き ──
        if (in_small
                and not pd.isna(rsi30)
                and rsi30 <= NOA_RSI_HI
                and macd_now < sig_now):
            matched["noa"] = base.copy()

        # ── ミネルヴィニ SEPA型: パーフェクトオーダー + ブレイクアウト ──
        if (in_minerv
                and not any(pd.isna(v) for v in [ma50, ma150, ma200, ma200_prev])
                and close > ma50 > ma150 > ma200
                and ma200 > ma200_prev
                and close >= wk52_lo * (1 + MINERVINI_MIN_RISE_FROM_LOW / 100)
                and close >= wk52_hi * (1 - MINERVINI_NEAR_HIGH_PCT / 100)
                and close > minerv_breakout_hi
                and vol_20x >= MINERVINI_VOL_MULT
                and len(df) >= 41):
            # 値固め確認①: ベース前半(40〜20日前)より収縮フェーズ(直近20日)の出来高が30%以上減少
            base_vol   = float(df["Volume"].iloc[-40:-20].mean())
            shrink_vol = float(df["Volume"].iloc[-20:].mean())
            volume_drying = base_vol > 0 and shrink_vol < base_vol * 0.8

            # 値固め確認②: 収縮フェーズのレンジ < ベース前半のレンジ
            pre_hi  = float(df["High"].iloc[-40:-20].max())
            pre_lo  = float(df["Low"].iloc[-40:-20].min())
            cons_hi = float(df["High"].iloc[-20:-1].max())
            cons_lo = float(df["Low"].iloc[-20:-1].min())
            range_contracting = (pre_hi - pre_lo) > 0 and (cons_hi - cons_lo) < (pre_hi - pre_lo)

            if volume_drying and range_contracting:
                matched["minervini"] = base.copy()

        if not matched:
            return None

        # ── 戦略ごとの買値目安・損切り・利確を付与 ──
        entry_prices = {
            "oversold_bounce":    close,
            "noa":                close,
        }
        rr_ratio = {
            "oversold_bounce":    OVERSOLD_BOUNCE_RR,
            "noa":                NOA_RR,
        }
        for sk in list(matched.keys()):
            if sk == "minervini":
                # 損切り = 収縮フェーズ(直近20日)の最安値の1%下、最大-15%でキャップ
                entry       = close
                consol_lo   = float(df["Low"].iloc[-20:-1].min())
                stop        = consol_lo * 0.99
                stop        = max(stop, entry * (1 - MINERVINI_STOP_PCT * 1.5))  # 最大-15%
                matched[sk]["entry_price"]  = entry
                matched[sk]["stop_loss"]    = stop
                matched[sk]["stop_capped"]  = stop == entry * (1 - MINERVINI_STOP_PCT * 1.5)
                matched[sk]["take_profit"]  = None  # トレーリングストップで管理
                continue
            entry     = entry_prices[sk]
            stop_atr  = entry - atr * 2.0
            stop_cap  = entry * 0.90          # 上限: -10%
            stop      = max(stop_atr, stop_cap)
            take      = entry + (entry - stop) * rr_ratio[sk]
            matched[sk]["entry_price"]  = entry
            matched[sk]["stop_loss"]    = stop
            matched[sk]["stop_capped"]  = stop > stop_atr   # 上限適用フラグ
            matched[sk]["take_profit"]  = take

        # 銘柄名はマッチした場合のみ取得
        try:
            info = stock.info or {}
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            name = ticker

        for v in matched.values():
            v["name"] = name

        return matched

    except Exception as e:
        logger.debug(f"{ticker}: スキップ ({e})")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# LINE Messaging API 通知
# ──────────────────────────────────────────────────────────────────────────────
def send_line_notify(message: str) -> bool:
    """LINE Messaging API の push message で送信する。5000文字超は分割して送る。"""
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        return False
    chunks  = [message[i:i+LINE_MAX_CHARS] for i in range(0, len(message), LINE_MAX_CHARS)]
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type":  "application/json",
    }
    success = True

    for i, chunk in enumerate(chunks):
        payload = {
            "to":       LINE_USER_ID,
            "messages": [{"type": "text", "text": chunk}],
        }
        try:
            r = requests.post(LINE_PUSH_URL, json=payload, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.error(f"LINE API エラー: {r.status_code} / {r.text}")
                success = False
        except Exception as e:
            logger.error(f"LINE 送信失敗: {e}")
            success = False
        if i < len(chunks) - 1:
            time.sleep(0.5)

    return success


def send_discord_notify(message: str) -> bool:
    """Discord Webhook にメッセージを送信する。2000文字超は分割して送る。"""
    if not DISCORD_WEBHOOK_URL:
        return False
    chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
    success = True
    for i, chunk in enumerate(chunks):
        try:
            r = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=10)
            if r.status_code not in (200, 204):
                logger.error(f"Discord API エラー: {r.status_code} / {r.text}")
                success = False
        except Exception as e:
            logger.error(f"Discord 送信失敗: {e}")
            success = False
        if i < len(chunks) - 1:
            time.sleep(0.5)
    return success


def send_notify(message: str) -> None:
    """LINE と Discord に同時送信する。"""
    send_line_notify(message)
    send_discord_notify(message)


def _now_str() -> str:
    return datetime.now().strftime(TIMESTAMP_FMT)


def _fmt_wma(val: float | None) -> str:
    return f"{val:,.1f}" if val is not None else "N/A"


def build_message(
    strategy_key: str,
    strategy_label: str,
    results: list[dict],
    shinyo_map: dict[str, dict | None] | None = None,
) -> str:
    now    = _now_str()
    header = f"【{strategy_label}】 {now}"

    if not results:
        return f"{header}\n条件を満たした銘柄はありませんでした。"

    # ── ランキングブロック（スコアルールが定義された戦略のみ）──
    ranking_block = build_ranking_block(
        strategy_key, strategy_label, results, shinyo_map or {}
    )

    # ── 詳細ブロックの並び順 ──
    if strategy_key == "breakout":
        results = sorted(results, key=lambda r: r["vol_20x"], reverse=True)
    elif strategy_key == "oversold_bounce":
        results = sorted(results, key=lambda r: r["change_pct"], reverse=True)
    else:
        results = sorted(results, key=lambda r: r["rsi"], reverse=True)

    detail_header = f"\n{'─' * 24}\n【詳細】" if ranking_block else ""
    lines = (
        ([ranking_block, detail_header] if ranking_block else [])
        + [header, f"該当銘柄: {len(results)} 件", "─" * 24]
    )

    for r in results:
        cap_oku  = r["market_cap"] / 1e8
        macd_arrow = "↑" if r['macd_dir'] == "up" else "↓"
        macd_str = f"{macd_arrow} {r['macd']:.3f} / SIG {r['macd_signal']:.3f}"
        vol_prev = f"{r['vol_ratio']:.2f} 倍" if r["vol_ratio"] > 0 else "N/A"

        lines += ["", f"▶ {r['code']}  {r['name']}",
                  f"  終値      : {r['close']:>8,.0f} 円",
                  f"  時価総額  : {cap_oku:>6.0f} 億円"]

        shinyo = (shinyo_map or {}).get(r["code"])
        shinyo_lines = _fmt_shinyo_block(shinyo)

        if strategy_key == "breakout":
            lines += [
                f"  20日高値  : {r['past_high_20']:>8,.0f} 円  突破 +{r['breakout_pct']:.2f}%",
                f"  前日比    : {r['change_pct']:>+7.2f}%",
                f"  出来高20比: {r['vol_20x']:.2f} 倍",
                f"  日足MA25  : {r['ma25_daily']:>8,.1f}",
                f"  週足MA25  : {_fmt_wma(r['ma25_weekly']):>8}",
                f"  [参考] RSI     : {r['rsi']:.1f}",
                f"  [参考] MACD    : {macd_str}",
                f"  [参考] 出来高前比: {vol_prev}",
                f"  ※始値目安 : {r['entry_price']:>8,.0f} 円（翌日始値で成行）",
                f"  損切り    : {r['stop_loss']:>8,.0f} 円（ATR×2.0 / {(r['stop_loss']-r['entry_price'])/r['entry_price']*100:.1f}%）",
                f"  利確目安  : {r['take_profit']:>8,.0f} 円（{(r['take_profit']-r['entry_price'])/r['entry_price']*100:+.1f}%）",
                f"  リスクリワード: 1:2",
            ] + shinyo_lines
        elif strategy_key == "oversold_bounce":
            lines += [
                f"  RSI(14)   : {r['rsi']:.1f}",
                f"  出来高20比: {r['vol_20x']:.2f} 倍",
                f"  前日比    : {r['change_pct']:>+7.2f}%",
                f"  日足MA25  : {r['ma25_daily']:>8,.1f}",
                f"  [参考] ATR: {r['atr']:.1f}",
                f"  ※始値目安 : {r['entry_price']:>8,.0f} 円（翌日始値で成行）",
                f"  損切り    : {r['stop_loss']:>8,.0f} 円（ATR×2.0 / {(r['stop_loss']-r['entry_price'])/r['entry_price']*100:.1f}%）",
                f"  利確目安  : {r['take_profit']:>8,.0f} 円（{(r['take_profit']-r['entry_price'])/r['entry_price']*100:+.1f}%）",
                f"  リスクリワード: 1:{OVERSOLD_BOUNCE_RR}",
            ] + shinyo_lines
        elif strategy_key == "noa":
            lines += [
                f"  RSI(30)   : {r.get('rsi30', float('nan')):.1f}",
                f"  MACD      : {r['macd']:.3f}（シグナル: {r['macd_signal']:.3f}）",
                f"  前日比    : {r['change_pct']:>+7.2f}%",
                f"  日足MA25  : {r['ma25_daily']:>8,.1f}",
                f"  ※始値目安 : {r['entry_price']:>8,.0f} 円（翌日始値で成行）",
                f"  損切り    : {r['stop_loss']:>8,.0f} 円（ATR×2.0 / {(r['stop_loss']-r['entry_price'])/r['entry_price']*100:.1f}%）",
                f"  利確目安  : {r['take_profit']:>8,.0f} 円（{(r['take_profit']-r['entry_price'])/r['entry_price']*100:+.1f}%）",
                f"  リスクリワード: 1:{NOA_RR}",
            ] + shinyo_lines
        elif strategy_key == "minervini":
            stop_pct_actual = (r['stop_loss'] - r['entry_price']) / r['entry_price'] * 100
            lines += [
                f"  ブレイクアウト: {r['breakout_pct']:>+.2f}%（20日高値比）",
                f"  出来高20比: {r['vol_20x']:.2f} 倍",
                f"  前日比    : {r['change_pct']:>+7.2f}%",
                f"  RSI(14)   : {r['rsi']:.1f}",
                f"  ※始値目安 : {r['entry_price']:>8,.0f} 円（翌日始値で成行）",
                f"  損切り    : {r['stop_loss']:>8,.0f} 円（収縮安値-1% / {stop_pct_actual:.1f}%）"
                + ("  ※上限キャップ適用" if r.get("stop_capped") else ""),
                f"  利確      : トレーリングストップ20%で管理（利確ライン設定なし）",
            ] + shinyo_lines

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# スクリーニング結果の保存・読み込み
# ──────────────────────────────────────────────────────────────────────────────
def save_results(all_results: dict[str, list[dict]], run_date: date) -> None:
    """スクリーニング結果を JSON ファイルに保存する。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run_date.isoformat()}.json"

    serializable: dict[str, list[dict]] = {}
    for strategy_key, results in all_results.items():
        serializable[strategy_key] = [
            {
                "code":         r["code"],
                "name":         r["name"],
                "close":        r["close"],
                "entry_price":  r["entry_price"],
                "stop_loss":    r["stop_loss"],
                "take_profit":  r["take_profit"],
                "stop_capped":  r["stop_capped"],
                "rsi":          r["rsi"],
                "rsi30":        r.get("rsi30"),
                "macd":         r["macd"],
                "macd_signal":  r["macd_signal"],
                "macd_dir":     r["macd_dir"],
            }
            for r in results
        ]

    path.write_text(
        json.dumps({"date": run_date.isoformat(), "strategies": serializable},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"スクリーニング結果を保存: {path.name}")


def save_performance(performances: list[dict], prev_date_str: str) -> None:
    """前日JSONにパフォーマンス追跡結果を書き込む。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{prev_date_str}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["performance"] = [
            {
                "code":         p["code"],
                "name":         p["name"],
                "entry_price":  p["entry_price"],
                "stop_loss":    p["stop_loss"],
                "take_profit":  p["take_profit"],
                "close_prev":   p["close_prev"],
                "close_now":    p["close_now"],
                "change_pct":   p["change_pct"],
                "hit_stop":     p["hit_stop"],
                "hit_take":     p["hit_take"],
                "strategies":   p["strategies"],
            }
            for p in performances
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"パフォーマンスを保存: {path.name} ({len(performances)} 件)")
    except Exception as e:
        logger.warning(f"パフォーマンス保存失敗: {e}")


def load_prev_results() -> dict | None:
    """直近の（今日より前の）スクリーニング結果 JSON を返す。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    today_str = date.today().isoformat()
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if path.stem == today_str:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"前日結果を読み込み: {path.name}")
            return data
        except Exception as e:
            logger.warning(f"前日結果の読み込み失敗: {path.name} ({e})")
    return None


def load_recent_results(max_hold: int = 20) -> list[dict]:
    """
    過去 max_hold 営業日以内のスクリーニング結果を全て読み込む。
    各エントリーに signal_date を付与して返す。
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    today = date.today()
    today_str = today.isoformat()
    collected: list[dict] = []

    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if path.stem == today_str:
            continue
        try:
            signal_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        hold_days = count_business_days(path.stem)
        if hold_days > max_hold:
            break   # 古い順に並んでいるので以降はすべて期限超え
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            collected.append({"signal_date": path.stem, "data": data})
            logger.debug(f"追跡対象: {path.name} (経過{hold_days}営業日)")
        except Exception as e:
            logger.warning(f"結果の読み込み失敗: {path.name} ({e})")

    logger.info(f"追跡対象ファイル: {len(collected)} 件（過去{max_hold}営業日以内）")
    return collected


# ──────────────────────────────────────────────────────────────────────────────
# 前日パフォーマンス分析
# ──────────────────────────────────────────────────────────────────────────────
def fetch_stock_performance(entry: dict) -> dict | None:
    """
    スクリーニング結果の 1 銘柄について現在のパフォーマンスを計算する。
    entry: save_results で保存した 1 銘柄分のdict に "strategies"・"signal_date"・"hold_days" を追加したもの。
    """
    ticker = entry["code"] + ".T"
    try:
        df = fetch_history(ticker, days=100)
        if df is None or len(df) < 2:
            return None

        close_now   = float(df["Close"].iloc[-1])
        close_prev  = float(entry["close"])
        entry_price = float(entry.get("entry_price") or close_prev)
        change_pct  = (close_now - entry_price) / entry_price * 100 if entry_price > 0 else 0.0

        rsi_now = float(calc_rsi(df["Close"]).iloc[-1])
        macd_s, sig_s = calc_macd(df["Close"])
        macd_now    = float(macd_s.iloc[-1])
        sig_now     = float(sig_s.iloc[-1])
        macd_dir_now = "up" if macd_now > sig_now else "down"

        stop_loss   = entry.get("stop_loss")
        take_profit = entry.get("take_profit")
        hit_stop    = stop_loss   is not None and close_now <= stop_loss
        hit_take    = take_profit is not None and close_now >= take_profit

        return {
            "code":             entry["code"],
            "name":             entry["name"],
            "signal_date":      entry.get("signal_date", ""),
            "hold_days":        entry.get("hold_days", 1),
            "close_prev":       close_prev,
            "entry_price":      entry_price,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "close_now":        close_now,
            "change_pct":       change_pct,
            "hit_stop":         hit_stop,
            "hit_take":         hit_take,
            "rsi_prev":         float(entry["rsi"]),
            "rsi_now":          rsi_now,
            "macd_prev":        float(entry["macd"]),
            "macd_signal_prev": float(entry["macd_signal"]),
            "macd_dir_prev":    entry["macd_dir"],
            "macd_now":         macd_now,
            "macd_signal_now":  sig_now,
            "macd_dir_now":     macd_dir_now,
            "strategies":       entry.get("strategies", []),
        }
    except Exception as e:
        logger.debug(f"{ticker}: パフォーマンス取得失敗 ({e})")
        return None


def _perf_comment(change_pct: float, rsi_prev: float, rsi_now: float,
                  macd_dir_prev: str, macd_dir_now: str) -> str:
    """価格変動・RSI・MACD の変化から日本語コメントを生成する。"""
    if change_pct >= 5.0:
        trend = "急騰継続"
    elif change_pct >= 1.0:
        trend = "上昇継続"
    elif change_pct >= -1.0:
        trend = "横ばい"
    elif change_pct >= -5.0:
        trend = "調整入り"
    else:
        trend = "急落"

    notes: list[str] = []
    if rsi_now >= 70:
        notes.append(f"RSI過熱圏({rsi_now:.0f})")
    elif rsi_now <= 30:
        notes.append(f"RSI売られ過ぎ({rsi_now:.0f})")
    elif rsi_now - rsi_prev >= 5:
        notes.append(f"RSI上昇({rsi_prev:.0f}→{rsi_now:.0f})")
    elif rsi_prev - rsi_now >= 5:
        notes.append(f"RSI低下({rsi_prev:.0f}→{rsi_now:.0f})")

    if macd_dir_prev == "up" and macd_dir_now == "down":
        notes.append("MACDデッドクロス")
    elif macd_dir_prev == "down" and macd_dir_now == "up":
        notes.append("MACDゴールデンクロス")

    return trend + (" / " + " / ".join(notes) if notes else "")


def build_performance_message(prev_date_str: str, performances: list[dict]) -> str:
    """追跡銘柄パフォーマンス LINE メッセージを生成する（最大20営業日追跡）。"""
    now    = _now_str()
    header = f"【保有追跡パフォーマンス】(最大20営業日) {now}"

    if not performances:
        return f"{header}\n追跡銘柄のデータを取得できませんでした。"

    performances = sorted(performances, key=lambda p: p["change_pct"], reverse=True)
    lines = [header, f"追跡銘柄: {len(performances)} 件", "─" * 24]

    for p in performances:
        comment      = _perf_comment(p["change_pct"], p["rsi_prev"], p["rsi_now"],
                                     p["macd_dir_prev"], p["macd_dir_now"])
        strategy_str = " / ".join(p["strategies"])
        hold_days    = p.get("hold_days", 1)
        sig_date     = p.get("signal_date", prev_date_str)

        alert = ""
        if p.get("hit_take"):
            alert = " ★利確到達"
        elif p.get("hit_stop"):
            alert = " ▼損切到達"

        entry  = p.get("entry_price")
        stop   = p.get("stop_loss")
        take   = p.get("take_profit")

        block = [
            "",
            f"▶ {p['code']}  {p['name']}{alert}",
            f"  シグナル日: {sig_date} ({hold_days}営業日経過)",
            f"  買値目安  : {entry:>8,.0f} 円" if entry else "",
            f"  現在値    : {p['close_now']:>8,.0f} 円  ({p['change_pct']:>+.2f}%)",
            f"  損切り    : {stop:>8,.0f} 円" if stop else "",
            f"  利確目安  : {take:>8,.0f} 円" if take else "",
            f"  評価      : {comment}",
            f"  RSI       : {p['rsi_prev']:.1f} → {p['rsi_now']:.1f}",
            f"  MACD方向  : {'↑' if p['macd_dir_prev'] == 'up' else '↓'} → {'↑' if p['macd_dir_now'] == 'up' else '↓'}",
            f"  戦略      : {strategy_str}",
        ]
        lines += [l for l in block if l != ""]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 需給情報取得（日本証券金融 taisyaku.jp CSV）
# ──────────────────────────────────────────────────────────────────────────────
_TAISYAKU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.taisyaku.jp/",
}
_ZANDAKA_URL = "https://www.taisyaku.jp/data/zandaka.csv"
_SHINA_URL   = "https://www.taisyaku.jp/data/shina.csv"

# CSVは営業日ごと更新なので当日1回だけ取得してキャッシュ
_taisyaku_cache: dict[str, pd.DataFrame | None] = {}   # "zandaka" / "shina"
_taisyaku_cache_date: str = ""


def _load_taisyaku_csvs() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """zandaka.csv と shina.csv を当日1回だけ取得してキャッシュする。"""
    global _taisyaku_cache_date

    today_str = date.today().isoformat()
    if _taisyaku_cache_date == today_str and _taisyaku_cache:
        return _taisyaku_cache.get("zandaka"), _taisyaku_cache.get("shina")

    _taisyaku_cache.clear()
    _taisyaku_cache_date = today_str

    # ── 信用残（zandaka.csv）──
    try:
        r = requests.get(_ZANDAKA_URL, headers=_TAISYAKU_HEADERS, timeout=30)
        r.raise_for_status()
        from io import StringIO as _StringIO
        df_z = pd.read_csv(_StringIO(r.content.decode("shift_jis", errors="replace")))
        df_z["銘柄コード"] = df_z["銘柄コード"].astype(str).str.strip()
        _taisyaku_cache["zandaka"] = df_z
        logger.info(f"taisyaku zandaka.csv 取得: {len(df_z)} 件")
    except Exception as e:
        logger.warning(f"zandaka.csv 取得失敗: {e}")
        _taisyaku_cache["zandaka"] = None

    # ── 逆日歩（shina.csv）── ヘッダー3行をスキップ
    try:
        r2 = requests.get(_SHINA_URL, headers=_TAISYAKU_HEADERS, timeout=30)
        r2.raise_for_status()
        from io import StringIO as _StringIO2
        df_s = pd.read_csv(
            _StringIO2(r2.content.decode("shift_jis", errors="replace")),
            skiprows=3,
        )
        df_s["コード"] = df_s["コード"].astype(str).str.strip()
        _taisyaku_cache["shina"] = df_s
        logger.info(f"taisyaku shina.csv 取得: {len(df_s)} 件（逆日歩発生銘柄）")
    except Exception as e:
        logger.warning(f"shina.csv 取得失敗: {e}")
        _taisyaku_cache["shina"] = None

    return _taisyaku_cache.get("zandaka"), _taisyaku_cache.get("shina")


def fetch_shinyo_info(code: str) -> dict | None:
    """
    taisyaku.jp CSVから信用取引情報を返す。
    返り値: {ratio, buying, selling, gyakuhibu} または None（データなし時）
      - ratio     : 信用倍率（float）
      - buying    : 信用買い残（株）
      - selling   : 信用売り残（株）
      - gyakuhibu : 逆日歩・日次単価（円/株/日, float | None）
    """
    df_z, df_s = _load_taisyaku_csvs()

    buying:  float | None = None
    selling: float | None = None
    ratio:   float | None = None

    if df_z is not None:
        row = df_z[df_z["銘柄コード"] == code]
        if not row.empty:
            r = row.iloc[0]
            try:
                buying  = float(r["融資残高株数"])
                selling = float(r["貸株残高株数"])
                ratio   = (buying / selling) if selling and selling > 0 else None
            except Exception:
                pass

    if buying is None:
        return None

    # 逆日歩（shina.csv に当日発生分のみ掲載）
    gyakuhibu: float | None = None
    if df_s is not None:
        srow = df_s[df_s["コード"] == code]
        if not srow.empty:
            try:
                total_fee = float(srow.iloc[0]["当日品貸料率（円）"])
                days      = float(srow.iloc[0]["当日品貸日数"])
                # 当日品貸料率 = 日次単価 × 品貸日数 → 日次単価に戻す
                if days > 0:
                    gyakuhibu = round(total_fee / days, 2)
            except Exception:
                pass

    return {
        "ratio":     ratio,
        "buying":    buying,
        "selling":   selling,
        "gyakuhibu": gyakuhibu,
    }


def fetch_shinyo_batch(codes: list[str], **_) -> dict[str, dict | None]:
    """複数銘柄の需給情報を返す。CSVは1回だけDLしてキャッシュ済みのため高速。"""
    return {code: fetch_shinyo_info(code) for code in codes}


_MEDALS = ["🥇", "🥈", "🥉"]

# 戦略ごとのスコア定義
_SCORE_RULES: dict[str, list[tuple]] = {
    # (条件関数, 点数, 説明)
    "oversold_bounce":    [],   # スコアなし
    "noa":                [],   # スコアなし
}


def calc_score(strategy_key: str, result: dict, shinyo: dict | None) -> int:
    return sum(
        pts
        for cond, pts, _ in _SCORE_RULES.get(strategy_key, [])
        if cond(result, shinyo)
    )


def build_ranking_block(
    strategy_key: str,
    strategy_label: str,
    results: list[dict],
    shinyo_map: dict[str, dict | None],
) -> str:
    """
    ランキングサマリーブロックを返す。
    ブレイクアウト型・結果0件・スコアルールなしの場合は空文字を返す。
    """
    if not results or not _SCORE_RULES.get(strategy_key):
        return ""

    with_data:    list[tuple[int, dict, dict]] = []
    without_data: list[dict]                   = []

    for r in results:
        shinyo = shinyo_map.get(r["code"])
        # ratio が None（売残ゼロで計算不能）も「データなし」扱い
        if shinyo is not None and shinyo.get("ratio") is not None:
            with_data.append((calc_score(strategy_key, r, shinyo), r, shinyo))
        else:
            without_data.append(r)

    if not with_data and not without_data:
        return ""

    with_data.sort(key=lambda x: x[0], reverse=True)
    top_n = min(3, len(with_data))
    title = f"🏆 {strategy_label} TOP{top_n}" if with_data else f"🏆 {strategy_label}"

    lines = [title, "━" * 16]

    if with_data:
        lines.append("【需給データあり】")
        for i, (score, r, shinyo) in enumerate(with_data):
            medal = _MEDALS[i] if i < len(_MEDALS) else "  "
            lines.append(f"{medal} {r['code']} {r['name']}（{score}点）")
            parts = []
            if shinyo.get("ratio") is not None:
                parts.append(f"信用倍率{shinyo['ratio']:.1f}倍")
            if shinyo.get("gyakuhibu") is not None:
                parts.append(f"逆日歩{shinyo['gyakuhibu']:.2f}円⚠️")
            if parts:
                lines.append(f"　{'・'.join(parts)}")

    if without_data:
        if with_data:
            lines.append("")
        lines.append("【需給データなし】")
        for r in without_data:
            lines.append(f"・{r['code']} {r['name']}")

    return "\n".join(lines)


def _fmt_shinyo_block(shinyo: dict | None) -> list[str]:
    """需給情報を通知フォーマットの行リストに変換する。取得失敗時は空リスト。"""
    if not shinyo:
        return []
    ratio_str = f"{shinyo['ratio']:.2f}倍" if shinyo["ratio"] is not None else "－"
    lines = [
        "  ─" * 8,
        "  📊 需給情報",
        f"  信用倍率：{ratio_str}",
        f"  信用買残：{int(shinyo['buying']):,}株",
        f"  信用売残：{int(shinyo['selling']):,}株",
    ]
    if shinyo.get("gyakuhibu") is not None:
        lines.append(f"  逆日歩　：{shinyo['gyakuhibu']:.2f}円/日 ⚠️")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# ポートフォリオ管理
# ──────────────────────────────────────────────────────────────────────────────
def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"positions": [], "history": []}


def save_portfolio(data: dict) -> None:
    PORTFOLIO_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def count_business_days(entry_date_str: str) -> int:
    """エントリー日から今日までの営業日数（土日除く）"""
    entry = date.fromisoformat(entry_date_str)
    today = date.today()
    if today <= entry:
        return 0
    count = 0
    d = entry + timedelta(days=1)
    while d <= today:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


def add_position(
    code: str,
    entry_price: float,
    strategy_type: str,
    stop: float | None = None,
    take: float | None = None,
) -> None:
    """保有銘柄を追加する。stop/take 未指定時は ATR から自動計算。"""
    ticker = code + ".T"

    name = code
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or code
    except Exception:
        pass

    if stop is None or take is None:
        df = fetch_history(ticker, days=100)
        if df is not None and len(df) >= 20:
            atr = calc_atr(df)
            if not pd.isna(atr) and atr > 0:
                if stop is None:
                    stop_atr = entry_price - atr * 2.0
                    stop = max(stop_atr, entry_price * 0.90)
                if take is None:
                    rr_map = {"oversold_bounce": OVERSOLD_BOUNCE_RR, "noa": NOA_RR}
                    rr  = rr_map.get(strategy_type, 1.5)
                    take = entry_price + (entry_price - stop) * rr

    portfolio = load_portfolio()
    portfolio["positions"] = [p for p in portfolio["positions"] if p["code"] != code]
    portfolio["positions"].append({
        "code":          code,
        "name":          name,
        "entry_price":   entry_price,
        "stop_loss":     round(stop,  1) if stop is not None else None,
        "take_profit":   round(take,  1) if take is not None else None,
        "strategy_type": strategy_type,
        "entry_date":    date.today().isoformat(),
        "status":        "open",
    })
    save_portfolio(portfolio)

    print(f"✅ ポジション追加: {code} {name}")
    stop_str = f"{stop:,.0f}円" if stop is not None else "未設定"
    take_str = f"{take:,.0f}円" if take is not None else "未設定"
    print(f"   買値: {entry_price:,.0f}円  損切り: {stop_str}  利確: {take_str}")


def close_position(code: str, exit_price: float, result: str) -> None:
    """保有銘柄を決済記録する。"""
    portfolio = load_portfolio()
    pos = next((p for p in portfolio["positions"] if p["code"] == code), None)
    if pos is None:
        print(f"❌ 銘柄 {code} は保有中ではありません。")
        return

    pnl_pct = (
        (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        if pos["entry_price"] > 0 else 0.0
    )
    closed = {
        **pos,
        "exit_price": exit_price,
        "exit_date":  date.today().isoformat(),
        "held_days":  count_business_days(pos["entry_date"]),
        "pnl_pct":    round(pnl_pct, 2),
        "result":     result,
        "status":     "closed",
    }
    portfolio["positions"] = [p for p in portfolio["positions"] if p["code"] != code]
    portfolio.setdefault("history", []).append(closed)
    save_portfolio(portfolio)

    print(f"✅ 決済記録: {code} {pos['name']}")
    print(f"   買値: {pos['entry_price']:,.0f}円 → 売値: {exit_price:,.0f}円"
          f"  損益: {pnl_pct:+.2f}%  結果: {result}")


def fetch_portfolio_updates() -> list[dict]:
    """保有中銘柄の現在値を取得してステータスを返す。"""
    positions = load_portfolio().get("positions", [])
    if not positions:
        return []

    updates: list[dict] = []
    for pos in positions:
        ticker = pos["code"] + ".T"
        try:
            df = fetch_history(ticker, days=10)
            if df is None or len(df) < 1:
                continue
            current = float(df["Close"].iloc[-1])
        except Exception:
            continue

        entry = pos["entry_price"]
        stop  = pos.get("stop_loss")
        take  = pos.get("take_profit")
        held  = count_business_days(pos["entry_date"])

        pnl_pct    = (current - entry) / entry * 100 if entry > 0 else 0.0
        stop_dist  = (stop  - current) / current * 100 if stop  is not None else None
        take_dist  = (take  - current) / current * 100 if take  is not None else None

        updates.append({
            "code":          pos["code"],
            "name":          pos["name"],
            "entry_price":   entry,
            "current_price": current,
            "stop_loss":     stop,
            "take_profit":   take,
            "strategy_type": pos.get("strategy_type", ""),
            "entry_date":    pos["entry_date"],
            "held_days":     held,
            "pnl_pct":       pnl_pct,
            "stop_dist_pct": stop_dist,
            "take_dist_pct": take_dist,
            "hit_stop":      stop is not None and current <= stop,
            "hit_take":      take is not None and current >= take,
            "expired":       held >= POSITION_MAX_DAYS,
        })

    return updates


def build_portfolio_message(updates: list[dict]) -> str | None:
    """保有中銘柄の LINE メッセージを生成する。"""
    if not updates:
        return None

    lines = [
        "━" * 16,
        f"📊 保有中銘柄（{len(updates)}件）",
        "━" * 16,
    ]

    for u in updates:
        if u["hit_stop"]:
            alert = "  ⚠️損切りサイン"
        elif u["hit_take"]:
            alert = "  🎯利確サイン"
        elif u["expired"]:
            alert = "  ⏰期間終了"
        else:
            alert = ""

        pnl_sign = "+" if u["pnl_pct"] >= 0 else ""
        block = [
            "",
            f"▶ {u['code']} {u['name']}",
        ]
        if alert:
            block.append(alert)
        block.append(
            f"  買値：{u['entry_price']:,.0f}円 → 現在：{u['current_price']:,.0f}円"
        )
        block.append(f"  損益：{pnl_sign}{u['pnl_pct']:.1f}%")
        if u["stop_loss"] is not None:
            block.append(
                f"  損切り：{u['stop_loss']:,.0f}円（まで{u['stop_dist_pct']:+.1f}%）"
            )
        if u["take_profit"] is not None:
            block.append(
                f"  利確：{u['take_profit']:,.0f}円（まで{u['take_dist_pct']:+.1f}%）"
            )
        block.append(f"  保有：{u['held_days'] + 1}日目")
        lines.extend(block)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# スクリーニング実行（3戦略並列）
# ──────────────────────────────────────────────────────────────────────────────
def run_screening(use_cache: bool = True) -> None:
    logger.info("=" * 60)
    logger.info(f"スクリーニング開始（3戦略 / 並列 {MAX_WORKERS} ワーカー）")
    _load_ohlcv_cache(skip=not use_cache)

    # ── 過去20営業日のパフォーマンスを事前取得 ──
    performance_msg: str | None = None
    recent_files = load_recent_results(max_hold=20)
    if recent_files:
        # 有効な3戦略のみ・unique な銘柄を集約（最新シグナル日を優先）
        TRACK_MAX = 20   # 追跡銘柄の上限
        seen: dict[str, dict] = {}
        for file_entry in recent_files:
            sig_date  = file_entry["signal_date"]
            hold_days = count_business_days(sig_date)
            for strategy_key, entries in file_entry["data"].get("strategies", {}).items():
                if strategy_key not in STRATEGIES:   # 廃止戦略は除外
                    continue
                strategy_label = STRATEGIES[strategy_key]
                for entry in entries:
                    code = entry["code"]
                    if code not in seen:
                        seen[code] = {
                            **entry,
                            "strategies":   [strategy_label],
                            "signal_date":  sig_date,
                            "hold_days":    hold_days,
                        }
                    else:
                        if strategy_label not in seen[code]["strategies"]:
                            seen[code]["strategies"].append(strategy_label)

        # 最新シグナル日順に並べ直して上限20件に絞る
        prev_entries = sorted(
            seen.values(),
            key=lambda e: (e["signal_date"], e["code"]),
            reverse=True,
        )[:TRACK_MAX]
        prev_date_str = recent_files[0]["signal_date"]   # 最新シグナル日
        logger.info(f"追跡銘柄 {len(prev_entries)} 件（上限{TRACK_MAX}件・過去20営業日）のパフォーマンスを取得中...")

        performances: list[dict] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            perf_futures = {executor.submit(fetch_stock_performance, e): e
                            for e in prev_entries}
            for future in as_completed(perf_futures):
                result = future.result()
                if result:
                    performances.append(result)

        performance_msg = build_performance_message(prev_date_str, performances)
        logger.info(f"パフォーマンス集計完了: {len(performances)}/{len(prev_entries)} 件取得")
        save_performance(performances, prev_date_str)

    # ── メイン スクリーニング ──
    try:
        tickers = fetch_jpx_stock_list()
    except RuntimeError as e:
        logger.error(str(e))
        send_notify(f"【エラー】銘柄リスト取得に失敗しました\n{e}")
        return

    total       = len(tickers)
    completed   = 0
    lock        = threading.Lock()
    all_results = {k: [] for k in STRATEGIES}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(screen_ticker, t): t for t in tickers}

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.debug(f"{ticker}: 例外 ({e})")
                result = None

            with lock:
                completed += 1
                cnt = completed

            if cnt % 200 == 0 or cnt == total:
                logger.info(f"進捗: {cnt}/{total} 件処理済み")

            if result:
                for strategy_key, data in result.items():
                    logger.info(
                        f"  ✓ [{STRATEGIES[strategy_key]}] "
                        f"{data['code']} {data['name']}"
                    )
                    all_results[strategy_key].append(data)

    # ── 結果を JSON 保存 ──
    save_results(all_results, date.today())

    # ── マッチ銘柄の需給情報を一括取得 ──
    all_matched_codes: list[str] = []
    seen_codes: set[str] = set()
    for results in all_results.values():
        for r in results:
            if r["code"] not in seen_codes:
                all_matched_codes.append(r["code"])
                seen_codes.add(r["code"])

    shinyo_map: dict[str, dict | None] = {}
    if all_matched_codes:
        logger.info(f"需給情報取得中: {len(all_matched_codes)} 銘柄...")
        shinyo_map = fetch_shinyo_batch(all_matched_codes, max_workers=4)
        ok = sum(1 for v in shinyo_map.values() if v)
        logger.info(f"需給情報取得完了: {ok}/{len(all_matched_codes)} 件成功")

    # ── LINE 通知: ポートフォリオ（保有中銘柄）を最初に送信 ──
    portfolio_updates = fetch_portfolio_updates()
    portfolio_msg = build_portfolio_message(portfolio_updates)
    if portfolio_msg:
        logger.info(f"ポートフォリオ {len(portfolio_updates)} 件 → LINE通知送信")
        send_line_notify(portfolio_msg)
        time.sleep(1)

    # ── 通知: 全戦略を1通にまとめて送信（0件の戦略はスキップ）──
    # LINE: ポートフォリオ追跡・パフォーマンス・新規シグナルすべて送信
    # Discord: 新規シグナルのみ送信（追跡はアプリで確認）
    line_parts: list[str] = []
    discord_parts: list[str] = []

    if performance_msg:
        line_parts.append(performance_msg)

    for strategy_key, strategy_label in STRATEGIES.items():
        if strategy_key not in NOTIFY_STRATEGIES:
            continue
        results = all_results[strategy_key]
        if not results:
            logger.info(f"[{strategy_label}] 0件 → スキップ")
            continue
        logger.info(f"[{strategy_label}] {len(results)} 件 → 通知に追加")
        msg = build_message(strategy_key, strategy_label, results, shinyo_map)
        line_parts.append(msg)
        discord_parts.append(msg)

    if line_parts:
        combined = "\n\n" + "─" * 30 + "\n\n".join(line_parts)
        send_line_notify(combined)
        logger.info("LINE: 全戦略の通知送信完了")
    else:
        logger.info("全戦略0件のためLINE通知なし")

    if discord_parts:
        combined = "\n\n" + "─" * 30 + "\n\n".join(discord_parts)
        send_discord_notify(combined)
        logger.info("Discord: 新規シグナルの通知送信完了")
    else:
        logger.info("新規シグナル0件のためDiscord通知なし")
    _save_ohlcv_cache()


# ──────────────────────────────────────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN または LINE_USER_ID が未設定です。LINE通知をスキップします。")

    def _arg(flag: str) -> str | None:
        """sys.argv から --flag VALUE を取得するヘルパー。"""
        try:
            return sys.argv[sys.argv.index(flag) + 1]
        except (ValueError, IndexError):
            return None

    # ── ポジション追加 ──
    if "--add-position" in sys.argv:
        code  = _arg("--code")
        entry = _arg("--entry")
        stype = _arg("--type")
        if not code or not entry or not stype:
            print("使い方: --add-position --code CODE --entry PRICE --type {oversold_bounce|noa} [--stop STOP] [--take TAKE]")
            sys.exit(1)
        if stype not in ("oversold_bounce", "noa"):
            print(f"--type は oversold_bounce / noa のいずれかを指定してください。")
            sys.exit(1)
        stop_val = _arg("--stop")
        take_val = _arg("--take")
        add_position(
            code, float(entry), stype,
            float(stop_val) if stop_val else None,
            float(take_val) if take_val else None,
        )
        sys.exit(0)

    # ── ポジション決済 ──
    if "--close-position" in sys.argv:
        code       = _arg("--code")
        exit_price = _arg("--exit")
        result     = _arg("--result")
        if not code or not exit_price or not result:
            print("使い方: --close-position --code CODE --exit PRICE --result {win|loss|break_even}")
            sys.exit(1)
        close_position(code, float(exit_price), result)
        sys.exit(0)

    if "--now" in sys.argv or "-n" in sys.argv:
        run_screening(use_cache=False)
    else:
        RUN_TIME = os.getenv("RUN_TIME", "16:00")
        if not re.fullmatch(r"\d{2}:\d{2}", RUN_TIME):
            logger.error(f"RUN_TIME は HH:MM 形式で指定してください（現在: {RUN_TIME!r}）")
            sys.exit(1)

        logger.info(f"スケジューラー起動 — 毎日 {RUN_TIME} に実行")
        logger.info("即時実行したい場合: python stock_screener.py --now")

        schedule.every().day.at(RUN_TIME).do(run_screening)

        while True:
            schedule.run_pending()
            time.sleep(30)
