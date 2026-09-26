#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファンダ主導型スクリーナー / バックテスト
==========================================
フィルター:
  ① 流動性 : 20日平均売買代金 ≥ 5000万円 / 時価総額 50〜300億円
  ② ファンダ: 前年同期比 EPS ≥+20% / 売上 ≥+15% / ROE ≥17%
             J-Quants /fins/statements（開示日ゲーティング）
  ③ テクニカル: MA200上昇（slope_days前比）+ 終値 > MA200
  ④ 希薄化フラグ: TDnet 第三者割当・新株予約権・転換社債（過去365日）

使い方:
  python fundamental_screener.py                         # 当日スクリーニング
  python fundamental_screener.py --backtest              # バックテスト（ohlcv_cache.pkl）
  python fundamental_screener.py --backtest backtest_cache.pkl
  python fundamental_screener.py --fetch-fins            # 財務キャッシュ強制更新
"""

import json, os, pickle, re, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent / ".env")

# ─── 定数 ─────────────────────────────────────────────────────────────────────
JQUANTS_API_BASE = "https://api.jquants.com/v2"
# V2: APIキー方式（ダッシュボードから発行、x-api-key ヘッダーで使用）
JQUANTS_API_KEY  = os.getenv("JQUANTS_API_KEY", os.getenv("JQUANTS_REFRESH_TOKEN", ""))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ① 流動性フィルター
MIN_MARKET_CAP   = 50  * 10**8   # 50億円
MAX_MARKET_CAP   = 300 * 10**8   # 300億円
MIN_AVG_TURNOVER = 50_000_000    # 5000万円/日（20日平均売買代金）

# ② ファンダフィルター（前年同期比）
EPS_GROWTH_MIN   = 20.0   # % YoY（スタンドアローン四半期）
SALES_GROWTH_MIN = 15.0   # % YoY
ROE_MIN          = 17.0   # %（当期純利益 / 純資産、年換算）

# ③ テクニカル
MA200_SLOPE_DAYS = 20     # MA200が何日前より上昇していればOK

# バックテスト設定
TRAIL_PCT    = 0.15       # トレーリングストップ（ピーク比下落率）
MAX_HOLD     = 120        # 最大保有日数
COOLDOWN     = 30         # 同一銘柄クールダウン（日）
HOLD_DAYS    = [20, 60, 120]  # 固定保有期間リターン出力用

# パス
FINS_CACHE_PATH = Path(__file__).parent / "fins_cache.pkl"
JPX_LIST_URL    = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
TDNET_URL = "https://www.release.tdnet.info/inbs/I_list_001_{code4}.html"

# 希薄化キーワード
DILUTION_KEYWORDS = ["第三者割当", "新株予約権", "転換社債"]

# 並列スレッド数（J-Quants財務取得）
MAX_FINS_WORKERS = 3   # J-Quants V2 レート制限対策（並列数を抑える）


# ─── J-Quants V2 認証 ─────────────────────────────────────────────────────────
def get_api_headers() -> dict:
    """V2 APIキー認証ヘッダーを返す"""
    if not JQUANTS_API_KEY:
        raise RuntimeError(".env の JQUANTS_API_KEY（または JQUANTS_REFRESH_TOKEN）が未設定です")
    return {"x-api-key": JQUANTS_API_KEY}


def check_api_key() -> None:
    """APIキーの疎通確認（軽量エンドポイントで確認）"""
    hdrs = get_api_headers()
    r = requests.get(f"{JQUANTS_API_BASE}/markets/calendar", headers=hdrs, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(
            f"J-Quants API 認証失敗: HTTP {r.status_code}\n"
            "→ ダッシュボードから新しいAPIキーを発行し、.env の JQUANTS_API_KEY に設定してください。"
        )


# ─── J-Quants 財務データ取得 ───────────────────────────────────────────────────

def fetch_fins_statements(code4: str) -> pd.DataFrame | None:
    """1銘柄の全四半期財務情報を取得（V2 /fins/summary、開示日順ソート済み）"""
    try:
        time.sleep(0.2)  # レート制限対策
        resp = requests.get(
            f"{JQUANTS_API_BASE}/fins/summary",
            params={"code": code4},
            headers=get_api_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        records = resp.json().get("data", [])
        if not records:
            return None
        df = pd.DataFrame(records)
        # V2 → 内部標準フィールド名にリネーム
        df = df.rename(columns={
            "DiscDate":   "DisclosedDate",
            "CurPerType": "TypeOfDocument",        # "1Q"/"2Q"/"3Q"/"FY"
            "CurFYEn":   "CurrentFiscalYearEndDate",
            "Sales":     "NetSales",
            "NP":        "Profit",
            "EPS":       "EarningsPerShare",
            "Eq":        "Equity",
        })
        df["DisclosedDate"] = pd.to_datetime(df["DisclosedDate"], errors="coerce")
        df = df.dropna(subset=["DisclosedDate"])
        df = df.sort_values("DisclosedDate").reset_index(drop=True)
        return df
    except Exception:
        return None


def build_fins_cache(
    tickers: list[str],
    existing: dict[str, pd.DataFrame] | None = None,
    req_interval: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    全銘柄の財務データを順次取得してキャッシュ構築。
    existing に前回の途中結果を渡すと未取得分だけ追加取得できる。
    req_interval: リクエスト間隔（秒）。レート制限対策。
    """
    result: dict[str, pd.DataFrame] = dict(existing) if existing else {}
    # 既取得済みはスキップ
    remaining = [t for t in tickers if t not in result]
    total = len(tickers)
    done  = total - len(remaining)

    print(f"  既取得: {done}銘柄  残り: {len(remaining)}銘柄")

    for ticker in remaining:
        code4 = ticker.replace(".T", "")
        df = fetch_fins_statements(code4)  # 内部で sleep(0.2) 済み
        if df is not None and len(df) > 0:
            result[ticker] = df
        done += 1
        extra_sleep = max(0.0, req_interval - 0.2)
        if extra_sleep > 0:
            time.sleep(extra_sleep)
        if done % 50 == 0 or done == total:
            print(f"  財務データ取得: {done}/{total}  取得済={len(result)}", flush=True)

    return result


# ─── 四半期スタンドアローン変換 ────────────────────────────────────────────────

def _extract_quarter(doc_type: str) -> int | None:
    """TypeOfDocument / CurPerType から四半期番号を返す（Q1=1, Q2=2, Q3=3, FY=4）"""
    s = str(doc_type).upper().strip()
    # V2 CurPerType: "1Q" "2Q" "3Q" "FY"
    if s == "1Q": return 1
    if s == "2Q": return 2
    if s == "3Q": return 3
    if s == "FY": return 4
    # V1 TypeOfDocument（後方互換）
    sl = s.lower()
    if "1stquarter" in sl: return 1
    if "2ndquarter" in sl: return 2
    if "3rdquarter" in sl: return 3
    if "fiscalyear"  in sl and "quarter" not in sl: return 4
    if "annual"      in sl and "quarter" not in sl: return 4
    return None


def to_standalone(fins_df: pd.DataFrame) -> pd.DataFrame:
    """
    J-Quants の累積値（Q1+Q2+...）からスタンドアローン単四半期値を計算する。
    追加カラム: sa_sales / sa_profit / sa_eps / sa_equity
    """
    df = fins_df.reset_index(drop=True).copy()

    for col in ["NetSales", "Profit", "EarningsPerShare", "Equity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["quarter"] = df["TypeOfDocument"].apply(_extract_quarter)

    fy_col = "CurrentFiscalYearEndDate" if "CurrentFiscalYearEndDate" in df.columns else None

    sa_sales  = np.full(len(df), np.nan)
    sa_profit = np.full(len(df), np.nan)
    sa_eps    = np.full(len(df), np.nan)

    for idx in range(len(df)):
        q = df.at[idx, "quarter"]
        if q is None or pd.isna(q):
            continue
        q = int(q)

        if q == 1:
            # Q1 のみ：累積値 = スタンドアローン値
            sa_sales[idx]  = df.at[idx, "NetSales"]
            sa_profit[idx] = df.at[idx, "Profit"]
            sa_eps[idx]    = df.at[idx, "EarningsPerShare"]
        else:
            # Q2/Q3/FY：前四半期の累積値を引いてスタンドアローン計算
            prev_q = q - 1
            if fy_col:
                fy_end = df.at[idx, fy_col]
                mask = (df["quarter"] == prev_q) & (df[fy_col] == fy_end) & (df.index < idx)
            else:
                mask = (df["quarter"] == prev_q) & (df.index < idx)

            prev_rows = df[mask]
            if len(prev_rows) > 0:
                prev = prev_rows.iloc[-1]
                sa_sales[idx]  = df.at[idx, "NetSales"]           - prev["NetSales"]
                sa_profit[idx] = df.at[idx, "Profit"]             - prev["Profit"]
                sa_eps[idx]    = df.at[idx, "EarningsPerShare"]   - prev["EarningsPerShare"]
            else:
                # 前四半期なし（データ欠落）→ 累積値をそのまま近似使用
                sa_sales[idx]  = df.at[idx, "NetSales"]
                sa_profit[idx] = df.at[idx, "Profit"]
                sa_eps[idx]    = df.at[idx, "EarningsPerShare"]

    df["sa_sales"]  = sa_sales
    df["sa_profit"] = sa_profit
    df["sa_eps"]    = sa_eps
    df["sa_equity"] = df["Equity"].values.copy()  # 残高はそのまま
    return df


# ─── ファンダ判定 ──────────────────────────────────────────────────────────────

def eval_fundamentals(fins_df: pd.DataFrame, as_of: datetime) -> tuple[bool, dict]:
    """
    as_of 時点で開示済みのデータのみ使用してファンダを評価する（データリーク防止）。
    returns: (passes_all, details_dict)
    """
    # 開示日ゲーティング（DisclosedDate が naive datetime の場合を想定）
    disc_dates = fins_df["DisclosedDate"].dt.tz_localize(None) \
        if fins_df["DisclosedDate"].dt.tz is not None \
        else fins_df["DisclosedDate"]

    avail = fins_df[disc_dates <= pd.Timestamp(as_of)].reset_index(drop=True)
    if len(avail) < 2:
        return False, {}

    avail_sa = to_standalone(avail)
    latest   = avail_sa.iloc[-1]

    q_num  = latest["quarter"]
    if q_num is None or pd.isna(q_num):
        return False, {}
    q_num = int(q_num)

    fy_col = "CurrentFiscalYearEndDate"
    if fy_col not in avail_sa.columns:
        return False, {}

    latest_fy_str = str(latest[fy_col])[:10]  # "YYYY-MM-DD"
    try:
        latest_fy_end = datetime.strptime(latest_fy_str, "%Y-%m-%d")
    except ValueError:
        return False, {}

    # 前年同期の決算期末（1年前）
    try:
        prev_fy_end = latest_fy_end.replace(year=latest_fy_end.year - 1)
    except ValueError:
        # うるう年2/29の場合
        prev_fy_end = latest_fy_end.replace(year=latest_fy_end.year - 1, day=28)
    prev_fy_str = prev_fy_end.strftime("%Y-%m")  # "YYYY-MM" でマッチ（日ずれ許容）

    prev_mask = (
        (avail_sa["quarter"] == q_num) &
        avail_sa[fy_col].astype(str).str.startswith(prev_fy_str)
    )
    prev_rows = avail_sa[prev_mask]
    if len(prev_rows) == 0:
        return False, {}
    prev = prev_rows.iloc[-1]

    def _pct_change(curr_val, prev_val):
        c = float(curr_val) if not pd.isna(curr_val) else None
        p = float(prev_val) if not pd.isna(prev_val) else None
        if c is None or p is None or p <= 0:
            return None
        return (c - p) / p * 100

    eps_growth   = _pct_change(latest["sa_eps"],    prev["sa_eps"])
    sales_growth = _pct_change(latest["sa_sales"],  prev["sa_sales"])

    # ROE: FY通期が使える場合はそれ、なければ直近四半期を年換算
    roe = None
    fy_mask = (avail_sa["quarter"] == 4) & \
              avail_sa[fy_col].astype(str).str.startswith(latest_fy_str[:7])
    fy_rows = avail_sa[fy_mask]
    if len(fy_rows) > 0:
        fy = fy_rows.iloc[-1]
        profit_fy = float(fy["sa_profit"]) if not pd.isna(fy["sa_profit"]) else None
        equity    = float(fy["sa_equity"]) if not pd.isna(fy["sa_equity"]) else None
        if profit_fy is not None and equity is not None and equity > 0:
            roe = profit_fy / equity * 100
    else:
        profit_q = float(latest["sa_profit"]) if not pd.isna(latest["sa_profit"]) else None
        equity   = float(latest["sa_equity"]) if not pd.isna(latest["sa_equity"]) else None
        if profit_q is not None and equity is not None and equity > 0:
            annualized = profit_q * (4 / max(q_num, 1))
            roe = annualized / equity * 100

    ok_eps   = eps_growth   is not None and eps_growth   >= EPS_GROWTH_MIN
    ok_sales = sales_growth is not None and sales_growth >= SALES_GROWTH_MIN
    ok_roe   = roe          is not None and roe           >= ROE_MIN

    details = {
        "eps_growth":   round(eps_growth,   1) if eps_growth   is not None else None,
        "sales_growth": round(sales_growth, 1) if sales_growth is not None else None,
        "roe":          round(roe,          1) if roe          is not None else None,
        "q_num":        q_num,
        "ok_eps":       ok_eps,
        "ok_sales":     ok_sales,
        "ok_roe":       ok_roe,
    }
    return ok_eps and ok_sales and ok_roe, details


# ─── テクニカル / 流動性 ───────────────────────────────────────────────────────

def check_technical(c: np.ndarray, i: int) -> bool:
    """MA200上昇 + 終値 > MA200"""
    need = 200 + MA200_SLOPE_DAYS
    if i < need:
        return False
    ma_now  = c[i - 199: i + 1].mean()
    ma_prev = c[i - 199 - MA200_SLOPE_DAYS: i + 1 - MA200_SLOPE_DAYS].mean()
    if np.isnan(ma_now) or np.isnan(ma_prev):
        return False
    return bool(c[i] > ma_now > ma_prev)


def check_liquidity(c: np.ndarray, v: np.ndarray, i: int) -> bool:
    """20日平均売買代金 ≥ MIN_AVG_TURNOVER"""
    if i < 20:
        return False
    avg_to = (c[i - 20: i] * v[i - 20: i]).mean()
    return bool(avg_to >= MIN_AVG_TURNOVER)


# ─── TDnet 希薄化フラグ ────────────────────────────────────────────────────────

def check_tdnet_dilution(code4: str, lookback_days: int = 365) -> tuple[bool, list[str]]:
    """
    TDnet の開示リストページをスクレイプして過去 lookback_days 日以内の
    希薄化イベント（第三者割当・新株予約権・転換社債）を検出する。
    """
    try:
        url  = TDNET_URL.format(code4=code4)
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; fundamental_screener)"},
        )
        if resp.status_code != 200:
            return False, []

        text    = resp.text
        cutoff  = datetime.now() - timedelta(days=lookback_days)
        # 日付パターン YYYY/MM/DD をページから抽出
        date_re = re.compile(r"(\d{4}/\d{2}/\d{2})")
        matched = []

        for kw in DILUTION_KEYWORDS:
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx == -1:
                    break
                # キーワード前後500文字から日付を探す
                context = text[max(0, idx - 500): idx + 100]
                dates_found = date_re.findall(context)
                for d in dates_found:
                    try:
                        dt = datetime.strptime(d, "%Y/%m/%d")
                        if dt >= cutoff:
                            matched.append(f"{d}: {kw}")
                            break
                    except ValueError:
                        continue
                start = idx + 1

        return len(matched) > 0, list(dict.fromkeys(matched))  # 重複除去
    except Exception:
        return False, []


# ─── スクリーニングモード ─────────────────────────────────────────────────────

def run_screener(fins_cache: dict[str, pd.DataFrame]) -> None:
    import yfinance as yf

    print("JPX銘柄リスト取得中...")
    try:
        resp    = requests.get(JPX_LIST_URL, timeout=30)
        resp.raise_for_status()
        df_jpx  = pd.read_excel(BytesIO(resp.content), dtype=str)
        mkt_col = next((c for c in df_jpx.columns if "市場" in str(c)), None)
        cod_col = next((c for c in df_jpx.columns if "コード" in str(c)), None)
        if not mkt_col or not cod_col:
            raise RuntimeError("JPX列が見つかりません")
        df_jpx  = df_jpx[df_jpx[mkt_col].str.contains("スタンダード|グロース|プライム", na=False)]
        tickers = [
            f"{str(c).strip()}.T"
            for c in df_jpx[cod_col]
            if str(c).strip().isdigit() and len(str(c).strip()) == 4
        ]
    except Exception as e:
        print(f"ERROR: JPX銘柄リスト取得失敗: {e}")
        return

    print(f"対象: {len(tickers)}銘柄  時価総額 {MIN_MARKET_CAP//10**8}〜{MAX_MARKET_CAP//10**8}億円")
    hits = []
    now  = datetime.now()

    for ticker in tickers:
        try:
            code4 = ticker.replace(".T", "")
            stock = yf.Ticker(ticker)
            mc    = getattr(stock.fast_info, "market_cap", None) or 0.0
            if not (MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP):
                continue

            df = stock.history(period="2y", auto_adjust=True)
            if df is None or len(df) < 230:
                continue
            c = df["Close"].values.astype(float)
            v = df["Volume"].values.astype(float)
            i = len(c) - 1

            if not check_liquidity(c, v, i):
                continue
            if not check_technical(c, i):
                continue

            fins = fins_cache.get(ticker)
            if fins is None:
                continue
            ok_fund, details = eval_fundamentals(fins, now)
            if not ok_fund:
                continue

            has_dil, dil_events = check_tdnet_dilution(code4)
            time.sleep(0.3)

            hits.append({
                "ticker":       ticker,
                "close":        round(c[i], 1),
                "market_cap_b": round(mc / 10**8, 1),
                "dilution":     has_dil,
                "dil_events":   dil_events,
                **details,
            })
            dil_mark = " ⚠️希薄化注意" if has_dil else ""
            print(
                f"  HIT: {ticker} ¥{c[i]:.0f} ({mc/10**8:.0f}億)  "
                f"EPS+{details.get('eps_growth')}%  "
                f"売上+{details.get('sales_growth')}%  "
                f"ROE {details.get('roe')}%{dil_mark}"
            )
        except Exception:
            continue

    # 出力
    print(f"\n{'=' * 60}")
    print(f"ファンダ型スクリーナー: {len(hits)}件ヒット")
    print(f"{'=' * 60}")

    for h in hits:
        dil_mark = " ⚠️希薄化注意" if h["dilution"] else ""
        print(
            f"  {h['ticker']}  ¥{h['close']}  {h['market_cap_b']}億円  "
            f"EPS+{h['eps_growth']}%  売上+{h['sales_growth']}%  "
            f"ROE {h['roe']}%{dil_mark}"
        )

    if hits and DISCORD_WEBHOOK_URL:
        _notify_discord(hits)


def _notify_discord(hits: list[dict]) -> None:
    lines = ["**ファンダ型スクリーナー** 本日のヒット\n"]
    for h in hits:
        dil = "  ⚠️ 希薄化注意" if h["dilution"] else ""
        lines.append(
            f"**{h['ticker']}** ¥{h['close']} ({h['market_cap_b']}億円)\n"
            f"  EPS: +{h['eps_growth']}%  売上: +{h['sales_growth']}%  "
            f"ROE: {h['roe']}%{dil}"
        )
    payload = {"content": "\n".join(lines)[:2000]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        print("Discord通知送信完了")
    except Exception as e:
        print(f"Discord通知失敗: {e}")


# ─── バックテストモード ────────────────────────────────────────────────────────

def _load_ohlcv_cache(cache_name: str) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """OHLCVキャッシュ読み込み。ohlcv_cache.pkl と backtest_cache.pkl の両構造に対応"""
    path = Path(__file__).parent / cache_name
    if not path.exists():
        raise FileNotFoundError(f"キャッシュが見つかりません: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)

    data: dict[str, pd.DataFrame] = {}
    caps: dict[str, float]        = {}
    raw_data = raw.get("data", {})

    for ticker, val in raw_data.items():
        if isinstance(val, dict) and "df" in val:
            data[ticker] = val["df"]
            caps[ticker] = float(val.get("market_cap", 0.0))
        elif isinstance(val, pd.DataFrame):
            data[ticker] = val
            caps[ticker] = 0.0

    return data, caps


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def run_backtest(
    cache_name: str,
    fins_cache: dict[str, pd.DataFrame],
    skip_fins: bool = False,
) -> None:
    print(f"\nOHLCVキャッシュ読み込み: {cache_name}")
    if skip_fins:
        print("モード: ①③ のみ（--skip-fins: ファンダフィルターをスキップ）")
    data, market_caps = _load_ohlcv_cache(cache_name)
    fins_count = sum(1 for t in data if t in fins_cache)
    print(f"全銘柄数: {len(data)}  財務データ有り: {fins_count}")

    all_hits: list[dict] = []
    survivorship_total   = len(data)

    for ticker in data:
        mc = market_caps.get(ticker, 0.0)
        # 時価総額フィルター（現在値で近似: ルックアヘッドバイアスあり・注記）
        if mc > 0 and not (MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP):
            continue

        fins = fins_cache.get(ticker)
        if not skip_fins and fins is None:
            continue

        df = _clean_df(data[ticker].copy())
        if len(df) < 230:
            continue

        c    = df["Close"].values.astype(float)
        v    = df["Volume"].values.astype(float)
        hi   = df["High"].values.astype(float)  if "High"  in df.columns else c.copy()
        lo   = df["Low"].values.astype(float)   if "Low"   in df.columns else c.copy()
        op   = df["Open"].values.astype(float)  if "Open"  in df.columns else c.copy()
        n    = len(c)
        dates = df.index

        last_hit_i = -COOLDOWN - 1

        for i in range(220, n - max(HOLD_DAYS) - 1):
            if i - last_hit_i <= COOLDOWN:
                continue

            # ① 流動性チェック
            if not check_liquidity(c, v, i):
                continue

            # ③ テクニカルチェック
            if not check_technical(c, i):
                continue

            # ② ファンダチェック（開示日ゲーティング）
            details: dict = {}
            if not skip_fins and fins is not None:
                dt_i  = dates[i]
                as_of = dt_i.to_pydatetime() if hasattr(dt_i, "to_pydatetime") \
                        else datetime(dt_i.year, dt_i.month, dt_i.day)
                if as_of.tzinfo is not None:
                    as_of = as_of.replace(tzinfo=None)
                ok_fund, details = eval_fundamentals(fins, as_of)
                if not ok_fund:
                    continue

            # エントリー: 翌日始値
            j0 = i + 1
            if j0 >= n:
                continue
            entry = op[j0]
            if entry <= 0:
                entry = c[j0]  # Openがなければ終値で代替
            if entry <= 0:
                continue

            # トレーリングストップシミュレーション
            trail_stop = entry * (1 - TRAIL_PCT)
            highest    = entry
            exit_price = None
            hold_days_actual = 0
            end = min(j0 + MAX_HOLD, n)

            for j in range(j0, end):
                if op[j] <= trail_stop:
                    exit_price       = op[j]
                    hold_days_actual = j - j0
                    break
                if hi[j] > highest:
                    highest    = hi[j]
                    trail_stop = max(trail_stop, highest * (1 - TRAIL_PCT))
                if lo[j] <= trail_stop:
                    exit_price       = trail_stop
                    hold_days_actual = j - j0
                    break

            if exit_price is None:
                exit_price       = c[min(end - 1, n - 1)]
                hold_days_actual = end - 1 - j0

            ret_trail = (exit_price - entry) / entry * 100

            # 固定保有リターン
            fwd = {}
            for nd in HOLD_DAYS:
                j2 = j0 + nd
                if j2 < n and c[j2] > 0:
                    fwd[nd] = (c[j2] - entry) / entry * 100
                else:
                    fwd[nd] = np.nan

            all_hits.append({
                "ticker":    ticker,
                "date":      dates[i].date().isoformat() if hasattr(dates[i], "date") else str(dates[i])[:10],
                "entry":     round(entry, 1),
                "exit":      round(exit_price, 1),
                "ret_trail": round(ret_trail, 2),
                "hold_days": hold_days_actual,
                **{f"ret_{nd}d": (round(fwd[nd], 2) if not np.isnan(fwd[nd]) else None) for nd in HOLD_DAYS},
                "eps_growth":   details.get("eps_growth"),
                "sales_growth": details.get("sales_growth"),
                "roe":          details.get("roe"),
                "q_num":        details.get("q_num"),
            })
            last_hit_i = i

    print(f"\n{'=' * 70}")
    mode_label = "①③ のみ" if skip_fins else "①②③"
    print(f"バックテスト結果 ({mode_label}): {len(all_hits)} 件")
    print(f"{'=' * 70}")
    print(f"⚠ サバイバーシップバイアス注記:")
    if not skip_fins:
        print(f"  - 全銘柄数 {survivorship_total} のうち財務データあり銘柄のみ対象")
    print(f"  - 時価総額フィルターはキャッシュ構築時の現在値で近似（ルックアヘッドバイアスあり）")
    print(f"  - 上場廃止銘柄はキャッシュに含まれない可能性あり")
    print(f"{'─' * 70}")

    if not all_hits:
        print("ヒットなし")
        return

    df_hits = pd.DataFrame(all_hits)

    # ──────────── 全体成績 ────────────
    print(f"\n【{mode_label} 全体 (ファンダ主導型)】")
    _print_perf("トレーリングストップ", df_hits["ret_trail"])
    for nd in HOLD_DAYS:
        col = f"ret_{nd}d"
        if col in df_hits.columns:
            _print_perf(f"固定{nd}日保有", df_hits[col].dropna())

    # 年別
    df_hits["year"] = df_hits["date"].str[:4].astype(int)
    print("\n  年別 (トレーリングストップ):")
    for yr in sorted(df_hits["year"].unique()):
        sub = df_hits[df_hits["year"] == yr]["ret_trail"]
        if len(sub) < 3:
            continue
        wins = sub[sub > 0]; losses = sub[sub <= 0]
        pf = (wins.sum() / losses.abs().sum()) if len(losses) > 0 and losses.abs().sum() > 0 else float("inf")
        print(f"  {yr}: 勝率{(sub>0).mean()*100:.0f}%  平均{sub.mean():+.1f}%  PF{pf:.2f}  n={len(sub)}")

    # ──────────── ④ 希薄化フラグ比較 ────────────
    print(f"\n{'─' * 70}")
    print("④ 希薄化フラグ判定中（TDnet, 現在データで近似）...")
    unique_tickers = df_hits["ticker"].unique().tolist()
    dilution_map: dict[str, bool] = {}
    for idx, t in enumerate(unique_tickers):
        has_dil, _ = check_tdnet_dilution(t.replace(".T", ""))
        dilution_map[t] = has_dil
        time.sleep(0.3)
        if (idx + 1) % 20 == 0:
            print(f"  TDnet確認: {idx+1}/{len(unique_tickers)}", flush=True)

    df_hits["dilution"] = df_hits["ticker"].map(dilution_map)
    with_dil    = df_hits[df_hits["dilution"] == True]
    without_dil = df_hits[df_hits["dilution"] == False]

    print(f"\n【希薄化フラグあり: {len(with_dil)}件】")
    if len(with_dil) > 0:
        _print_perf("  トレーリングストップ", with_dil["ret_trail"])
    else:
        print("  ─ データなし ─")

    print(f"\n【希薄化フラグなし: {len(without_dil)}件】")
    if len(without_dil) > 0:
        _print_perf("  トレーリングストップ", without_dil["ret_trail"])
    else:
        print("  ─ データなし ─")

    # ──────────── CSV保存 ────────────
    out = Path(__file__).parent / "results" / "fundamental_hits.csv"
    out.parent.mkdir(exist_ok=True)
    df_hits.sort_values("date", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nCSV保存: {out}")


def _print_perf(label: str, series: pd.Series) -> None:
    s = series.dropna()
    if len(s) == 0:
        print(f"  {label}: データなし")
        return
    wins   = s[s > 0]
    losses = s[s <= 0]
    wr     = (s > 0).mean() * 100
    avg    = s.mean()
    med    = s.median()
    pf_pos = wins.abs().sum()
    pf_neg = losses.abs().sum()
    pf     = pf_pos / pf_neg if pf_neg > 0 else float("inf")
    print(
        f"  {label}: "
        f"勝率 {wr:5.1f}%  平均 {avg:+6.2f}%  中央値 {med:+6.2f}%  "
        f"PF {pf:.2f}  n={len(s)}"
    )


# ─── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    args        = sys.argv[1:]
    is_backtest = "--backtest"   in args
    force_fetch = "--fetch-fins" in args
    skip_fins   = "--skip-fins"  in args   # J-Quants不要モード（①③のみ）
    cache_name  = next(
        (a for a in args if not a.startswith("-")),
        "ohlcv_cache.pkl",
    )

    # J-Quants APIキー確認（--skip-fins なら不要）
    if not skip_fins:
        print("J-Quants API 接続確認中...", end=" ", flush=True)
        try:
            check_api_key()
            print("OK")
        except RuntimeError as e:
            print(f"\n{e}")
            print("ヒント: J-Quantsが不要な場合は --skip-fins を追加してください")
            print("       python fundamental_screener.py --backtest --skip-fins")
            sys.exit(1)
    else:
        print("--skip-fins: J-Quants認証をスキップ（①③テクニカル+流動性のみ）")

    # 財務キャッシュ読み込み or 取得
    fins_cache: dict[str, pd.DataFrame] = {}
    today_str = date.today().isoformat()

    if not force_fetch and FINS_CACHE_PATH.exists():
        try:
            with open(FINS_CACHE_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved.get("date") == today_str:
                fins_cache = saved["data"]
                print(f"財務キャッシュ読み込み: {len(fins_cache)}銘柄 ({today_str})")
            else:
                print(f"財務キャッシュ期限切れ（{saved.get('date')} → 再取得）")
        except Exception as e:
            print(f"財務キャッシュ読み込み失敗: {e}")

    if not fins_cache and not skip_fins:
        if is_backtest:
            # バックテスト: OHLCVキャッシュの銘柄が対象
            try:
                bt_data, _ = _load_ohlcv_cache(cache_name)
                tickers = list(bt_data.keys())
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                sys.exit(1)
        else:
            # スクリーニング: JPXリストから取得
            resp    = requests.get(JPX_LIST_URL, timeout=30)
            df_jpx  = pd.read_excel(BytesIO(resp.content), dtype=str)
            mkt_col = next((c for c in df_jpx.columns if "市場" in str(c)), None)
            cod_col = next((c for c in df_jpx.columns if "コード" in str(c)), None)
            df_jpx  = df_jpx[df_jpx[mkt_col].str.contains("スタンダード|グロース|プライム", na=False)]
            tickers = [
                f"{str(c).strip()}.T" for c in df_jpx[cod_col]
                if str(c).strip().isdigit() and len(str(c).strip()) == 4
            ]

        print(f"\n財務データ取得中（{len(tickers)}銘柄 / 1リクエスト/秒）...")
        fins_cache = build_fins_cache(tickers, existing=fins_cache)
        print(f"財務データ取得完了: {len(fins_cache)}銘柄")

        with open(FINS_CACHE_PATH, "wb") as f:
            pickle.dump({"date": today_str, "data": fins_cache}, f)
        print(f"財務キャッシュ保存: {FINS_CACHE_PATH.name}")

    # 実行
    if is_backtest:
        run_backtest(cache_name, fins_cache, skip_fins=skip_fins)
    else:
        run_screener(fins_cache)


if __name__ == "__main__":
    main()
