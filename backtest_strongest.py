#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最強型 総合グリッドサーチ
=========================
RSI売られすぎ × 当日反発 × 各種フィルターの全組み合わせを探索

グリッド: rsi_hi × pct_thr × ma25_dev × atr_expand × vol_mult × mktcap_max × rr
         3      × 3       × 4        × 2          × 4        × 4          × 3 = 3,456通り

データ: backtest_cache.pkl (yfinance + J-Quants 補完・日付チェックで自動リフレッシュ)
評価:  WR≥60% / PF≥1.8 / 件/日≥0.2
"""

import itertools, os, pickle, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from stock_screener import calc_rsi, _get_jquants_id_token

# ── 定数 ──────────────────────────────────────────────────────────────────────
CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
CACHE_MAX_AGE = 3          # キャッシュが何日以上古ければリフレッシュするか
JPX_LIST_URL  = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JQUANTS_BASE  = "https://api.jquants.com/v1"
JQUANTS_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN", "")

MIN_HISTORY   = 100
MAX_HOLD      = 20
MAX_WORKERS   = 20
MIN_TURNOVER  = 30_000_000   # 20日平均売買代金 下限（円）

# ── 評価基準 ──────────────────────────────────────────────────────────────────
CRITERIA_WR  = 60.0
CRITERIA_PF  = 1.8
CRITERIA_SPD = 0.2

# ── グリッド ──────────────────────────────────────────────────────────────────
GRID = {
    "rsi_hi":     [35.0, 40.0, 45.0],
    "pct_thr":    [2.0, 3.0, 5.0],
    "ma25_dev":   [None, -3.0, -5.0, -10.0],   # None=フィルターなし
    "atr_expand": [False, True],
    "vol_mult":   [0.0, 1.5, 2.0, 3.0],         # 0.0=フィルターなし
    "mktcap_max": [50e9, 100e9, 200e9, None],    # None=フィルターなし
    "rr":         [1.5, 2.0, 2.5],
}

# フラグキーマップ（グリッド値 → preprocess結果のキー名）
_RSI_KEY  = {35.0: "r35", 40.0: "r40", 45.0: "r45"}
_PCT_KEY  = {2.0: "p2",  3.0: "p3",  5.0: "p5"}
_MA25_KEY = {-3.0: "m3", -5.0: "m5", -10.0: "m10"}
_VOL_KEY  = {1.5: "v15", 2.0: "v20", 3.0: "v30"}
_MC_KEY   = {50e9: "c50", 100e9: "c100", 200e9: "c200"}


# ══════════════════════════════════════════════════════════════════════════════
# データ取得・キャッシュ管理
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_jpx_tickers() -> list[str]:
    resp = requests.get(JPX_LIST_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content), dtype=str)
    mkt_col  = next((c for c in df.columns if "市場" in str(c)), None)
    code_col = next((c for c in df.columns if "コード" in str(c)), None)
    if not mkt_col or not code_col:
        raise RuntimeError("JPX列名が見つからない")
    mask = df[mkt_col].str.contains(
        r"(?=.*(?:スタンダード|グロース))(?=.*内国株式)", na=False, regex=True
    )
    return (df[mask][code_col].str.strip().str.zfill(4) + ".T").tolist()


def _fetch_jquants(code4: str, days: int = 730) -> pd.DataFrame | None:
    try:
        id_token  = _get_jquants_id_token()
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        to_date   = datetime.now().strftime("%Y%m%d")
        r = requests.get(
            f"{JQUANTS_BASE}/prices/daily_quotes",
            params={"code": code4, "from": from_date, "to": to_date},
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        records = r.json().get("daily_quotes", [])
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
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame | None]:
    code4 = ticker.replace(".T", "")
    # yfinance
    try:
        df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if df is not None and len(df) >= MIN_HISTORY:
            return ticker, df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    # J-Quants フォールバック
    if JQUANTS_TOKEN:
        df = _fetch_jquants(code4)
        if df is not None and len(df) >= MIN_HISTORY:
            return ticker, df
    return ticker, None


def build_or_load_cache() -> dict[str, pd.DataFrame]:
    """
    backtest_cache.pkl をロード。
    - 当日付なら即返す
    - CACHE_MAX_AGE 日以内なら J-Quants で差分のみ補完
    - それより古ければ全銘柄フル再取得
    """
    today_str = date.today().isoformat()

    # ── 既存キャッシュ確認 ──
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                stored = pickle.load(f)
            cache_date = stored.get("date", "")
            raw_data: dict[str, pd.DataFrame] = stored.get("data", {})
            cache_age = (date.today() - date.fromisoformat(cache_date)).days

            if cache_date == today_str:
                print(f"キャッシュ利用（当日）: {len(raw_data)} 銘柄")
                return raw_data

            print(f"キャッシュ: {cache_date}（{cache_age}日前）→ 差分補完")
        except Exception:
            raw_data = {}
            cache_age = 999
    else:
        raw_data = {}
        cache_age = 999
        print("キャッシュなし → 全件取得")

    # ── 銘柄リスト取得 ──
    print("JPX銘柄リスト取得中...")
    try:
        tickers = _fetch_jpx_tickers()
    except Exception as e:
        print(f"  リスト取得失敗: {e} → 既存キャッシュで続行")
        return raw_data
    print(f"  対象: {len(tickers)} 銘柄")

    # ── 不足・陳腐化銘柄を特定 ──
    if cache_age <= CACHE_MAX_AGE:
        # J-Quantsで差分のみ（最新N日）を補完
        missing = [t for t in tickers if t not in raw_data]
        to_fetch = missing
        mode = f"差分({len(missing)}銘柄)"
    else:
        to_fetch = tickers
        mode = "全件"

    if not to_fetch:
        print(f"補完不要  →  そのまま利用")
        return raw_data

    # ── 並列取得 ──
    print(f"データ取得中（{mode} / {len(to_fetch)}銘柄 / {MAX_WORKERS}並列）...")
    done = 0
    ok   = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in to_fetch}
        for fut in as_completed(futs):
            t, df = fut.result()
            done += 1
            if df is not None:
                raw_data[t] = df
                ok += 1
            if done % 300 == 0 or done == len(to_fetch):
                print(f"  {done}/{len(to_fetch)}  成功: {ok}")

    # ── 保存 ──
    print(f"キャッシュ保存中: {len(raw_data)} 銘柄...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print("  保存完了")
    return raw_data


# ══════════════════════════════════════════════════════════════════════════════
# 時価総額（株数 × 終値）
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        sh = yf.Ticker(ticker).fast_info.shares
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            t, sh = fut.result()
            result[t] = sh
            done += 1
            if done % 300 == 0 or done == len(tickers):
                ok = sum(1 for v in result.values() if v)
                print(f"  {done}/{len(tickers)}  取得: {ok}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 前処理（銘柄ごとに特徴量フラグを事前計算）
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(ticker: str, df_raw: pd.DataFrame,
               shares: float | None) -> dict | None:
    df = df_raw.copy()
    df = df[df["Close"] > 0].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < MIN_HISTORY:
        return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)

    # 翌日始値（エントリー価格）
    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    # RSI (14)
    rsi = calc_rsi(pd.Series(c)).values

    # 前日比 (%)
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    pct = np.where(prev_c > 0, (c - prev_c) / prev_c * 100, np.nan)

    # ATR (14)
    prev_c2 = np.roll(c, 1); prev_c2[0] = np.nan
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c2), np.abs(l - prev_c2)])
    atr = pd.Series(tr).rolling(14).mean().values

    # ATR拡大フラグ（直近3日平均 > 前3日平均）
    atr_s = pd.Series(atr)
    atr3d      = atr_s.rolling(3).mean().values
    atr3d_prev = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (atr3d > atr3d_prev) & ~np.isnan(atr3d) & ~np.isnan(atr3d_prev) & (atr3d_prev > 0)

    # MA25 乖離率 (%)
    ma25     = pd.Series(c).rolling(25).mean().values
    ma25_dev = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)

    # 出来高倍率（20日平均比）
    avg_v = pd.Series(v).rolling(20).mean().values
    vol_r = np.where(avg_v > 0, v / avg_v, np.nan)

    # 平均売買代金（20日）
    avg_to = pd.Series(c * v).rolling(20).mean().values

    # 時価総額
    mktcap = c * shares if shares else np.full(n, np.nan)

    # ベースフィルター（全条件共通）
    idx = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi)) & (~np.isnan(pct)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    # ── 事前計算フラグ ──
    flags: dict[str, np.ndarray] = {
        "base": base,
        # RSI上限
        "r35": (rsi <= 35.0) & ~np.isnan(rsi),
        "r40": (rsi <= 40.0) & ~np.isnan(rsi),
        "r45": (rsi <= 45.0) & ~np.isnan(rsi),
        # 前日比
        "p2":  (pct >= 2.0)  & ~np.isnan(pct),
        "p3":  (pct >= 3.0)  & ~np.isnan(pct),
        "p5":  (pct >= 5.0)  & ~np.isnan(pct),
        # MA25乖離
        "m3":  (ma25_dev <= -3.0)  & ~np.isnan(ma25_dev),
        "m5":  (ma25_dev <= -5.0)  & ~np.isnan(ma25_dev),
        "m10": (ma25_dev <= -10.0) & ~np.isnan(ma25_dev),
        # ATR拡大
        "atr_exp": atr_exp,
        # 出来高
        "v15": (vol_r >= 1.5) & ~np.isnan(vol_r),
        "v20": (vol_r >= 2.0) & ~np.isnan(vol_r),
        "v30": (vol_r >= 3.0) & ~np.isnan(vol_r),
        # 時価総額
        "c50":  np.isnan(mktcap) | (mktcap <= 50e9),
        "c100": np.isnan(mktcap) | (mktcap <= 100e9),
        "c200": np.isnan(mktcap) | (mktcap <= 200e9),
    }

    return {
        "close":  c,
        "next_o": next_o,
        "atr":    atr,
        "flags":  flags,
        "n":      n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# リターン計算（ベクトル化）
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rets(closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray,
               t_arr: np.ndarray) -> np.ndarray:
    n        = len(closes)
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut      = np.where(in_range, closes[safe_idx], np.nan)
    hit      = ((fut <= s_arr[:, np.newaxis]) | (fut >= t_arr[:, np.newaxis])) & in_range
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    ep       = closes[np.clip(vidx + 1 + fhp, 0, n - 1)]
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


def _get_mask(flags: dict, rsi_hi, pct_thr, ma25_dev, atr_expand,
              vol_mult, mktcap_max) -> np.ndarray:
    m = flags["base"] & flags[_RSI_KEY[rsi_hi]] & flags[_PCT_KEY[pct_thr]]
    if ma25_dev is not None:
        m = m & flags[_MA25_KEY[ma25_dev]]
    if atr_expand:
        m = m & flags["atr_exp"]
    if vol_mult > 0.0:
        m = m & flags[_VOL_KEY[vol_mult]]
    if mktcap_max is not None:
        m = m & flags[_MC_KEY[mktcap_max]]
    return m


def _metrics(rets: np.ndarray, trading_days: int, rr: float) -> dict:
    if len(rets) < 5:
        return {"n": len(rets), "wr": 0.0, "pf": 0.0, "ev": 0.0, "spd": 0.0}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets) * 100
    avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = len(rets) / trading_days if trading_days > 0 else 0.0
    return {"n": len(rets), "wr": wr, "pf": pf, "ev": ev, "spd": spd,
            "avg_w": avg_w, "avg_l": avg_l}


# ══════════════════════════════════════════════════════════════════════════════
# グリッドサーチ本体
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(all_proc: list[dict], trading_days: int) -> list[dict]:
    combos     = list(itertools.product(*GRID.values()))
    combo_keys = list(GRID.keys())
    n_combos   = len(combos)
    results    = []

    print(f"\nグリッドサーチ開始: {n_combos} 通り × {len(all_proc)} 銘柄")
    t0 = time.time()

    for ci, combo_vals in enumerate(combos, 1):
        p = dict(zip(combo_keys, combo_vals))
        all_rets: list[float] = []

        for td in all_proc:
            m = _get_mask(
                td["flags"],
                p["rsi_hi"], p["pct_thr"], p["ma25_dev"],
                p["atr_expand"], p["vol_mult"], p["mktcap_max"],
            )
            vidx = np.where(m)[0]
            if len(vidx) == 0:
                continue
            e = td["next_o"][vidx]
            a = td["atr"][vidx]
            s = np.maximum(e - a * 2.0, e * 0.90)
            t = e + (e - s) * p["rr"]
            valid = (e > 0) & (s > 0) & (t > 0) & (s < e) & (t > e)
            if not valid.any():
                continue
            rets = _calc_rets(td["close"], vidx[valid], e[valid], s[valid], t[valid])
            all_rets.extend(rets.tolist())

        m = _metrics(np.array(all_rets), trading_days, p["rr"])
        results.append({**p, **m})

        if ci % 200 == 0 or ci == n_combos:
            passed = sum(
                1 for r in results
                if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                and r["spd"] >= CRITERIA_SPD
            )
            elapsed = time.time() - t0
            eta     = elapsed / ci * (n_combos - ci)
            print(f"  {ci:4d}/{n_combos}  合格: {passed}件  "
                  f"経過: {elapsed:.0f}s  残: {eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_params(r: dict) -> str:
    mc   = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else "制限なし"
    ma25 = f"{r['ma25_dev']:+.0f}%" if r["ma25_dev"] is not None else "  なし"
    atr  = "ATR拡↑" if r["atr_expand"] else "     "
    vol  = f"vol{r['vol_mult']:.1f}x" if r["vol_mult"] > 0 else " vol-  "
    return (f"RSI≤{r['rsi_hi']:.0f}  +{r['pct_thr']:.0f}%  "
            f"MA25{ma25}  {atr}  {vol}  {mc}  RR{r['rr']}")


def print_results(results: list[dict]) -> None:
    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]

    print(f"\n{'='*80}")
    print(f"【最強型グリッドサーチ 結果】")
    print(f"  総計: {len(results)} 通り  |  合格: {len(qualified)} 通り")
    print(f"  評価基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")

    hdr = f"  {'条件':<60}  {'勝率':>7}  {'PF':>5}  {'EV':>7}  {'件数':>6}  {'件/日':>6}"
    sep = "  " + "-" * 95

    # ── 合格条件（WR順）──
    if qualified:
        top_wr = sorted(qualified, key=lambda r: (r["wr"], r["pf"]), reverse=True)[:20]
        print(f"\n▶ 合格条件 上位20件【WR順】")
        print(hdr); print(sep)
        for r in top_wr:
            print(f"  {_fmt_params(r):<60}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        top_pf = sorted(qualified, key=lambda r: (r["pf"], r["wr"]), reverse=True)[:10]
        print(f"\n▶ 合格条件 上位10件【PF順】")
        print(hdr); print(sep)
        for r in top_pf:
            print(f"  {_fmt_params(r):<60}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        top_ev = sorted(qualified, key=lambda r: r["ev"], reverse=True)[:10]
        print(f"\n▶ 合格条件 上位10件【期待値順】")
        print(hdr); print(sep)
        for r in top_ev:
            print(f"  {_fmt_params(r):<60}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        # ── ★ 最良条件 ──
        best = sorted(qualified,
                      key=lambda r: r["wr"] * 0.4 + r["pf"] * 20 + r["ev"] * 5,
                      reverse=True)[0]
        print(f"\n{'='*80}")
        print(f"★ 最良条件（WR×0.4 + PF×20 + EV×5 の総合スコア）")
        print(f"  RSI        : ≤ {best['rsi_hi']:.0f}")
        print(f"  前日比     : +{best['pct_thr']:.0f}% 以上")
        print(f"  MA25乖離   : {best['ma25_dev']:+.0f}% 以下" if best['ma25_dev'] else "  MA25乖離   : 条件なし")
        print(f"  ATR拡大    : {'あり' if best['atr_expand'] else 'なし'}")
        print(f"  出来高     : {best['vol_mult']:.1f}倍以上" if best['vol_mult'] > 0 else "  出来高     : 条件なし")
        print(f"  時価総額   : {best['mktcap_max']/1e8:.0f}億円以下" if best['mktcap_max'] else "  時価総額   : 条件なし")
        print(f"  RR         : 1:{best['rr']}")
        print(f"  ──────────")
        print(f"  勝率       : {best['wr']:.1f}%")
        print(f"  PF         : {best['pf']:.2f}")
        print(f"  期待値     : {best['ev']:+.2f}%/トレード")
        print(f"  件数       : {best['n']:,}件  ({best['spd']:.2f}件/日)")
        print(f"  平均利益   : {best['avg_w']:+.2f}%  平均損失: {best['avg_l']:+.2f}%")

    else:
        print("\n合格条件なし。緩和基準で上位10件を表示します。")
        top = sorted(results, key=lambda r: r["wr"] * r["pf"], reverse=True)[:10]
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt_params(r):<60}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

    # ── 分布サマリ ──
    print(f"\n{'='*80}")
    print("【条件ごとの分布】")
    for key, vals in GRID.items():
        if key == "rr":
            continue
        for v in vals:
            sub = [r for r in results if r[key] == v]
            if not sub:
                continue
            best_wr = max(r["wr"] for r in sub)
            best_pf = max(r["pf"] for r in sub)
            q = sum(1 for r in sub if r["wr"] >= CRITERIA_WR
                    and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD)
            lbl = f"{key}={v}"
            print(f"  {lbl:<25}  最高WR: {best_wr:5.1f}%  最高PF: {best_pf:4.2f}"
                  f"  合格: {q}/{len(sub)}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("最強型グリッドサーチ 開始")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())}"
          f" = {len(list(itertools.product(*GRID.values())))} 通り")
    print("=" * 80)

    # ── データ取得 ──
    raw_data = build_or_load_cache()

    # ── 株数取得（時価総額用）──
    tickers = list(raw_data.keys())
    print(f"\n株数取得中: {len(tickers)} 銘柄...")
    shares_map = fetch_all_shares(tickers)

    # ── 前処理 ──
    print(f"\n前処理中...")
    all_proc: list[dict] = []
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(tk, df_raw, shares_map.get(tk))
        if r is not None:
            all_proc.append(r)
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)}  有効: {len(all_proc)}")
    print(f"  完了: 有効 {len(all_proc)} 銘柄")

    # ── 取引日数推定 ──
    dates: set = set()
    for td in all_proc[:50]:
        # dfのインデックスは保持していないので行数ベースで推定
        pass
    # raw_dataのインデックスから推定
    sample_dfs = list(raw_data.values())[:30]
    all_dates: set = set()
    for df in sample_dfs:
        all_dates.update(df.index.tolist())
    trading_days = len(all_dates)
    print(f"推定取引日数: {trading_days} 日")

    # ── グリッドサーチ ──
    results = run_grid(all_proc, trading_days)

    # ── 結果表示 ──
    print_results(results)

    print(f"\n完了")


if __name__ == "__main__":
    main()
