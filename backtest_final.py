#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終統合グリッドサーチ ─ 全バックテスト知見の総合版
=====================================================
【全バックテストから得た知見を統合】
  ✓ RSI14（短期2/3/5日は日本株では無効と判明）
  ✓ IBS (Internal Bar Strength) ─ 安値引けで翌日反発
  ✓ 前日比（価格上昇確認）
  ✓ 出来高（vol制限なしは機能しないと判明 → 必須化）
  ✓ MA25乖離率
  ✓ ATR拡大（補完的フィルター）
  ✓ MA75方向（上昇トレンド中の逆張りは機能しないと判明）
  ✓ 連続下落後フラグ

【旧最強条件】同データで再計算して公平比較
  RSI14≤35 + 前日比+5% + MA25-10% + ATR拡大 + vol制限なし + RR2.5
  → backtest_strongest.py での結果: WR73.4% / PF1.85 / 0.56件/日

グリッド: 5×3×3×4×3×2×2×3×3×3 = 29,160 通り
複合スコア: WR × PF × √(件/日)
評価基準  : WR≥55% / PF≥1.5 / 件/日≥0.3
"""

import itertools, os, pickle, time, warnings
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

# ── 定数 ──────────────────────────────────────────────────────────────────────
CACHE_PATH    = Path(__file__).parent / "backtest_cache.pkl"
CACHE_MAX_AGE = 3
JPX_LIST_URL  = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JQUANTS_BASE  = "https://api.jquants.com/v1"
JQUANTS_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN", "")

MIN_HISTORY  = 100
MAX_HOLD     = 20
MAX_WORKERS  = 20
MIN_TURNOVER = 30_000_000
STOP_COEF    = 2.0
STOP_CAP     = 0.10

# ── 評価基準 ──────────────────────────────────────────────────────────────────
CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.3

# ── グリッド（全知見統合） ────────────────────────────────────────────────────
GRID = {
    "rsi_hi":     [20.0, 25.0, 30.0, 35.0, 40.0],  # RSI14上限    (5)
    "ibs_hi":     [None, 0.3, 0.5],                  # IBS上限      (3)  None=制限なし
    "pct_thr":    [None, 2.0, 5.0],                  # 前日比下限   (3)  None=制限なし
    "vol_mult":   [0.0, 1.5, 2.0, 3.0],              # 出来高倍率   (4)  0=制限なし
    "ma25_dev":   [None, -5.0, -10.0],               # MA25乖離上限 (3)  None=制限なし
    "atr_expand": [False, True],                      # ATR拡大      (2)
    "ma75_below": [False, True],                      # MA75下       (2)
    "cons_down":  [0, 1, 2],                          # 連続下落日数 (3)  0=制限なし
    "rr":         [1.5, 2.0, 2.5],                    # RR比         (3)
    "mktcap_max": [100e9, 200e9, None],               # 時価総額上限 (3)  None=制限なし
}
# 5×3×3×4×3×2×2×3×3×3 = 29,160 通り

# 旧最強条件（公平比較用に同データで再計算する）
OLD_BEST = {
    "rsi_hi": 35.0, "ibs_hi": None, "pct_thr": 5.0,
    "vol_mult": 0.0, "ma25_dev": -10.0, "atr_expand": True,
    "ma75_below": False, "cons_down": 0, "rr": 2.5, "mktcap_max": None,
}

# フラグキーマップ
_RSI_KEY  = {20.0: "r20", 25.0: "r25", 30.0: "r30", 35.0: "r35", 40.0: "r40"}
_IBS_KEY  = {0.3: "ibs03", 0.5: "ibs05"}
_PCT_KEY  = {2.0: "p2", 5.0: "p5"}
_VOL_KEY  = {1.5: "v15", 2.0: "v20", 3.0: "v30"}
_MA25_KEY = {-5.0: "m5", -10.0: "m10"}
_MC_KEY   = {100e9: "c100", 200e9: "c200"}
_CD_KEY   = {1: "cd1", 2: "cd2"}


# ══════════════════════════════════════════════════════════════════════════════
# データ取得・キャッシュ
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_jpx_tickers() -> list[str]:
    resp = requests.get(JPX_LIST_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content), dtype=str)
    mkt_col  = next((c for c in df.columns if "市場" in str(c)), None)
    code_col = next((c for c in df.columns if "コード" in str(c)), None)
    if not mkt_col or not code_col:
        raise RuntimeError("JPXリストの列名が変更されています")
    targets = ["プライム", "スタンダード", "グロース"]
    df = df[df[mkt_col].str.contains("|".join(targets), na=False)]
    codes = df[code_col].str.strip().tolist()
    return [f"{c}.T" for c in codes if c.isdigit() and len(c) == 4]


def _fetch_jquants(ticker: str, token: str) -> pd.DataFrame | None:
    code  = ticker.replace(".T", "")
    from_d = (date.today() - timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"{JQUANTS_BASE}/prices/daily_quotes",
            params={"code": code, "from": from_d},
            headers={"Authorization": f"Bearer {token}"}, timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("daily_quotes", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except Exception:
        return None


def _fetch_one(ticker: str, token: str) -> tuple[str, pd.DataFrame | None]:
    try:
        yf_df = yf.download(ticker, period="3y", interval="1d",
                            auto_adjust=True, progress=False, timeout=20)
        if yf_df is not None and len(yf_df) >= MIN_HISTORY:
            yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
            return ticker, yf_df[["Open","High","Low","Close","Volume"]]
    except Exception:
        pass
    if token:
        df = _fetch_jquants(ticker, token)
        if df is not None and len(df) >= MIN_HISTORY:
            return ticker, df
    return ticker, None


def load_cache() -> dict[str, pd.DataFrame]:
    today_str = date.today().isoformat()
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        cutoff = (date.today() - timedelta(days=CACHE_MAX_AGE)).isoformat()
        if cached.get("date", "") >= cutoff:
            print(f"キャッシュ利用（{cached['date']}）: {len(cached['data'])} 銘柄")
            return cached["data"]
        print("キャッシュ期限切れ → 再取得")
        raw_data = cached.get("data", {})
    else:
        print("キャッシュなし → 全件取得")
        raw_data = {}

    tickers  = _fetch_jpx_tickers()
    to_fetch = [t for t in tickers if t not in raw_data]
    token = ""
    if JQUANTS_TOKEN:
        try:
            token = _get_jquants_id_token(JQUANTS_TOKEN)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, t, token): t for t in to_fetch}
        done = ok = 0
        for fut in as_completed(futs):
            t, df = fut.result()
            done += 1
            if df is not None:
                raw_data[t] = df
                ok += 1
            if done % 300 == 0 or done == len(to_fetch):
                print(f"  {done}/{len(to_fetch)}  成功: {ok}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print(f"キャッシュ保存: {len(raw_data)} 銘柄")
    return raw_data


def fetch_all_shares(tickers: list[str]) -> dict[str, float | None]:
    def _get(t):
        try:
            sh = yf.Ticker(t).fast_info.shares
            return t, float(sh) if sh else None
        except Exception:
            return t, None

    result: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_get, t): t for t in tickers}
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
# 前処理
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi14(prices: np.ndarray) -> np.ndarray:
    period = 14
    n    = len(prices)
    rsi  = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    diff   = np.diff(prices)
    gains  = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    rsi[period] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = avg_g * 13 / 14 + gains[i] / 14
        avg_l = avg_l * 13 / 14 + losses[i] / 14
        rsi[i + 1] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


def preprocess(ticker: str, df_raw: pd.DataFrame,
               shares: float | None) -> dict | None:
    df = df_raw.copy()
    df = df[df["Close"] > 0].dropna(subset=["Open","High","Low","Close","Volume"])
    if len(df) < MIN_HISTORY:
        return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)

    # 翌日始値
    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    # ATR(14)
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr     = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr    = pd.Series(tr).rolling(14).mean().values

    # ATR拡大（直近3日平均 > 前3日平均）
    atr_s      = pd.Series(atr)
    atr3d      = atr_s.rolling(3).mean().values
    atr3d_prev = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (
        (atr3d > atr3d_prev)
        & ~np.isnan(atr3d) & ~np.isnan(atr3d_prev) & (atr3d_prev > 0)
    )

    # IBS
    rng = h - l
    ibs = np.where(rng > 0, (c - l) / rng, np.nan)

    # RSI(14)
    rsi14 = _calc_rsi14(c)

    # 前日比(%)
    pct = np.full(n, np.nan)
    pct[1:] = (c[1:] - c[:-1]) / c[:-1] * 100

    # MA25乖離率
    ma25     = pd.Series(c).rolling(25).mean().values
    ma25_dev = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)

    # MA75との位置
    ma75 = pd.Series(c).rolling(75).mean().values

    # 出来高倍率（20日平均比）
    avg_v = pd.Series(v).rolling(20).mean().values
    vol_r = np.where(avg_v > 0, v / avg_v, np.nan)

    # 平均売買代金(20日)
    avg_to = pd.Series(c * v).rolling(20).mean().values

    # 時価総額
    mktcap = c * shares if shares else np.full(n, np.nan)

    # 連続下落フラグ
    down = (pct < 0) & ~np.isnan(pct)
    up   = (pct > 0) & ~np.isnan(pct)
    prev_down1 = np.roll(down, 1); prev_down1[0] = False
    prev_down2 = np.roll(down, 2); prev_down2[:2] = False
    cd1 = up & prev_down1
    cd2 = up & prev_down1 & prev_down2

    # ベースフィルター
    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi14)) & (~np.isnan(pct)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    # 事前計算フラグ
    flags: dict[str, np.ndarray] = {
        "base": base,
        # RSI14
        "r20": (rsi14 <= 20.0) & ~np.isnan(rsi14),
        "r25": (rsi14 <= 25.0) & ~np.isnan(rsi14),
        "r30": (rsi14 <= 30.0) & ~np.isnan(rsi14),
        "r35": (rsi14 <= 35.0) & ~np.isnan(rsi14),
        "r40": (rsi14 <= 40.0) & ~np.isnan(rsi14),
        # IBS
        "ibs03": (ibs <= 0.3) & ~np.isnan(ibs),
        "ibs05": (ibs <= 0.5) & ~np.isnan(ibs),
        # 前日比
        "p2": (pct >= 2.0)  & ~np.isnan(pct),
        "p5": (pct >= 5.0)  & ~np.isnan(pct),
        # 出来高
        "v15": (vol_r >= 1.5) & ~np.isnan(vol_r),
        "v20": (vol_r >= 2.0) & ~np.isnan(vol_r),
        "v30": (vol_r >= 3.0) & ~np.isnan(vol_r),
        # MA25乖離
        "m5":  (ma25_dev <= -5.0)  & ~np.isnan(ma25_dev),
        "m10": (ma25_dev <= -10.0) & ~np.isnan(ma25_dev),
        # ATR拡大
        "atr_exp": atr_exp,
        # MA75下
        "ma75_below": (c < ma75) & ~np.isnan(ma75),
        # 連続下落
        "cd1": cd1,
        "cd2": cd2,
        # 時価総額
        "c100": np.isnan(mktcap) | (mktcap <= 100e9),
        "c200": np.isnan(mktcap) | (mktcap <= 200e9),
    }

    return {"close": c, "next_o": next_o, "atr": atr, "flags": flags, "n": n}


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


def _get_mask(flags: dict, p: dict) -> np.ndarray:
    m = flags["base"] & flags[_RSI_KEY[p["rsi_hi"]]]
    if p["ibs_hi"] is not None:
        m = m & flags[_IBS_KEY[p["ibs_hi"]]]
    if p["pct_thr"] is not None:
        m = m & flags[_PCT_KEY[p["pct_thr"]]]
    if p["vol_mult"] > 0.0:
        m = m & flags[_VOL_KEY[p["vol_mult"]]]
    if p["ma25_dev"] is not None:
        m = m & flags[_MA25_KEY[p["ma25_dev"]]]
    if p["atr_expand"]:
        m = m & flags["atr_exp"]
    if p["ma75_below"]:
        m = m & flags["ma75_below"]
    if p["cons_down"] >= 1:
        m = m & flags[_CD_KEY[p["cons_down"]]]
    if p["mktcap_max"] is not None:
        m = m & flags[_MC_KEY[p["mktcap_max"]]]
    return m


def _eval(all_proc: list[dict], p: dict, trading_days: int) -> dict:
    all_rets: list[float] = []
    for td in all_proc:
        m    = _get_mask(td["flags"], p)
        vidx = np.where(m)[0]
        if len(vidx) == 0:
            continue
        e = td["next_o"][vidx]
        a = td["atr"][vidx]
        s = np.maximum(e - a * STOP_COEF, e * (1 - STOP_CAP))
        t = e + (e - s) * p["rr"]
        valid = (e > 0) & (s > 0) & (t > 0) & (s < e) & (t > e)
        if not valid.any():
            continue
        rets = _calc_rets(td["close"], vidx[valid], e[valid], s[valid], t[valid])
        all_rets.extend(rets.tolist())

    rets = np.array(all_rets)
    if len(rets) < 5:
        return {**p, "n": len(rets), "wr": 0., "pf": 0., "ev": 0.,
                "spd": 0., "avg_w": 0., "avg_l": 0., "score": 0.}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets) * 100
    avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.
    avg_l  = float(losses.mean()) if len(losses) > 0 else 0.
    pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = len(rets) / trading_days
    score  = (wr / 100) * pf * (spd ** 0.5)
    return {**p, "n": len(rets), "wr": wr, "pf": pf, "ev": ev,
            "spd": spd, "avg_w": avg_w, "avg_l": avg_l, "score": score}


# ══════════════════════════════════════════════════════════════════════════════
# グリッドサーチ
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(all_proc: list[dict], trading_days: int) -> list[dict]:
    combos     = list(itertools.product(*GRID.values()))
    combo_keys = list(GRID.keys())
    n_combos   = len(combos)
    results    = []

    print(f"\nグリッドサーチ: {n_combos} 通り × {len(all_proc)} 銘柄")
    t0 = time.time()

    for ci, combo_vals in enumerate(combos, 1):
        p = dict(zip(combo_keys, combo_vals))

        # 無条件買い（IBS/RSI/pct/vol/MA 全てなし）はスキップ
        if (p["ibs_hi"] is None and p["pct_thr"] is None and
                p["vol_mult"] == 0.0 and p["ma25_dev"] is None and
                not p["atr_expand"] and not p["ma75_below"] and p["cons_down"] == 0):
            continue

        results.append(_eval(all_proc, p, trading_days))

        if ci % 1000 == 0 or ci == n_combos:
            passed  = sum(1 for r in results
                          if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                          and r["spd"] >= CRITERIA_SPD)
            elapsed = time.time() - t0
            eta     = elapsed / ci * (n_combos - ci)
            print(f"  {ci:6d}/{n_combos}  合格: {passed}件  "
                  f"経過: {elapsed:.0f}s  残: {eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(r: dict) -> str:
    ibs  = f"IBS≤{r['ibs_hi']}" if r["ibs_hi"] is not None else "IBS--"
    pct  = f"+{r['pct_thr']:.0f}%" if r["pct_thr"] is not None else "pct--"
    vol  = f"vol{r['vol_mult']:.1f}x" if r["vol_mult"] > 0 else "vol--"
    ma25 = f"MA25{r['ma25_dev']:+.0f}%" if r["ma25_dev"] is not None else "MA25--"
    atr  = "ATR↑" if r["atr_expand"] else "    "
    m75  = "MA75下" if r["ma75_below"] else "     "
    cd   = f"cd{r['cons_down']}日" if r["cons_down"] > 0 else "    "
    mc   = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else " 制限なし"
    return (f"RSI≤{r['rsi_hi']:.0f} {ibs} {pct} {vol} "
            f"{ma25} {atr} {m75} {cd} {mc} RR{r['rr']}")


def _show_box(title: str, r: dict) -> None:
    ibs  = f"≤ {r['ibs_hi']}" if r["ibs_hi"] is not None else "なし"
    pct  = f"≥ +{r['pct_thr']:.0f}%" if r["pct_thr"] is not None else "なし"
    vol  = f"≥ {r['vol_mult']:.1f}倍" if r["vol_mult"] > 0 else "なし"
    ma25 = f"≤ {r['ma25_dev']:+.0f}%" if r["ma25_dev"] is not None else "なし"
    cd_map = {0: "なし", 1: "1日連続下落後", 2: "2日連続下落後"}
    mc   = f"{r['mktcap_max']/1e8:.0f}億円以下" if r["mktcap_max"] else "なし"
    print(f"\n  {'─'*55}")
    print(f"  {title}")
    print(f"  {'─'*55}")
    print(f"  RSI(14) ≤ {r['rsi_hi']:.0f}")
    print(f"  IBS       : {ibs}")
    print(f"  前日比    : {pct}")
    print(f"  出来高    : {vol}")
    print(f"  MA25乖離  : {ma25}")
    print(f"  ATR拡大   : {'あり' if r['atr_expand'] else 'なし'}")
    print(f"  MA75下    : {'あり' if r['ma75_below'] else 'なし'}")
    print(f"  連続下落  : {cd_map[r['cons_down']]}")
    print(f"  時価総額  : {mc}")
    print(f"  RR        : 1:{r['rr']}")
    print(f"  {'─'*55}")
    print(f"  勝率      : {r['wr']:.1f}%")
    print(f"  PF        : {r['pf']:.2f}")
    print(f"  期待値    : {r['ev']:+.2f}%/トレード")
    print(f"  件数/日   : {r['spd']:.2f}件/日")
    print(f"  複合Score : {r['score']:.4f}")
    print(f"  平均利益  : {r['avg_w']:+.2f}%  平均損失: {r['avg_l']:+.2f}%")
    print(f"  サンプル  : {r['n']:,}件")


def print_results(results: list[dict], old_best_result: dict) -> None:
    qualified = [
        r for r in results
        if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
        and r["spd"] >= CRITERIA_SPD
    ]

    hdr = (f"  {'条件':<76}  {'勝率':>6}  {'PF':>5}  {'EV':>7}"
           f"  {'件/日':>6}  {'Score':>6}")
    sep = "  " + "-" * 112

    print(f"\n{'='*115}")
    print(f"【最終統合グリッドサーチ 結果】")
    print(f"  総計: {len(results)} 通り  |  合格: {len(qualified)} 通り")
    print(f"  基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  件/日≥{CRITERIA_SPD}")
    print(f"  複合スコア = WR × PF × √(件/日)")

    if not qualified:
        print("\n合格条件なし。上位20件（Score順）:")
        top = sorted(results, key=lambda r: r["score"], reverse=True)[:20]
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt(r):<76}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")
    else:
        # ── 複合スコア上位20件 ────────────────────────────────────────────────
        top_score = sorted(qualified, key=lambda r: r["score"], reverse=True)[:20]
        print(f"\n▶ 複合スコア上位20件")
        print(hdr); print(sep)
        for r in top_score:
            print(f"  {_fmt(r):<76}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── WR上位10件 ────────────────────────────────────────────────────────
        top_wr = sorted(qualified, key=lambda r: r["wr"], reverse=True)[:10]
        print(f"\n▶ WR重視 上位10件")
        print(hdr); print(sep)
        for r in top_wr:
            print(f"  {_fmt(r):<76}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── 件数上位10件 ──────────────────────────────────────────────────────
        top_spd = sorted(qualified, key=lambda r: r["spd"], reverse=True)[:10]
        print(f"\n▶ 件数重視 上位10件")
        print(hdr); print(sep)
        for r in top_spd:
            print(f"  {_fmt(r):<76}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── パレート前線 ──────────────────────────────────────────────────────
        print(f"\n▶ パレート前線（WR帯別 最多件数）")
        print(f"  {'WR帯':<12}  {'件/日':>6}  {'PF':>5}  {'EV':>7}  条件")
        print("  " + "-" * 95)
        for lo, hi in [(80,100),(75,80),(70,75),(65,70),(60,65),(55,60)]:
            band = [r for r in qualified if lo <= r["wr"] < hi]
            if band:
                best = max(band, key=lambda r: r["spd"])
                print(f"  WR{lo:>3}〜{hi}%  {best['spd']:>5.2f}/日"
                      f"  {best['pf']:>5.2f}  {best['ev']:>+6.2f}%  {_fmt(best)}")

    # ══════════════════════════════════════════════════════════════════════════
    # ★ 新旧最強条件 比較
    # ══════════════════════════════════════════════════════════════════════════
    new_best = max(qualified, key=lambda r: r["score"]) if qualified else None

    print(f"\n{'='*115}")
    print(f"★★ 新旧最強条件 直接比較（同一データ・同一バックテスト手法）")

    _show_box("【旧最強】RSI35 + 前日比+5% + MA25-10% + ATR拡大 (backtest_strongest.py と同条件)",
              old_best_result)

    if new_best:
        _show_box("【新最強】全バックテスト知見統合 (backtest_final.py 新型)",
                  new_best)

        print(f"\n  {'─'*55}")
        print(f"  【比較サマリ】")
        print(f"  {'指標':<12}  {'旧最強':>10}  {'新最強':>10}  {'差分':>10}")
        print(f"  {'─'*55}")
        metrics = [
            ("勝率",   "wr",  "%",   1),
            ("PF",     "pf",  "",    2),
            ("期待値", "ev",  "%",   2),
            ("件数/日","spd", "件",  2),
            ("Score",  "score","",   4),
        ]
        for lbl, key, unit, dec in metrics:
            old_v = old_best_result[key]
            new_v = new_best[key]
            diff  = new_v - old_v
            sign  = "+" if diff >= 0 else ""
            fmt   = f".{dec}f"
            print(f"  {lbl:<12}  {old_v:>9.{dec}f}{unit}  {new_v:>9.{dec}f}{unit}"
                  f"  {sign}{diff:>{dec+6}.{dec}f}{unit}")
        print(f"  {'─'*55}")

        # 総合判定
        print(f"\n  【総合判定】")
        if new_best["score"] > old_best_result["score"]:
            if new_best["wr"] >= old_best_result["wr"]:
                verdict = "新型が WR・Score 両方で旧最強を上回った"
            elif new_best["spd"] > old_best_result["spd"] * 2:
                verdict = (f"新型は WR で劣るが件数が"
                           f"{new_best['spd']/old_best_result['spd']:.1f}倍 → Score で上回る")
            else:
                verdict = "新型が複合スコアで旧最強を上回った"
        else:
            if old_best_result["wr"] > new_best["wr"]:
                verdict = (f"旧最強が WR {old_best_result['wr']:.1f}% で依然最強。"
                           f"新型は件数 {new_best['spd']:.2f}/日 で補完的")
            else:
                verdict = "旧最強が複合スコアで依然優位"
        print(f"  → {verdict}")

    # ── 各パラメータ別サマリ ──────────────────────────────────────────────────
    if qualified:
        print(f"\n{'='*115}")
        print("【各パラメータ別サマリ（合格条件のみ）】")

        def _show(key, vals_labels):
            print(f"\n  [{key}]")
            for v, lbl in vals_labels:
                sub = [r for r in qualified if r[key] == v]
                if not sub:
                    print(f"    {lbl:<25}  合格: 0件")
                    continue
                print(f"    {lbl:<25}  合格: {len(sub):5d}件  "
                      f"WR: {np.mean([r['wr'] for r in sub]):5.1f}%  "
                      f"件/日: {np.mean([r['spd'] for r in sub]):5.2f}  "
                      f"Score: {np.mean([r['score'] for r in sub]):.4f}")

        _show("rsi_hi",    [(20., "RSI≤20"), (25., "RSI≤25"), (30., "RSI≤30"),
                             (35., "RSI≤35"), (40., "RSI≤40")])
        _show("ibs_hi",    [(None,"IBS なし"), (0.3,"IBS≤0.3"), (0.5,"IBS≤0.5")])
        _show("pct_thr",   [(None,"前日比 なし"), (2., "前日比≥+2%"), (5., "前日比≥+5%")])
        _show("vol_mult",  [(0., "vol なし"), (1.5,"vol≥1.5x"),
                             (2., "vol≥2.0x"), (3., "vol≥3.0x")])
        _show("atr_expand",[(False,"ATR制限なし"), (True,"ATR拡大")])
        _show("ma75_below",[(False,"MA75 なし"), (True,"MA75下")])
        _show("cons_down", [(0,"連続下落 なし"), (1,"1日連続"), (2,"2日連続")])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_combos = 1
    for v in GRID.values():
        total_combos *= len(v)

    print("=" * 115)
    print("最終統合グリッドサーチ 開始")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())} = {total_combos} 通り")
    print("目的    : 全バックテスト知見を統合し、旧最強との公平比較を行う")
    print("=" * 115)

    raw_data   = load_cache()
    print(f"\n株数取得中: {len(raw_data)} 銘柄...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中...")
    all_proc: list[dict] = []
    for i, (ticker, df_raw) in enumerate(raw_data.items(), 1):
        td = preprocess(ticker, df_raw, shares_map.get(ticker))
        if td is not None:
            all_proc.append(td)
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)}  有効: {len(all_proc)}")
    print(f"  完了: {len(all_proc)} 銘柄")

    trading_days = int(np.mean([td["n"] for td in all_proc]))
    print(f"推定取引日数: {trading_days} 日")

    # 旧最強条件を同データで再計算
    print("\n旧最強条件を同データで再計算中...")
    old_best_result = _eval(all_proc, OLD_BEST, trading_days)
    print(f"  旧最強: WR={old_best_result['wr']:.1f}%  "
          f"PF={old_best_result['pf']:.2f}  "
          f"EV={old_best_result['ev']:+.2f}%  "
          f"{old_best_result['spd']:.2f}件/日")

    results = run_grid(all_proc, trading_days)
    print_results(results, old_best_result)
    print("\n完了")


if __name__ == "__main__":
    main()
