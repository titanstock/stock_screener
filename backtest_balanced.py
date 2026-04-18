#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バランス最適化グリッドサーチ
=============================
勝率・PF・ヒット件数の折り合いが最良の条件を探す

グリッド拡張:
  - RSI上限: 35→55（ヒット件数拡大）
  - 連続下落フラグ: 2日以上連続で下落後の反発（精度向上）
  - 出来高フィルター: None / 1.5x / 2x / 3x
  - MA25乖離: None / -3% / -5% / -10%

複合スコア = WR × PF × sqrt(件/日)  ← 3軸バランス最大化
評価基準   = WR≥55% / PF≥1.5 / 件/日≥0.3
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
CACHE_MAX_AGE = 3
JPX_LIST_URL  = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JQUANTS_BASE  = "https://api.jquants.com/v1"
JQUANTS_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN", "")

MIN_HISTORY   = 100
MAX_HOLD      = 20
MAX_WORKERS   = 20
MIN_TURNOVER  = 30_000_000

# ── 評価基準 ──────────────────────────────────────────────────────────────────
CRITERIA_WR  = 55.0    # 勝率
CRITERIA_PF  = 1.5     # プロフィットファクター
CRITERIA_SPD = 0.3     # 件/日

# ── グリッド ──────────────────────────────────────────────────────────────────
GRID = {
    "rsi_hi":       [35.0, 40.0, 45.0, 50.0, 55.0],  # RSI上限 (5値)
    "pct_thr":      [2.0, 3.0, 5.0],                   # 前日比下限 (3値)
    "cons_down":    [0, 1, 2],                          # 連続下落日数 0=制限なし (3値)
    "vol_mult":     [0.0, 1.5, 2.0, 3.0],              # 出来高倍率 0=制限なし (4値)
    "ma25_dev":     [None, -3.0, -5.0, -10.0],         # MA25乖離上限 (4値)
    "atr_expand":   [False, True],                      # ATR拡大 (2値)
    "mktcap_max":   [100e9, 200e9, None],               # 時価総額上限 (3値)
    "rr":           [1.5, 2.0, 2.5],                    # RR (3値)
}
# 5×3×3×4×4×2×3×3 = 6,480 通り

_RSI_KEY  = {35.0: "r35", 40.0: "r40", 45.0: "r45", 50.0: "r50", 55.0: "r55"}
_PCT_KEY  = {2.0: "p2", 3.0: "p3", 5.0: "p5"}
_VOL_KEY  = {1.5: "v15", 2.0: "v20", 3.0: "v30"}
_MA25_KEY = {-3.0: "m3", -5.0: "m5", -10.0: "m10"}
_MC_KEY   = {100e9: "c100", 200e9: "c200"}
_CD_KEY   = {1: "cd1", 2: "cd2"}   # 連続下落


# ══════════════════════════════════════════════════════════════════════════════
# データ取得・キャッシュ管理（backtest_strongest.py と共通）
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
    code = ticker.replace(".T", "")
    url = f"{JQUANTS_BASE}/prices/daily_quotes"
    headers = {"Authorization": f"Bearer {token}"}
    from_d  = (date.today() - timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        r = requests.get(url, params={"code": code, "from": from_d},
                         headers=headers, timeout=30)
        r.raise_for_status()
        rows = r.json().get("daily_quotes", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _fetch_one(ticker: str, token: str) -> tuple[str, pd.DataFrame | None]:
    try:
        yf_df = yf.download(ticker, period="3y", interval="1d",
                            auto_adjust=True, progress=False, timeout=20)
        if yf_df is not None and len(yf_df) >= MIN_HISTORY:
            yf_df.columns = [c[0] if isinstance(c, tuple) else c
                             for c in yf_df.columns]
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
        cache_date = cached.get("date", "")
        cutoff = (date.today() - timedelta(days=CACHE_MAX_AGE)).isoformat()
        if cache_date >= cutoff:
            print(f"キャッシュ利用（{cache_date}）: {len(cached['data'])} 銘柄")
            return cached["data"]
        print(f"キャッシュ期限切れ（{cache_date}）→ 再取得")
        raw_data: dict[str, pd.DataFrame] = cached.get("data", {})
    else:
        print("キャッシュなし → 全件取得")
        raw_data = {}

    tickers = _fetch_jpx_tickers()
    to_fetch = [t for t in tickers if t not in raw_data]
    print(f"取得対象: {len(to_fetch)} 銘柄（既存: {len(raw_data)}）")

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

    print(f"キャッシュ保存中: {len(raw_data)} 銘柄...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print("  保存完了")
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
# 前処理（銘柄ごとに特徴量フラグを事前計算）
# ══════════════════════════════════════════════════════════════════════════════

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

    # RSI(14)
    rsi = calc_rsi(pd.Series(c)).values

    # 前日比(%)
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    pct = np.where(prev_c > 0, (c - prev_c) / prev_c * 100, np.nan)

    # ATR(14)
    prev_c2 = np.roll(c, 1); prev_c2[0] = np.nan
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c2), np.abs(l - prev_c2)])
    atr = pd.Series(tr).rolling(14).mean().values

    # ATR拡大（直近3日平均 > 前3日平均）
    atr_s      = pd.Series(atr)
    atr3d      = atr_s.rolling(3).mean().values
    atr3d_prev = atr_s.shift(3).rolling(3).mean().values
    atr_exp = (
        (atr3d > atr3d_prev)
        & ~np.isnan(atr3d) & ~np.isnan(atr3d_prev)
        & (atr3d_prev > 0)
    )

    # MA25乖離率(%)
    ma25     = pd.Series(c).rolling(25).mean().values
    ma25_dev = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)

    # 出来高倍率（20日移動平均比）
    avg_v = pd.Series(v).rolling(20).mean().values
    vol_r = np.where(avg_v > 0, v / avg_v, np.nan)

    # 平均売買代金(20日)
    avg_to = pd.Series(c * v).rolling(20).mean().values

    # 時価総額
    mktcap = c * shares if shares else np.full(n, np.nan)

    # ── 連続下落フラグ ─────────────────────────────────────────────────────────
    # 当日は上昇(pct>0)で、直前1日が下落（cd1）
    # 当日は上昇(pct>0)で、直前2日が連続下落（cd2）
    down = (pct < 0) & ~np.isnan(pct)
    up   = (pct > 0) & ~np.isnan(pct)

    # cd1: 前日下落 & 今日上昇
    prev_down1 = np.roll(down, 1); prev_down1[0] = False
    cd1 = up & prev_down1

    # cd2: 前日・前々日ともに下落 & 今日上昇
    prev_down2 = np.roll(down, 2); prev_down2[:2] = False
    cd2 = up & prev_down1 & prev_down2

    # ── ベースフィルター ────────────────────────────────────────────────────────
    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi)) & (~np.isnan(pct)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    # ── 事前計算フラグ ──────────────────────────────────────────────────────────
    flags: dict[str, np.ndarray] = {
        "base":    base,
        # RSI上限
        "r35": (rsi <= 35.0) & ~np.isnan(rsi),
        "r40": (rsi <= 40.0) & ~np.isnan(rsi),
        "r45": (rsi <= 45.0) & ~np.isnan(rsi),
        "r50": (rsi <= 50.0) & ~np.isnan(rsi),
        "r55": (rsi <= 55.0) & ~np.isnan(rsi),
        # 前日比
        "p2": (pct >= 2.0) & ~np.isnan(pct),
        "p3": (pct >= 3.0) & ~np.isnan(pct),
        "p5": (pct >= 5.0) & ~np.isnan(pct),
        # 連続下落
        "cd1": cd1,
        "cd2": cd2,
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


def _get_mask(flags: dict, rsi_hi, pct_thr, cons_down,
              vol_mult, ma25_dev, atr_expand, mktcap_max) -> np.ndarray:
    m = flags["base"] & flags[_RSI_KEY[rsi_hi]] & flags[_PCT_KEY[pct_thr]]
    if cons_down >= 1:
        m = m & flags[_CD_KEY[cons_down]]
    if vol_mult > 0.0:
        m = m & flags[_VOL_KEY[vol_mult]]
    if ma25_dev is not None:
        m = m & flags[_MA25_KEY[ma25_dev]]
    if atr_expand:
        m = m & flags["atr_exp"]
    if mktcap_max is not None:
        m = m & flags[_MC_KEY[mktcap_max]]
    return m


def _metrics(rets: np.ndarray, trading_days: int) -> dict:
    if len(rets) < 5:
        return {"n": len(rets), "wr": 0.0, "pf": 0.0, "ev": 0.0, "spd": 0.0,
                "avg_w": 0.0, "avg_l": 0.0, "score": 0.0}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets) * 100
    avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = len(rets) / trading_days if trading_days > 0 else 0.0
    # 複合スコア = WR × PF × sqrt(件/日)  ← 3軸バランス
    score  = (wr / 100) * pf * (spd ** 0.5)
    return {"n": len(rets), "wr": wr, "pf": pf, "ev": ev, "spd": spd,
            "avg_w": avg_w, "avg_l": avg_l, "score": score}


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
                p["rsi_hi"], p["pct_thr"], p["cons_down"],
                p["vol_mult"], p["ma25_dev"], p["atr_expand"], p["mktcap_max"],
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

        m = _metrics(np.array(all_rets), trading_days)
        results.append({**p, **m})

        if ci % 500 == 0 or ci == n_combos:
            passed = sum(
                1 for r in results
                if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                and r["spd"] >= CRITERIA_SPD
            )
            elapsed = time.time() - t0
            eta     = elapsed / ci * (n_combos - ci)
            print(f"  {ci:5d}/{n_combos}  合格: {passed}件  "
                  f"経過: {elapsed:.0f}s  残: {eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_params(r: dict) -> str:
    mc   = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else "制限なし"
    ma25 = f"{r['ma25_dev']:+.0f}%" if r["ma25_dev"] is not None else "  なし"
    atr  = "ATR↑" if r["atr_expand"] else "    "
    vol  = f"vol{r['vol_mult']:.1f}x" if r["vol_mult"] > 0 else "vol--"
    cd   = f"cd{r['cons_down']}日" if r["cons_down"] > 0 else "cd--"
    return (
        f"RSI≤{r['rsi_hi']:.0f}  +{r['pct_thr']:.0f}%  {cd}  "
        f"MA25{ma25}  {atr}  {vol}  {mc}  RR{r['rr']}"
    )


def print_results(results: list[dict]) -> None:
    qualified = [
        r for r in results
        if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
        and r["spd"] >= CRITERIA_SPD
    ]

    print(f"\n{'='*90}")
    print(f"【バランス最適化グリッドサーチ 結果】")
    print(f"  総計: {len(results)} 通り  |  合格: {len(qualified)} 通り")
    print(f"  評価基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  件/日≥{CRITERIA_SPD}")
    print(f"  複合スコア = WR × PF × √(件/日)  ← ランキング軸")

    hdr = f"  {'条件':<68}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件/日':>6}  {'Score':>6}"
    sep = "  " + "-" * 103

    if qualified:
        # ★ 複合スコア上位（メイン）
        top_score = sorted(qualified, key=lambda r: r["score"], reverse=True)[:20]
        print(f"\n▶ 合格条件 上位20件【複合スコア順（WR×PF×√件数）】")
        print(hdr); print(sep)
        for r in top_score:
            print(f"  {_fmt_params(r):<68}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # WR重視
        top_wr = sorted(qualified, key=lambda r: r["wr"], reverse=True)[:10]
        print(f"\n▶ WR重視 上位10件")
        print(hdr); print(sep)
        for r in top_wr:
            print(f"  {_fmt_params(r):<68}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # 件数重視
        top_spd = sorted(qualified, key=lambda r: r["spd"], reverse=True)[:10]
        print(f"\n▶ 件数重視 上位10件")
        print(hdr); print(sep)
        for r in top_spd:
            print(f"  {_fmt_params(r):<68}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── ★ パレート前線（WR vs 件/日）──────────────────────────────────────
        print(f"\n▶ パレート前線（WR vs 件/日）")
        print(f"  {'WR区間':<12}  {'最高PF':>6}  {'件/日':>6}  条件")
        print("  " + "-" * 75)
        wr_bands = [(70, 100), (65, 70), (60, 65), (55, 60)]
        for lo, hi in wr_bands:
            band = [r for r in qualified if lo <= r["wr"] < hi]
            if band:
                best = max(band, key=lambda r: r["spd"])
                print(f"  WR {lo:>3}–{hi:<3}%    {best['pf']:>6.2f}  "
                      f"{best['spd']:>5.2f}/日  {_fmt_params(best)}")

        # ── ★ 最良条件 ───────────────────────────────────────────────────────────
        best = max(qualified, key=lambda r: r["score"])
        print(f"\n{'='*90}")
        print(f"★ 推奨条件（複合スコア最大）")
        print(f"  RSI上限    : ≤ {best['rsi_hi']:.0f}")
        print(f"  前日比     : +{best['pct_thr']:.0f}% 以上")
        _cd_map = {0: "なし", 1: "1日連続下落後", 2: "2日連続下落後"}
        print(f"  連続下落   : {_cd_map[best['cons_down']]}")
        print(f"  MA25乖離   : {best['ma25_dev']:+.0f}% 以下" if best["ma25_dev"] else "  MA25乖離   : 条件なし")
        print(f"  ATR拡大    : {'あり' if best['atr_expand'] else 'なし'}")
        print(f"  出来高     : {best['vol_mult']:.1f}倍以上" if best["vol_mult"] > 0 else "  出来高     : 条件なし")
        print(f"  時価総額   : {best['mktcap_max']/1e8:.0f}億円以下" if best["mktcap_max"] else "  時価総額   : 条件なし")
        print(f"  RR         : 1:{best['rr']}")
        print(f"  ──────────────────────")
        print(f"  勝率       : {best['wr']:.1f}%")
        print(f"  PF         : {best['pf']:.2f}")
        print(f"  期待値     : {best['ev']:+.2f}%/トレード")
        print(f"  件数/日    : {best['spd']:.2f}件/日")
        print(f"  複合スコア : {best['score']:.4f}")
        print(f"  平均利益   : {best['avg_w']:+.2f}%  平均損失: {best['avg_l']:+.2f}%")

    else:
        print("\n合格条件なし。上位20件（複合スコア順）を表示します。")
        top = sorted(results, key=lambda r: r["score"], reverse=True)[:20]
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt_params(r):<68}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

    # ── 各パラメータ別サマリ ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("【各パラメータ別サマリ（複合スコア平均）】")
    summary_keys = [
        ("rsi_hi",     [(35.0,"RSI35"),(40.0,"RSI40"),(45.0,"RSI45"),
                        (50.0,"RSI50"),(55.0,"RSI55")]),
        ("cons_down",  [(0,"連続なし"),(1,"1日連続"),(2,"2日連続")]),
        ("vol_mult",   [(0.0,"出来高制限なし"),(1.5,"vol1.5x"),
                        (2.0,"vol2.0x"),(3.0,"vol3.0x")]),
        ("ma25_dev",   [(None,"MA25制限なし"),(-3.0,"MA25-3%"),
                        (-5.0,"MA25-5%"),(-10.0,"MA25-10%")]),
        ("atr_expand", [(False,"ATR制限なし"),(True,"ATR拡大")]),
    ]
    for key, vals in summary_keys:
        best_score = 0.0
        for v, lbl in vals:
            sub = [r for r in qualified if r[key] == v] if qualified else []
            if not sub:
                continue
            avg_score = np.mean([r["score"] for r in sub])
            avg_wr    = np.mean([r["wr"]    for r in sub])
            avg_spd   = np.mean([r["spd"]   for r in sub])
            print(f"  {lbl:<20}  合格: {len(sub):4d}件  "
                  f"WR平均: {avg_wr:5.1f}%  件/日平均: {avg_spd:5.2f}  "
                  f"スコア平均: {avg_score:.4f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_combos = 1
    for v in GRID.values():
        total_combos *= len(v)

    print("=" * 90)
    print("バランス最適化グリッドサーチ 開始")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())} = {total_combos} 通り")
    print(f"新機能  : 連続下落フラグ（cd1/cd2）/ RSI上限を55まで拡張")
    print(f"複合スコア: WR × PF × √(件/日)")
    print("=" * 90)

    # ── データ読込 ─────────────────────────────────────────────────────────────
    raw_data = load_cache()

    # ── 株数取得（時価総額フィルター用） ────────────────────────────────────────
    print(f"\n株数取得中: {len(raw_data)} 銘柄...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    # ── 前処理 ─────────────────────────────────────────────────────────────────
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

    # ── グリッドサーチ ──────────────────────────────────────────────────────────
    results = run_grid(all_proc, trading_days)

    # ── 結果表示 ────────────────────────────────────────────────────────────────
    print_results(results)
    print("\n完了")


if __name__ == "__main__":
    main()
