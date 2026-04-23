#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出来高爆発モメンタム型 グリッドサーチ
======================================
売られすぎ反発型とは根本的に異なる原理：
  「出来高の爆発」を主シグナルとし RSI 制約を外す

グリッド:
  vol_mult × pct_thr × rsi_range × ma25_pos × mktcap_max × rr
  4       × 3       × 4         × 3        × 4          × 3 = 1,728 通り

データ: backtest_cache.pkl（当日キャッシュなら即利用、古ければ yfinance+J-Quants 補完）
評価:  WR≥55% / PF≥1.6 / 件/日≥0.5
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
MIN_HISTORY   = 60
MAX_HOLD      = 20
MAX_WORKERS   = 20
MIN_TURNOVER  = 30_000_000

# ── 評価基準（件数重視でやや緩める） ─────────────────────────────────────────
CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.6
CRITERIA_SPD = 0.5   # 1日0.5件以上

# ── グリッド ──────────────────────────────────────────────────────────────────
GRID = {
    "vol_mult":  [3.0, 5.0, 7.0, 10.0],
    "pct_thr":   [2.0, 3.0, 5.0],
    # (rsi_lo, rsi_hi) — None は制限なし
    "rsi_range": [
        (None, None),    # 制限なし
        (30.0, 60.0),    # 回復途上（やや売られすぎから中立）
        (40.0, 70.0),    # 中立〜強気
        (50.0, 80.0),    # 強気（モメンタム圏）
    ],
    # close vs MA25: None=制限なし / "above"=MA25上 / "below"=MA25下
    "ma25_pos":  [None, "above", "below"],
    "mktcap_max": [30e9, 100e9, 200e9, 300e9, None],
    "rr":        [1.5, 2.0, 2.5],
}


# ══════════════════════════════════════════════════════════════════════════════
# キャッシュ管理
# ══════════════════════════════════════════════════════════════════════════════

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
    try:
        df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if df is not None and len(df) >= MIN_HISTORY:
            return ticker, df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    if JQUANTS_TOKEN:
        df = _fetch_jquants(code4)
        if df is not None and len(df) >= MIN_HISTORY:
            return ticker, df
    return ticker, None


def build_or_load_cache() -> dict[str, pd.DataFrame]:
    today_str = date.today().isoformat()
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                stored = pickle.load(f)
            cache_date = stored.get("date", "")
            raw_data   = stored.get("data", {})
            cache_age  = (date.today() - date.fromisoformat(cache_date)).days
            if cache_date == today_str:
                print(f"キャッシュ利用（当日）: {len(raw_data)} 銘柄")
                return raw_data
            print(f"キャッシュ: {cache_date}（{cache_age}日前）→ 差分補完")
        except Exception:
            raw_data = {}
            cache_age = 999
    else:
        raw_data  = {}
        cache_age = 999
        print("キャッシュなし → 全件取得")

    # 銘柄リスト
    print("JPX銘柄リスト取得中...")
    try:
        resp = requests.get(JPX_LIST_URL, timeout=30)
        resp.raise_for_status()
        df_jpx = pd.read_excel(BytesIO(resp.content), dtype=str)
        mkt_col  = next((c for c in df_jpx.columns if "市場" in str(c)), None)
        code_col = next((c for c in df_jpx.columns if "コード" in str(c)), None)
        mask = df_jpx[mkt_col].str.contains(
            r"(?=.*(?:スタンダード|グロース))(?=.*内国株式)", na=False, regex=True
        )
        tickers = (df_jpx[mask][code_col].str.strip().str.zfill(4) + ".T").tolist()
        print(f"  {len(tickers)} 銘柄")
    except Exception as e:
        print(f"  リスト取得失敗: {e} → 既存キャッシュで続行")
        return raw_data

    to_fetch = [t for t in tickers if t not in raw_data] if cache_age <= CACHE_MAX_AGE else tickers
    if not to_fetch:
        return raw_data

    print(f"データ取得中: {len(to_fetch)} 銘柄...")
    done = ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in to_fetch}
        for fut in as_completed(futs):
            t, df = fut.result()
            done += 1
            if df is not None:
                raw_data[t] = df; ok += 1
            if done % 300 == 0 or done == len(to_fetch):
                print(f"  {done}/{len(to_fetch)} 成功: {ok}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print(f"キャッシュ保存: {len(raw_data)} 銘柄")
    return raw_data


# ══════════════════════════════════════════════════════════════════════════════
# 株数取得
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_shares(ticker: str) -> tuple[str, float | None]:
    try:
        sh = yf.Ticker(ticker).fast_info.shares
        return ticker, float(sh) if sh else None
    except Exception:
        return ticker, None


def fetch_all_shares(tickers: list[str]) -> dict[str, float | None]:
    res: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_shares, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            t, sh = fut.result()
            res[t] = sh
            done += 1
            if done % 300 == 0 or done == len(tickers):
                print(f"  {done}/{len(tickers)} 取得: {sum(1 for v in res.values() if v)}")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 前処理
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

    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    # RSI
    rsi = calc_rsi(pd.Series(c)).values

    # 前日比
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    pct = np.where(prev_c > 0, (c - prev_c) / prev_c * 100, np.nan)

    # ATR (ストップ計算用)
    prev_c2 = np.roll(c, 1); prev_c2[0] = np.nan
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c2), np.abs(l - prev_c2)])
    atr = pd.Series(tr).rolling(14).mean().values

    # MA25
    ma25 = pd.Series(c).rolling(25).mean().values

    # 出来高倍率（20日平均比）
    avg_v = pd.Series(v).rolling(20).mean().values
    vol_r = np.where(avg_v > 0, v / avg_v, np.nan)

    # 平均売買代金
    avg_to = pd.Series(c * v).rolling(20).mean().values

    # 時価総額
    mktcap = c * shares if shares else np.full(n, np.nan)

    # ベースフィルター
    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(vol_r)) &
        (~np.isnan(pct)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    # ── 事前フラグ ──
    flags: dict[str, np.ndarray] = {
        "base": base,
        # 出来高倍率
        "v3":  (vol_r >= 3.0)  & ~np.isnan(vol_r),
        "v5":  (vol_r >= 5.0)  & ~np.isnan(vol_r),
        "v7":  (vol_r >= 7.0)  & ~np.isnan(vol_r),
        "v10": (vol_r >= 10.0) & ~np.isnan(vol_r),
        # 前日比
        "p2": (pct >= 2.0) & ~np.isnan(pct),
        "p3": (pct >= 3.0) & ~np.isnan(pct),
        "p5": (pct >= 5.0) & ~np.isnan(pct),
        # RSI範囲
        "rsi_arr": rsi,
        # MA25位置
        "above_ma25": (c > ma25) & ~np.isnan(ma25),
        "below_ma25": (c < ma25) & ~np.isnan(ma25),
        # 時価総額
        "c30":  np.isnan(mktcap) | (mktcap <= 30e9),
        "c100": np.isnan(mktcap) | (mktcap <= 100e9),
        "c200": np.isnan(mktcap) | (mktcap <= 200e9),
        "c300": np.isnan(mktcap) | (mktcap <= 300e9),
    }

    return {
        "close":  c,
        "high":   h,
        "low":    l,
        "open":   o,
        "next_o": next_o,
        "atr":    atr,
        "flags":  flags,
        "n":      n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# リターン計算
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rets(highs, lows, opens, closes, vidx, e_arr, s_arr, t_arr):
    n        = len(closes)
    raw_idx  = vidx[:, np.newaxis] + np.arange(1, MAX_HOLD + 1)
    in_range = raw_idx < n
    safe_idx = np.where(in_range, raw_idx, n - 1)
    fut_h    = np.where(in_range, highs[safe_idx], np.nan)
    fut_l    = np.where(in_range, lows[safe_idx], np.nan)
    fut_o    = np.where(in_range, opens[safe_idx], np.nan)
    hit_sl   = ((fut_l <= s_arr[:, np.newaxis]) | (fut_o <= s_arr[:, np.newaxis])) & in_range
    hit_tp   = ((fut_h >= t_arr[:, np.newaxis]) | (fut_o >= t_arr[:, np.newaxis])) & in_range
    hit      = hit_sl | hit_tp
    has_hit  = hit.any(axis=1)
    has_fut  = in_range.any(axis=1)
    last_v   = np.where(has_fut, np.sum(in_range, axis=1) - 1, 0)
    fhp      = np.clip(np.where(has_hit, np.argmax(hit, axis=1), last_v), 0, MAX_HOLD - 1)
    exit_idx = np.clip(vidx + 1 + fhp, 0, n - 1)
    ex_o     = opens[exit_idx]
    ex_sl    = has_hit & ((lows[exit_idx] <= s_arr) | (ex_o <= s_arr))
    sl_price = np.where(ex_o <= s_arr, ex_o, s_arr)
    ep       = np.where(~has_hit, closes[exit_idx], np.where(ex_sl, sl_price, t_arr))
    rets     = np.where(has_fut, (ep - e_arr) / e_arr * 100, np.nan)
    return rets[~np.isnan(rets)]


_VOL_KEY  = {3.0: "v3", 5.0: "v5", 7.0: "v7", 10.0: "v10"}
_PCT_KEY  = {2.0: "p2", 3.0: "p3", 5.0: "p5"}
_MC_KEY   = {30e9: "c30", 100e9: "c100", 200e9: "c200", 300e9: "c300"}


def _get_mask(flags, vol_mult, pct_thr, rsi_range, ma25_pos, mktcap_max):
    m = flags["base"] & flags[_VOL_KEY[vol_mult]] & flags[_PCT_KEY[pct_thr]]

    rsi_lo, rsi_hi = rsi_range
    if rsi_lo is not None:
        m = m & (flags["rsi_arr"] >= rsi_lo) & ~np.isnan(flags["rsi_arr"])
    if rsi_hi is not None:
        m = m & (flags["rsi_arr"] <= rsi_hi) & ~np.isnan(flags["rsi_arr"])

    if ma25_pos == "above":
        m = m & flags["above_ma25"]
    elif ma25_pos == "below":
        m = m & flags["below_ma25"]

    if mktcap_max is not None:
        m = m & flags[_MC_KEY[mktcap_max]]

    return m


def _metrics(rets: np.ndarray, trading_days: int) -> dict:
    if len(rets) < 5:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "ev": 0.0, "spd": 0.0,
                "avg_w": 0.0, "avg_l": 0.0}
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
        all_rets: list[float] = []

        for td in all_proc:
            m = _get_mask(
                td["flags"],
                p["vol_mult"], p["pct_thr"], p["rsi_range"],
                p["ma25_pos"], p["mktcap_max"],
            )
            vidx = np.where(m)[0]
            if len(vidx) == 0:
                continue
            e = td["next_o"][vidx]
            a = td["atr"][vidx]
            s = np.maximum(e - a * 2.0, e * 0.90)
            t = e + (e - s) * p["rr"]
            valid = (e > 0) & (s < e) & (t > e)
            if not valid.any():
                continue
            rets = _calc_rets(td["high"], td["low"], td["open"], td["close"], vidx[valid], e[valid], s[valid], t[valid])
            all_rets.extend(rets.tolist())

        m = _metrics(np.array(all_rets), trading_days)
        results.append({**p, **m})

        if ci % 200 == 0 or ci == n_combos:
            passed = sum(
                1 for r in results
                if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                and r["spd"] >= CRITERIA_SPD
            )
            ela = time.time() - t0
            eta = ela / ci * (n_combos - ci)
            print(f"  {ci:4d}/{n_combos}  合格: {passed}件  経過:{ela:.0f}s  残:{eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════════════════

def _rsi_label(rsi_range) -> str:
    lo, hi = rsi_range
    if lo is None and hi is None:
        return "RSI:制限なし"
    if lo is None:
        return f"RSI≤{hi:.0f}"
    if hi is None:
        return f"RSI≥{lo:.0f}"
    return f"RSI{lo:.0f}-{hi:.0f}"


def _fmt(r: dict) -> str:
    mc  = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else "制限なし"
    ma  = {"above": "MA25上", "below": "MA25下", None: "MA25無"}[r["ma25_pos"]]
    rsi = _rsi_label(r["rsi_range"])
    return (f"vol{r['vol_mult']:.0f}x  +{r['pct_thr']:.0f}%  {rsi:<14}"
            f"  {ma}  {mc}  RR{r['rr']}")


def print_results(results: list[dict]) -> None:
    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]

    print(f"\n{'='*80}")
    print(f"【出来高爆発モメンタム型 グリッドサーチ結果】")
    print(f"  総計: {len(results)} 通り  合格: {len(qualified)} 通り")
    print(f"  評価基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  {CRITERIA_SPD}件/日以上")

    hdr = f"  {'条件':<58}  {'勝率':>7}  {'PF':>5}  {'EV':>7}  {'件数':>6}  {'件/日':>6}"
    sep = "  " + "-" * 96

    if qualified:
        # WR順
        top_wr = sorted(qualified, key=lambda r: (r["wr"], r["pf"]), reverse=True)[:15]
        print(f"\n▶ 合格 上位15件【WR順】")
        print(hdr); print(sep)
        for r in top_wr:
            print(f"  {_fmt(r):<58}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        # 件/日順（件数重視）
        top_spd = sorted(qualified, key=lambda r: r["spd"], reverse=True)[:15]
        print(f"\n▶ 合格 上位15件【件/日順（件数重視）】")
        print(hdr); print(sep)
        for r in top_spd:
            print(f"  {_fmt(r):<58}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        # PF順
        top_pf = sorted(qualified, key=lambda r: (r["pf"], r["wr"]), reverse=True)[:10]
        print(f"\n▶ 合格 上位10件【PF順】")
        print(hdr); print(sep)
        for r in top_pf:
            print(f"  {_fmt(r):<58}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

        # ★ 最良（件数×EV = 期待収益/日）
        best = sorted(qualified, key=lambda r: r["ev"] * r["spd"], reverse=True)[0]
        print(f"\n{'='*80}")
        print(f"★ 最良条件（期待収益/日 = EV × 件/日 最大）")
        print(f"  出来高       : {best['vol_mult']:.0f}倍以上")
        print(f"  前日比       : +{best['pct_thr']:.0f}% 以上")
        print(f"  RSI          : {_rsi_label(best['rsi_range'])}")
        _ma25_labels = {'above': 'MA25上', 'below': 'MA25下', None: '制限なし'}
        print(f"  MA25位置     : {_ma25_labels[best['ma25_pos']]}")
        print(f"  時価総額     : {best['mktcap_max']/1e8:.0f}億円以下" if best["mktcap_max"] else "  時価総額     : 制限なし")
        print(f"  RR           : 1:{best['rr']}")
        print(f"  ──────────────────────")
        print(f"  勝率         : {best['wr']:.1f}%")
        print(f"  PF           : {best['pf']:.2f}")
        print(f"  期待値       : {best['ev']:+.2f}%/トレード")
        print(f"  件数         : {best['n']:,}件  ({best['spd']:.2f}件/日)")
        print(f"  期待収益/日  : {best['ev']*best['spd']:+.3f}%")
        print(f"  平均利益     : {best['avg_w']:+.2f}%  平均損失: {best['avg_l']:+.2f}%")

    else:
        print("\n合格なし。上位10件（WR×PF順）を表示:")
        top = sorted(results, key=lambda r: r["wr"] * r["pf"], reverse=True)[:10]
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt(r):<58}  {r['wr']:>6.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['n']:>6,}  {r['spd']:>5.2f}/日")

    # 分布
    print(f"\n{'='*80}")
    print("【条件ごとの傾向】")
    for key in ["vol_mult", "pct_thr", "rsi_range", "ma25_pos"]:
        vals = GRID[key]
        for v in vals:
            sub = [r for r in results if r[key] == v]
            if not sub:
                continue
            best_wr  = max(r["wr"]  for r in sub)
            best_pf  = max(r["pf"]  for r in sub)
            best_spd = max(r["spd"] for r in sub)
            q = sum(1 for r in sub if r["wr"] >= CRITERIA_WR
                    and r["pf"] >= CRITERIA_PF and r["spd"] >= CRITERIA_SPD)
            lbl = (f"{key}={_rsi_label(v)}" if key == "rsi_range"
                   else f"{key}={v}")
            print(f"  {lbl:<30}  WR:{best_wr:5.1f}%  PF:{best_pf:4.2f}"
                  f"  件/日:{best_spd:4.2f}  合格:{q}/{len(sub)}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    n_combos = len(list(itertools.product(*GRID.values())))
    print("=" * 80)
    print("出来高爆発モメンタム型 グリッドサーチ")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())} = {n_combos} 通り")
    print("=" * 80)

    raw_data = build_or_load_cache()

    print(f"\n株数取得中: {len(raw_data)} 銘柄...")
    shares_map = fetch_all_shares(list(raw_data.keys()))

    print("\n前処理中...")
    all_proc: list[dict] = []
    for i, (tk, df_raw) in enumerate(raw_data.items(), 1):
        r = preprocess(tk, df_raw, shares_map.get(tk))
        if r is not None:
            all_proc.append(r)
        if i % 500 == 0:
            print(f"  {i}/{len(raw_data)}  有効: {len(all_proc)}")
    print(f"  完了: {len(all_proc)} 銘柄")

    sample_dfs = list(raw_data.values())[:30]
    all_dates: set = set()
    for df in sample_dfs:
        all_dates.update(df.index.tolist())
    trading_days = len(all_dates)
    print(f"推定取引日数: {trading_days} 日")

    results = run_grid(all_proc, trading_days)
    print_results(results)
    print("\n完了")


if __name__ == "__main__":
    main()
