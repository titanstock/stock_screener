#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
連続陰線下ヒゲ反発型 グリッドサーチ
=====================================
スクリーナー条件:
  - 直近 N 日連続陰線（前N日が全てclose < open）
  - 当日下ヒゲ比率 ≥ X%（= (min(open,close) - low) / (high - low) × 100）
  - 出来高 ≥ Y 倍
  - RSI(14) ≤ Z
  - 売買代金 ≥ 3000万/日
  - 時価総額 ≤ 上限

複合スコア = WR × PF × sqrt(件/日)
評価基準   = WR≥55% / PF≥1.5 / 件/日≥0.3
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

MIN_HISTORY  = 100
MAX_HOLD     = 20
MAX_WORKERS  = 20
MIN_TURNOVER = 30_000_000

CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.3

# ── グリッド ──────────────────────────────────────────────────────────────────
GRID = {
    "consec_days":  [3, 4, 5],                    # 連続陰線日数 (3)
    "shadow_pct":   [20.0, 30.0, 40.0],           # 下ヒゲ比率下限% (3)
    "vol_mult":     [0.0, 1.5, 2.0],              # 出来高倍率 0=制限なし (3)
    "rsi_hi":       [45.0, 55.0, 65.0],           # RSI上限 (3)
    "mktcap_max":   [30e9, 100e9, 200e9, None],   # 時価総額上限 (4)
    "rr":           [1.5, 2.0, 2.5],              # RR (3)
}
# 3×3×3×3×4×3 = 972 通り

_SHADOW_KEY = {20.0: "sh20", 30.0: "sh30", 40.0: "sh40"}
_VOL_KEY    = {1.5: "v15", 2.0: "v20"}
_RSI_KEY    = {45.0: "r45", 55.0: "r55", 65.0: "r65"}
_MC_KEY     = {30e9: "c30", 100e9: "c100", 200e9: "c200"}


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
    targets = ["スタンダード", "グロース"]
    df = df[df[mkt_col].str.contains("|".join(targets), na=False)]
    codes = df[code_col].str.strip().tolist()
    return [f"{c}.T" for c in codes if c.isdigit() and len(c) == 4]


def _fetch_jquants(ticker: str, token: str) -> pd.DataFrame | None:
    code = ticker.replace(".T", "")
    url  = f"{JQUANTS_BASE}/prices/daily_quotes"
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
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _fetch_one(ticker: str, token: str) -> tuple[str, pd.DataFrame | None]:
    try:
        yf_df = yf.download(ticker, period="3y", interval="1d",
                            auto_adjust=True, progress=False, timeout=20)
        if yf_df is not None and len(yf_df) >= MIN_HISTORY:
            yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
            return ticker, yf_df[["Open", "High", "Low", "Close", "Volume"]]
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

    tickers  = _fetch_jpx_tickers()
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

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"date": today_str, "data": raw_data}, f)
    print(f"キャッシュ保存完了: {len(raw_data)} 銘柄")
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
                print(f"  {done}/{len(tickers)}  取得: {sum(1 for v in result.values() if v)}")
    return result


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

    rsi    = calc_rsi(pd.Series(c)).values
    avg_v  = pd.Series(v).rolling(20).mean().values
    vol_r  = np.where(avg_v > 0, v / avg_v, np.nan)
    avg_to = pd.Series(c * v).rolling(20).mean().values

    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = pd.Series(tr).rolling(14).mean().values

    mktcap = c * shares if shares else np.full(n, np.nan)

    # 陰線フラグ（close < open）
    bear = (c < o).astype(float)

    # 下ヒゲ比率 = (min(open,close) - low) / (high - low) × 100
    body_lo  = np.minimum(c, o)
    rng      = h - l
    shadow   = np.where(rng > 0, (body_lo - l) / rng * 100, 0.0)

    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(rsi)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    flags: dict[str, np.ndarray] = {
        "base": base,
        # 下ヒゲ比率
        "sh20": (shadow >= 20.0),
        "sh30": (shadow >= 30.0),
        "sh40": (shadow >= 40.0),
        # 出来高
        "v15": (vol_r >= 1.5) & ~np.isnan(vol_r),
        "v20": (vol_r >= 2.0) & ~np.isnan(vol_r),
        # RSI上限
        "r45": (rsi <= 45.0) & ~np.isnan(rsi),
        "r55": (rsi <= 55.0) & ~np.isnan(rsi),
        "r65": (rsi <= 65.0) & ~np.isnan(rsi),
        # 時価総額
        "c30":  np.isnan(mktcap) | (mktcap <= 30e9),
        "c100": np.isnan(mktcap) | (mktcap <= 100e9),
        "c200": np.isnan(mktcap) | (mktcap <= 200e9),
    }

    # 連続陰線フラグ: 直前 consec_days 日が全て陰線
    bear_s = pd.Series(bear)
    for days in [3, 4, 5]:
        consec = bear_s.shift(1).rolling(days).sum().values
        flags[f"cb{days}"] = (consec >= days)

    return {
        "close":  c, "high": h, "low": l, "open": o,
        "next_o": next_o, "atr": atr, "flags": flags, "n": n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# リターン計算（高値/安値ベース）
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


def _get_mask(flags, consec_days, shadow_pct, vol_mult, rsi_hi, mktcap_max):
    m = flags["base"] & flags[f"cb{consec_days}"] & flags[_SHADOW_KEY[shadow_pct]]
    m = m & flags[_RSI_KEY[rsi_hi]]
    if vol_mult > 0.0:
        m = m & flags[_VOL_KEY[vol_mult]]
    if mktcap_max is not None:
        m = m & flags[_MC_KEY[mktcap_max]]
    return m


def _metrics(rets: np.ndarray, trading_days: int) -> dict:
    if len(rets) < 5:
        return {"n": len(rets), "wr": 0.0, "pf": 0.0, "ev": 0.0,
                "spd": 0.0, "avg_w": 0.0, "avg_l": 0.0, "score": 0.0}
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets) * 100
    avg_w  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_l  = float(losses.mean()) if len(losses) > 0 else 0.0
    pf     = abs(avg_w / avg_l)   if avg_l != 0  else 0.0
    ev     = wr / 100 * avg_w + (1 - wr / 100) * avg_l
    spd    = len(rets) / trading_days if trading_days > 0 else 0.0
    score  = (wr / 100) * pf * (spd ** 0.5)
    return {"n": len(rets), "wr": wr, "pf": pf, "ev": ev, "spd": spd,
            "avg_w": avg_w, "avg_l": avg_l, "score": score}


# ══════════════════════════════════════════════════════════════════════════════
# グリッドサーチ
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
                p["consec_days"], p["shadow_pct"],
                p["vol_mult"], p["rsi_hi"], p["mktcap_max"],
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
            rets = _calc_rets(td["high"], td["low"], td["open"], td["close"],
                              vidx[valid], e[valid], s[valid], t[valid])
            all_rets.extend(rets.tolist())

        m = _metrics(np.array(all_rets), trading_days)
        results.append({**p, **m})

        if ci % 100 == 0 or ci == n_combos:
            passed  = sum(1 for r in results
                          if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                          and r["spd"] >= CRITERIA_SPD)
            elapsed = time.time() - t0
            eta     = elapsed / ci * (n_combos - ci)
            print(f"  {ci:4d}/{n_combos}  合格: {passed}件  "
                  f"経過: {elapsed:.0f}s  残: {eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_params(r: dict) -> str:
    mc  = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else "制限なし"
    vol = f"vol{r['vol_mult']:.1f}x" if r["vol_mult"] > 0 else "vol--"
    return (f"陰線{r['consec_days']}日  下ヒゲ≥{r['shadow_pct']:.0f}%  "
            f"{vol}  RSI≤{r['rsi_hi']:.0f}  {mc}  RR{r['rr']}")


def print_results(results: list[dict]) -> None:
    qualified = [r for r in results
                 if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
                 and r["spd"] >= CRITERIA_SPD]

    print(f"\n{'='*90}")
    print(f"【連続陰線下ヒゲ反発型 グリッドサーチ 結果】")
    print(f"  総計: {len(results)} 通り  |  合格: {len(qualified)} 通り")
    print(f"  評価基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  件/日≥{CRITERIA_SPD}")
    print(f"  スクリーナー条件: 5日陰線 / 下ヒゲ≥30% / vol1.5x / RSI≤55 / 時価総額≤300億 / RR2.5")

    hdr = f"  {'条件':<58}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件/日':>6}  {'Score':>6}"
    sep = "  " + "-" * 93

    if qualified:
        top = sorted(qualified, key=lambda r: r["score"], reverse=True)[:20]
        print(f"\n▶ 合格条件 上位20件【複合スコア順】")
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt_params(r):<58}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        best = max(qualified, key=lambda r: r["score"])
        print(f"\n{'='*90}")
        print(f"★ 推奨条件（複合スコア最大）")
        print(f"  連続陰線   : {best['consec_days']}日")
        print(f"  下ヒゲ     : ≥ {best['shadow_pct']:.0f}%")
        print(f"  出来高     : {best['vol_mult']:.1f}倍以上" if best["vol_mult"] > 0 else "  出来高     : 条件なし")
        print(f"  RSI上限    : ≤ {best['rsi_hi']:.0f}")
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
            print(f"  {_fmt_params(r):<58}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_combos = 1
    for v in GRID.values():
        total_combos *= len(v)

    print("=" * 90)
    print("連続陰線下ヒゲ反発型 グリッドサーチ 開始")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())} = {total_combos} 通り")
    print("=" * 90)

    raw_data = load_cache()

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

    results = run_grid(all_proc, trading_days)
    print_results(results)
    print("\n完了")


if __name__ == "__main__":
    main()
