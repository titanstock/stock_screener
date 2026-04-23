#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBS・短期RSI 総合グリッドサーチ
================================
学術研究・定量バックテスト・専業トレーダーが重きを置く条件を全組み合わせで探索

① IBS (Internal Bar Strength) = (終値-安値)/(高値-安値)
   → 0に近いほど「安値引け」= 翌日反発の逆張りシグナル
   → 30年・日本株含む国際市場で確認済み（最も証拠が強い短期指標）

② 短期RSI（2 / 3 / 5 / 14日）× 閾値（20 / 30 / 40）
   → 14日RSIより2〜5日の方がスイングに有効（学術研究）
   → 閾値も30/70より20/80の極端値が有効

③ 出来高急増（High Volume Return Premium — Gervais et al. 2001）
④ MA75 トレンド方向（長期トレンドフィルター）
⑤ MA25 乖離率

グリッド: 4×4×4×4×3×4×3×3 = 27,648 通り
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

# ── 評価基準 ──────────────────────────────────────────────────────────────────
CRITERIA_WR  = 55.0
CRITERIA_PF  = 1.5
CRITERIA_SPD = 0.3

# ── グリッド ──────────────────────────────────────────────────────────────────
RSI_PERIODS     = [2, 3, 5, 14]
RSI_THRESHOLDS  = [20, 30, 40]

GRID = {
    "ibs_hi":     [0.2, 0.3, 0.5, None],       # IBS上限  (None=制限なし)
    "rsi_period": [2, 3, 5, 14],                 # RSI計算期間
    "rsi_hi":     [20, 30, 40, None],            # RSI上限  (None=制限なし)
    "vol_mult":   [0.0, 1.5, 2.0, 3.0],         # 出来高倍率 (0=制限なし)
    "ma75_pos":   [None, "above", "below"],       # MA75との位置関係
    "ma25_dev":   [None, -3.0, -5.0, -10.0],    # MA25乖離率上限 (None=制限なし)
    "rr":         [1.5, 2.0, 2.5],               # RR比
    "mktcap_max": [30e9, 100e9, 200e9, None],    # 時価総額上限 (None=制限なし)
}
# 4×4×4×4×3×4×3×3 = 27,648 通り

# フラグキーマップ
_IBS_KEY  = {0.2: "ibs02", 0.3: "ibs03", 0.5: "ibs05"}
_RSI_KEY  = {(p, h): f"r{p}_{h}" for p in RSI_PERIODS for h in RSI_THRESHOLDS}
_VOL_KEY  = {1.5: "v15", 2.0: "v20", 3.0: "v30"}
_MA25_KEY = {-3.0: "m3", -5.0: "m5", -10.0: "m10"}
_MC_KEY   = {30e9: "c30", 100e9: "c100", 200e9: "c200"}


# ══════════════════════════════════════════════════════════════════════════════
# データ取得・キャッシュ（backtest_strongest.py と共通構造）
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
    url = f"{JQUANTS_BASE}/prices/daily_quotes"
    from_d = (date.today() - timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        r = requests.get(url, params={"code": code, "from": from_d},
                         headers={"Authorization": f"Bearer {token}"}, timeout=30)
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
        cache_date = cached.get("date", "")
        cutoff = (date.today() - timedelta(days=CACHE_MAX_AGE)).isoformat()
        if cache_date >= cutoff:
            print(f"キャッシュ利用（{cache_date}）: {len(cached['data'])} 銘柄")
            return cached["data"]
        print(f"キャッシュ期限切れ → 再取得")
        raw_data: dict[str, pd.DataFrame] = cached.get("data", {})
    else:
        print("キャッシュなし → 全件取得")
        raw_data = {}

    tickers = _fetch_jpx_tickers()
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
# RSI ヘルパー（Wilder平滑化、複数期間対応）
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(prices: np.ndarray, period: int) -> np.ndarray:
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
        avg_g = avg_g * (period - 1) / period + gains[i] / period
        avg_l = avg_l * (period - 1) / period + losses[i] / period
        rsi[i + 1] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


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

    # 翌日始値（エントリー価格）
    next_o = np.full(n, np.nan)
    next_o[:-1] = o[1:]

    # ATR(14) — ストップロス計算用
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr     = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr    = pd.Series(tr).rolling(14).mean().values

    # IBS = (終値 - 安値) / (高値 - 安値)
    rng = h - l
    ibs = np.where(rng > 0, (c - l) / rng, np.nan)

    # RSI（複数期間）
    rsi_arrays = {p: _calc_rsi(c, p) for p in RSI_PERIODS}

    # MA25・MA75
    ma25 = pd.Series(c).rolling(25).mean().values
    ma75 = pd.Series(c).rolling(75).mean().values

    # MA25乖離率(%)
    ma25_dev = np.where(ma25 > 0, (c - ma25) / ma25 * 100, np.nan)

    # 出来高倍率（20日移動平均比）
    avg_v = pd.Series(v).rolling(20).mean().values
    vol_r = np.where(avg_v > 0, v / avg_v, np.nan)

    # 平均売買代金(20日)
    avg_to = pd.Series(c * v).rolling(20).mean().values

    # 時価総額
    mktcap = c * shares if shares else np.full(n, np.nan)

    # ── ベースフィルター ───────────────────────────────────────────────────────
    idx  = np.arange(n)
    base = (
        (~np.isnan(atr)) & (atr > 0) &
        (~np.isnan(next_o)) & (next_o > 0) &
        (~np.isnan(avg_to)) & (avg_to >= MIN_TURNOVER) &
        (~np.isnan(ibs)) &
        (idx >= MIN_HISTORY) & (idx < n - 1)
    )

    # ── 事前計算フラグ ─────────────────────────────────────────────────────────
    flags: dict[str, np.ndarray] = {
        "base": base,

        # IBS（安値引け）
        "ibs02": (ibs <= 0.2) & ~np.isnan(ibs),
        "ibs03": (ibs <= 0.3) & ~np.isnan(ibs),
        "ibs05": (ibs <= 0.5) & ~np.isnan(ibs),

        # MA75との位置関係
        "ma75_above": (c > ma75) & ~np.isnan(ma75),
        "ma75_below": (c < ma75) & ~np.isnan(ma75),

        # MA25乖離
        "m3":  (ma25_dev <= -3.0)  & ~np.isnan(ma25_dev),
        "m5":  (ma25_dev <= -5.0)  & ~np.isnan(ma25_dev),
        "m10": (ma25_dev <= -10.0) & ~np.isnan(ma25_dev),

        # 出来高
        "v15": (vol_r >= 1.5) & ~np.isnan(vol_r),
        "v20": (vol_r >= 2.0) & ~np.isnan(vol_r),
        "v30": (vol_r >= 3.0) & ~np.isnan(vol_r),

        # 時価総額
        "c30":  np.isnan(mktcap) | (mktcap <= 30e9),
        "c100": np.isnan(mktcap) | (mktcap <= 100e9),
        "c200": np.isnan(mktcap) | (mktcap <= 200e9),
    }

    # RSI閾値フラグ（全期間×全閾値）
    for period in RSI_PERIODS:
        rsi_arr = rsi_arrays[period]
        for hi in RSI_THRESHOLDS:
            flags[f"r{period}_{hi}"] = (rsi_arr <= hi) & ~np.isnan(rsi_arr)

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
# リターン計算（ベクトル化）
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rets(highs: np.ndarray, lows: np.ndarray, opens: np.ndarray,
               closes: np.ndarray, vidx: np.ndarray,
               e_arr: np.ndarray, s_arr: np.ndarray,
               t_arr: np.ndarray) -> np.ndarray:
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


def _get_mask(flags: dict, ibs_hi, rsi_period, rsi_hi,
              vol_mult, ma75_pos, ma25_dev, mktcap_max) -> np.ndarray:
    m = flags["base"].copy()
    if ibs_hi is not None:
        m = m & flags[_IBS_KEY[ibs_hi]]
    if rsi_hi is not None:
        m = m & flags[_RSI_KEY[(rsi_period, rsi_hi)]]
    if vol_mult > 0.0:
        m = m & flags[_VOL_KEY[vol_mult]]
    if ma75_pos == "above":
        m = m & flags["ma75_above"]
    elif ma75_pos == "below":
        m = m & flags["ma75_below"]
    if ma25_dev is not None:
        m = m & flags[_MA25_KEY[ma25_dev]]
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
    score  = (wr / 100) * pf * (spd ** 0.5)   # 複合スコア
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

        # 「無条件買い」の組み合わせはスキップ
        if (p["ibs_hi"] is None and p["rsi_hi"] is None and
                p["vol_mult"] == 0.0 and p["ma75_pos"] is None and p["ma25_dev"] is None):
            continue

        all_rets: list[float] = []
        for td in all_proc:
            m = _get_mask(
                td["flags"],
                p["ibs_hi"], p["rsi_period"], p["rsi_hi"],
                p["vol_mult"], p["ma75_pos"], p["ma25_dev"], p["mktcap_max"],
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
            rets = _calc_rets(td["high"], td["low"], td["open"], td["close"], vidx[valid], e[valid], s[valid], t[valid])
            all_rets.extend(rets.tolist())

        m = _metrics(np.array(all_rets), trading_days)
        results.append({**p, **m})

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
    rsi  = f"RSI{r['rsi_period']}d≤{r['rsi_hi']}" if r["rsi_hi"] is not None else f"RSI{r['rsi_period']}d--"
    vol  = f"vol{r['vol_mult']:.1f}x" if r["vol_mult"] > 0 else "vol--"
    ma75 = {"above": "MA75上", "below": "MA75下", None: "MA75--"}[r["ma75_pos"]]
    ma25 = f"MA25{r['ma25_dev']:+.0f}%" if r["ma25_dev"] is not None else "MA25--"
    mc   = f"{r['mktcap_max']/1e8:.0f}億" if r["mktcap_max"] else "  制限なし"
    return f"{ibs}  {rsi}  {vol}  {ma75}  {ma25}  {mc}  RR{r['rr']}"


def print_results(results: list[dict]) -> None:
    qualified = [
        r for r in results
        if r["wr"] >= CRITERIA_WR and r["pf"] >= CRITERIA_PF
        and r["spd"] >= CRITERIA_SPD
    ]

    hdr = f"  {'条件':<62}  {'勝率':>6}  {'PF':>5}  {'EV':>7}  {'件/日':>6}  {'Score':>6}"
    sep = "  " + "-" * 97

    print(f"\n{'='*97}")
    print(f"【IBS・短期RSI グリッドサーチ 結果】")
    print(f"  総計: {len(results)} 通り  |  合格: {len(qualified)} 通り")
    print(f"  基準: WR≥{CRITERIA_WR}%  PF≥{CRITERIA_PF}  件/日≥{CRITERIA_SPD}")
    print(f"  複合スコア = WR × PF × √(件/日)")

    if qualified:
        # ── 複合スコア上位20件 ────────────────────────────────────────────────
        top_score = sorted(qualified, key=lambda r: r["score"], reverse=True)[:20]
        print(f"\n▶ 複合スコア上位20件（WR×PF×√件数）")
        print(hdr); print(sep)
        for r in top_score:
            print(f"  {_fmt(r):<62}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── WR重視 ───────────────────────────────────────────────────────────
        top_wr = sorted(qualified, key=lambda r: r["wr"], reverse=True)[:10]
        print(f"\n▶ WR重視 上位10件")
        print(hdr); print(sep)
        for r in top_wr:
            print(f"  {_fmt(r):<62}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── 件数重視 ─────────────────────────────────────────────────────────
        top_spd = sorted(qualified, key=lambda r: r["spd"], reverse=True)[:10]
        print(f"\n▶ 件数重視 上位10件")
        print(hdr); print(sep)
        for r in top_spd:
            print(f"  {_fmt(r):<62}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

        # ── パレート前線 ──────────────────────────────────────────────────────
        print(f"\n▶ パレート前線（WR vs 件/日）")
        print(f"  {'WR帯':<12}  {'件/日':>6}  {'PF':>5}  条件")
        print("  " + "-" * 80)
        for lo, hi in [(75, 100), (70, 75), (65, 70), (60, 65), (55, 60)]:
            band = [r for r in qualified if lo <= r["wr"] < hi]
            if band:
                best = max(band, key=lambda r: r["spd"])
                print(f"  WR {lo:>3}〜{hi:<3}%  {best['spd']:>5.2f}/日  "
                      f"{best['pf']:>5.2f}  {_fmt(best)}")

        # ── ★ 推奨条件 ────────────────────────────────────────────────────────
        best = max(qualified, key=lambda r: r["score"])
        print(f"\n{'='*97}")
        print(f"★ 推奨条件（複合スコア最大）")
        print(f"  IBS上限       : {'≤ ' + str(best['ibs_hi']) if best['ibs_hi'] is not None else '条件なし'}")
        print(f"  RSI期間       : {best['rsi_period']}日")
        print(f"  RSI閾値       : {'≤ ' + str(best['rsi_hi']) if best['rsi_hi'] is not None else '条件なし'}")
        print(f"  出来高        : {best['vol_mult']:.1f}倍以上" if best["vol_mult"] > 0 else "  出来高        : 条件なし")
        _ma75_map = {"above": "MA75上（上昇トレンド）", "below": "MA75下（下落トレンド）", None: "条件なし"}
        print(f"  MA75位置      : {_ma75_map[best['ma75_pos']]}")
        print(f"  MA25乖離      : {best['ma25_dev']:+.0f}% 以下" if best["ma25_dev"] else "  MA25乖離      : 条件なし")
        print(f"  時価総額      : {best['mktcap_max']/1e8:.0f}億円以下" if best["mktcap_max"] else "  時価総額      : 条件なし")
        print(f"  RR            : 1:{best['rr']}")
        print(f"  ──────────────────────")
        print(f"  勝率          : {best['wr']:.1f}%")
        print(f"  PF            : {best['pf']:.2f}")
        print(f"  期待値        : {best['ev']:+.2f}%/トレード")
        print(f"  件数/日       : {best['spd']:.2f}件/日")
        print(f"  複合スコア    : {best['score']:.4f}")
        print(f"  平均利益      : {best['avg_w']:+.2f}%  平均損失: {best['avg_l']:+.2f}%")

    else:
        print("\n合格条件なし。上位20件（複合スコア順）を表示:")
        top = sorted(results, key=lambda r: r["score"], reverse=True)[:20]
        print(hdr); print(sep)
        for r in top:
            print(f"  {_fmt(r):<62}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}"
                  f"  {r['ev']:>+6.2f}%  {r['spd']:>5.2f}/日  {r['score']:>5.3f}")

    # ── 各パラメータ別サマリ ──────────────────────────────────────────────────
    print(f"\n{'='*97}")
    print("【各パラメータ別サマリ（合格条件の件/日平均・WR平均・スコア平均）】")

    def _show_param(key, vals_labels):
        print(f"\n  [{key}]")
        for v, lbl in vals_labels:
            sub = [r for r in qualified if r[key] == v] if qualified else []
            if not sub:
                print(f"    {lbl:<20}  合格: 0件")
                continue
            print(f"    {lbl:<20}  合格: {len(sub):5d}件  "
                  f"WR: {np.mean([r['wr'] for r in sub]):5.1f}%  "
                  f"件/日: {np.mean([r['spd'] for r in sub]):5.2f}  "
                  f"スコア: {np.mean([r['score'] for r in sub]):.4f}")

    _show_param("ibs_hi", [
        (0.2, "IBS≤0.2（強い安値引け）"),
        (0.3, "IBS≤0.3"),
        (0.5, "IBS≤0.5"),
        (None, "IBS制限なし"),
    ])
    _show_param("rsi_period", [
        (2,  "RSI(2日)"),
        (3,  "RSI(3日)"),
        (5,  "RSI(5日)"),
        (14, "RSI(14日)"),
    ])
    _show_param("rsi_hi", [
        (20,  "RSI≤20（極度売られすぎ）"),
        (30,  "RSI≤30"),
        (40,  "RSI≤40"),
        (None,"RSI制限なし"),
    ])
    _show_param("ma75_pos", [
        ("above", "MA75上（上昇トレンド）"),
        ("below", "MA75下（下落トレンド）"),
        (None,   "MA75制限なし"),
    ])
    _show_param("vol_mult", [
        (0.0, "出来高制限なし"),
        (1.5, "vol≥1.5x"),
        (2.0, "vol≥2.0x"),
        (3.0, "vol≥3.0x"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_combos = 1
    for v in GRID.values():
        total_combos *= len(v)

    print("=" * 97)
    print("IBS・短期RSI 総合グリッドサーチ 開始")
    print(f"グリッド: {' × '.join(str(len(v)) for v in GRID.values())} = {total_combos} 通り")
    print("新条件  : IBS（安値引け）/ RSI短期(2-14日) / MA75トレンド方向")
    print("複合スコア: WR × PF × √(件/日)")
    print("=" * 97)

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
