#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小型株リサーチ（検証専用・実売買には使わない）
================================================
仮説: 小型株はアナリストのカバーが薄く、好決算が株価に織り込まれるまで時間がかかる
      （決算後ドリフト / PEAD）。好決算の開示翌日に買えば、その遅れを取れるのではないか。

このファイルは「仮説が本物か」を公正に判定するための土台。
  - 合格基準はコードを書く前に固定する（下の PRE_REGISTERED。結果を見てから変えない）
  - 売買コスト・寄りのギャップ・ストップ安張り付きを反映する
  - 2025年以降のデータ（ホールドアウト）は最後に1回だけ使う

使い方:
  python smallcap_research.py check        # J-Quants で何のデータが取れるか診断
  python smallcap_research.py selftest     # 売買シミュレーションの動作確認（データ不要）
"""

import json, os, sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

JQUANTS_API_BASE = "https://api.jquants.com/v2"
JQUANTS_API_KEY  = os.getenv("JQUANTS_API_KEY", os.getenv("JQUANTS_REFRESH_TOKEN", ""))
HOLDOUT_LOG      = Path(__file__).parent / "results" / "holdout_used.json"

# ──────────────────────────────────────────────────────────────────────────────
# 事前登録（検証結果を見る前に固定。変更したら別の仮説として最初からやり直す）
# ──────────────────────────────────────────────────────────────────────────────
PRE_REGISTERED = {
    "universe":         "時価総額 50〜300億円（シグナル時点の株価×発行済株式数）",
    "min_turnover_yen": 30_000_000,          # 20日平均売買代金の下限
    "cost_round_trip":  0.6,                 # 往復コスト%（手数料0 + スプレッド/スリッページ）
    "stop_pct":         0.15,                # 初期損切り（エントリー比）
    "max_hold":         60,                  # 最大保有営業日
    "holdout_from":     "2025-01-01",        # これ以降は最終確認まで使わない
    # 合格基準（すべて満たすこと）
    "min_trades":       100,
    "min_pf":           1.3,
    "min_ev_pct":       0.0,
    "min_years_pf_ge_1": 0.75,               # 年別 PF≥1.0 の年の割合
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. 診断: J-Quants で何が取れるか
# ──────────────────────────────────────────────────────────────────────────────
def _jq_get(path: str, params: dict) -> tuple[int, dict | str]:
    try:
        r = requests.get(f"{JQUANTS_API_BASE}{path}", params=params,
                         headers={"x-api-key": JQUANTS_API_KEY}, timeout=20)
    except requests.RequestException as e:
        return -1, str(e)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:200]


def _rows(body) -> list[dict]:
    if not isinstance(body, dict):
        return []
    for key in ("data", "daily_quotes", "statements", "info"):
        if isinstance(body.get(key), list):
            return body[key]
    return []


def cmd_check() -> None:
    print("=== J-Quants 診断 ===")
    print("APIキー:", "あり" if JQUANTS_API_KEY else "なし（.env に JQUANTS_API_KEY を設定）")
    if not JQUANTS_API_KEY:
        return

    # 株価: 何年前まで取れるか（プランで上限が違う）
    for label, frm, to in [("5年前", "2021-06-01", "2021-06-10"),
                           ("2年前", "2024-10-01", "2024-10-10"),
                           ("直近", (date.today() - timedelta(days=20)).isoformat(), date.today().isoformat())]:
        st, body = _jq_get("/equities/bars/daily", {"code": "7203", "from": frm, "to": to})
        rows = _rows(body)
        print(f"株価[{label}]: HTTP {st} / {len(rows)}件" + ("" if st == 200 else f" / {str(body)[:120]}"))
        if rows and label == "直近":
            print("  株価の項目名:", ", ".join(rows[0].keys()))
            print("  最新日付:", max(r.get("Date", "") for r in rows))

    # 財務: 項目名を確認（営業利益・発行済株式数の名前が必要）
    st, body = _jq_get("/fins/summary", {"code": "7203"})
    rows = _rows(body)
    print(f"財務: HTTP {st} / {len(rows)}件" + ("" if st == 200 else f" / {str(body)[:120]}"))
    if rows:
        print("  財務の項目名:", ", ".join(rows[-1].keys()))
        dd = [r.get("DiscDate") or r.get("DisclosedDate") for r in rows]
        dd = [d for d in dd if d]
        if dd:
            print("  開示日の範囲:", min(dd), "〜", max(dd))

    # 上場銘柄一覧: 上場廃止銘柄を含められるか（生存者バイアス対策）
    for path in ("/equities/master", "/listed/info", "/equities/info"):
        st, body = _jq_get(path, {"date": "2021-06-01"})
        rows = _rows(body)
        print(f"銘柄一覧 {path}[2021-06-01時点]: HTTP {st} / {len(rows)}件")
        if rows:
            print("  銘柄一覧の項目名:", ", ".join(rows[0].keys()))
            break
    print("=== 診断ここまで（この出力をすべてチャットに貼ってください）===")


# ──────────────────────────────────────────────────────────────────────────────
# 2. 売買シミュレーション
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    ticker: str
    signal_date: date
    entry_date: date
    exit_date: date
    entry: float
    exit: float
    ret_pct: float        # コスト控除後
    reason: str           # stop / gap_stop / limit_down_then_open / time


def simulate_trade(df: pd.DataFrame, entry_i: int, ticker: str, signal_date: date,
                   stop_pct: float = PRE_REGISTERED["stop_pct"],
                   max_hold: int = PRE_REGISTERED["max_hold"],
                   cost_pct: float = PRE_REGISTERED["cost_round_trip"]) -> Trade | None:
    """entry_i 日の始値で買い、損切り or 最大保有日数で決済する。
    - 寄りで損切りを割っていたら始値で約定（ギャップ）
    - 高値=安値で損切りを割っている日（ストップ安張り付き）は寄りでも売れないので翌日始値
    """
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    n = len(df)
    if entry_i >= n or not o[entry_i] > 0:
        return None
    entry = o[entry_i]
    stop  = entry * (1 - stop_pct)
    last  = min(entry_i + max_hold - 1, n - 1)
    exit_i, exit_px, reason, pending = last, c[last], "time", False

    for j in range(entry_i, last + 1):
        if pending:                                   # 前日張り付きで売れなかった
            exit_i, exit_px, reason = j, o[j], "limit_down_then_open"
            break
        if j > entry_i and h[j] == l[j] and l[j] <= stop:
            pending = True                            # ストップ安張り付き: 寄りでも売れない
            continue
        if j > entry_i and o[j] <= stop:
            exit_i, exit_px, reason = j, o[j], "gap_stop"
            break
        if l[j] <= stop:
            exit_i, exit_px, reason = j, stop, "stop"
            break
    else:
        if pending:                                   # 最終日も張り付き → 終値で評価
            exit_i, exit_px, reason = last, c[last], "limit_down_then_open"

    ret = (exit_px / entry - 1) * 100 - cost_pct
    idx = df.index
    return Trade(ticker, signal_date, idx[entry_i].date(), idx[exit_i].date(),
                 float(entry), float(exit_px), float(ret), reason)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 判定
# ──────────────────────────────────────────────────────────────────────────────
def split_holdout(trades: list[Trade], use_holdout: bool) -> list[Trade]:
    cut = date.fromisoformat(PRE_REGISTERED["holdout_from"])
    if not use_holdout:
        return [t for t in trades if t.signal_date < cut]
    if HOLDOUT_LOG.exists():
        print(f"⚠️ ホールドアウトは使用済みです（{HOLDOUT_LOG.read_text(encoding='utf-8').strip()}）。"
              "この結果は参考値で、合否判定には使えません。")
    else:
        HOLDOUT_LOG.parent.mkdir(exist_ok=True)
        HOLDOUT_LOG.write_text(json.dumps({"used_at": date.today().isoformat()}), encoding="utf-8")
    return [t for t in trades if t.signal_date >= cut]


def evaluate(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "passed": False}
    r = np.array([t.ret_pct for t in trades])
    gain, loss = r[r > 0].sum(), -r[r <= 0].sum()
    by_year: dict[int, float] = {}
    for y in sorted({t.signal_date.year for t in trades}):
        ry = np.array([t.ret_pct for t in trades if t.signal_date.year == y])
        g, l = ry[ry > 0].sum(), -ry[ry <= 0].sum()
        by_year[y] = float(g / l) if l > 0 else float("inf")
    pf = float(gain / loss) if loss > 0 else float("inf")
    years_ok = sum(v >= 1.0 for v in by_year.values()) / len(by_year)
    reasons = pd.Series([t.reason for t in trades]).value_counts().to_dict()
    passed = (len(r) >= PRE_REGISTERED["min_trades"] and pf >= PRE_REGISTERED["min_pf"]
              and r.mean() > PRE_REGISTERED["min_ev_pct"]
              and years_ok >= PRE_REGISTERED["min_years_pf_ge_1"])
    return {"n": len(r), "win_rate": float((r > 0).mean() * 100), "pf": pf,
            "ev": float(r.mean()), "median": float(np.median(r)), "worst": float(r.min()),
            "by_year_pf": by_year, "years_pf_ge_1": years_ok, "exit_reasons": reasons,
            "passed": passed}


def format_report(title: str, ev: dict) -> str:
    if ev["n"] == 0:
        return f"【{title}】シグナル0件"
    yrs = "  ".join(f"{y}:{p:.2f}" for y, p in ev["by_year_pf"].items())
    return "\n".join([
        f"【{title}】（コスト{PRE_REGISTERED['cost_round_trip']}%控除後）",
        f"件数 {ev['n']} / 勝率 {ev['win_rate']:.1f}% / PF {ev['pf']:.2f} / 期待値 {ev['ev']:+.2f}%"
        f" / 中央値 {ev['median']:+.2f}% / 最悪 {ev['worst']:+.1f}%",
        f"年別PF: {yrs}",
        f"決済理由: {ev['exit_reasons']}",
        f"判定: {'✅ 合格' if ev['passed'] else '❌ 不合格'}"
        f"（基準: {PRE_REGISTERED['min_trades']}件以上・PF≥{PRE_REGISTERED['min_pf']}・期待値>0・"
        f"年別PF≥1.0が{PRE_REGISTERED['min_years_pf_ge_1']*100:.0f}%以上の年）",
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 動作確認（データ不要）
# ──────────────────────────────────────────────────────────────────────────────
def cmd_selftest() -> None:
    idx = pd.bdate_range("2024-01-01", periods=10)
    def mk(o, h, l, c):
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e5}, index=idx)
    base = [100] * 10
    # 1) 場中に損切り
    t = simulate_trade(mk(base, [101]*10, [100, 99, 84, 90, 90, 90, 90, 90, 90, 90], base), 0, "A", idx[0].date(), cost_pct=0)
    assert t.reason == "stop" and t.exit == 85.0, t
    # 2) 寄りで損切りを割る（ギャップ）→ 始値で約定
    t = simulate_trade(mk([100, 100, 70] + [70]*7, [101, 101, 72] + [72]*7, [99, 99, 69] + [69]*7, [100, 100, 70] + [70]*7), 0, "B", idx[0].date(), cost_pct=0)
    assert t.reason == "gap_stop" and t.exit == 70.0, t
    # 3) ストップ安張り付き（高値=安値）→ その日は売れず翌日始値
    t = simulate_trade(mk([100, 100, 80, 75] + [75]*6, [101, 101, 80, 76] + [76]*6, [99, 99, 80, 74] + [74]*6, [100, 100, 80, 75] + [75]*6), 0, "C", idx[0].date(), cost_pct=0)
    assert t.reason == "limit_down_then_open" and t.exit == 75.0 and t.exit_date == idx[3].date(), t
    t = simulate_trade(mk([100, 100, 90, 60] + [60]*6, [101, 101, 90, 61] + [61]*6, [99, 99, 90, 59] + [59]*6, [100, 100, 90, 60] + [60]*6), 0, "D", idx[0].date(), stop_pct=0.05, cost_pct=0)
    assert t.reason == "limit_down_then_open" and t.exit == 60.0, t
    # 4) 最大保有で終値決済 + コスト控除
    t = simulate_trade(mk(base, [101]*10, [99]*10, [110]*10), 0, "E", idx[0].date(), max_hold=5)
    assert t.reason == "time" and t.exit_date == idx[4].date() and abs(t.ret_pct - (10 - 0.6)) < 1e-9, t
    # 5) 判定
    ev = evaluate([Trade("X", date(2022, 1, 1), date(2022, 1, 2), date(2022, 2, 1), 100, 110, 10, "time"),
                   Trade("Y", date(2023, 1, 1), date(2023, 1, 2), date(2023, 2, 1), 100, 95, -5, "stop")])
    assert abs(ev["pf"] - 2.0) < 1e-9 and not ev["passed"], ev
    print("selftest OK: 損切り / ギャップ / ストップ安張り付き / 時間切れ / コスト / 判定")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        cmd_check()
    elif cmd == "selftest":
        cmd_selftest()
    else:
        print(__doc__)
