#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小型株リサーチ（検証専用・実売買には使わない）
================================================
仮説: 小型株はアナリストのカバーが薄く、好材料が株価に織り込まれるまで時間がかかる
      （決算後ドリフト / PEAD）。会社が通期の営業利益予想を上方修正した翌日に買えば、
      その遅れを取れるのではないか。
      ※J-Quants 無料プランは約2年分（2024-07〜）しかなく、前年同期比は比較対象がほぼ無い。
        1回前の開示と比べるだけで判定できる「予想の上方修正」を主仮説にした。

このファイルは「仮説が本物か」を公正に判定するための土台。
  - 合格基準はコードを書く前に固定する（下の PRE_REGISTERED。結果を見てから変えない）
  - 売買コスト・寄りのギャップ・ストップ安張り付きを反映する
  - 2025年以降のデータ（ホールドアウト）は最後に1回だけ使う

使い方:
  python smallcap_research.py check        # J-Quants で何のデータが取れるか診断
  python smallcap_research.py selftest     # 売買シミュレーションの動作確認（データ不要）
  python smallcap_research.py build        # 株価・財務を日付ごとにダウンロード（中断しても続きから再開）
  python smallcap_research.py backtest     # 検証期間（ホールドアウト前）で判定
  python smallcap_research.py final        # ホールドアウトで最終確認（1回だけ）
  python smallcap_research.py volspike     # 仮説2: ヨコヨコ→出来高急増 を検証期間で判定
  python smallcap_research.py volspike-final  # 仮説2のホールドアウト（1回だけ）
"""

import json, os, sys, time
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
DATA_DIR         = Path(__file__).parent / "data_cache" / "smallcap"

# ──────────────────────────────────────────────────────────────────────────────
# 事前登録（検証結果を見る前に固定。変更したら別の仮説として最初からやり直す）
# ──────────────────────────────────────────────────────────────────────────────
PRE_REGISTERED = {
    "data_note":        "J-Quants無料プラン（約2年分）。期間が短く相場局面が限られるため、合格しても信頼性は低い",
    "signal":           "通期営業利益予想(FOP)を前回開示比+10%以上に上方修正（前回FOP>0）。開示翌営業日の始値で買い",
    "min_revision":     0.10,
    "universe":         "時価総額 50〜300億円（シグナル時点の株価×(発行済株式数-自己株式)）",
    "min_mcap_yen":     50 * 10**8,
    "max_mcap_yen":     300 * 10**8,
    "min_turnover_yen": 30_000_000,          # 20日平均売買代金の下限
    "cost_round_trip":  0.6,                 # 往復コスト%（手数料0 + スプレッド/スリッページ）
    "stop_pct":         0.15,                # 初期損切り（エントリー比）
    "max_hold":         60,                  # 最大保有営業日
    "holdout_from":     "2025-10-01",        # これ以降は最終確認まで使わない（データが2年のため）
    # 合格基準（すべて満たすこと）
    "min_trades":       100,
    "min_pf":           1.3,
    "min_ev_pct":       0.0,
    "min_years_pf_ge_1": 0.75,               # 半期別 PF≥1.0 の期の割合
    "beat_control":     True,                # 条件なしの全開示（対照群）の期待値を上回ること
}


# 仮説2（ユーザーの観察）: 小型株が一定期間ヨコヨコのあと、出来高が突然膨らむと、
# その日〜翌日に一気に上がる。シグナルが分かるのは当日の大引け後なので、買えるのは翌日始値から。
# 結果を見る前に定義を固定する。条件の数値は変えず、変えるなら別仮説として扱う。
PRE_REGISTERED_VOLSPIKE = {
    **PRE_REGISTERED,
    "signal":           "直近20日(当日除く)の高値/安値-1 ≤15% のヨコヨコで、当日出来高 ≥ 直近20日平均の5倍、"
                        "かつ当日終値が前日終値より高い → 翌営業日の始値で買い",
    "flat_days":        20,
    "flat_range_max":   0.15,
    "vol_mult":         5.0,
    "min_turnover_yen": 10_000_000,          # 出来高急増の前（当日除く20日平均）で判定
    "turnover_before_signal": True,
    "stop_pct":         0.10,
    "max_hold":         5,                   # 「一気に上がる」短期の動きを検証
    "control_sample":   20_000,              # 対照群: 同ユニバースの無作為な日に買い
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
def split_holdout(trades: list[Trade], use_holdout: bool, hypothesis: str = "revision") -> list[Trade]:
    """ホールドアウトは仮説ごとに1回だけ使える。2回目以降は参考値と表示する。"""
    cut = date.fromisoformat(PRE_REGISTERED["holdout_from"])
    if not use_holdout:
        return [t for t in trades if t.signal_date < cut]
    log: dict = {}
    if HOLDOUT_LOG.exists():
        log = json.loads(HOLDOUT_LOG.read_text(encoding="utf-8"))
        if "used_at" in log:                       # 旧形式（仮説1のみ）
            log = {"revision": log["used_at"]}
    if hypothesis in log:
        print(f"⚠️ この仮説のホールドアウトは使用済みです（{log[hypothesis]}）。"
              "この結果は参考値で、合否判定には使えません。")
    else:
        log[hypothesis] = date.today().isoformat()
        HOLDOUT_LOG.parent.mkdir(exist_ok=True)
        HOLDOUT_LOG.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    return [t for t in trades if t.signal_date >= cut]


def _period(d: date) -> str:
    return f"{d.year}{'H1' if d.month <= 6 else 'H2'}"


def evaluate(trades: list[Trade], control_ev: float | None = None) -> dict:
    """事前登録の基準で判定する。control_ev は条件なし（対照群）の期待値。"""
    if not trades:
        return {"n": 0, "passed": False}
    r = np.array([t.ret_pct for t in trades])
    gain, loss = r[r > 0].sum(), -r[r <= 0].sum()
    by_period: dict[str, float] = {}
    for p in sorted({_period(t.signal_date) for t in trades}):
        rp = np.array([t.ret_pct for t in trades if _period(t.signal_date) == p])
        g, l = rp[rp > 0].sum(), -rp[rp <= 0].sum()
        by_period[p] = float(g / l) if l > 0 else float("inf")
    pf = float(gain / loss) if loss > 0 else float("inf")
    periods_ok = sum(v >= 1.0 for v in by_period.values()) / len(by_period)
    reasons = pd.Series([t.reason for t in trades]).value_counts().to_dict()
    beats = control_ev is None or r.mean() > control_ev
    passed = (len(r) >= PRE_REGISTERED["min_trades"] and pf >= PRE_REGISTERED["min_pf"]
              and r.mean() > PRE_REGISTERED["min_ev_pct"]
              and periods_ok >= PRE_REGISTERED["min_years_pf_ge_1"]
              and (beats or not PRE_REGISTERED["beat_control"]))
    return {"n": len(r), "win_rate": float((r > 0).mean() * 100), "pf": pf,
            "ev": float(r.mean()), "median": float(np.median(r)), "worst": float(r.min()),
            "by_period_pf": by_period, "periods_pf_ge_1": periods_ok, "exit_reasons": reasons,
            "control_ev": control_ev, "passed": passed}


def format_report(title: str, ev: dict, judge: bool = True) -> str:
    if ev["n"] == 0:
        return f"【{title}】シグナル0件"
    per = "  ".join(f"{p}:{v:.2f}" for p, v in ev["by_period_pf"].items())
    lines = [
        f"【{title}】（コスト{PRE_REGISTERED['cost_round_trip']}%控除後）",
        f"件数 {ev['n']} / 勝率 {ev['win_rate']:.1f}% / PF {ev['pf']:.2f} / 期待値 {ev['ev']:+.2f}%"
        f" / 中央値 {ev['median']:+.2f}% / 最悪 {ev['worst']:+.1f}%",
        f"半期別PF: {per}",
        f"決済理由: {ev['exit_reasons']}",
    ]
    if judge:
        ctrl = "" if ev["control_ev"] is None else f"・対照群の期待値{ev['control_ev']:+.2f}%超"
        lines.append(
            f"判定: {'✅ 合格' if ev['passed'] else '❌ 不合格'}"
            f"（基準: {PRE_REGISTERED['min_trades']}件以上・PF≥{PRE_REGISTERED['min_pf']}・期待値>0・"
            f"半期PF≥1.0が{PRE_REGISTERED['min_years_pf_ge_1']*100:.0f}%以上{ctrl}）")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. データ取得（日付ごと・中断再開可）
# ──────────────────────────────────────────────────────────────────────────────
_req_interval = float(os.getenv("SMALLCAP_REQ_INTERVAL", "1.0") or 1.0)
_last_req = 0.0


class JQError(RuntimeError):
    pass


def _jq_fetch_all(path: str, params: dict) -> list[dict]:
    """ページ送り・レート制限(429)・一時エラーに対応して全件取得する。"""
    global _req_interval, _last_req
    rows: list[dict] = []
    p = dict(params)
    while True:
        for attempt in range(8):
            wait = _last_req + _req_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_req = time.time()
            st, body = _jq_get(path, p)
            if st == 429:
                _req_interval = min(max(_req_interval * 2, 1.0), 20.0)
                print(f"  レート制限 → 60秒待機（以後{_req_interval:.0f}秒間隔）", flush=True)
                time.sleep(60)
                continue
            if st == -1 or st >= 500:
                time.sleep(min(5 * 2 ** attempt, 120))
                continue
            break
        if st != 200:
            raise JQError(f"{path} {p}: HTTP {st} {str(body)[:150]}")
        rows += _rows(body)
        key = body.get("pagination_key") if isinstance(body, dict) else None
        if not key:
            return rows
        p["pagination_key"] = key


def _plan_range() -> tuple[date, date]:
    """契約プランで取得できる期間を、エラーメッセージから読み取る。"""
    import re
    st, body = _jq_get("/equities/bars/daily", {"code": "7203", "from": "2000-01-04", "to": "2000-01-05"})
    m = re.search(r"(\d{4}-\d{2}-\d{2}) ~ (\d{4}-\d{2}-\d{2})", str(body))
    if m:
        return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
    return date.today() - timedelta(days=730), date.today()


def _business_days(frm: date, to: date) -> list[date]:
    try:
        rows = _jq_fetch_all("/markets/calendar", {"from": frm.isoformat(), "to": to.isoformat()})
        hol_key = next((k for k in rows[0] if "hol" in k.lower()), None) if rows else None
        if hol_key:
            return [date.fromisoformat(r["Date"][:10]) for r in rows if str(r[hol_key]) in ("1", "2")]
    except (JQError, KeyError, ValueError):
        pass
    return [d.date() for d in pd.bdate_range(frm, to)]


BAR_COLS = ["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "O", "H", "L", "C", "Vo"]


def _save(path: Path, rows: list[dict], keep: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df = pd.DataFrame(rows)
    if keep:
        df = df[[c for c in keep if c in df.columns]]      # 容量削減（必要な列だけ）
    df.to_pickle(tmp)
    tmp.replace(path)                                   # 途中で止まっても壊れたファイルを残さない


def cmd_build() -> None:
    if not JQUANTS_API_KEY:
        print(".env に JQUANTS_API_KEY がありません")
        return
    frm, to = _plan_range()
    days = [d for d in _business_days(frm, to) if frm <= d <= min(to, date.today())]
    print(f"取得期間: {frm} 〜 {to}（営業日 {len(days)}日）", flush=True)

    # 財務は日付指定が使えるか先に確認（使えなければ銘柄ごとに取得）
    probe = days[len(days) // 2: len(days) // 2 + 10]
    try:
        fins_by_date = any(_jq_fetch_all("/fins/summary", {"date": d.isoformat()}) for d in probe)
    except JQError:
        fins_by_date = False
    print(f"財務の取得方式: {'日付ごと' if fins_by_date else '銘柄ごと'}", flush=True)

    jobs = [("bars", d.isoformat(), "/equities/bars/daily", {"date": d.isoformat()}) for d in days]
    if fins_by_date:
        jobs += [("fins", d.isoformat(), "/fins/summary", {"date": d.isoformat()}) for d in days]
    todo = [j for j in jobs if not (DATA_DIR / j[0] / f"{j[1]}.pkl").exists()]
    print(f"未取得: {len(todo)} / {len(jobs)} 件（1件あたり約{_req_interval:.0f}秒）", flush=True)
    t0 = time.time()
    for i, (kind, key, path, params) in enumerate(todo, 1):
        _save(DATA_DIR / kind / f"{key}.pkl", _jq_fetch_all(path, params),
              BAR_COLS if kind == "bars" else None)
        if i % 25 == 0 or i == len(todo):
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"  {i}/{len(todo)} 件完了（残り約{eta:.0f}分）", flush=True)

    if not fins_by_date:
        codes = sorted(load_bars()["Code"].unique())
        todo = [c for c in codes if not (DATA_DIR / "fins_code" / f"{c}.pkl").exists()]
        print(f"財務（銘柄ごと）未取得: {len(todo)} 銘柄", flush=True)
        for i, c in enumerate(todo, 1):
            _save(DATA_DIR / "fins_code" / f"{c}.pkl", _jq_fetch_all("/fins/summary", {"code": c}))
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} 銘柄完了", flush=True)
    print("✅ ダウンロード完了", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. シグナル作成と検証
# ──────────────────────────────────────────────────────────────────────────────
def _pick(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def load_bars() -> pd.DataFrame:
    parts = [pd.read_pickle(f) for f in sorted((DATA_DIR / "bars").glob("*.pkl"))]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = pd.DataFrame({
        "Code":  df["Code"].astype(str),
        "Date":  pd.to_datetime(df["Date"]),
        "Open":  _pick(df, "AdjO", "AdjustmentOpen", "O"),
        "High":  _pick(df, "AdjH", "AdjustmentHigh", "H"),
        "Low":   _pick(df, "AdjL", "AdjustmentLow", "L"),
        "Close": _pick(df, "AdjC", "AdjustmentClose", "C"),
        "RawC":  _pick(df, "C", "Close", "AdjC"),
        "Vo":    _pick(df, "Vo", "Volume", "AdjVo"),
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values(["Code", "Date"])


def load_fins() -> pd.DataFrame:
    files = sorted((DATA_DIR / "fins").glob("*.pkl")) + sorted((DATA_DIR / "fins_code").glob("*.pkl"))
    parts = [pd.read_pickle(f) for f in files]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["Code"] = df["Code"].astype(str)
    df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    df["shares"] = _pick(df, "ShOutFY") - _pick(df, "TrShFY").fillna(0)
    sort_cols = ["Code", "DiscDate"] + (["DiscNo"] if "DiscNo" in df.columns else [])
    df = df.dropna(subset=["DiscDate"]).drop_duplicates(subset=[c for c in sort_cols if c in df.columns])
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_events(fins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(上方修正イベント, 対照群=全開示イベント) を返す。どちらも Code/DiscDate を持つ。"""
    # 同じ決算期の営業利益予想を開示順に並べる。期初予想は前年度の本決算資料に
    # 「来期予想」(NxFOP) として載るので、それも同じ系列に入れる
    order = fins["DiscNo"] if "DiscNo" in fins.columns else pd.Series(range(len(fins)), index=fins.index)
    cur = pd.DataFrame({"Code": fins["Code"], "DiscDate": fins["DiscDate"], "order": order,
                        "target": fins.get("CurFYEn"),
                        "fc": _pick(fins, "FOP").fillna(_pick(fins, "FNCOP"))})
    nxt = pd.DataFrame({"Code": fins["Code"], "DiscDate": fins["DiscDate"], "order": order,
                        "target": fins.get("NxFYEn"),
                        "fc": _pick(fins, "NxFOP").fillna(_pick(fins, "NxFNCOP"))})
    tl = (pd.concat([cur, nxt]).dropna(subset=["target", "fc"])
          .sort_values(["Code", "target", "DiscDate", "order"]))
    prev = tl.groupby(["Code", "target"])["fc"].shift(1)
    revisions = tl[(prev > 0) & (tl["fc"] / prev - 1 >= PRE_REGISTERED["min_revision"])]
    control = fins.drop_duplicates(subset=["Code", "DiscDate"])
    return revisions[["Code", "DiscDate"]], control[["Code", "DiscDate"]]


def run_events(events: pd.DataFrame, bars: pd.DataFrame, fins: pd.DataFrame,
               cfg: dict = PRE_REGISTERED) -> tuple[list[Trade], dict]:
    """ユニバース（時価総額・流動性）を満たすイベントを、イベント翌営業日の始値で売買する。"""
    stats = {"events": len(events), "no_price": 0, "out_of_universe": 0, "illiquid": 0,
             "insufficient_window": 0, "duplicate": 0}
    data_end = bars["Date"].max()
    by_code = {c: g.set_index("Date") for c, g in bars.groupby("Code")}
    shares_by_code = {c: g.dropna(subset=["shares"]).set_index("DiscDate")["shares"]
                      for c, g in fins.groupby("Code")}
    trades: list[Trade] = []
    busy_until: dict[str, pd.Timestamp] = {}
    for ev in events.sort_values("DiscDate").itertuples():
        df = by_code.get(ev.Code)
        if df is None:
            stats["no_price"] += 1
            continue
        if busy_until.get(ev.Code, pd.Timestamp.min) >= ev.DiscDate:
            stats["duplicate"] += 1
            continue
        entry_i = int(df.index.searchsorted(ev.DiscDate, side="right"))   # 開示日の翌営業日
        if entry_i >= len(df) or entry_i < 20:
            stats["no_price"] += 1
            continue
        sh = shares_by_code.get(ev.Code)
        sh = sh[sh.index <= ev.DiscDate] if sh is not None else None
        if sh is None or sh.empty:
            stats["out_of_universe"] += 1
            continue
        mcap = df["RawC"].iloc[entry_i - 1] * sh.iloc[-1]
        if not (cfg["min_mcap_yen"] <= mcap <= cfg["max_mcap_yen"]):
            stats["out_of_universe"] += 1
            continue
        end = entry_i - 1 if cfg.get("turnover_before_signal") else entry_i
        turnover = (df["RawC"] * df["Vo"]).iloc[max(0, end - 20): end].mean()
        if not turnover >= cfg["min_turnover_yen"]:
            stats["illiquid"] += 1
            continue
        # 保有期間が標本の終わりで切れるイベントは除外（上場廃止で途切れた銘柄は残す）
        if entry_i + cfg["max_hold"] > len(df) and df.index[-1] >= data_end:
            stats["insufficient_window"] += 1
            continue
        t = simulate_trade(df, entry_i, ev.Code, ev.DiscDate.date(),
                           stop_pct=cfg["stop_pct"], max_hold=cfg["max_hold"],
                           cost_pct=cfg["cost_round_trip"])
        if t:
            trades.append(t)
            busy_until[ev.Code] = pd.Timestamp(t.exit_date)
    return trades, stats


def cmd_backtest(use_holdout: bool) -> None:
    bars, fins = load_bars(), load_fins()
    print(f"データ: 株価 {bars['Date'].min().date()}〜{bars['Date'].max().date()} "
          f"{bars['Code'].nunique()}銘柄 / 財務 {len(fins)}件 "
          f"({fins['DiscDate'].min().date()}〜{fins['DiscDate'].max().date()})")
    revisions, control = build_events(fins)
    print(f"上方修正イベント(ユニバース判定前): {len(revisions)}件 / 全開示: {len(control)}件")
    print(f"※{PRE_REGISTERED['data_note']}")
    cut = date.fromisoformat(PRE_REGISTERED["holdout_from"])
    ctrl_trades, ctrl_stats = run_events(control, bars, fins)
    main_trades, main_stats = run_events(revisions, bars, fins)
    print(f"  除外内訳 対照群: {ctrl_stats}")
    print(f"  除外内訳 上方修正: {main_stats}")
    # 対照群は比較用なのでホールドアウト使用の記録はしない。主仮説だけ記録する
    ctrl_trades = [t for t in ctrl_trades if (t.signal_date >= cut) == use_holdout]
    main_trades = split_holdout(main_trades, use_holdout)

    phase = "ホールドアウト(最終確認)" if use_holdout else "検証期間"
    ctrl = evaluate(ctrl_trades)
    print()
    print(format_report(f"{phase} 対照群: 全開示の翌日に買い", ctrl, judge=False))
    print()
    print(format_report(f"{phase} 上方修正+10%の翌日に買い", evaluate(main_trades, ctrl.get("ev"))))


def build_volspike_events(bars: pd.DataFrame) -> pd.DataFrame:
    """ヨコヨコ後の出来高急増（上昇引け）の日を返す。DiscDate=シグナル日（大引け後に判明）。"""
    cfg, out = PRE_REGISTERED_VOLSPIKE, []
    n = cfg["flat_days"]
    for code, g in bars.groupby("Code"):
        g = g.set_index("Date")
        hi = g["High"].shift(1).rolling(n).max()
        lo = g["Low"].shift(1).rolling(n).min()
        avg_vo = g["Vo"].shift(1).rolling(n).mean()
        prev_c = g["Close"].shift(1)
        hit = ((hi / lo - 1 <= cfg["flat_range_max"]) & (avg_vo > 0)
               & (g["Vo"] >= avg_vo * cfg["vol_mult"]) & (g["Close"] > prev_c))
        if hit.any():
            nxt_open = g["Open"].shift(-1)
            sub = g[hit]
            out.append(pd.DataFrame({
                "Code": code, "DiscDate": sub.index,
                "day_ret": (sub["Close"] / prev_c[hit] - 1).to_numpy() * 100,        # 当日の上昇（買えない）
                "gap": (nxt_open[hit] / sub["Close"] - 1).to_numpy() * 100,           # 翌朝のギャップ（買えない）
            }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Code", "DiscDate", "day_ret", "gap"])


def cmd_volspike(use_holdout: bool) -> None:
    cfg = PRE_REGISTERED_VOLSPIKE
    bars, fins = load_bars(), load_fins()
    print(f"データ: 株価 {bars['Date'].min().date()}〜{bars['Date'].max().date()} {bars['Code'].nunique()}銘柄")
    print(f"仮説: {cfg['signal']}")
    print(f"決済: 損切り-{cfg['stop_pct']*100:.0f}% / 最大{cfg['max_hold']}営業日 / コスト往復{cfg['cost_round_trip']}%")
    print(f"※{cfg['data_note']}")
    events = build_volspike_events(bars)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(bars), size=min(cfg["control_sample"], len(bars)), replace=False)
    control = bars.iloc[pick][["Code", "Date"]].rename(columns={"Date": "DiscDate"})

    main_trades, main_stats = run_events(events, bars, fins, cfg)
    ctrl_trades, ctrl_stats = run_events(control, bars, fins, cfg)
    print(f"出来高急増シグナル(ユニバース判定前): {len(events)}件")
    print(f"  除外内訳 シグナル: {main_stats}")
    print(f"  除外内訳 対照群: {ctrl_stats}")

    # 買う前に終わってしまう値動き（当日の上昇・翌朝のギャップ）を実際に売買した銘柄で集計
    traded = {(t.ticker, t.signal_date) for t in main_trades}
    ev = events[[(c, d.date()) in traded for c, d in zip(events["Code"], events["DiscDate"])]]
    if len(ev):
        print(f"  買えない部分: シグナル当日の上昇 中央値{ev['day_ret'].median():+.1f}% / "
              f"翌朝ギャップ 中央値{ev['gap'].median():+.1f}%（平均{ev['gap'].mean():+.1f}%）")

    cut = date.fromisoformat(cfg["holdout_from"])
    ctrl_trades = [t for t in ctrl_trades if (t.signal_date >= cut) == use_holdout]
    main_trades = split_holdout(main_trades, use_holdout, "volspike")
    phase = "ホールドアウト(最終確認)" if use_holdout else "検証期間"
    ctrl = evaluate(ctrl_trades)
    print()
    print(format_report(f"{phase} 対照群: 無作為な日に買い", ctrl, judge=False))
    print()
    print(format_report(f"{phase} ヨコヨコ→出来高急増の翌日に買い", evaluate(main_trades, ctrl.get("ev"))))


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
                   Trade("Y", date(2022, 8, 1), date(2022, 8, 2), date(2022, 9, 1), 100, 95, -5, "stop")])
    assert abs(ev["pf"] - 2.0) < 1e-9 and not ev["passed"] and set(ev["by_period_pf"]) == {"2022H1", "2022H2"}, ev
    print("selftest OK: 損切り / ギャップ / ストップ安張り付き / 時間切れ / コスト / 判定")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        cmd_check()
    elif cmd == "selftest":
        cmd_selftest()
    elif cmd == "build":
        cmd_build()
    elif cmd == "backtest":
        cmd_backtest(use_holdout=False)
    elif cmd == "final":
        cmd_backtest(use_holdout=True)
    elif cmd == "volspike":
        cmd_volspike(use_holdout=False)
    elif cmd == "volspike-final":
        cmd_volspike(use_holdout=True)
    else:
        print(__doc__)
