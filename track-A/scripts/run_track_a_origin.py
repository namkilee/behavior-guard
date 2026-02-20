#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# =========================
# Env / Utils
# =========================
def env(key: str, default=None, cast=None):
    v = os.environ.get(key, "")
    if v == "":
        v = default
    if cast and v is not None:
        return cast(v)
    return v


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def day_range(day: str):
    # day: YYYY-MM-DD
    d = datetime.strptime(day, "%Y-%m-%d")
    start = d.strftime("%Y-%m-%d 00:00:00")
    end = (d + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    return start, end


def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=False)


def topk_str(values, k=3):
    c = Counter([v for v in values if pd.notna(v)])
    items = c.most_common(k)
    return ", ".join([f"{a}({b})" for a, b in items])


def guess_time_col(df: pd.DataFrame):
    for cand in ["event_ts", "timestamp", "start_time", "created_at", "ts"]:
        if cand in df.columns:
            return cand
    return None


def inspect_parquet(path: Path, head_n: int = 5):
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    print(f"\n[INSPECT] {path}")
    print("rows:", len(df), "cols:", len(df.columns))
    print("columns:", df.columns.tolist())
    print("\ndtypes:\n", df.dtypes)
    print("\nhead:\n", df.head(head_n).to_string(index=False))


# =========================
# Explanation layer helpers
# =========================
def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "session_start" in df.columns and "session_end" in df.columns:
        s = safe_to_datetime(df["session_start"])
        e = safe_to_datetime(df["session_end"])
        df["duration_s"] = (e - s).dt.total_seconds()
    return df


def robust_z(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    med = X.median(axis=0)
    mad = (X.sub(med)).abs().median(axis=0).replace(0, 1e-9)
    z = (X.sub(med)).div(mad)
    return z


def add_why_top5(df: pd.DataFrame, feature_cols: list[str], topn: int = 5) -> pd.DataFrame:
    df = df.copy()
    if not feature_cols:
        df["why_top5"] = ""
        return df

    z = robust_z(df, feature_cols)

    why = []
    for i in range(len(df)):
        s = z.iloc[i]
        top = s.abs().sort_values(ascending=False).head(topn).index.tolist()
        parts = [f"{k}:{s[k]:+.2f}" for k in top]
        why.append(", ".join(parts))
    df["why_top5"] = why
    return df


def add_error_rate_from_outcome(df: pd.DataFrame, outcome_col="outcome_class") -> pd.DataFrame:
    df = df.copy()
    if outcome_col not in df.columns:
        return df
    # 매우 단순한 휴리스틱(필요하면 너희 outcome taxonomy에 맞춰 개선)
    s = df[outcome_col].astype(str)
    is_err = s.str.contains("fail|error|deny|blocked|429|5xx", case=False, regex=True)
    df["is_error"] = is_err
    return df


# =========================
# Drilldown builders
# =========================
KEY_COLS = ["user_id", "client_name", "session_key"]


def is_event_level_raw(raw: pd.DataFrame) -> bool:
    """세션당 row가 여러 개인지로 이벤트 레벨 여부 추정."""
    for c in KEY_COLS:
        if c not in raw.columns:
            return False
    sizes = raw.groupby(KEY_COLS).size()
    return (sizes.max() if len(sizes) else 0) > 1


def build_drilldown_event_level(raw: pd.DataFrame, top_keys: pd.DataFrame) -> pd.DataFrame:
    # raw: 이벤트/관측치 1행 = 1 event 가정
    for c in KEY_COLS:
        if c not in raw.columns:
            raise ValueError(f"raw missing required key column: {c}")

    filt = raw.merge(top_keys[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    if filt.empty:
        return pd.DataFrame()

    time_col = guess_time_col(filt)

    def agg_group(g: pd.DataFrame) -> pd.Series:
        out = {
            "n_events_dd": int(len(g)),
            "n_tokens_dd": int(g["token"].nunique()) if "token" in g.columns else np.nan,
            "n_outcomes_dd": int(g["outcome_class"].nunique()) if "outcome_class" in g.columns else np.nan,
            "token_top3": topk_str(g["token"], 3) if "token" in g.columns else "",
            "route_group_top3": topk_str(g["route_group"], 3) if "route_group" in g.columns else "",
            "outcome_top3": topk_str(g["outcome_class"], 3) if "outcome_class" in g.columns else "",
        }
        if time_col:
            out["first_ts"] = g[time_col].min()
            out["last_ts"] = g[time_col].max()

        if "outcome_class" in g.columns:
            s = g["outcome_class"].astype(str)
            is_err = s.str.contains("fail|error|deny|blocked|429|5xx", case=False, regex=True)
            out["error_rate_dd"] = float(is_err.mean())
        return pd.Series(out)

    dd = filt.groupby(KEY_COLS, dropna=False).apply(agg_group).reset_index()
    return dd


def parse_maybe_list(x):
    """세션 1행 요약 raw에서 list/array 또는 문자열로 저장된 값을 최대한 list로 변환."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    # pandas may store arrays as np.ndarray/object
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        # 아주 단순: "a,b,c" 형태
        if "," in s and "[" not in s and "{" not in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        # JSON-like list 시도
        if (s.startswith("[") and s.endswith("]")):
            try:
                import json
                v = json.loads(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                return [s]
        return [s]
    # 기타 타입은 문자열화
    return [str(x)]


def build_drilldown_session_level(raw: pd.DataFrame, top_keys: pd.DataFrame) -> pd.DataFrame:
    """
    raw가 세션당 1행 요약일 때:
    - tokens/outcomes/route_group 같은 컬럼이 array(또는 문자열)로 들어있으면 요약
    - 없으면 drilldown이 빈약할 수밖에 없으니 가능한 최소만 제공
    """
    for c in KEY_COLS:
        if c not in raw.columns:
            raise ValueError(f"raw missing required key column: {c}")

    filt = raw.merge(top_keys[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    if filt.empty:
        return pd.DataFrame()

    # 후보 컬럼 이름들 (네 스키마에 맞춰 추가 가능)
    token_col = "token" if "token" in filt.columns else ("tokens" if "tokens" in filt.columns else None)
    outcome_col = "outcome_class" if "outcome_class" in filt.columns else ("outcomes" if "outcomes" in filt.columns else None)
    route_col = "route_group" if "route_group" in filt.columns else ("route_groups" if "route_groups" in filt.columns else None)

    rows = []
    for _, r in filt.iterrows():
        tokens = parse_maybe_list(r[token_col]) if token_col else []
        outs = parse_maybe_list(r[outcome_col]) if outcome_col else []
        routes = parse_maybe_list(r[route_col]) if route_col else []

        row = {
            "user_id": r["user_id"],
            "client_name": r["client_name"],
            "session_key": r["session_key"],
            "n_events_dd": int(r["n_events"]) if "n_events" in filt.columns and pd.notna(r.get("n_events")) else (len(tokens) if tokens else np.nan),
            "n_tokens_dd": len(set(tokens)) if tokens else np.nan,
            "n_outcomes_dd": len(set(outs)) if outs else np.nan,
            "token_top3": topk_str(tokens, 3) if tokens else "",
            "route_group_top3": topk_str(routes, 3) if routes else "",
            "outcome_top3": topk_str(outs, 3) if outs else "",
        }

        if outs:
            s = pd.Series([str(x) for x in outs])
            is_err = s.str.contains("fail|error|deny|blocked|429|5xx", case=False, regex=True)
            row["error_rate_dd"] = float(is_err.mean())
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Track A core
# =========================
def score_with_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # id columns to exclude from feature vector
    id_candidates = ["user_id", "client_name", "session_key", "session_start", "session_end"]
    id_cols = [c for c in id_candidates if c in df.columns]

    # exclude model outputs if rerun
    exclude = set(id_cols + ["anomaly_score", "risk_score", "risk_pct", "duration_s", "why_top5"])
    feature_cols = [c for c in df.columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No feature columns found. Check your session_features parquet schema.")

    # Ensure numeric features
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # try cast numeric; non-numeric -> NaN -> 0
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0).astype(float)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    anomaly_score = -iso.decision_function(X_scaled)
    df["anomaly_score"] = anomaly_score

    # risk_score: p1~p99 scaling + clip
    p1, p99 = np.quantile(anomaly_score, [0.01, 0.99])
    diff = float(p99 - p1)
    if diff < 1e-9:
        # fallback: percentile
        df["risk_score"] = pd.Series(anomaly_score).rank(pct=True) * 100
    else:
        df["risk_score"] = ((df["anomaly_score"] - p1) / (diff) * 100).clip(0, 100)

    # risk_pct: 항상 보기 좋은 퍼센타일(설명용)
    df["risk_pct"] = pd.Series(anomaly_score).rank(pct=True) * 100

    return df, feature_cols


def main():
    ap = argparse.ArgumentParser(description="BehaviorGuard Track A (Parquet-first) with Explanation Layer")
    ap.add_argument("--day", help="YYYY-MM-DD (pipeline friendly)")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--out-dir", default=None)

    ap.add_argument("--features-parquet", default=None, help="Override features parquet path")
    ap.add_argument("--raw-parquet", default=None, help="Override raw parquet path")

    ap.add_argument("--save-csv", type=int, default=None)
    ap.add_argument("--save-parquet", type=int, default=None)

    # inspect helper
    ap.add_argument("--inspect-parquet", default=None, help="Print schema/head of parquet file then exit")

    args = ap.parse_args()

    if args.inspect_parquet:
        inspect_parquet(Path(args.inspect_parquet))
        return

    day = args.day or env("DAY", None)
    if not day:
        raise ValueError("Provide --day YYYY-MM-DD or set DAY in env.")

    out_dir = env("OUT_DIR", args.out_dir or "out")
    topk = env("TOPK", args.topk or 100, cast=int)
    save_csv = env("SAVE_CSV", args.save_csv if args.save_csv is not None else 1, cast=int)
    save_parquet = env("SAVE_PARQUET", args.save_parquet if args.save_parquet is not None else 1, cast=int)

    out_path = ensure_dir(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # default parquet paths (match build_exports.sh)
    features_path = Path(args.features_parquet or env("FEATURES_PARQUET", f"{out_dir}/session_features_{day}.parquet"))
    raw_path = Path(args.raw_parquet or env("RAW_PARQUET", f"{out_dir}/sessions_raw_{day}.parquet"))

    if not features_path.exists():
        raise FileNotFoundError(f"features parquet not found: {features_path}")

    print(f"[INFO] Loading features: {features_path}")
    df = pd.read_parquet(features_path)
    if df.empty:
        print("[INFO] features parquet empty.")
        return

    # score
    df = add_duration(df)
    df_scored, feature_cols = score_with_isolation_forest(df)

    # explanation: why_top5 from robust z-scores
    df_scored = add_why_top5(df_scored, feature_cols, topn=5)

    # top-k selection
    top = df_scored.sort_values(["risk_pct", "risk_score"], ascending=False).head(topk).copy()

    # drilldown
    dd = pd.DataFrame()
    if raw_path.exists():
        print(f"[INFO] Loading raw for drilldown: {raw_path}")
        raw = pd.read_parquet(raw_path)
        if raw.empty:
            print("[WARN] raw parquet is empty, drilldown will be empty.")
        else:
            # Ensure keys exist; if not, drilldown can't join
            missing_keys = [c for c in KEY_COLS if c not in raw.columns]
            if missing_keys:
                print(f"[WARN] raw parquet missing key cols {missing_keys}. Cannot drilldown-join.")
            else:
                if is_event_level_raw(raw):
                    print("[INFO] raw looks EVENT-level (multiple rows per session). Building event-level drilldown.")
                    dd = build_drilldown_event_level(raw, top)
                else:
                    print("[WARN] raw looks SESSION-level (1 row per session). Building session-level drilldown fallback.")
                    dd = build_drilldown_session_level(raw, top)
    else:
        print(f"[WARN] raw parquet not found: {raw_path} (drilldown will be empty)")

    # merge drilldown summaries into top summary (always on keys)
    if not dd.empty:
        top = top.merge(dd, on=KEY_COLS, how="left")
    else:
        # still include placeholders to make the UI consistent
        for c in ["n_events_dd", "n_tokens_dd", "n_outcomes_dd", "error_rate_dd", "route_group_top3", "outcome_top3", "token_top3"]:
            if c not in top.columns:
                top[c] = np.nan if c.endswith("_dd") or c.endswith("_rate_dd") else ""

    # summary columns (human friendly)
    summary_cols = [
        "risk_pct", "risk_score", "anomaly_score",
        "user_id", "client_name", "session_key",
        "n_events", "duration_s",
        "n_events_dd", "n_tokens_dd", "n_outcomes_dd",
        "error_rate_dd",
        "route_group_top3", "outcome_top3", "token_top3",
        "why_top5",
    ]
    summary_cols = [c for c in summary_cols if c in top.columns]
    summary = top[summary_cols].copy()

    # output files
    base = f"{day}_{run_id}"
    summary_csv = out_path / f"trackA_summary_top_{base}.csv"
    drill_csv = out_path / f"trackA_drilldown_{base}.csv"
    scored_csv = out_path / f"trackA_features_scored_{base}.csv"

    if save_csv:
        df_scored.to_csv(scored_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        if not dd.empty:
            dd2 = dd.merge(top[KEY_COLS + ["risk_pct", "risk_score"]], on=KEY_COLS, how="left") if "risk_score" in top.columns else dd
            dd2.to_csv(drill_csv, index=False)

    if save_parquet:
        df_scored.to_parquet(out_path / f"trackA_features_scored_{base}.parquet", index=False)
        summary.to_parquet(out_path / f"trackA_summary_top_{base}.parquet", index=False)
        if not dd.empty:
            dd2 = dd.merge(top[KEY_COLS + ["risk_pct", "risk_score"]], on=KEY_COLS, how="left") if "risk_score" in top.columns else dd
            dd2.to_parquet(out_path / f"trackA_drilldown_{base}.parquet", index=False)

    # console print (readable)
    print("\n=== TOP-K SUMMARY (human-friendly) ===\n")
    print(summary.head(min(30, len(summary))).to_string(index=False))

    if not dd.empty:
        print("\n=== DRILLDOWN (first 20) ===\n")
        cols = [c for c in ["risk_pct", "risk_score"] + KEY_COLS + ["n_events_dd","error_rate_dd","route_group_top3","outcome_top3","token_top3","first_ts","last_ts"] if c in dd.columns or c in top.columns]
        show = dd.merge(top[KEY_COLS + ["risk_pct","risk_score"]], on=KEY_COLS, how="left") if "risk_score" in top.columns else dd
        cols = [c for c in cols if c in show.columns]
        print(show[cols].head(20).to_string(index=False))
    else:
        print("\n[INFO] Drilldown is empty or not informative. (Likely raw parquet is session-level or missing keys.)\n")

    print("\nSaved outputs to:", str(out_path))
    print(" -", summary_csv if save_csv else f"{out_path}/trackA_summary_top_{base}.parquet")
    if not dd.empty:
        print(" -", drill_csv if save_csv else f"{out_path}/trackA_drilldown_{base}.parquet")


if __name__ == "__main__":
    main()
