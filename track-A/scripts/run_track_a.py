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
    for cand in ["event_time", "event_ts", "timestamp", "start_time", "created_at", "ts"]:
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


def robust_stats(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    med = X.median(axis=0)
    mad = (X.sub(med)).abs().median(axis=0)
    return X, med, mad


def add_why_top5(df: pd.DataFrame, feature_cols: list[str], topn: int = 5) -> pd.DataFrame:
    """
    기존 robust z-score 기반이지만,
    MAD=0(거의 대부분 0인 count feature)에서 z가 폭발하는 문제를 해결:
      - med==0 & mad==0 & value>0  =>  rare_nonzero로 표시
      - 그 외는 z = (x-med)/(mad or 1e-9)
    출력 포맷:
      feature=value (med=..., z=...)  또는  feature=value (rare_nonzero)
    """
    df = df.copy()
    if not feature_cols:
        df["why_top5"] = ""
        return df

    X, med, mad = robust_stats(df, feature_cols)

    # z 계산(안전)
    mad_safe = mad.replace(0, np.nan)
    z = (X.sub(med)).div(mad_safe)
    z = z.replace([np.inf, -np.inf], np.nan)

    why = []
    for i in range(len(df)):
        row = X.iloc[i]
        zrow = z.iloc[i]

        # rare_nonzero 후보(대부분 0인데 이 row만 0이 아님)
        rare = []
        for k in feature_cols:
            if (med[k] == 0) and (mad[k] == 0) and (row[k] != 0):
                rare.append(k)

        # z 기반 top (rare는 z가 NaN일 수 있으니 별도로)
        zabs = zrow.abs().fillna(0.0)
        top_z = zabs.sort_values(ascending=False).head(topn).index.tolist()

        # 우선: rare_nonzero를 앞쪽에 배치(최대 topn까지만)
        picked = []
        for k in rare:
            if len(picked) >= topn:
                break
            picked.append(k)
        for k in top_z:
            if len(picked) >= topn:
                break
            if k not in picked:
                picked.append(k)

        parts = []
        for k in picked:
            if (med[k] == 0) and (mad[k] == 0) and (row[k] != 0):
                parts.append(f"{k}={row[k]:.2f} (rare_nonzero)")
            else:
                zv = zrow[k]
                if pd.isna(zv):
                    # z를 계산할 수 없으면 간단 표기
                    parts.append(f"{k}={row[k]:.2f} (med={med[k]:.2f})")
                else:
                    parts.append(f"{k}={row[k]:.2f} (med={med[k]:.2f}, z={zv:+.2f})")

        why.append("; ".join(parts))

    df["why_top5"] = why
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


def is_list_like(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def is_packed_session_raw(raw: pd.DataFrame) -> bool:
    """
    세션당 1행인데 이벤트가 array/list로 들어있는 형태인지 감지.
    예: tokens/outcomes/route_groups/event_times/dt_buckets
    """
    for c in KEY_COLS:
        if c not in raw.columns:
            return False

    candidates = ["tokens", "outcomes", "route_groups", "event_times", "dt_buckets"]
    present = [c for c in candidates if c in raw.columns]
    if not present:
        return False

    sample = raw[present].head(50)
    for c in present:
        if sample[c].apply(is_list_like).any():
            return True
    return False


def parse_maybe_list(x):
    """세션 1행 요약 raw에서 list/array 또는 문자열로 저장된 값을 최대한 list로 변환."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        # 아주 단순: "a,b,c"
        if "," in s and "[" not in s and "{" not in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        # JSON-like list
        if (s.startswith("[") and s.endswith("]")):
            try:
                import json
                v = json.loads(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def explode_packed_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    세션 1행 + 배열(raw)을 이벤트 단위 row로 explode.
    표준 컬럼명:
      token, outcome_class, route_group, dt_bucket, event_time
    """
    raw = raw.copy()

    rename_map = {}
    if "tokens" in raw.columns:
        rename_map["tokens"] = "token"
    if "outcomes" in raw.columns:
        rename_map["outcomes"] = "outcome_class"
    if "route_groups" in raw.columns:
        rename_map["route_groups"] = "route_group"
    if "dt_buckets" in raw.columns:
        rename_map["dt_buckets"] = "dt_bucket"
    if "event_times" in raw.columns:
        rename_map["event_times"] = "event_time"

    raw = raw.rename(columns=rename_map)

    explode_cols = [c for c in ["token", "outcome_class", "route_group", "dt_bucket", "event_time"] if c in raw.columns]

    # 문자열/ndarray 등 대응
    for c in explode_cols:
        raw[c] = raw[c].apply(parse_maybe_list)

    # 길이 불일치 안전장치: 가장 짧은 길이에 맞춰 자름
    def trim_row(r):
        lens = []
        for c in explode_cols:
            v = r[c]
            lens.append(len(v) if isinstance(v, list) else 0)
        m = min([x for x in lens if x > 0], default=0)
        if m == 0:
            return r
        for c in explode_cols:
            if isinstance(r[c], list) and len(r[c]) != m:
                r[c] = r[c][:m]
        return r

    raw = raw.apply(trim_row, axis=1)

    ev = raw.explode(explode_cols, ignore_index=True)

    if "event_time" in ev.columns:
        ev["event_time"] = safe_to_datetime(ev["event_time"])

    return ev


def _peak_window_30s(t: pd.Series, window_s: int = 30):
    """
    t: datetime series sorted/cleaned inside
    returns (start_ts, end_ts, max_count)
    """
    if t is None:
        return (pd.NaT, pd.NaT, 0)
    tt = pd.to_datetime(t.dropna(), errors="coerce").dropna().sort_values()
    if len(tt) == 0:
        return (pd.NaT, pd.NaT, 0)
    if len(tt) == 1:
        return (tt.iloc[0], tt.iloc[0], 1)

    arr = tt.values.astype("datetime64[ns]")
    win = np.timedelta64(window_s, "s")
    j = 0
    best_i = 0
    best_j = 0
    best = 1
    for i in range(len(arr)):
        while arr[i] - arr[j] > win:
            j += 1
        c = i - j + 1
        if c > best:
            best = c
            best_i = i
            best_j = j
    start = tt.iloc[best_j]
    end = tt.iloc[best_i]
    return (start, end, int(best))


def _fmt_ts(ts):
    if ts is None or pd.isna(ts):
        return ""
    # 초 단위까지만
    try:
        return pd.to_datetime(ts).strftime("%H:%M:%S")
    except Exception:
        return str(ts)


def build_drilldown_event_level(raw: pd.DataFrame, top_keys: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트 레벨 raw 또는 explode된 이벤트 raw를 받아 drilldown을 생성.
    요구사항(2~4):
      - burst_score, route_skew
      - timeline_1line
      - top_error_route
    """
    for c in KEY_COLS:
        if c not in raw.columns:
            raise ValueError(f"raw missing required key column: {c}")

    filt = raw.merge(top_keys[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    if filt.empty:
        return pd.DataFrame()

    time_col = guess_time_col(filt)

    def agg_group(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))

        # Top3 strings
        token_top3 = topk_str(g["token"], 3) if "token" in g.columns else ""
        route_top3 = topk_str(g["route_group"], 3) if "route_group" in g.columns else ""
        outcome_top3 = topk_str(g["outcome_class"], 3) if "outcome_class" in g.columns else ""
        dt_top3 = topk_str(g["dt_bucket"], 3) if "dt_bucket" in g.columns else ""

        # Counts/ratios
        error_rate = np.nan
        rate_limited_rate = np.nan
        top_error_route = ""
        top1_route_ratio = np.nan

        if "outcome_class" in g.columns:
            s = g["outcome_class"].astype(str)
            is_err = s.str.contains("fail|error|deny|blocked|429|5xx", case=False, regex=True)
            error_rate = float(is_err.mean())

            is_rl = s.str.contains("rate_limited|429", case=False, regex=True)
            rate_limited_rate = float(is_rl.mean())

            # top_error_route: 에러가 가장 많은 route_group
            if "route_group" in g.columns:
                err_g = g.loc[is_err, "route_group"]
                if len(err_g):
                    vc = err_g.value_counts(dropna=True)
                    if len(vc):
                        top_error_route = f"{vc.index[0]}({int(vc.iloc[0])})"

        if "route_group" in g.columns and n > 0:
            vc = g["route_group"].value_counts(dropna=True)
            if len(vc):
                top1_route_ratio = float(vc.iloc[0] / n)

        # Time/burst
        first_ts = pd.NaT
        last_ts = pd.NaT
        peak_start = pd.NaT
        peak_end = pd.NaT
        peak_events = 0
        max_events_in_30s = np.nan
        p50_gap_s = np.nan
        p95_gap_s = np.nan

        if time_col and time_col in g.columns:
            tt = pd.to_datetime(g[time_col], errors="coerce")
            first_ts = tt.min()
            last_ts = tt.max()

            # gaps
            t_sorted = tt.dropna().sort_values()
            if len(t_sorted) >= 2:
                gaps = t_sorted.diff().dt.total_seconds().dropna()
                if len(gaps):
                    p50_gap_s = float(np.quantile(gaps, 0.50))
                    p95_gap_s = float(np.quantile(gaps, 0.95))
            # peak window
            peak_start, peak_end, peak_events = _peak_window_30s(t_sorted, window_s=30)
            max_events_in_30s = float(peak_events)

        # signals
        route_skew = float(top1_route_ratio) if pd.notna(top1_route_ratio) else np.nan
        burst_score = float(max_events_in_30s / (n + 1e-9)) if pd.notna(max_events_in_30s) else np.nan

        # timeline 1-line
        tl_parts = []
        if pd.notna(first_ts) and pd.notna(last_ts):
            tl_parts.append(f"span={_fmt_ts(first_ts)}~{_fmt_ts(last_ts)}")
        if peak_events and pd.notna(peak_start) and pd.notna(peak_end):
            tl_parts.append(f"peak30s={peak_events}@{_fmt_ts(peak_start)}~{_fmt_ts(peak_end)}")
        if top_error_route:
            tl_parts.append(f"top_error_route={top_error_route}")
        if pd.notna(error_rate):
            tl_parts.append(f"error_rate={error_rate:.0%}")
        if pd.notna(rate_limited_rate):
            tl_parts.append(f"rate_limited_rate={rate_limited_rate:.0%}")
        timeline_1line = " | ".join(tl_parts)

        out = {
            "n_events_dd": n,
            "n_tokens_dd": int(pd.Series(g["token"]).dropna().nunique()) if "token" in g.columns else np.nan,
            "n_outcomes_dd": int(pd.Series(g["outcome_class"]).dropna().nunique()) if "outcome_class" in g.columns else np.nan,
            "token_top3": token_top3,
            "route_group_top3": route_top3,
            "outcome_top3": outcome_top3,
            "dt_bucket_top3": dt_top3,

            "error_rate_dd": error_rate,
            "rate_limited_rate_dd": rate_limited_rate,

            "route_skew": route_skew,
            "burst_score": burst_score,

            "first_ts": first_ts,
            "last_ts": last_ts,
            "peak_30s_start": peak_start,
            "peak_30s_end": peak_end,
            "peak_30s_events": int(peak_events) if peak_events else 0,
            "max_events_in_30s": max_events_in_30s,
            "p50_gap_s": p50_gap_s,
            "p95_gap_s": p95_gap_s,

            "top_error_route": top_error_route,
            "timeline_1line": timeline_1line,
        }
        return pd.Series(out)

    dd = filt.groupby(KEY_COLS, dropna=False).apply(agg_group).reset_index()
    return dd


def build_drilldown_from_packed_raw(raw: pd.DataFrame, top_keys: pd.DataFrame) -> pd.DataFrame:
    ev = explode_packed_raw(raw)
    return build_drilldown_event_level(ev, top_keys)


def build_drilldown_session_level(raw: pd.DataFrame, top_keys: pd.DataFrame) -> pd.DataFrame:
    """
    진짜 세션 1행 요약(배열도 없음)일 때의 fallback.
    """
    for c in KEY_COLS:
        if c not in raw.columns:
            raise ValueError(f"raw missing required key column: {c}")

    filt = raw.merge(top_keys[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    if filt.empty:
        return pd.DataFrame()

    token_col = "token" if "token" in filt.columns else ("tokens" if "tokens" in filt.columns else None)
    outcome_col = "outcome_class" if "outcome_class" in filt.columns else ("outcomes" if "outcomes" in filt.columns else None)
    route_col = "route_group" if "route_group" in filt.columns else ("route_groups" if "route_groups" in filt.columns else None)

    rows = []
    for _, r in filt.iterrows():
        tokens = parse_maybe_list(r[token_col]) if token_col else []
        outs = parse_maybe_list(r[outcome_col]) if outcome_col else []
        routes = parse_maybe_list(r[route_col]) if route_col else []

        # 간이 route_skew 추정
        route_skew = np.nan
        if routes:
            c = Counter([x for x in routes if x is not None])
            top1 = c.most_common(1)[0][1] if c else 0
            route_skew = float(top1 / (len(routes) + 1e-9))

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
            "dt_bucket_top3": "",
            "error_rate_dd": np.nan,
            "rate_limited_rate_dd": np.nan,
            "route_skew": route_skew,
            "burst_score": np.nan,
            "max_events_in_30s": np.nan,
            "p50_gap_s": np.nan,
            "p95_gap_s": np.nan,
            "top_error_route": "",
            "timeline_1line": "",
        }

        if outs:
            s = pd.Series([str(x) for x in outs])
            is_err = s.str.contains("fail|error|deny|blocked|429|5xx", case=False, regex=True)
            is_rl = s.str.contains("rate_limited|429", case=False, regex=True)
            row["error_rate_dd"] = float(is_err.mean())
            row["rate_limited_rate_dd"] = float(is_rl.mean())
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Track A core
# =========================
def score_with_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    id_candidates = ["user_id", "client_name", "session_key", "session_start", "session_end"]
    id_cols = [c for c in id_candidates if c in df.columns]

    exclude = set(id_cols + ["anomaly_score", "risk_score", "risk_pct", "duration_s", "why_top5", "why_ranked", "risk_tags"])
    feature_cols = [c for c in df.columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No feature columns found. Check your session_features parquet schema.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
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

    p1, p99 = np.quantile(anomaly_score, [0.01, 0.99])
    diff = float(p99 - p1)
    if diff < 1e-9:
        df["risk_score"] = pd.Series(anomaly_score).rank(pct=True) * 100
    else:
        df["risk_score"] = ((df["anomaly_score"] - p1) / (diff) * 100).clip(0, 100)

    df["risk_pct"] = pd.Series(anomaly_score).rank(pct=True) * 100

    return df, feature_cols


# =========================
# Post-merge explanation: tags + ranked why
# =========================
def add_risk_tags_and_ranked_why(top: pd.DataFrame) -> pd.DataFrame:
    """
    (2) signals: burst_score, route_skew (already computed in drilldown)
    (3) why priority order:
        error_rate
        volume/burst
        duration
        token heavy
        route skew
    + tags
    """
    top = top.copy()

    def fmt_pct(x):
        return "" if pd.isna(x) else f"{x:.0%}"

    tags_list = []
    why_ranked = []

    for _, r in top.iterrows():
        tags = []
        why = []

        n = r.get("n_events_dd", np.nan)
        if pd.isna(n):
            n = r.get("n_events", np.nan)

        # 1) error/rate-limit signals
        er = r.get("error_rate_dd", np.nan)
        rl = r.get("rate_limited_rate_dd", np.nan)
        if pd.notna(er) and er >= 0.20:
            tags.append("ERROR_HEAVY")
            why.append(f"error_rate={fmt_pct(er)}")
        if pd.notna(rl) and rl >= 0.20:
            tags.append("RATE_LIMIT_HEAVY")
            why.append(f"rate_limited_rate={fmt_pct(rl)}")

        # 2) volume/burst
        max30 = r.get("max_events_in_30s", np.nan)
        bs = r.get("burst_score", np.nan)
        if pd.notna(max30) and max30 >= 20:
            tags.append("BURST")
            why.append(f"burst_peak30s={int(max30)}")
        elif pd.notna(bs) and bs >= 0.05:
            tags.append("BURST_DENSE")
            why.append(f"burst_score={bs:.2f}")

        # 3) duration
        dur = r.get("duration_s", np.nan)
        if pd.notna(dur) and dur >= 1800:  # 30분 이상이면 표시(연구용 기본값)
            tags.append("LONG_DURATION")
            why.append(f"duration_s={dur:.0f}")

        # 4) token heavy (여기서 token은 네 raw "token" 카테고리/버킷이지만)
        utok = r.get("n_tokens_dd", np.nan)
        if pd.notna(utok) and utok >= 10:
            tags.append("TOKEN_DIVERSE")
            why.append(f"n_unique_tokens={int(utok)}")

        # 5) route skew
        rs = r.get("route_skew", np.nan)
        if pd.notna(rs) and rs >= 0.90:
            tags.append("ROUTE_SKEW")
            why.append(f"route_skew={rs:.0%}")

        # 보강: why_top5를 뒤에 붙이되, 너무 길면 앞부분만
        wt = r.get("why_top5", "")
        if isinstance(wt, str) and wt.strip():
            why.append(f"features: {wt}")

        tags_list.append(",".join(tags))
        why_ranked.append(" | ".join(why))

    top["risk_tags"] = tags_list
    top["why_ranked"] = why_ranked
    return top


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="BehaviorGuard Track A (Parquet-first) with Explanation Layer")
    ap.add_argument("--day", help="YYYY-MM-DD (pipeline friendly)")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--out-dir", default=None)

    ap.add_argument("--features-parquet", default=None, help="Override features parquet path")
    ap.add_argument("--raw-parquet", default=None, help="Override raw parquet path")

    ap.add_argument("--save-csv", type=int, default=None)
    ap.add_argument("--save-parquet", type=int, default=None)

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

    # explanation: why_top5 (stable)
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
            missing_keys = [c for c in KEY_COLS if c not in raw.columns]
            if missing_keys:
                print(f"[WARN] raw parquet missing key cols {missing_keys}. Cannot drilldown-join.")
            else:
                if is_event_level_raw(raw):
                    print("[INFO] raw looks EVENT-level (multiple rows per session). Building event-level drilldown.")
                    dd = build_drilldown_event_level(raw, top)
                elif is_packed_session_raw(raw):
                    print("[INFO] raw looks PACKED session-level (arrays). Exploding then building event-level drilldown.")
                    dd = build_drilldown_from_packed_raw(raw, top)
                else:
                    print("[WARN] raw looks SESSION-level (1 row per session, no arrays). Building fallback drilldown.")
                    dd = build_drilldown_session_level(raw, top)
    else:
        print(f"[WARN] raw parquet not found: {raw_path} (drilldown will be empty)")

    # merge drilldown summaries into top summary
    if not dd.empty:
        top = top.merge(dd, on=KEY_COLS, how="left")
    else:
        # placeholders
        for c in [
            "n_events_dd", "n_tokens_dd", "n_outcomes_dd",
            "error_rate_dd", "rate_limited_rate_dd",
            "route_skew", "burst_score",
            "max_events_in_30s", "p50_gap_s", "p95_gap_s",
            "dt_bucket_top3",
            "route_group_top3", "outcome_top3", "token_top3",
            "timeline_1line", "top_error_route",
            "peak_30s_start", "peak_30s_end", "peak_30s_events",
        ]:
            if c not in top.columns:
                top[c] = np.nan if (c.endswith("_dd") or c.endswith("_rate_dd") or c.endswith("_s") or c.startswith("peak_") or c in ["route_skew","burst_score","max_events_in_30s"]) else ""

    # post-merge: tags + ranked why (priority)
    top = add_risk_tags_and_ranked_why(top)

    # summary columns
    summary_cols = [
        "risk_pct", "risk_score", "anomaly_score",
        "user_id", "client_name", "session_key",
        "n_events", "duration_s",
        "n_events_dd", "n_tokens_dd", "n_outcomes_dd",
        "error_rate_dd", "rate_limited_rate_dd",
        "burst_score", "route_skew",
        "max_events_in_30s", "p50_gap_s", "p95_gap_s",
        "dt_bucket_top3",
        "route_group_top3", "outcome_top3", "token_top3",
        "top_error_route",
        "risk_tags",
        "timeline_1line",
        "why_ranked",
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

    # console print
    print("\n=== TOP-K SUMMARY (human-friendly) ===\n")
    print(summary.head(min(30, len(summary))).to_string(index=False))

    if not dd.empty:
        print("\n=== DRILLDOWN (first 20) ===\n")
        show = dd.merge(top[KEY_COLS + ["risk_pct", "risk_score"]], on=KEY_COLS, how="left") if "risk_score" in top.columns else dd
        cols = [c for c in [
            "risk_pct", "risk_score",
            "user_id", "client_name", "session_key",
            "n_events_dd",
            "error_rate_dd", "rate_limited_rate_dd",
            "burst_score", "route_skew",
            "max_events_in_30s", "p50_gap_s", "p95_gap_s",
            "peak_30s_events", "peak_30s_start", "peak_30s_end",
            "dt_bucket_top3",
            "route_group_top3", "outcome_top3", "token_top3",
            "top_error_route",
            "timeline_1line",
            "first_ts", "last_ts"
        ] if c in show.columns]
        print(show[cols].head(20).to_string(index=False))
    else:
        print("\n[INFO] Drilldown is empty or not informative.\n")

    print("\nSaved outputs to:", str(out_path))
    print(" -", summary_csv if save_csv else f"{out_path}/trackA_summary_top_{base}.parquet")
    if not dd.empty:
        print(" -", drill_csv if save_csv else f"{out_path}/trackA_drilldown_{base}.parquet")


if __name__ == "__main__":
    main()