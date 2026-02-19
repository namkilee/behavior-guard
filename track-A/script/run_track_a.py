import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

try:
    import clickhouse_connect
except Exception:
    clickhouse_connect = None


# ----------------------------
# Utils
# ----------------------------
def resolve_env(key: str, default=None, cast=None):
    v = os.environ.get(key, "")
    if v == "":
        v = default
    if cast and v is not None:
        return cast(v)
    return v


def ensure_out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def day_range(day: str):
    # day: YYYY-MM-DD
    start = f"{day} 00:00:00"
    d = datetime.strptime(day, "%Y-%m-%d")
    end = (d + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    return start, end


def ch_tuple_escape(s: str) -> str:
    return "'" + (s or "").replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_in_tuples(df_top: pd.DataFrame) -> str:
    tuples = []
    for _, row in df_top.iterrows():
        tuples.append(
            "("
            + ",".join(
                [
                    ch_tuple_escape(str(row["user_id"])),
                    ch_tuple_escape(str(row["client_name"])),
                    ch_tuple_escape(str(row["session_key"])),
                ]
            )
            + ")"
        )
    return ",".join(tuples) if tuples else "(NULL,NULL,NULL)"


def load_sql_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def apply_placeholders(sql: str, mapping: dict) -> str:
    for k, v in mapping.items():
        sql = sql.replace(k, v)
    if "{{" in sql and "}}" in sql:
        leftovers = [seg for seg in sql.split() if "{{" in seg and "}}" in seg]
        if leftovers:
            raise ValueError(f"Unreplaced placeholders found: {set(leftovers)}")
    return sql


# ----------------------------
# Data loading
# ----------------------------
def load_features_from_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_raw_from_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def ch_client_from_env():
    if clickhouse_connect is None:
        raise RuntimeError("clickhouse_connect is not installed, cannot use clickhouse source.")

    host = resolve_env("CH_HOST", "localhost")
    port = resolve_env("CH_PORT", 8123, cast=int)
    user = resolve_env("CH_USER", "default")
    password = resolve_env("CH_PASSWORD", "")
    database = resolve_env("CH_DATABASE", "default")

    return clickhouse_connect.get_client(
        host=host, port=port, username=user, password=password, database=database
    )


def load_features_from_clickhouse(feature_sql_path: str, day_start: str, day_end: str) -> pd.DataFrame:
    ch = ch_client_from_env()
    sql_raw = load_sql_file(feature_sql_path)
    sql = apply_placeholders(sql_raw, {"{{DAY_START}}": day_start, "{{DAY_END}}": day_end})
    return ch.query_df(sql)


def load_drilldown_from_clickhouse(drill_sql_path: str, day_start: str, day_end: str, in_tuples: str) -> pd.DataFrame:
    ch = ch_client_from_env()
    sql_raw = load_sql_file(drill_sql_path)
    sql = apply_placeholders(
        sql_raw,
        {"{{DAY_START}}": day_start, "{{DAY_END}}": day_end, "{{IN_TUPLES}}": in_tuples},
    )
    return ch.query_df(sql)


# ----------------------------
# Drilldown for parquet mode
# ----------------------------
def drilldown_from_raw_parquet(raw: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """
    raw parquet가 세션 단위가 아니라 "이벤트/관측치 row"라는 가정 하에,
    Top-K 세션만 필터링해서 간단 집계(drilldown summary)를 만든다.

    raw에 아래 컬럼이 있으면 활용:
      - user_id, client_name, session_key (필수)
      - token, outcome_class, route_group, dt_bucket 등(있으면 카운트/요약)
      - timestamp / event_ts / start_time 등(있으면 기간)
    """
    key_cols = ["user_id", "client_name", "session_key"]
    for c in key_cols:
        if c not in raw.columns:
            raise ValueError(f"raw parquet missing required column: {c}")

    # Top-K 키로 inner join 필터
    keys = top[key_cols].drop_duplicates()
    filt = raw.merge(keys, on=key_cols, how="inner")

    if filt.empty:
        return pd.DataFrame()

    # 가능한 컬럼들로 요약
    agg = {c: "count" for c in key_cols[:1]}  # dummy count source
    # 더 의미 있는 집계
    if "token" in filt.columns:
        agg["token"] = "nunique"
    if "outcome_class" in filt.columns:
        agg["outcome_class"] = "nunique"
    if "route_group" in filt.columns:
        agg["route_group"] = "nunique"
    if "dt_bucket" in filt.columns:
        agg["dt_bucket"] = "nunique"

    # time span
    time_col = None
    for cand in ["event_ts", "timestamp", "start_time", "created_at"]:
        if cand in filt.columns:
            time_col = cand
            break

    if time_col:
        # min/max
        grouped = filt.groupby(key_cols).agg(
            n_events=(key_cols[0], "count"),
            n_tokens=("token", "nunique") if "token" in filt.columns else (key_cols[0], "count"),
            n_outcomes=("outcome_class", "nunique") if "outcome_class" in filt.columns else (key_cols[0], "count"),
            first_ts=(time_col, "min"),
            last_ts=(time_col, "max"),
        ).reset_index()
    else:
        grouped = filt.groupby(key_cols).agg(
            n_events=(key_cols[0], "count"),
            n_tokens=("token", "nunique") if "token" in filt.columns else (key_cols[0], "count"),
            n_outcomes=("outcome_class", "nunique") if "outcome_class" in filt.columns else (key_cols[0], "count"),
        ).reset_index()

    # risk_score join
    out = grouped.merge(
        top[key_cols + ["risk_score"]],
        on=key_cols,
        how="left",
    ).sort_values("risk_score", ascending=False)

    return out


# ----------------------------
# Model
# ----------------------------
def score_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    id_cols = ["user_id", "client_name", "session_key", "session_start", "session_end"]
    # 존재하는 id 컬럼만 제외
    id_cols_present = [c for c in id_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols_present]

    if not feature_cols:
        raise ValueError("No feature columns found (after excluding id columns).")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

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
    df = df.copy()
    df["anomaly_score"] = anomaly_score

    p1, p99 = np.quantile(anomaly_score, [0.01, 0.99])
    df["risk_score"] = ((df["anomaly_score"] - p1) / (p99 - p1 + 1e-9) * 100).clip(0, 100)

    return df


def main():
    ap = argparse.ArgumentParser(description="BehaviorGuard Track A (Parquet-first; ClickHouse optional)")
    ap.add_argument("--day", help="YYYY-MM-DD (preferred for pipeline)")
    ap.add_argument("--day-start")
    ap.add_argument("--day-end")

    ap.add_argument("--source", choices=["auto", "parquet", "clickhouse"], default="auto")

    # parquet inputs (default: out/)
    ap.add_argument("--features-parquet", help="session_features parquet path")
    ap.add_argument("--raw-parquet", help="sessions_raw parquet path (for drilldown)")

    # clickhouse inputs (optional)
    ap.add_argument("--feature-sql", help="feature SQL path (built sql ok)")
    ap.add_argument("--drilldown-sql", help="drilldown SQL path")

    # outputs
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-csv", type=int, default=None)
    ap.add_argument("--save-parquet", type=int, default=None)

    args = ap.parse_args()

    # env defaults (pipeline에서 export되어 내려온다고 가정)
    out_dir = resolve_env("OUT_DIR", args.out_dir or "out")
    topk = resolve_env("TOPK", args.topk or 100, cast=int)
    save_csv = resolve_env("SAVE_CSV", args.save_csv if args.save_csv is not None else 1, cast=int)
    save_parquet = resolve_env("SAVE_PARQUET", args.save_parquet if args.save_parquet is not None else 1, cast=int)

    # day window
    if args.day:
        day_start, day_end = day_range(args.day)
    else:
        day_start = args.day_start or resolve_env("DAY_START", None)
        day_end = args.day_end or resolve_env("DAY_END", None)
        if not day_start or not day_end:
            raise ValueError("Provide --day (preferred) or DAY_START/DAY_END (args/env).")

    out_path = ensure_out_dir(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # parquet default paths (match build_exports.sh)
    day_for_paths = args.day
    if not day_for_paths:
        # DAY_START 기준으로 파일명 추정(YYYY-MM-DD)
        day_for_paths = str(day_start).split(" ")[0]

    features_parquet = Path(args.features_parquet or resolve_env("FEATURES_PARQUET", f"{out_dir}/session_features_{day_for_paths}.parquet"))
    raw_parquet = Path(args.raw_parquet or resolve_env("RAW_PARQUET", f"{out_dir}/sessions_raw_{day_for_paths}.parquet"))

    # clickhouse sql defaults (built sql 사용)
    feature_sql_path = args.feature_sql or resolve_env("FEATURE_SQL_PATH", "sql/build/export_session_features.built.sql")
    drill_sql_path = args.drilldown_sql or resolve_env("DRILLDOWN_SQL_PATH", "sql/bg_sessions_drilldown.sql")

    # -------------------------
    # Load features
    # -------------------------
    df_features = pd.DataFrame()

    if args.source in ["auto", "parquet"]:
        if features_parquet.exists():
            print(f"[INFO] Loading features from parquet: {features_parquet}")
            df_features = load_features_from_parquet(features_parquet)
        elif args.source == "parquet":
            raise FileNotFoundError(f"Features parquet not found: {features_parquet}")

    if df_features.empty and args.source in ["auto", "clickhouse"]:
        print(f"[INFO] Loading features from ClickHouse using SQL: {feature_sql_path}")
        df_features = load_features_from_clickhouse(feature_sql_path, day_start, day_end)

    if df_features.empty:
        print("[INFO] No rows returned for features.")
        return

    # -------------------------
    # Score
    # -------------------------
    df_scored = score_features(df_features)
    top = df_scored.sort_values("risk_score", ascending=False).head(topk).copy()

    # -------------------------
    # Drilldown
    # -------------------------
    drill = pd.DataFrame()

    # Parquet drilldown (preferred in current mode)
    if raw_parquet.exists():
        print(f"[INFO] Loading raw from parquet for drilldown: {raw_parquet}")
        raw = load_raw_from_parquet(raw_parquet)
        if not raw.empty:
            drill = drilldown_from_raw_parquet(raw, top)

    # If parquet drilldown didn't work and ClickHouse is allowed, try ClickHouse drilldown SQL
    if drill.empty and args.source in ["auto", "clickhouse"] and drill_sql_path:
        if clickhouse_connect is None:
            print("[WARN] clickhouse_connect not available, skipping clickhouse drilldown.")
        else:
            print(f"[INFO] Loading drilldown from ClickHouse using SQL: {drill_sql_path}")
            in_tuples = build_in_tuples(top)
            drill = load_drilldown_from_clickhouse(drill_sql_path, day_start, day_end, in_tuples)

            if not drill.empty and "risk_score" not in drill.columns:
                drill = drill.merge(
                    top[["user_id", "client_name", "session_key", "risk_score"]],
                    on=["user_id", "client_name", "session_key"],
                    how="left",
                ).sort_values("risk_score", ascending=False)

    # -------------------------
    # Save
    # -------------------------
    features_out_csv = out_path / f"trackA_features_{run_id}.csv"
    top_out_csv = out_path / f"trackA_top_{run_id}.csv"
    drill_out_csv = out_path / f"trackA_drilldown_{run_id}.csv"

    if save_csv:
        df_scored.to_csv(features_out_csv, index=False)
        top.to_csv(top_out_csv, index=False)
        if not drill.empty:
            drill.to_csv(drill_out_csv, index=False)

    if save_parquet:
        df_scored.to_parquet(out_path / f"trackA_features_{run_id}.parquet", index=False)
        top.to_parquet(out_path / f"trackA_top_{run_id}.parquet", index=False)
        if not drill.empty:
            drill.to_parquet(out_path / f"trackA_drilldown_{run_id}.parquet", index=False)

    # -------------------------
    # Console summary
    # -------------------------
    print("\n=== TOP ANOMALOUS SESSIONS (summary) ===\n")
    show_cols = ["user_id", "client_name", "session_key", "session_start", "session_end", "n_events", "risk_score"]
    show_cols = [c for c in show_cols if c in top.columns]
    print(top[show_cols].to_string(index=False))

    if not drill.empty:
        print("\n=== DRILLDOWN (first 10) ===\n")
        drill_show_cols = ["risk_score", "user_id", "client_name", "session_key", "n_events", "n_tokens", "n_outcomes", "first_ts", "last_ts"]
        drill_show_cols = [c for c in drill_show_cols if c in drill.columns]
        print(drill[drill_show_cols].head(10).to_string(index=False))
    else:
        print("\n[INFO] Drilldown is empty (raw parquet missing or schema mismatch, and clickhouse drilldown not used).\n")

    print("\nSaved outputs to:", str(out_path))


if __name__ == "__main__":
    main()
