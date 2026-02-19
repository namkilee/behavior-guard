import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import clickhouse_connect
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def load_env_file(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def resolve_value(env_key, arg_value, default_value=None, cast=None):
    if env_key in os.environ and os.environ[env_key] != "":
        value = os.environ[env_key]
    elif arg_value is not None:
        value = arg_value
    else:
        value = default_value
    if cast and value is not None:
        return cast(value)
    return value


def load_sql_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def apply_placeholders(sql: str, mapping: dict) -> str:
    for k, v in mapping.items():
        sql = sql.replace(k, v)
    # 남은 {{...}} 있으면 빠르게 실패
    if "{{" in sql and "}}" in sql:
        # 완전한 템플릿 엔진은 아니지만, 실험용으로 충분
        leftovers = [seg for seg in sql.split() if "{{" in seg and "}}" in seg]
        if leftovers:
            raise ValueError(f"Unreplaced placeholders found: {set(leftovers)}")
    return sql


def ch_tuple_escape(s: str) -> str:
    # ClickHouse SQL string literal escape
    return "'" + (s or "").replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_in_tuples(df_top: pd.DataFrame) -> str:
    # (user_id, client_name, session_key) IN ((...),(...))
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


def ensure_out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    load_env_file()

    ap = argparse.ArgumentParser(description="BehaviorGuard Track A: Statistical baseline + Drilldown")
    ap.add_argument("--sql", help="Feature SQL file path")
    ap.add_argument("--drilldown-sql", help="Drilldown SQL file path")
    ap.add_argument("--day-start")
    ap.add_argument("--day-end")

    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--database")

    ap.add_argument("--topk", type=int)
    ap.add_argument("--out-dir", help="Output directory")
    ap.add_argument("--save-csv", type=int, help="1/0")
    ap.add_argument("--save-parquet", type=int, help="1/0")

    args = ap.parse_args()

    feature_sql_path = resolve_value("FEATURE_SQL_PATH", args.sql, "sql/bg_session_features.sql")
    drill_sql_path = resolve_value("DRILLDOWN_SQL_PATH", args.drilldown_sql, "sql/bg_sessions_drilldown.sql")

    day_start = resolve_value("DAY_START", args.day_start, None)
    day_end = resolve_value("DAY_END", args.day_end, None)
    if not day_start or not day_end:
        raise ValueError("DAY_START and DAY_END must be provided (env or arg).")

    host = resolve_value("CH_HOST", args.host, "localhost")
    port = resolve_value("CH_PORT", args.port, 8123, cast=int)
    user = resolve_value("CH_USER", args.user, "default")
    password = resolve_value("CH_PASSWORD", args.password, "")
    database = resolve_value("CH_DATABASE", args.database, "default")

    topk = resolve_value("TOPK", args.topk, 100, cast=int)
    out_dir = resolve_value("OUT_DIR", args.out_dir, "out")
    save_csv = resolve_value("SAVE_CSV", args.save_csv, 1, cast=int)
    save_parquet = resolve_value("SAVE_PARQUET", args.save_parquet, 1, cast=int)

    out_path = ensure_out_dir(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    ch = clickhouse_connect.get_client(
        host=host, port=port, username=user, password=password, database=database
    )

    # ---- 1) Feature query ----
    feature_sql_raw = load_sql_file(feature_sql_path)
    feature_sql = apply_placeholders(
        feature_sql_raw,
        {
            "{{DAY_START}}": day_start,
            "{{DAY_END}}": day_end,
        },
    )

    df = ch.query_df(feature_sql)
    if df.empty:
        print("No rows returned from feature query.")
        return

    id_cols = ["user_id", "client_name", "session_key", "session_start", "session_end"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    X = df[feature_cols].fillna(0.0).astype(float)
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
    df["risk_score"] = ((df["anomaly_score"] - p1) / (p99 - p1 + 1e-9) * 100).clip(0, 100)

    top = df.sort_values("risk_score", ascending=False).head(topk).copy()

    # ---- 2) Drilldown query for Top-K ----
    in_tuples = build_in_tuples(top)

    drill_sql_raw = load_sql_file(drill_sql_path)
    drill_sql = apply_placeholders(
        drill_sql_raw,
        {
            "{{DAY_START}}": day_start,
            "{{DAY_END}}": day_end,
            "{{IN_TUPLES}}": in_tuples,
        },
    )

    drill = ch.query_df(drill_sql)

    # risk_score를 drilldown 결과에도 붙이기(조인)
    drill = drill.merge(
        top[["user_id", "client_name", "session_key", "risk_score"]],
        on=["user_id", "client_name", "session_key"],
        how="left",
    ).sort_values("risk_score", ascending=False)

    # ---- 3) Save outputs ----
    features_out = out_path / f"trackA_features_{run_id}.csv"
    top_out = out_path / f"trackA_top_{run_id}.csv"
    drill_out = out_path / f"trackA_drilldown_{run_id}.csv"

    if save_csv:
        df.to_csv(features_out, index=False)
        top.to_csv(top_out, index=False)
        drill.to_csv(drill_out, index=False)

    if save_parquet:
        df.to_parquet(out_path / f"trackA_features_{run_id}.parquet", index=False)
        top.to_parquet(out_path / f"trackA_top_{run_id}.parquet", index=False)
        drill.to_parquet(out_path / f"trackA_drilldown_{run_id}.parquet", index=False)

    # ---- 4) Console summary ----
    print("\n=== TOP ANOMALOUS SESSIONS (summary) ===\n")
    show_cols = [
        "user_id", "client_name", "session_key",
        "session_start", "session_end",
        "n_events", "risk_score"
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    print(top[show_cols].to_string(index=False))

    print("\n=== DRILLDOWN (first 10) ===\n")
    drill_show_cols = ["risk_score", "user_id", "client_name", "session_key", "n_events", "tokens", "outcomes"]
    drill_show_cols = [c for c in drill_show_cols if c in drill.columns]
    print(drill[drill_show_cols].head(10).to_string(index=False))

    print("\nSaved outputs to:", str(out_path))


if __name__ == "__main__":
    main()
