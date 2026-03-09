#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import polars as pl

TRACKC_ROOT = "out/track_c"
TOP_K = 50

precision_rows = []
top_rows = []
summary_rows = []

score_files = glob.glob(f"{TRACKC_ROOT}/*/scores/*_scores.parquet")

if not score_files:
    raise RuntimeError("No score files found")

for path in score_files:

    parts = path.split("/")
    client = parts[2]

    filename = os.path.basename(path)
    model = filename.split("_")[0]

    df = pl.read_parquet(path)

    if "score_mean_nll" in df.columns:
        score_col = "score_mean_nll"
    else:
        score_col = "score_mean_ce"

    df = df.with_columns(
        pl.col(score_col).alias("score")
    )

    df_sorted = df.sort("score", descending=True)

    # -------------------------
    # Top anomaly sessions
    # -------------------------

    top_df = df_sorted.head(TOP_K)

    for r in top_df.iter_rows(named=True):
        top_rows.append({
            "client": client,
            "model": model,
            "user_id": r.get("user_id"),
            "session_key": r.get("session_key"),
            "score": r["score"]
        })

    # -------------------------
    # score summary
    # -------------------------

    summary_rows.append({
        "client": client,
        "model": model,
        "mean_score": df["score"].mean(),
        "p95_score": df["score"].quantile(0.95),
        "max_score": df["score"].max(),
        "num_sessions": len(df)
    })

    # -------------------------
    # simple Precision@K proxy
    # -------------------------

    for k in [50,100,200,500]:

        k = min(k, len(df_sorted))

        subset = df_sorted.head(k)

        precision_rows.append({
            "client": client,
            "model": model,
            "k": k,
            "mean_score_topk": subset["score"].mean()
        })


# -----------------------------
# Save outputs
# -----------------------------

os.makedirs("report", exist_ok=True)

pl.DataFrame(summary_rows).write_csv(
    "report/report_trackc_model_summary.csv"
)

pl.DataFrame(top_rows).write_csv(
    "report/report_trackc_top_sessions.csv"
)

pl.DataFrame(precision_rows).write_csv(
    "report/report_trackc_precision_proxy.csv"
)

print("Track C report files created:")

print("report/report_trackc_model_summary.csv")
print("report/report_trackc_top_sessions.csv")
print("report/report_trackc_precision_proxy.csv")