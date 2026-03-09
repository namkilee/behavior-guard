#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track C report generator

What it does
------------
1) Reads Track C score parquet files from:
   out/track_c/<client>/scores/*_scores.parquet

2) Produces report CSVs:
   - report_trackc_model_summary.csv
   - report_trackc_top_sessions.csv
   - report_trackc_topk_score_summary.csv

3) Produces plots:
   - learning curves (if log files are available)
   - token schema comparison
   - anomaly score histograms
   - top anomaly session charts

Notes
-----
- This script is intentionally robust to slight filename/schema differences.
- It does NOT compute true precision/recall unless you provide ground-truth labels.
- "topk_score_summary" is a score concentration summary, not true Precision@K.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import polars as pl


# -----------------------------
# Helpers
# -----------------------------

MODEL_PATTERNS = [
    ("c1", "C1"),
    ("c2", "C2"),
    ("c3", "C3"),
]


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def infer_model_from_name(name: str) -> str:
    s = name.lower()
    for needle, label in MODEL_PATTERNS:
        if needle in s:
            return label
    return "UNKNOWN"


def infer_version_from_name(name: str) -> str:
    m = re.search(r"(v[0-9]+)", name.lower())
    return m.group(1) if m else "unknown"


def parse_score_file_info(path: str, trackc_root: str) -> dict:
    """
    Expected path shape:
      <trackc_root>/<client>/scores/<something>_scores.parquet
    """
    p = Path(path)
    parts = p.parts

    client = "unknown_client"
    if "track_c" in parts:
        idx = parts.index("track_c")
        if idx + 1 < len(parts):
            client = parts[idx + 1]
    else:
        # fallback: assume .../<client>/scores/file
        if len(parts) >= 3:
            client = parts[-3]

    stem = p.stem  # e.g. c2_small_tx_lm_v1_scores
    base = re.sub(r"_scores$", "", stem)

    version = infer_version_from_name(base)
    model = infer_model_from_name(base)

    # keep a richer experiment label for plotting/debugging
    experiment = base

    return {
        "client": client,
        "model": model,
        "version": version,
        "experiment": experiment,
        "score_file": str(p),
    }


def pick_score_column(df: pl.DataFrame) -> str:
    candidates = [
        "score",
        "score_mean_nll",
        "score_mean_ce",
        "anomaly_score",
        "risk_score",
        "session_score",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a usable score column. Columns={df.columns}")


def standardize_score_df(df: pl.DataFrame) -> pl.DataFrame:
    score_col = pick_score_column(df)

    required_like = {
        "user_id": None,
        "session_key": None,
    }
    for c in required_like:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    # keep extra context columns if present
    keep = [c for c in [
        "user_id",
        "session_key",
        "outcomes",
        "route_groups",
        "tokens",
        "n_events",
        "session_start",
        "session_end",
        score_col,
    ] if c in df.columns]

    out = df.select(keep).with_columns(
        pl.col(score_col).cast(pl.Float64).alias("score")
    )

    if score_col != "score":
        out = out.drop(score_col)

    return out


def summarize_scores(df: pl.DataFrame) -> dict:
    return {
        "mean_score": float(df["score"].mean()) if len(df) else math.nan,
        "median_score": float(df["score"].median()) if len(df) else math.nan,
        "p90_score": float(df["score"].quantile(0.90)) if len(df) else math.nan,
        "p95_score": float(df["score"].quantile(0.95)) if len(df) else math.nan,
        "p99_score": float(df["score"].quantile(0.99)) if len(df) else math.nan,
        "max_score": float(df["score"].max()) if len(df) else math.nan,
        "min_score": float(df["score"].min()) if len(df) else math.nan,
        "num_sessions": int(len(df)),
    }



def parse_learning_logs(log_paths: Iterable[str]) -> pl.DataFrame:
    """
    Parse learning logs.

    Supported cases
    ---------------
    1) Separate log per experiment:
         cline_c1_v0.log
         clientA_c2_v2_train.log

    2) One combined terminal log created by:
         bash run_trackc_3gpu.sh | tee trackc_3gpu.log

       In this case, the parser tries to track the current experiment
       from command lines such as:
         python train_c1_lstm_lm.py --client cline --version v0
         CUDA_VISIBLE_DEVICES=0 python ... --client cline --version v1

       It can also recover context from saved-path lines such as:
         [DONE] best_val_loss=... saved to out/track_c/cline/runs/c1_lstm_lm/v0_123456/best.pt

    Important
    ---------
    If a combined log contains only val_loss lines and no command lines
    (or saved paths) indicating client/version, exact version separation
    is impossible because the training code prints:
      [C1][epoch 1] val_loss=...
    without client/version in the line itself.
    """
    rows = []

    val_pat = re.compile(r"\[(C[123])\]\[epoch\s+(\d+)\]\s+val_loss=([0-9.]+)")
    # command/start line examples:
    # python train_c1_lstm_lm.py --client cline --version v0
    # CUDA_VISIBLE_DEVICES=0 python3 train_c2_small_tx_lm.py --trackc-root out/track_c --client foo --version v2
    cmd_pat = re.compile(
        r"(train_(c[123])[^ ]*\.py).*?--client\s+([A-Za-z0-9_.\-]+).*?--version\s+(v[0-9]+)",
        re.IGNORECASE,
    )
    # saved path examples:
    # [DONE] best_val_loss=1.2345 saved to out/track_c/cline/runs/c1_lstm_lm/v0_123456/best.pt
    done_pat = re.compile(
        r"saved to .*?/track_c/([^/\s]+)/runs/[^/\s]+/(v[0-9]+)_[^/\s]*/best\.pt",
        re.IGNORECASE,
    )

    for path in log_paths:
        name = Path(path).name

        # filename-based fallback
        file_client = "unknown_client"
        file_version = infer_version_from_name(name)
        file_model = infer_model_from_name(name)

        m_client = re.match(r"(.+?)_(c[123])", name.lower())
        if m_client:
            file_client = m_client.group(1)

        current = {
            "client": file_client,
            "model": file_model,
            "version": file_version,
        }

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()

                m_cmd = cmd_pat.search(line)
                if m_cmd:
                    _, model_tag, client, version = m_cmd.groups()
                    current = {
                        "client": client,
                        "model": model_tag.upper(),
                        "version": version.lower(),
                    }
                    continue

                m_done = done_pat.search(line)
                if m_done:
                    client, version = m_done.groups()
                    # keep current model if already known; done lines do not always include explicit C1/C2/C3 tag
                    current["client"] = client
                    current["version"] = version.lower()

                m_val = val_pat.search(line)
                if m_val:
                    line_model, epoch, val_loss = m_val.groups()

                    # Prefer model from the val line itself.
                    model = line_model.upper()

                    # Use current tracked client/version if available.
                    client = current.get("client") or file_client
                    version = current.get("version") or file_version

                    rows.append({
                        "log_file": path,
                        "client": client,
                        "model": model,
                        "version": version,
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                    })

    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "log_file": pl.Utf8,
            "client": pl.Utf8,
            "model": pl.Utf8,
            "version": pl.Utf8,
            "epoch": pl.Int64,
            "val_loss": pl.Float64,
        }
    )


# -----------------------------
# Plotting
# -----------------------------

def save_learning_curves(df: pl.DataFrame, out_dir: str) -> None:
    if len(df) == 0:
        return

    safe_mkdir(out_dir)

    # one plot per client
    for client in df["client"].unique().to_list():
        sub = df.filter(pl.col("client") == client).sort(["model", "version", "epoch"])

        plt.figure(figsize=(9, 5))
        seen = sub.select(["model", "version"]).unique().iter_rows(named=True)
        for key in seen:
            s = sub.filter(
                (pl.col("model") == key["model"]) &
                (pl.col("version") == key["version"])
            ).sort("epoch")
            if len(s) == 0:
                continue
            label = f'{key["model"]}-{key["version"]}'
            plt.plot(s["epoch"].to_list(), s["val_loss"].to_list(), marker="o", label=label)

        plt.xlabel("epoch")
        plt.ylabel("validation loss")
        plt.title(f"Track C Learning Curves - {client}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{client}_learning_curves.png", dpi=180)
        plt.close()


def save_token_schema_comparison(summary_df: pl.DataFrame, out_dir: str) -> None:
    if len(summary_df) == 0:
        return

    safe_mkdir(out_dir)

    # grouped bar: versions on x-axis, separate bars by model, one figure per client
    for client in summary_df["client"].unique().to_list():
        sub = summary_df.filter(pl.col("client") == client)
        versions = sorted(sub["version"].unique().to_list())
        models = sorted(sub["model"].unique().to_list())

        if not versions or not models:
            continue

        x = list(range(len(versions)))
        width = 0.25 if len(models) >= 3 else 0.35

        plt.figure(figsize=(9, 5))
        for i, model in enumerate(models):
            vals = []
            for ver in versions:
                row = sub.filter((pl.col("model") == model) & (pl.col("version") == ver))
                vals.append(float(row["mean_score"][0]) if len(row) else math.nan)
            offsets = [v + (i - (len(models)-1)/2) * width for v in x]
            plt.bar(offsets, vals, width=width, label=model)

        plt.xticks(x, versions)
        plt.xlabel("token schema version")
        plt.ylabel("mean anomaly score")
        plt.title(f"Track C Token Schema Comparison - {client}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{client}_token_schema_comparison.png", dpi=180)
        plt.close()


def save_anomaly_histograms(score_long_df: pl.DataFrame, out_dir: str, bins: int = 50) -> None:
    if len(score_long_df) == 0:
        return

    safe_mkdir(out_dir)

    series = score_long_df.select(["client", "model", "version"]).unique().iter_rows(named=True)
    for key in series:
        sub = score_long_df.filter(
            (pl.col("client") == key["client"]) &
            (pl.col("model") == key["model"]) &
            (pl.col("version") == key["version"])
        )
        scores = sub["score"].to_list()
        if not scores:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=bins)
        plt.xlabel("anomaly score")
        plt.ylabel("count")
        plt.title(f'Anomaly Score Histogram - {key["client"]} / {key["model"]}-{key["version"]}')
        plt.tight_layout()
        plt.savefig(
            Path(out_dir) / f'{key["client"]}_{key["model"]}_{key["version"]}_hist.png',
            dpi=180
        )
        plt.close()


def save_top_anomaly_charts(top_df: pl.DataFrame, out_dir: str, max_items: int = 20) -> None:
    if len(top_df) == 0:
        return

    safe_mkdir(out_dir)

    series = top_df.select(["client", "model", "version"]).unique().iter_rows(named=True)
    for key in series:
        sub = top_df.filter(
            (pl.col("client") == key["client"]) &
            (pl.col("model") == key["model"]) &
            (pl.col("version") == key["version"])
        ).sort("rank")

        if len(sub) == 0:
            continue

        sub = sub.head(max_items)
        labels = [
            f'{u if u is not None else "NA"} / {s if s is not None else "NA"}'
            for u, s in zip(sub["user_id"].to_list(), sub["session_key"].to_list())
        ]
        scores = sub["score"].to_list()

        plt.figure(figsize=(10, max(5, len(labels) * 0.35)))
        plt.barh(range(len(labels)), scores)
        plt.yticks(range(len(labels)), labels)
        plt.gca().invert_yaxis()
        plt.xlabel("anomaly score")
        plt.ylabel("session")
        plt.title(f'Top Anomaly Sessions - {key["client"]} / {key["model"]}-{key["version"]}')
        plt.tight_layout()
        plt.savefig(
            Path(out_dir) / f'{key["client"]}_{key["model"]}_{key["version"]}_top_sessions.png',
            dpi=180
        )
        plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackc-root", default="out/track_c")
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--log-glob", default="", help="Optional. Example: 'logs/trackc/*.log'")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    plot_dir = report_dir / "plots"
    safe_mkdir(report_dir)
    safe_mkdir(plot_dir)

    score_files = sorted(glob.glob(f"{args.trackc_root}/*/scores/*_scores.parquet"))
    if not score_files:
        raise RuntimeError(f"No score files found under: {args.trackc_root}/*/scores/*_scores.parquet")

    summary_rows = []
    top_rows = []
    topk_rows = []
    score_long_dfs = []

    for path in score_files:
        info = parse_score_file_info(path, args.trackc_root)
        raw_df = pl.read_parquet(path)
        df = standardize_score_df(raw_df)

        # enrich
        df = df.with_columns(
            pl.lit(info["client"]).alias("client"),
            pl.lit(info["model"]).alias("model"),
            pl.lit(info["version"]).alias("version"),
            pl.lit(info["experiment"]).alias("experiment"),
        )
        score_long_dfs.append(df.select([
            "client", "model", "version", "experiment", "user_id", "session_key", "score"
        ]))

        # summary
        row = {
            "client": info["client"],
            "model": info["model"],
            "version": info["version"],
            "experiment": info["experiment"],
            "score_file": info["score_file"],
        }
        row.update(summarize_scores(df))
        summary_rows.append(row)

        # top sessions
        df_sorted = df.sort("score", descending=True)
        top_df = df_sorted.head(args.top_k)

        for rank, r in enumerate(top_df.iter_rows(named=True), start=1):
            top_rows.append({
                "client": info["client"],
                "model": info["model"],
                "version": info["version"],
                "experiment": info["experiment"],
                "rank": rank,
                "user_id": r.get("user_id"),
                "session_key": r.get("session_key"),
                "score": r["score"],
            })

        # top-k score concentration summaries
        for k0 in [10, 20, 50, 100, 200, 500]:
            k = min(k0, len(df_sorted))
            if k == 0:
                continue
            subset = df_sorted.head(k)
            topk_rows.append({
                "client": info["client"],
                "model": info["model"],
                "version": info["version"],
                "experiment": info["experiment"],
                "k": k,
                "mean_score_topk": float(subset["score"].mean()),
                "min_score_topk": float(subset["score"].min()),
                "max_score_topk": float(subset["score"].max()),
            })

    summary_df = pl.DataFrame(summary_rows).sort(["client", "model", "version"])
    top_df = pl.DataFrame(top_rows).sort(["client", "model", "version", "rank"])
    topk_df = pl.DataFrame(topk_rows).sort(["client", "model", "version", "k"])
    score_long_df = pl.concat(score_long_dfs, how="diagonal_relaxed").sort(
        ["client", "model", "version", "score"], descending=[False, False, False, True]
    )

    # save CSVs
    summary_csv = report_dir / "report_trackc_model_summary.csv"
    top_csv = report_dir / "report_trackc_top_sessions.csv"
    topk_csv = report_dir / "report_trackc_topk_score_summary.csv"
    score_long_csv = report_dir / "report_trackc_score_long.csv"

    summary_df.write_csv(summary_csv)
    top_df.write_csv(top_csv)
    topk_df.write_csv(topk_csv)
    score_long_df.write_csv(score_long_csv)

    # learning curves (optional)
    log_df = pl.DataFrame()
    if args.log_glob:
        logs = sorted(glob.glob(args.log_glob))
        if logs:
            log_df = parse_learning_logs(logs)
            if len(log_df) > 0:
                log_df.write_csv(report_dir / "report_trackc_learning_curves.csv")
                save_learning_curves(log_df, plot_dir / "learning_curves")

    # plots from scores
    save_token_schema_comparison(summary_df, plot_dir / "token_schema_comparison")
    save_anomaly_histograms(score_long_df, plot_dir / "anomaly_histograms")
    save_top_anomaly_charts(top_df, plot_dir / "top_anomaly_sessions")

    print("Track C report files created:")
    print(f"- {summary_csv}")
    print(f"- {top_csv}")
    print(f"- {topk_csv}")
    print(f"- {score_long_csv}")
    if len(log_df) > 0:
        print(f"- {report_dir / 'report_trackc_learning_curves.csv'}")
        print(f"- {plot_dir / 'learning_curves'}")
    print(f"- {plot_dir / 'token_schema_comparison'}")
    print(f"- {plot_dir / 'anomaly_histograms'}")
    print(f"- {plot_dir / 'top_anomaly_sessions'}")


if __name__ == "__main__":
    main()
