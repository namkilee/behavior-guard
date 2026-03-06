#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_df(input_path: str) -> pl.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory: {input_path}")
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")

    return pl.read_parquet(input_path)


def add_duration_seconds(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("session_end") - pl.col("session_start"))
            .dt.total_milliseconds()
            .truediv(1000.0)
        ).alias("duration_sec")
    )


def make_summary(df: pl.DataFrame) -> dict:
    period = df.select([
        pl.col("session_start").min().alias("analysis_start"),
        pl.col("session_end").max().alias("analysis_end"),
    ]).to_dicts()[0]

    stats = df.select([
        pl.len().alias("total_sessions"),
        pl.col("user_id").drop_nulls().n_unique().alias("unique_users"),
        pl.col("client_name").drop_nulls().n_unique().alias("unique_clients"),
        pl.col("n_events").mean().alias("avg_events_per_session"),
        pl.col("n_events").median().alias("median_events_per_session"),
        pl.col("n_events").max().alias("max_events_per_session"),
        pl.col("duration_sec").mean().alias("avg_duration_sec"),
        pl.col("duration_sec").median().alias("median_duration_sec"),
        pl.col("duration_sec").max().alias("max_duration_sec"),
    ]).to_dicts()[0]

    result = {**period, **stats}
    return result


def save_summary(summary: dict, out_dir: str) -> None:
    out_path = Path(out_dir) / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def build_sessions_per_day(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("session_start").dt.date().alias("day"))
        .group_by("day")
        .agg(pl.len().alias("session_count"))
        .sort("day")
    )


def plot_sessions_per_day(day_df: pl.DataFrame, out_dir: str) -> None:
    x = day_df["day"].to_list()
    y = day_df["session_count"].to_list()

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Sessions")
    plt.title("Sessions per Day")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "sessions_per_day.png", dpi=150)
    plt.close()


def plot_n_events_hist(df: pl.DataFrame, out_dir: str, bins: int) -> None:
    values = df["n_events"].drop_nulls().to_list()

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.xlabel("Events per Session")
    plt.ylabel("Number of Sessions")
    plt.title("Session Event Distribution")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "n_events_hist.png", dpi=150)
    plt.close()


def plot_duration_vs_events(df: pl.DataFrame, out_dir: str, sample_n: int | None) -> None:
    plot_df = df.select(["duration_sec", "n_events"]).drop_nulls()

    if sample_n is not None and sample_n > 0 and plot_df.height > sample_n:
        plot_df = plot_df.sample(n=sample_n, shuffle=True, seed=42)

    x = plot_df["duration_sec"].to_list()
    y = plot_df["n_events"].to_list()

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.25)
    plt.xlabel("Session Duration (sec)")
    plt.ylabel("Events per Session")
    plt.title("Session Duration vs Events")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "duration_vs_events.png", dpi=150)
    plt.close()


def build_top_sessions(df: pl.DataFrame, top_k: int) -> pl.DataFrame:
    cols = [
        "user_id",
        "client_name",
        "session_key",
        "session_start",
        "session_end",
        "duration_sec",
        "n_events",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    return (
        df.select(existing_cols)
        .sort(["n_events", "duration_sec"], descending=[True, False])
        .head(top_k)
    )


def build_client_counts(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("client_name")
        .agg(pl.len().alias("session_count"))
        .sort("session_count", descending=True)
    )


def plot_client_counts(client_df: pl.DataFrame, out_dir: str) -> None:
    labels = [str(x) for x in client_df["client_name"].to_list()]
    values = client_df["session_count"].to_list()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel("Client")
    plt.ylabel("Sessions")
    plt.title("Sessions by Client")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "client_session_counts.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report-ready stats and plots from merged sessions_raw parquet."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to merged parquet file or directory containing parquet files",
    )
    parser.add_argument(
        "--out-dir",
        default="report_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K sessions by n_events",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for n_events histogram",
    )
    parser.add_argument(
        "--scatter-sample",
        type=int,
        default=20000,
        help="Sample size for scatter plot; use 0 to disable sampling",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = load_df(args.input)
    df = add_duration_seconds(df)

    summary = make_summary(df)
    save_summary(summary, args.out_dir)

    day_df = build_sessions_per_day(df)
    day_df.write_csv(Path(args.out_dir) / "sessions_per_day.csv")
    plot_sessions_per_day(day_df, args.out_dir)

    plot_n_events_hist(df, args.out_dir, bins=args.hist_bins)

    scatter_sample = None if args.scatter_sample == 0 else args.scatter_sample
    plot_duration_vs_events(df, args.out_dir, sample_n=scatter_sample)

    top_df = build_top_sessions(df, top_k=args.top_k)
    top_df.write_csv(Path(args.out_dir) / "top_sessions_by_events.csv")

    client_df = build_client_counts(df)
    client_df.write_csv(Path(args.out_dir) / "client_session_counts.csv")
    plot_client_counts(client_df, args.out_dir)

    print(f"[DONE] Outputs saved to: {args.out_dir}")
    print("Generated files:")
    for name in [
        "summary.json",
        "sessions_per_day.csv",
        "sessions_per_day.png",
        "n_events_hist.png",
        "duration_vs_events.png",
        "top_sessions_by_events.csv",
        "client_session_counts.csv",
        "client_session_counts.png",
    ]:
        print(f" - {Path(args.out_dir) / name}")


if __name__ == "__main__":
    main()