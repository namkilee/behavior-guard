#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BehaviorGuard Track B (Research Mode) - v0.1 scaffolding
- B1: First-order Markov NLL (add-1 smoothing)
- B2: N-gram rarity (bigram + trigram)
- B3: Entropy deviation (|H_session - H_med|)

Input (default):
  ./artifacts/sessions_packed.parquet

Output (default):
  ./artifacts/track_b/

Join key (must be preserved end-to-end):
  (project_id, day, user_id_norm, session_id_norm)

Token definition (Frozen v0.1):
  token = "{route_group}:{normalized_outcome}"

Ordering (Frozen v0.1):
  event_time ASC, observation_id ASC
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import pandas as pd

from track_b.preprocess import build_event_frame
from track_b.models import (
    score_b1_markov_nll,
    score_b2_ngram_rarity,
    score_b3_entropy_deviation,
)
from track_b.artifacts import write_summary_and_drilldown
from track_b.metadata import build_run_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BehaviorGuard Track B (Research)")

    p.add_argument("--input", default="./artifacts/sessions_packed.parquet", help="Input parquet path")
    p.add_argument("--outdir", default="./artifacts/track_b", help="Output directory")

    # Model selection
    p.add_argument("--model", choices=["b1", "b2", "b3", "all"], default="b1", help="Which model to run")

    # Top-K
    p.add_argument("--topk", type=int, default=200, help="Top-K sessions to emit")

    # Partitioning (Frozen: (project_id, day))
    p.add_argument("--partition-by", default="project_id,day", help="Comma-separated partition columns")

    # Column hints (robustness)
    p.add_argument("--col-project", default="project_id")
    p.add_argument("--col-user", default="user_id_norm")
    p.add_argument("--col-session", default="session_id_norm")
    p.add_argument("--col-event-time", default="event_time")
    p.add_argument("--col-observation-id", default="observation_id")
    p.add_argument("--col-route-group", default="route_group")
    p.add_argument("--col-outcome", default="normalized_outcome")

    # If packed schema differs, allow specifying source fields for building route/outcome
    p.add_argument("--src-route", default="route_group", help="Source field used as route_group if route_group missing")
    p.add_argument("--src-outcome", default="outcome", help="Source field used to normalize outcome if normalized_outcome missing")

    # Explode fields: comma-separated list columns if known (optional)
    p.add_argument("--explode-cols", default="", help="Comma-separated list columns to explode (optional)")

    # Output format
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output artifact format")

    return p.parse_args()


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    if not in_path.exists():
        print(f"[FATAL] input not found: {in_path}", file=sys.stderr)
        return 2

    # Read input (sessions_packed)
    df_in = pd.read_parquet(in_path)

    # Build exploded + tokenized event-level frame
    explode_cols = [c.strip() for c in args.explode_cols.split(",") if c.strip()] or None
    events = build_event_frame(
        df_in=df_in,
        col_project=args.col_project,
        col_user=args.col_user,
        col_session=args.col_session,
        col_event_time=args.col_event_time,
        col_observation_id=args.col_observation_id,
        col_route_group=args.col_route_group,
        col_outcome=args.col_outcome,
        src_route=args.src_route,
        src_outcome=args.src_outcome,
        explode_cols=explode_cols,
    )

    # Run metadata
    run_meta = build_run_metadata(
        track="B",
        track_version="0.1",
        input_path=str(in_path),
        outdir=str(outdir),
        args=vars(args),
        n_rows_input=int(len(df_in)),
        n_rows_events=int(len(events)),
    )
    (outdir / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    partition_cols = [c.strip() for c in args.partition_by.split(",") if c.strip()]
    topk = int(args.topk)

    # Run selected model(s)
    model_runs: list[tuple[str, pd.DataFrame]] = []
    if args.model in ("b1", "all"):
        scores_b1 = score_b1_markov_nll(events, partition_cols=partition_cols)
        model_runs.append(("b1", scores_b1))

    if args.model in ("b2", "all"):
        scores_b2 = score_b2_ngram_rarity(events, partition_cols=partition_cols, n_values=(2, 3))
        model_runs.append(("b2", scores_b2))

    if args.model in ("b3", "all"):
        scores_b3 = score_b3_entropy_deviation(events, partition_cols=partition_cols)
        model_runs.append(("b3", scores_b3))

    # Emit artifacts per model
    for model_name, scores in model_runs:
        model_dir = outdir / model_name
        _ensure_outdir(model_dir)

        write_summary_and_drilldown(
            model_name=model_name,
            events=events,
            session_scores=scores,
            outdir=model_dir,
            topk=topk,
            fmt=args.format,
        )

    print(f"[OK] Track B finished. outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())