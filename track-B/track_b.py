#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import pandas as pd

from track_b.common_builder import build_sessions_packed_file
from track_b.preprocess import explode_canonical_packed_to_events
from track_b.models import score_b1_markov_nll, score_b2_ngram_rarity, score_b3_entropy_deviation
from track_b.artifacts import write_summary_and_drilldown
from track_b.metadata import build_run_metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BehaviorGuard Track B (Research)")

    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--raw", default=None, help="Raw parquet (default: out/sessions_raw_{day}.parquet)")
    ap.add_argument("--packed", default=None, help="Packed parquet (default: out/common/sessions_packed_{day}.parquet)")
    ap.add_argument("--outdir", default="out/track_b", help="Output directory (default: out/track_b)")
    ap.add_argument("--common-dir", default="out/common", help="Common artifacts directory (default: out/common)")

    ap.add_argument("--build-packed", type=int, default=1, help="1 to build packed if missing")
    ap.add_argument("--overwrite-packed", type=int, default=0, help="1 to overwrite packed")

    ap.add_argument("--model", choices=["b1", "b2", "b3", "all"], default="b1")
    ap.add_argument("--topk", type=int, default=200)

    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    day = args.day

    raw_path = Path(args.raw or f"out/sessions_raw_{day}.parquet")
    common_dir = Path(args.common_dir)
    packed_path = Path(args.packed or (common_dir / f"sessions_packed_{day}.parquet"))
    manifest_path = common_dir / f"schema_manifest_{day}.json"

    # Ensure packed exists (common input)
    if not packed_path.exists():
        if not args.build_packed:
            print(f"[FATAL] packed not found: {packed_path} (use --build-packed 1)", file=sys.stderr)
            return 2
        build_sessions_packed_file(
            day=day,
            in_path=raw_path,
            out_path=packed_path,
            manifest_path=manifest_path,
            overwrite=bool(args.overwrite_packed),
        )

    # Read packed and explode to event-level for sequence scoring
    packed = pd.read_parquet(packed_path)
    events = explode_canonical_packed_to_events(packed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = build_run_metadata(
        track="B",
        track_version="0.1.1",
        input_path=str(packed_path),
        outdir=str(outdir),
        args=vars(args),
        n_rows_input=int(len(packed)),
        n_rows_events=int(len(events)),
    )
    (outdir / f"run_metadata_{day}.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    topk = int(args.topk)

    model_runs: list[tuple[str, pd.DataFrame]] = []
    if args.model in ("b1", "all"):
        model_runs.append(("b1", score_b1_markov_nll(events, partition_cols=["client_name", "day"])))
    if args.model in ("b2", "all"):
        model_runs.append(("b2", score_b2_ngram_rarity(events, partition_cols=["client_name", "day"], n_values=(2, 3))))
    if args.model in ("b3", "all"):
        model_runs.append(("b3", score_b3_entropy_deviation(events, partition_cols=["client_name", "day"])))

    # Emit artifacts
    for model_name, scores in model_runs:
        model_dir = outdir / day / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        write_summary_and_drilldown(
            model_name=model_name,
            events=events,
            session_scores=scores,
            outdir=model_dir,
            topk=topk,
            fmt=args.format,
        )

    print(f"[OK] Track B done. outdir={outdir}/{day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())