#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge daily parquet files into one parquet.")
    p.add_argument("--input-dir", required=True, help="Directory containing daily parquet files")
    p.add_argument("--start-day", required=True, help="Start day (YYYY-MM-DD)")
    p.add_argument("--end-day", required=True, help="End day (YYYY-MM-DD)")
    p.add_argument("--prefix", required=True, help="File prefix, e.g. session_features_ or sessions_raw_")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any daily parquet is missing within the requested date range",
    )
    return p.parse_args()


def daterange(start_day: str, end_day: str):
    start = datetime.strptime(start_day, "%Y-%m-%d").date()
    end = datetime.strptime(end_day, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("start_day must be <= end_day")

    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"[FATAL] input-dir not found: {input_dir}", file=sys.stderr)
        return 1

    files: list[Path] = []
    missing: list[Path] = []

    for day in daterange(args.start_day, args.end_day):
        fp = input_dir / f"{args.prefix}{day}.parquet"
        if fp.exists():
            files.append(fp)
        else:
            missing.append(fp)

    if missing:
        print("[WARN] Missing files:")
        for fp in missing:
            print(f"[WARN] - {fp}")

        if args.strict_missing:
            print("[FATAL] strict-missing enabled and some files are missing.", file=sys.stderr)
            return 1

    if not files:
        print("[FATAL] No parquet files found to merge.", file=sys.stderr)
        return 1

    print("[INFO] Files to merge:")
    for fp in files:
        print(f"[INFO] - {fp}")

    dfs = [pl.read_parquet(fp) for fp in files]
    merged = pl.concat(dfs, how="diagonal_relaxed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output_path)

    print(f"[INFO] Wrote merged parquet: {output_path}")
    print(f"[INFO] Rows: {merged.height}, Cols: {merged.width}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())