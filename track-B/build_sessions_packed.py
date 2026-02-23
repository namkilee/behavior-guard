#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from track_b.common_builder import build_sessions_packed_file


def main() -> int:
    ap = argparse.ArgumentParser(description="Build canonical sessions_packed_{day}.parquet for Track A/B")
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--in", dest="in_path", default=None, help="Input raw parquet (default: out/sessions_raw_{day}.parquet)")
    ap.add_argument("--outdir", default="out/common", help="Output directory (default: out/common)")
    ap.add_argument("--overwrite", type=int, default=0, help="1 to overwrite if exists")

    args = ap.parse_args()

    day = args.day
    in_path = Path(args.in_path or f"out/sessions_raw_{day}.parquet")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"sessions_packed_{day}.parquet"
    manifest_path = outdir / f"schema_manifest_{day}.json"

    build_sessions_packed_file(
        day=day,
        in_path=in_path,
        out_path=out_path,
        manifest_path=manifest_path,
        overwrite=bool(args.overwrite),
    )

    print(f"[OK] Built: {out_path}")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())