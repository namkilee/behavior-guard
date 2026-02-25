#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple, Dict

import polars as pl


DEFAULT_POS_OUTCOMES = {"error", "rate_limited", "blocked"}


def load_tracka_sessions(in_glob: str) -> pl.DataFrame:
    paths = sorted(glob.glob(in_glob)) if any(ch in in_glob for ch in "*?[]") else [in_glob]
    if not paths:
        raise SystemExit(f"No input files matched: {in_glob}")
    dfs = [pl.read_parquet(p) for p in paths]
    df = pl.concat(dfs, how="diagonal_relaxed")
    # need join keys + outcomes
    need = ["client_name", "day", "user_id", "session_key", "outcomes"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"TrackA session parquet missing columns: {missing}. Found: {df.columns}")
    return df.select(need)


def make_weak_label(outcomes: List[str], pos_set: set[str]) -> int:
    if outcomes is None:
        return 0
    for o in outcomes:
        if o is None:
            continue
        s = str(o).strip().lower()
        if s in pos_set:
            return 1
    return 0


def precision_at_k(labels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = labels[:k]
    return sum(top) / float(len(top)) if top else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracka-in", required=True, help='TrackA session parquet glob, e.g. "out/common/sessions_packed_*.parquet"')
    ap.add_argument("--scores", required=True, help="Scores parquet path produced by score_trackc_models.py")
    ap.add_argument("--client", default="", help="Optional client_name filter (recommended)")
    ap.add_argument("--pos-outcomes", default="error,rate_limited,blocked")
    ap.add_argument("--ks", default="50,100,200,500,1000")
    ap.add_argument("--score-cols", default="", help="Comma separated score columns to evaluate. Default: all score_* cols")
    args = ap.parse_args()

    pos_set = set([s.strip().lower() for s in args.pos_outcomes.split(",") if s.strip()])
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    df_a = load_tracka_sessions(args.tracka_in)
    df_s = pl.read_parquet(args.scores)

    if args.client:
        df_a = df_a.filter(pl.col("client_name") == args.client)
        df_s = df_s.filter(pl.col("client_name") == args.client)

    # Join labels onto scored sessions
    df = df_s.join(df_a, on=["client_name", "day", "user_id", "session_key"], how="left")

    # Build weak label
    df = df.with_columns(
        pl.col("outcomes").map_elements(lambda xs: make_weak_label(xs, pos_set)).alias("label")
    )

    # Pick score columns
    if args.score_cols.strip():
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    else:
        score_cols = [c for c in df.columns if c.startswith("score_")]

    if not score_cols:
        raise ValueError("No score columns found. Pass --score-cols explicitly.")

    rows = []
    for sc in score_cols:
        d2 = df.sort(sc, descending=True)
        labels = d2["label"].to_list()
        n_pos = int(sum(labels))
        n_all = len(labels)

        for k in ks:
            p = precision_at_k(labels, min(k, n_all))
            rows.append({
                "score_col": sc,
                "K": k,
                "precision_at_k": p,
                "n_scored": n_all,
                "n_pos_total": n_pos,
                "pos_outcomes": ",".join(sorted(pos_set)),
            })

    out = pl.DataFrame(rows).sort(["score_col", "K"])
    print(out)

    # Optional: write sidecar
    out_path = os.path.splitext(args.scores)[0] + "_precision_at_k.parquet"
    out.write_parquet(out_path, compression="zstd")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()