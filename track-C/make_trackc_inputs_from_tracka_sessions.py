#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track C preprocessing from Track A session-level parquet.

Creates three canonical inputs:
- v0: BOS + tokens + EOS
- v1: BOS + per-event [TOK:<token>, OUT:<out>, DT:<dt>, GAP:<gap>, EOV] + EOS
- v2: v1 + RG:<route_group> + OP:<op_from_token> + OUT2:<out_from_token> + DT2:<dt_from_token> (if parsable)

Also:
- deterministic split(train/val/test) by hash(client|user|session_key)
- tail-truncate to max_seq_len, pad to max_seq_len, attention_mask
- outputs client-partitioned parquet files

Assumptions:
- arrays are aligned, but we enforce safety.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


RESERVED_TOKENS = [
    "PAD",
    "UNK",
    "BOS",
    "EOS",
    "EOV",
    "OUT:<NA>",
    "DT:<NA>",
    "RG:<NA>",
    "GAP:<BOS>",
]


TOKEN_PARSE_RE = re.compile(
    r"(?:^|[|])client=(?P<client>[^|]+)"
    r"[|]op=(?P<op>[^|]+)"
    r"[|]out=(?P<out>[^|]+)"
    r"[|]dt=(?P<dt>[^|]+)(?:$|[|])"
)


def stable_hash_0_99(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 100


def assign_split(client: str, user_id: str, session_key: str) -> str:
    h = stable_hash_0_99(f"{client}|{user_id}|{session_key}")
    if h < 90:
        return "train"
    if h < 95:
        return "val"
    return "test"


def to_dt_seconds(x: Any) -> Optional[float]:
    # event_times could be datetime, ISO string, or int (epoch ms/sec)
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.timestamp()
    if isinstance(x, (int, float)):
        # heuristic: if large, treat as ms
        if x > 10_000_000_000:  # ~2286-11-20 in seconds
            return float(x) / 1000.0
        return float(x)
    if isinstance(x, str):
        xs = x.strip()
        if not xs:
            return None
        # try iso
        try:
            # handles "2026-01-06 02:55:43" and iso variants
            return datetime.fromisoformat(xs.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


def bucket_gap_seconds(gap_s: Optional[float]) -> str:
    if gap_s is None:
        return "<NA>"
    if gap_s < 0.2:
        return "<0.2s"
    if gap_s < 1:
        return "0.2-1s"
    if gap_s < 5:
        return "1-5s"
    if gap_s < 30:
        return "5-30s"
    if gap_s < 120:
        return "30-120s"
    if gap_s < 600:
        return "2-10m"
    return ">10m"


def parse_token_fields(token: str) -> Dict[str, str]:
    m = TOKEN_PARSE_RE.search(token or "")
    if not m:
        return {"op": "<NA>", "out": "<NA>", "dt": "<NA>"}
    d = m.groupdict()
    return {
        "op": (d.get("op") or "<NA>").strip(),
        "out": (d.get("out") or "<NA>").strip(),
        "dt": (d.get("dt") or "<NA>").strip(),
    }


@dataclass(frozen=True)
class BuildConfig:
    max_seq_len: int = 2048
    pad_token: str = "PAD"


def tail_truncate(tokens: List[str], max_len: int) -> List[str]:
    if len(tokens) <= max_len:
        return tokens
    return tokens[-max_len:]


def pad_to(tokens: List[str], max_len: int, pad_token: str) -> Tuple[List[str], List[int]]:
    # attention_mask: 1 for real token, 0 for PAD
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    mask = [1] * len(tokens)
    if len(tokens) < max_len:
        n_pad = max_len - len(tokens)
        tokens = tokens + [pad_token] * n_pad
        mask = mask + [0] * n_pad
    return tokens, mask


def align_lists(
    tokens: List[str],
    outcomes: Optional[List[str]],
    route_groups: Optional[List[str]],
    dt_buckets: Optional[List[str]],
    event_times: Optional[List[Any]],
) -> Tuple[List[str], List[str], List[str], List[str], List[Any]]:
    n = len(tokens)
    def fix(lst: Optional[List[Any]], fill: Any) -> List[Any]:
        if lst is None:
            return [fill] * n
        if len(lst) == n:
            return lst
        if len(lst) > n:
            return lst[:n]
        return lst + [fill] * (n - len(lst))

    return (
        tokens,
        [str(x) if x is not None and str(x).strip() else "<NA>" for x in fix(outcomes, "<NA>")],
        [str(x) if x is not None and str(x).strip() else "<NA>" for x in fix(route_groups, "<NA>")],
        [str(x) if x is not None and str(x).strip() else "<NA>" for x in fix(dt_buckets, "<NA>")],
        fix(event_times, None),
    )


def build_v0(cfg: BuildConfig, toks: List[str]) -> List[str]:
    return ["BOS"] + toks + ["EOS"]


def build_v1(
    cfg: BuildConfig,
    toks: List[str],
    outcomes: List[str],
    dt_buckets: List[str],
    event_times: List[Any],
) -> List[str]:
    seq: List[str] = ["BOS"]
    prev_ts: Optional[float] = None

    for i, tok in enumerate(toks):
        ts = to_dt_seconds(event_times[i]) if i < len(event_times) else None
        gap_tok = "GAP:<BOS>" if prev_ts is None else f"GAP:{bucket_gap_seconds((ts - prev_ts) if (ts is not None and prev_ts is not None) else None)}"
        out_tok = f"OUT:{outcomes[i] if i < len(outcomes) else '<NA>'}"
        dt_tok = f"DT:{dt_buckets[i] if i < len(dt_buckets) else '<NA>'}"

        seq.extend([
            f"TOK:{tok}",
            out_tok if out_tok != "OUT:" else "OUT:<NA>",
            dt_tok if dt_tok != "DT:" else "DT:<NA>",
            gap_tok,
            "EOV",
        ])
        prev_ts = ts if ts is not None else prev_ts

    seq.append("EOS")
    return seq


def build_v2(
    cfg: BuildConfig,
    toks: List[str],
    outcomes: List[str],
    route_groups: List[str],
    dt_buckets: List[str],
    event_times: List[Any],
) -> List[str]:
    seq: List[str] = ["BOS"]
    prev_ts: Optional[float] = None

    for i, tok in enumerate(toks):
        parsed = parse_token_fields(tok)
        ts = to_dt_seconds(event_times[i]) if i < len(event_times) else None
        gap_tok = "GAP:<BOS>" if prev_ts is None else f"GAP:{bucket_gap_seconds((ts - prev_ts) if (ts is not None and prev_ts is not None) else None)}"

        out1 = outcomes[i] if i < len(outcomes) else "<NA>"
        rg1 = route_groups[i] if i < len(route_groups) else "<NA>"
        dt1 = dt_buckets[i] if i < len(dt_buckets) else "<NA>"

        seq.extend([
            f"RG:{rg1}",
            f"OUT:{out1}",
            f"DT:{dt1}",
            gap_tok,
            f"TOK:{tok}",
            f"OP:{parsed['op']}",
            f"OUT2:{parsed['out']}",
            f"DT2:{parsed['dt']}",
            "EOV",
        ])
        prev_ts = ts if ts is not None else prev_ts

    seq.append("EOS")
    return seq


def process_file(in_path: str) -> pl.DataFrame:
    df = pl.read_parquet(in_path)

    required = ["user_id", "client_name", "session_key", "tokens"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_path}. Found: {df.columns}")

    # Ensure list types
    # polars may store list columns; if not, we still try.
    return df


def make_outputs_for_client(
    df_client: pl.DataFrame,
    out_dir: str,
    cfg: BuildConfig,
    versions: List[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # We'll build row-wise using pl.map_elements for performance and simplicity.
    def build_row(row: Dict[str, Any]) -> Dict[str, Any]:
        client = str(row.get("client_name") or "")
        user_id = str(row.get("user_id") or "")
        session_key = str(row.get("session_key") or "")
        split = assign_split(client, user_id, session_key)

        toks = row.get("tokens") or []
        outcomes = row.get("outcomes")
        route_groups = row.get("route_groups")
        dt_buckets = row.get("dt_buckets")
        event_times = row.get("event_times")

        toks, outcomes2, route_groups2, dt_buckets2, event_times2 = align_lists(
            list(toks),
            list(outcomes) if outcomes is not None else None,
            list(route_groups) if route_groups is not None else None,
            list(dt_buckets) if dt_buckets is not None else None,
            list(event_times) if event_times is not None else None,
        )

        out: Dict[str, Any] = {
            "client_name": client,
            "user_id": user_id,
            "session_key": session_key,
            "day": row.get("day"),
            "session_start": row.get("session_start"),
            "session_end": row.get("session_end"),
            "n_events": int(row.get("n_events") or len(toks)),
            "split": split,
        }

        for v in versions:
            if v == "v0":
                seq = build_v0(cfg, toks)
            elif v == "v1":
                seq = build_v1(cfg, toks, outcomes2, dt_buckets2, event_times2)
            elif v == "v2":
                seq = build_v2(cfg, toks, outcomes2, route_groups2, dt_buckets2, event_times2)
            else:
                raise ValueError(v)

            seq = tail_truncate(seq, cfg.max_seq_len)
            seq_len = len(seq)
            seq_padded, attn = pad_to(seq, cfg.max_seq_len, cfg.pad_token)

            out[f"tokens_{v}"] = seq_padded
            out[f"attention_mask_{v}"] = attn
            out[f"seq_len_{v}"] = seq_len

        return out

    # Convert to python dicts row-wise
    rows = df_client.to_dicts()
    out_rows = [build_row(r) for r in rows]
    out_df = pl.DataFrame(out_rows)

    # Write per version parquet separately (cleaner downstream)
    for v in versions:
        cols = [
            "client_name", "day", "user_id", "session_key",
            "session_start", "session_end", "n_events", "split",
            f"tokens_{v}", f"attention_mask_{v}", f"seq_len_{v}",
        ]
        # Some inputs may not have day/session_start/session_end; keep if present
        cols = [c for c in cols if c in out_df.columns]
        out_path = os.path.join(out_dir, f"sessions_tokens_{v}.parquet")
        out_df.select(cols).write_parquet(out_path, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_glob", required=True, help="Input parquet path or glob. e.g. out/common/sessions_tracka_*.parquet")
    ap.add_argument("--out", dest="out_root", default="out/track_c", help="Output root dir.")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--versions", default="v0,v1,v2", help="Comma separated: v0,v1,v2")
    ap.add_argument("--clients", default="", help="Optional comma-separated client_name allowlist")
    args = ap.parse_args()

    in_paths = sorted(glob.glob(args.in_glob)) if any(ch in args.in_glob for ch in "*?[]") else [args.in_glob]
    if not in_paths:
        raise SystemExit(f"No input files matched: {args.in_glob}")

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    cfg = BuildConfig(max_seq_len=args.max_seq_len)

    os.makedirs(args.out_root, exist_ok=True)
    spec_dir = os.path.join(args.out_root, "_spec")
    os.makedirs(spec_dir, exist_ok=True)
    with open(os.path.join(spec_dir, "reserved_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(RESERVED_TOKENS, f, ensure_ascii=False, indent=2)

    allow_clients = set([c.strip() for c in args.clients.split(",") if c.strip()]) if args.clients else set()

    # Read all, concat
    dfs = [process_file(p) for p in in_paths]
    df_all = pl.concat(dfs, how="diagonal_relaxed")

    if allow_clients:
        df_all = df_all.filter(pl.col("client_name").is_in(list(allow_clients)))

    # Partition by client and write
    clients = df_all.select(pl.col("client_name").unique()).to_series().to_list()
    for client in clients:
        df_client = df_all.filter(pl.col("client_name") == client)
        out_dir = os.path.join(args.out_root, str(client), "tokens")
        make_outputs_for_client(df_client, out_dir, cfg, versions)

    print(f"[OK] wrote Track C token datasets under: {args.out_root}")


if __name__ == "__main__":
    main()