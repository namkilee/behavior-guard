#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import polars as pl


def load_vocab(vocab_json: str) -> Dict[str, int]:
    with open(vocab_json, "r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], unk_id: int) -> List[int]:
    return [vocab.get(t, unk_id) for t in tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackc-root", default="out/track_c")
    ap.add_argument("--client", required=True)
    ap.add_argument("--version", required=True, choices=["v0", "v1", "v2"])
    args = ap.parse_args()

    token_path = os.path.join(args.trackc_root, args.client, "tokens", f"sessions_tokens_{args.version}.parquet")
    vocab_path = os.path.join(args.trackc_root, args.client, "vocab", args.version, "vocab.json")
    out_dir = os.path.join(args.trackc_root, args.client, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sessions_ids_{args.version}.parquet")

    df = pl.read_parquet(token_path)
    vocab = load_vocab(vocab_path)
    unk_id = vocab.get("UNK", 1)

    tok_col = f"tokens_{args.version}"
    attn_col = f"attention_mask_{args.version}"
    len_col = f"seq_len_{args.version}"

    # Map tokens -> ids
    df2 = df.with_columns(
        pl.col(tok_col).map_elements(lambda xs: tokens_to_ids(xs, vocab, unk_id)).alias(f"token_ids_{args.version}")
    )

    keep = [
        "client_name", "day", "user_id", "session_key",
        "session_start", "session_end", "n_events", "split",
        f"token_ids_{args.version}", attn_col, len_col,
    ]
    keep = [c for c in keep if c in df2.columns]
    df2.select(keep).write_parquet(out_path, compression="zstd")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()