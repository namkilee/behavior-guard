#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import polars as pl


def load_reserved_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab_from_parquet(
    parquet_path: str,
    token_col: str,
    reserved: List[str],
    min_freq: int,
    max_vocab: int,
) -> Tuple[Dict[str, int], List[str], Dict]:
    df = pl.read_parquet(parquet_path, columns=[token_col])
    # explode list -> count
    s = df.select(pl.col(token_col).explode()).to_series()
    # Polars may include None
    tokens = [t for t in s.to_list() if t is not None]

    ctr = Counter(tokens)

    # Remove reserved from counting (we will add them first)
    for rt in reserved:
        ctr.pop(rt, None)

    # Filter by min_freq
    items = [(tok, freq) for tok, freq in ctr.items() if freq >= min_freq]
    # Deterministic sort: freq desc, token asc
    items.sort(key=lambda x: (-x[1], x[0]))

    # Cap size: reserved + others
    keep = items[: max(0, max_vocab - len(reserved))]

    id2token: List[str] = []
    vocab: Dict[str, int] = {}

    for t in reserved:
        vocab[t] = len(id2token)
        id2token.append(t)

    for tok, _ in keep:
        if tok in vocab:
            continue
        vocab[tok] = len(id2token)
        id2token.append(tok)

    # Coverage (approx): fraction of token occurrences covered by vocab
    total = sum(ctr.values()) + sum(Counter(tokens)[rt] for rt in reserved if rt in Counter(tokens))
    covered = 0
    for t, f in Counter(tokens).items():
        if t in vocab:
            covered += f
    coverage = float(covered) / float(total) if total else 1.0

    stats = {
        "token_col": token_col,
        "min_freq": min_freq,
        "max_vocab": max_vocab,
        "reserved_size": len(reserved),
        "vocab_size": len(vocab),
        "coverage": coverage,
    }
    return vocab, id2token, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackc-root", default="out/track_c", help="Track C root dir.")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-vocab", type=int, default=50000)
    ap.add_argument("--versions", default="v0,v1,v2")
    ap.add_argument("--clients", default="", help="Optional comma-separated allowlist of client_name")
    args = ap.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    allow_clients = set([c.strip() for c in args.clients.split(",") if c.strip()]) if args.clients else set()

    reserved_path = os.path.join(args.trackc_root, "_spec", "reserved_tokens.json")
    reserved = load_reserved_tokens(reserved_path)

    # Discover clients
    clients = []
    for name in os.listdir(args.trackc_root):
        if name.startswith("_"):
            continue
        p = os.path.join(args.trackc_root, name)
        if os.path.isdir(p):
            clients.append(name)

    if allow_clients:
        clients = [c for c in clients if c in allow_clients]

    for client in clients:
        token_dir = os.path.join(args.trackc_root, client, "tokens")
        vocab_root = os.path.join(args.trackc_root, client, "vocab")
        os.makedirs(vocab_root, exist_ok=True)

        for v in versions:
            parquet_path = os.path.join(token_dir, f"sessions_tokens_{v}.parquet")
            if not os.path.exists(parquet_path):
                print(f"[SKIP] missing {parquet_path}")
                continue

            token_col = f"tokens_{v}"
            vocab, id2token, stats = build_vocab_from_parquet(
                parquet_path=parquet_path,
                token_col=token_col,
                reserved=reserved,
                min_freq=args.min_freq,
                max_vocab=args.max_vocab,
            )

            out_dir = os.path.join(vocab_root, v)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
            with open(os.path.join(out_dir, "id2token.json"), "w", encoding="utf-8") as f:
                json.dump(id2token, f, ensure_ascii=False)
            with open(os.path.join(out_dir, "vocab_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            print(f"[OK] client={client} version={v} vocab_size={stats['vocab_size']} coverage={stats['coverage']:.4f}")

    print("[DONE]")


if __name__ == "__main__":
    main()