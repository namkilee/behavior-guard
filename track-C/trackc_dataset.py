#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrackCDatasetConfig:
    trackc_root: str
    client: str
    version: str  # v0|v1|v2
    split: str    # train|val|test


class TrackCParquetDataset(Dataset):
    """
    Loads session-level ids parquet created by make_ids.py:
    - token_ids_{v}: List[int] length=max_seq_len
    - attention_mask_{v}: List[int] length=max_seq_len
    - seq_len_{v}: int (unpadded length)
    """

    def __init__(self, cfg: TrackCDatasetConfig):
        super().__init__()
        path = os.path.join(cfg.trackc_root, cfg.client, "dataset", f"sessions_ids_{cfg.version}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset parquet: {path}\n"
                f"Run: python make_ids.py --trackc-root {cfg.trackc_root} --client {cfg.client!r} --version {cfg.version}"
            )

        v = cfg.version
        tok_col = f"token_ids_{v}"
        attn_col = f"attention_mask_{v}"
        len_col = f"seq_len_{v}"

        df = pl.read_parquet(path)

        if "split" not in df.columns:
            raise ValueError("Dataset parquet must include 'split' column (train/val/test).")

        df = df.filter(pl.col("split") == cfg.split)

        # Keep only required columns for speed
        keep = [tok_col, attn_col, len_col]
        df = df.select([c for c in keep if c in df.columns])

        self._tok = df[tok_col].to_list()
        self._attn = df[attn_col].to_list()
        self._len = df[len_col].to_list()

        if not (len(self._tok) == len(self._attn) == len(self._len)):
            raise ValueError("Mismatched dataset lengths after loading parquet.")

    def __len__(self) -> int:
        return len(self._tok)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self._tok[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self._attn[idx], dtype=torch.long),
            "seq_len": torch.tensor(self._len[idx], dtype=torch.long),
        }


def get_vocab_size(trackc_root: str, client: str, version: str) -> int:
    import json
    vocab_path = os.path.join(trackc_root, client, "vocab", version, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Missing vocab.json: {vocab_path}\n"
            f"Run: python build_vocab.py --trackc-root {trackc_root} --versions {version}"
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)