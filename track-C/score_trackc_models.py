#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Literal, Dict, Any, List

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trackc_dataset import TrackCParquetDataset, TrackCDatasetConfig, get_vocab_size


# ----------------------------
# Models (must match training)
# ----------------------------

class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x, _ = self.lstm(x)
        x = self.ln(x)
        return self.head(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SmallTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float, max_len: int, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x = self.pos(x)

        T = input_ids.size(1)
        causal = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        key_pad = (attention_mask == 0)  # True for PAD

        x = self.enc(x, mask=causal, src_key_padding_mask=key_pad)
        x = self.ln(x)
        return self.head(x)


class SeqAutoEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.enc = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )
        self.enc_proj = nn.Linear(2 * d_model, d_model)
        self.dec = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        enc_out, _ = self.enc(x)
        _ = self.enc_proj(enc_out)  # reserved for later conditioning; keep structure stable
        dec_out, _ = self.dec(x)
        h = self.ln(dec_out)
        return self.head(h)


# ----------------------------
# Per-session scoring
# ----------------------------

def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # returns [B, T] cross-entropy per position (no reduction)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    B, T, V = logits.shape

    # reshape() is safe for non-contiguous tensors (will copy if needed)
    loss = loss_fct(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
    ).reshape(B, T)

    return loss


def score_lm_batch(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int = 0, tail_k: int = 128
) -> Dict[str, torch.Tensor]:
    # Next token: predict ids[:,1:] from logits[:,:-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]  # [B, T-1]

    ce = per_token_ce(shift_logits, shift_labels)  # [B, T-1]
    ce = ce * shift_mask.float()

    denom = shift_mask.float().sum(dim=1).clamp_min(1.0)
    mean_nll = ce.sum(dim=1) / denom

    # tail mean (last tail_k tokens among valid)
    # simple approach: take last tail_k positions of sequence (already padded), mask handles validity.
    ce_tail = ce[:, -tail_k:] if ce.size(1) >= tail_k else ce
    mask_tail = shift_mask[:, -tail_k:] if shift_mask.size(1) >= tail_k else shift_mask
    denom_tail = mask_tail.float().sum(dim=1).clamp_min(1.0)
    tail_nll = ce_tail.sum(dim=1) / denom_tail

    # max token nll (useful for spikes)
    ce_masked = ce.masked_fill(shift_mask == 0, float("-inf"))
    max_nll = ce_masked.max(dim=1).values
    max_nll = torch.where(torch.isfinite(max_nll), max_nll, torch.zeros_like(max_nll))  # all-pad safety
    return {"score_mean_nll": mean_nll, "score_tail_nll": tail_nll, "score_max_nll": max_nll}


def score_ae_batch(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int = 0, tail_k: int = 128
) -> Dict[str, torch.Tensor]:
    ce = per_token_ce(logits, input_ids)  # [B,T]
    ce = ce * attention_mask.float()

    denom = attention_mask.float().sum(dim=1).clamp_min(1.0)
    mean_ce = ce.sum(dim=1) / denom

    ce_tail = ce[:, -tail_k:] if ce.size(1) >= tail_k else ce
    mask_tail = attention_mask[:, -tail_k:] if attention_mask.size(1) >= tail_k else attention_mask
    denom_tail = mask_tail.float().sum(dim=1).clamp_min(1.0)
    tail_ce = ce_tail.sum(dim=1) / denom_tail

    max_ce = (ce + (1.0 - attention_mask.float()) * (-1e9)).max(dim=1).values
    return {"score_mean_ce": mean_ce, "score_tail_ce": tail_ce, "score_max_ce": max_ce}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackc-root", default="out/track_c")
    ap.add_argument("--client", required=True)
    ap.add_argument("--version", required=True, choices=["v0", "v1", "v2"])
    ap.add_argument("--model", required=True, choices=["c1", "c2", "c3"])
    ap.add_argument("--ckpt", required=True, help="Path to best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tail-k", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-heads", type=int, default=8)        # for c2
    ap.add_argument("--max-len", type=int, default=2048)      # for c2 pos-enc
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = get_vocab_size(args.trackc_root, args.client, args.version)

    # dataset
    ds = TrackCParquetDataset(TrackCDatasetConfig(args.trackc_root, args.client, args.version, args.split))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model init
    if args.model == "c1":
        model = LSTMLM(vocab_size, args.d_model, args.n_layers, args.dropout).to(device)
        scorer = "lm"
    elif args.model == "c2":
        model = SmallTransformerLM(vocab_size, args.d_model, args.n_heads, max(args.n_layers, 1), args.dropout, max_len=args.max_len).to(device)
        scorer = "lm"
    else:
        model = SeqAutoEncoder(vocab_size, args.d_model, args.n_layers, args.dropout).to(device)
        scorer = "ae"

    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # We need join keys: load from the parquet that includes split + keys
    base_path = os.path.join(args.trackc_root, args.client, "dataset", f"sessions_ids_{args.version}.parquet")
    base_df = pl.read_parquet(base_path).filter(pl.col("split") == args.split)

    # Score in the same order as dataset rows.
    # TrackCParquetDataset loads rows in parquet order after filtering; we mirrored that order in base_df.
    scores: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            logits = model(input_ids, attn)
            if scorer == "lm":
                out = score_lm_batch(logits, input_ids, attn, pad_id=0, tail_k=args.tail_k)
            else:
                out = score_ae_batch(logits, input_ids, attn, pad_id=0, tail_k=args.tail_k)

            for k, v in out.items():
                scores.setdefault(k, []).extend(v.detach().cpu().tolist())

    # Attach scores
    out_df = base_df.select([c for c in ["client_name", "day", "user_id", "session_key", "split"] if c in base_df.columns]).with_columns(
        [pl.Series(name=k, values=v) for k, v in scores.items()]
    )

    out_dir = os.path.join(args.trackc_root, args.client, "scores")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}_{args.version}_{args.split}_scores.parquet")
    out_df.write_parquet(out_path, compression="zstd")

    print(f"[OK] wrote scores: {out_path}")
    print("Columns:", out_df.columns)


if __name__ == "__main__":
    main()