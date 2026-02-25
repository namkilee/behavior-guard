#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trackc_dataset import TrackCParquetDataset, TrackCDatasetConfig, get_vocab_size


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x = self.pos(x)

        # causal mask: prevent attending to future tokens
        T = input_ids.size(1)
        causal = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        # key padding mask: True for PAD
        key_pad = (attention_mask == 0)

        x = self.enc(x, mask=causal, src_key_padding_mask=key_pad)
        x = self.ln(x)
        return self.head(x)


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size(0), -1)
    loss = (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp_min(1.0)
    return loss


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 3
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_len: int = 2048
    num_workers: int = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackc-root", default="out/track_c")
    ap.add_argument("--client", required=True)
    ap.add_argument("--version", required=True, choices=["v0", "v1", "v2"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    cfg = TrainCfg(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = get_vocab_size(args.trackc_root, args.client, args.version)
    model = SmallTransformerLM(vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout, max_len=cfg.max_len).to(device)

    train_ds = TrackCParquetDataset(TrackCDatasetConfig(args.trackc_root, args.client, args.version, "train"))
    val_ds   = TrackCParquetDataset(TrackCDatasetConfig(args.trackc_root, args.client, args.version, "val"))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    run_dir = os.path.join(args.trackc_root, args.client, "runs", "c2_small_tx_lm", f"{args.version}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    def evaluate() -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn = batch["attention_mask"].to(device, non_blocking=True)
                logits = model(input_ids, attn)
                loss = causal_lm_loss(logits, input_ids, attn, pad_id=0)
                total += float(loss.item())
                n += 1
        model.train()
        return total / max(n, 1)

    best = 1e9
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            logits = model(input_ids, attn)
            loss = causal_lm_loss(logits, input_ids, attn, pad_id=0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 100 == 0:
                print(f"[C2][epoch {epoch}] step {step} loss={loss.item():.4f}")

        val_loss = evaluate()
        print(f"[C2][epoch {epoch}] val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    print(f"[DONE] best_val_loss={best:.4f} saved to {run_dir}/best.pt")


if __name__ == "__main__":
    main()