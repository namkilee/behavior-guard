#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trackc_dataset import TrackCParquetDataset, TrackCDatasetConfig, get_vocab_size


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
        logits = self.head(x)
        return logits


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Next-token prediction:
    - predict input_ids[:, 1:] from logits[:, :-1]
    - mask out PAD positions (and also the first token has no target)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size(0), -1)

    # also mask by attention_mask
    loss = (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp_min(1.0)
    return loss


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 3
    d_model: int = 256
    n_layers: int = 2
    dropout: float = 0.1
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
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    cfg = TrainCfg(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = get_vocab_size(args.trackc_root, args.client, args.version)
    model = LSTMLM(vocab_size, cfg.d_model, cfg.n_layers, cfg.dropout).to(device)

    train_ds = TrackCParquetDataset(TrackCDatasetConfig(args.trackc_root, args.client, args.version, "train"))
    val_ds   = TrackCParquetDataset(TrackCDatasetConfig(args.trackc_root, args.client, args.version, "val"))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    run_dir = os.path.join(args.trackc_root, args.client, "runs", "c1_lstm_lm", f"{args.version}_{int(time.time())}")
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
                print(f"[C1][epoch {epoch}] step {step} loss={loss.item():.4f}")

        val_loss = evaluate()
        print(f"[C1][epoch {epoch}] val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    print(f"[DONE] best_val_loss={best:.4f} saved to {run_dir}/best.pt")


if __name__ == "__main__":
    main()