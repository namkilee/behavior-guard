from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import math
import pandas as pd
import numpy as np


def _scale_to_risk_0_100(raw: pd.Series, *, p50: float, p95: float) -> pd.Series:
    """
    Frozen plan:
      raw -> risk_score_seq (0..100) using p50/p95.
    """
    x = raw.astype(float)

    if not np.isfinite(p50) or not np.isfinite(p95) or p95 <= p50:
        # fallback: min-max
        lo = np.nanmin(x.values) if len(x) else 0.0
        hi = np.nanmax(x.values) if len(x) else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - lo) / (hi - lo) * 100.0).clip(0, 100)

    # <=p50 -> 0, >=p95 -> 100, linear between
    y = (x - p50) / (p95 - p50) * 100.0
    return y.clip(0, 100)


def _session_key_cols(events: pd.DataFrame) -> list[str]:
    # Frozen join key: (project_id, day, user_id_norm, session_id_norm)
    for c in ["project_id", "day", "user_id_norm", "session_id_norm"]:
        if c not in events.columns:
            raise ValueError(f"Missing required join key column: {c}")
    return ["project_id", "day", "user_id_norm", "session_id_norm"]


def _build_session_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with one row per session and a list of tokens in-order.
    Columns: join keys + tokens(list[str]) + n_events
    """
    key = _session_key_cols(events)

    # group tokens in order
    g = events.groupby(key, sort=False)["token"].apply(list).reset_index()
    g["n_events"] = g["token"].map(len).astype(int)
    return g


# -----------------------
# B1: First-order Markov NLL
# -----------------------
def score_b1_markov_nll(events: pd.DataFrame, *, partition_cols: Sequence[str]) -> pd.DataFrame:
    """
    partition-wise train transition table using add-1 smoothing.
    session raw score: average NLL per transition (higher = more anomalous)
    output columns:
      join keys + seq_raw + risk_score_seq + n_events
      plus model hints: n_transitions
    """
    key = _session_key_cols(events)
    sess = _build_session_sequences(events)

    # For partitioned training, operate per (project_id, day) by default
    for c in partition_cols:
        if c not in sess.columns:
            raise ValueError(f"partition col missing in session frame: {c}")

    rows = []

    for pvals, part_df in sess.groupby(list(partition_cols), sort=False):
        # Build transition counts over all sessions in this partition
        # States are tokens
        trans_counts: dict[tuple[str, str], int] = {}
        out_counts: dict[str, int] = {}
        vocab: set[str] = set()

        for toks in part_df["token"].tolist():
            vocab.update(toks)
            if len(toks) < 2:
                continue
            for a, b in zip(toks[:-1], toks[1:]):
                trans_counts[(a, b)] = trans_counts.get((a, b), 0) + 1
                out_counts[a] = out_counts.get(a, 0) + 1

        V = max(len(vocab), 1)

        def trans_prob(a: str, b: str) -> float:
            # add-1 Laplace: (c(a,b)+1) / (c(a,*) + V)
            num = trans_counts.get((a, b), 0) + 1
            den = out_counts.get(a, 0) + V
            return num / den

        # Score each session
        for _, r in part_df.iterrows():
            toks = r["token"]
            n_events = int(r["n_events"])
            n_trans = max(n_events - 1, 0)

            if n_trans <= 0:
                seq_raw = 0.0
            else:
                nll = 0.0
                for a, b in zip(toks[:-1], toks[1:]):
                    p = trans_prob(a, b)
                    nll += -math.log(p)
                seq_raw = nll / n_trans  # avg per transition

            row = {c: r[c] for c in key}
            row.update(
                {
                    "model": "b1",
                    "seq_raw": float(seq_raw),
                    "n_events": n_events,
                    "n_transitions": int(n_trans),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    # scale per-partition using p50/p95 of seq_raw (partition-local scaling)
    out["risk_score_seq"] = 0.0
    for pvals, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    # Ensure direction: Higher = more anomalous (already true)
    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


# -----------------------
# B2: N-gram rarity (bigram + trigram)
# -----------------------
def score_b2_ngram_rarity(
    events: pd.DataFrame, *,
    partition_cols: Sequence[str],
    n_values: Tuple[int, ...] = (2, 3),
) -> pd.DataFrame:
    """
    For each partition, build n-gram counts (n in {2,3} by default).
    Session rarity raw: mean over n-grams of -log( (count+1) / (total_ngrams + V) )
      (add-1 smoothing over n-gram vocab)
    Higher = rarer = more anomalous.
    """
    key = _session_key_cols(events)
    sess = _build_session_sequences(events)

    rows = []

    for pvals, part_df in sess.groupby(list(partition_cols), sort=False):
        # Build n-gram counts
        counts: dict[tuple[str, ...], int] = {}
        total = 0
        vocab: set[tuple[str, ...]] = set()

        for toks in part_df["token"].tolist():
            for n in n_values:
                if len(toks) < n:
                    continue
                for i in range(len(toks) - n + 1):
                    ng = tuple(toks[i : i + n])
                    counts[ng] = counts.get(ng, 0) + 1
                    vocab.add(ng)
                    total += 1

        V = max(len(vocab), 1)
        denom = total + V

        def ng_prob(ng: tuple[str, ...]) -> float:
            return (counts.get(ng, 0) + 1) / denom

        for _, r in part_df.iterrows():
            toks = r["token"]
            n_events = int(r["n_events"])

            # collect all ngrams for this session
            ngrams = []
            for n in n_values:
                if len(toks) < n:
                    continue
                for i in range(len(toks) - n + 1):
                    ngrams.append(tuple(toks[i : i + n]))

            if not ngrams:
                seq_raw = 0.0
                n_ngrams = 0
            else:
                s = 0.0
                for ng in ngrams:
                    p = ng_prob(ng)
                    s += -math.log(p)
                n_ngrams = len(ngrams)
                seq_raw = s / n_ngrams

            row = {c: r[c] for c in key}
            row.update(
                {
                    "model": "b2",
                    "seq_raw": float(seq_raw),
                    "n_events": n_events,
                    "n_ngrams": int(n_ngrams),
                    "ngram_ns": ",".join(map(str, n_values)),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)

    out["risk_score_seq"] = 0.0
    for pvals, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


# -----------------------
# B3: Entropy deviation
# -----------------------
def _entropy_from_tokens(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    s = pd.Series(tokens, dtype="string")
    p = (s.value_counts(normalize=True)).values
    # Shannon entropy (nats)
    return float(-(p * np.log(p)).sum())


def score_b3_entropy_deviation(events: pd.DataFrame, *, partition_cols: Sequence[str]) -> pd.DataFrame:
    """
    For each partition:
      H_session = entropy(token distribution)
      H_med = median(H_session)
      raw = abs(H_session - H_med)
    Higher deviation = more anomalous.
    """
    key = _session_key_cols(events)
    sess = _build_session_sequences(events)

    rows = []
    for pvals, part_df in sess.groupby(list(partition_cols), sort=False):
        ent = part_df["token"].map(_entropy_from_tokens).astype(float)
        h_med = float(np.nanmedian(ent.values)) if len(ent) else 0.0

        for (idx, r), h in zip(part_df.iterrows(), ent.values):
            raw = abs(float(h) - h_med)
            row = {c: r[c] for c in key}
            row.update(
                {
                    "model": "b3",
                    "seq_raw": float(raw),
                    "n_events": int(r["n_events"]),
                    "entropy_session": float(h),
                    "entropy_median_partition": float(h_med),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)

    out["risk_score_seq"] = 0.0
    for pvals, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)