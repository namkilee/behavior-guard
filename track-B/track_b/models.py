from __future__ import annotations

import math
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


JOIN_KEY = ["client_name", "day", "user_id", "session_key"]


def _scale_to_risk_0_100(raw: pd.Series, *, p50: float, p95: float) -> pd.Series:
    x = raw.astype(float)
    if not np.isfinite(p50) or not np.isfinite(p95) or p95 <= p50:
        lo = np.nanmin(x.values) if len(x) else 0.0
        hi = np.nanmax(x.values) if len(x) else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - lo) / (hi - lo) * 100.0).clip(0, 100)
    return (((x - p50) / (p95 - p50)) * 100.0).clip(0, 100)


def _build_session_sequences(events: pd.DataFrame) -> pd.DataFrame:
    for c in JOIN_KEY:
        if c not in events.columns:
            raise ValueError(f"events missing key col: {c}")
    g = events.groupby(JOIN_KEY, sort=False)["token"].apply(list).reset_index()
    g["n_events"] = g["token"].map(len).astype(int)
    return g


def score_b1_markov_nll(events: pd.DataFrame, *, partition_cols: Sequence[str]) -> pd.DataFrame:
    sess = _build_session_sequences(events)
    for c in partition_cols:
        if c not in sess.columns:
            raise ValueError(f"partition col missing: {c}")

    rows = []
    for _, part_df in sess.groupby(list(partition_cols), sort=False):
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
            return (trans_counts.get((a, b), 0) + 1) / (out_counts.get(a, 0) + V)

        for _, r in part_df.iterrows():
            toks = r["token"]
            n_events = int(r["n_events"])
            n_trans = max(n_events - 1, 0)

            if n_trans <= 0:
                seq_raw = 0.0
            else:
                nll = 0.0
                for a, b in zip(toks[:-1], toks[1:]):
                    nll += -math.log(trans_prob(a, b))
                seq_raw = nll / n_trans

            row = {c: r[c] for c in JOIN_KEY}
            row.update({"model": "b1", "seq_raw": float(seq_raw), "n_events": n_events, "n_transitions": int(n_trans)})
            rows.append(row)

    out = pd.DataFrame(rows)
    out["risk_score_seq"] = 0.0
    for _, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def score_b2_ngram_rarity(
    events: pd.DataFrame, *,
    partition_cols: Sequence[str],
    n_values: Tuple[int, ...] = (2, 3),
) -> pd.DataFrame:
    sess = _build_session_sequences(events)
    rows = []

    for _, part_df in sess.groupby(list(partition_cols), sort=False):
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
                    s += -math.log(ng_prob(ng))
                n_ngrams = len(ngrams)
                seq_raw = s / n_ngrams

            row = {c: r[c] for c in JOIN_KEY}
            row.update({"model": "b2", "seq_raw": float(seq_raw), "n_events": n_events, "n_ngrams": int(n_ngrams), "ngram_ns": "2,3"})
            rows.append(row)

    out = pd.DataFrame(rows)
    out["risk_score_seq"] = 0.0
    for _, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def _entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    s = pd.Series(tokens, dtype="string")
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum())


def score_b3_entropy_deviation(events: pd.DataFrame, *, partition_cols: Sequence[str]) -> pd.DataFrame:
    sess = _build_session_sequences(events)

    rows = []
    for _, part_df in sess.groupby(list(partition_cols), sort=False):
        ent = part_df["token"].map(_entropy).astype(float)
        h_med = float(np.nanmedian(ent.values)) if len(ent) else 0.0

        for (_, r), h in zip(part_df.iterrows(), ent.values):
            raw = abs(float(h) - h_med)
            row = {c: r[c] for c in JOIN_KEY}
            row.update({"model": "b3", "seq_raw": float(raw), "n_events": int(r["n_events"]), "entropy_session": float(h), "entropy_median_partition": float(h_med)})
            rows.append(row)

    out = pd.DataFrame(rows)
    out["risk_score_seq"] = 0.0
    for _, idx in out.groupby(list(partition_cols), sort=False).groups.items():
        raw = out.loc[idx, "seq_raw"]
        p50 = float(np.nanpercentile(raw.values, 50))
        p95 = float(np.nanpercentile(raw.values, 95))
        out.loc[idx, "risk_score_seq"] = _scale_to_risk_0_100(raw, p50=p50, p95=p95)

    return out.sort_values(["risk_score_seq", "seq_raw"], ascending=[False, False], kind="mergesort").reset_index(drop=True)