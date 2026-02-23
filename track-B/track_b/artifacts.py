from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import math


JOIN_KEY = ["project_id", "day", "user_id_norm", "session_id_norm"]


def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)


def _topk_sessions(session_scores: pd.DataFrame, topk: int) -> pd.DataFrame:
    # Higher risk_score_seq = more anomalous (Frozen)
    return session_scores.sort_values(
        ["risk_score_seq", "seq_raw"],
        ascending=[False, False],
        kind="mergesort",
    ).head(topk).reset_index(drop=True)


def _drilldown_for_b1(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Drilldown includes:
      - ordered events
      - token
      - prev_token / next_token
    (transition probability explanation is doable but requires access to trained table;
     in research mode we keep it lightweight and deterministic)
    """
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")

    # prev/next token within session
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")
    dd["prev_token"] = dd.groupby(JOIN_KEY, sort=False)["token"].shift(1)
    dd["next_token"] = dd.groupby(JOIN_KEY, sort=False)["token"].shift(-1)
    return dd


def _drilldown_for_b2(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")
    # n-gram windows (2,3) as strings for human scan
    tokens = dd.groupby(JOIN_KEY, sort=False)["token"].apply(list).reset_index()
    tokens["bigrams"] = tokens["token"].map(lambda t: ["|".join(t[i:i+2]) for i in range(max(len(t)-1, 0))])
    tokens["trigrams"] = tokens["token"].map(lambda t: ["|".join(t[i:i+3]) for i in range(max(len(t)-2, 0))])
    # explode to rows for readability
    b2 = tokens.explode("bigrams", ignore_index=True).rename(columns={"bigrams": "ngram_bigram"})
    b3 = tokens.explode("trigrams", ignore_index=True).rename(columns={"trigrams": "ngram_trigram"})
    out = b2.merge(b3[JOIN_KEY + ["ngram_trigram"]], on=JOIN_KEY, how="outer")
    return out


def _drilldown_for_b3(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")

    # top tokens by frequency per session for quick "shape" inspection
    def top_tokens(toks: list[str], k: int = 10) -> list[str]:
        s = pd.Series(toks, dtype="string")
        vc = s.value_counts().head(k)
        return [f"{idx}({int(v)})" for idx, v in vc.items()]

    agg = dd.groupby(JOIN_KEY, sort=False)["token"].apply(list).reset_index()
    agg["top_tokens"] = agg["token"].map(top_tokens)
    # also keep ordered raw events for deep scan
    # We'll join to each event row so reviewer can filter by session key then see top_tokens
    dd = dd.merge(agg[JOIN_KEY + ["top_tokens"]], on=JOIN_KEY, how="left")
    return dd


def write_summary_and_drilldown(
    *,
    model_name: str,
    events: pd.DataFrame,
    session_scores: pd.DataFrame,
    outdir: Path,
    topk: int,
    fmt: str,
) -> None:
    """
    Emits:
      - summary_topk.(parquet|csv)
      - drilldown_topk.(parquet|csv)
      - all_scores.(parquet|csv)  (optional but useful)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # sanity: key presence
    for c in JOIN_KEY:
        if c not in session_scores.columns:
            raise ValueError(f"session_scores missing key col: {c}")
        if c not in events.columns:
            raise ValueError(f"events missing key col: {c}")

    topk_df = _topk_sessions(session_scores, topk)

    # Summary: keep a stable, reviewer-friendly subset first, plus model fields
    base_cols = JOIN_KEY + ["risk_score_seq", "seq_raw", "n_events"]
    model_cols = [c for c in session_scores.columns if c not in base_cols]
    summary_cols = base_cols + model_cols

    summary = topk_df[summary_cols].copy()
    _write_df(summary, outdir / "summary_topk", fmt)

    # Drilldown: model specific
    topk_keys = topk_df[JOIN_KEY].drop_duplicates()

    if model_name == "b1":
        drill = _drilldown_for_b1(events, topk_keys)
    elif model_name == "b2":
        drill = _drilldown_for_b2(events, topk_keys)
    elif model_name == "b3":
        drill = _drilldown_for_b3(events, topk_keys)
    else:
        drill = events.merge(topk_keys, on=JOIN_KEY, how="inner")

    _write_df(drill, outdir / "drilldown_topk", fmt)

    # All scores (for evaluation / overlap / stability later)
    _write_df(session_scores, outdir / "all_scores", fmt)