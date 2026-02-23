from __future__ import annotations

from pathlib import Path
import pandas as pd

JOIN_KEY = ["client_name", "day", "user_id", "session_key"]


def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)


def _topk_sessions(session_scores: pd.DataFrame, topk: int) -> pd.DataFrame:
    return session_scores.sort_values(
        ["risk_score_seq", "seq_raw"],
        ascending=[False, False],
        kind="mergesort",
    ).head(topk).reset_index(drop=True)


def _drilldown_b1(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")
    dd["prev_token"] = dd.groupby(JOIN_KEY, sort=False)["token"].shift(1)
    dd["next_token"] = dd.groupby(JOIN_KEY, sort=False)["token"].shift(-1)
    return dd


def _drilldown_b2(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")

    tokens = dd.groupby(JOIN_KEY, sort=False)["token"].apply(list).reset_index()
    tokens["bigram"] = tokens["token"].map(lambda t: ["|".join(t[i:i+2]) for i in range(max(len(t)-1, 0))])
    tokens["trigram"] = tokens["token"].map(lambda t: ["|".join(t[i:i+3]) for i in range(max(len(t)-2, 0))])

    b2 = tokens.explode("bigram", ignore_index=True).rename(columns={"bigram": "ngram_bigram"})
    b3 = tokens.explode("trigram", ignore_index=True).rename(columns={"trigram": "ngram_trigram"})
    return b2.merge(b3[JOIN_KEY + ["ngram_trigram"]], on=JOIN_KEY, how="outer")


def _drilldown_b3(events: pd.DataFrame, topk_keys: pd.DataFrame) -> pd.DataFrame:
    dd = events.merge(topk_keys[JOIN_KEY], on=JOIN_KEY, how="inner")
    dd = dd.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort")

    def top_tokens(toks: list[str], k: int = 10) -> list[str]:
        s = pd.Series(toks, dtype="string")
        vc = s.value_counts().head(k)
        return [f"{idx}({int(v)})" for idx, v in vc.items()]

    agg = dd.groupby(JOIN_KEY, sort=False)["token"].apply(list).reset_index()
    agg["top_tokens"] = agg["token"].map(top_tokens)
    return dd.merge(agg[JOIN_KEY + ["top_tokens"]], on=JOIN_KEY, how="left")


def write_summary_and_drilldown(
    *,
    model_name: str,
    events: pd.DataFrame,
    session_scores: pd.DataFrame,
    outdir: Path,
    topk: int,
    fmt: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for c in JOIN_KEY:
        if c not in session_scores.columns:
            raise ValueError(f"session_scores missing key col: {c}")
        if c not in events.columns:
            raise ValueError(f"events missing key col: {c}")

    topk_df = _topk_sessions(session_scores, topk)
    topk_keys = topk_df[JOIN_KEY].drop_duplicates()

    base_cols = JOIN_KEY + ["risk_score_seq", "seq_raw", "n_events"]
    model_cols = [c for c in session_scores.columns if c not in base_cols]
    summary = topk_df[base_cols + model_cols].copy()

    _write_df(summary, outdir / "summary_topk", fmt)

    if model_name == "b1":
        drill = _drilldown_b1(events, topk_keys)
    elif model_name == "b2":
        drill = _drilldown_b2(events, topk_keys)
    elif model_name == "b3":
        drill = _drilldown_b3(events, topk_keys)
    else:
        drill = events.merge(topk_keys, on=JOIN_KEY, how="inner")

    _write_df(drill, outdir / "drilldown_topk", fmt)
    _write_df(session_scores, outdir / "all_scores", fmt)