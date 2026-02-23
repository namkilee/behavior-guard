from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd
import numpy as np


# -----------------------
# Outcome normalization
# (Track A 규칙을 "그대로"라고 했지만, 여기선 Track B 단독 실행을 위해
# 최소한의 호환 규칙을 제공한다.
# - 실제 Track A normalize 함수가 따로 있으면, 이 함수를 그걸로 교체하면 됨.
# -----------------------
def normalize_outcome(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "unknown"

    s = str(raw).strip().lower()
    if not s:
        return "unknown"

    # Common-ish buckets (ok / error / rate_limited / blocked / timeout / unknown)
    # Track A에서 쓰던 bucket명이 다르면 여기 mapping만 맞추면 됨.
    if s in {"ok", "success", "completed"}:
        return "ok"

    if "rate" in s and "limit" in s:
        return "rate_limited"
    if s in {"429"}:
        return "rate_limited"

    if "block" in s or "denied" in s or "forbidden" in s:
        return "blocked"
    if s in {"403"}:
        return "blocked"

    if "timeout" in s or "timed out" in s:
        return "timeout"
    if s in {"408"}:
        return "timeout"

    # errors
    if s in {"error", "failed", "failure", "exception"}:
        return "error"
    if s in {"500", "502", "503", "504"}:
        return "error"
    if "error" in s or "exception" in s or "fail" in s:
        return "error"

    return s


def _infer_list_columns(df: pd.DataFrame) -> list[str]:
    # Heuristic: column where any cell is list-like
    candidates: list[str] = []
    for c in df.columns:
        # sample a few values
        ser = df[c]
        for v in ser.head(50).tolist():
            if isinstance(v, (list, tuple, np.ndarray)):
                candidates.append(c)
                break
    return candidates


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce", utc=False)


def build_event_frame(
    df_in: pd.DataFrame,
    *,
    col_project: str,
    col_user: str,
    col_session: str,
    col_event_time: str,
    col_observation_id: str,
    col_route_group: str,
    col_outcome: str,
    src_route: str,
    src_outcome: str,
    explode_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Input: sessions_packed parquet (likely packed arrays)
    Output: event-level frame with:
      - join key cols: project_id, day, user_id_norm, session_id_norm
      - ordering cols: event_time, observation_id
      - token cols: route_group, normalized_outcome, token
    """
    df = df_in.copy()

    # Determine columns to explode
    if explode_cols is None:
        explode_cols = _infer_list_columns(df)

    # If we have at least one list column, explode them consistently
    # Strategy:
    # - If multiple explode cols exist, explode sequentially.
    # - Assumes aligned list lengths per row (typical packed schema).
    for c in explode_cols:
        if c in df.columns:
            df = df.explode(c, ignore_index=True)

    # Ensure required identity columns exist
    for required in (col_project, col_user, col_session):
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    # event_time / observation_id fallback if absent
    if col_event_time not in df.columns:
        raise ValueError(
            f"Missing event_time column '{col_event_time}'. "
            f"If packed schema differs, rename or adjust --col-event-time."
        )
    if col_observation_id not in df.columns:
        # not strictly fatal, but ordering spec includes it
        # create stable surrogate if missing
        df[col_observation_id] = np.arange(len(df), dtype=np.int64)

    # route_group
    if col_route_group not in df.columns:
        if src_route in df.columns:
            df[col_route_group] = df[src_route].astype("string")
        else:
            df[col_route_group] = "unknown_route"

    # normalized_outcome
    if col_outcome not in df.columns:
        if src_outcome in df.columns:
            df[col_outcome] = df[src_outcome].map(normalize_outcome).astype("string")
        else:
            df[col_outcome] = "unknown"

    # Parse datetimes and day
    df[col_event_time] = _safe_to_datetime(df[col_event_time])
    # day partition key
    df["day"] = df[col_event_time].dt.date.astype("string")

    # Token (Frozen v0.1)
    df["token"] = (df[col_route_group].astype("string") + ":" + df[col_outcome].astype("string")).astype("string")

    # Keep only columns we need + passthrough (drilldown might need extras)
    # We'll preserve everything, but enforce ordering and key presence.
    # Ordering (Frozen v0.1): event_time ASC, observation_id ASC
    df = df.sort_values([col_project, "day", col_user, col_session, col_event_time, col_observation_id], kind="mergesort")

    return df