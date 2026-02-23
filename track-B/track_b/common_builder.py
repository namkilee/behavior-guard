from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


# Track A 현실 기준 키
KEY_COLS = ["user_id", "client_name", "session_key"]


def _is_list_like(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _parse_maybe_list(x: Any) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        # naive CSV
        if "," in s and "[" not in s and "{" not in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        # json list
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce", utc=False)


def _guess_time_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["event_time", "event_ts", "timestamp", "start_time", "created_at", "ts"]:
        if cand in df.columns:
            return cand
    return None


def _is_event_level_raw(raw: pd.DataFrame) -> bool:
    if any(c not in raw.columns for c in KEY_COLS):
        return False
    sizes = raw.groupby(KEY_COLS).size()
    return (sizes.max() if len(sizes) else 0) > 1


def _is_packed_session_raw(raw: pd.DataFrame) -> bool:
    if any(c not in raw.columns for c in KEY_COLS):
        return False
    candidates = ["tokens", "outcomes", "route_groups", "event_times", "dt_buckets"]
    present = [c for c in candidates if c in raw.columns]
    if not present:
        return False
    sample = raw[present].head(50)
    for c in present:
        if sample[c].apply(_is_list_like).any():
            return True
    return False


def _ensure_day_column(df: pd.DataFrame, *, day: str) -> pd.Series:
    # canonical day is fixed by CLI argument (research run/day partition)
    return pd.Series([day] * len(df), index=df.index, dtype="string")


def _build_token(route_group: pd.Series, outcome_class: pd.Series) -> pd.Series:
    return (route_group.astype("string") + ":" + outcome_class.astype("string")).astype("string")


def _canonicalize_from_event_level(raw: pd.DataFrame, *, day: str) -> pd.DataFrame:
    df = raw.copy()

    # required keys
    for c in KEY_COLS:
        if c not in df.columns:
            raise ValueError(f"raw missing required key col: {c}")

    # normalize cols to expected names for canonical packing
    if "route_group" not in df.columns and "route_groups" in df.columns:
        df["route_group"] = df["route_groups"]
    if "outcome_class" not in df.columns and "outcomes" in df.columns:
        df["outcome_class"] = df["outcomes"]
    if "event_time" not in df.columns and "event_times" in df.columns:
        df["event_time"] = df["event_times"]

    # event_time
    time_col = _guess_time_col(df)
    if time_col is None:
        # allow missing event_time: use row order
        df["event_time"] = pd.NaT
        time_col = "event_time"
    else:
        if time_col != "event_time":
            df["event_time"] = df[time_col]
            time_col = "event_time"

    df["event_time"] = _safe_to_datetime(df["event_time"])

    # outcome/route defaults
    if "route_group" not in df.columns:
        df["route_group"] = "unknown_route"
    if "outcome_class" not in df.columns:
        df["outcome_class"] = "unknown"

    # ordering: event_time asc, then stable row order
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(KEY_COLS + ["event_time", "_row_idx"], kind="mergesort")

    # observation_id surrogate per session in order
    df["observation_id"] = df.groupby(KEY_COLS, sort=False).cumcount().astype(np.int64)

    # token
    df["token"] = _build_token(df["route_group"], df["outcome_class"])

    # pack to one row per session with arrays
    packed = (
        df.groupby(KEY_COLS, sort=False)
        .agg(
            day=("event_time", lambda _: day),
            event_time=("event_time", lambda s: s.tolist()),
            observation_id=("observation_id", lambda s: s.tolist()),
            route_group=("route_group", lambda s: s.astype("string").tolist()),
            outcome_class=("outcome_class", lambda s: s.astype("string").tolist()),
            token=("token", lambda s: s.astype("string").tolist()),
        )
        .reset_index()
    )

    # optional dt_bucket if exists
    if "dt_bucket" in df.columns:
        dtp = df.groupby(KEY_COLS, sort=False)["dt_bucket"].apply(lambda s: s.tolist()).reset_index()
        packed = packed.merge(dtp, on=KEY_COLS, how="left")

    return packed


def _canonicalize_from_packed_session(raw: pd.DataFrame, *, day: str) -> pd.DataFrame:
    df = raw.copy()

    # required keys
    for c in KEY_COLS:
        if c not in df.columns:
            raise ValueError(f"raw missing required key col: {c}")

    # rename packed columns to canonical names if present
    rename_map = {}
    if "tokens" in df.columns:
        rename_map["tokens"] = "token"
    if "outcomes" in df.columns:
        rename_map["outcomes"] = "outcome_class"
    if "route_groups" in df.columns:
        rename_map["route_groups"] = "route_group"
    if "event_times" in df.columns:
        rename_map["event_times"] = "event_time"
    if "dt_buckets" in df.columns:
        rename_map["dt_buckets"] = "dt_bucket"
    df = df.rename(columns=rename_map)

    # parse lists
    for c in ["token", "outcome_class", "route_group", "event_time", "dt_bucket"]:
        if c in df.columns:
            df[c] = df[c].apply(_parse_maybe_list)

    # trim to min aligned length across present list cols
    explode_cols = [c for c in ["token", "outcome_class", "route_group", "event_time", "dt_bucket"] if c in df.columns]

    def trim_row(r: pd.Series) -> pd.Series:
        lens = []
        for c in explode_cols:
            v = r[c]
            lens.append(len(v) if isinstance(v, list) else 0)
        m = min([x for x in lens if x > 0], default=0)
        if m == 0:
            return r
        for c in explode_cols:
            if isinstance(r[c], list) and len(r[c]) != m:
                r[c] = r[c][:m]
        return r

    df = df.apply(trim_row, axis=1)

    # ensure route/outcome exist
    if "route_group" not in df.columns:
        df["route_group"] = [[] for _ in range(len(df))]
    if "outcome_class" not in df.columns:
        df["outcome_class"] = [[] for _ in range(len(df))]

    # ensure token exists (build if absent/empty)
    if "token" not in df.columns:
        # build token from route/outcome lists
        def build_tok_list(r: pd.Series) -> list[str]:
            rg = r["route_group"]
            oc = r["outcome_class"]
            m = min(len(rg), len(oc))
            return [f"{str(rg[i])}:{str(oc[i])}" for i in range(m)]
        df["token"] = df.apply(build_tok_list, axis=1)
    else:
        # if token list is empty but route/outcome exist, rebuild
        def ensure_tok(r: pd.Series) -> list[str]:
            t = r["token"]
            if isinstance(t, list) and len(t) > 0:
                return [str(x) for x in t]
            rg = r["route_group"]
            oc = r["outcome_class"]
            m = min(len(rg), len(oc))
            return [f"{str(rg[i])}:{str(oc[i])}" for i in range(m)]
        df["token"] = df.apply(ensure_tok, axis=1)

    # event_time parse
    if "event_time" in df.columns:
        def parse_time_list(x: list) -> list:
            s = pd.to_datetime(pd.Series(x), errors="coerce", utc=False)
            return s.tolist()
        df["event_time"] = df["event_time"].apply(parse_time_list)
    else:
        df["event_time"] = [[] for _ in range(len(df))]

    # observation_id surrogate (0..n-1)
    def obs_ids(tok_list: list) -> list[int]:
        n = len(tok_list) if isinstance(tok_list, list) else 0
        return list(range(n))

    df["observation_id"] = df["token"].apply(obs_ids)

    # day fixed
    df["day"] = _ensure_day_column(df, day=day)

    # keep only canonical columns (+dt_bucket if exists)
    keep = KEY_COLS + ["day", "event_time", "observation_id", "route_group", "outcome_class", "token"]
    if "dt_bucket" in df.columns:
        keep.append("dt_bucket")

    # ensure lists are lists of strings for route/outcome/token
    df["route_group"] = df["route_group"].apply(lambda arr: [str(x) for x in arr] if isinstance(arr, list) else [])
    df["outcome_class"] = df["outcome_class"].apply(lambda arr: [str(x) for x in arr] if isinstance(arr, list) else [])
    df["token"] = df["token"].apply(lambda arr: [str(x) for x in arr] if isinstance(arr, list) else [])

    return df[keep].copy()


def build_sessions_packed_df(raw: pd.DataFrame, *, day: str) -> pd.DataFrame:
    if any(c not in raw.columns for c in KEY_COLS):
        raise ValueError(f"raw must include key cols {KEY_COLS}")

    if _is_event_level_raw(raw):
        packed = _canonicalize_from_event_level(raw, day=day)
        packed["source_mode"] = "event_level"
        return packed

    if _is_packed_session_raw(raw):
        packed = _canonicalize_from_packed_session(raw, day=day)
        packed["source_mode"] = "packed_session"
        return packed

    # fallback: treat as session-level but not packed; try to use existing columns
    # if it has token/outcome/route as scalar, wrap into 1-element arrays
    df = raw.copy()
    df["day"] = _ensure_day_column(df, day=day)

    for c in ["route_group", "outcome_class", "token", "event_time"]:
        if c in df.columns and not df[c].apply(_is_list_like).any():
            df[c] = df[c].apply(lambda x: [x] if pd.notna(x) else [])
        elif c not in df.columns:
            df[c] = [[] for _ in range(len(df))]

    if "token" not in df.columns or df["token"].map(len).fillna(0).eq(0).all():
        # build from route/outcome
        def build_tok_list(r: pd.Series) -> list[str]:
            rg = r["route_group"]
            oc = r["outcome_class"]
            m = min(len(rg), len(oc))
            return [f"{str(rg[i])}:{str(oc[i])}" for i in range(m)]
        df["token"] = df.apply(build_tok_list, axis=1)

    df["observation_id"] = df["token"].apply(lambda t: list(range(len(t))) if isinstance(t, list) else [])

    keep = KEY_COLS + ["day", "event_time", "observation_id", "route_group", "outcome_class", "token"]
    df = df[keep].copy()
    df["source_mode"] = "session_fallback"
    return df


def build_schema_manifest(packed: pd.DataFrame, *, day: str, in_path: str, out_path: str) -> dict:
    return {
        "behaviorguard": {
            "component": "common_sessions_packed_builder",
            "schema_version": "0.1",
            "day": day,
        },
        "io": {"input": in_path, "output": out_path},
        "columns": packed.columns.tolist(),
        "dtypes": {c: str(packed.dtypes[c]) for c in packed.columns},
        "notes": [
            "Canonical join key: (client_name, day, user_id, session_key)",
            "token = route_group:outcome_class",
            "observation_id is surrogate if missing upstream",
        ],
    }


def build_sessions_packed_file(
    *,
    day: str,
    in_path: Path,
    out_path: Path,
    manifest_path: Path,
    overwrite: bool = False,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {out_path} (use --overwrite 1)")

    raw = pd.read_parquet(in_path)
    if raw.empty:
        raise ValueError(f"input parquet is empty: {in_path}")

    packed = build_sessions_packed_df(raw, day=day)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed.to_parquet(out_path, index=False)

    manifest = build_schema_manifest(
        packed=packed,
        day=day,
        in_path=str(in_path),
        out_path=str(out_path),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")