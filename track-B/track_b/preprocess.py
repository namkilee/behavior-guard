from __future__ import annotations

import numpy as np
import pandas as pd


JOIN_KEY = ["client_name", "day", "user_id", "session_key"]


def explode_canonical_packed_to_events(packed: pd.DataFrame) -> pd.DataFrame:
    """
    Input: canonical packed (one row per session)
      - event_time: list
      - observation_id: list
      - route_group: list
      - outcome_class: list
      - token: list

    Output: event-level dataframe with guaranteed columns:
      JOIN_KEY + event_time + observation_id + route_group + outcome_class + token
    """
    df = packed.copy()

    for c in JOIN_KEY:
        if c not in df.columns:
            raise ValueError(f"packed missing required key col: {c}")

    required_arrays = ["event_time", "observation_id", "route_group", "outcome_class", "token"]
    for c in required_arrays:
        if c not in df.columns:
            raise ValueError(f"packed missing required array col: {c}")

    # explode arrays together
    explode_cols = required_arrays
    ev = df.explode(explode_cols, ignore_index=True)

    # normalize dtypes
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce", utc=False)
    ev["observation_id"] = pd.to_numeric(ev["observation_id"], errors="coerce").fillna(-1).astype(np.int64)

    ev["route_group"] = ev["route_group"].astype("string")
    ev["outcome_class"] = ev["outcome_class"].astype("string")
    ev["token"] = ev["token"].astype("string")

    # ordering guarantee (Frozen ordering rule adapted to our schema)
    ev = ev.sort_values(JOIN_KEY + ["event_time", "observation_id"], kind="mergesort").reset_index(drop=True)
    return ev