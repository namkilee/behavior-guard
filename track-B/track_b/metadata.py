from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import subprocess
from typing import Any, Dict


def _try_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def build_run_metadata(
    *,
    track: str,
    track_version: str,
    input_path: str,
    outdir: str,
    args: Dict[str, Any],
    n_rows_input: int,
    n_rows_events: int,
) -> Dict[str, Any]:
    return {
        "behaviorguard": {
            "track": track,
            "track_version": track_version,
            "mode": "research",
        },
        "runtime": {
            "utc_started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else None,
            "git_commit": _try_git_commit(),
        },
        "io": {
            "input_path": input_path,
            "outdir": outdir,
            "n_rows_input": n_rows_input,
            "n_rows_events": n_rows_events,
        },
        "args": args,
    }