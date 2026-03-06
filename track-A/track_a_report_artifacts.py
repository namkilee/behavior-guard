#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_latest(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return Path(matches[-1])


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p}")


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def split_tags(value) -> list[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def pick_case(summary: pd.DataFrame, tag: str) -> pd.Series | None:
    if "risk_tags" not in summary.columns:
        return None
    mask = summary["risk_tags"].fillna("").str.contains(fr"(^|,){tag}(,|$)", regex=True)
    sub = summary.loc[mask].copy()
    if sub.empty:
        return None
    sort_cols = [c for c in ["risk_pct", "risk_score", "anomaly_score"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=False)
    return sub.iloc[0]


def plot_histogram(series: pd.Series, title: str, xlabel: str, outpath: Path) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Session count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_bar(labels: list[str], values: list[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    if not labels:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def build_tables(features: pd.DataFrame, summary: pd.DataFrame, drilldown: pd.DataFrame | None, out_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    feature_num_cols = [
        "anomaly_score", "risk_score", "risk_pct", "n_events", "duration_s",
        "error_rate_dd", "rate_limited_rate_dd", "burst_score", "max_events_in_30s",
        "p50_gap_s", "p95_gap_s",
    ]
    features = to_numeric(features, feature_num_cols)
    summary = to_numeric(summary, feature_num_cols)
    if drilldown is not None:
        drilldown = to_numeric(drilldown, feature_num_cols)

    # 1) overall score summary
    score_metric = "risk_pct" if "risk_pct" in features.columns else "anomaly_score"
    s = features[score_metric].dropna()
    overall_rows = [
        ("total_sessions", int(len(features))),
        ("unique_users", int(features["user_id"].nunique()) if "user_id" in features.columns else np.nan),
        ("unique_clients", int(features["client_name"].nunique()) if "client_name" in features.columns else np.nan),
        (f"{score_metric}_p50", float(s.quantile(0.50)) if len(s) else np.nan),
        (f"{score_metric}_p95", float(s.quantile(0.95)) if len(s) else np.nan),
        (f"{score_metric}_p99", float(s.quantile(0.99)) if len(s) else np.nan),
        (f"{score_metric}_max", float(s.max()) if len(s) else np.nan),
    ]
    if score_metric == "risk_pct":
        overall_rows.extend([
            ("sessions_risk_pct_ge_95", int((features["risk_pct"] >= 95).sum())),
            ("sessions_risk_pct_ge_99", int((features["risk_pct"] >= 99).sum())),
        ])
    overall_df = pd.DataFrame(overall_rows, columns=["metric", "value"])
    out = out_dir / "table_trackA_overall_score_summary.csv"
    overall_df.to_csv(out, index=False)
    outputs["overall_table"] = out

    # 2) top anomalous sessions table
    top_cols = [
        c for c in [
            "risk_pct", "risk_score", "anomaly_score", "user_id", "client_name", "session_key",
            "n_events", "duration_s", "error_rate_dd", "rate_limited_rate_dd", "burst_score",
            "route_skew_pct", "max_events_in_30s", "risk_tags", "why_ranked",
        ] if c in summary.columns
    ]
    top_df = summary.copy()
    sort_cols = [c for c in ["risk_pct", "risk_score", "anomaly_score"] if c in top_df.columns]
    if sort_cols:
        top_df = top_df.sort_values(sort_cols, ascending=False)
    top_df = top_df[top_cols].head(20).copy()
    top_df.insert(0, "rank", range(1, len(top_df) + 1))
    out = out_dir / "table_trackA_top20_sessions.csv"
    top_df.to_csv(out, index=False)
    outputs["top20_table"] = out

    # 3) risk tag frequency
    tags = []
    if "risk_tags" in summary.columns:
        for value in summary["risk_tags"]:
            tags.extend(split_tags(value))
    tag_counts = pd.Series(tags, dtype="object").value_counts().rename_axis("risk_tag").reset_index(name="count")
    out = out_dir / "table_trackA_risk_tag_counts.csv"
    tag_counts.to_csv(out, index=False)
    outputs["risk_tag_table"] = out

    # 4) client composition within top-K
    if "client_name" in summary.columns:
        client_counts = summary["client_name"].fillna("UNKNOWN").value_counts().rename_axis("client_name").reset_index(name="count")
        client_counts["ratio"] = client_counts["count"] / max(len(summary), 1)
        out = out_dir / "table_trackA_topk_client_composition.csv"
        client_counts.to_csv(out, index=False)
        outputs["client_table"] = out

    # 5) representative cases
    cases = []
    for tag in ["BURST", "ROUTE_SKEW", "ERROR_HEAVY", "RATE_LIMIT_HEAVY", "LONG_DURATION"]:
        row = pick_case(summary, tag)
        if row is None:
            continue
        evidence_parts = []
        for c in ["max_events_in_30s", "burst_score", "route_skew_pct", "error_rate_dd", "rate_limited_rate_dd", "timeline_1line", "why_ranked"]:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
                evidence_parts.append(f"{c}={row[c]}")
        cases.append({
            "pattern": tag,
            "user_id": row.get("user_id", ""),
            "client_name": row.get("client_name", ""),
            "session_key": row.get("session_key", ""),
            "risk_pct": row.get("risk_pct", np.nan),
            "risk_score": row.get("risk_score", np.nan),
            "evidence": " | ".join(evidence_parts),
        })
    case_df = pd.DataFrame(cases)
    out = out_dir / "table_trackA_representative_cases.csv"
    case_df.to_csv(out, index=False)
    outputs["cases_table"] = out

    return outputs


def build_figures(features: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    features = to_numeric(features, ["anomaly_score", "risk_score", "risk_pct"])

    if "anomaly_score" in features.columns:
        out = out_dir / "fig_trackA_anomaly_score_hist.png"
        plot_histogram(features["anomaly_score"], "Track A anomaly score distribution", "anomaly_score", out)
        outputs["anomaly_hist"] = out

    if "risk_pct" in features.columns:
        out = out_dir / "fig_trackA_risk_pct_hist.png"
        plot_histogram(features["risk_pct"], "Track A risk percentile distribution", "risk_pct", out)
        outputs["risk_pct_hist"] = out

    if "risk_tags" in summary.columns:
        tags = []
        for value in summary["risk_tags"]:
            tags.extend(split_tags(value))
        vc = pd.Series(tags, dtype="object").value_counts().head(10)
        if not vc.empty:
            out = out_dir / "fig_trackA_risk_tag_top10.png"
            plot_bar(vc.index.tolist(), vc.values.tolist(), "Top risk tags within Track A Top-K", "risk_tag", "count", out)
            outputs["risk_tag_bar"] = out

    if "client_name" in summary.columns:
        vc = summary["client_name"].fillna("UNKNOWN").value_counts().head(10)
        if not vc.empty:
            out = out_dir / "fig_trackA_topk_client_composition.png"
            plot_bar(vc.index.tolist(), vc.values.tolist(), "Top-K client composition", "client_name", "count", out)
            outputs["client_bar"] = out

    return outputs


def build_markdown_summary(features: pd.DataFrame, summary: pd.DataFrame, table_paths: dict[str, Path], fig_paths: dict[str, Path], out_dir: Path) -> Path:
    features = to_numeric(features, ["risk_pct", "anomaly_score"])
    score_metric = "risk_pct" if "risk_pct" in features.columns else "anomaly_score"
    s = features[score_metric].dropna()

    lines = [
        "# Track A Results Summary",
        "",
        "## Key findings",
        "",
        f"- Total scored sessions: **{len(features):,}**",
    ]
    if "user_id" in features.columns:
        lines.append(f"- Unique users: **{features['user_id'].nunique():,}**")
    if "client_name" in features.columns:
        lines.append(f"- Unique clients: **{features['client_name'].nunique():,}**")
    if len(s):
        lines.append(f"- {score_metric} p95: **{s.quantile(0.95):.2f}**")
        lines.append(f"- {score_metric} p99: **{s.quantile(0.99):.2f}**")
        lines.append(f"- {score_metric} max: **{s.max():.2f}**")
    if score_metric == "risk_pct":
        lines.append(f"- Sessions with risk_pct >= 95: **{int((features['risk_pct'] >= 95).sum()):,}**")
        lines.append(f"- Sessions with risk_pct >= 99: **{int((features['risk_pct'] >= 99).sum()):,}**")

    if "risk_tags" in summary.columns:
        tags = []
        for value in summary["risk_tags"]:
            tags.extend(split_tags(value))
        vc = pd.Series(tags, dtype="object").value_counts().head(5)
        if not vc.empty:
            lines += ["", "## Top-K pattern composition", ""]
            for tag, count in vc.items():
                lines.append(f"- {tag}: **{count}** cases in Top-K")

    lines += [
        "",
        "## Generated artifacts",
        "",
        f"- Overall table: `{table_paths.get('overall_table', '')}`",
        f"- Top-20 table: `{table_paths.get('top20_table', '')}`",
        f"- Risk tag table: `{table_paths.get('risk_tag_table', '')}`",
        f"- Representative cases: `{table_paths.get('cases_table', '')}`",
    ]
    for name, path in fig_paths.items():
        lines.append(f"- Figure ({name}): `{path}`")

    out = out_dir / "trackA_results_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate report-ready tables and figures from Track A outputs")
    ap.add_argument("--features-scored", help="trackA_features_scored_*.csv/parquet")
    ap.add_argument("--summary-top", help="trackA_summary_top_*.csv/parquet")
    ap.add_argument("--drilldown", help="trackA_drilldown_*.csv/parquet")
    ap.add_argument("--input-dir", default="out", help="Directory to search latest Track A outputs when explicit paths are omitted")
    ap.add_argument("--output-dir", default="out/report_trackA", help="Directory to write report-ready artifacts")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(args.output_dir)

    features_path = Path(args.features_scored) if args.features_scored else find_latest(str(input_dir / "trackA_features_scored_*.*"))
    summary_path = Path(args.summary_top) if args.summary_top else find_latest(str(input_dir / "trackA_summary_top_*.*"))

    drilldown_path = None
    if args.drilldown:
        drilldown_path = Path(args.drilldown)
    else:
        matches = sorted(glob.glob(str(input_dir / "trackA_drilldown_*.*")))
        if matches:
            drilldown_path = Path(matches[-1])

    print(f"[INFO] features_scored = {features_path}")
    print(f"[INFO] summary_top     = {summary_path}")
    if drilldown_path:
        print(f"[INFO] drilldown       = {drilldown_path}")
    else:
        print("[INFO] drilldown       = (not provided)")

    features = read_any(features_path)
    summary = read_any(summary_path)
    drilldown = read_any(drilldown_path) if drilldown_path else None

    table_paths = build_tables(features, summary, drilldown, out_dir)
    fig_paths = build_figures(features, summary, out_dir)
    md_path = build_markdown_summary(features, summary, table_paths, fig_paths, out_dir)

    print("\n[INFO] Generated artifacts:")
    for p in list(table_paths.values()) + list(fig_paths.values()) + [md_path]:
        print(" -", p)


if __name__ == "__main__":
    main()
