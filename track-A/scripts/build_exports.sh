#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

need_cmd clickhouse-client
need_cmd cat
need_cmd mkdir
need_cmd date

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DAY="${1:-${DAY:-}}"
[[ -n "$DAY" ]] || die "DAY is required. Usage: scripts/build_exports.sh YYYY-MM-DD"
[[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "Invalid DAY format: $DAY"

# required env (pipeline.sh에서 export되어 내려온다고 가정)
: "${CH_HOST:?Missing CH_HOST}"
: "${CH_PORT:?Missing CH_PORT}"
: "${CH_USER:?Missing CH_USER}"
: "${CH_PASSWORD:?Missing CH_PASSWORD}"
: "${CH_DATABASE:?Missing CH_DATABASE}"

OUT_DIR="${OUT_DIR:-out}"
DAILY_DIR="${DAILY_DIR:-$OUT_DIR/daily}"
SQL_BUILD_DIR="${SQL_BUILD_DIR:-sql/build}"

DAY_START="${DAY} 00:00:00"
DAY_END="$(date -d "$DAY +1 day" +"%Y-%m-%d") 00:00:00"

SQL_FEATURES_PATH="$ROOT_DIR/$SQL_BUILD_DIR/export_session_features.built.sql"
SQL_RAW_PATH="$ROOT_DIR/$SQL_BUILD_DIR/export_sessions_raw.built.sql"

[[ -f "$SQL_FEATURES_PATH" ]] || die "Built SQL not found: $SQL_FEATURES_PATH (run build_sql.sh first)"
[[ -f "$SQL_RAW_PATH" ]] || die "Built SQL not found: $SQL_RAW_PATH (run build_sql.sh first)"

mkdir -p "$ROOT_DIR/$DAILY_DIR"

FEATURE_OUT="$ROOT_DIR/$DAILY_DIR/session_features_${DAY}.parquet"
RAW_OUT="$ROOT_DIR/$DAILY_DIR/sessions_raw_${DAY}.parquet"

echo "[INFO] Export DAY=$DAY"
echo "[INFO] Range: $DAY_START ~ $DAY_END"
echo "[INFO] Using built SQL dir: $SQL_BUILD_DIR"
echo "[INFO] DAILY_DIR=$DAILY_DIR"

clickhouse-client \
  --host "$CH_HOST" \
  --port "$CH_PORT" \
  --user "$CH_USER" \
  --password "$CH_PASSWORD" \
  --database "$CH_DATABASE" \
  --param_day_start "$DAY_START" \
  --param_day_end "$DAY_END" \
  --query "$(cat "$SQL_FEATURES_PATH")" \
  > "$FEATURE_OUT"

clickhouse-client \
  --host "$CH_HOST" \
  --port "$CH_PORT" \
  --user "$CH_USER" \
  --password "$CH_PASSWORD" \
  --database "$CH_DATABASE" \
  --param_day_start "$DAY_START" \
  --param_day_end "$DAY_END" \
  --query "$(cat "$SQL_RAW_PATH")" \
  > "$RAW_OUT"

[[ -s "$FEATURE_OUT" ]] || die "Exported file is empty: $FEATURE_OUT"
[[ -s "$RAW_OUT" ]] || die "Exported file is empty: $RAW_OUT"

echo "[INFO] Done."
echo "[INFO] - $FEATURE_OUT"
echo "[INFO] - $RAW_OUT"