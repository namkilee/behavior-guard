#!/usr/bin/env bash
set -Eeuo pipefail
die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd(){ command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

need_cmd clickhouse-client
need_cmd cat
need_cmd mkdir
need_cmd date

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DAY="${1:-${DAY:-}}"
[[ -n "$DAY" ]] || die "DAY is required. Usage: scripts/build_exports.sh YYYY-MM-DD"
[[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "Invalid DAY format: $DAY"

# required env (pipeline.sh에서 export되어 내려온다고 가정)
: "${CH_HOST:?}"
: "${CH_PORT:?}"
: "${CH_USER:?}"
: "${CH_PASSWORD:?}"
: "${CH_DATABASE:?}"
: "${SQL_SESSION_FEATURES:?}"
: "${SQL_SESSIONS_RAW:?}"

OUT_DIR="${OUT_DIR:-out}"

DAY_START="${DAY} 00:00:00"
DAY_END="$(date -d "$DAY +1 day" +"%Y-%m-%d") 00:00:00"

SQL_FEATURES_PATH="$ROOT_DIR/$SQL_SESSION_FEATURES"
SQL_RAW_PATH="$ROOT_DIR/$SQL_SESSIONS_RAW"

mkdir -p "$ROOT_DIR/$OUT_DIR"

clickhouse-client \
  --host "$CH_HOST" --port "$CH_PORT" --user "$CH_USER" --password "$CH_PASSWORD" \
  --database "$CH_DATABASE" \
  --param_day_start "$DAY_START" \
  --param_day_end "$DAY_END" \
  --query "$(cat "$SQL_FEATURES_PATH")" \
  > "$ROOT_DIR/$OUT_DIR/session_features_${DAY}.parquet"

clickhouse-client \
  --host "$CH_HOST" --port "$CH_PORT" --user "$CH_USER" --password "$CH_PASSWORD" \
  --database "$CH_DATABASE" \
  --param_day_start "$DAY_START" \
  --param_day_end "$DAY_END" \
  --query "$(cat "$SQL_RAW_PATH")" \
  > "$ROOT_DIR/$OUT_DIR/sessions_raw_${DAY}.parquet"
