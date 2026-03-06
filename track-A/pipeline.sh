#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }

# .env loader (KEY=VALUE, ignores blanks/comments)
load_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || die ".env not found at: $env_file"

  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

need_cmd bash
need_cmd date
need_cmd python3

load_env "$ENV_FILE"

START_DAY="${1:-${START_DAY:-}}"
END_DAY="${2:-${END_DAY:-$START_DAY}}"

[[ -n "$START_DAY" ]] || die "START_DAY is required. Usage: ./pipeline.sh YYYY-MM-DD [YYYY-MM-DD]"
[[ -n "$END_DAY" ]] || die "END_DAY is required."
[[ "$START_DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "Invalid START_DAY format: $START_DAY"
[[ "$END_DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "Invalid END_DAY format: $END_DAY"

if [[ "$START_DAY" > "$END_DAY" ]]; then
  die "START_DAY must be <= END_DAY"
fi

# required env
: "${CH_HOST:?Missing CH_HOST in .env}"
: "${CH_PORT:?Missing CH_PORT in .env}"
: "${CH_USER:?Missing CH_USER in .env}"
: "${CH_PASSWORD:?Missing CH_PASSWORD in .env}"
: "${CH_DATABASE:?Missing CH_DATABASE}"

OUT_DIR="${OUT_DIR:-out}"
DAILY_DIR="${DAILY_DIR:-$OUT_DIR/daily}"
MERGED_DIR="${MERGED_DIR:-$OUT_DIR/merged}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$ROOT_DIR/$DAILY_DIR" "$ROOT_DIR/$MERGED_DIR"

echo "[INFO] Pipeline START_DAY=$START_DAY END_DAY=$END_DAY"
echo "[INFO] Using ENV_FILE=$ENV_FILE"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] DAILY_DIR=$DAILY_DIR"
echo "[INFO] MERGED_DIR=$MERGED_DIR"
echo "[INFO] SKIP_EXISTING=$SKIP_EXISTING"

# 1) Build SQL once
bash "$ROOT_DIR/scripts/build_sql.sh"

CURRENT_DAY="$START_DAY"
END_EXCLUSIVE="$(date -d "$END_DAY +1 day" +"%Y-%m-%d")"

while [[ "$CURRENT_DAY" < "$END_EXCLUSIVE" ]]; do
  echo "----------------------------------------"
  echo "[INFO] Processing DAY=$CURRENT_DAY"
  echo "----------------------------------------"

  FEATURE_FILE="$ROOT_DIR/$DAILY_DIR/session_features_${CURRENT_DAY}.parquet"
  RAW_FILE="$ROOT_DIR/$DAILY_DIR/sessions_raw_${CURRENT_DAY}.parquet"

  if [[ "$SKIP_EXISTING" == "1" && -f "$FEATURE_FILE" && -f "$RAW_FILE" ]]; then
    echo "[INFO] Skip DAY=$CURRENT_DAY (already exported)"
  else
    bash "$ROOT_DIR/scripts/build_exports.sh" "$CURRENT_DAY"
  fi

  CURRENT_DAY="$(date -d "$CURRENT_DAY +1 day" +"%Y-%m-%d")"
done

# 2) Merge daily parquet files
python3 "$ROOT_DIR/scripts/merge_parquets.py" \
  --input-dir "$ROOT_DIR/$DAILY_DIR" \
  --start-day "$START_DAY" \
  --end-day "$END_DAY" \
  --prefix "session_features_" \
  --output "$ROOT_DIR/$MERGED_DIR/session_features_${START_DAY}_${END_DAY}.parquet"

python3 "$ROOT_DIR/scripts/merge_parquets.py" \
  --input-dir "$ROOT_DIR/$DAILY_DIR" \
  --start-day "$START_DAY" \
  --end-day "$END_DAY" \
  --prefix "sessions_raw_" \
  --output "$ROOT_DIR/$MERGED_DIR/sessions_raw_${START_DAY}_${END_DAY}.parquet"

echo "[INFO] Pipeline finished."
echo "[INFO] Merged files:"
echo "[INFO] - $ROOT_DIR/$MERGED_DIR/session_features_${START_DAY}_${END_DAY}.parquet"
echo "[INFO] - $ROOT_DIR/$MERGED_DIR/sessions_raw_${START_DAY}_${END_DAY}.parquet"

# 3) Optional: Run Track A on merged features
if [[ "${RUN_TRACK_A:-0}" == "1" ]]; then
  echo "[INFO] RUN_TRACK_A=1 -> running Track A on merged features"

  python3 "$ROOT_DIR/run_track_a_file.py" \
    --input "$ROOT_DIR/$MERGED_DIR/session_features_${START_DAY}_${END_DAY}.parquet" \
    --day "${START_DAY}_${END_DAY}"
fi