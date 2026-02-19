#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }

# .env loader (KEY=VALUE, ignores blanks/comments)
load_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || die ".env not found at: $env_file"
  # shellcheck disable=SC2163
  export $(grep -vE '^\s*#|^\s*$' "$env_file" | sed -E 's/\r$//' | xargs -d '\n') || true
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

DAY="${1:-${DAY:-}}"
[[ -n "$DAY" ]] || die "DAY is required. Usage: ./pipeline.sh YYYY-MM-DD  (or set DAY in .env)"
[[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "Invalid DAY format: $DAY"

load_env "$ENV_FILE"

# (선택) 공통 변수 미리 체크 - 여기서 한 번만 검증해도 됨
: "${CH_HOST:?Missing CH_HOST in .env}"
: "${CH_PORT:?Missing CH_PORT in .env}"
: "${CH_USER:?Missing CH_USER in .env}"
: "${CH_PASSWORD:?Missing CH_PASSWORD in .env}"
: "${CH_DATABASE:?Missing CH_DATABASE in .env}"

echo "[INFO] Pipeline DAY=$DAY"
echo "[INFO] Using ENV_FILE=$ENV_FILE"

# 1) SQL build (built sql 생성)
bash "$ROOT_DIR/scripts/build_sql.sh"

# 2) Export parquet (DAY 필요)
bash "$ROOT_DIR/scripts/build_exports.sh" "$DAY"

# 3) Run Track A (선택)
# python "$ROOT_DIR/run_track_a_file.py" --day "$DAY"
