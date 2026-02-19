#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd(){ command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

need_cmd perl
need_cmd cat
need_cmd mkdir
need_cmd grep

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override 가능 (pipeline/env에서 통제)
SQL_DIR="${SQL_DIR:-$ROOT_DIR/sql}"
BUILD_DIR="${SQL_BUILD_DIR:-$SQL_DIR/build}"

mkdir -p "$BUILD_DIR"

SESSIONS_QUERY_FILE="${SESSIONS_QUERY_FILE:-$SQL_DIR/sessions_query.sql}"
RAW_TEMPLATE_FILE="${RAW_TEMPLATE_FILE:-$SQL_DIR/export_sessions_raw.sql}"
FEAT_TEMPLATE_FILE="${FEAT_TEMPLATE_FILE:-$SQL_DIR/export_session_features.sql}"

[[ -f "$SESSIONS_QUERY_FILE" ]] || die "Missing: $SESSIONS_QUERY_FILE"
[[ -f "$RAW_TEMPLATE_FILE" ]] || die "Missing: $RAW_TEMPLATE_FILE"
[[ -f "$FEAT_TEMPLATE_FILE" ]] || die "Missing: $FEAT_TEMPLATE_FILE"

SESSIONS_QUERY="$(cat "$SESSIONS_QUERY_FILE")"
[[ -n "$SESSIONS_QUERY" ]] || die "sessions_query.sql is empty: $SESSIONS_QUERY_FILE"

OUT_RAW="$BUILD_DIR/export_sessions_raw.built.sql"
OUT_FEAT="$BUILD_DIR/export_session_features.built.sql"

# Replace placeholder safely (multi-line safe)
SESSIONS_QUERY="$SESSIONS_QUERY" perl -0777 -pe '
  BEGIN { die "SESSIONS_QUERY env missing/empty\n" unless defined $ENV{SESSIONS_QUERY} && length $ENV{SESSIONS_QUERY}; }
  s{/\*__SESSIONS_QUERY__\*/}{$ENV{SESSIONS_QUERY}}g;
' "$RAW_TEMPLATE_FILE" > "$OUT_RAW"

SESSIONS_QUERY="$SESSIONS_QUERY" perl -0777 -pe '
  BEGIN { die "SESSIONS_QUERY env missing/empty\n" unless defined $ENV{SESSIONS_QUERY} && length $ENV{SESSIONS_QUERY}; }
  s{/\*__SESSIONS_QUERY__\*/}{$ENV{SESSIONS_QUERY}}g;
' "$FEAT_TEMPLATE_FILE" > "$OUT_FEAT"

# quick sanity: placeholder leftovers + 실제로 들어갔는지 체크
if grep -q '/\*__SESSIONS_QUERY__\*/' "$OUT_RAW" "$OUT_FEAT"; then
  die "Placeholder not replaced in built SQL (/*__SESSIONS_QUERY__*/ remains)."
fi

echo "[INFO] Built:"
echo " - $OUT_RAW"
echo " - $OUT_FEAT"
