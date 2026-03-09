#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# Config (edit or override via env)
# =========================
TRACKA_IN_GLOB="${TRACKA_IN_GLOB:-out/common/sessions_packed_*.parquet}"
TRACKC_ROOT="${TRACKC_ROOT:-out/track_c}"

VERSIONS_CSV="${VERSIONS_CSV:-v1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

MIN_FREQ="${MIN_FREQ:-3}"
MAX_VOCAB="${MAX_VOCAB:-50000}"

EPOCHS_C1="${EPOCHS_C1:-2}"
EPOCHS_C2="${EPOCHS_C2:-2}"
EPOCHS_C3="${EPOCHS_C3:-2}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"

D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS_TX="${N_LAYERS_TX:-4}"
N_LAYERS_LSTM="${N_LAYERS_LSTM:-2}"
N_LAYERS_AE="${N_LAYERS_AE:-2}"

SPLIT_SCORE="${SPLIT_SCORE:-test}"
TAIL_K="${TAIL_K:-128}"
KS="${KS:-50,100,200,500,1000}"
POS_OUTCOMES="${POS_OUTCOMES:-error,rate_limited,blocked}"

ONLY_CLIENT="${ONLY_CLIENT:-}"   # e.g. "Cline SR"

# =========================
# Skip / Resume flags
# =========================
FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_VOCAB="${FORCE_VOCAB:-0}"
FORCE_IDS="${FORCE_IDS:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_SCORE="${FORCE_SCORE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_VOCAB="${SKIP_VOCAB:-0}"
SKIP_IDS="${SKIP_IDS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SCORE="${SKIP_SCORE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# =========================
# Helpers
# =========================
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[FATAL] missing cmd: $1" >&2; exit 1; }; }

latest_run_dir_by_prefix() {
  # $1 = runs base dir, $2 = prefix like "v1_"
  local base="$1"
  local prefix="$2"
  ls -1dt "$base"/"${prefix}"* 2>/dev/null | head -n 1 || true
}

have_any_match() {
  local pattern="$1"
  compgen -G "$pattern" > /dev/null 2>&1
}

client_dataset_exists() {
  local client="$1"
  local version="$2"
  have_any_match "$TRACKC_ROOT/$client/dataset/*${version}*.parquet"
}

client_vocab_exists() {
  local client="$1"
  local version="$2"
  have_any_match "$TRACKC_ROOT/$client/vocab/*${version}*" || \
  have_any_match "$TRACKC_ROOT/$client/*vocab*/*${version}*" || \
  have_any_match "$TRACKC_ROOT/$client/vocab/*"
}

client_ids_exists() {
  local client="$1"
  local version="$2"
  [[ -f "$TRACKC_ROOT/$client/dataset/sessions_ids_${version}.parquet" ]]
}

client_ckpt_exists() {
  local client="$1"
  local version="$2"
  local c1_base="$TRACKC_ROOT/$client/runs/c1_lstm_lm"
  local c2_base="$TRACKC_ROOT/$client/runs/c2_small_tx_lm"
  local c3_base="$TRACKC_ROOT/$client/runs/c3_seq_ae"

  local c1_run c2_run c3_run
  c1_run="$(latest_run_dir_by_prefix "$c1_base" "${version}_")"
  c2_run="$(latest_run_dir_by_prefix "$c2_base" "${version}_")"
  c3_run="$(latest_run_dir_by_prefix "$c3_base" "${version}_")"

  [[ -n "${c1_run:-}" && -f "$c1_run/best.pt" ]] &&
  [[ -n "${c2_run:-}" && -f "$c2_run/best.pt" ]] &&
  [[ -n "${c3_run:-}" && -f "$c3_run/best.pt" ]]
}

client_score_exists() {
  local client="$1"
  local version="$2"
  [[ -f "$TRACKC_ROOT/$client/scores/c1_${version}_${SPLIT_SCORE}_scores.parquet" ]] &&
  [[ -f "$TRACKC_ROOT/$client/scores/c2_${version}_${SPLIT_SCORE}_scores.parquet" ]] &&
  [[ -f "$TRACKC_ROOT/$client/scores/c3_${version}_${SPLIT_SCORE}_scores.parquet" ]]
}

log_skip() { echo "[SKIP] $*"; }
log_run()  { echo "[RUN ] $*"; }

# =========================
# Preconditions
# =========================
need_cmd python
need_cmd ls
need_cmd awk
need_cmd sed
need_cmd find
need_cmd sort
need_cmd basename

echo "[INFO] TRACKA_IN_GLOB=$TRACKA_IN_GLOB"
echo "[INFO] TRACKC_ROOT=$TRACKC_ROOT"
echo "[INFO] VERSIONS=$VERSIONS_CSV"
echo "[INFO] ONLY_CLIENT=${ONLY_CLIENT:-<ALL>}"
echo "[INFO] SPLIT_SCORE=$SPLIT_SCORE"

# =========================
# 1) Discover / Prepare client list
# =========================
CLIENTS=()

if [[ -n "$ONLY_CLIENT" ]]; then
  CLIENTS=("$ONLY_CLIENT")
else
  if [[ -d "$TRACKC_ROOT" ]]; then
    while IFS= read -r d; do
      bn="$(basename "$d")"
      [[ "$bn" == _* ]] && continue
      CLIENTS+=("$bn")
    done < <(find "$TRACKC_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
fi

# If no clients yet, we may need step 1 first. That's okay.
IFS=',' read -r -a VERSIONS <<< "$VERSIONS_CSV"

# =========================
# 2) Dataset build
# =========================
need_dataset_build=0
if [[ "$SKIP_DATASET" == "1" ]]; then
  log_skip "dataset step disabled by SKIP_DATASET=1"
else
  if [[ "$FORCE_DATASET" == "1" ]]; then
    need_dataset_build=1
  else
    if [[ ${#CLIENTS[@]} -eq 0 ]]; then
      need_dataset_build=1
    else
      for c in "${CLIENTS[@]}"; do
        for v in "${VERSIONS[@]}"; do
          if ! client_dataset_exists "$c" "$v"; then
            need_dataset_build=1
            break 2
          fi
        done
      done
    fi
  fi
fi

if [[ "$need_dataset_build" == "1" ]]; then
  log_run "STEP 1/6 make_trackc_inputs_from_tracka_sessions.py"
  python make_trackc_inputs_from_tracka_sessions.py \
    --in "$TRACKA_IN_GLOB" \
    --out "$TRACKC_ROOT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --versions "$VERSIONS_CSV" \
    ${ONLY_CLIENT:+--clients "$ONLY_CLIENT"}
else
  log_skip "dataset already exists"
fi

# Refresh clients after dataset build
if [[ -z "$ONLY_CLIENT" ]]; then
  CLIENTS=()
  while IFS= read -r d; do
    bn="$(basename "$d")"
    [[ "$bn" == _* ]] && continue
    CLIENTS+=("$bn")
  done < <(find "$TRACKC_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
else
  CLIENTS=("$ONLY_CLIENT")
fi

if [[ ${#CLIENTS[@]} -eq 0 ]]; then
  echo "[FATAL] No clients found under $TRACKC_ROOT" >&2
  exit 1
fi

echo "[INFO] Clients: ${CLIENTS[*]}"

# =========================
# 3) Build vocab
# =========================
need_vocab_build=0
if [[ "$SKIP_VOCAB" == "1" ]]; then
  log_skip "vocab step disabled by SKIP_VOCAB=1"
else
  if [[ "$FORCE_VOCAB" == "1" ]]; then
    need_vocab_build=1
  else
    for c in "${CLIENTS[@]}"; do
      for v in "${VERSIONS[@]}"; do
        if ! client_vocab_exists "$c" "$v"; then
          need_vocab_build=1
          break 2
        fi
      done
    done
  fi
fi

if [[ "$need_vocab_build" == "1" ]]; then
  log_run "STEP 2/6 build_vocab.py"
  python build_vocab.py \
    --trackc-root "$TRACKC_ROOT" \
    --min-freq "$MIN_FREQ" \
    --max-vocab "$MAX_VOCAB" \
    --versions "$VERSIONS_CSV" \
    ${ONLY_CLIENT:+--clients "$ONLY_CLIENT"}
else
  log_skip "vocab already exists"
fi

# =========================
# 4) Make ids
# =========================
if [[ "$SKIP_IDS" == "1" ]]; then
  log_skip "ids step disabled by SKIP_IDS=1"
else
  log_run "STEP 3/6 make_ids.py (client x version)"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      if [[ "$FORCE_IDS" == "1" ]] || ! client_ids_exists "$c" "$v"; then
        echo "  - ids: client='$c' version='$v'"
        python make_ids.py --trackc-root "$TRACKC_ROOT" --client "$c" --version "$v"
      else
        log_skip "ids exists: client='$c' version='$v'"
      fi
    done
  done
fi

# =========================
# 5) Train C1/C2/C3
# =========================
if [[ "$SKIP_TRAIN" == "1" ]]; then
  log_skip "train step disabled by SKIP_TRAIN=1"
else
  log_run "STEP 4/6 Train C1/C2/C3"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      if [[ "$FORCE_TRAIN" == "1" ]] || ! client_ckpt_exists "$c" "$v"; then
        echo "  [TRAIN] client='$c' version='$v'"

        python train_c1_lstm_lm.py \
          --trackc-root "$TRACKC_ROOT" \
          --client "$c" \
          --version "$v" \
          --epochs "$EPOCHS_C1" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --d-model "$D_MODEL" \
          --n-layers "$N_LAYERS_LSTM"

        python train_c2_small_tx_lm.py \
          --trackc-root "$TRACKC_ROOT" \
          --client "$c" \
          --version "$v" \
          --epochs "$EPOCHS_C2" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --d-model "$D_MODEL" \
          --n-heads "$N_HEADS" \
          --n-layers "$N_LAYERS_TX" \
          --max-len "$MAX_SEQ_LEN"

        python train_c3_seq_ae.py \
          --trackc-root "$TRACKC_ROOT" \
          --client "$c" \
          --version "$v" \
          --epochs "$EPOCHS_C3" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --d-model "$D_MODEL" \
          --n-layers "$N_LAYERS_AE"
      else
        log_skip "checkpoints already exist: client='$c' version='$v'"
      fi
    done
  done
fi

# =========================
# 6) Score
# =========================
if [[ "$SKIP_SCORE" == "1" ]]; then
  log_skip "score step disabled by SKIP_SCORE=1"
else
  log_run "STEP 5/6 Score models on split='$SPLIT_SCORE'"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      if [[ "$FORCE_SCORE" != "1" ]] && client_score_exists "$c" "$v"; then
        log_skip "scores already exist: client='$c' version='$v' split='$SPLIT_SCORE'"
        continue
      fi

      echo "  [SCORE] client='$c' version='$v' split='$SPLIT_SCORE'"

      c1_base="$TRACKC_ROOT/$c/runs/c1_lstm_lm"
      c2_base="$TRACKC_ROOT/$c/runs/c2_small_tx_lm"
      c3_base="$TRACKC_ROOT/$c/runs/c3_seq_ae"

      c1_run="$(latest_run_dir_by_prefix "$c1_base" "${v}_")"
      c2_run="$(latest_run_dir_by_prefix "$c2_base" "${v}_")"
      c3_run="$(latest_run_dir_by_prefix "$c3_base" "${v}_")"

      [[ -n "$c1_run" ]] || { echo "[FATAL] No run dir for client='$c' version='$v' under $c1_base" >&2; exit 1; }
      [[ -n "$c2_run" ]] || { echo "[FATAL] No run dir for client='$c' version='$v' under $c2_base" >&2; exit 1; }
      [[ -n "$c3_run" ]] || { echo "[FATAL] No run dir for client='$c' version='$v' under $c3_base" >&2; exit 1; }

      c1_ckpt="$c1_run/best.pt"
      c2_ckpt="$c2_run/best.pt"
      c3_ckpt="$c3_run/best.pt"

      [[ -f "$c1_ckpt" ]] || { echo "[FATAL] missing ckpt: $c1_ckpt" >&2; exit 1; }
      [[ -f "$c2_ckpt" ]] || { echo "[FATAL] missing ckpt: $c2_ckpt" >&2; exit 1; }
      [[ -f "$c3_ckpt" ]] || { echo "[FATAL] missing ckpt: $c3_ckpt" >&2; exit 1; }

      echo "    [CKPT] c1=$c1_ckpt"
      echo "    [CKPT] c2=$c2_ckpt"
      echo "    [CKPT] c3=$c3_ckpt"

      python score_trackc_models.py \
        --trackc-root "$TRACKC_ROOT" \
        --client "$c" \
        --version "$v" \
        --model c1 \
        --ckpt "$c1_ckpt" \
        --split "$SPLIT_SCORE" \
        --batch-size "$BATCH_SIZE" \
        --tail-k "$TAIL_K" \
        --d-model "$D_MODEL" \
        --n-layers "$N_LAYERS_LSTM"

      python score_trackc_models.py \
        --trackc-root "$TRACKC_ROOT" \
        --client "$c" \
        --version "$v" \
        --model c2 \
        --ckpt "$c2_ckpt" \
        --split "$SPLIT_SCORE" \
        --batch-size "$BATCH_SIZE" \
        --tail-k "$TAIL_K" \
        --d-model "$D_MODEL" \
        --n-heads "$N_HEADS" \
        --n-layers "$N_LAYERS_TX" \
        --max-len "$MAX_SEQ_LEN"

      python score_trackc_models.py \
        --trackc-root "$TRACKC_ROOT" \
        --client "$c" \
        --version "$v" \
        --model c3 \
        --ckpt "$c3_ckpt" \
        --split "$SPLIT_SCORE" \
        --batch-size "$BATCH_SIZE" \
        --tail-k "$TAIL_K" \
        --d-model "$D_MODEL" \
        --n-layers "$N_LAYERS_AE"
    done
  done
fi

# =========================
# 7) Eval
# =========================
if [[ "$SKIP_EVAL" == "1" ]]; then
  log_skip "eval step disabled by SKIP_EVAL=1"
else
  log_run "STEP 6/6 Precision@K"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      for m in c1 c2 c3; do
        scores_path="$TRACKC_ROOT/$c/scores/${m}_${v}_${SPLIT_SCORE}_scores.parquet"
        [[ -f "$scores_path" ]] || { echo "[WARN] missing scores: $scores_path (skip)"; continue; }

        if [[ "$FORCE_EVAL" != "1" ]]; then
          # eval output file 여부를 모르는 상태라 skip 기준은 두지 않고 항상 eval 수행
          :
        fi

        echo "  [EVAL] client='$c' version='$v' model='$m'"
        python eval_precision_at_k.py \
          --tracka-in "$TRACKA_IN_GLOB" \
          --scores "$scores_path" \
          --client "$c" \
          --ks "$KS" \
          --pos-outcomes "$POS_OUTCOMES"
      done
    done
  done
fi

echo "[OK] Track C resume runner completed."