#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# Config (edit or override via env)
# =========================
TRACKA_IN_GLOB="${TRACKA_IN_GLOB:-out/common/sessions_packed_*.parquet}"
TRACKC_ROOT="${TRACKC_ROOT:-out/track_c}"

VERSIONS_CSV="${VERSIONS_CSV:-v1}"          # e.g. v1 or v0,v1,v2
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

MIN_FREQ="${MIN_FREQ:-3}"
MAX_VOCAB="${MAX_VOCAB:-50000}"

# Train epochs
EPOCHS_C1="${EPOCHS_C1:-2}"
EPOCHS_C2="${EPOCHS_C2:-2}"
EPOCHS_C3="${EPOCHS_C3:-2}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"

# Transformer / model config
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS_TX="${N_LAYERS_TX:-4}"
N_LAYERS_LSTM="${N_LAYERS_LSTM:-2}"
N_LAYERS_AE="${N_LAYERS_AE:-2}"

SPLIT_SCORE="${SPLIT_SCORE:-test}"
TAIL_K="${TAIL_K:-128}"
KS="${KS:-50,100,200,500,1000}"
POS_OUTCOMES="${POS_OUTCOMES:-error,rate_limited,blocked}"

ONLY_CLIENT="${ONLY_CLIENT:-}"  # e.g. "Cline SR"

# GPU mapping
GPU_C1="${GPU_C1:-0}"
GPU_C2="${GPU_C2:-1}"
GPU_C3="${GPU_C3:-2}"

# Resume / Skip flags
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

client_ckpt_exists_for_model() {
  local client="$1"
  local version="$2"
  local model="$3"

  local base=""
  case "$model" in
    c1) base="$TRACKC_ROOT/$client/runs/c1_lstm_lm" ;;
    c2) base="$TRACKC_ROOT/$client/runs/c2_small_tx_lm" ;;
    c3) base="$TRACKC_ROOT/$client/runs/c3_seq_ae" ;;
    *) echo "[FATAL] unknown model: $model" >&2; exit 1 ;;
  esac

  local run_dir
  run_dir="$(latest_run_dir_by_prefix "$base" "${version}_")"
  [[ -n "${run_dir:-}" && -f "$run_dir/best.pt" ]]
}

client_ckpt_exists_all() {
  local client="$1"
  local version="$2"
  client_ckpt_exists_for_model "$client" "$version" c1 &&
  client_ckpt_exists_for_model "$client" "$version" c2 &&
  client_ckpt_exists_for_model "$client" "$version" c3
}

client_score_exists_for_model() {
  local client="$1"
  local version="$2"
  local model="$3"
  [[ -f "$TRACKC_ROOT/$client/scores/${model}_${version}_${SPLIT_SCORE}_scores.parquet" ]]
}

client_score_exists_all() {
  local client="$1"
  local version="$2"
  client_score_exists_for_model "$client" "$version" c1 &&
  client_score_exists_for_model "$client" "$version" c2 &&
  client_score_exists_for_model "$client" "$version" c3
}

log_skip() { echo "[SKIP] $*"; }
log_run()  { echo "[RUN ] $*"; }

run_bg() {
  local name="$1"
  shift
  echo "[BG  ] $name"
  "$@" &
  BG_PIDS+=("$!")
  BG_NAMES+=("$name")
}

wait_all_bg() {
  local i
  local failed=0
  for i in "${!BG_PIDS[@]}"; do
    local pid="${BG_PIDS[$i]}"
    local name="${BG_NAMES[$i]}"
    if wait "$pid"; then
      echo "[DONE] $name"
    else
      echo "[FAIL] $name" >&2
      failed=1
    fi
  done
  BG_PIDS=()
  BG_NAMES=()
  [[ "$failed" == "0" ]] || { echo "[FATAL] One or more background jobs failed." >&2; exit 1; }
}

get_ckpt_path() {
  local client="$1"
  local version="$2"
  local model="$3"

  local base=""
  case "$model" in
    c1) base="$TRACKC_ROOT/$client/runs/c1_lstm_lm" ;;
    c2) base="$TRACKC_ROOT/$client/runs/c2_small_tx_lm" ;;
    c3) base="$TRACKC_ROOT/$client/runs/c3_seq_ae" ;;
    *) echo "[FATAL] unknown model: $model" >&2; exit 1 ;;
  esac

  local run_dir
  run_dir="$(latest_run_dir_by_prefix "$base" "${version}_")"
  [[ -n "$run_dir" ]] || { echo "[FATAL] No run dir for client='$client' version='$version' model='$model'" >&2; exit 1; }

  local ckpt="$run_dir/best.pt"
  [[ -f "$ckpt" ]] || { echo "[FATAL] missing ckpt: $ckpt" >&2; exit 1; }
  echo "$ckpt"
}

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
need_cmd nvidia-smi

echo "[INFO] TRACKA_IN_GLOB=$TRACKA_IN_GLOB"
echo "[INFO] TRACKC_ROOT=$TRACKC_ROOT"
echo "[INFO] VERSIONS=$VERSIONS_CSV"
echo "[INFO] ONLY_CLIENT=${ONLY_CLIENT:-<ALL>}"
echo "[INFO] SPLIT_SCORE=$SPLIT_SCORE"
echo "[INFO] GPU map: c1->$GPU_C1 c2->$GPU_C2 c3->$GPU_C3"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# =========================
# 0) Discover / Prepare client list
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

IFS=',' read -r -a VERSIONS <<< "$VERSIONS_CSV"

# =========================
# 1) Dataset build
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

# refresh clients
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
# 2) Build vocab
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
# 3) Make ids
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
# 4) Train C1/C2/C3 in parallel across 3 GPUs
# =========================
if [[ "$SKIP_TRAIN" == "1" ]]; then
  log_skip "train step disabled by SKIP_TRAIN=1"
else
  log_run "STEP 4/6 Train C1/C2/C3 (3-GPU parallel)"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      echo "  [TRAIN-GROUP] client='$c' version='$v'"

      BG_PIDS=()
      BG_NAMES=()

      if [[ "$FORCE_TRAIN" == "1" ]] || ! client_ckpt_exists_for_model "$c" "$v" c1; then
        run_bg "train c1 client=$c version=$v gpu=$GPU_C1" \
          env CUDA_VISIBLE_DEVICES="$GPU_C1" \
          python train_c1_lstm_lm.py \
            --trackc-root "$TRACKC_ROOT" \
            --client "$c" \
            --version "$v" \
            --epochs "$EPOCHS_C1" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --d-model "$D_MODEL" \
            --n-layers "$N_LAYERS_LSTM"
      else
        log_skip "checkpoint exists: c1 client='$c' version='$v'"
      fi

      if [[ "$FORCE_TRAIN" == "1" ]] || ! client_ckpt_exists_for_model "$c" "$v" c2; then
        run_bg "train c2 client=$c version=$v gpu=$GPU_C2" \
          env CUDA_VISIBLE_DEVICES="$GPU_C2" \
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
      else
        log_skip "checkpoint exists: c2 client='$c' version='$v'"
      fi

      if [[ "$FORCE_TRAIN" == "1" ]] || ! client_ckpt_exists_for_model "$c" "$v" c3; then
        run_bg "train c3 client=$c version=$v gpu=$GPU_C3" \
          env CUDA_VISIBLE_DEVICES="$GPU_C3" \
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
        log_skip "checkpoint exists: c3 client='$c' version='$v'"
      fi

      if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
        wait_all_bg
      fi
    done
  done
fi

# =========================
# 5) Score C1/C2/C3 in parallel across 3 GPUs
# =========================
if [[ "$SKIP_SCORE" == "1" ]]; then
  log_skip "score step disabled by SKIP_SCORE=1"
else
  log_run "STEP 5/6 Score models on split='$SPLIT_SCORE' (3-GPU parallel)"
  for c in "${CLIENTS[@]}"; do
    for v in "${VERSIONS[@]}"; do
      echo "  [SCORE-GROUP] client='$c' version='$v' split='$SPLIT_SCORE'"

      BG_PIDS=()
      BG_NAMES=()

      if [[ "$FORCE_SCORE" == "1" ]] || ! client_score_exists_for_model "$c" "$v" c1; then
        c1_ckpt="$(get_ckpt_path "$c" "$v" c1)"
        run_bg "score c1 client=$c version=$v gpu=$GPU_C1" \
          env CUDA_VISIBLE_DEVICES="$GPU_C1" \
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
      else
        log_skip "score exists: c1 client='$c' version='$v' split='$SPLIT_SCORE'"
      fi

      if [[ "$FORCE_SCORE" == "1" ]] || ! client_score_exists_for_model "$c" "$v" c2; then
        c2_ckpt="$(get_ckpt_path "$c" "$v" c2)"
        run_bg "score c2 client=$c version=$v gpu=$GPU_C2" \
          env CUDA_VISIBLE_DEVICES="$GPU_C2" \
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
      else
        log_skip "score exists: c2 client='$c' version='$v' split='$SPLIT_SCORE'"
      fi

      if [[ "$FORCE_SCORE" == "1" ]] || ! client_score_exists_for_model "$c" "$v" c3; then
        c3_ckpt="$(get_ckpt_path "$c" "$v" c3)"
        run_bg "score c3 client=$c version=$v gpu=$GPU_C3" \
          env CUDA_VISIBLE_DEVICES="$GPU_C3" \
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
      else
        log_skip "score exists: c3 client='$c' version='$v' split='$SPLIT_SCORE'"
      fi

      if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
        wait_all_bg
      fi
    done
  done
fi

# =========================
# 6) Eval (CPU/serial)
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

echo "[OK] Track C 3-GPU runner completed."