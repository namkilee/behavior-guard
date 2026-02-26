#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# Config (edit or override via env)
# =========================
TRACKA_IN_GLOB="${TRACKA_IN_GLOB:-out/common/sessions_packed_*.parquet}"
TRACKC_ROOT="${TRACKC_ROOT:-out/track_c}"

VERSIONS_CSV="${VERSIONS_CSV:-v1}"          # 추천 기본: v1부터. 필요하면 v0,v1,v2
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

MIN_FREQ="${MIN_FREQ:-3}"
MAX_VOCAB="${MAX_VOCAB:-50000}"

# Train epochs (quick loop)
EPOCHS_C1="${EPOCHS_C1:-2}"
EPOCHS_C2="${EPOCHS_C2:-2}"
EPOCHS_C3="${EPOCHS_C3:-2}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"

# Transformer config
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS_TX="${N_LAYERS_TX:-4}"
N_LAYERS_LSTM="${N_LAYERS_LSTM:-2}"
N_LAYERS_AE="${N_LAYERS_AE:-2}"

SPLIT_SCORE="${SPLIT_SCORE:-test}"
TAIL_K="${TAIL_K:-128}"
KS="${KS:-50,100,200,500,1000}"
POS_OUTCOMES="${POS_OUTCOMES:-error,rate_limited,blocked}"

# Optional: run only one client
ONLY_CLIENT="${ONLY_CLIENT:-}"  # e.g. "Cline SR"

# =========================
# Helpers
# =========================
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[FATAL] missing cmd: $1" >&2; exit 1; }; }

latest_run_dir() {
  # $1 = runs base dir, returns latest (by name sorting) non-empty dir
  local base="$1"
  ls -1dt "$base"/* 2>/dev/null | head -n 1 || true
}

latest_run_dir_by_prefix() {
  # $1 = runs base dir, $2 = prefix like "v1_"
  local base="$1"
  local prefix="$2"
  ls -1dt "$base"/"${prefix}"* 2>/dev/null | head -n 1 || true
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

echo "[INFO] TRACKA_IN_GLOB=$TRACKA_IN_GLOB"
echo "[INFO] TRACKC_ROOT=$TRACKC_ROOT"
echo "[INFO] VERSIONS=$VERSIONS_CSV"
echo "[INFO] ONLY_CLIENT=${ONLY_CLIENT:-<ALL>}"

# =========================
# 1) Build Track C inputs (v0/v1/v2)
# =========================
echo "[STEP] 1/6 make_trackc_inputs_from_tracka_sessions.py"
python make_trackc_inputs_from_tracka_sessions.py \
  --in "$TRACKA_IN_GLOB" \
  --out "$TRACKC_ROOT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --versions "$VERSIONS_CSV" \
  ${ONLY_CLIENT:+--clients "$ONLY_CLIENT"}

# =========================
# 2) Build vocab
# =========================
echo "[STEP] 2/6 build_vocab.py"
python build_vocab.py \
  --trackc-root "$TRACKC_ROOT" \
  --min-freq "$MIN_FREQ" \
  --max-vocab "$MAX_VOCAB" \
  --versions "$VERSIONS_CSV" \
  ${ONLY_CLIENT:+--clients "$ONLY_CLIENT"}

# =========================
# Discover clients under trackc root
# =========================
echo "[INFO] Discovering clients..."
CLIENTS=()
while IFS= read -r d; do
  bn="$(basename "$d")"
  [[ "$bn" == _* ]] && continue
  CLIENTS+=("$bn")
done < <(find "$TRACKC_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ -n "$ONLY_CLIENT" ]]; then
  CLIENTS=("$ONLY_CLIENT")
fi

if [[ ${#CLIENTS[@]} -eq 0 ]]; then
  echo "[FATAL] No clients found under $TRACKC_ROOT" >&2
  exit 1
fi

echo "[INFO] Clients: ${CLIENTS[*]}"

# =========================
# 3) Make ids for each client/version
# =========================
echo "[STEP] 3/6 make_ids.py (client x version)"
IFS=',' read -r -a VERSIONS <<< "$VERSIONS_CSV"
for c in "${CLIENTS[@]}"; do
  for v in "${VERSIONS[@]}"; do
    echo "  - ids: client='$c' version='$v'"
    python make_ids.py --trackc-root "$TRACKC_ROOT" --client "$c" --version "$v"
  done
done

# =========================
# 4) Train C1/C2/C3 for each client/version
# =========================
echo "[STEP] 4/6 Train C1/C2/C3"
for c in "${CLIENTS[@]}"; do
  for v in "${VERSIONS[@]}"; do
    echo "  [TRAIN] client='$c' version='$v'"

    # C1
    python train_c1_lstm_lm.py \
      --trackc-root "$TRACKC_ROOT" \
      --client "$c" \
      --version "$v" \
      --epochs "$EPOCHS_C1" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --d-model "$D_MODEL" \
      --n-layers "$N_LAYERS_LSTM"

    # C2
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

    # C3
    python train_c3_seq_ae.py \
      --trackc-root "$TRACKC_ROOT" \
      --client "$c" \
      --version "$v" \
      --epochs "$EPOCHS_C3" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --d-model "$D_MODEL" \
      --n-layers "$N_LAYERS_AE"
  done
done

# =========================
# 5) Score (test/val/train) using latest checkpoints (VERSION-AWARE)
# =========================
echo "[STEP] 5/6 Score models on split='$SPLIT_SCORE' (version-aware ckpt selection)"
for c in "${CLIENTS[@]}"; do
  for v in "${VERSIONS[@]}"; do
    echo "  [SCORE] client='$c' version='$v' split='$SPLIT_SCORE'"

    # Runs base dirs (NOTE: must match train scripts)
    c1_base="$TRACKC_ROOT/$c/runs/c1_lstm_lm"
    c2_base="$TRACKC_ROOT/$c/runs/c2_small_tx_lm"
    c3_base="$TRACKC_ROOT/$c/runs/c3_seq_ae"

    # Version-aware: select latest run whose dirname starts with "${v}_"
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

    # C1
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

    # C2
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

    # C3
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

# =========================
# 6) Precision@K evaluation
# =========================
echo "[STEP] 6/6 Precision@K"
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

echo "[OK] Track C e2e completed."