#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-check}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CHECKPOINT="${CHECKPOINT:-$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedseen_s3072/weights_final_step_6420.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/evaluation/history3/h3_speedseen_s3072}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_SEEDS=(42 43 44 45 46 47)

COMMON=(
  --checkpoint "$CHECKPOINT"
  --stablewm-repo "$STABLEWM_REPO"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

run_e1() {
  "$PYTHON_BIN" scripts/run_tworoom_history3_eval.py \
    --suite e1-paired-pred \
    "${COMMON[@]}"
}

run_id() {
  "$PYTHON_BIN" scripts/run_tworoom_id_retention.py \
    --checkpoint "$CHECKPOINT" \
    --stablewm-repo "$STABLEWM_REPO" \
    --output-dir "$OUTPUT_DIR/id_retention" \
    --output "$OUTPUT_DIR/id_retention_n50x6.json" \
    --eval-seeds "${EVAL_SEEDS[@]}" \
    --num-eval 50 \
    --devices 0 1 2 3 \
    --reuse-existing
}

run_e4() {
  extra=()
  if [[ "${SKIP_CATALOG_REPLAY:-0}" == "1" ]]; then
    extra+=(--skip-catalog-replay)
  fi
  "$PYTHON_BIN" scripts/run_tworoom_history3_eval.py \
    --suite e4-context-plan \
    "${COMMON[@]}" \
    --eval-seeds "${EVAL_SEEDS[@]}" \
    --num-eval 50 \
    --reuse-existing \
    "${extra[@]}"
}

case "$MODE" in
  check)
    "$PYTHON_BIN" scripts/run_tworoom_history3_eval.py \
      --suite check \
      "${COMMON[@]}"
    ;;
  e1)
    run_e1
    ;;
  id)
    run_id
    ;;
  e4)
    run_e4
    ;;
  all)
    run_e1
    run_id
    run_e4
    ;;
  *)
    echo "Usage: bash scripts/run_h3_speedseen_eval.sh [check|e1|id|e4|all]" >&2
    exit 2
    ;;
esac
