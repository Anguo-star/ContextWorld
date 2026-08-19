#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VARIANT="${1:-multi}"
MODE="${2:-preflight}"
TRAINING_SEED="${3:-3072}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-ad2bc44579f2b5b65c004fd2c9d8edc8ebaa43ce}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h7_training_v1.yaml"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

case "$VARIANT" in
  original)
    MODEL_ID=H7_OriginalOnly
    RUN_PREFIX=h7_action_delay_original_only
    ;;
  single)
    MODEL_ID=H7_ActionDelay_SingleControl
    RUN_PREFIX=h7_action_delay_single_control
    ;;
  multi)
    MODEL_ID=H7_ActionDelay_Multi
    RUN_PREFIX=h7_action_delay_multi
    ;;
  *)
    echo "第一个参数必须是 original、single 或 multi" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    PROFILE=icl_formal
    RUN_KIND=confirmation
    EXTRA_ARGS=(--preflight-only)
    ;;
  memory-smoke)
    PROFILE=smoke
    RUN_KIND=adapter_smoke
    EXTRA_ARGS=(
      --epoch-size 2048
      --validation-epoch-size 1024
      --max-epochs 1
      --batch-size 128
      --num-workers 2
      --devices 8
      --precision bf16-mixed
      --accumulate-grad-batches 1
      --limit-train-batches 2
      --limit-val-batches 1
      --expected-optimizer-steps 2
    )
    ;;
  formal)
    PROFILE=icl_formal
    RUN_KIND=confirmation
    EXTRA_ARGS=()
    ;;
  *)
    echo "第二个参数必须是 preflight、memory-smoke 或 formal" >&2
    exit 2
    ;;
esac

MODE_SLUG="${MODE//-/_}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${MODE_SLUG}_s${TRAINING_SEED}}"
REPORT="$REPORT_DIR/${RUN_NAME}.json"
mkdir -p "$REPORT_DIR" "$MPLCONFIGDIR"

echo "[action-delay-h7] variant=$VARIANT mode=$MODE seed=$TRAINING_SEED"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind "$RUN_KIND" \
  --profile "$PROFILE" \
  --resume-policy never \
  --seed "$TRAINING_SEED" \
  --data-split-seed 3072 \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --logger-backend none \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}"

echo "[action-delay-h7] report=$REPORT"
