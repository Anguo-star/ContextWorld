#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

FAMILY="${1:?第一个参数必须是 lewm 或 pldm}"
TRAINING_SEED="${2:-3072}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-ad2bc44579f2b5b65c004fd2c9d8edc8ebaa43ce}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"

case "$FAMILY" in
  lewm)
    MODEL_ID=H7_ActionDelay_Core_v3_LeWM
    CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h7_core_lewm_v3.yaml"
    ;;
  pldm)
    MODEL_ID=H7_ActionDelay_Core_v3_PLDM
    CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h7_core_pldm_v3.yaml"
    ;;
  *)
    echo "第一个参数必须是 lewm 或 pldm" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-h7_action_delay_core_v3_${FAMILY}_formal_s${TRAINING_SEED}}"
REPORT="$REPORT_DIR/${RUN_NAME}.json"
mkdir -p "$REPORT_DIR" "${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[action-delay-h7-core-v3] family=$FAMILY seed=$TRAINING_SEED"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind confirmation \
  --profile icl_core_v3 \
  --resume-policy never \
  --seed "$TRAINING_SEED" \
  --data-split-seed 3072 \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --logger-backend none \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT"

echo "[action-delay-h7-core-v3] report=$REPORT"
