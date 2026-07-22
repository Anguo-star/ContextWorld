#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-preflight}"
VARIANT="${MODEL_VARIANT:-multi}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
DATA_SPLIT_SEED=3072
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/training/logs}"
BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_door_training_v2.yaml"
ORIGINAL_H5="${CONTEXTWORLD_TWOROOM_H5:-${ORIGINAL_H5:-}}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export MPLCONFIGDIR

case "$VARIANT" in
  fixed)
    MODEL_ID=M_door_fixed49_v2
    RUN_PREFIX=h3_door_fixed49_v2
    ;;
  multi)
    MODEL_ID=M_door_multi_v2
    RUN_PREFIX=h3_door_multi_v2
    ;;
  *)
    echo "MODEL_VARIANT must be fixed or multi" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_s${TRAINING_SEED}}"
    RUN_KIND=confirmation
    REPORT="$REPORT_DIR/${RUN_NAME}_preflight.json"
    LOG="$LOG_DIR/${RUN_NAME}_preflight.log"
    EXTRA_ARGS=(--profile additive --preflight-only)
    ;;
  smoke)
    RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
    RUN_KIND=adapter_smoke
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile smoke --devices 1 --num-workers 2)
    ;;
  fresh)
    RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_s${TRAINING_SEED}}"
    RUN_KIND=confirmation
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_fresh_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile additive --resume-policy never)
    ;;
  train)
    RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_s${TRAINING_SEED}}"
    RUN_KIND=confirmation
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile additive --resume-policy auto)
    ;;
  resume)
    RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_s${TRAINING_SEED}}"
    RUN_KIND=confirmation
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_resume_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile additive --resume-policy required)
    ;;
  *)
    echo "Usage: bash scripts/run_h3_door_train.sh [preflight|smoke|fresh|train|resume]" >&2
    exit 2
    ;;
esac

mkdir -p "$REPORT_DIR" "$LOG_DIR" "$MPLCONFIGDIR"
ORIGINAL_ARGS=()
if [[ -n "$ORIGINAL_H5" ]]; then
  ORIGINAL_ARGS=(--original-h5 "$ORIGINAL_H5")
fi

echo "[door-v2] variant=$VARIANT mode=$MODE run=$RUN_NAME seed=$TRAINING_SEED"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind "$RUN_KIND" \
  --seed "$TRAINING_SEED" \
  --data-split-seed "$DATA_SPLIT_SEED" \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  "${ORIGINAL_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"

echo "[door-v2] report=$REPORT"
echo "[door-v2] log=$LOG"
