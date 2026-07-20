#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODEL="${1:-}"
MODE="${2:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/training/logs}"
BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_original_ability_reconstruction_v1.yaml"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"

case "$MODEL" in
  origheldout)
    MODEL_ID=M_origheldout
    RUN_NAME="h3_origheldout_s${TRAINING_SEED}"
    PROFILE=formal
    ;;
  synth5matched)
    MODEL_ID=M_synth5matched
    RUN_NAME="h3_synth5matched_s${TRAINING_SEED}"
    PROFILE=formal
    ;;
  origplus_synth5)
    MODEL_ID=M_origplus_synth5
    RUN_NAME="h3_origplus_synth5_s${TRAINING_SEED}"
    PROFILE=additive
    ;;
  *)
    echo "Usage: bash scripts/run_h3_original_ability_train.sh {origheldout|synth5matched|origplus_synth5} [preflight|smoke|fresh|train|resume]" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    REPORT="$REPORT_DIR/${RUN_NAME}_preflight.json"
    LOG="$LOG_DIR/${RUN_NAME}_preflight.log"
    EXTRA_ARGS=(--profile "$PROFILE" --preflight-only)
    ;;
  smoke)
    RUN_NAME="${RUN_NAME}_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile smoke --devices 1 --num-workers 2)
    ;;
  fresh)
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_fresh_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile "$PROFILE" --resume-policy never)
    ;;
  train)
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile "$PROFILE" --resume-policy auto)
    ;;
  resume)
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_resume_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile "$PROFILE" --resume-policy required)
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$REPORT_DIR" "$LOG_DIR" "$MPLCONFIGDIR"
"$PYTHON_BIN" scripts/train_tworoom_step1.py \
  --model-id "$MODEL_ID" \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind confirmation \
  --seed "$TRAINING_SEED" \
  --data-split-seed 3072 \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"
