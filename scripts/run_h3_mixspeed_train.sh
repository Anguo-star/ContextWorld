#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
DATA_SPLIT_SEED=3072
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/training/logs}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export MPLCONFIGDIR

case "$MODE" in
  preflight)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_s${TRAINING_SEED}}"
    RUN_KIND="${RUN_KIND:-pilot}"
    REPORT="$REPORT_DIR/${RUN_NAME}_preflight.json"
    LOG="$LOG_DIR/${RUN_NAME}_preflight.log"
    EXTRA_ARGS=(--profile formal --preflight-only)
    ;;
  ddp-smoke)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_ddp_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
    RUN_KIND=adapter_smoke
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile smoke --devices 2 --num-workers 2)
    ;;
  formal-smoke)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_4gpu_accum_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
    RUN_KIND=adapter_smoke
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(
      --profile smoke
      --devices 4
      --batch-size 128
      --accumulate-grad-batches 2
      --epoch-size 2048
      --validation-epoch-size 1024
      --num-workers 2
      --limit-train-batches 1.0
      --limit-val-batches 1
    )
    ;;
  train)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_s${TRAINING_SEED}}"
    if [[ "$TRAINING_SEED" == "3072" ]]; then
      RUN_KIND="${RUN_KIND:-pilot}"
    else
      RUN_KIND="${RUN_KIND:-confirmation}"
    fi
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    EXTRA_ARGS=(--profile formal --resume-policy auto)
    ;;
  fresh)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_s${TRAINING_SEED}}"
    if [[ "$TRAINING_SEED" == "3072" ]]; then
      RUN_KIND="${RUN_KIND:-pilot}"
    else
      RUN_KIND="${RUN_KIND:-confirmation}"
    fi
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_fresh_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile formal --resume-policy never)
    ;;
  resume)
    RUN_NAME="${RUN_NAME:-h3_mixspeed_s${TRAINING_SEED}}"
    if [[ "$TRAINING_SEED" == "3072" ]]; then
      RUN_KIND="${RUN_KIND:-pilot}"
    else
      RUN_KIND="${RUN_KIND:-confirmation}"
    fi
    REPORT="$REPORT_DIR/${RUN_NAME}.json"
    LOG="$LOG_DIR/${RUN_NAME}_resume_$(date -u +%Y%m%dT%H%M%SZ).log"
    EXTRA_ARGS=(--profile formal --resume-policy required)
    ;;
  *)
    echo "Usage: bash scripts/run_h3_mixspeed_train.sh [preflight|ddp-smoke|formal-smoke|fresh|train|resume]" >&2
    exit 2
    ;;
esac

mkdir -p "$REPORT_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

echo "[H3-MixSpeed] mode=$MODE run=$RUN_NAME train_seed=$TRAINING_SEED split_seed=$DATA_SPLIT_SEED"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id M_speed \
  --run-name "$RUN_NAME" \
  --run-kind "$RUN_KIND" \
  --seed "$TRAINING_SEED" \
  --data-split-seed "$DATA_SPLIT_SEED" \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"

echo "[H3-MixSpeed] report=$REPORT"
echo "[H3-MixSpeed] log=$LOG"
