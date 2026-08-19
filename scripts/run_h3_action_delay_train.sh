#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VARIANT="${1:-multi}"
MODE="${2:-formal}"
TRAINING_SEED="${3:-3072}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/training/logs}"
CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h3_training_v1.yaml"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"

case "$VARIANT" in
  single)
    MODEL_ID=H3_ActionDelay_SingleControl
    RUN_PREFIX=h3_action_delay_single_control
    ;;
  multi)
    MODEL_ID=H3_ActionDelay_Multi
    RUN_PREFIX=h3_action_delay_multi
    ;;
  *)
    echo "第一个参数必须是 single 或 multi" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    PROFILE=icl_formal
    RUN_KIND=confirmation
    EXTRA_ARGS=(--preflight-only)
    ;;
  formal)
    PROFILE=icl_formal
    RUN_KIND=confirmation
    EXTRA_ARGS=()
    ;;
  *)
    echo "第二个参数必须是 preflight 或 formal" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${MODE}_s${TRAINING_SEED}}"
REPORT="$REPORT_DIR/${RUN_NAME}.json"
LOG="$LOG_DIR/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$REPORT_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

echo "[action-delay-h3] variant=$VARIANT mode=$MODE seed=$TRAINING_SEED run=$RUN_NAME"
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
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"

echo "[action-delay-h3] report=$REPORT"
echo "[action-delay-h3] log=$LOG"
