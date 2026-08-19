#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VARIANT="${1:-original}"
TRAINING_SEED="${2:-3072}"
DEVICE="${3:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CHECKPOINT_ROOT="$ARTIFACT_ROOT/training/runs/checkpoints"
REPORT_ROOT="$ARTIFACT_ROOT/training/reports"
RESULT_ROOT="$ARTIFACT_ROOT/evaluation/history3/action_delay_validation_v1/results"
mkdir -p "$RESULT_ROOT"

case "$VARIANT" in
  original)
    MODEL_ID=H3_Original_LEWM
    CHECKPOINT="$CHECKPOINT_ROOT/h3_origheldout_s3072/weights_final_step_6420.pt"
    TRAINING_REPORT="$REPORT_ROOT/h3_origheldout_s3072.json"
    OUTPUT="$RESULT_ROOT/original_s3072.json"
    EXPECTED_STEPS=6420
    ;;
  single)
    MODEL_ID=H3_ActionDelay_SingleControl
    CHECKPOINT="$CHECKPOINT_ROOT/h3_action_delay_single_control_formal_s${TRAINING_SEED}/weights_final_step_1024.pt"
    TRAINING_REPORT="$REPORT_ROOT/h3_action_delay_single_control_formal_s${TRAINING_SEED}.json"
    OUTPUT="$RESULT_ROOT/single_s${TRAINING_SEED}.json"
    EXPECTED_STEPS=1024
    ;;
  multi)
    MODEL_ID=H3_ActionDelay_Multi
    CHECKPOINT="$CHECKPOINT_ROOT/h3_action_delay_multi_formal_s${TRAINING_SEED}/weights_final_step_1024.pt"
    TRAINING_REPORT="$REPORT_ROOT/h3_action_delay_multi_formal_s${TRAINING_SEED}.json"
    OUTPUT="$RESULT_ROOT/multi_s${TRAINING_SEED}.json"
    EXPECTED_STEPS=1024
    ;;
  *)
    echo "第一个参数必须是 original、single 或 multi" >&2
    exit 2
    ;;
esac

echo "[action-delay-h3-eval] variant=$VARIANT seed=$TRAINING_SEED device=$DEVICE"
"$PYTHON_BIN" "$ROOT/scripts/eval_tworoom_action_delay_h3.py" \
  --model-id "$MODEL_ID" \
  --training-seed "$TRAINING_SEED" \
  --checkpoint "$CHECKPOINT" \
  --training-report "$TRAINING_REPORT" \
  --expected-training-steps "$EXPECTED_STEPS" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --batch-size 128
