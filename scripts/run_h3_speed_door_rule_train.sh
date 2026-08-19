#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VARIANT="${1:-joint}"
MODE="${2:-preflight}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
CONFIG="$ROOT/configs/benchmark/tworoom_speed_door_rule_h3_training_v1.yaml"

case "$VARIANT" in
  speed)
    MODEL_ID=H3_SpeedOnly_PLDM
    RUN_PREFIX=h3_speed_only_pldm
    ;;
  joint)
    MODEL_ID=H3_SpeedDoorJoint_PLDM
    RUN_PREFIX=h3_speed_door_joint_pldm
    ;;
  *)
    echo "第一个参数必须是 speed 或 joint" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    PROFILE=passage_pilot
    RUN_KIND=pilot
    EXTRA_ARGS=(--preflight-only)
    ;;
  formal)
    PROFILE=passage_formal
    RUN_KIND=confirmation
    EXTRA_ARGS=()
    ;;
  *)
    echo "第二个参数必须是 preflight 或 formal" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${PROFILE}_s${TRAINING_SEED}}"
REPORT_SUFFIX=""
if [[ "$MODE" == "preflight" ]]; then
  REPORT_SUFFIX="_preflight"
fi
REPORT="$REPORT_DIR/${RUN_NAME}${REPORT_SUFFIX}.json"
mkdir -p "$REPORT_DIR"

python "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind "$RUN_KIND" \
  --profile "$PROFILE" \
  --resume-policy never \
  --seed "$TRAINING_SEED" \
  --data-split-seed 3072 \
  --stablewm-repo ../stable-worldmodel \
  --stablewm-ref 5864b74980f6ed328fd0045e777b3865962eff43 \
  --audit-concurrency 8 \
  --logger-backend none \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}"
