#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

ROLE="${1:?第一个参数必须是 speed_only、door_only 或 joint}"
MODE="${2:-preflight}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
PROTOCOL_ROOT="$ARTIFACT_ROOT/protocols/speed_door_rule_h3_v2"

case "$ROLE" in
  speed_only)
    MODEL_ID=H3_SpeedDoorV2_SpeedOnly_PLDM
    RUN_PREFIX=h3_sdr_v2_speed_only_pldm
    CONFIG="$PROTOCOL_ROOT/speed_only_training.yaml"
    ;;
  door_only)
    MODEL_ID=H3_SpeedDoorV2_DoorOnly_PLDM
    RUN_PREFIX=h3_sdr_v2_door_only_pldm
    CONFIG="$PROTOCOL_ROOT/door_only_training.yaml"
    ;;
  joint)
    MODEL_ID=H3_SpeedDoorV2_Joint_PLDM
    RUN_PREFIX=h3_sdr_v2_joint_pldm
    CONFIG="$PROTOCOL_ROOT/joint_training.yaml"
    ;;
  *)
    echo "第一个参数必须是 speed_only、door_only 或 joint" >&2
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
  --stablewm-ref 4b6f5d94693631ce64ed1f561dd0bc1d23ca38fa \
  --audit-concurrency 8 \
  --logger-backend none \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}"
