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
    MODEL_ID=H7_ActionDelay_Curriculum_v4_LeWM
    CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h7_curriculum_lewm_v4.yaml"
    ;;
  pldm)
    MODEL_ID=H7_ActionDelay_Curriculum_v4_PLDM
    CONFIG="$ROOT/configs/benchmark/tworoom_action_delay_h7_curriculum_pldm_v4.yaml"
    ;;
  *)
    echo "第一个参数必须是 lewm 或 pldm" >&2
    exit 2
    ;;
esac

case "${FAMILY}:${TRAINING_SEED}" in
  pldm:3072)
    INITIAL_SHA=f4ae94b78079e1a05e5b42df1901f516f2441958ce8b52ae6c8f958b19b5971b
    INITIAL_CONFIG_SHA=3a8f75e37724814c5907cff184c91cf57e7a6852ed216a5bb10dbb059e2078e6
    ;;
  pldm:4096)
    INITIAL_SHA=d4497f3bd9a76d4f31db0645715a8a4353eb220bfe7ee16e56116c6e2097db59
    INITIAL_CONFIG_SHA=2dda5daff2996f2037a6d59cdf218df0083f5fb98bfba2b5e9dd43b4ec6a23d8
    ;;
  pldm:5120)
    INITIAL_SHA=922b2a714614a3c8b751efa3cfaf373823f3d5860f1766fa33aed2315957196d
    INITIAL_CONFIG_SHA=fe334479dbc34b33860b69d81b362596dd51782d734366d2c3dec9799543811e
    ;;
  lewm:3072)
    INITIAL_SHA=deb594026aa0c55898aee3e50d1b959364432ddeda83ea9327b00d197acb7710
    INITIAL_CONFIG_SHA=c3c35b581c02aad4441353512544e4c2a6e1dc321fe3f3f05201189f116316ad
    ;;
  lewm:4096)
    INITIAL_SHA=c74ca7659c2551c2ca93f981676438e72b3e279903ef683e9a6d65e4f42bfb0b
    INITIAL_CONFIG_SHA=d8fa860a5db0b609160df67bd52a90e13186646744da603221909ecf76ff49e2
    ;;
  lewm:5120)
    INITIAL_SHA=29e453a94ffebc109849a71166cfbe8826f6e074c85e84041a5430e465cde522
    INITIAL_CONFIG_SHA=54db2644ceda774cc18832743e637a7dbfa91b68faf644c7d307f23beb0eed30
    ;;
  *)
    echo "训练 seed 必须是 3072、4096 或 5120" >&2
    exit 2
    ;;
esac

INITIAL_CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/h7_action_delay_paired_${FAMILY}_formal_s${TRAINING_SEED}/weights_final_step_1024.pt"
if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "第一阶段 checkpoint 不存在：$INITIAL_CHECKPOINT" >&2
  exit 1
fi
INITIAL_CONFIG="$(dirname "$INITIAL_CHECKPOINT")/config.json"
if [[ ! -f "$INITIAL_CONFIG" ]]; then
  echo "第一阶段 checkpoint 配置不存在：$INITIAL_CONFIG" >&2
  exit 1
fi
if [[ "$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')" != "$INITIAL_SHA" ]]; then
  echo "第一阶段 checkpoint 哈希不一致：$INITIAL_CHECKPOINT" >&2
  exit 1
fi
if [[ "$(sha256sum "$INITIAL_CONFIG" | awk '{print $1}')" != "$INITIAL_CONFIG_SHA" ]]; then
  echo "第一阶段 checkpoint 配置哈希不一致：$INITIAL_CONFIG" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-h7_action_delay_curriculum_v4_${FAMILY}_formal_s${TRAINING_SEED}}"
REPORT="$REPORT_DIR/${RUN_NAME}.json"
mkdir -p "$REPORT_DIR" "${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[action-delay-h7-curriculum-v4] family=$FAMILY seed=$TRAINING_SEED"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind confirmation \
  --profile icl_core_v3 \
  --resume-policy never \
  --initialization-checkpoint "$INITIAL_CHECKPOINT" \
  --initialization-checkpoint-sha256 "$INITIAL_SHA" \
  --seed "$TRAINING_SEED" \
  --data-split-seed 3072 \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --logger-backend none \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT"

echo "[action-delay-h7-curriculum-v4] report=$REPORT"
