#!/usr/bin/env bash
set -euo pipefail

# Logging uses the same environment contract as Stable-WorldModel's
# scripts/train/run_trainer.sh. The shell entry defaults to SwanLab; set
# logger_backend=none for local-only loss_trace.jsonl logging. Optional:
# swanlab_project, swanlab_workspace, swanlab_experiment_name, swanlab_id,
# swanlab_logdir, swanlab_mode, swanlab_collect_hardware,
# swanlab_hardware_monitor, and swanlab_log_hyperparams.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VARIANT="${1:-mixed}"
MODE="${2:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_SEED="${TRAINING_SEED:-3072}"
DATA_SPLIT_SEED=3072
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/training/runs}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/training/reports}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/training/logs}"
BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_hidden_passage_h3_training_v1.yaml"
ORIGINAL_H5="${CONTEXTWORLD_TWOROOM_H5:-${ORIGINAL_H5:-}}"
OBJECTIVE_ARGS=()
DIAGNOSTIC_CHECKPOINTS=0
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/contextworld-matplotlib}"
export MPLCONFIGDIR

is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

logger_backend="${logger_backend:-swanlab}"
if [[ "$logger_backend" == "swanlab" ]] && \
   ! is_true "${swanlab_enabled:-True}"; then
  logger_backend=none
fi

if [[ -n "${LOCAL_RANK+x}" ]]; then
  echo "拒绝从带 LOCAL_RANK 的外部环境启动；本脚本只负责单节点根进程，DDP 子进程由 Lightning 创建。" >&2
  exit 2
fi

for internal_var in \
  CONTEXTWORLD_H3_RANK0_ATTESTATION_V1 \
  CONTEXTWORLD_H3_RANK0_ATTESTATION_V2 \
  CONTEXTWORLD_H3_RANK0_SECRET \
  CONTEXTWORLD_H3_RANK0_ISSUER
do
  if [[ -n "${!internal_var+x}" ]]; then
    echo "拒绝继承内部 hidden-passage 启动变量: $internal_var" >&2
    exit 2
  fi
done

if [[ "$VARIANT" == "all" ]]; then
  for item in passable blocked mixed; do
    "$0" "$item" "$MODE"
  done
  exit 0
fi

case "$VARIANT" in
  passable)
    MODEL_ID=H3_Passage_PassableOnly
    RUN_PREFIX=h3_passage_passable_only
    ;;
  blocked)
    MODEL_ID=H3_Passage_BlockedOnly
    RUN_PREFIX=h3_passage_blocked_only
    ;;
  mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules
    ;;
  lewm-std-cov-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_std18_cov12
    OBJECTIVE_ARGS=(--lewm-std-weight 18 --lewm-cov-weight 12)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-sigreg-0p3-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_sigreg0p3
    OBJECTIVE_ARGS=(--lewm-sigreg-weight 0.3)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-sigreg-0p9-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_sigreg0p9
    OBJECTIVE_ARGS=(--lewm-sigreg-weight 0.9)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-sigreg-1p3-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_sigreg1p3
    OBJECTIVE_ARGS=(--lewm-sigreg-weight 1.3)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-sigreg-1p65-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_sigreg1p65
    OBJECTIVE_ARGS=(--lewm-sigreg-weight 1.65)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-sigreg-2p05-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_sigreg2p05
    OBJECTIVE_ARGS=(--lewm-sigreg-weight 2.05)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  lewm-visreg-mixed)
    MODEL_ID=H3_Passage_MixedRules
    RUN_PREFIX=h3_passage_mixed_rules_lewm_visreg0p09
    OBJECTIVE_ARGS=(--lewm-regularizer visreg --lewm-visreg-weight 0.09)
    DIAGNOSTIC_CHECKPOINTS=1
    AUDIT_CONCURRENCY=8
    ;;
  fixed-mixed)
    MODEL_ID=H3_Passage_MixedRules_FrozenRepresentation
    RUN_PREFIX=h3_passage_mixed_rules_fixed_representation_v2
    BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml"
    AUDIT_CONCURRENCY=8
    ;;
  pldm-mixed)
    MODEL_ID=H3_Passage_MixedRules_PLDMObjective
    RUN_PREFIX=h3_passage_mixed_rules_pldm_objective
    BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_hidden_passage_h3_pldm_training_v1.yaml"
    AUDIT_CONCURRENCY=8
    ;;
  pldm-fixed-mixed)
    MODEL_ID=H3_Passage_MixedRules_PLDMObjective_FrozenRepresentation
    RUN_PREFIX=h3_passage_mixed_rules_pldm_objective_fixed_representation
    BENCHMARK_CONFIG="$ROOT/configs/benchmark/tworoom_hidden_passage_h3_pldm_fixed_representation_training_v1.yaml"
    AUDIT_CONCURRENCY=8
    ;;
  *)
    echo "第一个参数必须是 passable、blocked、mixed、lewm-std-cov-mixed、lewm-sigreg-0p3-mixed、lewm-sigreg-0p9-mixed、lewm-sigreg-1p3-mixed、lewm-sigreg-1p65-mixed、lewm-sigreg-2p05-mixed、lewm-visreg-mixed、fixed-mixed、pldm-mixed、pldm-fixed-mixed 或 all" >&2
    exit 2
    ;;
esac

case "$MODE" in
  preflight)
    PROFILE=passage_pilot
    RUN_KIND=pilot
    RESUME_POLICY=never
    EXTRA_ARGS=(--preflight-only)
    ;;
  smoke)
    PROFILE=smoke
    RUN_KIND=adapter_smoke
    RESUME_POLICY=never
    EXTRA_ARGS=()
    ;;
  smoke-8gpu)
    PROFILE=smoke
    RUN_KIND=adapter_smoke
    RESUME_POLICY=never
    EXTRA_ARGS=(--devices 8 --num-workers 2)
    ;;
  pilot)
    PROFILE=passage_pilot
    RUN_KIND=pilot
    RESUME_POLICY=never
    EXTRA_ARGS=()
    ;;
  pilot-resume)
    PROFILE=passage_pilot
    RUN_KIND=pilot
    RESUME_POLICY=required
    EXTRA_ARGS=()
    ;;
  formal)
    PROFILE=passage_formal
    RUN_KIND=confirmation
    RESUME_POLICY=never
    EXTRA_ARGS=()
    ;;
  formal-resume)
    PROFILE=passage_formal
    RUN_KIND=confirmation
    RESUME_POLICY=required
    EXTRA_ARGS=()
    ;;
  *)
    echo "第二个参数必须是 preflight、smoke、smoke-8gpu、pilot、pilot-resume、formal 或 formal-resume" >&2
    exit 2
    ;;
esac

if [[ "$DIAGNOSTIC_CHECKPOINTS" == "1" ]]; then
  case "$MODE" in
    smoke|smoke-8gpu)
      DIAGNOSTIC_MAX_STEP=2
      ;;
    preflight|pilot|pilot-resume)
      DIAGNOSTIC_MAX_STEP=256
      ;;
    formal|formal-resume)
      DIAGNOSTIC_MAX_STEP=1024
      ;;
  esac
  for diagnostic_step in 1 2 4 8 16 32 64 128 256 512 1024; do
    if (( diagnostic_step <= DIAGNOSTIC_MAX_STEP )); then
      OBJECTIVE_ARGS+=(
        --diagnostic-checkpoint-step "$diagnostic_step"
      )
    fi
  done
fi

RUN_LABEL="$PROFILE"
if [[ "$MODE" == "smoke-8gpu" ]]; then
  RUN_LABEL=smoke8gpu
fi
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${RUN_LABEL}_s${TRAINING_SEED}}"
if [[ "$MODE" == "preflight" ]]; then
  REPORT="$REPORT_DIR/${RUN_NAME}_preflight.json"
else
  REPORT="$REPORT_DIR/${RUN_NAME}.json"
fi
LOG="$LOG_DIR/${RUN_NAME}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$REPORT_DIR" "$LOG_DIR" "$MPLCONFIGDIR"
ORIGINAL_ARGS=()
if [[ -n "$ORIGINAL_H5" ]]; then
  ORIGINAL_ARGS=(--original-h5 "$ORIGINAL_H5")
fi

AUDIT_CONCURRENCY="${AUDIT_CONCURRENCY:-1}"
LOGGER_ARGS=(--logger-backend "$logger_backend")
if [[ -n "${swanlab_project:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-project "$swanlab_project")
fi
if [[ -n "${swanlab_workspace:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-workspace "$swanlab_workspace")
fi
if [[ -n "${swanlab_experiment_name:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-experiment-name "$swanlab_experiment_name")
fi
if [[ -n "${swanlab_id:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-id "$swanlab_id")
fi
if [[ -n "${swanlab_logdir:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-logdir "$swanlab_logdir")
fi
if [[ -n "${swanlab_mode:-}" ]]; then
  LOGGER_ARGS+=(--swanlab-mode "$swanlab_mode")
fi
if is_true "${swanlab_collect_hardware:-False}"; then
  LOGGER_ARGS+=(--swanlab-collect-hardware)
fi
if is_true "${swanlab_hardware_monitor:-False}"; then
  LOGGER_ARGS+=(--swanlab-hardware-monitor)
fi
if is_true "${swanlab_log_hyperparams:-False}"; then
  LOGGER_ARGS+=(--swanlab-log-hyperparams)
fi

if [[ "$logger_backend" == "swanlab" ]] && \
   [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  swanlab login -k "$SWANLAB_API_KEY"
fi

echo "[hidden-passage-h3] variant=$VARIANT mode=$MODE run=$RUN_NAME seed=$TRAINING_SEED logger=$logger_backend audit_concurrency=$AUDIT_CONCURRENCY concurrent_runs_per_release=1"
"$PYTHON_BIN" "$ROOT/scripts/train_tworoom_step1.py" \
  --model-id "$MODEL_ID" \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --run-name "$RUN_NAME" \
  --run-kind "$RUN_KIND" \
  --profile "$PROFILE" \
  --resume-policy "$RESUME_POLICY" \
  --seed "$TRAINING_SEED" \
  --data-split-seed "$DATA_SPLIT_SEED" \
  --stablewm-repo "$STABLEWM_REPO" \
  --stablewm-ref "$STABLEWM_REF" \
  --audit-concurrency "$AUDIT_CONCURRENCY" \
  "${OBJECTIVE_ARGS[@]}" \
  "${LOGGER_ARGS[@]}" \
  "${ORIGINAL_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"

echo "[hidden-passage-h3] report=$REPORT"
echo "[hidden-passage-h3] log=$LOG"
