#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODEL="${1:-}"
SUITE="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
EVAL_ROOT="$ARTIFACT_ROOT/evaluation/history3/original_ability_reconstruction"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
ORIGINAL_CATALOG="$EVAL_ROOT/original_heldout_eval_catalog.json"
MATCHED_CATALOG="$EVAL_ROOT/speed5_matched_eval_catalog.json"
ROLLOUT_CATALOG="$EVAL_ROOT/rollout_catalog.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3}"
SEEDS=(42 43 44 45 46 47)

case "$MODEL" in
  origheldout)
    SLUG=h3_origheldout_s3072
    CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/$SLUG/weights_final_step_6420.pt"
    ;;
  synth5matched)
    SLUG=h3_synth5matched_s3072
    CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/$SLUG/weights_final_step_6420.pt"
    ;;
  origplus_synth5)
    SLUG=h3_origplus_synth5_s3072
    CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/$SLUG/weights_final_step_12840.pt"
    ;;
  speedfull)
    SLUG=h3_speedfull_s3072
    CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/$SLUG/weights_final_step_12840.pt"
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_original_ability_eval.sh {origheldout|synth5matched|origplus_synth5|speedfull} {planning-original|planning-matched|rollout|e1|e4-noctx|e4}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$EVAL_ROOT/$SLUG"
mkdir -p "$OUTPUT_DIR"

run_planning_seed() {
  local seed="$1"
  local device="$2"
  local catalog="$3"
  local stem="$4"
  local output="$OUTPUT_DIR/${stem}_s${seed}.json"
  local log="$OUTPUT_DIR/${stem}_s${seed}.log"
  echo "[$SLUG $stem] seed=$seed physical_device=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_ability_catalog.py \
    --catalog "$catalog" \
    --checkpoint "$CHECKPOINT" \
    --normalizer "$NORMALIZER" \
    --output "$output" \
    --seed "$seed" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 \
    --eval-budget 50 \
    --horizon 5 \
    --receding-horizon 5 \
    --cem-samples 300 \
    --cem-steps 30 \
    --cem-topk 30 \
    >"$log" 2>&1
}

run_planning_suite() {
  local catalog="$1"
  local stem="$2"
  for ((offset=0; offset<${#SEEDS[@]}; offset+=${#DEVICES[@]})); do
    local pids=()
    local labels=()
    for ((index=0; index<${#DEVICES[@]}; index++)); do
      local position=$((offset + index))
      if [[ "$position" -ge "${#SEEDS[@]}" ]]; then
        break
      fi
      local seed="${SEEDS[$position]}"
      local device="${DEVICES[$index]}"
      run_planning_seed "$seed" "$device" "$catalog" "$stem" &
      pids+=("$!")
      labels+=("seed=$seed/device=$device")
    done
    for ((index=0; index<${#pids[@]}; index++)); do
      if ! wait "${pids[$index]}"; then
        echo "[$SLUG $stem] failed ${labels[$index]}" >&2
        exit 1
      fi
      echo "[$SLUG $stem] completed ${labels[$index]}"
    done
  done
}

case "$SUITE" in
  planning-original)
    run_planning_suite "$ORIGINAL_CATALOG" planning_original_heldout
    ;;
  planning-matched)
    run_planning_suite "$MATCHED_CATALOG" planning_speed5_matched
    ;;
  rollout)
    device="${DEVICES[0]}"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_rollout_error.py \
      --catalog "$ROLLOUT_CATALOG" \
      --checkpoint "$CHECKPOINT" \
      --normalizer "$NORMALIZER" \
      --output "$OUTPUT_DIR/rollout_error.json" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --batch-size 16 \
      2>&1 | tee "$OUTPUT_DIR/rollout_error.log"
    ;;
  e1)
    device="${DEVICES[0]}"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_icl.py \
      --catalog "$ARTIFACT_ROOT/evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json" \
      --checkpoint "$CHECKPOINT" \
      --original-h5 /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/tworoom.h5 \
      --normalizer "$NORMALIZER" \
      --output "$OUTPUT_DIR/e1_speed_paired.json" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --encode-batch-size 64 \
      --predictor-batch-size 128 \
      --seed 3072 \
      --family speed \
      2>&1 | tee "$OUTPUT_DIR/e1_speed_paired.log"
    ;;
  e4-noctx)
    run_e4_noctx_seed() {
      local seed="$1"
      local device="$2"
      local replay_args=()
      if [[ "$seed" != "42" ]]; then
        replay_args+=(--skip-catalog-replay)
      fi
      CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_icl_nocontext_planning.py \
        --checkpoint "$CHECKPOINT" \
        --normalizer "$NORMALIZER" \
        --output "$OUTPUT_DIR/e4_speed_noctx_n50_s${seed}.json" \
        --stablewm-repo "$STABLEWM_REPO" \
        --stablewm-ref "$STABLEWM_REF" \
        --device cuda:0 \
        --seed "$seed" \
        "${replay_args[@]}" \
        >"$OUTPUT_DIR/e4_speed_noctx_n50_s${seed}.log" 2>&1
    }
    for ((offset=0; offset<${#SEEDS[@]}; offset+=${#DEVICES[@]})); do
      pids=()
      labels=()
      for ((index=0; index<${#DEVICES[@]}; index++)); do
        position=$((offset + index))
        if [[ "$position" -ge "${#SEEDS[@]}" ]]; then
          break
        fi
        seed="${SEEDS[$position]}"
        device="${DEVICES[$index]}"
        echo "[$SLUG e4-noctx] seed=$seed physical_device=$device"
        run_e4_noctx_seed "$seed" "$device" &
        pids+=("$!")
        labels+=("seed=$seed/device=$device")
      done
      for ((index=0; index<${#pids[@]}; index++)); do
        if ! wait "${pids[$index]}"; then
          echo "[$SLUG e4-noctx] failed ${labels[$index]}" >&2
          exit 1
        fi
        echo "[$SLUG e4-noctx] completed ${labels[$index]}"
      done
    done
    ;;
  e4)
    CHECKPOINT="$CHECKPOINT" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    NORMALIZER="$NORMALIZER" \
    RUN_LABEL="$SLUG E4" \
    DEVICES="${DEVICES[*]}" \
    STABLEWM_REPO="$STABLEWM_REPO" \
    STABLEWM_REF="$STABLEWM_REF" \
      bash scripts/run_h3_speedfull_e4_parallel.sh "${SEEDS[@]}"
    ;;
  *)
    echo "Unsupported suite: $SUITE" >&2
    exit 2
    ;;
esac
