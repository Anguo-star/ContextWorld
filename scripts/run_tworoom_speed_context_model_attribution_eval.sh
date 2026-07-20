#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CONFIG="$ROOT/configs/benchmark/tworoom_speed_context_model_attribution_v1.yaml"
EVAL_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
CATALOG_ROOT="$EVAL_ROOT/catalogs"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"

SEEDS=(42 43 44 45 46 47)
DIRECTIONS=(wrong_slow wrong_fast)
MODELS=(
  "h3_origheldout_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_origheldout_s3072/weights_final_step_6420.pt|7d141b86cca49145444a69bff89c71ede69e8cf8252bfb933e656c3e2e962b54"
  "h3_synth5matched_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_synth5matched_s3072/weights_final_step_6420.pt|565de1fe25015603831fdfa9b0c3dee4f711ad1d929b00bf814ffee254c36283"
  "h3_origplus_synth5_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_origplus_synth5_s3072/weights_final_step_12840.pt|79e1c63a00e4b24e7c65d5bdc096afcfec7edd838e58475280637923991b3474"
)

catalog_for() {
  local direction="$1"
  echo "$CATALOG_ROOT/heldout_${direction}.json"
}

audit_static() {
  "$PYTHON_BIN" \
    scripts/analyze_tworoom_speed_context_model_attribution.py \
    --config "$CONFIG" \
    --audit-only
}

is_reusable_result() {
  local path="$1"
  local checkpoint_sha="$2"
  local seed="$3"
  [[ -s "$path" ]] && jq -e \
    --arg checkpoint_sha "$checkpoint_sha" \
    --argjson seed "$seed" \
    '
      .status == "passed"
      and .checkpoint.sha256 == $checkpoint_sha
      and .protocol.eval_seed == $seed
      and .protocol.eval_budget == 50
      and .protocol.horizon == 5
      and .protocol.receding_horizon == 5
      and .protocol.cem_num_samples == 300
      and .protocol.cem_steps == 30
      and .protocol.cem_topk == 30
      and (.records | length) == 100
      and ([.records[] | select(.condition == "correct")] | length) == 50
      and ([.records[] | select(.condition == "wrong")] | length) == 50
    ' "$path" >/dev/null
}

run_job() {
  local model_slug="$1"
  local checkpoint="$2"
  local checkpoint_sha="$3"
  local direction="$4"
  local seed="$5"
  local device="$6"
  local catalog
  catalog="$(catalog_for "$direction")"
  local raw_root="$EVAL_ROOT/$model_slug"
  local output="$raw_root/${direction}_n50_s${seed}.json"
  local log="$raw_root/${direction}_n50_s${seed}.log"
  mkdir -p "$raw_root"
  if is_reusable_result "$output" "$checkpoint_sha" "$seed"; then
    echo "[model attribution] reusing model=$model_slug direction=$direction seed=$seed"
    return
  fi
  mapfile -t templates < <(
    jq -r '.geometry_bank[].template_id' "$catalog"
  )
  echo "[model attribution] model=$model_slug direction=$direction seed=$seed device=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
    scripts/eval_tworoom_icl_planning.py \
    --catalog "$catalog" \
    --checkpoint "$checkpoint" \
    --normalizer "$NORMALIZER" \
    --output "$output" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 \
    --seed "$seed" \
    --num-eval 50 \
    --run-kind confirmation \
    --speeds 5.0 5.1 \
    --templates "${templates[@]}" \
    --eval-budget 50 \
    --horizon 5 \
    --receding-horizon 5 \
    --cem-batch-size 1 \
    --cem-num-samples 300 \
    --cem-var-scale 1.0 \
    --cem-steps 30 \
    --cem-topk 30 \
    --skip-catalog-replay \
    >"$log" 2>&1
}

run_batched() {
  local jobs=()
  local model
  for model in "${MODELS[@]}"; do
    local model_slug="${model%%|*}"
    local remainder="${model#*|}"
    local checkpoint="${remainder%%|*}"
    local checkpoint_sha="${remainder#*|}"
    local direction
    for direction in "${DIRECTIONS[@]}"; do
      local seed
      for seed in "${SEEDS[@]}"; do
        jobs+=(
          "$model_slug|$checkpoint|$checkpoint_sha|$direction|$seed"
        )
      done
    done
  done

  local offset
  for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
    local pids=()
    local labels=()
    local index
    for ((index=0; index<${#DEVICES[@]}; index++)); do
      local position=$((offset + index))
      if [[ "$position" -ge "${#jobs[@]}" ]]; then
        break
      fi
      local job="${jobs[$position]}"
      IFS='|' read -r model_slug checkpoint checkpoint_sha direction seed <<< "$job"
      local device="${DEVICES[$index]}"
      run_job \
        "$model_slug" \
        "$checkpoint" \
        "$checkpoint_sha" \
        "$direction" \
        "$seed" \
        "$device" &
      pids+=("$!")
      labels+=("$model_slug/$direction/seed=$seed/device=$device")
    done
    for ((index=0; index<${#pids[@]}; index++)); do
      if ! wait "${pids[$index]}"; then
        echo "[model attribution] failed ${labels[$index]}" >&2
        return 1
      fi
      echo "[model attribution] completed ${labels[$index]}"
    done
  done
}

analyze() {
  "$PYTHON_BIN" \
    scripts/analyze_tworoom_speed_context_model_attribution.py \
    --config "$CONFIG"
}

case "$MODE" in
  audit)
    audit_static
    ;;
  run)
    audit_static
    run_batched
    ;;
  analyze)
    analyze
    ;;
  all)
    audit_static
    run_batched
    analyze
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_speed_context_model_attribution_eval.sh [audit|run|analyze|all]" >&2
    exit 2
    ;;
esac
