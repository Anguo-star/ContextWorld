#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
SOURCE_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
OUTPUT_ROOT="$ARTIFACT_ROOT/evaluation/history3/planner_mechanism_v1/closed_loop"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

MODELS=(
  "h3_speedfull_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"
  "h3_origplus_synth5_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_origplus_synth5_s3072/weights_final_step_12840.pt"
)
read -r -a SEEDS <<< "${SEED_LIST:-42 43 44 45 46 47}"
read -r -a DIRECTIONS <<< "${DIRECTION_LIST:-wrong_slow wrong_fast}"
MODEL_FILTER="${MODEL_FILTER:-}"
BUDGET_FILTER="${BUDGET_FILTER:-}"

run_job() {
  local model_slug="$1" checkpoint="$2" direction="$3" seed="$4" budget="$5" device="$6"
  local catalog="$SOURCE_ROOT/catalogs/heldout_${direction}.json"
  local out_dir="$OUTPUT_ROOT/budget_${budget}/${model_slug}"
  local output="$out_dir/${direction}_n50_s${seed}.json"
  local log="$out_dir/${direction}_n50_s${seed}.log"
  mkdir -p "$out_dir"
  if [[ -s "$output" ]] && jq -e \
    --argjson budget "$budget" \
    '.status == "passed" and .protocol.eval_budget == $budget and
     (.records | length) == 100 and
     ([.records[].trajectory.raw_steps_executed] | length) == 100' \
    "$output" >/dev/null; then
    echo "[mechanism] reusing $model_slug $direction seed=$seed budget=$budget"
    return
  fi
  mapfile -t templates < <(jq -r '.geometry_bank[].template_id' "$catalog")
  echo "[mechanism] $model_slug $direction seed=$seed budget=$budget gpu=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_icl_planning.py \
    --catalog "$catalog" --checkpoint "$checkpoint" --normalizer "$NORMALIZER" \
    --output "$output" --stablewm-repo "$STABLEWM_REPO" --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 --seed "$seed" --num-eval 50 --run-kind confirmation \
    --speeds 5.0 5.1 --templates "${templates[@]}" --eval-budget "$budget" \
    --horizon 5 --receding-horizon 5 --cem-batch-size 1 \
    --cem-num-samples 300 --cem-var-scale 1.0 --cem-steps 30 --cem-topk 30 \
    --skip-catalog-replay >"$log" 2>&1
}

jobs=()
for model in "${MODELS[@]}"; do
  IFS='|' read -r slug checkpoint <<< "$model"
  [[ -n "$MODEL_FILTER" && "$slug" != "$MODEL_FILTER" ]] && continue
  budgets=(50)
  if [[ "$slug" == "h3_speedfull_s3072" ]]; then budgets=(50 75 100); fi
  for budget in "${budgets[@]}"; do
    [[ -n "$BUDGET_FILTER" && "$budget" != "$BUDGET_FILTER" ]] && continue
    for direction in "${DIRECTIONS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        jobs+=("$slug|$checkpoint|$direction|$seed|$budget")
      done
    done
  done
done

for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
  pids=()
  labels=()
  for ((index=0; index<${#DEVICES[@]}; index++)); do
    position=$((offset + index))
    [[ "$position" -ge "${#jobs[@]}" ]] && break
    IFS='|' read -r slug checkpoint direction seed budget <<< "${jobs[$position]}"
    run_job "$slug" "$checkpoint" "$direction" "$seed" "$budget" "${DEVICES[$index]}" &
    pids+=("$!")
    labels+=("$slug/$direction/s$seed/b$budget")
  done
  for ((index=0; index<${#pids[@]}; index++)); do
    if ! wait "${pids[$index]}"; then
      echo "[mechanism] failed ${labels[$index]}" >&2
      exit 1
    fi
    echo "[mechanism] completed ${labels[$index]}"
  done
done
