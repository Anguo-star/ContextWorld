#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
SOURCE_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
OUTPUT_ROOT="$ARTIFACT_ROOT/evaluation/history3/cem_config_impact_v1/closed_loop"
CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"
read -r -a SEEDS <<< "${SEED_LIST:-42 43 44 45 46 47}"
read -r -a DIRECTIONS <<< "${DIRECTION_LIST:-wrong_slow wrong_fast}"
CONFIG_FILTER="${CONFIG_FILTER:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# slug | horizon | receding horizon | samples | iterations | top-k
CONFIGURATIONS=(
  "horizon10|10|5|300|30|30"
  "samples600|5|5|600|30|30"
  "iterations60|5|5|300|60|30"
)

mapfile -t TEMPLATES < <(
  jq -r '.geometry_bank[].template_id' \
    "$SOURCE_ROOT/catalogs/heldout_wrong_slow.json"
)

run_job() {
  local slug="$1" horizon="$2" receding="$3" samples="$4"
  local iterations="$5" topk="$6" direction="$7" seed="$8" device="$9"
  local catalog="$SOURCE_ROOT/catalogs/heldout_${direction}.json"
  local out_dir="$OUTPUT_ROOT/$slug/h3_speedfull_s3072"
  local output="$out_dir/${direction}_n50_s${seed}.json"
  local log="$out_dir/${direction}_n50_s${seed}.log"
  mkdir -p "$out_dir"

  if [[ -s "$output" ]] && jq -e \
    --argjson seed "$seed" \
    --argjson horizon "$horizon" \
    --argjson receding "$receding" \
    --argjson samples "$samples" \
    --argjson iterations "$iterations" \
    --argjson topk "$topk" \
    '.status == "passed" and
     .protocol.eval_seed == $seed and
     .protocol.eval_budget == 100 and
     .protocol.horizon == $horizon and
     .protocol.receding_horizon == $receding and
     .protocol.cem_num_samples == $samples and
     .protocol.cem_steps == $iterations and
     .protocol.cem_topk == $topk and
     (.records | length) == 100 and
     ([.records[].trajectory.raw_steps_executed] | length) == 100' \
    "$output" >/dev/null; then
    echo "[cem impact] reusing $slug $direction seed=$seed"
    return
  fi

  echo "[cem impact] $slug $direction seed=$seed gpu=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
    scripts/eval_tworoom_icl_planning.py \
    --catalog "$catalog" \
    --checkpoint "$CHECKPOINT" \
    --normalizer "$NORMALIZER" \
    --output "$output" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 \
    --seed "$seed" \
    --num-eval 50 \
    --run-kind confirmation \
    --speeds 5.0 5.1 \
    --templates "${TEMPLATES[@]}" \
    --eval-budget 100 \
    --horizon "$horizon" \
    --receding-horizon "$receding" \
    --cem-batch-size 1 \
    --cem-num-samples "$samples" \
    --cem-var-scale 1.0 \
    --cem-steps "$iterations" \
    --cem-topk "$topk" \
    --skip-catalog-replay >"$log" 2>&1
}

jobs=()
for configuration in "${CONFIGURATIONS[@]}"; do
  IFS='|' read -r slug horizon receding samples iterations topk \
    <<< "$configuration"
  [[ -n "$CONFIG_FILTER" && "$slug" != "$CONFIG_FILTER" ]] && continue
  for direction in "${DIRECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=(
        "$slug|$horizon|$receding|$samples|$iterations|$topk|$direction|$seed"
      )
    done
  done
done

for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
  pids=()
  labels=()
  for ((index=0; index<${#DEVICES[@]}; index++)); do
    position=$((offset + index))
    [[ "$position" -ge "${#jobs[@]}" ]] && break
    IFS='|' read -r slug horizon receding samples iterations topk direction seed \
      <<< "${jobs[$position]}"
    run_job \
      "$slug" "$horizon" "$receding" "$samples" "$iterations" "$topk" \
      "$direction" "$seed" "${DEVICES[$index]}" &
    pids+=("$!")
    labels+=("$slug/$direction/s$seed")
  done
  for ((index=0; index<${#pids[@]}; index++)); do
    if ! wait "${pids[$index]}"; then
      echo "[cem impact] failed ${labels[$index]}" >&2
      exit 1
    fi
    echo "[cem impact] completed ${labels[$index]}"
  done
done
