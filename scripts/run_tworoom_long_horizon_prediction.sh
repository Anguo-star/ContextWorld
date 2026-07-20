#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
SOURCE_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
OUTPUT_ROOT="$ARTIFACT_ROOT/evaluation/history3/cem_config_impact_v1/long_horizon_prediction/h3_speedfull_s3072"
CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5}"
read -r -a SEEDS <<< "${SEED_LIST:-42 43 44 45 46 47}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mapfile -t TEMPLATES < <(
  jq -r '.geometry_bank[].template_id' \
    "$SOURCE_ROOT/catalogs/heldout_wrong_slow.json"
)
mkdir -p "$OUTPUT_ROOT"

pids=()
labels=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  device="${DEVICES[$((index % ${#DEVICES[@]}))]}"
  output="$OUTPUT_ROOT/long_horizon_prediction_n50_s${seed}.json"
  log="$OUTPUT_ROOT/long_horizon_prediction_n50_s${seed}.log"
  if [[ -s "$output" ]] && jq -e \
    --argjson seed "$seed" \
    '.status == "passed" and
     .eval_seed == $seed and
     .protocol.prediction_horizon_raw_steps == 50 and
     (.records | length) == 50 and
     ([.records[].probe.last_25_raw_actions_are_zero] | all)' \
    "$output" >/dev/null; then
    echo "[long prediction] reusing seed=$seed"
    continue
  fi
  echo "[long prediction] seed=$seed gpu=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
    scripts/eval_tworoom_long_horizon_prediction.py \
    --slow-catalog "$SOURCE_ROOT/catalogs/heldout_wrong_slow.json" \
    --fast-catalog "$SOURCE_ROOT/catalogs/heldout_wrong_fast.json" \
    --checkpoint "$CHECKPOINT" \
    --normalizer "$NORMALIZER" \
    --output "$output" \
    --templates "${TEMPLATES[@]}" \
    --seed "$seed" \
    --num-eval 50 \
    --device cuda:0 \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" >"$log" 2>&1 &
  pids+=("$!")
  labels+=("s$seed")
done

for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[long prediction] failed ${labels[$index]}" >&2
    exit 1
  fi
  echo "[long prediction] completed ${labels[$index]}"
done
