#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CHECKPOINT="${CHECKPOINT:-$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedseen_s3072/weights_final_step_6420.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/evaluation/history3/h3_speedseen_s3072}"
ORIGINAL_H5="${ORIGINAL_H5:-/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/tworoom.h5}"
LEGACY_CODE_ROOT="${LEGACY_CODE_ROOT:-/opt/huawei/explorer-env/dataset/ag_data/code/wm_exp}"
CATALOG="${CATALOG:-$ARTIFACT_ROOT/evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3}"
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 43 44 45 46 47)
fi

mkdir -p "$OUTPUT_DIR"

run_seed() {
  local seed="$1"
  local device="$2"
  local output="$OUTPUT_DIR/e4_speed_ctx_n50_s${seed}.json"
  local log="$OUTPUT_DIR/e4_speed_ctx_n50_s${seed}.log"
  local replay_args=()
  if [[ "$seed" != "42" ]]; then
    replay_args+=(--skip-catalog-replay)
  fi
  echo "[H3-SpeedSeen E4] seed=$seed physical_device=$device output=$output"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" scripts/eval_tworoom_icl_planning.py \
    --catalog "$CATALOG" \
    --checkpoint "$CHECKPOINT" \
    --legacy-code-root "$LEGACY_CODE_ROOT" \
    --original-h5 "$ORIGINAL_H5" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 \
    --seed "$seed" \
    --num-eval 50 \
    --run-kind confirmation \
    --speeds 3.1 3.3 3.5 4.1 5.0 5.1 5.9 7.0 \
    --templates s0 s1 s2 s3 \
    --eval-budget 50 \
    --img-size 224 \
    --horizon 5 \
    --receding-horizon 5 \
    --cem-batch-size 1 \
    --cem-num-samples 300 \
    --cem-var-scale 1.0 \
    --cem-steps 30 \
    --cem-topk 30 \
    --output "$output" \
    "${replay_args[@]}" \
    >"$log" 2>&1
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
    run_seed "$seed" "$device" &
    pids+=("$!")
    labels+=("seed=$seed/device=$device")
  done
  for ((index=0; index<${#pids[@]}; index++)); do
    if ! wait "${pids[$index]}"; then
      echo "[H3-SpeedSeen E4] failed ${labels[$index]}" >&2
      exit 1
    fi
    echo "[H3-SpeedSeen E4] completed ${labels[$index]}"
  done
done

echo "[H3-SpeedSeen E4] raw seeds completed: ${SEEDS[*]}"
"$PYTHON_BIN" scripts/run_tworoom_history3_eval.py \
  --suite e4-context-plan \
  --checkpoint "$CHECKPOINT" \
  --legacy-code-root "$LEGACY_CODE_ROOT" \
  --original-h5 "$ORIGINAL_H5" \
  --stablewm-repo "$STABLEWM_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --eval-seeds "${SEEDS[@]}" \
  --num-eval 50 \
  --skip-catalog-replay \
  --reuse-existing
echo "[H3-SpeedSeen E4] summary=$OUTPUT_DIR/e4_speed_ctx_n50x${#SEEDS[@]}.json"
