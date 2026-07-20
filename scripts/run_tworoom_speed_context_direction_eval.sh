#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CONFIG="$ROOT/configs/benchmark/tworoom_speed_context_direction_eval_v2.yaml"
EVAL_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
CATALOG_ROOT="$EVAL_ROOT/catalogs"
RAW_ROOT="$EVAL_ROOT/h3_speedfull_s3072"
BUILD_REPORT="$CATALOG_ROOT/catalog_build_report.json"
SUMMARY="$EVAL_ROOT/formal_summary_n50x6.json"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"

SEEDS=(42 43 44 45 46 47)
DIRECTIONS=(wrong_slow wrong_fast)

is_passed() {
  local path="$1"
  [[ -s "$path" ]] && jq -e '.status == "passed"' "$path" >/dev/null
}

catalog_for() {
  local direction="$1"
  echo "$CATALOG_ROOT/heldout_${direction}.json"
}

build_catalogs() {
  local config_sha
  config_sha="$(sha256sum "$CONFIG" | awk '{print $1}')"
  if is_passed "$BUILD_REPORT" && \
    [[ "$(jq -r '.config.sha256' "$BUILD_REPORT")" == "$config_sha" ]] && \
    jq -e '.cross_catalog_audit.passed == true' "$BUILD_REPORT" >/dev/null; then
    echo "[directional build] reusing paired catalogs"
    return
  fi
  echo "[directional build] generating and replaying both catalogs before scoring"
  "$PYTHON_BIN" scripts/build_tworoom_speed_context_direction_catalogs.py \
    --config "$CONFIG" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF"
}

run_job() {
  local direction="$1"
  local seed="$2"
  local device="$3"
  local catalog
  catalog="$(catalog_for "$direction")"
  local output="$RAW_ROOT/${direction}_n50_s${seed}.json"
  local log="$RAW_ROOT/${direction}_n50_s${seed}.log"
  mkdir -p "$RAW_ROOT"
  if is_passed "$output"; then
    echo "[directional eval] reusing direction=$direction seed=$seed"
    return
  fi
  mapfile -t templates < <(
    jq -r '.geometry_bank[].template_id' "$catalog"
  )
  echo "[directional eval] direction=$direction seed=$seed device=$device"
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
  for direction in "${DIRECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("$direction|$seed")
    done
  done
  for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
    local pids=()
    local labels=()
    for ((index=0; index<${#DEVICES[@]}; index++)); do
      local position=$((offset + index))
      if [[ "$position" -ge "${#jobs[@]}" ]]; then
        break
      fi
      local job="${jobs[$position]}"
      local direction="${job%%|*}"
      local seed="${job#*|}"
      local device="${DEVICES[$index]}"
      run_job "$direction" "$seed" "$device" &
      pids+=("$!")
      labels+=("$direction/seed=$seed/device=$device")
    done
    for ((index=0; index<${#pids[@]}; index++)); do
      if ! wait "${pids[$index]}"; then
        echo "[directional eval] failed ${labels[$index]}" >&2
        return 1
      fi
      echo "[directional eval] completed ${labels[$index]}"
    done
  done
}

analyze() {
  echo "[directional analyze] verifying 50×6 counts and aggregating three contexts"
  "$PYTHON_BIN" scripts/analyze_tworoom_speed_context_direction_eval.py \
    --config "$CONFIG"
}

case "$MODE" in
  build)
    build_catalogs
    ;;
  run)
    run_batched
    ;;
  analyze)
    analyze
    ;;
  all)
    build_catalogs
    run_batched
    analyze
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_speed_context_direction_eval.sh [build|run|analyze|all]" >&2
    exit 2
    ;;
esac

echo "[directional] mode=$MODE build_report=$BUILD_REPORT summary=$SUMMARY"
