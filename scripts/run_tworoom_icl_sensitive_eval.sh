#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
CONFIG="$ROOT/configs/benchmark/tworoom_speed_icl_sensitive_eval_v1.yaml"
EVAL_ROOT="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v1"
CATALOG_ROOT="$EVAL_ROOT/catalogs"
CALIBRATION_CATALOG="$CATALOG_ROOT/calibration_bank.json"
SELECTED_CATALOG="$CATALOG_ROOT/heldout_selected.json"
CALIBRATION_ROOT="$EVAL_ROOT/calibration"
FORMAL_ROOT="$EVAL_ROOT/formal"
SELECTION="$EVAL_ROOT/calibration_selection.json"
SUMMARY="$EVAL_ROOT/formal_summary_n50x6.json"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
SPEEDFULL_CHECKPOINT="$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"

SPEEDS=(3.1 3.3 3.5 4.1 5.0 5.1 5.9 7.0)
CALIBRATION_SEEDS=(2401 2402)
FORMAL_SEEDS=(42 43 44 45 46 47)
MODELS=(
  "h3_origheldout_s3072:weights_final_step_6420.pt"
  "h3_synth5matched_s3072:weights_final_step_6420.pt"
  "h3_origplus_synth5_s3072:weights_final_step_12840.pt"
  "h3_speedfull_s3072:weights_final_step_12840.pt"
)

is_passed() {
  local path="$1"
  [[ -s "$path" ]] && jq -e '.status == "passed"' "$path" >/dev/null
}

speed_slug() {
  local speed="$1"
  local normalized="${speed%.0}"
  echo "${normalized/./p}"
}

build_catalogs() {
  local report="$CATALOG_ROOT/catalog_build_report.json"
  local config_sha
  config_sha="$(sha256sum "$CONFIG" | awk '{print $1}')"
  if is_passed "$report" && \
    [[ "$(jq -r '.config.sha256' "$report")" == "$config_sha" ]]; then
    echo "[ICL-sensitive build] reusing current catalog build"
    return
  fi
  echo "[ICL-sensitive build] generating calibration and untouched heldout banks"
  "$PYTHON_BIN" scripts/build_tworoom_icl_sensitive_catalogs.py \
    --config "$CONFIG" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF"
}

run_calibration_job() {
  local speed="$1"
  local seed="$2"
  local device="$3"
  local slug
  slug="$(speed_slug "$speed")"
  local output_dir="$CALIBRATION_ROOT/h3_speedfull_s3072"
  local output="$output_dir/paired_speed${slug}_n72_s${seed}.json"
  local log="$output_dir/paired_speed${slug}_n72_s${seed}.log"
  mkdir -p "$output_dir"
  if is_passed "$output"; then
    echo "[ICL-sensitive calibration] reusing speed=$speed seed=$seed"
    return
  fi
  mapfile -t templates < <(
    jq -r '.geometry_bank[].template_id' "$CALIBRATION_CATALOG"
  )
  echo "[ICL-sensitive calibration] speed=$speed seed=$seed device=$device"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
    scripts/eval_tworoom_icl_planning.py \
    --catalog "$CALIBRATION_CATALOG" \
    --checkpoint "$SPEEDFULL_CHECKPOINT" \
    --normalizer "$NORMALIZER" \
    --output "$output" \
    --stablewm-repo "$STABLEWM_REPO" \
    --stablewm-ref "$STABLEWM_REF" \
    --device cuda:0 \
    --seed "$seed" \
    --num-eval 72 \
    --run-kind confirmation \
    --speeds "$speed" \
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

run_calibration() {
  local jobs=()
  for speed in "${SPEEDS[@]}"; do
    for seed in "${CALIBRATION_SEEDS[@]}"; do
      jobs+=("$speed|$seed")
    done
  done
  run_batched calibration run_calibration_job "${jobs[@]}"
}

select_distances() {
  echo "[ICL-sensitive select] applying frozen distance gates"
  "$PYTHON_BIN" scripts/analyze_tworoom_icl_sensitive_eval.py \
    select --config "$CONFIG"
}

run_formal_job() {
  local model_spec="$1"
  local seed="$2"
  local device="$3"
  local model_slug="${model_spec%%:*}"
  local checkpoint_name="${model_spec#*:}"
  local checkpoint="$ARTIFACT_ROOT/training/runs/checkpoints/$model_slug/$checkpoint_name"
  local output_dir="$FORMAL_ROOT/$model_slug"
  local noctx_output="$output_dir/none_n50_s${seed}.json"
  local paired_output="$output_dir/paired_n50_s${seed}.json"
  local noctx_log="$output_dir/none_n50_s${seed}.log"
  local paired_log="$output_dir/paired_n50_s${seed}.log"
  mkdir -p "$output_dir"
  mapfile -t templates < <(
    jq -r '.geometry_bank[].template_id' "$SELECTED_CATALOG"
  )

  if ! is_passed "$noctx_output"; then
    echo "[ICL-sensitive formal none] model=$model_slug seed=$seed device=$device"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
      scripts/eval_tworoom_icl_nocontext_planning.py \
      --catalog "$SELECTED_CATALOG" \
      --checkpoint "$checkpoint" \
      --normalizer "$NORMALIZER" \
      --output "$noctx_output" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --seed "$seed" \
      --num-eval 50 \
      --speeds "${SPEEDS[@]}" \
      --templates "${templates[@]}" \
      --eval-budget 50 \
      --horizon 5 \
      --receding-horizon 5 \
      --cem-num-samples 300 \
      --cem-steps 30 \
      --cem-topk 30 \
      --skip-catalog-replay \
      >"$noctx_log" 2>&1
  else
    echo "[ICL-sensitive formal none] reusing model=$model_slug seed=$seed"
  fi

  if ! is_passed "$paired_output"; then
    echo "[ICL-sensitive formal paired] model=$model_slug seed=$seed device=$device"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
      scripts/eval_tworoom_icl_planning.py \
      --catalog "$SELECTED_CATALOG" \
      --checkpoint "$checkpoint" \
      --normalizer "$NORMALIZER" \
      --output "$paired_output" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --seed "$seed" \
      --num-eval 50 \
      --run-kind confirmation \
      --speeds "${SPEEDS[@]}" \
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
      >"$paired_log" 2>&1
  else
    echo "[ICL-sensitive formal paired] reusing model=$model_slug seed=$seed"
  fi
}

run_formal() {
  if ! jq -e '.formal_eval_authorized == true' "$SELECTION" >/dev/null; then
    echo "[ICL-sensitive formal] stopped: no calibration distance passed frozen gates"
    return
  fi
  local jobs=()
  for model_spec in "${MODELS[@]}"; do
    for seed in "${FORMAL_SEEDS[@]}"; do
      jobs+=("$model_spec|$seed")
    done
  done
  run_batched formal run_formal_job "${jobs[@]}"
}

analyze_formal() {
  if ! jq -e '.formal_eval_authorized == true' "$SELECTION" >/dev/null; then
    echo "[ICL-sensitive analyze] no formal summary: calibration stopped the protocol"
    return
  fi
  echo "[ICL-sensitive analyze] aggregating four-model heldout results"
  "$PYTHON_BIN" scripts/analyze_tworoom_icl_sensitive_eval.py \
    formal --config "$CONFIG"
}

diagnose_stop() {
  if jq -e '.formal_eval_authorized == false' "$SELECTION" >/dev/null; then
    echo "[ICL-sensitive diagnose] decomposing the frozen calibration stop"
    "$PYTHON_BIN" scripts/analyze_tworoom_icl_sensitive_eval.py \
      diagnose --config "$CONFIG"
  fi
}

run_batched() {
  local stage="$1"
  local function_name="$2"
  shift 2
  local jobs=("$@")
  for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
    local pids=()
    local labels=()
    for ((index=0; index<${#DEVICES[@]}; index++)); do
      local position=$((offset + index))
      if [[ "$position" -ge "${#jobs[@]}" ]]; then
        break
      fi
      local job="${jobs[$position]}"
      local left="${job%%|*}"
      local right="${job#*|}"
      local device="${DEVICES[$index]}"
      "$function_name" "$left" "$right" "$device" &
      pids+=("$!")
      labels+=("$left/$right/device=$device")
    done
    for ((index=0; index<${#pids[@]}; index++)); do
      if ! wait "${pids[$index]}"; then
        echo "[ICL-sensitive $stage] failed ${labels[$index]}" >&2
        return 1
      fi
      echo "[ICL-sensitive $stage] completed ${labels[$index]}"
    done
  done
}

case "$MODE" in
  build)
    build_catalogs
    ;;
  calibrate)
    run_calibration
    ;;
  select)
    select_distances
    ;;
  formal)
    run_formal
    ;;
  analyze)
    analyze_formal
    ;;
  diagnose)
    diagnose_stop
    ;;
  all)
    build_catalogs
    run_calibration
    select_distances
    run_formal
    analyze_formal
    diagnose_stop
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_icl_sensitive_eval.sh [build|calibrate|select|formal|analyze|diagnose|all]" >&2
    exit 2
    ;;
esac

if [[ -s "$SELECTION" ]] && \
  jq -e '.formal_eval_authorized == false' "$SELECTION" >/dev/null; then
  echo "[ICL-sensitive] mode=$MODE selection=$SELECTION formal_summary=not_authorized"
else
  echo "[ICL-sensitive] mode=$MODE selection=$SELECTION summary=$SUMMARY"
fi
