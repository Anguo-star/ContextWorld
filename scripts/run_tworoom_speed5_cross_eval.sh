#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
STABLEWM_REF="${STABLEWM_REF:-5864b74980f6ed328fd0045e777b3865962eff43}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
EVAL_ROOT="$ARTIFACT_ROOT/evaluation/history3/original_ability_reconstruction"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
CATALOG="$ARTIFACT_ROOT/evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json"
SUMMARY="$EVAL_ROOT/speed5_cross_eval_n50x6.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"
SEEDS=(42 43 44 45 46 47)
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

run_job() {
  local model_spec="$1"
  local seed="$2"
  local device="$3"
  local slug="${model_spec%%:*}"
  local checkpoint_name="${model_spec#*:}"
  local checkpoint="$ARTIFACT_ROOT/training/runs/checkpoints/$slug/$checkpoint_name"
  local output_dir="$EVAL_ROOT/$slug/speed5_cross_eval"
  local noctx_output="$output_dir/e4_speed5_noctx_n50_s${seed}.json"
  local ctx_output="$output_dir/e4_speed5_ctx_n50_s${seed}.json"
  mkdir -p "$output_dir"

  if ! is_passed "$noctx_output"; then
    echo "[$slug speed5 noctx] seed=$seed device=$device"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
      scripts/eval_tworoom_icl_nocontext_planning.py \
      --catalog "$CATALOG" \
      --checkpoint "$checkpoint" \
      --normalizer "$NORMALIZER" \
      --output "$noctx_output" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --seed "$seed" \
      --num-eval 50 \
      --speeds 5.0 \
      --templates s0 s1 s2 s3 \
      --eval-budget 50 \
      --horizon 5 \
      --receding-horizon 5 \
      --cem-num-samples 300 \
      --cem-steps 30 \
      --cem-topk 30 \
      --skip-catalog-replay \
      >"$output_dir/e4_speed5_noctx_n50_s${seed}.log" 2>&1
  else
    echo "[$slug speed5 noctx] reusing seed=$seed"
  fi

  if ! is_passed "$ctx_output"; then
    echo "[$slug speed5 context] seed=$seed device=$device"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
      scripts/eval_tworoom_icl_planning.py \
      --catalog "$CATALOG" \
      --checkpoint "$checkpoint" \
      --normalizer "$NORMALIZER" \
      --output "$ctx_output" \
      --stablewm-repo "$STABLEWM_REPO" \
      --stablewm-ref "$STABLEWM_REF" \
      --device cuda:0 \
      --seed "$seed" \
      --num-eval 50 \
      --run-kind confirmation \
      --speeds 5.0 \
      --templates s0 s1 s2 s3 \
      --eval-budget 50 \
      --horizon 5 \
      --receding-horizon 5 \
      --cem-batch-size 1 \
      --cem-num-samples 300 \
      --cem-var-scale 1.0 \
      --cem-steps 30 \
      --cem-topk 30 \
      --skip-catalog-replay \
      >"$output_dir/e4_speed5_ctx_n50_s${seed}.log" 2>&1
  else
    echo "[$slug speed5 context] reusing seed=$seed"
  fi
}

run_all() {
  local jobs=()
  for model_spec in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("$model_spec|$seed")
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
      local model_spec="${job%%|*}"
      local seed="${job#*|}"
      local device="${DEVICES[$index]}"
      run_job "$model_spec" "$seed" "$device" &
      pids+=("$!")
      labels+=("${model_spec%%:*}/seed=$seed/device=$device")
    done
    for ((index=0; index<${#pids[@]}; index++)); do
      if ! wait "${pids[$index]}"; then
        echo "[speed5 cross-eval] failed ${labels[$index]}" >&2
        exit 1
      fi
      echo "[speed5 cross-eval] completed ${labels[$index]}"
    done
  done
}

analyze() {
  "$PYTHON_BIN" scripts/analyze_tworoom_speed5_cross_eval.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --output "$SUMMARY"
}

case "$MODE" in
  run)
    run_all
    ;;
  analyze)
    analyze
    ;;
  all)
    run_all
    analyze
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_speed5_cross_eval.sh [run|analyze|all]" >&2
    exit 2
    ;;
esac

echo "[speed5 cross-eval] mode=$MODE summary=$SUMMARY"
