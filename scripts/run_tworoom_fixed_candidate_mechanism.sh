#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
SOURCE="$ARTIFACT_ROOT/evaluation/history3/icl_sensitive_v2_directional"
OUTPUT="$ARTIFACT_ROOT/evaluation/history3/planner_mechanism_v1/fixed_candidates"
NORMALIZER="$ARTIFACT_ROOT/splits/tworoom_original_train_s3072_normalizer.json"
read -r -a DEVICES <<< "${DEVICES:-0 1 2 3 4 5 6 7}"
MODELS=(
  "h3_speedfull_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt"
  "h3_origplus_synth5_s3072|$ARTIFACT_ROOT/training/runs/checkpoints/h3_origplus_synth5_s3072/weights_final_step_12840.pt"
)
read -r -a SEEDS <<< "${SEED_LIST:-42 43 44 45 46 47}"
MODEL_FILTER="${MODEL_FILTER:-}"
mapfile -t TEMPLATES < <(jq -r '.geometry_bank[].template_id' "$SOURCE/catalogs/heldout_wrong_slow.json")
jobs=()
for model in "${MODELS[@]}"; do
  [[ -n "$MODEL_FILTER" && "${model%%|*}" != "$MODEL_FILTER" ]] && continue
  for seed in "${SEEDS[@]}"; do jobs+=("$model|$seed"); done
done
for ((offset=0; offset<${#jobs[@]}; offset+=${#DEVICES[@]})); do
  pids=()
  labels=()
  for ((i=0; i<${#DEVICES[@]}; i++)); do
    pos=$((offset+i)); [[ "$pos" -ge "${#jobs[@]}" ]] && break
    IFS='|' read -r slug checkpoint seed <<< "${jobs[$pos]}"
    out_dir="$OUTPUT/$slug"; mkdir -p "$out_dir"
    out="$out_dir/fixed_candidates_n50_s${seed}.json"
    if [[ -s "$out" ]] && jq -e \
      '.status=="passed" and .candidate_generation_version=="goal_directed_mixture_v1"
       and (.records|length)==50' "$out" >/dev/null; then
      echo "[fixed candidates] reusing $slug seed=$seed"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" "$PYTHON_BIN" \
      scripts/eval_tworoom_fixed_candidate_mechanism.py \
      --slow-catalog "$SOURCE/catalogs/heldout_wrong_slow.json" \
      --fast-catalog "$SOURCE/catalogs/heldout_wrong_fast.json" \
      --checkpoint "$checkpoint" --normalizer "$NORMALIZER" --output "$out" \
      --templates "${TEMPLATES[@]}" --seed "$seed" --device cuda:0 \
      >"$out_dir/fixed_candidates_n50_s${seed}.log" 2>&1 &
    pids+=("$!"); labels+=("$slug/s$seed")
  done
  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[$i]}"; then echo "failed ${labels[$i]}" >&2; exit 1; fi
    echo "[fixed candidates] completed ${labels[$i]}"
  done
done
