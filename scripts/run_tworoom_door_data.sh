#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-compile}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHARDS_PER_CONFIG="${SHARDS_PER_CONFIG:-32}"
CONFIGS=(
  "$ROOT/configs/synthesis/tworoom_door_fixed49_matched_v2.yaml"
  "$ROOT/configs/synthesis/tworoom_door_multi_v2.yaml"
)

run_smoke() {
  local config="$1"
  shift
  "$PYTHON_BIN" -m contextworld.synthesis.smoke \
    --config "$config" \
    "$@"
}

collect_config() {
  local config="$1"
  local pids=()
  local index
  for ((index=0; index<SHARDS_PER_CONFIG; index++)); do
    "$PYTHON_BIN" "$ROOT/scripts/collect_tworoom_synthesis_shard.py" \
      --config "$config" \
      --shard-index "$index" \
      --num-shards "$SHARDS_PER_CONFIG" &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Door data shard collection failed for $config" >&2
    return 1
  fi
}

case "$MODE" in
  compile)
    for config in "${CONFIGS[@]}"; do
      run_smoke "$config" --compile-only
    done
    ;;
  generate-parallel)
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    collect_config "${CONFIGS[0]}" &
    fixed_pid=$!
    collect_config "${CONFIGS[1]}" &
    multi_pid=$!
    failed=0
    wait "$fixed_pid" || failed=1
    wait "$multi_pid" || failed=1
    if [[ "$failed" -ne 0 ]]; then
      echo "Door data collection did not complete for both paired recipes" >&2
      exit 1
    fi
    for config in "${CONFIGS[@]}"; do
      run_smoke "$config" --resume
    done
    ;;
  resume)
    for config in "${CONFIGS[@]}"; do
      run_smoke "$config" --resume
    done
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_door_data.sh [compile|generate-parallel|resume]" >&2
    exit 2
    ;;
esac
