#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-compile}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT/configs/synthesis/tworoom_synth5_matched_v2.yaml"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

case "$MODE" in
  freeze-reference)
    "$PYTHON_BIN" scripts/analyze_tworoom_synth5_matched.py \
      --config "$CONFIG" \
      --freeze-reference
    exit 0
    ;;
  compile)
    EXTRA_ARGS=(--compile-only)
    ;;
  generate)
    EXTRA_ARGS=()
    ;;
  generate-parallel)
    SHARDS="${SHARDS:-32}"
    pids=()
    for ((index=0; index<SHARDS; index++)); do
      "$PYTHON_BIN" scripts/collect_tworoom_synthesis_shard.py \
        --config "$CONFIG" \
        --shard-index "$index" \
        --num-shards "$SHARDS" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    "$PYTHON_BIN" -m contextworld.synthesis.smoke \
      --config "$CONFIG" \
      --resume
    "$PYTHON_BIN" scripts/analyze_tworoom_synth5_matched.py \
      --config "$CONFIG"
    exit 0
    ;;
  resume)
    EXTRA_ARGS=(--resume)
    ;;
  analyze)
    "$PYTHON_BIN" scripts/analyze_tworoom_synth5_matched.py \
      --config "$CONFIG"
    exit 0
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_synth5_matched_data.sh [freeze-reference|compile|generate|generate-parallel|resume|analyze]" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" -m contextworld.synthesis.smoke \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
