#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-compile}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT/configs/synthesis/tworoom_speed_full_v1.yaml"

case "$MODE" in
  compile)
    EXTRA_ARGS=(--compile-only)
    ;;
  generate)
    EXTRA_ARGS=()
    ;;
  generate-parallel)
    SHARDS="${SHARDS:-32}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    echo "[TwoRoom-SpeedFull] mode=$MODE config=$CONFIG shards=$SHARDS"
    pids=()
    for ((index=0; index<SHARDS; index++)); do
      "$PYTHON_BIN" "$ROOT/scripts/collect_tworoom_synthesis_shard.py" \
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
    exit 0
    ;;
  resume)
    EXTRA_ARGS=(--resume)
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_speedfull_data.sh [compile|generate|generate-parallel|resume]" >&2
    exit 2
    ;;
esac

echo "[TwoRoom-SpeedFull] mode=$MODE config=$CONFIG"
"$PYTHON_BIN" -m contextworld.synthesis.smoke \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
