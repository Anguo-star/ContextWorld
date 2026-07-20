#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-compile}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORMAL_CONFIG="$ROOT/configs/synthesis/tworoom_speed_clean_v1.yaml"
SMOKE_CONFIG="$ROOT/configs/synthesis/tworoom_speed_clean_smoke_v1.yaml"

case "$MODE" in
  compile)
    CONFIG="$FORMAL_CONFIG"
    EXTRA_ARGS=(--compile-only)
    ;;
  smoke)
    CONFIG="$SMOKE_CONFIG"
    EXTRA_ARGS=()
    ;;
  smoke-resume)
    CONFIG="$SMOKE_CONFIG"
    EXTRA_ARGS=(--resume)
    ;;
  generate)
    CONFIG="$FORMAL_CONFIG"
    EXTRA_ARGS=()
    ;;
  generate-parallel)
    SHARDS="${SHARDS:-4}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    echo "[TwoRoom-SpeedClean] mode=$MODE config=$FORMAL_CONFIG shards=$SHARDS"
    pids=()
    for ((index=0; index<SHARDS; index++)); do
      "$PYTHON_BIN" "$ROOT/scripts/collect_tworoom_synthesis_shard.py" \
        --config "$FORMAL_CONFIG" \
        --shard-index "$index" \
        --num-shards "$SHARDS" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    "$PYTHON_BIN" -m contextworld.synthesis.smoke \
      --config "$FORMAL_CONFIG" \
      --resume
    exit 0
    ;;
  resume)
    CONFIG="$FORMAL_CONFIG"
    EXTRA_ARGS=(--resume)
    ;;
  *)
    echo "Usage: bash scripts/run_tworoom_speedclean_data.sh [compile|smoke|smoke-resume|generate|generate-parallel|resume]" >&2
    exit 2
    ;;
esac

echo "[TwoRoom-SpeedClean] mode=$MODE config=$CONFIG"
"$PYTHON_BIN" -m contextworld.synthesis.smoke \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
