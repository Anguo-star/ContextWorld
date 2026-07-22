#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:-preflight}"
read -r -a SEEDS <<< "${TRAINING_SEEDS:-3072 4096 5120}"

for seed in "${SEEDS[@]}"; do
  if [[ "$MODE" == "preflight" ]]; then
    MODEL_VARIANT=fixed TRAINING_SEED="$seed" \
      bash scripts/run_h3_door_train.sh preflight
    MODEL_VARIANT=multi TRAINING_SEED="$seed" \
      bash scripts/run_h3_door_train.sh preflight
    continue
  fi

  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=23271 MODEL_VARIANT=fixed TRAINING_SEED="$seed" \
    bash scripts/run_h3_door_train.sh "$MODE" &
  fixed_pid=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=23272 MODEL_VARIANT=multi TRAINING_SEED="$seed" \
    bash scripts/run_h3_door_train.sh "$MODE" &
  multi_pid=$!

  failed=0
  wait "$fixed_pid" || failed=1
  wait "$multi_pid" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "[door-v2] paired training failed for seed=$seed" >&2
    exit 1
  fi
  echo "[door-v2] paired training complete seed=$seed"
done
