#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <model-slug> <device> [batch-size]" >&2
  exit 2
fi

MODEL_SLUG="$1"
DEVICE="$2"
BATCH_SIZE="${3:-128}"

python scripts/eval_tworoom_action_delay_h3_multistep.py \
  --model "${MODEL_SLUG}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}"
