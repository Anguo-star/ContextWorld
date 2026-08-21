#!/usr/bin/env bash
# Single entry point for cloud training jobs.
#
# The job template ends in `bash ${run_shell_script} "$@"`, so the platform
# holds one script path and a set of environment variables -- there is no
# place to type extra commands. Everything a run needs therefore has to be
# either set in the launch GUI or defaulted here.
#
# Set in the GUI:
#
#   work_dir=/path/to/ContextWorld
#   run_shell_script=scripts/cloud_train.sh
#   CW_TASK=original   CW_ENV=tworoom   CW_FAMILY=prejepa   CW_ALL_SEEDS=1
#
# Everything below is resolved automatically and only needs setting when
# detection fails or you want a different location.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

# --- Data root -------------------------------------------------------------
# The cloud mounts this as /opt/huawei/dataset/ag_data; the development box
# has an extra `explorer-env` segment. Detect rather than hardcode, so the
# same script works in both without an edit.
if [ -z "${CW_DATA_ROOT:-}" ]; then
  for candidate in \
    /opt/huawei/dataset/ag_data \
    /opt/huawei/explorer-env/dataset/ag_data \
    "$(dirname "$(dirname "$ROOT")")"
  do
    if [ -d "$candidate/data/world_model" ]; then
      CW_DATA_ROOT="$candidate"
      break
    fi
  done
fi

if [ -z "${CW_DATA_ROOT:-}" ]; then
  echo "[cloud-train] cannot locate the data root." >&2
  echo "[cloud-train] set CW_DATA_ROOT to the directory holding data/world_model" >&2
  exit 2
fi

# --- Stable-WorldModel checkout --------------------------------------------
# Training runs upstream's own scripts/train/<family>.py, which lives in a
# source checkout -- the pip-installed package does not ship it.
if [ -z "${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}" ]; then
  for candidate in \
    "$CW_DATA_ROOT/data/world_model/context_world/upstream/stable-worldmodel-875e607fc08aa72e" \
    "$CW_DATA_ROOT/pkg_x86/stable-worldmodel" \
    "$CW_DATA_ROOT/code/stable-worldmodel"
  do
    if [ -d "$candidate/scripts/train" ]; then
      export CONTEXTWORLD_STABLE_WORLDMODEL_REPO="$candidate"
      break
    fi
  done
fi

# --- Data roots ------------------------------------------------------------
# Three distinct trees live under the data root, and conflating them is the
# failure this section exists to prevent:
#
#   <root>/data/world_model/quentinll/          original LeWM open data
#   <root>/data/world_model/context_world/      ContextWorld's own outputs
#                             synthesis/          synthesized benchmark data
#                             training/           checkpoints and run logs
#                             upstream/           the SWM source checkout
#
# Original-task training reads the first. Benchmark training reads the second.
# A single variable covering both would silently train on the wrong data, so
# they are exported separately.
#
# CONTEXTWORLD_DATASET_ROOT is the directory a dataset name is resolved
# against: `quentinll/tworoom.h5` hangs off data/world_model, so that is the
# root -- not the quentinll directory itself.
: "${CONTEXTWORLD_DATASET_ROOT:=$CW_DATA_ROOT/data/world_model}"
export CONTEXTWORLD_DATASET_ROOT

# CONTEXTWORLD_ARTIFACT_ROOT is where ContextWorld reads synthesized data and
# writes runs. Without it, contextworld.paths guesses from the checkout's
# location (repo.parents[1]/data/world_model/context_world), which is wrong
# whenever work_dir is not two levels below the data root -- exactly the
# cloud's situation.
: "${CONTEXTWORLD_ARTIFACT_ROOT:=$CONTEXTWORLD_DATASET_ROOT/context_world}"
export CONTEXTWORLD_ARTIFACT_ROOT

if [ ! -d "$CONTEXTWORLD_ARTIFACT_ROOT" ]; then
  echo "[cloud-train] artifact root does not exist: $CONTEXTWORLD_ARTIFACT_ROOT" >&2
  echo "[cloud-train] set CONTEXTWORLD_ARTIFACT_ROOT to the context_world directory" >&2
  exit 2
fi

# --- Pretrained backbone ---------------------------------------------------
# prejepa loads facebook/dinov2-small through transformers. Point the hub
# cache at wherever the weights were placed, and prefer offline so a missing
# file fails loudly instead of silently pulling from the network mid-run.
if [ -z "${HF_HUB_CACHE:-}" ] && [ -d "$CW_DATA_ROOT/models" ]; then
  export HF_HUB_CACHE="$CW_DATA_ROOT/models"
fi
if [ -n "${HF_HUB_CACHE:-}" ] && [ -z "${HF_HUB_OFFLINE:-}" ]; then
  if [ -d "$HF_HUB_CACHE/models--facebook--dinov2-small" ]; then
    export HF_HUB_OFFLINE=1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[cloud-train] data_root=$CW_DATA_ROOT"
echo "[cloud-train] stablewm=${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-<unset>}"
echo "[cloud-train] original data (datasets)=${CONTEXTWORLD_DATASET_ROOT}"
echo "[cloud-train] contextworld data (artifacts)=${CONTEXTWORLD_ARTIFACT_ROOT}"
echo "[cloud-train] hf_cache=${HF_HUB_CACHE:-<default>} offline=${HF_HUB_OFFLINE:-0}"

exec "$PYTHON_BIN" "$ROOT/scripts/cloud_train.py" "$@"
