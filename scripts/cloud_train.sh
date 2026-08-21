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
#   CW_TASK=original   CW_ENV=tworoom   CW_FAMILY=prejepa
#   CW_SEEDS=3072
#
# Submit one seed per requeueable SLURM job or array task. Comma-separated
# seeds remain available for non-SLURM serial sweeps.
#
# For the four standard original tasks, pass one dataset root and one shared
# checkpoint root as absolute paths:
#
#   CONTEXTWORLD_DATASET_ROOT=/path/to/data/world_model
#   CW_CHECKPOINT_ROOT=/path/to/dino-wm-output
#
# CW_ENV selects the task-specific YAML/data mapping. CW_DATASET remains an
# optional exact-file override for non-standard layouts. Auto-detection is a
# local convenience and never overrides an explicitly supplied path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

POST_TRAIN_EVAL=0
case "${CW_POST_TRAIN_EVAL:-0}" in
  1|true|TRUE|yes|YES|on|ON) POST_TRAIN_EVAL=1 ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "[cloud-train] CW_POST_TRAIN_EVAL must be a boolean" >&2
    exit 2
    ;;
esac

EVAL_ONLY=0
case "${CW_EVAL_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON) EVAL_ONLY=1 ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "[cloud-train] CW_EVAL_ONLY must be a boolean" >&2
    exit 2
    ;;
esac

# --- Optional umbrella root ------------------------------------------------
# The cloud mounts this as /opt/huawei/dataset/ag_data; the development box
# has an extra `explorer-env` segment. Detect rather than hardcode, so the
# same script works in both without an edit.  This umbrella root is only a
# fallback: a job that supplies the concrete paths below does not need it.
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
if [ -z "${CONTEXTWORLD_DATASET_ROOT:-}" ] && \
   [ -z "${CW_DATASET:-}" ] && [ -n "${CW_DATA_ROOT:-}" ]; then
  CONTEXTWORLD_DATASET_ROOT="$CW_DATA_ROOT/data/world_model"
fi
if [ -n "${CONTEXTWORLD_DATASET_ROOT:-}" ]; then
  export CONTEXTWORLD_DATASET_ROOT
fi

# A configured root is the public cloud contract for standard original-data
# jobs. Reject a relative value before Python can resolve it against work_dir
# and make an accidental location look valid. An exact CW_DATASET is
# authoritative and intentionally ignores a stale inherited root.
if [ "${CW_TASK:-}" = "original" ] && [ -z "${CW_DATASET:-}" ] && \
   [ -n "${CONTEXTWORLD_DATASET_ROOT:-}" ]; then
  case "$CONTEXTWORLD_DATASET_ROOT" in
    /*) ;;
    *)
      echo "[cloud-train] CONTEXTWORLD_DATASET_ROOT must be absolute: $CONTEXTWORLD_DATASET_ROOT" >&2
      exit 2
      ;;
  esac
fi

# CONTEXTWORLD_ARTIFACT_ROOT is where ContextWorld reads synthesized data and
# writes runs. Without it, contextworld.paths guesses from the checkout's
# location (repo.parents[1]/data/world_model/context_world), which is wrong
# whenever work_dir is not two levels below the data root -- exactly the
# cloud's situation.
if [ -z "${CONTEXTWORLD_ARTIFACT_ROOT:-}" ] && \
   { [ "${CW_TASK:-}" != "original" ] || [ "$POST_TRAIN_EVAL" = "1" ] || \
     [ "$EVAL_ONLY" = "1" ]; } && \
   [ -n "${CW_DATA_ROOT:-}" ]; then
  CONTEXTWORLD_ARTIFACT_ROOT="$CW_DATA_ROOT/data/world_model/context_world"
fi
if [ -n "${CONTEXTWORLD_ARTIFACT_ROOT:-}" ]; then
  export CONTEXTWORLD_ARTIFACT_ROOT
fi

# An original-data run does not read the ContextWorld artifact tree when its
# dataset is supplied explicitly. Benchmark capability runs do.
if { { [ "${CW_TASK:-}" != "original" ] && \
       [ "${CW_FAMILY:-lewm}" != "prejepa" ]; } || \
     [ "$POST_TRAIN_EVAL" = "1" ] || [ "$EVAL_ONLY" = "1" ]; } && \
   [ ! -d "${CONTEXTWORLD_ARTIFACT_ROOT:-}" ]; then
  echo "[cloud-train] benchmark artifact root does not exist: ${CONTEXTWORLD_ARTIFACT_ROOT:-<unset>}" >&2
  echo "[cloud-train] set CONTEXTWORLD_ARTIFACT_ROOT to the context_world directory" >&2
  exit 2
fi

# --- Stable-WorldModel checkout --------------------------------------------
# Training runs upstream's own scripts/train/<family>.py, which lives in a
# source checkout -- the pip-installed package does not ship it.
if [ -z "${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}" ]; then
  for candidate in \
    "${CONTEXTWORLD_ARTIFACT_ROOT:-}/upstream/stable-worldmodel-875e607fc08aa72e" \
    "${CW_DATA_ROOT:-}/data/world_model/context_world/upstream/stable-worldmodel-875e607fc08aa72e" \
    "${CW_DATA_ROOT:-}/pkg_x86/stable-worldmodel" \
    "${CW_DATA_ROOT:-}/code/stable-worldmodel"
  do
    if [ -d "$candidate/scripts/train" ]; then
      CONTEXTWORLD_STABLE_WORLDMODEL_REPO="$candidate"
      export CONTEXTWORLD_STABLE_WORLDMODEL_REPO
      break
    fi
  done
fi

# --- Explicit dataset and checkpoint paths ---------------------------------
# CW_DATASET names the exact dataset for an original-task run.  It is routed
# all the way to Stable-WorldModel; it is not merely printed in the banner.
if [ -n "${CW_DATASET:-}" ]; then
  case "$CW_DATASET" in
    /*) ;;
    *)
      echo "[cloud-train] CW_DATASET must be an absolute path: $CW_DATASET" >&2
      exit 2
      ;;
  esac
  if [ ! -e "$CW_DATASET" ]; then
    echo "[cloud-train] dataset does not exist: $CW_DATASET" >&2
    exit 2
  fi
fi

# Stable-WorldModel stores real checkpoints under
# $STABLEWM_HOME/checkpoints.  CW_OUTPUT only controls per-run Hydra/output
# files, so expose a separate, accurately named cloud variable.
if [ -n "${CW_CHECKPOINT_ROOT:-}" ]; then
  case "$CW_CHECKPOINT_ROOT" in
    /*) ;;
    *)
      echo "[cloud-train] CW_CHECKPOINT_ROOT must be absolute: $CW_CHECKPOINT_ROOT" >&2
      exit 2
      ;;
  esac
  CW_CHECKPOINT_ROOT="$(readlink -m -- "$CW_CHECKPOINT_ROOT")"
  if [ -n "${STABLEWM_HOME:-}" ] && \
     [ "$(readlink -m -- "$STABLEWM_HOME")" != "$CW_CHECKPOINT_ROOT" ]; then
    echo "[cloud-train] CW_CHECKPOINT_ROOT and STABLEWM_HOME disagree" >&2
    exit 2
  fi
  STABLEWM_HOME="$CW_CHECKPOINT_ROOT"
  export STABLEWM_HOME
fi

# StablePretraining owns optimizer/scheduler/epoch checkpointing and native
# scheduler requeue. Its default (~/.cache/stable-pretraining) is ephemeral in
# cloud containers, so persist that state beside StableWM's inference weights.
if [ -n "${STABLEWM_HOME:-}" ]; then
  STABLEWM_HOME="$(readlink -m -- "$STABLEWM_HOME")"
  export STABLEWM_HOME
  if [ -n "${SPT_CACHE_DIR:-}" ] && \
     [ "$(readlink -m -- "$SPT_CACHE_DIR")" != "$STABLEWM_HOME" ]; then
    echo "[cloud-train] SPT_CACHE_DIR and CW_CHECKPOINT_ROOT disagree" >&2
    exit 2
  fi
  SPT_CACHE_DIR="$STABLEWM_HOME"
  export SPT_CACHE_DIR
fi

if [ "${CW_TASK:-}" = "original" ]; then
  if [ -z "${CW_DATASET:-}" ] && \
     [ ! -d "${CONTEXTWORLD_DATASET_ROOT:-}" ]; then
    echo "[cloud-train] original training needs CW_DATASET or CONTEXTWORLD_DATASET_ROOT" >&2
    exit 2
  fi
fi

# The public family-profile entry handles original runs for all three
# families and benchmark PreJEPA runs. Both write real Stable-WorldModel
# checkpoints and therefore require an explicit storage root. Frozen LeWM /
# PLDM component launchers retain their own registered output contracts.
if { [ "${CW_TASK:-}" = "original" ] || \
     [ "${CW_FAMILY:-lewm}" = "prejepa" ]; } && \
   [ -z "${STABLEWM_HOME:-}" ]; then
  echo "[cloud-train] StableWM profile training needs CW_CHECKPOINT_ROOT or STABLEWM_HOME" >&2
  exit 2
fi

# --- Pretrained backbone ---------------------------------------------------
# prejepa loads facebook/dinov2-small through transformers. Point the hub
# cache at wherever the weights were placed, and prefer offline so a missing
# file fails loudly instead of silently pulling from the network mid-run.
if [ -z "${HF_HUB_CACHE:-}" ] && [ -d "${CW_DATA_ROOT:-}/models" ]; then
  export HF_HUB_CACHE="$CW_DATA_ROOT/models"
fi
if [ -n "${HF_HUB_CACHE:-}" ] && [ -z "${HF_HUB_OFFLINE:-}" ]; then
  if [ -d "$HF_HUB_CACHE/models--facebook--dinov2-small" ]; then
    export HF_HUB_OFFLINE=1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[cloud-train] checkout=$ROOT"
echo "[cloud-train] umbrella data root=${CW_DATA_ROOT:-<not needed>}"
echo "[cloud-train] stablewm=${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-<unset>}"
echo "[cloud-train] original dataset=${CW_DATASET:-<selected from ${CONTEXTWORLD_DATASET_ROOT:-upstream defaults}>}"
echo "[cloud-train] contextworld data=${CONTEXTWORLD_ARTIFACT_ROOT:-<not needed>}"
echo "[cloud-train] checkpoint root=${STABLEWM_HOME:-<upstream default>}"
echo "[cloud-train] spt cache=${SPT_CACHE_DIR:-<upstream default>}"
echo "[cloud-train] run output=${CW_OUTPUT:-<launcher default>}"
echo "[cloud-train] hf_cache=${HF_HUB_CACHE:-<default>} offline=${HF_HUB_OFFLINE:-0}"
echo "[cloud-train] logger=${CW_LOGGER:-none}"
echo "[cloud-train] post_train_eval=$POST_TRAIN_EVAL"
echo "[cloud-train] eval_only=$EVAL_ONLY"
echo "[cloud-train] eval_result_subdir=${CW_EVAL_RESULT_SUBDIR:-<default>}"
if [ "${CW_LOGGER:-none}" = "swanlab" ]; then
  # Authentication is deliberately performed by run_stablewm_train.py via
  # the Python SDK immediately before training. Passing the key to
  # `swanlab login -k ...` here would expose it in the process argument list.
  echo "[cloud-train] swanlab auth=python-sdk-before-training mode=${CW_SWANLAB_MODE:-cloud}"
fi

# Every cloud training request has one public Python entry. For a historical
# release reproduction, run_stablewm_train.py selects the frozen task recipe
# internally; it does not start a second router process.
if [ "${CW_TASK:-}" != "original" ]; then
  export CW_COMPONENT="${CW_COMPONENT:-${CW_TASK:-}}"
fi
if [ "${CW_FAMILY:-lewm}" = "prejepa" ] && \
   [ -z "${CW_BATCH_SIZE:-}" ]; then
  export CW_BATCH_SIZE=128
fi
echo "[cloud-train] route=stablewm-train"

# Huawei's startup_cce.sh exports the GUI's custom parameters and also
# repeats them as ``--NAME value`` arguments. The exported environment is the
# cloud contract; forwarding the duplicate argv would make argparse reject
# names such as --CW_FAMILY and could echo secret values in its error message.
# Keep ordinary lowercase launcher options working for local invocations,
# while consuming only platform metadata and environment-variable mirrors.
FORWARD_ARGS=()
IGNORED_PLATFORM_ARGS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --[A-Z]*|--work_dir|--work_dir=*|--run_shell_script|--run_shell_script=*|\
    --ag_data|--ag_data=*|--np|--np=*)
      IGNORED_PLATFORM_ARGS=$((IGNORED_PLATFORM_ARGS + 1))
      if [[ "$1" == *=* ]]; then
        shift
      elif [ "$#" -ge 2 ] && [[ "$2" != --* ]]; then
        shift 2
      else
        shift
      fi
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done
if [ "$IGNORED_PLATFORM_ARGS" -gt 0 ]; then
  echo "[cloud-train] ignored duplicate platform arguments=$IGNORED_PLATFORM_ARGS"
fi
exec "$PYTHON_BIN" "$ROOT/scripts/run_stablewm_train.py" "${FORWARD_ARGS[@]}"
