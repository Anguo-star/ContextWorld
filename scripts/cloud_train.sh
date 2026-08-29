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
#   CW_METHOD=native|coja_v1   (optional; defaults to native)
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

PRINT_ONLY=0
case "${CW_PRINT_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON) PRINT_ONLY=1 ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "[cloud-train] CW_PRINT_ONLY must be a boolean" >&2
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
# Three distinct data trees live under the data root, and conflating them is the
# failure this section exists to prevent:
#
#   <root>/data/world_model/quentinll/          original LeWM open data
#   <root>/data/world_model/ContextWorld-v1/    public benchmark bundle
#   <root>/data/world_model/context_world/      ContextWorld's own outputs
#                             synthesis/          synthesized benchmark data
#                             training/           checkpoints and run logs
#                             upstream/           the SWM source checkout
#
# Original-task training reads the first. Current LeWM, VIS-WM, PLDM and PreJEPA
# component training plus every public post-training ICL suite read the clean
# bundle; CEM continues to read the first tree. The internal tree is retained
# only for an explicitly selected historical LeWM/PLDM release reproduction.
# A single variable covering these roles would silently select the wrong data.
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

# Current component runs default to the common ContextWorld-v1 joint-scratch
# recipe. Historical LeWM/PLDM reproduction must be selected explicitly; it
# is the only cloud route that still needs the private artifact tree.
CW_TARGET="${CW_TASK:-${CW_COMPONENT:-}}"
CW_TRAINING_TRACK="${CW_TRAINING_TRACK:-joint_scratch_v1}"
case "$CW_TRAINING_TRACK" in
  joint_scratch_v1|historical_release) ;;
  *)
    echo "[cloud-train] CW_TRAINING_TRACK must be joint_scratch_v1 or historical_release" >&2
    exit 2
    ;;
esac
export CW_TRAINING_TRACK
HISTORICAL_RELEASE=0
if [ "$CW_TRAINING_TRACK" = "historical_release" ]; then
  HISTORICAL_RELEASE=1
  if [ -z "$CW_TARGET" ] || [ "$CW_TARGET" = "original" ]; then
    echo "[cloud-train] historical_release is valid only for a benchmark component" >&2
    exit 2
  fi
  if [ "${CW_FAMILY:-lewm}" != "lewm" ] && \
     [ "${CW_FAMILY:-lewm}" != "pldm" ]; then
    echo "[cloud-train] historical_release exists only for LeWM and PLDM" >&2
    exit 2
  fi
  if [ -n "${CW_DATASET:-}" ]; then
    echo "[cloud-train] historical_release owns its dataset; omit CW_DATASET" >&2
    exit 2
  fi
fi

# The Python launcher owns method validation and fails closed from the profile.
CW_METHOD="${CW_METHOD:-native}"
export CW_METHOD

# The clean Hugging Face export is a multi-table benchmark bundle, not one
# raw StableWM table. run_stablewm_train.py turns it into a lazy registered
# dataset view for current component training and passes the same root to all
# public Development ICL post-evaluation. Public Test remains closed.
NEEDS_BENCHMARK_ROOT=0
if [ "$HISTORICAL_RELEASE" = "0" ] && \
   { [ "$POST_TRAIN_EVAL" = "1" ] || [ "$EVAL_ONLY" = "1" ] || \
     { [ -n "$CW_TARGET" ] && [ "$CW_TARGET" != "original" ] && \
       [ -z "${CW_DATASET:-}" ]; }; }; then
  NEEDS_BENCHMARK_ROOT=1
fi
if [ -z "${CONTEXTWORLD_BENCHMARK_ROOT:-}" ] && \
   [ "$NEEDS_BENCHMARK_ROOT" = "1" ] && \
   [ -n "${CONTEXTWORLD_DATASET_ROOT:-}" ]; then
  CONTEXTWORLD_BENCHMARK_ROOT="$CONTEXTWORLD_DATASET_ROOT/ContextWorld-v1"
fi
if [ -n "${CONTEXTWORLD_BENCHMARK_ROOT:-}" ]; then
  case "$CONTEXTWORLD_BENCHMARK_ROOT" in
    /*) ;;
    *)
      echo "[cloud-train] CONTEXTWORLD_BENCHMARK_ROOT must be absolute: $CONTEXTWORLD_BENCHMARK_ROOT" >&2
      exit 2
      ;;
  esac
  export CONTEXTWORLD_BENCHMARK_ROOT
fi
if [ "$NEEDS_BENCHMARK_ROOT" = "1" ]; then
  if [ ! -f "${CONTEXTWORLD_BENCHMARK_ROOT:-}/task_registry.json" ] || \
     [ ! -f "${CONTEXTWORLD_BENCHMARK_ROOT:-}/manifest.jsonl" ] || \
     [ ! -f "${CONTEXTWORLD_BENCHMARK_ROOT:-}/manifest.sha256" ]; then
    echo "[cloud-train] ContextWorld-v1 bundle is incomplete: ${CONTEXTWORLD_BENCHMARK_ROOT:-<unset>}" >&2
    echo "[cloud-train] set CONTEXTWORLD_BENCHMARK_ROOT to the clean export root" >&2
    exit 2
  fi
fi

# CONTEXTWORLD_ARTIFACT_ROOT names the private historical archive. It is not
# an input to current training or public Development evaluation. Keep its
# path explicit for the frozen release launchers rather than letting
# contextworld.paths infer a cloud-incorrect location from the checkout.
if [ -z "${CONTEXTWORLD_ARTIFACT_ROOT:-}" ] && \
   [ "$HISTORICAL_RELEASE" = "1" ] && \
   [ -n "${CW_DATA_ROOT:-}" ]; then
  CONTEXTWORLD_ARTIFACT_ROOT="$CW_DATA_ROOT/data/world_model/context_world"
fi
if [ -n "${CONTEXTWORLD_ARTIFACT_ROOT:-}" ]; then
  export CONTEXTWORLD_ARTIFACT_ROOT
fi

# Frozen release reproductions still read the private release artifact tree.
if [ "$HISTORICAL_RELEASE" = "1" ] && \
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
if [ -z "${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}" ] && \
   [ "$HISTORICAL_RELEASE" = "1" ]; then
  for candidate in \
    "${CONTEXTWORLD_ARTIFACT_ROOT:-}/upstream/stable-worldmodel-875e607fc08aa72e" \
    "${CW_DATA_ROOT:-}/data/world_model/context_world/upstream/stable-worldmodel-875e607fc08aa72e"
  do
    if [ -d "$candidate/scripts/train" ]; then
      CONTEXTWORLD_STABLE_WORLDMODEL_REPO="$candidate"
      export CONTEXTWORLD_STABLE_WORLDMODEL_REPO
      break
    fi
  done
fi

# --- Explicit dataset and checkpoint paths ---------------------------------
# CW_DATASET names an exact custom dataset. It is routed all the way to
# Stable-WorldModel; standard current component runs should omit it so the
# manifest-bound ContextWorld-v1 view is selected.
if [ -n "${CW_DATASET:-}" ]; then
  case "$CW_DATASET" in
    /*|contextworld://v1/*) ;;
    *)
      echo "[cloud-train] CW_DATASET must be an absolute path: $CW_DATASET" >&2
      exit 2
      ;;
  esac
  if [[ "$CW_DATASET" != contextworld://v1/* ]] && [ ! -e "$CW_DATASET" ]; then
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
# ContextWorld also uses the persisted SPT last.ckpt for identity-checked
# full-state recovery when the cloud platform submits a replacement job.
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

# The current family-profile entry handles original and component runs for all
# three built-in families. They write real Stable-WorldModel checkpoints and
# therefore require an explicit storage root. Frozen historical launchers
# retain their own registered output contracts.
if [ "$HISTORICAL_RELEASE" = "0" ] && [ -z "${STABLEWM_HOME:-}" ]; then
  echo "[cloud-train] StableWM profile training needs CW_CHECKPOINT_ROOT or STABLEWM_HOME" >&2
  exit 2
fi

# A shared Stable-WorldModel checkout may be updated while a long training job
# is running. Copy only its runtime source into persistent checkpoint storage
# so training and every post-training evaluator import exactly the same files.
# This deliberately does not require the cloud image to provide the Git CLI.
# Print-only planning remains read-only; historical release recipes retain
# their own frozen source contract.
_stablewm_head() {
  local repo="$1"
  local git_dir head ref common_dir candidate
  if [ -d "$repo/.git" ]; then
    git_dir="$repo/.git"
  elif [ -f "$repo/.git" ]; then
    git_dir="$(sed -n 's/^gitdir: //p' "$repo/.git")"
    [ -n "$git_dir" ] || return 1
    case "$git_dir" in
      /*) ;;
      *) git_dir="$(readlink -m -- "$repo/$git_dir")" ;;
    esac
  else
    return 1
  fi
  [ -r "$git_dir/HEAD" ] || return 1
  head="$(tr -d '\r\n' < "$git_dir/HEAD")"
  if [[ ! "$head" == "ref: "* ]]; then
    [[ "$head" =~ ^[0-9a-f]{40}$ ]] || return 1
    printf '%s\n' "$head"
    return 0
  fi

  ref="${head#ref: }"
  common_dir="$git_dir"
  if [ -r "$git_dir/commondir" ]; then
    candidate="$(tr -d '\r\n' < "$git_dir/commondir")"
    case "$candidate" in
      /*) common_dir="$candidate" ;;
      *) common_dir="$(readlink -m -- "$git_dir/$candidate")" ;;
    esac
  fi
  for candidate in "$git_dir/$ref" "$common_dir/$ref"; do
    if [ -r "$candidate" ]; then
      head="$(tr -d '\r\n' < "$candidate")"
      [[ "$head" =~ ^[0-9a-f]{40}$ ]] || return 1
      printf '%s\n' "$head"
      return 0
    fi
  done
  if [ -r "$common_dir/packed-refs" ]; then
    head="$(awk -v ref="$ref" \
      '$1 !~ /^[#^]/ && $2 == ref { print $1; exit }' \
      "$common_dir/packed-refs")"
    [[ "$head" =~ ^[0-9a-f]{40}$ ]] || return 1
    printf '%s\n' "$head"
    return 0
  fi
  return 1
}

_stablewm_source_sha256() {
  local repo="$1"
  (
    cd "$repo"
    {
      find stable_worldmodel scripts -type f \
        ! -path '*/__pycache__/*' ! -name '*.pyc' -print0
      [ ! -f pyproject.toml ] || printf '%s\0' pyproject.toml
    } | LC_ALL=C sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

STABLEWM_LIVE_REPO="${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}"
if [ "$HISTORICAL_RELEASE" = "0" ] && [ "$PRINT_ONLY" = "0" ]; then
  if [ -z "$STABLEWM_LIVE_REPO" ]; then
    echo "[cloud-train] Stable-WorldModel checkout is not configured" >&2
    exit 2
  fi
  STABLEWM_LIVE_REPO="$(readlink -m -- "$STABLEWM_LIVE_REPO")"
  if [ ! -d "$STABLEWM_LIVE_REPO/stable_worldmodel" ] || \
     [ ! -d "$STABLEWM_LIVE_REPO/scripts/train" ] || \
     [ ! -d "$STABLEWM_LIVE_REPO/scripts/plan" ]; then
    echo "[cloud-train] Stable-WorldModel source tree is incomplete: $STABLEWM_LIVE_REPO" >&2
    exit 2
  fi

  STABLEWM_SOURCE_SHA256="$(_stablewm_source_sha256 "$STABLEWM_LIVE_REPO")"
  OBSERVED_STABLEWM_HEAD="$(_stablewm_head "$STABLEWM_LIVE_REPO" || true)"
  STABLEWM_REF_KIND="git-head"
  if [ -n "${CW_STABLEWM_REF:-}" ]; then
    STABLEWM_REF="$CW_STABLEWM_REF"
    STABLEWM_REF_KIND="explicit"
  elif [ -n "$OBSERVED_STABLEWM_HEAD" ]; then
    STABLEWM_REF="$OBSERVED_STABLEWM_HEAD"
  else
    STABLEWM_REF="${STABLEWM_SOURCE_SHA256:0:40}"
    STABLEWM_REF_KIND="content-sha256-prefix"
  fi
  if [[ ! "$STABLEWM_REF" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[cloud-train] CW_STABLEWM_REF must be a 40-digit lowercase SHA" >&2
    exit 2
  fi

  STABLEWM_SNAPSHOT_PARENT="$STABLEWM_HOME/.contextworld/stable-worldmodel"
  STABLEWM_SNAPSHOT="$STABLEWM_SNAPSHOT_PARENT/$STABLEWM_REF"
  mkdir -p "$STABLEWM_SNAPSHOT_PARENT"
  if ! command -v flock >/dev/null 2>&1; then
    echo "[cloud-train] flock is required to create the shared Stable-WorldModel snapshot safely" >&2
    exit 2
  fi
  exec 9>"$STABLEWM_SNAPSHOT_PARENT/$STABLEWM_REF.lock"
  if ! flock -w 120 9; then
    echo "[cloud-train] timed out waiting for Stable-WorldModel snapshot $STABLEWM_REF" >&2
    exit 2
  fi
  if [ -e "$STABLEWM_SNAPSHOT" ] && [ -z "${CW_STABLEWM_REF:-}" ]; then
    EXISTING_SOURCE_SHA256="$(_stablewm_source_sha256 "$STABLEWM_SNAPSHOT")"
    if [ "$EXISTING_SOURCE_SHA256" != "$STABLEWM_SOURCE_SHA256" ]; then
      echo "[cloud-train] live SWM files differ from the existing snapshot at ref $STABLEWM_REF; commit the source update before restarting" >&2
      exit 2
    fi
  fi
  if [ ! -e "$STABLEWM_SNAPSHOT" ]; then
    if [ -n "${CW_STABLEWM_REF:-}" ] && \
       [ -n "$OBSERVED_STABLEWM_HEAD" ] && \
       [ "$STABLEWM_REF" != "$OBSERVED_STABLEWM_HEAD" ]; then
      echo "[cloud-train] requested SWM ref $STABLEWM_REF is not materialized; live source is $OBSERVED_STABLEWM_HEAD" >&2
      exit 2
    fi
    STABLEWM_SNAPSHOT_TMP="$(mktemp -d \
      "$STABLEWM_SNAPSHOT_PARENT/.${STABLEWM_REF}.XXXXXX")"
    COPY_ITEMS=(stable_worldmodel scripts)
    [ ! -f "$STABLEWM_LIVE_REPO/pyproject.toml" ] || \
      COPY_ITEMS+=(pyproject.toml)
    if ! tar -C "$STABLEWM_LIVE_REPO" \
        --exclude='*/__pycache__' --exclude='*.pyc' \
        -cf - "${COPY_ITEMS[@]}" \
        | tar -C "$STABLEWM_SNAPSHOT_TMP" -xf -; then
      rm -rf -- "$STABLEWM_SNAPSHOT_TMP"
      echo "[cloud-train] failed to copy the Stable-WorldModel runtime snapshot" >&2
      exit 2
    fi
    SNAPSHOT_SOURCE_SHA256="$(_stablewm_source_sha256 "$STABLEWM_SNAPSHOT_TMP")"
    CURRENT_SOURCE_SHA256="$(_stablewm_source_sha256 "$STABLEWM_LIVE_REPO")"
    if [ "$SNAPSHOT_SOURCE_SHA256" != "$STABLEWM_SOURCE_SHA256" ] || \
       [ "$CURRENT_SOURCE_SHA256" != "$STABLEWM_SOURCE_SHA256" ]; then
      rm -rf -- "$STABLEWM_SNAPSHOT_TMP"
      echo "[cloud-train] Stable-WorldModel source changed while it was being snapshotted; restart the job" >&2
      exit 2
    fi
    mkdir -p "$STABLEWM_SNAPSHOT_TMP/.git"
    printf '%s\n' "$STABLEWM_REF" > "$STABLEWM_SNAPSHOT_TMP/.git/HEAD"
    printf '%s\n' "$STABLEWM_SOURCE_SHA256" \
      > "$STABLEWM_SNAPSHOT_TMP/.contextworld_source_sha256"
    mv -- "$STABLEWM_SNAPSHOT_TMP" "$STABLEWM_SNAPSHOT"
  fi
  SNAPSHOT_REF="$(_stablewm_head "$STABLEWM_SNAPSHOT" || true)"
  if [ "$SNAPSHOT_REF" != "$STABLEWM_REF" ] || \
     [ ! -d "$STABLEWM_SNAPSHOT/stable_worldmodel" ] || \
     [ ! -d "$STABLEWM_SNAPSHOT/scripts/train" ] || \
     [ ! -d "$STABLEWM_SNAPSHOT/scripts/plan" ]; then
    echo "[cloud-train] Stable-WorldModel snapshot is incomplete or has the wrong revision: $STABLEWM_SNAPSHOT" >&2
    exit 2
  fi
  flock -u 9
  exec 9>&-

  CONTEXTWORLD_STABLE_WORLDMODEL_REPO="$STABLEWM_SNAPSHOT"
  CW_STABLEWM_REF="$STABLEWM_REF"
  export CONTEXTWORLD_STABLE_WORLDMODEL_REPO CW_STABLEWM_REF
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
if [ -n "$STABLEWM_LIVE_REPO" ] && \
   [ "$STABLEWM_LIVE_REPO" != "${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}" ]; then
  echo "[cloud-train] stablewm live=$STABLEWM_LIVE_REPO"
  echo "[cloud-train] stablewm ref=$CW_STABLEWM_REF kind=$STABLEWM_REF_KIND"
  echo "[cloud-train] stablewm source_sha256=$STABLEWM_SOURCE_SHA256"
fi
echo "[cloud-train] stablewm=${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-<unset>}"
echo "[cloud-train] original dataset=${CW_DATASET:-<selected from ${CONTEXTWORLD_DATASET_ROOT:-upstream defaults}>}"
echo "[cloud-train] benchmark bundle=${CONTEXTWORLD_BENCHMARK_ROOT:-<not needed>}"
echo "[cloud-train] historical artifact archive=${CONTEXTWORLD_ARTIFACT_ROOT:-<not needed>}"
echo "[cloud-train] checkpoint root=${STABLEWM_HOME:-<upstream default>}"
echo "[cloud-train] spt cache=${SPT_CACHE_DIR:-<upstream default>}"
echo "[cloud-train] run output=${CW_OUTPUT:-<launcher default>}"
echo "[cloud-train] hf_cache=${HF_HUB_CACHE:-<default>} offline=${HF_HUB_OFFLINE:-0}"
echo "[cloud-train] logger=${CW_LOGGER:-none}"
echo "[cloud-train] training_track=$CW_TRAINING_TRACK"
echo "[cloud-train] method=$CW_METHOD"
echo "[cloud-train] post_train_eval=$POST_TRAIN_EVAL"
echo "[cloud-train] eval_only=$EVAL_ONLY"
echo "[cloud-train] eval_result_subdir=${CW_EVAL_RESULT_SUBDIR:-<default>}"
if [ "${CW_LOGGER:-none}" = "swanlab" ]; then
  # Authentication is deliberately performed by run_stablewm_train.py via
  # the Python SDK immediately before training. Passing the key to
  # `swanlab login -k ...` here would expose it in the process argument list.
  echo "[cloud-train] swanlab auth=python-sdk-before-training mode=${CW_SWANLAB_MODE:-cloud}"
fi

# Every cloud training request has one public Python entry. For an explicit historical
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
if [ "$HISTORICAL_RELEASE" = "0" ] && \
   [ -n "${CONTEXTWORLD_STABLE_WORLDMODEL_REPO:-}" ]; then
  # Keep a forwarded lowercase --stablewm-repo from bypassing the frozen
  # source. argparse uses the final occurrence for this scalar option.
  FORWARD_ARGS+=(
    --stablewm-repo "$CONTEXTWORLD_STABLE_WORLDMODEL_REPO"
  )
fi
exec "$PYTHON_BIN" "$ROOT/scripts/run_stablewm_train.py" "${FORWARD_ARGS[@]}"
