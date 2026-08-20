#!/usr/bin/env bash
# Single entry point for cloud training jobs.
#
# The job template ends in `bash ${run_shell_script} "$@"`, so the platform
# holds one script path. Point run_shell_script at this file once and switch
# tasks with CW_TASK / CW_FAMILY instead of editing the job configuration.
#
#   work_dir=/path/to/ContextWorld
#   run_shell_script=scripts/cloud_train.sh
#   CW_TASK=speed CW_FAMILY=lewm CW_MODE=formal
#
# Routing lives in cloud_train.py, where it is testable. This file exists to
# satisfy the template's `bash <script>` contract and to resolve the repo
# root, so the job template need not care where the checkout landed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$ROOT/scripts/cloud_train.py" "$@"
