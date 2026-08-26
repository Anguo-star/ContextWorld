"""Activate ContextWorld's StablePretraining bridge in every DDP rank.

Python imports ``sitecustomize`` during interpreter startup.  The launcher
adds this directory to ``PYTHONPATH`` only for an upstream StableWM training
process, so Lightning child ranks receive exactly the same Manager wiring as
rank zero without modifying the StableWM checkout.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path


if os.environ.get("CONTEXTWORLD_SPT_BRIDGE") == "1":
    try:
        if os.environ.get("CONTEXTWORLD_STABLEWM_BUNDLE") == "1":
            # Lance explicitly does not support DataLoader's Linux ``fork``
            # default.  Select a clean interpreter for every worker before
            # the upstream trainer imports torch or constructs a DataLoader.
            # Lightning's ``ddp`` strategy launches ranks as subprocesses, so
            # this changes only nested worker creation for the public bundle.
            worker_start_method = os.environ.get(
                "CONTEXTWORLD_DATALOADER_START_METHOD", "spawn"
            )
            if worker_start_method != "spawn":
                raise RuntimeError(
                    "ContextWorld-v1 StableWM training requires "
                    "CONTEXTWORLD_DATALOADER_START_METHOD=spawn"
            )
            mp.set_start_method("spawn", force=True)

            from contextworld.training.stablewm_bundle import (
                register_stablewm_bundle_format,
            )

            register_stablewm_bundle_format()

        from scripts.run_stablewm_family_entry import _install_manager_bridge

        resume_value = os.environ.get("CONTEXTWORLD_SPT_RESUME_CHECKPOINT")
        _install_manager_bridge(
            run_name=os.environ["CONTEXTWORLD_SPT_RUN_NAME"],
            identity_sha256=os.environ["CONTEXTWORLD_SPT_IDENTITY_SHA256"],
            resume_checkpoint=Path(resume_value) if resume_value else None,
        )
    except BaseException as exc:
        # Python normally reports an Exception raised by sitecustomize and
        # continues startup. Continuing here could silently train from epoch
        # zero, so convert every bridge failure into a fatal startup error.
        sys.stderr.write(f"ContextWorld StableWM bootstrap failed: {exc}\n")
        raise SystemExit(2) from exc
