#!/usr/bin/env python3
"""Infrastructure relaunch of the killed TwoRoom cell (ckpt 15322, eval 47).

The frozen parallel wave lost exactly one cell to an external kill (no Python
traceback, no receipt, only the 46-byte pre-CEM reservation stub).  Under the
amendment's ``infrastructure_retry_requires_new_recovery_identity`` policy this
launcher:

  * verifies the recovery preregistration and the frozen runner identity;
  * verifies the frozen cell command script flag-by-flag against the amendment
    identities pinned in the recovery preregistration;
  * verifies the stale reservation stub byte-for-byte, records it, removes it;
  * relaunches the identical frozen command once (device substitution only);
  * writes one additive, x-exclusive relaunch receipt.

It never modifies the runner, the command, thresholds, or any other cell, and
the per-seed result file is written only by the frozen runner itself.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT
    / "configs/benchmark/"
    "complete_comparison_portal_exit_pldm_seed15322_eval47_infra_relaunch_recovery_v1.yaml"
)
STUB_BYTES = b'{"seed": 47, "status": "reserved_before_cem"}\n'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "--device":
        raise SystemExit("usage: ... --device cuda:N")
    device = sys.argv[2]
    if not re.fullmatch(r"cuda:[0-7]", device):
        raise SystemExit(f"unexpected device: {device}")

    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    if prereg["recovery_id"] != (
        "complete_comparison_portal_exit_pldm_seed15322_eval47_infra_relaunch_recovery_v1"
    ):
        raise RuntimeError("Unexpected recovery preregistration")
    cell = prereg["parent_identity"]["cell"]

    # -- frozen runner identity ------------------------------------------------
    runner_spec = prereg["parent_identity"]["frozen_tworoom_runner"]
    runner = ROOT / runner_spec["path"]
    if (
        file_sha256(runner) != runner_spec["sha256"]
        or runner.stat().st_size != int(runner_spec["size_bytes"])
    ):
        raise RuntimeError("Frozen TwoRoom runner drifted; refusing to relaunch")

    # -- frozen cell command script, flag by flag ------------------------------
    template = Path(
        "/private/tmp/claude-501/-Users-wwzz-Downloads-proxyclawd/"
        "88833871-9d1c-410a-8425-a5a54e5377ef/scratchpad/driver/"
        "eval_portal_exit_pldm_15322_s47.sh"
    )
    text = template.read_text(encoding="utf-8")
    for line in ("export MUJOCO_GL=egl", "export PYTHONUNBUFFERED=1"):
        if line not in text:
            raise RuntimeError(f"Frozen command script lost `{line}`")
    flags = dict(
        re.findall(r"(--[a-z0-9-]+) (\S+)", text.replace("\\\n", " "))
    )
    output = ROOT / cell["output"]
    expected_flags = {
        "--stable-worldmodel-" + "root": cell["runtime_root"],
        "--expected-ref": cell["runtime_ref"],
        "--checkpoint": cell["checkpoint"]["path"],
        "--expected-checkpoint-sha256": cell["checkpoint"]["sha256"],
        "--expected-checkpoint-size": str(cell["checkpoint"]["size_bytes"]),
        "--expected-config-sha256": cell["checkpoint"]["config_sha256"],
        "--expected-config-size": str(cell["checkpoint"]["config_size_bytes"]),
        "--expected-catalog-sha256": cell["catalog_sha256"],
        "--expected-catalog-size": str(cell["catalog_size_bytes"]),
        "--expected-normalizer-sha256": cell["normalizer_sha256"],
        "--expected-normalizer-size": str(cell["normalizer_size_bytes"]),
        "--expected-source-sha256": cell["source_sha256"],
        "--expected-source-size": str(cell["source_size_bytes"]),
        "--seed": "47",
        "--device": "DEVICE_PLACEHOLDER",
        "--output": str(output),
    }
    for flag, expected in expected_flags.items():
        if flags.get(flag) != expected:
            raise RuntimeError(
                f"Frozen command drifted at {flag}: {flags.get(flag)!r}"
            )

    # -- stale reservation stub ------------------------------------------------
    stub_spec = prereg["failure_event"]["stale_reservation_stub"]
    stub = ROOT / stub_spec["path"]
    observed = stub.read_bytes()
    if observed != STUB_BYTES or len(observed) != int(stub_spec["size_bytes"]):
        raise RuntimeError(
            "Reservation stub is not the expected 46-byte pre-CEM marker; "
            "refusing to remove it"
        )
    stub_sha = hashlib.sha256(observed).hexdigest()
    log_spec = prereg["failure_event"]["runner_log"]
    failed_log = ROOT / log_spec["path"]
    failed_log_sha = file_sha256(failed_log)
    dmesg = subprocess.run(
        ["dmesg", "-T"], capture_output=True, text=True, check=False
    )
    oom_lines = [
        line
        for line in dmesg.stdout.splitlines()
        if re.search(r"oom|out of memory|killed process", line, re.IGNORECASE)
    ][-8:]
    stub.unlink()

    # -- relaunch --------------------------------------------------------------
    run_script = template.with_name("run_portal_exit_pldm_15322_s47_recovery_v1.sh")
    substituted = text.replace("DEVICE_PLACEHOLDER", device)
    if f"--device {device}" not in substituted.replace("\\\n  ", ""):
        raise RuntimeError("Device substitution failed")
    run_script.write_text(substituted, encoding="utf-8")
    relaunch_log = (
        ROOT
        / "artifacts/evaluation/complete_reference_comparison_v1/logs/cem/"
        "portal_exit_pldm_15322_s47_recovery_v1.log"
    )
    with relaunch_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            ["bash", str(run_script)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    receipt = {
        "schema_version": 1,
        "recovery_id": prereg["recovery_id"],
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": file_sha256(PREREG),
        },
        "failure_evidence": {
            "runner_log": {
                "path": log_spec["path"],
                "sha256": failed_log_sha,
                "python_traceback_present": False,
            },
            "stale_reservation_stub": {
                "path": stub_spec["path"],
                "sha256": stub_sha,
                "size_bytes": len(observed),
                "removed": True,
            },
            "kernel_oom_grep_tail": oom_lines,
            "result_observed_before_failure": False,
        },
        "relaunch": {
            "command_script": str(run_script),
            "template": str(template),
            "device": device,
            "pid": process.pid,
            "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "output_written_by_frozen_runner_only": True,
        },
    }
    receipt_path = ROOT / prereg["outputs"]["relaunch_receipt"]["path"]
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "sha256": file_sha256(receipt_path),
                "pid": process.pid,
                "device": device,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
