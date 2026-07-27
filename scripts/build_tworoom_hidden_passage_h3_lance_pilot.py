#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage import make_templates
from contextworld.evaluation.hidden_passage_lance import (
    HiddenPassageLanceCase,
    audit_actual_lewm_adapter,
    audit_hidden_passage_lance_case,
    audit_hidden_passage_lance_pairs,
    case_identity,
    collect_hidden_passage_lance_case,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_lance_pilot_v1.yaml"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_stable_worldmodel_tracked_clean(stable_repo: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(stable_repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Stable-WorldModel has tracked changes; the pilot requires the "
            "pinned implementation exactly"
        )


def _prepare_output(
    output_root: Path,
    *,
    benchmark: str,
    refresh_existing: bool,
) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    if not refresh_existing:
        raise FileExistsError(
            f"Refusing to overwrite existing pilot output {output_root}"
        )
    report_path = output_root / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "--refresh-existing requires the prior build_report.json"
        )
    old = json.loads(report_path.read_text(encoding="utf-8"))
    if old.get("benchmark") != benchmark:
        raise ValueError(
            "Refusing to replace output belonging to another benchmark"
        )
    shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect, reload and replay the TwoRoom hidden-passage "
            "History-3 Lance pilot"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    print(f"[h3-pilot] pinned StableWM {stable_commit}", flush=True)
    _assert_stable_worldmodel_tracked_clean(stable_repo)
    output_root = resolve_contextworld_path(
        args.output_root or config["output_root"],
        repo_root=ROOT,
    )
    _prepare_output(
        output_root,
        benchmark=str(config["benchmark"]),
        refresh_existing=bool(args.refresh_existing),
    )

    geometry = config["geometry"]
    pixel_codec = dict(config["storage"]["pixel_codec"])
    case_reports: list[dict] = []
    case_assets: dict[str, dict] = {}
    cases: list[HiddenPassageLanceCase] = []
    for direction, direction_config in geometry["directions"].items():
        template = make_templates(
            door_positions=[int(geometry["door_position"])],
            directions=[direction],
            doorway_offsets_px=[float(geometry["doorway_offset_px"])],
            catalog_seed=int(geometry["catalog_seed"]),
            goal_state=tuple(map(float, direction_config["goal_state"])),
        )[0]
        for rule in geometry["rules"]:
            case_id = f"{direction}-{rule}"
            print(f"[h3-pilot] collect {case_id}", flush=True)
            case = HiddenPassageLanceCase(
                case_id=case_id,
                direction=direction,
                rule=str(rule),
                table_path=output_root / "tables" / f"{case_id}.lance",
            )
            reference = collect_hidden_passage_lance_case(
                swm,
                template=template,
                rule=str(rule),
                table_path=case.table_path,
                pixel_codec=pixel_codec,
            )
            print(f"[h3-pilot] audit {case_id}", flush=True)
            case_report, assets = audit_hidden_passage_lance_case(
                swm,
                case=case,
                template=template,
                reference=reference,
                pixel_codec=pixel_codec,
            )
            case_report["table"] = portable_contextworld_path(
                case.table_path,
                repo_root=ROOT,
            )
            case_reports.append(case_report)
            case_assets[case_id] = assets
            cases.append(case)

    print("[h3-pilot] audit paired Lance clips", flush=True)
    pair_audit = audit_hidden_passage_lance_pairs(case_assets)
    adapter_config = config["model_adapter"]
    checkpoint = resolve_contextworld_path(
        adapter_config["checkpoint"],
        repo_root=ROOT,
    )
    normalizer = resolve_contextworld_path(
        adapter_config["normalizer"],
        repo_root=ROOT,
    )
    if file_sha256(checkpoint) != adapter_config["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    if file_sha256(normalizer) != adapter_config["normalizer_sha256"]:
        raise RuntimeError(f"Normalizer hash mismatch: {normalizer}")
    print("[h3-pilot] run actual LeWM adapter audit", flush=True)
    model_adapter_audit = audit_actual_lewm_adapter(
        checkpoint=checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(config["stable_worldmodel"]["repo"]),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=str(args.device or adapter_config["device"]),
        case_assets=case_assets,
    )

    checks = {
        "all_four_lance_cases_pass": (
            len(case_reports) == 4
            and all(row["passed"] for row in case_reports)
        ),
        "both_direction_pairs_pass": pair_audit["passed"],
        "actual_checkpoint_adapter_passes": model_adapter_audit["passed"],
        "stable_worldmodel_commit_exact": (
            stable_commit == config["stable_worldmodel"]["commit"]
        ),
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "claim_limit": (
            "World-to-Lance restoration and real LeWM input isolation only; "
            "this is not a hidden-passage ICL result"
        ),
        "checks": checks,
        "history3_layout": config["history3"],
        "cases": [
            {
                **case_identity(case),
                "table_path": portable_contextworld_path(
                    case.table_path,
                    repo_root=ROOT,
                ),
            }
            for case in cases
        ],
        "case_audits": case_reports,
        "pair_audit": pair_audit,
        "model_adapter_audit": model_adapter_audit,
        "identity": {
            "config": portable_contextworld_path(
                config_path,
                repo_root=ROOT,
            ),
            "config_sha256": file_sha256(config_path),
            "config_canonical_sha256": _canonical_sha256(
                {
                    key: value
                    for key, value in config.items()
                    if key != "_config_path"
                }
            ),
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
            "stable_worldmodel_tracked_tree_clean": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
        },
        "interpretation": {
            "passed_means": (
                "The exact History-3 evidence survives World collection, "
                "lossless Lance reload, hidden-rule restoration and the "
                "real LeWM adapter boundary"
            ),
            "does_not_mean": (
                "No model has yet been trained or scored for hidden-passage "
                "in-context adaptation"
            ),
            "next_gate": (
                "freeze paired single-rule and mixed-rule training data, "
                "then run the 2-by-3 next-latent Validation matrix"
            ),
        },
    }
    write_json(output_root / "build_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
