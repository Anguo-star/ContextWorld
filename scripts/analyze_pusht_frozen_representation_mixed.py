#!/usr/bin/env python3
"""Seal the frozen-representation Push-T mechanism experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world"
)
DEFAULT_ROOT = ARTIFACT_ROOT / (
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "frozen_representation_mixed_seed6145_step2048"
)
DEFAULT_NATIVE_CONTROL = ARTIFACT_ROOT / (
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048/standard_pusht_cem/"
    "mixed_native_0p09/aggregate.json"
)
TRAINING_PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/"
    "pusht_hidden_actuation_frozen_representation_mixed_v1.yaml"
)
CEM_PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/"
    "pusht_hidden_actuation_frozen_representation_standard_cem_"
    "diagnostic_v1.yaml"
)
PREDICTION_GATES = {
    "target_selection": 0.95,
    "correct_history": 0.95,
    "rule_switch": 0.95,
    "worst_mode": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--native-control",
        type=Path,
        default=DEFAULT_NATIVE_CONTROL,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only(items: list[dict[str, Any]], context: str) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"{context}: expected one item, found {len(items)}")
    return items[0]


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def paired_outcomes(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, int]:
    counts = {
        "both_success": 0,
        "candidate_only": 0,
        "control_only": 0,
        "both_fail": 0,
    }
    if len(candidate["seeds"]) != len(control["seeds"]):
        raise ValueError("Candidate and control seed counts differ")
    for candidate_seed, control_seed in zip(
        candidate["seeds"],
        control["seeds"],
    ):
        if candidate_seed["eval_seed"] != control_seed["eval_seed"]:
            raise ValueError("Candidate and control seed ordering differs")
        candidate_rows = candidate_seed["episode_successes"]
        control_rows = control_seed["episode_successes"]
        if len(candidate_rows) != len(control_rows):
            raise ValueError("Candidate and control query counts differ")
        for candidate_success, control_success in zip(
            candidate_rows,
            control_rows,
        ):
            if candidate_success and control_success:
                counts["both_success"] += 1
            elif candidate_success:
                counts["candidate_only"] += 1
            elif control_success:
                counts["control_only"] += 1
            else:
                counts["both_fail"] += 1
    return counts


def main() -> None:
    args = parse_args()
    root = Path(os.path.abspath(args.root.expanduser()))
    output = (
        Path(os.path.abspath(args.output.expanduser()))
        if args.output
        else root / "summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    paths = {
        "training_protocol": TRAINING_PROTOCOL,
        "standard_cem_diagnostic_protocol": CEM_PROTOCOL,
        "training_and_hidden_prediction": root
        / "training/mixed_report.json",
        "standard_pusht_cem_diagnostic": root
        / "standard_pusht_cem_diagnostic/aggregate.json",
        "replay_matched_native_standard_cem": (
            args.native_control.expanduser().resolve()
        ),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    training_report = load(paths["training_and_hidden_prediction"])
    training = only(training_report["results"], "training report")
    if training["variant"] != "mixed_frozen_image_native_0p09":
        raise ValueError(f"Unexpected variant {training['variant']!r}")
    if training["seed"] != 6145 or training["optimizer_steps"] != 2048:
        raise ValueError("Unexpected training seed or step count")
    final_snapshot = training["snapshots"][-1]
    if final_snapshot["optimizer_step"] != 2048:
        raise ValueError("Training report does not end at step 2048")
    initial_metrics = training["snapshots"][0]["hidden_evaluation"]
    hidden_metrics = final_snapshot["hidden_evaluation"]

    freeze = training["representation_freeze"]
    first_gradient = freeze["first_step_gradient_audit"]
    audit = {
        "frozen_modules": freeze["frozen_modules"],
        "trainable_top_level_modules": freeze[
            "trainable_top_level_modules"
        ],
        "optimizer_excludes_frozen_parameters": freeze[
            "optimizer_excludes_frozen_parameters"
        ],
        "frozen_parameters_have_no_gradient": first_gradient[
            "frozen_parameters_have_no_gradient"
        ],
        "trainable_parameters_have_nonzero_gradient": first_gradient[
            "trainable_parameters_have_nonzero_gradient"
        ],
        "native_sigreg_requires_grad": first_gradient[
            "native_sigreg_requires_grad"
        ],
        "frozen_state_sha256_before": freeze[
            "frozen_state_sha256_before"
        ],
        "frozen_state_sha256_after": freeze["frozen_state_sha256_after"],
        "frozen_state_unchanged": freeze["frozen_state_unchanged"],
        "trainable_state_changed": freeze["trainable_state_changed"],
    }
    audit["passed"] = (
        audit["frozen_modules"] == ["encoder", "projector"]
        and audit["trainable_top_level_modules"]
        == ["action_encoder", "pred_proj", "predictor"]
        and audit["optimizer_excludes_frozen_parameters"]
        and audit["frozen_parameters_have_no_gradient"]
        and audit["trainable_parameters_have_nonzero_gradient"]
        and not audit["native_sigreg_requires_grad"]
        and audit["frozen_state_unchanged"]
        and audit["trainable_state_changed"]
    )
    if not audit["passed"]:
        raise RuntimeError(f"Frozen implementation audit failed: {audit}")

    prediction = {
        "target_selection": hidden_metrics[
            "two_real_future_target_selection_rate"
        ],
        "correct_history": hidden_metrics[
            "correct_history_preference_rate"
        ],
        "rule_switch": hidden_metrics["correct_rule_switch_rate"],
        "worst_mode": hidden_metrics["worst_mode_target_selection_rate"],
        "prediction_mse": hidden_metrics["prediction_mse"],
        "paired_to_unrelated_ratio_initial": initial_metrics[
            "representation_geometry"
        ]["prediction_space"]["paired_to_unrelated_ratio"],
        "paired_to_unrelated_ratio_final": hidden_metrics[
            "representation_geometry"
        ]["prediction_space"]["paired_to_unrelated_ratio"],
    }
    prediction["frozen_geometry_exactly_stable"] = (
        prediction["paired_to_unrelated_ratio_initial"]
        == prediction["paired_to_unrelated_ratio_final"]
    )
    prediction["gate_passed"] = all(
        prediction[name] >= threshold
        for name, threshold in PREDICTION_GATES.items()
    )

    standard_report = load(paths["standard_pusht_cem_diagnostic"])
    native_report = load(paths["replay_matched_native_standard_cem"])
    if (
        standard_report["query_catalog"]["sha256"]
        != native_report["query_catalog"]["sha256"]
    ):
        raise ValueError("Standard CEM query catalogs do not match")
    standard_model = only(standard_report["models"], "standard CEM")
    native_model = only(native_report["models"], "native standard CEM")
    standard = dict(standard_model["aggregate"])
    native = dict(native_model["aggregate"])
    standard["replay_matched_native_success_count"] = native[
        "success_count"
    ]
    standard["replay_matched_native_success_rate"] = native["success_rate"]
    standard["difference_from_replay_matched_native"] = (
        standard["success_rate"] - native["success_rate"]
    )
    standard["noninferiority_margin_absolute"] = 0.05
    standard["noninferior"] = (
        standard["difference_from_replay_matched_native"] >= -0.05
    )
    standard["paired_outcomes"] = paired_outcomes(
        standard_model,
        native_model,
    )
    standard["query_catalog_sha256"] = standard_report["query_catalog"][
        "sha256"
    ]

    payload = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "experiment": "frozen_planning_representation_mixed_training_v1",
        "owner": "ContextWorld",
        "status": "complete_prediction_and_standard_cem_gates_failed",
        "method": {
            "display_name_zh": "固定规划表示的混合训练",
            "training_seed": 6145,
            "optimizer_steps": 2048,
            "frozen_modules": ["encoder", "projector"],
            "trainable_modules": [
                "predictor",
                "action_encoder",
                "pred_proj",
            ],
            "native_sigreg_weight": 0.09,
        },
        "implementation_audit": audit,
        "hidden_prediction": prediction,
        "standard_pusht_cem_diagnostic": standard,
        "hidden_actuation_cem": {
            "status": "skipped_by_registered_prediction_gate",
        },
        "decision": {
            "hidden_prediction_passed": prediction["gate_passed"],
            "standard_pusht_noninferior": standard["noninferior"],
            "jointly_passed": False,
            "interpretation": (
                "Fixing the source Encoder and Projector preserves a readable "
                "history-dependent direction and partially recovers standard "
                "CEM relative to condition-protected joint training, but it "
                "passes neither the hidden-future gate nor standard-task "
                "noninferiority. Target-representation drift is therefore "
                "not a sufficient explanation of the planning regression."
            ),
        },
        "checkpoint": training["final_checkpoint"],
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    atomic_write(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": payload["status"],
                "decision": payload["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
