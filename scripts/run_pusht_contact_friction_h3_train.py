#!/usr/bin/env python3
"""Train one PushT History-3 contact-friction ICL checkpoint.

The model sees only RGB frames and actions.  Every mixed batch contains
standard PushT replay plus complete low/high contact-friction pairs.  The
Loader Validation split is used only to monitor training; the independent
Validation split is never opened by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source_root in (
    ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.paths import (  # noqa: E402
    artifact_root,
    resolve_contextworld_path,
)
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


DEFAULT_ORIGINAL_H5: Path | None = None
DEFAULT_ORIGINAL_LANCE: Path | None = None
DEFAULT_CHECKPOINT: Path | None = None
MODEL_VARIANTS = {
    "lewm": "mixed_dynamics_response_sigreg_0p02",
    "pldm": "mixed_pldm_joint",
}
LEWM_PAIRED_FUTURE_RANKING_VARIANT = (
    "mixed_frozen_image_paired_future_ranking_1p00"
)
LEWM_PAIRED_FUTURE_MATCHING_VARIANT = (
    "mixed_frozen_image_paired_future_matching_1p00"
)
LEWM_PAIRED_FUTURE_FIT_VARIANT = (
    "mixed_frozen_image_paired_future_fit_1p00"
)
LEWM_PAIRED_FUTURE_PROJECTED_CENTER_VARIANT = (
    "mixed_frozen_image_paired_future_projected_center_1p00"
)
LEWM_PAIRED_FUTURE_RESPONSE_LOG_NORM_VARIANT = (
    "mixed_frozen_image_paired_future_response_log_norm_1p00"
)
LEWM_PAIRED_FUTURE_PROJECTED_GEOMETRY_VARIANT = (
    "mixed_frozen_image_paired_future_projected_geometry_1p00"
)
mixed.VARIANT_WEIGHTS.update(
    {
        LEWM_PAIRED_FUTURE_RANKING_VARIANT: (
            "paired_future_ranking",
            1.0,
            "paired_future_ranking",
        ),
        LEWM_PAIRED_FUTURE_MATCHING_VARIANT: (
            "paired_future_matching",
            1.0,
            "paired_future_matching",
        ),
        LEWM_PAIRED_FUTURE_FIT_VARIANT: (
            "paired_future_fit",
            1.0,
            "paired_future_fit",
        ),
        LEWM_PAIRED_FUTURE_PROJECTED_CENTER_VARIANT: (
            "paired_future_projected_center",
            1.0,
            "paired_future_projected_center",
        ),
        LEWM_PAIRED_FUTURE_RESPONSE_LOG_NORM_VARIANT: (
            "paired_future_response_log_norm",
            1.0,
            "paired_future_response_log_norm",
        ),
        LEWM_PAIRED_FUTURE_PROJECTED_GEOMETRY_VARIANT: (
            "paired_future_projected_geometry",
            1.0,
            "paired_future_projected_geometry",
        ),
    }
)
mixed.FROZEN_IMAGE_VARIANTS.update(
    {
        LEWM_PAIRED_FUTURE_RANKING_VARIANT,
        LEWM_PAIRED_FUTURE_MATCHING_VARIANT,
        LEWM_PAIRED_FUTURE_FIT_VARIANT,
        LEWM_PAIRED_FUTURE_PROJECTED_CENTER_VARIANT,
        LEWM_PAIRED_FUTURE_RESPONSE_LOG_NORM_VARIANT,
        LEWM_PAIRED_FUTURE_PROJECTED_GEOMETRY_VARIANT,
    }
)
DIAGNOSTIC_VARIANTS = {
    "lewm": {
        "mixed_native_sigreg_0p09",
        "mixed_frozen_image_native_0p09",
        "mixed_dynamics_response_sigreg_0p02",
        "mixed_dynamics_response_sigreg_0p05",
        LEWM_PAIRED_FUTURE_RANKING_VARIANT,
        LEWM_PAIRED_FUTURE_MATCHING_VARIANT,
        LEWM_PAIRED_FUTURE_FIT_VARIANT,
        LEWM_PAIRED_FUTURE_PROJECTED_CENTER_VARIANT,
        LEWM_PAIRED_FUTURE_RESPONSE_LOG_NORM_VARIANT,
        LEWM_PAIRED_FUTURE_PROJECTED_GEOMETRY_VARIANT,
    },
    "pldm": {"mixed_pldm_joint"},
}
CAPABILITY_SLUG = "contact_friction"
CAPABILITY_DISPLAY = "contact-friction"
HIDDEN_FIELD = "hidden_contact_friction"
TRAINER_DESCRIPTION = __doc__
ORIGINAL_BATCH_KEY = "original_pusht_samples_per_batch"
ACTION_STATS_LOADER = pilot.original_action_stats
ACTION_INPUT_DIM = 10


def _training_split(
    path: Path,
    *,
    expected_pairs: int,
    action_stats: dict[str, Any],
) -> pilot.MaterializedSplit:
    """Bulk-read the formal Lance table while preserving pair order."""

    arrays = _read_lance_pairs(
        path,
        expected_pairs=expected_pairs,
        expected_split="train",
    )
    shape = (
        2 * expected_pairs,
        4,
        3,
        arrays.low_pixels.shape[2],
        arrays.low_pixels.shape[3],
    )
    pixels = torch.empty(shape, dtype=torch.uint8)
    pixels[0::2] = torch.from_numpy(
        arrays.low_pixels.copy()
    ).permute(0, 1, 4, 2, 3)
    pixels[1::2] = torch.from_numpy(
        arrays.high_pixels.copy()
    ).permute(0, 1, 4, 2, 3)
    raw_action = torch.from_numpy(
        arrays.raw_action_blocks.copy()
    ).reshape(expected_pairs, 4, ACTION_INPUT_DIM)
    action = pilot.normalize_action_blocks(
        raw_action.float(),
        action_stats,
    ).repeat_interleave(2, dim=0)
    return pilot.MaterializedSplit(
        pixels=pixels,
        action=action,
        pair_count=expected_pairs,
    )


def _loader_validation(
    path: Path,
    *,
    expected_pairs: int,
    action_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    arrays = _read_lance_pairs(
        path,
        expected_pairs=expected_pairs,
        expected_split="loader_validation",
    )

    def pixels(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values.copy()).permute(0, 1, 4, 2, 3)

    action = torch.from_numpy(
        arrays.raw_action_blocks.copy()
    ).reshape(expected_pairs, 4, ACTION_INPUT_DIM)
    return {
        "low_pixels": pixels(arrays.low_pixels),
        "high_pixels": pixels(arrays.high_pixels),
        "action": pilot.normalize_action_blocks(
            action.float(),
            action_stats,
        ),
        "low_states": torch.from_numpy(arrays.low_states.copy()),
        "high_states": torch.from_numpy(arrays.high_states.copy()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TRAINER_DESCRIPTION)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Diagnostic-only compatible data root override. The formal "
            "default comes from the release contract."
        ),
    )
    parser.add_argument("--model", choices=tuple(MODEL_VARIANTS), required=True)
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Optional registered diagnostic variant. The formal default is "
            "selected from --model."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-h5", type=Path, default=DEFAULT_ORIGINAL_H5)
    parser.add_argument(
        "--original-lance",
        type=Path,
        default=DEFAULT_ORIGINAL_LANCE,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--contrast-scales",
        type=Path,
        default=None,
        help="Frozen source scales required by dynamics-response variants",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the frozen recipe without writing output",
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help=(
            "Diagnostic-only budget override. The formal default comes "
            "from the release contract."
        ),
    )
    return parser.parse_args()


def _resolve_release_input(
    specification: dict[str, Any] | str,
    *,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if isinstance(specification, str):
        return resolve_contextworld_path(specification, repo_root=ROOT)
    environment = str(specification.get("environment_variable", ""))
    configured = os.environ.get(environment) if environment else None
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = specification.get("bundled_artifact_path")
    if bundled:
        candidate = artifact_root(ROOT) / str(bundled)
        if candidate.exists():
            return candidate.resolve()
    for key in ("path", "checkpoint"):
        logical_path = specification.get(key)
        if logical_path:
            return resolve_contextworld_path(logical_path, repo_root=ROOT)
    raise ValueError(
        "Required upstream training input is not installed; set "
        f"{environment!r} or provide the bundled artifact"
    )


def _training_inputs(
    release: dict[str, Any],
    *,
    model: str,
    original_h5: Path | None,
    original_lance: Path | None,
    checkpoint: Path | None,
) -> tuple[Path, Path, Path]:
    upstream = release["training"].get("upstream", {})
    original_h5_path = _resolve_release_input(
        upstream["original_h5"], explicit=original_h5
    )
    original_lance_path = _resolve_release_input(
        upstream["original_lance"], explicit=original_lance
    )
    checkpoint_specification = upstream.get("initialization")
    if checkpoint_specification is None:
        initialization = release["training"].get("initialization")
        if initialization is not None:
            checkpoint_specification = initialization
        else:
            checkpoint_specification = release["training"][
                "reference_matrix"
            ]["initial_checkpoints"][model]
    checkpoint_path = _resolve_release_input(
        checkpoint_specification,
        explicit=checkpoint,
    )
    return original_h5_path, original_lance_path, checkpoint_path


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    matrix = release["training"]["reference_matrix"]
    followup = release["training"].get("learnability_followup", {})
    reference_open = matrix["status"] in {
        "planned_not_executed",
        "in_progress",
        "completed",
        "completed_failed_prediction_gate",
    }
    failed_development_replay = bool(
        matrix["status"] == "failed_development"
        and args.seed
        in {
            int(value)
            for value in matrix.get("completed_development_seeds", ())
        }
    )
    formal_variant = MODEL_VARIANTS[args.model]
    if failed_development_replay:
        endpoint = matrix["reported_endpoint"]
        endpoint_family = str(endpoint["model_family"]).lower()
        if endpoint_family != args.model:
            raise RuntimeError(
                "Failed-Development replay model does not match the "
                "registered endpoint"
            )
        formal_variant = str(endpoint["recipe"])
        if args.variant not in {None, formal_variant}:
            raise ValueError(
                "Failed-Development replay must use its registered recipe"
            )
    followup_open = followup.get("status") in {
        "development_recipe_search_in_progress",
        "three_seed_in_progress",
    }
    if (
        not reference_open
        and not followup_open
        and not failed_development_replay
    ):
        raise RuntimeError(
            "The frozen reference matrix permits only its registered "
            "reproduction run"
        )
    allowed_seeds = tuple(int(value) for value in matrix["training_seeds"])
    if args.seed not in allowed_seeds:
        raise ValueError(
            f"Training seed must be one of {allowed_seeds}, got {args.seed}"
        )

    common = matrix["common"]
    registered_optimizer_steps = int(common["optimizer_steps"])
    if int(common["fixed_checkpoint_step"]) != registered_optimizer_steps:
        raise RuntimeError(
            "Release optimizer_steps and fixed_checkpoint_step differ"
        )
    registered_monitor_steps = common.get(
        "loader_validation_monitor_steps",
        common.get("development_monitor_steps"),
    )
    if registered_monitor_steps is None:
        raise RuntimeError("Release is missing Development monitor steps")
    monitor_steps = tuple(int(value) for value in registered_monitor_steps)
    if (
        not monitor_steps
        or tuple(sorted(set(monitor_steps))) != monitor_steps
        or monitor_steps[-1] != registered_optimizer_steps
    ):
        raise RuntimeError(
            "Release monitor steps must be unique, increasing, and end at "
            "the fixed optimizer step"
        )
    # The shared trainer exposes snapshot points as a module-level contract.
    # Bind it to this component's release so logs, checkpoints, and the YAML
    # cannot silently disagree about a 4096- versus 8192-step run.
    mixed.SNAPSHOT_STEPS = (0, *monitor_steps)
    optimizer_steps = (
        registered_optimizer_steps
        if args.optimizer_steps is None
        else int(args.optimizer_steps)
    )
    if optimizer_steps <= 0:
        raise ValueError("--optimizer-steps must be positive")
    if args.dry_run:
        data_root = (
            resolve_contextworld_path(
                release["data"]["artifact_tree"]["root"],
                repo_root=ROOT,
            )
            if args.data_root is None
            else args.data_root.expanduser().resolve()
        )
        original_h5, original_lance, checkpoint = _training_inputs(
            release,
            model=args.model,
            original_h5=args.original_h5,
            original_lance=args.original_lance,
            checkpoint=args.checkpoint,
        )
        required = (
            data_root / "manifest.json",
            original_h5,
            original_lance,
            checkpoint,
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing training input(s):\n" + "\n".join(map(str, missing))
            )
        variant = args.variant or formal_variant
        print(
            json.dumps(
                {
                    "status": "ready",
                    "release_id": release["release_id"],
                    "model": args.model,
                    "variant": variant,
                    "seed": args.seed,
                    "optimizer_steps": optimizer_steps,
                    "data_root": str(data_root),
                    "original_h5": str(original_h5),
                    "original_lance": str(original_lance),
                    "checkpoint": str(checkpoint),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    data_root = (
        resolve_contextworld_path(
            release["data"]["artifact_tree"]["root"],
            repo_root=ROOT,
        )
        if args.data_root is None
        else args.data_root.expanduser().resolve()
    )
    data_manifest_path = data_root / "manifest.json"
    data_manifest = json.loads(
        data_manifest_path.read_text(encoding="utf-8")
    )
    if args.data_root is None:
        expected_train = int(release["data"]["pair_counts"]["train"])
        expected_development = int(
            release["data"]["pair_counts"]["loader_validation"]
        )
        train_table = release["data"]["lance_tables"]["train"]
        development_table = release["data"]["lance_tables"][
            "loader_validation"
        ]
    else:
        expected_train = int(data_manifest["pair_counts"]["train"])
        expected_development = int(
            data_manifest["pair_counts"]["loader_validation"]
        )
        train_table = data_manifest["splits"]["train"]["table_path"]
        development_table = data_manifest["splits"][
            "loader_validation"
        ]["table_path"]
        reuse = data_manifest["splits"]["loader_validation"].get(
            "frozen_split_reuse"
        )
        expected_source_manifest = release["data"]["manifest_sha256"]
        if not reuse or not all(
            (
                reuse.get("passed") is True,
                reuse.get("pair_identity_preserved") is True,
                reuse.get("model_visible_bytes_preserved") is True,
                reuse.get("source_manifest_sha256")
                == expected_source_manifest,
                reuse.get("source_table_sha256")
                == reuse.get("destination_table_sha256"),
                reuse.get("destination_table_sha256")
                == data_manifest["splits"]["loader_validation"][
                    "table_sha256"
                ],
            )
        ):
            raise RuntimeError(
                "Data-root override did not preserve the frozen "
                "Development split"
            )
    train_path = data_root / train_table
    loader_validation_path = data_root / development_table
    original_h5, original_lance, checkpoint = _training_inputs(
        release,
        model=args.model,
        original_h5=args.original_h5,
        original_lance=args.original_lance,
        checkpoint=args.checkpoint,
    )
    required = (
        data_manifest_path,
        train_path,
        loader_validation_path,
        original_h5,
        original_lance,
        checkpoint,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing training input(s):\n" + "\n".join(map(str, missing))
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    action_stats = ACTION_STATS_LOADER(original_h5)
    observed_manifest_sha256 = file_sha256(data_manifest_path)
    print(
        f"Bulk-loading {expected_train:,} "
        f"complete {CAPABILITY_DISPLAY} pairs",
        flush=True,
    )
    hidden = _training_split(
        train_path,
        expected_pairs=expected_train,
        action_stats=action_stats,
    )
    if hidden.pair_count != expected_train:
        raise RuntimeError(
            f"Expected {expected_train} train pairs, got {hidden.pair_count}"
        )
    print("Loading the isolated Loader Validation split", flush=True)
    evaluation = _loader_validation(
        loader_validation_path,
        expected_pairs=expected_development,
        action_stats=action_stats,
    )

    variant = args.variant or formal_variant
    if variant not in DIAGNOSTIC_VARIANTS[args.model]:
        raise ValueError(
            f"Variant {variant!r} is not registered for model {args.model}"
        )
    uses_contrast_scales = (
        mixed.VARIANT_WEIGHTS[variant][0] == "dynamics_response"
    )
    if uses_contrast_scales and args.contrast_scales is None:
        raise ValueError(
            "--contrast-scales is required for dynamics-response training"
        )
    contrast_scales = None
    contrast_scale_receipt = None
    if uses_contrast_scales:
        (
            contrast_scales,
            contrast_scale_receipt,
        ) = mixed.load_contrast_scales(
            args.contrast_scales.expanduser().resolve()
        )
    (output / "config.json").write_text(
        json.dumps(
            mixed.model_config(args.model),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "status": (
            f"{CAPABILITY_SLUG}_reference_training"
            if variant == formal_variant
            else f"{CAPABILITY_SLUG}_learnability_diagnostic"
        ),
        "release": {
            "path": str(release_path),
            "sha256": file_sha256(release_path),
            "release_id": release["release_id"],
        },
        "data": {
            "root": str(data_root),
            "manifest_sha256": observed_manifest_sha256,
            "release_manifest_sha256": release["data"][
                "manifest_sha256"
            ],
            "data_root_override": args.data_root is not None,
            "frozen_development_reused_from_release": (
                args.data_root is not None
            ),
            "train_pairs": hidden.pair_count,
            "loader_validation_pairs": int(
                evaluation["low_pixels"].shape[0]
            ),
            "independent_validation_opened": False,
        },
        "model": args.model,
        "variant": variant,
        "formal_reference_recipe": (
            variant == formal_variant
            and optimizer_steps == int(common["optimizer_steps"])
            and args.data_root is None
        ),
        "optimizer_steps": optimizer_steps,
        "seed": args.seed,
        "visible_fields": ["pixels", "action"],
        "forbidden_fields": [
            HIDDEN_FIELD,
            "hidden_mode",
            "physics_state",
            "physical_state",
            "pair_id",
            "catalog_index",
        ],
        "upstream": {
            "original_h5": {
                "path": str(original_h5),
                "bytes": original_h5.stat().st_size,
            },
            "original_lance": str(original_lance),
            "initial_checkpoint": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            },
        },
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    (output / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = mixed.train_variant(
        variant=variant,
        checkpoint=checkpoint,
        original_path=original_lance,
        hidden=hidden,
        evaluation=evaluation,
        action_stats=action_stats,
        output=output,
        device=device,
        seed=args.seed,
        max_steps=optimizer_steps,
        batch_size=int(common["batch_size"]),
        original_batch_size=int(
            common[ORIGINAL_BATCH_KEY]
        ),
        eval_batch_size=args.eval_batch_size,
        learning_rate=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        gradient_clip_norm=float(common["gradient_clip_norm"]),
        num_workers=args.num_workers,
        contrast_scales=contrast_scales,
        contrast_scale_receipt=contrast_scale_receipt,
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "provenance": provenance,
        "fixed_checkpoint_step": optimizer_steps,
        "loader_validation_used_for_selection": False,
        "independent_validation_used_for_selection": False,
        "result": result,
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "model": args.model,
                "seed": args.seed,
                "report": str(report_path),
                "checkpoint": result["final_checkpoint"],
                "loader_validation": result["snapshots"][-1][
                    "hidden_evaluation"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
