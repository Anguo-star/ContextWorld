#!/usr/bin/env python3
"""Train one Cube History-3 "can the gripper hold it?" checkpoint."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def _install_pinned_loss_compatibility(stable_loss: Any) -> dict[str, Any]:
    """Adapt only eager, unused diagnostics missing from Cube's pinned ref."""

    missing_diagnostics: list[str] = []

    class _UnavailablePinnedDiagnostic(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, *args: Any, **kwargs: Any):
            raise RuntimeError(
                "This optional diagnostic is unavailable in Cube's pinned "
                "Stable-WorldModel runtime"
            )

    for name in (
        "DynamicsResponseSIGReg",
        "GroupBalancedSIGReg",
        "ScaleCalibratedConditionalSIGReg",
    ):
        if not hasattr(stable_loss, name):
            setattr(stable_loss, name, _UnavailablePinnedDiagnostic)
            missing_diagnostics.append(name)

    conditional = stable_loss.ConditionalSIGReg
    parameters = inspect.signature(conditional.__init__).parameters
    missing_keywords = [
        name
        for name in ("include_unpaired", "complete_haar_population")
        if name not in parameters
    ]
    adapter_installed = bool(missing_keywords)
    if adapter_installed:

        class _PinnedConditionalSIGReg(conditional):
            def __init__(
                self,
                knots: int = 17,
                num_proj: int = 1024,
                randomize_pair_orientation: bool = True,
                include_unpaired: bool = False,
                complete_haar_population: bool = False,
            ) -> None:
                if include_unpaired or complete_haar_population:
                    raise RuntimeError(
                        "Cube's pinned ConditionalSIGReg adapter supports only "
                        "the legacy false/false constructor path"
                    )
                super().__init__(
                    knots=knots,
                    num_proj=num_proj,
                    randomize_pair_orientation=randomize_pair_orientation,
                )
                self.contextworld_include_unpaired = False
                self.contextworld_complete_haar_population = False

        _PinnedConditionalSIGReg.__name__ = "PinnedConditionalSIGReg"
        _PinnedConditionalSIGReg.__qualname__ = "PinnedConditionalSIGReg"
        setattr(stable_loss, "ConditionalSIGReg", _PinnedConditionalSIGReg)

    return {
        "conditional_sigreg_constructor_adapter_installed": adapter_installed,
        "conditional_sigreg_missing_keywords": missing_keywords,
        "conditional_sigreg_false_only": True,
        "unavailable_eager_diagnostic_sentinels": sorted(missing_diagnostics),
    }


_PINNED_LOSS_COMPATIBILITY: dict[str, Any] | None = None


def _prepare_cube_stable_runtime() -> Path | None:
    """Select Cube's pinned runtime without changing frozen shared trainers."""

    configured = os.environ.get("CONTEXTWORLD_STABLE_WORLDMODEL_REPO")
    if not configured:
        return None
    pinned = Path(configured).expanduser().resolve()
    release_path = ROOT / (
        "configs/benchmark/"
        "cube_gripper_carry_h3_v4r1_reference_training_prereg_v3.yaml"
    )
    if "--release-config" in sys.argv:
        try:
            release_path = Path(
                sys.argv[sys.argv.index("--release-config") + 1]
            ).expanduser().resolve()
        except IndexError as error:
            raise ValueError("--release-config requires a path") from error
    document = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("preregistration_id") != (
        "contextworld_cube_gripper_carry_h3_v4r1_reference_training_v3"
    ):
        raise ValueError("Cube reference trainer accepts only the v4r1 preregistration")
    runtime = document["runtime"]["stable_worldmodel"]
    expected_repo = Path(str(runtime["repo"])).expanduser()
    if not expected_repo.is_absolute():
        expected_repo = (ROOT / expected_repo).resolve()
    else:
        expected_repo = expected_repo.resolve()
    if pinned != expected_repo:
        raise RuntimeError(
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO does not match the Cube preregistration"
        )
    default = (ROOT.parent / "stable-worldmodel").resolve()
    if not pinned.is_dir() or pinned.is_symlink():
        raise FileNotFoundError(f"Pinned Cube Stable-WorldModel repo missing: {pinned}")
    git_environment = os.environ.copy()
    git_environment["SUDO_UID"] = str(pinned.stat().st_uid)
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={pinned}",
            "-C",
            str(pinned),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=git_environment,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={pinned}",
            "-C",
            str(pinned),
            "status",
            "--porcelain",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=git_environment,
    ).stdout
    if commit != str(runtime["expected_ref"]) or dirty:
        raise RuntimeError("Cube pinned Stable-WorldModel ref/cleanliness drifted")
    for name, entry in runtime["required_files"].items():
        path = pinned / str(entry["path"])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Cube pinned runtime file missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest != str(entry["sha256"])
            or path.stat().st_size != int(entry["size_bytes"])
        ):
            raise RuntimeError(f"Cube pinned runtime file drifted: {name}")

    # The shared PushT modules add their historical sibling repo only when it
    # is absent.  Pre-register both paths, with Cube's pinned repo first, so
    # their frozen source can remain byte-for-byte unchanged.
    for value in (str(default), str(pinned)):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    # Commit 875e predates optional PushT diagnostics and two constructor
    # switches.  The frozen shared engine instantiates those losses eagerly
    # even though Cube never selects them.  Process-local compatibility keeps
    # the shared released trainer byte-for-byte unchanged and fails loudly if
    # an unavailable semantic path is ever selected.
    from stable_worldmodel.wm import loss as stable_loss

    imported_loss = Path(stable_loss.__file__).resolve()
    if pinned not in imported_loss.parents:
        raise RuntimeError(
            f"Cube imported Stable-WorldModel loss from {imported_loss}, not {pinned}"
        )

    global _PINNED_LOSS_COMPATIBILITY
    _PINNED_LOSS_COMPATIBILITY = _install_pinned_loss_compatibility(stable_loss)
    return pinned


_PINNED_STABLE_RUNTIME = _prepare_cube_stable_runtime()

from contextworld.benchmarks.cube_grasp_rule_icl_data import (  # noqa: E402
    _read_lance_pairs,
)
from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    CUBE_REFERENCE_TRAINING_ID,
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    expected_cube_reference_training_cell,
    load_cube_reference_training_prereg,
)
import run_pusht_contact_friction_h3_train as trainer  # noqa: E402

if _PINNED_STABLE_RUNTIME is not None:
    trainer.STABLE_WORLD_MODEL_ROOT = _PINNED_STABLE_RUNTIME
    trainer.mixed.STABLE_WORLD_MODEL_ROOT = _PINNED_STABLE_RUNTIME
    trainer.pilot.STABLE_WORLD_MODEL_ROOT = _PINNED_STABLE_RUNTIME


CUBE_RAW_ACTION_DIM = 5
CUBE_ACTION_BLOCK_STEPS = 5
CUBE_ACTION_INPUT_DIM = CUBE_RAW_ACTION_DIM * CUBE_ACTION_BLOCK_STEPS
_ACTIVE_TRAINING_CONTRACT: dict[str, Any] | None = None


def _install_cube_action_dimensions() -> None:
    """Bind every shared training layer to a five-axis, five-step block.

    The shared PushT implementation defaults to two raw axes (10 flattened
    values).  Cube must replace all four dimension contracts before any data
    are materialized or a model is instantiated; changing only the outer
    wrapper would leave a latent 10-value assumption in the materialized split
    or action encoder.
    """

    trainer.ACTION_INPUT_DIM = CUBE_ACTION_INPUT_DIM
    trainer.pilot.ACTION_DIM = CUBE_RAW_ACTION_DIM
    trainer.pilot.ACTION_INPUT_DIM = CUBE_ACTION_INPUT_DIM
    trainer.mixed.ACTION_INPUT_DIM = CUBE_ACTION_INPUT_DIM


def _require_cube_action_blocks(values: np.ndarray, *, field: str) -> None:
    expected = (CUBE_ACTION_BLOCK_STEPS, CUBE_RAW_ACTION_DIM)
    if values.ndim < 2 or tuple(values.shape[-2:]) != expected:
        raise ValueError(
            f"{field} must end in five raw steps x five Cube action axes; "
            f"got {tuple(values.shape)}"
        )


def _read_cube_lance_pairs(
    path: Path, *, expected_pairs: int, expected_split: str
):
    arrays = _read_lance_pairs(
        path,
        expected_pairs=expected_pairs,
        expected_split=expected_split,
    )
    _require_cube_action_blocks(
        arrays.raw_action_blocks,
        field=f"Cube {expected_split} action blocks",
    )
    return arrays


def _finite_action_stats(path: Path) -> dict[str, Any]:
    count = 0
    total = np.zeros(CUBE_RAW_ACTION_DIM, dtype=np.float64)
    square_total = np.zeros(CUBE_RAW_ACTION_DIM, dtype=np.float64)
    with h5py.File(path, "r", swmr=True) as handle:
        actions = handle["action"]
        if actions.ndim != 2 or actions.shape[1] != CUBE_RAW_ACTION_DIM:
            raise ValueError(
                "Cube action normalization source must have shape "
                f"(rows, {CUBE_RAW_ACTION_DIM}); got {tuple(actions.shape)}"
            )
        for start in range(0, int(actions.shape[0]), 200_000):
            batch = actions[start : start + 200_000].astype(np.float64)
            batch = batch[np.isfinite(batch).all(axis=1)]
            count += len(batch)
            total += batch.sum(axis=0)
            square_total += np.square(batch).sum(axis=0)
    if not count:
        raise RuntimeError(f"No finite Cube actions in {path}")
    mean = total / count
    variance = square_total / count - np.square(mean)
    result = {
        "count": count,
        "mean": mean.astype(np.float32),
        "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "method": "population_zscore_after_excluding_nonfinite_rows",
    }
    if _ACTIVE_TRAINING_CONTRACT is not None:
        frozen = _ACTIVE_TRAINING_CONTRACT["evaluation"]["action_normalization"]
        frozen_mean = np.asarray(frozen["mean"], dtype=np.float32)
        frozen_std = np.asarray(frozen["std_population"], dtype=np.float32)
        if not np.allclose(result["mean"], frozen_mean, rtol=0.0, atol=1e-7):
            raise RuntimeError("Cube action-normalization mean drifted from preregistration")
        if not np.allclose(result["std"], frozen_std, rtol=0.0, atol=1e-7):
            raise RuntimeError("Cube action-normalization std drifted from preregistration")
    return result


def _loader_validation(
    path: Path,
    *,
    expected_pairs: int,
    action_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    arrays = _read_cube_lance_pairs(
        path,
        expected_pairs=expected_pairs,
        expected_split="loader_validation",
    )

    def pixels(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values.copy()).permute(0, 1, 4, 2, 3)

    action = torch.from_numpy(arrays.raw_action_blocks.copy()).reshape(
        expected_pairs, 4, CUBE_ACTION_INPUT_DIM
    )
    return {
        "low_pixels": pixels(arrays.cannot_hold_pixels),
        "high_pixels": pixels(arrays.can_hold_pixels),
        "action": trainer.pilot.normalize_action_blocks(
            action.float(), action_stats
        ),
        "low_states": torch.from_numpy(arrays.cannot_hold_states.copy()),
        "high_states": torch.from_numpy(arrays.can_hold_states.copy()),
    }


def _install_cube_diagnostic_name() -> None:
    original = trainer.pilot.evaluate_model

    def evaluate_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        evaluation = args[1] if len(args) > 1 else kwargs["evaluation"]
        height_gap = torch.abs(
            evaluation["high_states"][:, 3, 4]
            - evaluation["low_states"][:, 3, 4]
        )
        result.pop("physical_future_block_gap_px", None)
        result["physical_future_cube_height_gap_m"] = {
            "minimum": float(height_gap.min()),
            "mean": float(height_gap.mean()),
            "maximum": float(height_gap.max()),
        }
        return result

    trainer.pilot.evaluate_model = evaluate_model


def _load_training_contract_for_requested_model(
    path: Path | str,
) -> dict[str, Any]:
    """Load only the frozen v4r1 reference-training v3 preregistration."""

    global _ACTIVE_TRAINING_CONTRACT

    config_path = Path(path).expanduser().resolve()
    if (
        _ACTIVE_TRAINING_CONTRACT is not None
        and _ACTIVE_TRAINING_CONTRACT.get("_config_path") is not None
        and Path(_ACTIVE_TRAINING_CONTRACT["_config_path"]).resolve() == config_path
    ):
        return _ACTIVE_TRAINING_CONTRACT
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("preregistration_id") == (
        CUBE_REFERENCE_TRAINING_ID
    ):
        release = load_cube_reference_training_prereg(config_path)
        _ACTIVE_TRAINING_CONTRACT = release
        return release
    raise ValueError(
        "Cube reference trainer accepts only the frozen v4r1 v3 preregistration; "
        "legacy failed-Development replays must use their archived runner"
    )


def _install_fail_closed_formal_args() -> None:
    """Reject scientific overrides for the v4r1 v3 formal training contract."""

    original_parse_args = trainer.parse_args

    def parse_args():
        global _ACTIVE_TRAINING_CONTRACT

        _ACTIVE_TRAINING_CONTRACT = None
        args = original_parse_args()
        release_path = args.release_config.expanduser().resolve()
        if release_path != DEFAULT_CUBE_REFERENCE_TRAINING_PREREG.resolve():
            raise ValueError(
                "Cube formal training requires the canonical v4r1 v3 prereg path"
            )
        release = load_cube_reference_training_prereg(release_path, require_freeze=True)
        forbidden = {
            "--data-root": args.data_root,
            "--variant": args.variant,
            "--optimizer-steps": args.optimizer_steps,
            "--original-h5": args.original_h5,
            "--original-lance": args.original_lance,
            "--checkpoint": args.checkpoint,
            "--contrast-scales": args.contrast_scales,
        }
        used = [name for name, value in forbidden.items() if value is not None]
        if used:
            raise ValueError(
                "Cube v4r1 formal training forbids scientific input/recipe "
                "overrides: " + ", ".join(used)
            )
        common = release["training"]["reference_matrix"]["common"]
        if int(args.num_workers) != int(common["data_loader_workers"]):
            raise ValueError("Cube formal training data-loader worker count drifted")
        if int(args.eval_batch_size) != int(common["loader_validation_batch_size"]):
            raise ValueError("Cube formal training monitor batch size drifted")
        cell = expected_cube_reference_training_cell(
            release,
            model_family=args.model,
            training_seed=args.seed,
            repo_root=ROOT,
        )
        if args.output.expanduser().resolve() != Path(cell["job_root"]):
            raise ValueError("Cube formal training output is not its authorized matrix cell")
        _ACTIVE_TRAINING_CONTRACT = release
        return args

    trainer.parse_args = parse_args


def main() -> None:
    if _PINNED_STABLE_RUNTIME is None:
        raise RuntimeError(
            "Cube formal training requires CONTEXTWORLD_STABLE_WORLDMODEL_REPO"
        )
    _install_cube_action_dimensions()
    paired_future_fit_variant = "mixed_frozen_image_paired_future_fit_1p00"
    trainer.mixed.VARIANT_WEIGHTS[paired_future_fit_variant] = (
        "paired_future_fit",
        1.0,
        "paired_future_fit",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(paired_future_fit_variant)
    trainer.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_CUBE_REFERENCE_TRAINING_PREREG
    )
    trainer.load_contact_friction_icl_release = (
        _load_training_contract_for_requested_model
    )
    trainer._read_lance_pairs = _read_cube_lance_pairs
    trainer._loader_validation = _loader_validation
    # Formal inputs are resolved from the release YAML (or its documented
    # environment variables).  Non-empty argparse defaults would silently
    # bypass that portable provenance contract.
    trainer.DEFAULT_ORIGINAL_H5 = None
    trainer.DEFAULT_ORIGINAL_LANCE = None
    trainer.DEFAULT_CHECKPOINT = None
    trainer.MODEL_VARIANTS = {
        "lewm": paired_future_fit_variant,
        "pldm": "mixed_pldm_joint",
    }
    trainer.DIAGNOSTIC_VARIANTS = {
        "lewm": {paired_future_fit_variant},
        "pldm": {"mixed_pldm_joint"},
    }
    trainer.CAPABILITY_SLUG = "cube_gripper_carry"
    trainer.CAPABILITY_DISPLAY = "Cube gripper-lift-to-cube coupling"
    trainer.HIDDEN_FIELD = "hidden_grasp_enabled"
    trainer.TRAINER_DESCRIPTION = __doc__
    trainer.ORIGINAL_BATCH_KEY = "original_cube_samples_per_batch"
    trainer.ACTION_STATS_LOADER = _finite_action_stats
    _install_fail_closed_formal_args()
    _install_cube_diagnostic_name()
    trainer.main()


if __name__ == "__main__":
    main()
