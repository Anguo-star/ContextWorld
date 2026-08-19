from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Literal, Mapping

from contextworld.benchmarks.cube_grasp_rule_icl_data import (
    CubeGraspRuleEvalArrays,
    _read_lance_pairs,
)
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    cube_grasp_rule_prediction_gate,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    EXPECTED_AUTHORIZATION_BASIS,
    EXPECTED_BASIS_KEYS,
    REGISTRATION_ID,
    assert_portable_tree,
    exact_value_equal,
    lexical_absolute,
    read_json,
    read_yaml,
    require_regular_file,
    resolve_no_symlink_contextworld_path,
    tree_identity,
    validate_historical_evidence,
    validate_registration_preregistration_contract,
)
from contextworld.paths import repository_root


CUBE_GRASP_RULE_V4R1_RELEASE_ID = (
    "contextworld_cube_gripper_carry_icl_history3_v4r1"
)
CUBE_GRASP_RULE_V4R1_COMPONENT_ID = "cube_gripper_carry"
DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG = repository_root() / (
    "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
)
CUBE_GRASP_RULE_V4R1_MODES = ("cannot_hold", "can_hold")
EXPECTED_TRAINING_SEEDS = (17321, 17322, 17323)
EXPECTED_RELEASE_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "component_id",
        "release_status",
        "candidate_date",
        "scope",
        "runtime",
        "identity",
        "data",
        "reference_method",
        "evaluation",
        "scoring",
        "reference_results",
        "claim_boundary",
        "distribution",
    }
)
EXPECTED_IDENTITY_KEYS = frozenset(
    {
        "public_api",
        "package",
        "adapters",
        "data_api",
        "score_api",
        "legacy_data_reader",
        "metric_api",
        "paired_latent_response_api",
        "command_line",
        "registration_contract",
        "registration_preregistration",
        "registration_freeze_receipt",
        "registration_freezer",
        "projection_packager",
        "path_resolution",
    }
)
EXPECTED_SCOPE = {
    "environment": "Cube",
    "capability": "infer_hidden_gripper_carry_rule_from_recent_interaction",
    "display_name_zh": "Cube 夹爪携带规则 ICL",
    "history_tokens": 3,
    "context_transitions": 2,
    "raw_action_dim": 5,
    "raw_steps_per_action_block": 5,
    "flattened_action_input_dim": 25,
    "prediction_horizon_action_blocks": 1,
    "grasp_modes": ["cannot_hold", "can_hold"],
    "hidden_values": {"cannot_hold": 0.0, "can_hold": 1.0},
    "public_test_included": True,
    "sealed_test_included": False,
}
EXPECTED_EXTERNAL_EVALUATION_POLICY = {
    "external_evaluation_allowed": True,
    "formal_reference_mutation": False,
    "formal_scoreboard_eligible": False,
    "reference_rerun": False,
}
EXPECTED_PROTECTED_PATHS = [
    "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
    "configs/benchmark/cube_gripper_carry_h3_v4r1_suite_registration_prereg_v1.yaml",
    "configs/benchmark/cube_gripper_carry_h3_v4r1_public_recovery_prereg_v1.yaml",
    "configs/benchmark/cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml",
    "artifacts/evaluation/history3/cube_gripper_carry_h3_v4r1_suite_registration_v1",
    "artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1",
    "artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1",
    "artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1_recovery_v1",
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1",
    "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1",
    "artifacts/evaluation/history3/cube_gripper_carry_h3_public_release_v1",
    "artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1",
]
EXPECTED_TREE = {
    "root": "artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1",
    "files": 12,
    "bytes": 203259751,
    "sha256": "2373d0cefabdc4a81d02ac18a59a1245f8ee4a02a5e2a24bb3333cb74a71ad7f",
}
EXPECTED_TREE_BEFORE_SUCCESS = {
    "files": 11,
    "bytes": 203257134,
    "sha256": "d608d684089f0287a1a7d994c7bfb0f91258de215f92dd8ec21e59be4bce2e66",
}
EXPECTED_TABLES = {
    "train": {
        "path": "train.lance",
        "source": "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1/train.lance",
        "pair_count": 2048,
        "rows": 16384,
        "files": 3,
        "bytes": 162330955,
        "sha256": "d1afff921ef7580ecb8a832514b59c6d2b000351ede7bc6e22517fd19fae0a45",
    },
    "loader_validation": {
        "path": "loader_validation.lance",
        "source": "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1/loader_validation.lance",
        "pair_count": 256,
        "rows": 2048,
        "files": 3,
        "bytes": 20354755,
        "sha256": "c51a2c74b5aa4163c5338fcf15fbf38dc2d6cda07800385ae487a60d9c2ce0d8",
    },
    "validation": {
        "path": "validation.lance",
        "source": "artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/validation.lance",
        "pair_count": 256,
        "rows": 2048,
        "files": 3,
        "bytes": 20544003,
        "sha256": "ba72017e5f47408b3cd351398a323e626f052ee1a9e252698f3c78efd550fb6f",
    },
}
EXPECTED_DATA_ARTIFACTS = {
    "packaging_started": {
        "path": EXPECTED_TREE["root"] + "/_PACKAGING_STARTED.json",
        "sha256": "6b27879f6739e4771a85dc50a09709b114ff84e777e6ab32f01867f80950e73e",
        "size_bytes": 935,
    },
    "portable_provenance": {
        "path": EXPECTED_TREE["root"] + "/portable_provenance.json",
        "sha256": "90b7a21fbcbdbc6c49911cececc687fbe35a3d2f1fb62eb01750175d5667550d",
        "size_bytes": 26486,
    },
    "success": {
        "path": EXPECTED_TREE["root"] + "/_SUCCESS.json",
        "sha256": "2ed379af0fad8c3ccc96b8174b52a8fac7fc3e1ffd1cc933960427f4b88d625c",
        "size_bytes": 2617,
    },
}
EXPECTED_CHECKPOINTS = [
    {
        "training_seed": 17321,
        "optimizer_step": 4096,
        "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/reference_training_v3/lewm_seed17321/mixed_frozen_image_paired_future_fit_1p00_step4096.pt",
        "sha256": "b7a9380bee03b057c2290a61bd71933b6ccbb693c803079a77822457b7359b70",
        "size_bytes": 72282568,
        "model_state_sha256": "9c743045571c380da71db1bdb8e6157a4adadb1cdc5cd1cc4148b07d2ad3e57d",
    },
    {
        "training_seed": 17322,
        "optimizer_step": 4096,
        "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/reference_training_v3/lewm_seed17322/mixed_frozen_image_paired_future_fit_1p00_step4096.pt",
        "sha256": "8a5c72abcb8bb068cd0d6fbfb8d1b5c81df3c32f2d77b57ba1b6b10c607c43f5",
        "size_bytes": 72282568,
        "model_state_sha256": "143be4561eb8d190bd458978c784b7a5509dd2e7f2567569292843273ab7c069",
    },
    {
        "training_seed": 17323,
        "optimizer_step": 4096,
        "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/reference_training_v3/lewm_seed17323/mixed_frozen_image_paired_future_fit_1p00_step4096.pt",
        "sha256": "cf35f84bc6776c43a735318ad55847b9de0dda32f263af36053243c51d3ad54e",
        "size_bytes": 72282568,
        "model_state_sha256": "e6d2c48fdf42557ba896199562e8896f9c8c250df7adc490644ad7116a1bab75",
    },
]
EXPECTED_NORMALIZATION = {
    "source": "upstream_cube_h5_finite_action_values_excluding_last_10000_rows",
    "mean": [
        0.010884696617722511,
        -0.003141433000564575,
        0.002646582666784525,
        0.00042392866453155875,
        0.1592525690793991,
    ],
    "std_population": [
        0.28941982984542847,
        0.393716961145401,
        0.6431365013122559,
        0.3928016126155853,
        0.2503073513507843,
    ],
    "finite_action_values": 2000000,
    "excluded_action_values": 10000,
}
EXPECTED_GATES = {
    "correct_future_rate_minimum": 0.75,
    "correct_history_rate_minimum": 0.75,
    "context_switch_rate_minimum": 0.90,
    "worst_rule_correct_future_rate_minimum": 0.70,
    "target_latent_separation_required": True,
    "response_gain_minimum": 0.50,
    "normalized_response_error_strict_maximum": 1.00,
}
EXPECTED_UNCERTAINTY = {
    "method": "paired_query_bootstrap",
    "unit": "rule_matched_query_pair",
    "resamples": 10000,
    "confidence_level": 0.95,
    "random_seed": 2026080314,
    "lower_bound_minimum": {
        "correct_future_rate": 0.70,
        "correct_history_rate": 0.70,
        "context_switch_rate": 0.85,
    },
}
EXPECTED_RETENTION = {
    "status": "passed_retention",
    "evaluation": "standard_cube_cem",
    "episodes_per_checkpoint": 300,
    "eval_seeds": [42, 43, 44],
    "episodes_per_eval_seed": 100,
    "baseline_successes": 198,
    "noninferiority_margin_successes": 15,
    "candidate_successes_by_training_seed": {
        17321: 186,
        17322: 183,
        17323: 185,
    },
    "requirement_before_positive_reference_claim": True,
}
EXPECTED_PUBLIC_TEST_STATE = {
    "generated": True,
    "hashed": True,
    "opened": True,
    "read": True,
    "scored": True,
    "used_for_training_or_selection": False,
}
EXPECTED_RECOVERY_CLAIMS = {
    "local_data_and_scoring_release_packaging_allowed": True,
    "positive_reference_public_claim_allowed": True,
    "public_test_completed": True,
    "public_test_rerun_allowed": False,
    "public_test_score_reporting_allowed": True,
    "suite_registration_allowed": False,
}
EXPECTED_PROJECTION_CLAIM_BOUNDARY = {
    "public_reference_family": "lewm",
    "pldm_public_result_included": False,
    "public_test_rerun_during_packaging": False,
    "recovery_decision_granted_suite_registration": False,
    "suite_registration_requires_separate_final_audit": True,
}
AGGREGATE_METRICS = (
    "correct_future_rate",
    "correct_history_rate",
    "context_switch_rate",
    "worst_rule_correct_future_rate",
    "other_minus_correct_mse_margin_mean",
    "joint_icl_pair_success_rate",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_identity(path: Path) -> dict[str, Any]:
    """Return a no-follow tree identity, rejecting all symlinks/special nodes."""

    return tree_identity(path)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Cube v4r1 release field {label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"Cube v4r1 release field {label} has unexpected shape: "
            f"expected {sorted(expected)}, got {sorted(value)}"
        )


def _identity_specification(value: Any, *, label: str) -> dict[str, Any]:
    specification = _mapping(value, label=label)
    _require_exact_keys(
        specification, {"path", "sha256", "size_bytes"}, label=label
    )
    path = Path(str(specification["path"]))
    sha256 = str(specification["sha256"])
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(specification["size_bytes"], bool)
        or int(specification["size_bytes"]) <= 0
    ):
        raise ValueError(f"Invalid Cube v4r1 identity: {label}")
    return specification


def _release_config_path(path: Path | str) -> Path:
    candidate = lexical_absolute(Path(path))
    require_regular_file(candidate, label="Cube v4r1 release config")
    return candidate


def load_cube_grasp_rule_v4r1_icl_release(
    path: Path | str = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = _release_config_path(path)
    payload = read_yaml(config_path, label="Cube v4r1 release config")
    _require_exact_keys(payload, EXPECTED_RELEASE_KEYS, label="top level")
    identity_header = {
        name: payload.get(name)
        for name in (
            "schema_version",
            "release_id",
            "component_id",
            "release_status",
            "candidate_date",
        )
    }
    expected_header = {
        "schema_version": 1,
        "release_id": CUBE_GRASP_RULE_V4R1_RELEASE_ID,
        "component_id": CUBE_GRASP_RULE_V4R1_COMPONENT_ID,
        "release_status": "public_test_release_candidate",
        "candidate_date": "2026-08-14",
    }
    if not exact_value_equal(identity_header, expected_header):
        raise ValueError("Cube v4r1 release identity/status drifted")
    if not exact_value_equal(payload["scope"], EXPECTED_SCOPE):
        raise ValueError("Cube v4r1 5D History=3 scope contract drifted")

    runtime = _mapping(payload["runtime"], label="runtime")
    _require_exact_keys(
        runtime,
        {"supported_adapters", "extension_contract", "contextworld", "stable_worldmodel"},
        label="runtime",
    )
    if runtime["supported_adapters"] != [
        "stable_worldmodel_lewm_cube_grasp_rule_v1",
        "stable_worldmodel_pldm_cube_grasp_rule_v1",
    ] or runtime["extension_contract"] != (
        "contextworld.benchmarks.adapters.CubeGraspRuleICLModelAdapter"
    ):
        raise ValueError("Cube v4r1 runtime adapter contract drifted")
    contextworld_runtime = _mapping(
        runtime["contextworld"], label="runtime.contextworld"
    )
    stable_runtime = _mapping(
        runtime["stable_worldmodel"], label="runtime.stable_worldmodel"
    )
    _require_exact_keys(
        contextworld_runtime, {"repository", "package_version"}, label="runtime.contextworld"
    )
    _require_exact_keys(
        stable_runtime, {"source", "repo", "expected_ref"}, label="runtime.stable_worldmodel"
    )
    if (
        contextworld_runtime["repository"]
        != "https://github.com/Anguo-star/ContextWorld"
        or contextworld_runtime["package_version"] != "0.1.0"
        or stable_runtime["source"]
        != "https://github.com/galilai-group/stable-worldmodel"
        or stable_runtime["repo"] != "../stable-worldmodel"
        or stable_runtime["expected_ref"]
        != "875e607fc08aa72eacb94d5d178127804134cc06"
    ):
        raise ValueError("Cube v4r1 runtime provenance drifted")

    identities = _mapping(payload["identity"], label="identity")
    _require_exact_keys(identities, EXPECTED_IDENTITY_KEYS, label="identity")
    for name, value in identities.items():
        _identity_specification(value, label=f"identity.{name}")

    data = _mapping(payload["data"], label="data")
    _require_exact_keys(
        data,
        {"protocol", "artifact_tree", "tree_before_success_marker", "tables", "artifacts"},
        label="data",
    )
    if data["protocol"] != "cube_gripper_carry_rule_history3_v4r1_release_projection_v1":
        raise ValueError("Cube v4r1 data protocol drifted")
    if not exact_value_equal(data["artifact_tree"], EXPECTED_TREE):
        raise ValueError("Cube v4r1 portable artifact tree drifted")
    if not exact_value_equal(
        data["tree_before_success_marker"], EXPECTED_TREE_BEFORE_SUCCESS
    ):
        raise ValueError("Cube v4r1 pre-success tree drifted")
    if not exact_value_equal(data["tables"], EXPECTED_TABLES):
        raise ValueError("Cube v4r1 portable table contract drifted")
    if not exact_value_equal(data["artifacts"], EXPECTED_DATA_ARTIFACTS):
        raise ValueError("Cube v4r1 portable marker identities drifted")
    for name, value in data["artifacts"].items():
        _identity_specification(value, label=f"data.artifacts.{name}")

    method = _mapping(payload["reference_method"], label="reference_method")
    _require_exact_keys(
        method,
        {"model_family", "method_name", "training_recipe", "fixed_checkpoint_step", "checkpoints"},
        label="reference_method",
    )
    if not exact_value_equal(
        method,
        {
            "model_family": "lewm",
            "method_name": "lewm_mixed_frozen_image_paired_future_fit_1p00",
            "training_recipe": "mixed_frozen_image_paired_future_fit_1p00",
            "fixed_checkpoint_step": 4096,
            "checkpoints": EXPECTED_CHECKPOINTS,
        },
    ):
        raise ValueError("Cube v4r1 fixed LeWM reference method drifted")

    evaluation = _mapping(payload["evaluation"], label="evaluation")
    _require_exact_keys(
        evaluation,
        {"pair_count", "lance_table", "action_normalization", "adapter_contract"},
        label="evaluation",
    )
    expected_adapter = {
        "history_tokens": 3,
        "context_transitions": 2,
        "raw_action_dim": 5,
        "raw_steps_per_action_block": 5,
        "flattened_action_input_dim": 25,
        "prediction_horizon_action_blocks": 1,
    }
    if (
        evaluation["pair_count"] != 256
        or evaluation["lance_table"] != "validation.lance"
        or not exact_value_equal(evaluation["action_normalization"], EXPECTED_NORMALIZATION)
        or not exact_value_equal(evaluation["adapter_contract"], expected_adapter)
    ):
        raise ValueError("Cube v4r1 evaluation contract drifted")

    scoring = _mapping(payload["scoring"], label="scoring")
    _require_exact_keys(
        scoring,
        {"hidden_future_prediction", "method_level", "original_task_retention"},
        label="scoring",
    )
    hidden = _mapping(
        scoring["hidden_future_prediction"], label="scoring.hidden_future_prediction"
    )
    _require_exact_keys(hidden, {"gates", "uncertainty"}, label="scoring.hidden_future_prediction")
    if not exact_value_equal(hidden["gates"], EXPECTED_GATES) or not exact_value_equal(
        hidden["uncertainty"], EXPECTED_UNCERTAINTY
    ):
        raise ValueError("Cube v4r1 prediction gates drifted")
    if not exact_value_equal(
        scoring["method_level"],
        {"training_seeds_required": 3, "all_three_checkpoints_must_pass": True},
    ):
        raise ValueError("Cube v4r1 method-level gate drifted")
    if not exact_value_equal(scoring["original_task_retention"], EXPECTED_RETENTION):
        raise ValueError("Cube v4r1 retention contract drifted")

    reference_results = _mapping(payload["reference_results"], label="reference_results")
    _require_exact_keys(reference_results, EXPECTED_BASIS_KEYS, label="reference_results")
    if not exact_value_equal(reference_results, EXPECTED_AUTHORIZATION_BASIS):
        raise ValueError("Cube v4r1 frozen historical evidence identities drifted")
    for name, value in reference_results.items():
        _identity_specification(value, label=f"reference_results.{name}")

    claim = _mapping(payload["claim_boundary"], label="claim_boundary")
    _require_exact_keys(
        claim,
        {
            "external_evaluation",
            "formal_reference_source",
            "frozen_reference_result_mutable",
            "pldm_public_reference_included",
            "historical_failed_public_v1_preserved",
            "suite_membership_requires_final_registration_audit",
            "protected_paths",
        },
        label="claim_boundary",
    )
    if (
        not exact_value_equal(
            claim["external_evaluation"], EXPECTED_EXTERNAL_EVALUATION_POLICY
        )
        or claim["formal_reference_source"]
        != "canonical_frozen_cube_public_recovery_v1"
        or claim["frozen_reference_result_mutable"] is not False
        or claim["pldm_public_reference_included"] is not False
        or claim["historical_failed_public_v1_preserved"] is not True
        or claim["suite_membership_requires_final_registration_audit"] is not True
        or not exact_value_equal(claim["protected_paths"], EXPECTED_PROTECTED_PATHS)
    ):
        raise ValueError("Cube v4r1 claim boundary drifted")
    if not exact_value_equal(
        payload["distribution"],
        {
            "channel": "local_technical_release_candidate",
            "python_package_entry_point": "contextworld-cube-gripper-carry",
            "public_download_ready": False,
            "license_bundle_ready": False,
        },
    ):
        raise ValueError("Cube v4r1 distribution boundary drifted")
    return {**payload, "_config_path": str(config_path)}


@dataclass(frozen=True)
class CubeV4R1ArtifactLayout:
    kind: Literal["source", "bundle"]
    projection_root: Path
    public_table: Path
    provenance: Path


def _resolved_logical(
    value: str, *, repo_root: Path, label: str, allow_missing: bool = False
) -> Path:
    return resolve_no_symlink_contextworld_path(
        value,
        repo_root=repo_root,
        label=label,
        allow_missing=allow_missing,
    )


def resolve_cube_grasp_rule_v4r1_layout(
    release: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    layout: Literal["auto", "source", "bundle"] = "auto",
) -> CubeV4R1ArtifactLayout:
    if layout not in {"auto", "source", "bundle"}:
        raise ValueError("layout must be auto, source or bundle")
    root = lexical_absolute(repo_root or repository_root())
    projection = _resolved_logical(
        str(release["data"]["artifact_tree"]["root"]),
        repo_root=root,
        label="Cube portable projection",
        allow_missing=True,
    )
    source_available = True
    for name, entry in release["reference_results"].items():
        candidate = _resolved_logical(
            str(entry["path"]),
            repo_root=root,
            label=f"Cube historical evidence {name}",
            allow_missing=True,
        )
        if not candidate.is_file() or candidate.is_symlink():
            source_available = False
            break
    if layout == "source" and not source_available:
        raise FileNotFoundError("Cube source evidence is incomplete")
    if layout == "bundle" and not projection.is_dir():
        raise FileNotFoundError("Cube portable projection is missing")
    selected: Literal["source", "bundle"] = (
        "source"
        if layout == "source" or (layout == "auto" and source_available)
        else "bundle"
    )
    if not projection.is_dir():
        raise FileNotFoundError("Cube portable projection is missing")
    return CubeV4R1ArtifactLayout(
        kind=selected,
        projection_root=projection,
        public_table=projection / "validation.lance",
        provenance=projection / "portable_provenance.json",
    )


class CubeGraspRuleV4R1ICLEvalDataset:
    """Frozen 256-pair Cube v4r1 Public Test from the portable projection."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
        repo_root: Path | None = None,
        layout: Literal["auto", "source", "bundle"] = "auto",
    ) -> None:
        self.repo_root = lexical_absolute(repo_root or repository_root())
        self.release = release or load_cube_grasp_rule_v4r1_icl_release(
            release_config
        )
        self.layout = resolve_cube_grasp_rule_v4r1_layout(
            self.release, repo_root=self.repo_root, layout=layout
        )
        self._arrays: CubeGraspRuleEvalArrays | None = None

    @property
    def arrays(self) -> CubeGraspRuleEvalArrays:
        if self._arrays is None:
            expected = {
                key: self.release["data"]["tables"]["validation"][key]
                for key in ("files", "bytes", "sha256")
            }
            if directory_identity(self.layout.public_table) != expected:
                raise RuntimeError("Cube v4r1 Public table identity drifted")
            arrays = _read_lance_pairs(
                self.layout.public_table,
                expected_pairs=256,
                expected_split="validation",
            )
            if arrays.raw_action_blocks.shape != (256, 4, 5, 5):
                raise RuntimeError("Cube v4r1 Public actions are not four 5x5 blocks")
            self._arrays = arrays
        return self._arrays

    def describe(self) -> dict[str, Any]:
        return {
            "artifact_layout": self.layout.kind,
            "logical_root": self.release["data"]["artifact_tree"]["root"],
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "context_transitions": 2,
            "raw_action_dim": 5,
            "raw_steps_per_action_block": 5,
            "flattened_action_input_dim": 25,
            "prediction_horizon_action_blocks": 1,
            "grasp_modes": list(CUBE_GRASP_RULE_V4R1_MODES),
            "online_environment_calls": 0,
        }


def _file_audit(
    specification: Mapping[str, Any], *, repo_root: Path, label: str
) -> dict[str, Any]:
    try:
        path = _resolved_logical(
            str(specification["path"]), repo_root=repo_root, label=label
        )
        require_regular_file(path, label=label)
        observed = file_sha256(path)
        observed_size = path.stat().st_size
        passed = bool(
            observed == specification["sha256"]
            and observed_size == int(specification["size_bytes"])
        )
        error = None
    except Exception as caught:
        path = Path(str(specification["path"]))
        observed = None
        observed_size = None
        passed = False
        error = f"{type(caught).__name__}: {caught}"
    return {
        "logical_path": specification["path"],
        "expected_sha256": specification["sha256"],
        "observed_sha256": observed,
        "expected_size_bytes": int(specification["size_bytes"]),
        "observed_size_bytes": observed_size,
        "error": error,
        "passed": passed,
    }


def _portable_json(path: Path, *, projection_root: Path, label: str) -> dict[str, Any]:
    assert_portable_tree(projection_root)
    require_regular_file(path, label=label)
    return read_json(path, label=label)


def _expected_projected_tables(release: Mapping[str, Any]) -> dict[str, Any]:
    return {
        split: {
            "source": row["source"],
            "bundled_path": row["path"],
            "pair_count": row["pair_count"],
            "rows": row["rows"],
            "files": row["files"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for split, row in release["data"]["tables"].items()
    }


def _normalized_source_public_reference(evidence: Mapping[str, Any]) -> dict[str, Any]:
    reference = dict(evidence["public_reference"])
    reference["checkpoint_results"] = [
        {
            "training_seed": int(seed),
            "model_family": "lewm",
            "training_recipe": reference["training_recipe"],
            **row,
        }
        for seed, row in sorted(
            reference["checkpoint_results"].items(), key=lambda item: int(item[0])
        )
    ]
    return reference


def _validate_projection(
    release: Mapping[str, Any],
    *,
    artifact_layout: CubeV4R1ArtifactLayout,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    observed_tree = tree_identity(artifact_layout.projection_root)
    expected_tree = {
        key: release["data"]["artifact_tree"][key]
        for key in ("files", "bytes", "sha256")
    }
    if observed_tree != expected_tree:
        raise RuntimeError("Cube portable projection tree identity drifted")
    for split, specification in release["data"]["tables"].items():
        observed = tree_identity(
            artifact_layout.projection_root / specification["path"]
        )
        expected = {
            key: specification[key] for key in ("files", "bytes", "sha256")
        }
        if observed != expected:
            raise RuntimeError(f"Cube projected table identity drifted: {split}")
    for name, specification in release["data"]["artifacts"].items():
        path = artifact_layout.projection_root / Path(specification["path"]).name
        require_regular_file(path, label=f"Cube projection marker {name}")
        if (
            file_sha256(path) != specification["sha256"]
            or path.stat().st_size != int(specification["size_bytes"])
        ):
            raise RuntimeError(f"Cube projection marker identity drifted: {name}")
    provenance = _portable_json(
        artifact_layout.provenance,
        projection_root=artifact_layout.projection_root,
        label="Cube portable provenance",
    )
    _require_exact_keys(
        provenance,
        {
            "schema_version",
            "projection_id",
            "registration_id",
            "release_id",
            "component_id",
            "status",
            "source_tables",
            "historical_evidence",
            "data_contract",
            "public_reference",
            "original_task_retention",
            "claim_boundary",
        },
        label="portable provenance",
    )
    expected_identity = {
        "schema_version": 1,
        "projection_id": "contextworld_cube_gripper_carry_h3_v4r1_projection_v1",
        "registration_id": REGISTRATION_ID,
        "release_id": release["release_id"],
        "component_id": release["component_id"],
        "status": "portable_projection_content_complete",
    }
    if not exact_value_equal(
        {key: provenance.get(key) for key in expected_identity}, expected_identity
    ):
        raise RuntimeError("Cube portable provenance identity drifted")
    if not exact_value_equal(
        provenance["source_tables"], _expected_projected_tables(release)
    ):
        raise RuntimeError("Cube projected source-table identities drifted")
    if not exact_value_equal(
        provenance["historical_evidence"], release["reference_results"]
    ):
        raise RuntimeError("Cube projected historical evidence identities drifted")
    if not exact_value_equal(
        provenance["claim_boundary"], EXPECTED_PROJECTION_CLAIM_BOUNDARY
    ):
        raise RuntimeError("Cube portable provenance claim boundary drifted")
    started = read_json(
        artifact_layout.projection_root / "_PACKAGING_STARTED.json",
        label="Cube packaging-started marker",
    )
    success = read_json(
        artifact_layout.projection_root / "_SUCCESS.json",
        label="Cube packaging-success marker",
    )
    if (
        started.get("status")
        != "portable_projection_packaging_started_namespace_consumed"
        or started.get("projection_id") != provenance["projection_id"]
        or started.get("registration_id") != REGISTRATION_ID
        or started.get("output") != release["data"]["artifact_tree"]["root"]
        or any(
            started.get(name) is not False
            for name in ("public_test_rerun", "model_or_checkpoint_read", "rerun_authorized")
        )
        or success.get("status") != "portable_release_projection_published"
        or success.get("projection_id") != provenance["projection_id"]
        or success.get("registration_id") != REGISTRATION_ID
        or success.get("release_id") != release["release_id"]
        or success.get("tree_before_success_marker")
        != release["data"]["tree_before_success_marker"]
        or success.get("source_tables") != provenance["source_tables"]
        or any(
            success.get(name) is not False
            for name in ("public_test_rerun", "model_or_checkpoint_read", "rerun_authorized")
        )
    ):
        raise RuntimeError("Cube portable projection marker state drifted")

    source_evidence: dict[str, Any] | None = None
    if artifact_layout.kind == "source":
        prereg_spec = release["identity"]["registration_preregistration"]
        prereg_path = _resolved_logical(
            str(prereg_spec["path"]),
            repo_root=repo_root,
            label="Cube Suite-registration preregistration",
        )
        if (
            file_sha256(prereg_path) != prereg_spec["sha256"]
            or prereg_path.stat().st_size != int(prereg_spec["size_bytes"])
        ):
            raise RuntimeError("Cube Suite-registration preregistration identity drifted")
        prereg = read_yaml(
            prereg_path, label="Cube Suite-registration preregistration"
        )
        validate_registration_preregistration_contract(
            prereg, preregistration_path=prereg_path
        )
        source_evidence = validate_historical_evidence(prereg, repo_root=repo_root)
        if (
            not exact_value_equal(
                _normalized_source_public_reference(source_evidence),
                provenance["public_reference"],
            )
            or not exact_value_equal(
                source_evidence["original_task_retention"],
                provenance["original_task_retention"],
            )
            or not exact_value_equal(
                source_evidence["data_contract"], provenance["data_contract"]
            )
            or not exact_value_equal(
                source_evidence["authorization_basis"],
                provenance["historical_evidence"],
            )
        ):
            raise RuntimeError("Cube source evidence and portable projection differ")
    return provenance, source_evidence


def _validate_compact_public_reference(
    release: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    reference = _mapping(
        provenance.get("public_reference"), label="projection.public_reference"
    )
    _require_exact_keys(
        reference,
        {
            "status",
            "model_family",
            "training_recipe",
            "training_seeds",
            "checkpoints_passed",
            "checkpoints_required",
            "passed",
            "aggregate",
            "checkpoint_results",
            "public_test",
            "recovery_decision",
        },
        label="projection.public_reference",
    )
    if (
        reference["status"] != "completed_one_use_public_scoring"
        or reference["model_family"] != "lewm"
        or reference["training_recipe"] != release["reference_method"]["training_recipe"]
        or reference["training_seeds"] != list(EXPECTED_TRAINING_SEEDS)
        or reference["checkpoints_passed"] != 3
        or reference["checkpoints_required"] != 3
        or reference["passed"] is not True
        or not exact_value_equal(reference["public_test"], EXPECTED_PUBLIC_TEST_STATE)
    ):
        raise RuntimeError("Cube compact Public reference state drifted")
    recovery = _mapping(
        reference["recovery_decision"], label="public_reference.recovery_decision"
    )
    if (
        recovery.get("status")
        != "public_test_release_candidate_reference_passed"
        or not exact_value_equal(recovery.get("claims"), EXPECTED_RECOVERY_CLAIMS)
        or not exact_value_equal(
            recovery.get("public_evaluation"),
            {
                "model_family": "lewm",
                "training_seeds": list(EXPECTED_TRAINING_SEEDS),
                "checkpoints_passed": 3,
                "checkpoints_required": 3,
                "passed": True,
            },
        )
    ):
        raise RuntimeError("Cube recovery decision drifted")

    rows = reference["checkpoint_results"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise RuntimeError("Cube Public reference must contain three checkpoints")
    expected = {
        int(row["training_seed"]): row
        for row in release["reference_method"]["checkpoints"]
    }
    aggregates: dict[str, list[float]] = {name: [] for name in AGGREGATE_METRICS}
    per_seed: dict[str, Any] = {}
    for row in rows:
        _require_exact_keys(
            row,
            {
                "training_seed",
                "model_family",
                "training_recipe",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "model_state_sha256",
                "metrics",
                "gate",
            },
            label="public_reference.checkpoint_result",
        )
        seed = int(row["training_seed"])
        specification = expected.get(seed)
        if specification is None or (
            row["model_family"] != "lewm"
            or row["training_recipe"] != release["reference_method"]["training_recipe"]
            or row["checkpoint_sha256"] != specification["sha256"]
            or int(row["checkpoint_size_bytes"]) != int(specification["size_bytes"])
            or row["model_state_sha256"] != specification["model_state_sha256"]
            or int(specification["optimizer_step"]) != 4096
            or not str(specification["path"]).endswith("_step4096.pt")
        ):
            raise RuntimeError(f"Cube Public checkpoint identity drifted: {seed}")
        metrics = _mapping(row["metrics"], label=f"public metrics {seed}")
        gate = cube_grasp_rule_prediction_gate(metrics, release=dict(release))
        if not exact_value_equal(row["gate"], gate) or gate["passed"] is not True:
            raise RuntimeError(f"Cube Public gate failed to reproduce: {seed}")
        for name in AGGREGATE_METRICS:
            aggregates[name].append(float(metrics[name]))
        per_seed[str(seed)] = {
            "training_seed": seed,
            "checkpoint_sha256": specification["sha256"],
            "checkpoint_size_bytes": int(specification["size_bytes"]),
            "model_state_sha256": specification["model_state_sha256"],
            "metrics": metrics,
            "gate": gate,
        }
    if sorted(int(seed) for seed in per_seed) != list(EXPECTED_TRAINING_SEEDS):
        raise RuntimeError("Cube Public checkpoint seed set drifted")
    recomputed_aggregate = {
        name: {
            "mean": float(statistics.mean(values)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }
        for name, values in aggregates.items()
    }
    if not exact_value_equal(reference["aggregate"], recomputed_aggregate):
        raise RuntimeError("Cube Public aggregate drifted")

    retention = _mapping(
        provenance.get("original_task_retention"),
        label="projection.original_task_retention",
    )
    _require_exact_keys(
        retention,
        {"status", "baseline_success_count", "comparisons", "passed"},
        label="projection.original_task_retention",
    )
    comparisons = retention["comparisons"]
    expected_candidates = EXPECTED_RETENTION["candidate_successes_by_training_seed"]
    if (
        retention["status"] != "passed_retention"
        or retention["baseline_success_count"] != 198
        or retention["passed"] is not True
        or not isinstance(comparisons, list)
        or len(comparisons) != 3
    ):
        raise RuntimeError("Cube original-task retention state drifted")
    retention_rows: list[dict[str, Any]] = []
    for row in comparisons:
        _require_exact_keys(
            row,
            {
                "training_seed",
                "checkpoint_sha256",
                "baseline_successes",
                "candidate_successes",
                "evaluation_count",
                "noninferiority_margin_successes",
                "success_delta",
                "passed",
            },
            label="original_task_retention.comparison",
        )
        seed = int(row["training_seed"])
        specification = expected.get(seed)
        candidate = expected_candidates.get(seed)
        if specification is None or candidate is None or (
            row["checkpoint_sha256"] != specification["sha256"]
            or row["baseline_successes"] != 198
            or row["candidate_successes"] != candidate
            or row["evaluation_count"] != 300
            or row["noninferiority_margin_successes"] != 15
            or row["success_delta"] != candidate - 198
            or row["passed"] is not True
            or candidate < 198 - 15
        ):
            raise RuntimeError(f"Cube retention comparison drifted: {seed}")
        retention_rows.append(dict(row))
    retention_rows.sort(key=lambda row: int(row["training_seed"]))
    if [int(row["training_seed"]) for row in retention_rows] != list(
        EXPECTED_TRAINING_SEEDS
    ):
        raise RuntimeError("Cube retention seed set drifted")
    return {
        "model_family": "lewm",
        "method_name": release["reference_method"]["method_name"],
        "training_recipe": release["reference_method"]["training_recipe"],
        "training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "per_seed": per_seed,
        "aggregate": recomputed_aggregate,
        "original_task_retention": {
            "baseline_success_count": 198,
            "comparisons": retention_rows,
            "passed": True,
        },
        "checkpoints_passed": 3,
        "checkpoints_required": 3,
        "formal_reference_source": "canonical_frozen_cube_public_recovery_v1",
        "external_result": False,
        "recomputed": True,
        "recovery_decision_granted_suite_registration": False,
        "passed": True,
    }


def recompute_cube_grasp_rule_v4r1_public_reference(
    release: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    layout: Literal["auto", "source", "bundle"] = "auto",
) -> dict[str, Any]:
    root = lexical_absolute(repo_root or repository_root())
    artifact_layout = resolve_cube_grasp_rule_v4r1_layout(
        release, repo_root=root, layout=layout
    )
    provenance, _ = _validate_projection(
        release, artifact_layout=artifact_layout, repo_root=root
    )
    return _validate_compact_public_reference(release, provenance=provenance)


def _causal_contract(provenance: Mapping[str, Any]) -> dict[str, Any]:
    data_contract = _mapping(provenance.get("data_contract"), label="data_contract")
    development = _mapping(
        data_contract.get("development_splits"), label="development_splits"
    )
    public = _mapping(data_contract.get("public_split"), label="public_split")
    expected_counts = {
        "train": 512,
        "loader_validation": 64,
        "public": 64,
    }
    split_rows = {
        "train": development.get("train", {}),
        "loader_validation": development.get("loader_validation", {}),
        "public": public,
    }
    profiles_passed = all(
        row.get("action_anchor_counts")
        == {
            "endpoint4": expected_counts[name],
            "front_hold": expected_counts[name],
            "plateau": expected_counts[name],
            "ramp4": expected_counts[name],
        }
        and row.get("profile_constraint_extrema", {}).get("probe_sum")
        == {"maximum": 0.0, "minimum": 0.0}
        and row.get("profile_constraint_extrema", {}).get("probe_final_z")
        == {"maximum": 0.0, "minimum": 0.0}
        for name, row in split_rows.items()
    )
    development_isolation = data_contract.get("development_isolation", {})
    public_isolation = data_contract.get("public_isolation", {})
    expected_public_isolation = {
        "source_episode_overlap_with_all_prior_content",
        "action_profile_overlap_with_all_prior_content",
        "scene_template_overlap_with_all_prior_content",
        "pair_content_overlap_with_all_prior_content",
        "query_pixel_overlap_with_all_prior_content",
    }
    development_disjoint = bool(development_isolation.get("passed") is True) and all(
        development_isolation.get(name, {}).get("count") == 0
        for name in (
            "exact_action_profile_id_overlap",
            "pair_content_hash_overlap",
            "query_pixel_hash_overlap",
            "scene_template_content_hash_overlap",
            "source_episode_overlap",
        )
    )
    public_disjoint = set(public_isolation) == expected_public_isolation and all(
        value == 0 for value in public_isolation.values()
    )
    table_bindings = data_contract.get("source_table_report_bindings", {})
    binding_passed = all(
        table_bindings.get(split)
        == {
            "split": split,
            "pair_count": row["pair_count"],
            "table_path": row["path"],
            "table_files": row["files"],
            "table_bytes": row["bytes"],
            "table_sha256": row["sha256"],
        }
        for split, row in EXPECTED_TABLES.items()
    )
    causal_flags = data_contract.get("causal_data_contract", {})
    checks = {
        "development_passed": data_contract.get("development_passed") is True,
        "public_passed": data_contract.get("public_passed") is True,
        "causal_pair_contract_passed": causal_flags
        == {"development_passed": True, "public_passed": True, "passed": True},
        "four_template_pair_balance": profiles_passed,
        "sum_p_zero_and_final_p_zero": profiles_passed,
        "development_split_disjoint": development_disjoint,
        "public_split_disjoint": public_disjoint,
        "source_tables_bound_to_build_reports": binding_passed,
    }
    return {"checks": checks, "passed": all(checks.values())}


def audit_cube_grasp_rule_v4r1_icl_release(
    *,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
    layout: Literal["auto", "source", "bundle"] = "auto",
) -> dict[str, Any]:
    root = lexical_absolute(repo_root or repository_root())
    release = load_cube_grasp_rule_v4r1_icl_release(release_config)
    artifact_layout = resolve_cube_grasp_rule_v4r1_layout(
        release, repo_root=root, layout=layout
    )
    identity_items = {
        name: entry
        for name, entry in release["identity"].items()
        if not (
            artifact_layout.kind == "bundle"
            and name == "registration_freeze_receipt"
        )
    }
    files = {
        f"identity.{name}": _file_audit(
            entry, repo_root=root, label=f"Cube release identity {name}"
        )
        for name, entry in identity_items.items()
    }
    files.update(
        {
            f"data.{name}": _file_audit(
                entry, repo_root=root, label=f"Cube projection artifact {name}"
            )
            for name, entry in release["data"]["artifacts"].items()
        }
    )
    if artifact_layout.kind == "source":
        files.update(
            {
                f"reference_results.{name}": _file_audit(
                    entry, repo_root=root, label=f"Cube historical evidence {name}"
                )
                for name, entry in release["reference_results"].items()
            }
        )

    try:
        observed_tree = directory_identity(artifact_layout.projection_root)
        expected_tree = {
            key: release["data"]["artifact_tree"][key]
            for key in ("files", "bytes", "sha256")
        }
        tree_passed = observed_tree == expected_tree
    except Exception as error:
        observed_tree = {"error": f"{type(error).__name__}: {error}"}
        expected_tree = {
            key: release["data"]["artifact_tree"][key]
            for key in ("files", "bytes", "sha256")
        }
        tree_passed = False

    try:
        provenance, source_evidence = _validate_projection(
            release, artifact_layout=artifact_layout, repo_root=root
        )
        projection_passed = True
    except Exception as error:
        provenance = {"error": f"{type(error).__name__}: {error}"}
        source_evidence = None
        projection_passed = False

    table_audits: dict[str, Any] = {}
    for split, specification in release["data"]["tables"].items():
        path = artifact_layout.projection_root / specification["path"]
        try:
            observed = directory_identity(path)
            expected = {
                key: specification[key] for key in ("files", "bytes", "sha256")
            }
            passed = observed == expected
            row_count = None
            if full:
                import lance

                row_count = int(lance.dataset(path).count_rows())
                passed = passed and row_count == int(specification["rows"])
            table_audits[split] = {
                "expected": {**expected, "rows": specification["rows"]},
                "observed": observed,
                "observed_rows": row_count,
                "full_row_count_checked": bool(full),
                "passed": passed,
            }
        except Exception as error:
            table_audits[split] = {
                "error": f"{type(error).__name__}: {error}", "passed": False
            }

    try:
        dataset = CubeGraspRuleV4R1ICLEvalDataset(
            release=release, repo_root=root, layout=artifact_layout.kind
        )
        description = dataset.describe()
        public_passed = (
            description["pair_count"] == 256
            and dataset.arrays.raw_action_blocks.shape == (256, 4, 5, 5)
        )
        description["passed"] = public_passed
    except Exception as error:
        description = {"error": f"{type(error).__name__}: {error}", "passed": False}

    causal = _causal_contract(provenance) if projection_passed else {
        "error": "portable projection validation failed", "passed": False
    }
    try:
        public_reference = _validate_compact_public_reference(
            release, provenance=provenance
        )
    except Exception as error:
        public_reference = {
            "error": f"{type(error).__name__}: {error}", "passed": False
        }
    passed = bool(
        all(row["passed"] for row in files.values())
        and tree_passed
        and projection_passed
        and all(row["passed"] for row in table_audits.values())
        and description.get("passed") is True
        and causal.get("passed") is True
        and public_reference.get("passed") is True
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "component_id": release["component_id"],
        "release_config": release["_config_path"],
        "artifact_layout": artifact_layout.kind,
        "full": bool(full),
        "files": files,
        "artifact_tree": {
            "expected": expected_tree,
            "observed": observed_tree,
            "passed": tree_passed,
        },
        "tables": table_audits,
        "portable_projection": {"passed": projection_passed},
        "source_historical_evidence_revalidated": source_evidence is not None,
        "source_only_registration_freeze_receipt_required": (
            artifact_layout.kind == "source"
        ),
        "public_test": description,
        "public_reference": public_reference,
        "original_task_retention": public_reference.get(
            "original_task_retention", {}
        ),
        "causal_data_contract": causal,
        "external_evaluation_policy": release["claim_boundary"]["external_evaluation"],
        "release_checks": {
            "recovery_lineage_historical_not_rerun": True,
            "pldm_formal_public_row_absent": True,
            "external_results_formal_scoreboard_ineligible": True,
            "suite_membership_requires_separate_final_audit": True,
            "local_release_candidate": True,
        },
        "status": "passed" if passed else "failed",
        "passed": passed,
    }


__all__ = [
    "CUBE_GRASP_RULE_V4R1_COMPONENT_ID",
    "CUBE_GRASP_RULE_V4R1_MODES",
    "CUBE_GRASP_RULE_V4R1_RELEASE_ID",
    "DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG",
    "EXPECTED_EXTERNAL_EVALUATION_POLICY",
    "EXPECTED_PROTECTED_PATHS",
    "CubeGraspRuleV4R1ICLEvalDataset",
    "CubeV4R1ArtifactLayout",
    "audit_cube_grasp_rule_v4r1_icl_release",
    "directory_identity",
    "file_sha256",
    "load_cube_grasp_rule_v4r1_icl_release",
    "recompute_cube_grasp_rule_v4r1_public_reference",
    "resolve_cube_grasp_rule_v4r1_layout",
]
