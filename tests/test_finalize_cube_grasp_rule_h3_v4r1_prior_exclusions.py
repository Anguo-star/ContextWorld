from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.finalize_cube_grasp_rule_h3_v4r1_prior_exclusions as finalizer


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _content_entry(values: list[str], field: str) -> dict[str, Any]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": finalizer.canonical_content_digest(values, field_name=field),
    }


def _source_entry(values: list[int]) -> dict[str, Any]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": finalizer.excluded_source_episodes_sha256(values),
    }


def _closed_public() -> dict[str, Any]:
    return {
        "access_status": "closed_not_read_not_scored",
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(finalizer, "EXPECTED_OLD_SOURCE_COUNT", 2)
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_OLD_CONTENT_COUNTS",
        {field: 2 for field in finalizer.CONTENT_FIELDS},
    )
    monkeypatch.setattr(finalizer, "EXPECTED_FAILED_COUNT", 2)
    monkeypatch.setattr(finalizer, "EXPECTED_FINAL_SOURCE_COUNT", 4)
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_FINAL_CONTENT_COUNTS",
        {field: 4 for field in finalizer.CONTENT_FIELDS},
    )
    paths = {
        "old": tmp_path / "old_prior.json",
        "failed": tmp_path / "failed_attempt.json",
        "query": tmp_path / "query_reconstruction.json",
        "prereg": tmp_path / "v4r1_prereg.yaml",
        "freeze": tmp_path / "v4r1_freeze.json",
        "output": tmp_path / "v4r1_prior_final.json",
    }
    source = {
        "symbol": "upstream_cube_single_expert_h5",
        "sha256": _digest("source-h5"),
        "size_bytes": 123456,
        "row_count": 1000,
        "episode_count": 100,
        "path_recorded": False,
    }
    old_episodes = [0, 1]
    failed_episodes = [10, 11]
    old_content = {
        field: [_digest(f"old-{field}-{index}") for index in range(2)]
        for field in finalizer.CONTENT_FIELDS
    }
    failed_content = {
        field: [_digest(f"failed-{field}-{index}") for index in range(2)]
        for field in finalizer.CONTENT_FIELDS
    }
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_FINAL_SOURCE_SHA256",
        finalizer.excluded_source_episodes_sha256(
            sorted(old_episodes + failed_episodes)
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_FINAL_CONTENT_SHA256",
        {
            field: finalizer.canonical_content_digest(
                sorted(old_content[field] + failed_content[field]),
                field_name=field,
            )
            for field in finalizer.CONTENT_FIELDS
        },
    )
    old = {
        "schema_version": 1,
        "protocol_id": finalizer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": finalizer.OLD_PRIOR_RECEIPT_ID,
        "status": finalizer.OLD_PRIOR_STATUS,
        "checks_passed": True,
        "preregistration": {
            "path": "configs/benchmark/old-v4.yaml",
            "sha256": _digest("old-prereg"),
            "size_bytes": 111,
        },
        "freeze_receipt": {
            "path": "artifacts/evaluation/old-v4-freeze.json",
            "sha256": _digest("old-freeze"),
            "size_bytes": 222,
        },
        "source_h5": source,
        "coverage": {field: True for field in finalizer.OLD_COVERAGE_FIELDS},
        "input_artifacts": [
            {
                "role": "v3_formal",
                "path": "artifacts/evaluation/v3/formal.json",
                "sha256": _digest("old-artifact"),
                "size_bytes": 333,
                "preserved_extension": {"exact": True},
            }
        ],
        "excluded_source_episodes": old_episodes,
        "excluded_source_episode_count": len(old_episodes),
        "excluded_source_episodes_sha256": (
            finalizer.excluded_source_episodes_sha256(old_episodes)
        ),
        "prior_content_exclusions": {
            field: _content_entry(values, field)
            for field, values in old_content.items()
        },
        "public_test": _closed_public(),
        "reference_model_training_or_scoring": False,
    }
    _write_json(paths["old"], old)
    old_identity = _identity(paths["old"])
    monkeypatch.setattr(
        finalizer, "EXPECTED_OLD_PRIOR_SHA256", old_identity["sha256"]
    )

    failed_pairs = []
    jpeg_values = []
    for index in range(2):
        jpeg = _digest(f"jpeg-{index}")
        jpeg_values.append(jpeg)
        failed_pairs.append(
            {
                "pair_id": f"cube-carry-v4-train-{index:06d}",
                "catalog_index": 1_000_000 + index,
                "source_row": 100 + index,
                "source_episode": failed_episodes[index],
                "source_step": 2 + index,
                "action_anchor_id": ("endpoint4", "plateau")[index],
                "action_profile_id": failed_content["action_profile_ids"][index],
                "scene_template_content_hash": failed_content[
                    "scene_template_content_hashes"
                ][index],
                "pair_content_hash": failed_content["pair_content_hashes"][index],
                "query_jpeg_sha256": jpeg,
            }
        )
    failed_inputs = {
        "preregistration": copy.deepcopy(old["preregistration"]),
        "freeze_receipt": copy.deepcopy(old["freeze_receipt"]),
        "prior_exclusion_receipt": old_identity,
        "builder_snapshot": {
            "sha256": _digest("builder"),
            "size_bytes": 444,
        },
        "request_json": {"sha256": _digest("request"), "size_bytes": 555},
        "partial_train_fragment": {
            "sha256": _digest("fragment"),
            "size_bytes": 666,
        },
        "source_h5": copy.deepcopy(source),
    }
    failed = {
        "schema_version": 1,
        "protocol_id": finalizer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": finalizer.FAILED_RECEIPT_ID,
        "status": finalizer.FAILED_RECEIPT_STATUS,
        "checks_passed": True,
        "build_passed": False,
        "formal_build_attempt_consumed": True,
        "retry_authorized_under_original_preregistration": False,
        "input_identities": failed_inputs,
        "failed_attempt_content": {
            "split": "train",
            "row_count": 16,
            "episode_count": 4,
            "pair_count": 2,
            "source_episodes": _source_entry(failed_episodes),
            "prior_content_exclusions": {
                field: _content_entry(failed_content[field], field)
                for field in finalizer.DIRECT_FAILED_CONTENT_FIELDS
            },
            "query_pixel_hash_status": (
                "pending_deterministic_raw_reconstruction_not_present_in_fragment"
            ),
            "query_jpeg_sha256": {
                "values": sorted(jpeg_values),
                "count": 2,
                "sha256": finalizer.forensic_query_jpeg_digest(
                    sorted(jpeg_values)
                ),
                "digest_namespace": (
                    finalizer.FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE
                ),
                "role": "forensic_binding_only_not_raw_query_pixel_hash",
            },
            "pairs": failed_pairs,
            "profile_constraints": {"passed": True},
            "prior_overlap": {
                "source_episode_count": 0,
                "action_profile_id_count": 0,
                "scene_template_content_hash_count": 0,
                "pair_content_hash_count": 0,
                "query_pixel_hash_count": None,
                "query_pixel_hash_overlap_status": (
                    "not_computable_until_raw_query_reconstruction"
                ),
                "passed_for_directly_inspectable_identities": True,
            },
        },
        "scope": {
            "public_test": _closed_public(),
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
        },
        "recovery_policy": {
            key: True
            for key in (
                "original_v4_preregistration_attempt_budget_exhausted",
                "original_failed_tree_must_remain_immutable",
                "silent_retry_or_overwrite_forbidden",
                "newly_frozen_recovery_preregistration_required",
                "failed_source_action_scene_pair_and_reconstructed_raw_query_must_be_excluded",
            )
        },
    }
    _write_json(paths["failed"], failed)
    failed_identity = _identity(paths["failed"])
    monkeypatch.setattr(
        finalizer, "EXPECTED_FAILED_ATTEMPT_SHA256", failed_identity["sha256"]
    )

    query_inputs = copy.deepcopy(failed_inputs)
    query_inputs["failed_attempt_receipt"] = failed_identity
    query_inputs["physics_snapshot"] = {
        "sha256": _digest("physics"),
        "size_bytes": 777,
    }
    query_pairs = []
    for index, pair in enumerate(failed_pairs):
        query_pairs.append(
            {
                **pair,
                "split": "train",
                "raw_query_pixel_hash": failed_content["query_pixel_hashes"][index],
            }
        )
    query = {
        "schema_version": 1,
        "protocol_id": finalizer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": finalizer.QUERY_RECEIPT_ID,
        "status": finalizer.QUERY_RECEIPT_STATUS,
        "checks_passed": True,
        "failed_attempt_receipt": failed_identity,
        "input_identities": query_inputs,
        "reconstruction_contract": {
            key: True
            for key in (
                "jpeg_reencoding_bitwise_equal_to_fragment",
                "reencoded_query_jpegs_match_fragment",
                "stored_paired_query_jpegs_equal",
                "raw_query_hashes_unique",
                "raw_query_prior_overlap_zero",
                "builder_snapshot_loaded_by_explicit_path",
                "physics_snapshot_loaded_by_explicit_path",
                "all_inputs_reverified_unchanged_after_replay",
            )
        },
        "failed_attempt_content": {
            "split": "train",
            "row_count": 16,
            "episode_count": 4,
            "pair_count": 2,
            "source_episodes": _source_entry(failed_episodes),
            "prior_content_exclusions": {
                field: _content_entry(failed_content[field], field)
                for field in finalizer.CONTENT_FIELDS
            },
            "pairs": query_pairs,
        },
        "prior_overlap": {
            **{
                field: {"count": 0, "values": []}
                for field in ("source_episode", *finalizer.CONTENT_FIELDS)
            },
            "passed": True,
        },
        "public_test": _closed_public(),
        "rgb_probe": {"opened": False, "run": False, "scored": False},
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
    }
    query["reconstruction_contract"].update(
        {
            "dataset_manifest_opened": False,
            "lance_written": False,
            "replayed_mode": "cannot_hold_only",
            "query_model_step_idx": 2,
            "jpeg_quality": 95,
        }
    )
    _write_json(paths["query"], query)
    query_identity = _identity(paths["query"])
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_QUERY_RECONSTRUCTION_SHA256",
        query_identity["sha256"],
    )

    recovery_inputs = {
        "old_final_prior_receipt": old_identity,
        "failed_formal_attempt_receipt": failed_identity,
        "query_reconstruction_receipt": query_identity,
        "source_h5": source,
    }
    prereg = {
        "schema_version": 1,
        "scientific_protocol_id": finalizer.SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": finalizer.RECOVERY_AUTHORIZATION_ID,
        "status": finalizer.PREREG_STATUS,
        "recovery_inputs": recovery_inputs,
        "public_test": _closed_public(),
        "reference_model_training_or_scoring_authorized": False,
    }
    # JSON is a valid YAML subset and keeps the fixture deterministic.
    _write_json(paths["prereg"], prereg)
    prereg_identity = _identity(paths["prereg"])
    freeze = {
        "schema_version": 1,
        "protocol_id": finalizer.SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": finalizer.RECOVERY_AUTHORIZATION_ID,
        "status": finalizer.FREEZE_STATUS,
        "checks_passed": True,
        "preregistration": prereg_identity,
        "authorization_inputs": recovery_inputs,
        "public_test": _closed_public(),
        "reference_model_training_or_scoring_authorized": False,
    }
    _write_json(paths["freeze"], freeze)
    return {
        "paths": paths,
        "old": old,
        "failed": failed,
        "query": query,
        "prereg": prereg,
        "freeze": freeze,
        "source": source,
        "old_content": old_content,
        "failed_content": failed_content,
        "old_episodes": old_episodes,
        "failed_episodes": failed_episodes,
    }


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    paths = fixture["paths"]
    return finalizer.finalize(
        old_final_prior=paths["old"],
        failed_attempt_receipt=paths["failed"],
        query_reconstruction_receipt=paths["query"],
        prereg_path=paths["prereg"],
        freeze_receipt_path=paths["freeze"],
        output=paths["output"],
    )


def _refresh_hash(
    monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Any], name: str
) -> None:
    constant = {
        "old": "EXPECTED_OLD_PRIOR_SHA256",
        "failed": "EXPECTED_FAILED_ATTEMPT_SHA256",
        "query": "EXPECTED_QUERY_RECONSTRUCTION_SHA256",
    }[name]
    monkeypatch.setattr(
        finalizer,
        constant,
        hashlib.sha256(fixture["paths"][name].read_bytes()).hexdigest(),
    )


def test_digest_namespaces_are_stable() -> None:
    episodes = [2, 5, 9]
    assert finalizer.excluded_source_episodes_sha256(episodes) == hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0"
        + b"".join(value.to_bytes(8, "little", signed=True) for value in episodes)
    ).hexdigest()
    values = sorted(["01" * 32, "ab" * 32])
    assert finalizer.canonical_content_digest(
        values, field_name="action_profile_ids"
    ) == hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        b"action_profile_ids\0"
        + b"".join(bytes.fromhex(value) for value in values)
    ).hexdigest()


def test_success_unions_all_five_complete_sets_and_preserves_old_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _run(fixture)
    assert receipt["protocol_id"] == finalizer.SCIENTIFIC_PROTOCOL_ID
    assert receipt["scientific_protocol_id"] == finalizer.SCIENTIFIC_PROTOCOL_ID
    assert receipt["recovery_authorization_id"] == finalizer.RECOVERY_AUTHORIZATION_ID
    assert receipt["status"] == finalizer.FREEZE_STATUS
    assert receipt["coverage"] == {
        **{field: True for field in finalizer.OLD_COVERAGE_FIELDS},
        "v4_failed_formal_attempts": True,
    }
    assert receipt["input_artifacts"][:1] == fixture["old"]["input_artifacts"]
    assert [value["role"] for value in receipt["input_artifacts"][-2:]] == [
        "v4_failed_formal_attempts",
        "v4_failed_formal_attempts",
    ]
    assert receipt["excluded_source_episodes"] == [0, 1, 10, 11]
    assert receipt["excluded_source_episode_count"] == 4
    assert receipt["excluded_source_episodes_sha256"] == (
        finalizer.excluded_source_episodes_sha256([0, 1, 10, 11])
    )
    for field in finalizer.CONTENT_FIELDS:
        expected = sorted(
            fixture["old_content"][field] + fixture["failed_content"][field]
        )
        entry = receipt["prior_content_exclusions"][field]
        assert entry["values"] == expected
        assert entry["count"] == 4
        assert entry["sha256"] == finalizer.canonical_content_digest(
            expected, field_name=field
        )
    assert receipt["failed_attempt_old_prior_overlap"]["passed"] is True
    assert receipt["public_test"]["read"] is False
    assert receipt["rgb_probe"]["run"] is False
    assert receipt["reference_model_training_or_scoring"] is False
    assert receipt["reference_model_optimizer_steps"] == 0
    assert json.loads(fixture["paths"]["output"].read_text()) == receipt


@pytest.mark.parametrize(
    "field",
    (
        "source_episodes",
        "action_profile_ids",
        "scene_template_content_hashes",
        "pair_content_hashes",
        "query_pixel_hashes",
    ),
)
def test_any_old_prior_overlap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    query = json.loads(fixture["paths"]["query"].read_text())
    failed = json.loads(fixture["paths"]["failed"].read_text())
    if field == "source_episodes":
        values = [0, 11]
        query["failed_attempt_content"]["source_episodes"] = _source_entry(values)
        failed["failed_attempt_content"]["source_episodes"] = _source_entry(values)
        query["failed_attempt_content"]["pairs"][0]["source_episode"] = 0
        failed["failed_attempt_content"]["pairs"][0]["source_episode"] = 0
    else:
        old_value = fixture["old_content"][field][0]
        pair_key = {
            "action_profile_ids": "action_profile_id",
            "scene_template_content_hashes": "scene_template_content_hash",
            "pair_content_hashes": "pair_content_hash",
        }.get(field)
        retained_value = (
            failed["failed_attempt_content"]["pairs"][1][pair_key]
            if pair_key is not None
            else query["failed_attempt_content"]["pairs"][1][
                "raw_query_pixel_hash"
            ]
        )
        query["failed_attempt_content"]["prior_content_exclusions"][field] = (
            _content_entry([old_value, retained_value], field)
        )
        if field in finalizer.DIRECT_FAILED_CONTENT_FIELDS:
            failed["failed_attempt_content"]["prior_content_exclusions"][field] = (
                copy.deepcopy(
                    query["failed_attempt_content"]["prior_content_exclusions"][field]
                )
            )
            failed["failed_attempt_content"]["pairs"][0][pair_key] = old_value
            query["failed_attempt_content"]["pairs"][0][pair_key] = old_value
        else:
            query["failed_attempt_content"]["pairs"][0][
                "raw_query_pixel_hash"
            ] = old_value
    # Re-establish the query -> failed receipt identity chain after any A edit.
    if field in ("source_episodes", *finalizer.DIRECT_FAILED_CONTENT_FIELDS):
        _write_json(fixture["paths"]["failed"], failed)
        _refresh_hash(monkeypatch, fixture, "failed")
        failed_identity = _identity(fixture["paths"]["failed"])
        query["failed_attempt_receipt"] = failed_identity
        query["input_identities"]["failed_attempt_receipt"] = failed_identity
    _write_json(fixture["paths"]["query"], query)
    _refresh_hash(monkeypatch, fixture, "query")
    # Authorization inputs must bind the deliberately mutated evidence.
    failed_identity = _identity(fixture["paths"]["failed"])
    query_identity = _identity(fixture["paths"]["query"])
    prereg = json.loads(fixture["paths"]["prereg"].read_text())
    prereg["recovery_inputs"]["failed_formal_attempt_receipt"] = failed_identity
    prereg["recovery_inputs"]["query_reconstruction_receipt"] = query_identity
    _write_json(fixture["paths"]["prereg"], prereg)
    freeze = json.loads(fixture["paths"]["freeze"].read_text())
    freeze["authorization_inputs"]["failed_formal_attempt_receipt"] = failed_identity
    freeze["authorization_inputs"]["query_reconstruction_receipt"] = query_identity
    freeze["preregistration"] = _identity(fixture["paths"]["prereg"])
    _write_json(fixture["paths"]["freeze"], freeze)
    with pytest.raises(RuntimeError, match="overlap"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_query_must_bind_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    query = json.loads(fixture["paths"]["query"].read_text())
    query["failed_attempt_receipt"]["sha256"] = "00" * 32
    _write_json(fixture["paths"]["query"], query)
    _refresh_hash(monkeypatch, fixture, "query")
    with pytest.raises(RuntimeError, match="failed attempt.*mismatch"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    (
        ("old", ("public_test", "read", True), "Public"),
        ("failed", ("scope", "reference_model_training_or_scoring", True), "model"),
        ("query", ("public_test", "opened", True), "Public"),
        ("query", (None, "reference_model_optimizer_steps", 1), "model"),
        ("prereg", ("public_test", "hashed", True), "Public"),
        ("freeze", (None, "reference_model_training_or_scoring_authorized", True), "model"),
    ),
)
def test_public_or_model_evidence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mutation: tuple[str | None, str, Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["paths"][name]
    value = json.loads(path.read_text())
    parent, key, replacement = mutation
    target = value if parent is None else value[parent]
    target[key] = replacement
    _write_json(path, value)
    if name in ("old", "failed", "query"):
        _refresh_hash(monkeypatch, fixture, name)
    with pytest.raises(RuntimeError, match=message):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_authorization_source_binding_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    prereg = json.loads(fixture["paths"]["prereg"].read_text())
    prereg["recovery_inputs"]["source_h5"]["sha256"] = "00" * 32
    _write_json(fixture["paths"]["prereg"], prereg)
    with pytest.raises(RuntimeError, match="source H5.*mismatch"):
        _run(fixture)


def test_strict_final_count_gate_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finalizer, "EXPECTED_FINAL_SOURCE_COUNT", 5)
    with pytest.raises(RuntimeError, match="final source count"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_output_is_x_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["paths"]["output"].write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(fixture)
    assert fixture["paths"]["output"].read_text() == "owned\n"


def test_postflight_input_mutation_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = finalizer._verify_unchanged
    called = False

    def mutate_then_verify(path: Path, raw: bytes, *, label: str) -> None:
        nonlocal called
        if not called:
            called = True
            fixture["paths"]["freeze"].write_text("{}\n", encoding="utf-8")
        original(path, raw, label=label)

    monkeypatch.setattr(finalizer, "_verify_unchanged", mutate_then_verify)
    with pytest.raises(RuntimeError, match="mutated"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_cli_requires_all_explicit_inputs() -> None:
    with pytest.raises(SystemExit):
        finalizer.parse_args([])
    args = finalizer.parse_args(
        [
            "--old-final-prior",
            "old.json",
            "--failed-attempt-receipt",
            "failed.json",
            "--query-reconstruction-receipt",
            "query.json",
            "--prereg",
            "v4r1.yaml",
            "--freeze-receipt",
            "freeze.json",
            "--output",
            "final.json",
        ]
    )
    assert args.query_reconstruction_receipt == Path("query.json")


@pytest.mark.parametrize(
    "value", ("public/receipt.json", "x/validation.lance/receipt.json")
)
def test_public_shaped_paths_are_rejected(value: str) -> None:
    with pytest.raises(RuntimeError, match="Public"):
        finalizer._reject_public_path(value, label="test")


def test_script_has_no_lance_probe_model_or_public_execution_surface() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "lance" not in imported
    cli_flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert all(
        token not in flag
        for flag in cli_flags
        for token in ("public", "probe", "train", "score", "lance", "source-h5")
    )
