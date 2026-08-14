from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import scripts.reconstruct_cube_grasp_rule_h3_v4_failed_attempt_query_hashes as recovery


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _pair(index: int) -> recovery.FragmentPair:
    jpeg = f"jpeg-{index}".encode("ascii")
    scene = _digest(f"scene-{index}")
    profile = _digest(f"profile-{index}")
    pair_hash = hashlib.sha256(
        bytes.fromhex(scene) + bytes.fromhex(profile)
    ).hexdigest()
    return recovery.FragmentPair(
        pair_id=f"cube-carry-v4-train-{index:06d}",
        split="train",
        catalog_index=1_000_000 + index,
        source_row=100 + index,
        source_episode=10 + index,
        source_step=20 + index,
        action_anchor_id=("endpoint4", "plateau")[index % 2],
        action_profile_id=profile,
        scene_template_content_hash=scene,
        pair_content_hash=pair_hash,
        query_jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),
        query_jpeg=jpeg,
    )


def _replay(pair: recovery.FragmentPair) -> recovery.ReplayResult:
    return recovery.ReplayResult(
        pair_id=pair.pair_id,
        raw_query_pixel_hash=_digest(f"raw-{pair.pair_id}"),
        query_jpeg_sha256=pair.query_jpeg_sha256,
    )


def _set_entry(values: list[str], field_name: str) -> dict[str, object]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": recovery.canonical_content_digest(
            values, field_name=field_name
        ),
    }


def _failed_receipt(pairs: list[recovery.FragmentPair]) -> dict[str, object]:
    sets = recovery._fragment_sets(pairs)
    source = sets["source_episodes"]
    jpeg_values = sets["query_jpeg_sha256"]
    return {
        "schema_version": 1,
        "protocol_id": recovery.PROTOCOL,
        "receipt_id": recovery.FAILED_RECEIPT_ID,
        "status": recovery.FAILED_STATUS,
        "checks_passed": True,
        "build_passed": False,
        "formal_build_attempt_consumed": True,
        "retry_authorized_under_original_preregistration": False,
        "scope": {
            "public_test": {
                "access_status": "closed_not_read_not_scored",
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
        },
        "failed_attempt_content": {
            "split": "train",
            "row_count": 8 * len(pairs),
            "episode_count": 2 * len(pairs),
            "pair_count": len(pairs),
            "source_episodes": {
                "values": source,
                "count": len(source),
                "sha256": recovery.excluded_source_episodes_sha256(source),
            },
            "prior_content_exclusions": {
                field: _set_entry(sets[field], field)
                for field in recovery.FRAGMENT_CONTENT_FIELDS
            },
            "query_pixel_hash_status": (
                "pending_deterministic_raw_reconstruction_not_present_in_fragment"
            ),
            "query_jpeg_sha256": {
                "values": jpeg_values,
                "count": len(jpeg_values),
                "sha256": recovery.forensic_query_jpeg_digest(jpeg_values),
                "digest_namespace": (
                    recovery.FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE
                ),
                "role": "forensic_binding_only_not_raw_query_pixel_hash",
            },
            "pairs": [
                {
                    key: value
                    for key, value in pair.public_record().items()
                    if key != "split"
                }
                for pair in pairs
            ],
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
    }


def _fragment_table(pair_count: int = 2) -> pa.Table:
    rows: list[dict[str, object]] = []
    for index in range(pair_count):
        jpeg = f"jpeg-{index}".encode("ascii")
        blocks = np.zeros((4, 5, 5), dtype=np.float32)
        blocks[0, :, 2] = np.asarray([0.25, 0.25, -0.25, -0.25, 0])
        blocks[1, :, 2] = -blocks[0, :, 2]
        blocks[2, :, 2] = blocks[0, :, 2]
        blocks[:3, :, 4] = np.float32(0.4 + 0.1 * index)
        profile = hashlib.sha256(
            np.ascontiguousarray(blocks).tobytes()
        ).hexdigest()
        scene = _digest(f"fragment-scene-{index}")
        pair_hash = hashlib.sha256(
            bytes.fromhex(scene) + bytes.fromhex(profile)
        ).hexdigest()
        for mode in recovery.EXPECTED_MODES:
            for step in recovery.EXPECTED_MODEL_STEPS:
                rows.append(
                    {
                        "model_step_idx": step,
                        "pixels": jpeg if step == recovery.QUERY_MODEL_STEP else b"x",
                        "action_block": blocks[step].reshape(-1).tolist(),
                        "pair_id": f"cube-carry-v4-train-{index:06d}",
                        "hidden_mode": mode,
                        "split": "train",
                        "catalog_index": 1_000_000 + index,
                        "source_row": 100 + index,
                        "source_episode": 10 + index,
                        "source_step": 20 + index,
                        "action_anchor_id": ("endpoint4", "plateau")[index],
                        "action_profile_id": profile,
                        "scene_template_content_hash": scene,
                        "pair_content_hash": pair_hash,
                    }
                )
    schema = pa.schema(
        [
            pa.field("model_step_idx", pa.int32()),
            pa.field("pixels", pa.binary()),
            pa.field("action_block", pa.list_(pa.float32(), 25)),
            pa.field("pair_id", pa.string()),
            pa.field("hidden_mode", pa.string()),
            pa.field("split", pa.string()),
            pa.field("catalog_index", pa.int32()),
            pa.field("source_row", pa.int64()),
            pa.field("source_episode", pa.int32()),
            pa.field("source_step", pa.int32()),
            pa.field("action_anchor_id", pa.string()),
            pa.field("action_profile_id", pa.string()),
            pa.field("scene_template_content_hash", pa.string()),
            pa.field("pair_content_hash", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def test_small_fragment_and_failed_receipt_schema_round_trip() -> None:
    pairs, row_count = recovery.extract_fragment_pairs(_fragment_table())
    assert row_count == 16
    assert len(pairs) == 2
    recovery.validate_failed_attempt_content(
        _failed_receipt(pairs), pairs=pairs, row_count=row_count
    )


def test_reencoded_jpeg_mismatch_is_rejected() -> None:
    pair = _pair(0)

    def mismatch(value: recovery.FragmentPair) -> recovery.ReplayResult:
        result = _replay(value)
        return recovery.ReplayResult(
            pair_id=result.pair_id,
            raw_query_pixel_hash=result.raw_query_pixel_hash,
            query_jpeg_sha256=_digest("wrong-jpeg"),
        )

    with pytest.raises(RuntimeError, match="JPEG identity mismatch"):
        recovery.reconstruct_query_hashes(
            [pair], prior_query_hashes=set(), replay=mismatch
        )


def test_raw_query_collision_is_rejected() -> None:
    pairs = [_pair(0), _pair(1)]
    collision = _digest("same-raw-frame")

    def replay(value: recovery.FragmentPair) -> recovery.ReplayResult:
        return recovery.ReplayResult(
            pair_id=value.pair_id,
            raw_query_pixel_hash=collision,
            query_jpeg_sha256=value.query_jpeg_sha256,
        )

    with pytest.raises(RuntimeError, match="collision/duplicate"):
        recovery.reconstruct_query_hashes(
            pairs, prior_query_hashes=set(), replay=replay
        )


def test_prior_raw_query_overlap_is_rejected() -> None:
    pair = _pair(0)
    raw = _replay(pair).raw_query_pixel_hash
    with pytest.raises(RuntimeError, match="overlap prior exclusions"):
        recovery.reconstruct_query_hashes(
            [pair], prior_query_hashes={raw}, replay=_replay
        )


def test_bound_input_mutation_is_rejected(tmp_path: Path) -> None:
    value = tmp_path / "snapshot.py"
    value.write_bytes(b"frozen\n")
    declared = recovery._identity(value)
    value.write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        recovery._verify_file_identity(value, declared, label="snapshot")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        recovery.write_receipt_exclusive(output, {"checks_passed": True})
    assert output.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    "value",
    (
        "artifacts/public/receipt.json",
        "artifacts/validation.lance/data/file.lance",
    ),
)
def test_public_shaped_paths_are_rejected(value: str) -> None:
    with pytest.raises(RuntimeError, match="Public-Test"):
        recovery.reject_public_path(value, label="test")


def test_tool_has_no_lance_write_or_public_model_cli() -> None:
    source_path = Path(recovery.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write_dataset" not in called_attributes
    args = recovery.parse_args(
        [
            "--failed-attempt-receipt",
            "failed.json",
            "--fragment",
            "fragment.lance",
            "--builder-snapshot",
            "builder.py",
            "--physics-snapshot",
            "physics.py",
            "--source-h5",
            "source.h5",
            "--prereg",
            "prereg.yaml",
            "--freeze-receipt",
            "freeze.json",
            "--prior-exclusion-receipt",
            "prior.json",
            "--request-json",
            "request.json",
            "--output",
            "out.json",
        ]
    )
    assert args.workers == 16
    assert args.jpeg_quality == 95
    # argparse Namespace exposes values, so inspect parser behavior through
    # the accepted option names in source without invoking any work path.
    source = source_path.read_text(encoding="utf-8")
    for forbidden in ("--public", "--probe", "--train", "--score"):
        assert f'parser.add_argument("{forbidden}"' not in source
