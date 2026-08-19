from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from contextworld.evaluation.icl_catalog import (
    validate_context_query_catalog,
)
from contextworld.evaluation.speed_cube import build_speed_cube_catalog
from contextworld.synthesis.stablewm import load_stable_worldmodel


def test_speed_cube_has_exact_static_query_and_replay(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pinned = Path("/tmp/stable-worldmodel-5864")
    configured = str(pinned if pinned.is_dir() else root.parent / "stable-worldmodel")
    for name in tuple(sys.modules):
        if name == "stable_worldmodel" or name.startswith("stable_worldmodel."):
            del sys.modules[name]
    load_stable_worldmodel(
        root,
        configured,
        "5864b74980f6ed328fd0045e777b3865962eff43",
    )
    catalog_path = tmp_path / "cube.json"
    catalog = build_speed_cube_catalog(
        repo_root=tmp_path,
        output_catalog=catalog_path,
        payload_root=tmp_path / "payloads",
        split="validation",
        distances=[72],
        variants_per_distance=1,
        geometry_seed=2026072101,
        catalog_seed=2026072102,
        stable_worldmodel_commit="test",
        speeds=(3.4, 4.8, 6.9),
        track_name="unseen_interpolation",
    )

    assert catalog["summary"]["bundles"] == 3
    assert catalog["summary"]["matrix_cells"] == 9
    assert catalog["summary"]["static_query_groups"] == 1
    assert catalog["summary"]["static_query_pixel_audit_passed"]

    bundles = catalog["bundles"]
    assert len({bundle["static_query_id"] for bundle in bundles}) == 1
    assert len({bundle["query_pixels_sha256"] for bundle in bundles}) == 1
    assert {
        bundle["same_speed_condition"] for bundle in bundles
    } == {"history_low", "history_mid", "history_high"}
    for bundle in bundles:
        assert list(bundle["conditions"]) == [
            "history_low",
            "history_mid",
            "history_high",
        ]
        payload = np.load(tmp_path / bundle["payload"])
        assert payload["query_state"].shape == (2,)
        assert payload["candidate_states"].shape == (3, 2)

    validation = validate_context_query_catalog(
        catalog_path,
        repo_root=tmp_path,
        replay_simulator=True,
        family="speed",
    )
    assert validation["passed"], json.dumps(
        validation["failures"], indent=2
    )
    assert validation["context_rollouts_checked"] == 18
