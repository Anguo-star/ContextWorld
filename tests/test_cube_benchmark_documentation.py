from __future__ import annotations

import hashlib
from pathlib import Path
import re

import yaml

from contextworld.benchmarks.suite_data import (
    DEFAULT_SUITE_V2_RELEASE_CONFIG,
    load_icl_suite_release,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"
SUITE_V1_CONFIG = ROOT / "configs/benchmark/contextworld_icl_suite_v1.yaml"
SUITE_V2_CONFIG = (
    DEFAULT_SUITE_V2_RELEASE_CONFIG
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cube_release_uses_the_five_part_benchmark_template() -> None:
    document = PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    heading = "### 6.9 Cube 夹爪携带规则"
    start = document.index(heading)
    end = document.index("\n## 7. 接入新的 latent 世界模型", start)
    section = document[start:end]

    assert re.findall(r"^#### (.+)$", section, flags=re.MULTILINE) == [
        "任务目标",
        "数据构成",
        "评测方法",
        "基线表现",
        "适用范围",
    ]
    assert "authorized_not_generated_not_opened_not_read_not_scored" not in document
    assert "首个 Public v1 命名空间因发布元数据封装失败而永久封存" in document
    assert "旧 Public v1 的失败不是模型或科学数据门失败" in section
    assert "成员资格以 canonical" in section
    assert "不是自包含分发包" in section
    assert "77.73%、79.10% 和 78.52%" in section
    assert "186、183、185/300" in section


def test_cube_release_adds_suite_v2_without_rewriting_suite_v1() -> None:
    suite_v1 = yaml.safe_load(SUITE_V1_CONFIG.read_text(encoding="utf-8"))
    suite_v2 = load_icl_suite_release(SUITE_V2_CONFIG)

    assert len(suite_v1["components"]) == 8
    assert suite_v1["public_results"]["formal_reference_rows"] == 10
    assert all(
        "cube" not in component.lower() for component in suite_v1["components"]
    )
    assert len(suite_v2["components"]) == 9
    assert suite_v2["public_results"]["formal_reference_rows"] == 11
    assert len(suite_v2["public_results"]["components_with_formal_results"]) == 7
    assert suite_v2["components"]["cube_gripper_carry"][
        "reference_result_status"
    ] == "passed_public_test_3_of_3"
    assert (
        suite_v2["repository"]["public_document"]["sha256"]
        == _sha256(PUBLIC_DOCUMENT)
    )
    assert "contextworld-cube-gripper-carry" in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_cube_release_has_navigation_and_preserves_frozen_handoff() -> None:
    navigation = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    protocols = (ROOT / "docs/protocols/README.md").read_text(
        encoding="utf-8"
    )
    handoff = (
        ROOT
        / "docs/protocols/"
        "Cube_Gripper_Carry_History3_v4r1_Pre_Public_Handoff.md"
    ).read_text(encoding="utf-8")
    failure = (
        ROOT
        / "docs/protocols/"
        "Cube_Gripper_Carry_History3_v4r1_Public_v1_Generation_Failure.md"
    ).read_text(encoding="utf-8")

    assert "6.9 Cube 夹爪携带规则" in navigation
    assert "Public v1 失败与恢复边界" in protocols
    assert "reference_development_decision_v3.json" in handoff
    assert "original_task_retention_decision_v2.json" in handoff
    assert (
        "797e5a9722435257fae55e1f9d97424cc77d2d3779576833322b84160375954f"
        in handoff
    )
    assert (
        "12dbe11eb4cf025359987962dfd869e73e0deb0ecb0eca007fad727889a07ef0"
        in handoff
    )
    assert "generation_failed_before_publication_not_model_read_not_scored" in failure
    assert "a8c985f2f13fff93a0ac3629ffb5feee19803848ec15b6b2ac128ca7fb0e1965" in failure
    assert "fc5e6e21b43af548102c105ec21e75bdd7542808f3ede818d65c683063907fcc" in failure
