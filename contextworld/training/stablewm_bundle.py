"""Runtime StableWM view over the public ``ContextWorld-v1`` bundle.

The public bundle is a release artifact, not a single StableWM table.  This
module exposes it through StableWM's documented format registry without
rewriting or duplicating image payloads.  A compact ``contextworld://`` URI
binds the bundle manifest, component, split, payload and optional
original/synthetic mixture.  The reader then composes the registered Lance
members lazily inside each DataLoader worker.

Only the model-visible ``pixels`` and ``action`` columns are returned.  This
is intentional: component metadata remains in the release tables for audit,
but is not a privileged model input.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, reduce
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .groups import (
    LogicalGroupDataset,
    RelationBatchSampler,
    ScenarioBalancedDataset,
)


URI_PREFIX = "contextworld://v1/"
FORMAT_NAME = "contextworld_bundle"
MODEL_COLUMNS = ("pixels", "action")
_DELAY_PATTERN = re.compile(r"(?:^|[-_])d(?P<delay>\d+)(?:[-_])")
CONDITIONAL_JOINT_METHOD = "coja_v1"
CONDITIONAL_JOINT_COMPONENTS = frozenset(
    {"action_strength", "contact_friction", "portal_exit", "robot_arm_mass"}
)
CONDITIONAL_JOINT_GROUP_WIDTH = 2
CONDITIONAL_JOINT_GROUP_COLUMN = "conditional_joint_group"
PUBLIC_RELATION_COLUMN = "pair_id"
DEVELOPMENT_EVALUATION_SCHEMA_VERSION = "contextworld.development_evaluation.v1"
DEVELOPMENT_EVALUATION_STATUS = "public_development_only"
TWOROOM_NORMALIZER_PROTOCOL = "tworoom_original_train_s3072_unbiased_zscore_v1"
TWOROOM_NORMALIZER_SCOPE = "original_9000_train_episodes_only"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    return Path(str(path))


@lru_cache(maxsize=8)
def _bundle_contract(root_value: str) -> dict[str, Any]:
    root = _safe_absolute_path(root_value, label="ContextWorld bundle root")
    if not root.is_dir():
        raise ValueError(f"Missing ContextWorld bundle root: {root}")

    manifest = root / "manifest.jsonl"
    manifest_receipt = root / "manifest.sha256"
    registry = root / "task_registry.json"
    for path in (manifest, manifest_receipt, registry):
        # Hugging Face's local snapshot cache commonly represents files as
        # symlinks into its content-addressed blob store.  Their bytes remain
        # bound by the release hashes below, so require readability rather
        # than rejecting that standard cache layout.
        if not path.is_file():
            raise ValueError(f"Missing ContextWorld release file: {path}")

    receipt_fields = manifest_receipt.read_text(encoding="utf-8").strip().split()
    if len(receipt_fields) != 2 or receipt_fields[1] != "manifest.jsonl":
        raise ValueError(f"Malformed ContextWorld manifest receipt: {manifest_receipt}")
    observed_manifest_sha = _sha256_file(manifest)
    if observed_manifest_sha != receipt_fields[0]:
        raise ValueError(
            "ContextWorld manifest identity mismatch: "
            f"expected={receipt_fields[0]} observed={observed_manifest_sha}"
        )

    registry_record = None
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed manifest.jsonl record at line {line_number}"
                ) from exc
            if record.get("path") == "task_registry.json":
                registry_record = record
    if not isinstance(registry_record, dict):
        raise ValueError("manifest.jsonl does not bind task_registry.json")
    observed_registry_sha = _sha256_file(registry)
    if observed_registry_sha != registry_record.get("sha256"):
        raise ValueError(
            "ContextWorld task registry identity mismatch: "
            f"expected={registry_record.get('sha256')} "
            f"observed={observed_registry_sha}"
        )

    payload = json.loads(registry.read_text(encoding="utf-8"))
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"Invalid ContextWorld task registry: {registry}")
    return {
        "root": root,
        "manifest_sha256": observed_manifest_sha,
        "task_registry_sha256": observed_registry_sha,
        "registry": payload,
    }


def _urlsafe_json(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def _decode_urlsafe_json(value: str) -> dict[str, Any]:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed contextworld:// dataset identifier") from exc
    if not isinstance(payload, dict):
        raise ValueError("ContextWorld dataset identifier must encode a mapping")
    return payload


def _conditional_joint_contract(
    component_id: str, *, payload_id: str | None, method: str | None
) -> dict[str, Any] | None:
    if method is None:
        return None
    if method != CONDITIONAL_JOINT_METHOD:
        raise ValueError(
            f"Unsupported ContextWorld training method: {method!r}; "
            f"expected {CONDITIONAL_JOINT_METHOD!r}"
        )
    if component_id not in CONDITIONAL_JOINT_COMPONENTS:
        supported = ", ".join(sorted(CONDITIONAL_JOINT_COMPONENTS))
        raise ValueError(
            f"{CONDITIONAL_JOINT_METHOD} is registered only for components "
            f"with audited public pair identities ({supported}), not "
            f"{component_id!r}"
        )
    if payload_id != "data":
        raise ValueError(
            f"{component_id} conditional-joint training requires the "
            f"registered data payload, observed={payload_id!r}"
        )
    return {
        "method": method,
        "group_width": CONDITIONAL_JOINT_GROUP_WIDTH,
        "relation_kind": "public_pair_identity_v1",
    }


def build_contextworld_dataset_uri(
    bundle_root: str | Path,
    *,
    component: str,
    split: str = "training",
    payload_id: str | None = None,
    original_dataset: str | Path | None = None,
    original_weight: float = 0.0,
    synthetic_weight: float = 1.0,
    epoch_size: int | None = None,
    conditional_joint_method: str | None = None,
) -> str:
    """Return a Hydra-safe, immutable runtime dataset identifier."""

    root = _safe_absolute_path(bundle_root, label="ContextWorld bundle root")
    if split not in {"training", "development"}:
        raise ValueError(f"Unsupported ContextWorld split: {split!r}")
    if not component:
        raise ValueError("ContextWorld component must be non-empty")
    if (
        not math.isfinite(original_weight)
        or not math.isfinite(synthetic_weight)
        or original_weight < 0
        or synthetic_weight <= 0
    ):
        raise ValueError("Dataset mixture weights must be non-negative")
    if original_weight and original_dataset is None:
        raise ValueError("An original dataset is required when original_weight > 0")
    if original_dataset is not None:
        original = _safe_absolute_path(original_dataset, label="Original dataset")
    else:
        original = None
    if epoch_size is not None and epoch_size <= 0:
        raise ValueError("epoch_size must be positive")

    spec: dict[str, Any] = {
        "root": str(root),
        "component": component,
        "split": split,
        "payload_id": payload_id,
        "original_dataset": str(original) if original else None,
        "weights": {
            "original": float(original_weight),
            "synthetic": float(synthetic_weight),
        },
        "epoch_size": epoch_size,
        "conditional_joint_method": conditional_joint_method,
    }
    # Validate before a GPU job is rendered.  The URI contains no query
    # ``=`` characters, which keeps it safe as a Hydra override value.
    _describe_spec(spec)
    return URI_PREFIX + _urlsafe_json(spec)


def parse_contextworld_dataset_uri(uri: str) -> dict[str, Any]:
    if not str(uri).startswith(URI_PREFIX):
        raise ValueError(f"Not a ContextWorld runtime dataset URI: {uri!r}")
    return _decode_urlsafe_json(str(uri)[len(URI_PREFIX) :])


def _component(contract: Mapping[str, Any], component_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in contract["registry"]["components"]
        if value.get("component_id") == component_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate ContextWorld component: {component_id}")
    return matches[0]


def resolve_contextworld_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Validate a ``ContextWorld-v1`` root and return its release identity.

    This is deliberately a read-only release resolver.  It validates the
    manifest receipt and the manifest-bound task registry, then exposes the
    two hashes that must be recorded with a Development evaluation receipt.
    It does not inspect, infer, or fall back to any private artifact tree.
    """

    contract = _bundle_contract(str(bundle_root))
    registry = contract["registry"]
    components = registry.get("components")
    assert isinstance(components, list)  # already checked by _bundle_contract
    return {
        "schema_version": "contextworld.bundle-resolution.v1",
        "bundle_root": str(contract["root"]),
        "manifest_sha256": contract["manifest_sha256"],
        "task_registry_sha256": contract["task_registry_sha256"],
        "release_status": registry.get("release_status"),
        "public_test_policy": registry.get("public_test", {}).get("policy"),
        "component_ids": tuple(
            str(component.get("component_id", "")) for component in components
        ),
    }


def _development_evaluation(component: Mapping[str, Any]) -> dict[str, Any]:
    """Read the public Development-only contract from one registry component."""

    value = component.get("development_evaluation")
    component_id = component.get("component_id")
    if not isinstance(value, dict):
        raise ValueError(
            f"ContextWorld component {component_id!r} has no Development "
            "evaluation contract; refresh the ContextWorld-v1 metadata first"
        )
    if value.get("schema_version") != DEVELOPMENT_EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Development evaluation contract for {component_id!r}"
        )
    if value.get("status") != DEVELOPMENT_EVALUATION_STATUS:
        raise ValueError(
            f"Development evaluation is not publicly runnable for {component_id!r}"
        )
    if value.get("split") != "development":
        raise ValueError(
            f"Development evaluation must select split='development' for "
            f"{component_id!r}"
        )
    if not isinstance(value.get("payload_id"), str) or not value["payload_id"]:
        raise ValueError(
            f"Development evaluation has no payload_id for {component_id!r}"
        )
    if not isinstance(value.get("payload"), dict):
        raise ValueError(
            f"Development evaluation has no payload binding for {component_id!r}"
        )
    return value


def _assert_development_payload_binding(
    contract: Mapping[str, Any],
    component: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    payload: Mapping[str, Any],
    members: Sequence[Path],
) -> None:
    """Ensure registry metadata and manifest describe the same member set."""

    binding = evaluation["payload"]
    for field in ("public_path", "payload_kind", "lance_table_count"):
        if binding.get(field) != payload.get(field):
            raise ValueError(
                "Development payload binding disagrees with task payload for "
                f"{component.get('component_id')!r}: {field}"
            )
    expected_members = payload.get("members")
    if binding.get("members") != expected_members:
        raise ValueError(
            "Development payload member binding disagrees with task payload for "
            f"{component.get('component_id')!r}"
        )

    # ``manifest.jsonl`` binds every distributed payload file.  Checking its
    # release metadata is cheap and makes sure a registry entry cannot name a
    # directory outside its declared Development payload.  Rehashing every
    # image on each evaluator startup would turn this resolver into a large
    # I/O job, so content hashes remain available through the signed manifest.
    root = Path(contract["root"])
    member_prefixes = [
        path.relative_to(root).as_posix().rstrip("/") + "/" for path in members
    ]
    covered = [False] * len(member_prefixes)
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed manifest.jsonl record at line {line_number}"
                ) from exc
            if (
                row.get("role") != "dataset_payload"
                or row.get("component") != component.get("component_id")
                or row.get("split") != "development"
            ):
                continue
            path = row.get("path")
            if not isinstance(path, str):
                continue
            for index, prefix in enumerate(member_prefixes):
                if path.startswith(prefix):
                    covered[index] = True
    if not all(covered):
        missing = [
            str(member)
            for member, present in zip(members, covered)
            if not present
        ]
        raise ValueError(
            "Development payload member is not bound by manifest.jsonl: "
            + ", ".join(missing)
        )


def _resolve_development_normalizer(
    contract: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> Path | None:
    """Validate the optional portable normalizer bound by the manifest."""

    value = evaluation.get("normalizer_path")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Development normalizer_path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe Development normalizer path: {value}")
    root = Path(contract["root"])
    path = root / relative
    # Metadata files can be represented as content-addressed snapshot links by
    # standard HF caches.  Their bytes are still checked against the manifest.
    if not path.is_file():
        raise ValueError(f"Missing Development normalizer: {path}")

    manifest_row = None
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed manifest.jsonl record at line {line_number}"
                ) from exc
            if row.get("path") == relative.as_posix():
                manifest_row = row
                break
    if not isinstance(manifest_row, dict) or manifest_row.get("role") != "release_metadata":
        raise ValueError(
            f"Development normalizer is not bound as release metadata: {relative}"
        )
    observed_sha = _sha256_file(path)
    if observed_sha != manifest_row.get("sha256"):
        raise ValueError(
            "Development normalizer hash mismatch: "
            f"expected={manifest_row.get('sha256')} observed={observed_sha}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed Development normalizer: {path}") from exc
    if payload.get("protocol") != TWOROOM_NORMALIZER_PROTOCOL:
        raise ValueError(f"Unsupported Development normalizer protocol: {path}")
    if payload.get("statistics_scope") != TWOROOM_NORMALIZER_SCOPE:
        raise ValueError(f"Invalid Development normalizer scope: {path}")
    columns = payload.get("columns")
    if not isinstance(columns, dict):
        raise ValueError(f"Development normalizer has no columns mapping: {path}")
    for column in ("action", "proprio"):
        values = columns.get(column)
        if not isinstance(values, dict):
            raise ValueError(f"Development normalizer has no {column} statistics")
        mean = values.get("mean")
        std = values.get("std_unbiased")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or len(mean) != len(std)
            or not mean
            or any(
                not isinstance(item, (int, float)) or not math.isfinite(float(item))
                for item in mean
            )
            or any(
                not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) <= 0.0
                for item in std
            )
        ):
            raise ValueError(
                f"Invalid Development normalizer statistics for {column}: {path}"
            )
    action_normalization = evaluation.get("action_normalization")
    if not isinstance(action_normalization, dict):
        raise ValueError("Development evaluation has no inline action normalization")
    if (
        action_normalization.get("mean") != columns["action"]["mean"]
        or action_normalization.get("std") != columns["action"]["std_unbiased"]
        or action_normalization.get("std_estimator") != "unbiased"
    ):
        raise ValueError(
            "Development normalizer and inline action normalization disagree"
        )
    return path


def resolve_contextworld_component(
    bundle_root: str | Path, *, component: str
) -> dict[str, Any]:
    """Resolve one public component while retaining the bundle identity.

    This low-level component lookup also works for a pre-migration clean
    bundle.  Call :func:`resolve_contextworld_development_payload` when a
    runnable Development evaluator is required; that path insists on the new
    Development contract and its bound payload.
    """

    contract = _bundle_contract(str(bundle_root))
    resolved = _component(contract, component)
    identity = resolve_contextworld_bundle(bundle_root)
    return {
        **identity,
        "component_id": resolved["component_id"],
        "dataset_id": resolved["dataset_id"],
        "environment": resolved["environment"],
        "history_length": int(resolved["history_length"]),
        "action_dimension": int(resolved["action_dimension"]),
        "frameskip": int(resolved["frameskip"]),
        "development_evaluation": copy.deepcopy(
            resolved.get("development_evaluation")
        ),
    }


def resolve_contextworld_development_payload(
    bundle_root: str | Path, *, component: str
) -> dict[str, Any]:
    """Resolve manifest-bound Development Lance members for one component.

    Returned ``member_paths`` are absolute local paths; ``relative_members``
    and the manifest/registry hashes make the selection reproducible in a
    result receipt.  The function never resolves a private artifact root.
    """

    contract = _bundle_contract(str(bundle_root))
    resolved = _component(contract, component)
    evaluation = _development_evaluation(resolved)
    payload = _select_payload(
        resolved,
        split=str(evaluation["split"]),
        payload_id=str(evaluation["payload_id"]),
    )
    root = Path(contract["root"])
    members = _payload_members(root, payload)
    _assert_development_payload_binding(
        contract, resolved, evaluation, payload, members
    )
    normalizer = _resolve_development_normalizer(contract, evaluation)
    return {
        **resolve_contextworld_component(bundle_root, component=component),
        "payload_id": payload["payload_id"],
        "payload_kind": payload["payload_kind"],
        "relative_members": tuple(
            path.relative_to(root).as_posix() for path in members
        ),
        "member_paths": tuple(str(path) for path in members),
        "normalizer_path": str(normalizer) if normalizer is not None else None,
    }


def resolve_contextworld_development_payload_members(
    bundle_root: str | Path, *, component: str
) -> tuple[Path, ...]:
    """Return only the safe, manifest-bound Development Lance directories."""

    resolved = resolve_contextworld_development_payload(
        bundle_root, component=component
    )
    return tuple(Path(value) for value in resolved["member_paths"])


def _select_payload(
    component: Mapping[str, Any], *, split: str, payload_id: str | None
) -> dict[str, Any]:
    matches = [
        value
        for value in component.get("payloads", [])
        if value.get("split") == split
        and (payload_id is None or value.get("payload_id") == payload_id)
    ]
    if len(matches) != 1:
        choices = sorted(
            {
                str(value.get("payload_id"))
                for value in component.get("payloads", [])
                if value.get("split") == split
            }
        )
        raise ValueError(
            f"Component {component.get('component_id')!r} split {split!r} "
            f"needs one payload_id; available={choices}"
        )
    return matches[0]


def _payload_members(root: Path, payload: Mapping[str, Any]) -> list[Path]:
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        single = payload.get("single_dataset_path")
        members = [single] if isinstance(single, str) and single else []
    if not members:
        raise ValueError(f"Payload has no registered members: {payload.get('public_path')}")

    resolved: list[Path] = []
    for value in members:
        if not isinstance(value, str) or not value:
            raise ValueError("ContextWorld registry member must be a non-empty path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe ContextWorld registry member: {value}")
        path = root / relative
        if not path.is_dir() or path.is_symlink() or path.suffix != ".lance":
            raise ValueError(f"Missing or unsafe registered Lance member: {path}")
        resolved.append(path)
    if len(resolved) != int(payload.get("lance_table_count", len(resolved))):
        raise ValueError(
            "ContextWorld registry member count mismatch: "
            f"declared={payload.get('lance_table_count')} observed={len(resolved)}"
        )
    return resolved


def _describe_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = _safe_absolute_path(str(spec.get("root", "")), label="Bundle root")
    contract = _bundle_contract(str(root))
    component = _component(contract, str(spec.get("component", "")))
    split = str(spec.get("split", ""))
    payload = _select_payload(
        component,
        split=split,
        payload_id=(
            str(spec["payload_id"]) if spec.get("payload_id") is not None else None
        ),
    )
    members = _payload_members(root, payload)
    original_value = spec.get("original_dataset")
    if original_value is not None:
        original = _safe_absolute_path(str(original_value), label="Original dataset")
        # The original LeWM distribution intentionally exposes short, stable
        # symlink names (for example ``quentinll/tworoom.h5``).  They are a
        # normal part of the public data layout, so require a readable file
        # rather than rejecting the symlink spelling itself.
        if not original.is_file():
            raise ValueError(f"Missing original dataset: {original}")
    weights = spec.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("ContextWorld dataset identifier has no mixture weights")
    original_weight = float(weights.get("original", 0.0))
    synthetic_weight = float(weights.get("synthetic", 0.0))
    if (
        not math.isfinite(original_weight)
        or not math.isfinite(synthetic_weight)
        or original_weight < 0
        or synthetic_weight <= 0
    ):
        raise ValueError(f"Invalid ContextWorld mixture weights: {weights}")
    if bool(original_weight) != bool(original_value):
        raise ValueError("Original dataset and original mixture weight disagree")

    relative_members = [str(path.relative_to(root)) for path in members]
    member_list_sha256 = hashlib.sha256(
        ("\n".join(relative_members) + "\n").encode("utf-8")
    ).hexdigest()
    conditional_joint = _conditional_joint_contract(
        str(component["component_id"]),
        payload_id=payload.get("payload_id"),
        method=(
            str(spec["conditional_joint_method"])
            if spec.get("conditional_joint_method") is not None
            else None
        ),
    )
    return {
        "schema_version": "contextworld.stablewm-runtime-dataset.v1",
        "root": str(root),
        "manifest_sha256": contract["manifest_sha256"],
        "task_registry_sha256": contract["task_registry_sha256"],
        "component": component["component_id"],
        "dataset_id": component["dataset_id"],
        "environment": component["environment"],
        "history_length": int(component["history_length"]),
        "action_dimension": int(component["action_dimension"]),
        "frameskip": int(component["frameskip"]),
        "split": split,
        "payload_id": payload.get("payload_id"),
        "payload_kind": payload.get("payload_kind"),
        "sequence_schema": payload.get("stable_worldmodel_sequence_schema"),
        "adapter": payload.get("stable_worldmodel_adapter_required"),
        "member_count": len(members),
        # Keep checkpoint identities compact even for Speed's 512 tables.
        # The registry and manifest hashes bind the exact paths; this digest
        # makes the selected member list explicit without copying it into
        # every run marker.
        "member_list_sha256": member_list_sha256,
        "original_dataset": original_value,
        "weights": {
            "original": original_weight,
            "synthetic": synthetic_weight,
        },
        "epoch_size": spec.get("epoch_size"),
        "conditional_joint": conditional_joint,
    }


def describe_contextworld_dataset(uri: str) -> dict[str, Any]:
    """Return the cheap release/recipe identity recorded beside checkpoints."""

    return _describe_spec(parse_contextworld_dataset_uri(uri))


def _decode_images(values: Sequence[bytes]):
    import torch
    from PIL import Image

    frames = []
    for value in values:
        with Image.open(io.BytesIO(bytes(value))) as image:
            array = np.asarray(image.convert("RGB")).copy()
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames)


def _numeric(values: Sequence[Any]):
    import torch

    return torch.as_tensor(np.asarray(list(values), dtype=np.float32))


class _ProjectedLanceSequence:
    """Minimal numeric projection for legacy tables with string metadata."""

    def __init__(
        self,
        path: Path,
        *,
        num_steps: int,
        frameskip: int,
        keys_to_load: Sequence[str],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        import lance

        self.path = path
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.span = self.num_steps * self.frameskip
        self._keys = list(keys_to_load)
        self._transform = transform
        # Lance/Arrow dataset handles are native objects and must not be
        # inherited by Linux DataLoader workers through ``fork``.  Inspect the
        # member while constructing the lightweight Python index, then drop
        # the handle before the DataLoader is created.  Every worker opens its
        # own handle lazily on first use.
        dataset = lance.dataset(str(path))
        names = set(dataset.schema.names)
        required = {"episode_idx", "step_idx", *self._keys}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Legacy Lance projection {path} misses {missing}")
        self._action_dim = _fixed_list_size(dataset.schema, "action")
        self._offsets, self._lengths = _episode_structure(
            dataset, step_column="step_idx"
        )
        self._clip_counts = np.maximum(self._lengths - self.span + 1, 0)
        self._clip_cumulative = np.cumsum(self._clip_counts, dtype=np.int64)
        self._dataset = None
        self._dataset_pid = None

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    @property
    def episode_count(self) -> int:
        return len(self._offsets)

    def episode_clip_range(self, episode: int) -> tuple[int, int]:
        """Return ``(first_clip_index, clip_count)`` for one raw episode."""

        previous = 0 if episode == 0 else int(self._clip_cumulative[episode - 1])
        return previous, int(self._clip_counts[episode])

    def episode_relation_keys(self, column: str) -> list[str]:
        return _episode_relation_keys(
            self.path, self._offsets, self._lengths, column
        )

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value) -> None:
        self._transform = value

    def __len__(self) -> int:
        return int(self._clip_cumulative[-1]) if len(self._clip_cumulative) else 0

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self._clip_cumulative, index, side="right"))
        previous = 0 if episode == 0 else int(self._clip_cumulative[episode - 1])
        return episode, int(index - previous)

    def _open(self):
        pid = os.getpid()
        if self._dataset is None or self._dataset_pid != pid:
            import lance

            self._dataset = lance.dataset(str(self.path))
            self._dataset_pid = pid
        return self._dataset

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self._locate(index)
        offset = int(self._offsets[episode] + start)
        rows = list(range(offset, offset + self.span))
        table = self._open().take(rows, columns=self._keys)
        return self._sample_from_table(table)

    def _sample_from_table(self, table) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        for key in self._keys:
            values = table.column(key).to_pylist()
            if key == "pixels":
                sample[key] = _decode_images(values[:: self.frameskip])
            elif key == "action":
                sample[key] = _numeric(values)
            else:
                sample[key] = _numeric(values[:: self.frameskip])
        if self._transform is not None:
            sample = self._transform(sample)
        if "action" in sample:
            sample["action"] = sample["action"].reshape(self.num_steps, -1)
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        rows: list[int] = []
        for index in indices:
            episode, start = self._locate(int(index))
            offset = int(self._offsets[episode] + start)
            rows.extend(range(offset, offset + self.span))
        table = self._open().take(rows, columns=self._keys)
        return [
            self._sample_from_table(table.slice(position * self.span, self.span))
            for position in range(len(indices))
        ]

    def get_dim(self, column: str) -> int:
        if column == "action":
            return self._action_dim
        return _fixed_list_size(self._open().schema, column)

    def get_col_data(self, column: str) -> np.ndarray:
        values = self._open().to_table(columns=[column]).column(column).to_pylist()
        return np.asarray(values, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_dataset"] = None
        state["_dataset_pid"] = None
        state.pop("_trainer", None)
        return state


class _CubeBlockedSequence:
    """One four-model-step sample per frozen Cube release episode."""

    raw_action_dim = 5
    raw_steps_per_block = 5

    def __init__(
        self,
        path: Path,
        *,
        num_steps: int,
        frameskip: int,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        import lance

        if int(num_steps) != 4 or int(frameskip) != 5:
            raise ValueError(
                "Cube blocked projection requires num_steps=4 and frameskip=5"
            )
        self.path = path
        self.num_steps = 4
        self.frameskip = 5
        self._transform = transform
        # See _ProjectedLanceSequence: do not retain a native Lance handle in
        # the parent process that will later fork DataLoader workers.
        dataset = lance.dataset(str(path))
        required = {"episode_idx", "model_step_idx", "pixels", "action_block"}
        missing = sorted(required - set(dataset.schema.names))
        if missing:
            raise ValueError(f"Cube blocked projection {path} misses {missing}")
        width = _fixed_list_size(dataset.schema, "action_block")
        if width != self.raw_action_dim * self.raw_steps_per_block:
            raise ValueError(f"Cube action_block width must be 25, observed={width}")
        self._offsets, lengths = _episode_structure(
            dataset, step_column="model_step_idx"
        )
        if any(int(length) != self.num_steps for length in lengths):
            raise ValueError("Every Cube projection episode must contain four model steps")
        self._dataset = None
        self._dataset_pid = None

    @property
    def column_names(self) -> list[str]:
        return ["pixels", "action"]

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value) -> None:
        self._transform = value

    def __len__(self) -> int:
        return len(self._offsets)

    def _open(self):
        pid = os.getpid()
        if self._dataset is None or self._dataset_pid != pid:
            import lance

            self._dataset = lance.dataset(str(self.path))
            self._dataset_pid = pid
        return self._dataset

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        offset = int(self._offsets[index])
        table = self._open().take(
            list(range(offset, offset + self.num_steps)),
            columns=["pixels", "action_block"],
        )
        return self._sample_from_table(table)

    def _sample_from_table(self, table) -> dict[str, Any]:
        action = _numeric(table.column("action_block").to_pylist()).reshape(
            self.num_steps, self.raw_steps_per_block, self.raw_action_dim
        )
        sample = {
            "pixels": _decode_images(table.column("pixels").to_pylist()),
            "action": action,
        }
        if self._transform is not None:
            sample = self._transform(sample)
        sample["action"] = sample["action"].reshape(self.num_steps, -1)
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        rows: list[int] = []
        for index in indices:
            normalized = int(index)
            if normalized < 0:
                normalized += len(self)
            if not 0 <= normalized < len(self):
                raise IndexError(index)
            offset = int(self._offsets[normalized])
            rows.extend(range(offset, offset + self.num_steps))
        table = self._open().take(
            rows, columns=["pixels", "action_block"]
        )
        return [
            self._sample_from_table(
                table.slice(position * self.num_steps, self.num_steps)
            )
            for position in range(len(indices))
        ]

    def get_dim(self, column: str) -> int:
        if column != "action":
            raise KeyError(column)
        return self.raw_action_dim

    def get_col_data(self, column: str) -> np.ndarray:
        if column != "action":
            raise KeyError(column)
        values = self._open().to_table(columns=["action_block"]).column(0).to_pylist()
        return np.asarray(values, dtype=np.float32).reshape(-1, self.raw_action_dim)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_dataset"] = None
        state["_dataset_pid"] = None
        state.pop("_trainer", None)
        return state


def _fixed_list_size(schema, column: str) -> int:
    field_type = schema.field(column).type
    size = getattr(field_type, "list_size", None)
    if not isinstance(size, int):
        raise ValueError(f"Column {column!r} is not a fixed-size numeric vector")
    return int(size)


def _episode_structure(dataset, *, step_column: str) -> tuple[np.ndarray, np.ndarray]:
    table = dataset.to_table(columns=["episode_idx", step_column])
    episodes = np.asarray(table.column("episode_idx").to_pylist(), dtype=np.int64)
    steps = np.asarray(table.column(step_column).to_pylist(), dtype=np.int64)
    if not len(episodes):
        raise ValueError("ContextWorld Lance member is empty")
    starts = np.concatenate(([0], np.flatnonzero(episodes[1:] != episodes[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(episodes)]))
    if len(np.unique(episodes[starts])) != len(starts):
        raise ValueError("Lance episodes are not contiguous")
    for start, end in zip(starts, ends):
        if not np.array_equal(steps[start:end], np.arange(end - start)):
            raise ValueError(f"Non-contiguous {step_column} in Lance member")
    return starts.astype(np.int64), (ends - starts).astype(np.int64)


def _episode_relation_keys(
    path: Path,
    offsets: np.ndarray,
    lengths: np.ndarray,
    column: str,
) -> list[str]:
    """Read one public, constant-per-episode relation column.

    This runs while the lightweight Python index is built, never inside the
    model input path.  The value is used only to decide which episodes form a
    relation; it is not added to the sample and cannot reach the model.
    """

    import lance

    dataset = lance.dataset(str(path))
    if column not in set(dataset.schema.names):
        raise ValueError(
            f"Conditional-joint training needs the public {column!r} relation "
            f"column, which {path} does not publish"
        )
    values = dataset.to_table(columns=[column]).column(column).to_pylist()
    keys: list[str] = []
    for start, length in zip(offsets, lengths):
        window = {
            str(value) for value in values[int(start) : int(start) + int(length)]
        }
        if len(window) != 1:
            raise ValueError(
                f"Public relation column {column!r} is not constant within an "
                f"episode of {path}; the relation identity is ambiguous"
            )
        keys.append(window.pop())
    return keys


def _paired_episode_relations(leaf: Any) -> list[tuple[int, int]]:
    """Return aligned clip indices for every public same-query pair."""

    episodes: dict[str, list[int]] = {}
    for episode, key in enumerate(
        leaf.episode_relation_keys(PUBLIC_RELATION_COLUMN)
    ):
        episodes.setdefault(key, []).append(episode)
    relations: list[tuple[int, int]] = []
    for key in sorted(episodes):
        arms = episodes[key]
        if len(arms) != CONDITIONAL_JOINT_GROUP_WIDTH:
            raise ValueError(
                f"Public relation {key!r} has {len(arms)} episodes; expected 2"
            )
        ranges = [leaf.episode_clip_range(episode) for episode in arms]
        counts = {count for _, count in ranges}
        if len(counts) != 1:
            raise ValueError(
                f"Public relation {key!r} has unequal arm clip counts"
            )
        for offset in range(counts.pop()):
            relations.append(
                tuple(start + offset for start, _ in ranges)
            )
    if not relations:
        raise ValueError("Conditional-joint training publishes no pair relations")
    return relations

def _integer_weight_counts(weights: Sequence[float]) -> list[int]:
    values = [Fraction(str(float(weight))).limit_denominator(10_000) for weight in weights]
    total = sum(values, start=Fraction(0, 1))
    normalized = [value / total for value in values]
    denominator = math.lcm(*(value.denominator for value in normalized))
    counts = [value.numerator * (denominator // value.denominator) for value in normalized]
    divisor = reduce(math.gcd, counts)
    return [value // divisor for value in counts]


def _default_epoch_size(groups: Mapping[str, Any], weights: Mapping[str, float]) -> int:
    names = list(groups)
    counts = _integer_weight_counts([weights[name] for name in names])
    cycles = max(
        math.ceil(len(groups[name]) / count)
        for name, count in zip(names, counts)
    )
    return cycles * sum(counts)


@dataclass(frozen=True)
class _RegisteredActionNormalizerSource:
    """Tiny, picklable source used solely to fit StableWM action scaling.

    An explicitly synthetic-only component must not silently learn its action
    scaling from synthetic trajectories when its public contract registers the
    original-data statistics.  StableWM's :class:`ZScoreScaler` only needs
    ``get_col_data('action')``.  Two symmetric rows reproduce a declared
    *population* standard deviation exactly up to normal floating-point
    rounding, without adding original data as a training dependency.
    """

    action: np.ndarray

    def get_col_data(self, column: str) -> np.ndarray:
        if column != "action":
            raise KeyError(
                "Registered ContextWorld normalizer source exposes only 'action', "
                f"not {column!r}"
            )
        return self.action


def _registered_action_normalizer_source(
    component: Mapping[str, Any], *, action_dim: int
) -> _RegisteredActionNormalizerSource:
    """Build a fail-closed normalizer source from public component metadata.

    The Development contract is manifest-bound through the task registry
    before this function is reached.  The contract's stats describe original
    action data, while ``ZScoreScaler.fit`` always uses population standard
    deviation.  Encoding ``mean - std`` and ``mean + std`` therefore passes
    the registered values to that scaler without reading an original H5 file.
    """

    component_id = component.get("component_id")
    raw_component_dim = component.get("action_dimension")
    if (
        isinstance(raw_component_dim, bool)
        or not isinstance(raw_component_dim, int)
        or raw_component_dim <= 0
        or raw_component_dim != action_dim
    ):
        raise ValueError(
            "ContextWorld component action_dimension does not match its runtime "
            f"action dimension for {component_id!r}"
        )

    evaluation = _development_evaluation(component)
    normalization = evaluation.get("action_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError(
            "ContextWorld synthetic-only normalization requires "
            f"development_evaluation.action_normalization for {component_id!r}"
        )
    if normalization.get("transform") != "zscore":
        raise ValueError(
            "ContextWorld synthetic-only action normalization must use zscore "
            f"for {component_id!r}"
        )
    estimator = normalization.get("std_estimator")
    if estimator not in {"population", "unbiased"}:
        raise ValueError(
            "ContextWorld synthetic-only action normalization has unsupported "
            f"std_estimator for {component_id!r}: {estimator!r}"
        )

    mean_values = normalization.get("mean")
    std_values = normalization.get("std")
    if not isinstance(mean_values, list) or not isinstance(std_values, list):
        raise ValueError(
            "ContextWorld synthetic-only action normalization must provide list "
            f"mean/std values for {component_id!r}"
        )
    if len(mean_values) != action_dim or len(std_values) != action_dim:
        raise ValueError(
            "ContextWorld synthetic-only action normalization dimensions must "
            f"match action_dimension={action_dim} for {component_id!r}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (*mean_values, *std_values)
    ):
        raise ValueError(
            "ContextWorld synthetic-only action normalization must contain "
            f"numeric mean/std values for {component_id!r}"
        )
    try:
        mean = np.asarray(mean_values, dtype=np.float64)
        std = np.asarray(std_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ContextWorld synthetic-only action normalization must contain "
            f"numeric mean/std values for {component_id!r}"
        ) from exc
    if (
        mean.shape != (action_dim,)
        or std.shape != (action_dim,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            "ContextWorld synthetic-only action normalization has invalid "
            f"mean/std values for {component_id!r}"
        )

    # Fail rather than let addition/subtraction overflow and produce a
    # different scaler than the registered release contract.
    samples = np.stack((mean - std, mean + std), axis=0)
    if not np.isfinite(samples).all():
        raise ValueError(
            "ContextWorld synthetic-only action normalization cannot be encoded "
            f"as finite ZScoreScaler samples for {component_id!r}"
        )
    return _RegisteredActionNormalizerSource(action=samples)


class _ConditionalJointSubset:
    """Flat rows whose synthetic half is partitioned into complete pairs."""

    def __init__(self, runtime: "_RuntimeDataset", singles: Any, relations: Any):
        import torch

        self.runtime = runtime
        self.singles = torch.as_tensor(singles, dtype=torch.long).clone()
        self.relations = torch.as_tensor(relations, dtype=torch.long).clone()
        if self.singles.ndim != 1:
            raise ValueError("Conditional-joint singles must be one-dimensional")
        if self.relations.ndim != 2 or self.relations.size(1) != 2:
            raise ValueError("Conditional-joint relations must have shape (P,2)")
        self._global_indices = torch.cat(
            (self.singles, self.relations.reshape(-1)), dim=0
        )

    @property
    def column_names(self) -> list[str]:
        return [*self.runtime.column_names, CONDITIONAL_JOINT_GROUP_COLUMN]

    def __len__(self) -> int:
        return int(self._global_indices.numel())

    def _group_id(self, index: int) -> int:
        if index < self.singles.numel():
            return -1
        return (index - int(self.singles.numel())) // 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = dict(self.runtime[int(self._global_indices[index])])
        sample[CONDITIONAL_JOINT_GROUP_COLUMN] = self._group_id(index)
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        normalized = []
        for value in indices:
            index = int(value)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(value)
            normalized.append(index)
        global_indices = [int(self._global_indices[index]) for index in normalized]
        samples = self.runtime.__getitems__(global_indices)
        output = []
        for index, sample in zip(normalized, samples):
            value = dict(sample)
            value[CONDITIONAL_JOINT_GROUP_COLUMN] = self._group_id(index)
            output.append(value)
        return output

    def configure_train_loader(
        self, train_cfg: Mapping[str, Any], *, seed: int
    ) -> dict[str, Any]:
        import torch

        config = dict(train_cfg)
        batch_size = int(config.pop("batch_size"))
        config.pop("shuffle", None)
        config.pop("drop_last", None)
        config.pop("sampler", None)
        config.pop("batch_sampler", None)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = int(
                os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            )
            # Lightning constructs rank zero's loader before it exports
            # WORLD_SIZE, then re-executes the child ranks.  CUDA visibility
            # is already final at that point, so it is the correct fallback
            # for the unified ``devices=auto`` single-node contract.
            world_size = int(
                os.environ.get(
                    "WORLD_SIZE", str(max(1, torch.cuda.device_count()))
                )
            )
        relation_start = int(self.singles.numel())
        relation_ids = torch.arange(self.relations.size(0), dtype=torch.long)
        local_relations = torch.stack(
            (
                relation_start + 2 * relation_ids,
                relation_start + 2 * relation_ids + 1,
            ),
            dim=1,
        )
        config["batch_sampler"] = RelationBatchSampler(
            torch.arange(self.singles.numel(), dtype=torch.long),
            local_relations,
            batch_size=batch_size,
            epoch_row_count=len(self),
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
        return config

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_trainer", None)
        return state


class _RuntimeDataset:
    """Protocol-complete facade over the virtual group schedule."""

    def __init__(
        self,
        dataset: Any,
        leaves: Sequence[Any],
        *,
        normalizer_source=None,
        conditional_relations: Sequence[Sequence[int]] | None = None,
    ):
        self.dataset = dataset
        self.leaves = list(leaves)
        self.normalizer_source = normalizer_source
        self.conditional_relations = conditional_relations
        self._transform = None

    @property
    def column_names(self) -> list[str]:
        return list(self.dataset.column_names)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value) -> None:
        self._transform = value
        for leaf in self.leaves:
            leaf.transform = value

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __getitems__(self, indices: list[int]):
        getter = getattr(self.dataset, "__getitems__", None)
        return getter(indices) if getter else [self.dataset[index] for index in indices]

    def get_dim(self, column: str) -> int:
        dimensions = {int(leaf.get_dim(column)) for leaf in self.leaves}
        if len(dimensions) != 1:
            raise ValueError(f"Mixed dataset column {column!r} dimensions differ: {dimensions}")
        return dimensions.pop()

    def get_col_data(self, column: str) -> np.ndarray:
        source = self.normalizer_source
        if source is not None:
            return np.asarray(source.get_col_data(column))
        return np.concatenate(
            [np.asarray(leaf.get_col_data(column)) for leaf in self.leaves], axis=0
        )

    def split_for_training(
        self, *, train_fraction: float, generator: Any
    ) -> tuple[Any, Any]:
        """Split complete Contact relations and retain flat model inputs."""

        import torch

        if self.conditional_relations is None:
            return torch.utils.data.random_split(
                self,
                lengths=[train_fraction, 1.0 - train_fraction],
                generator=generator,
            )
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must be strictly between zero and one")
        if not isinstance(self.dataset, LogicalGroupDataset):
            raise ValueError("Conditional-joint training requires the registered mixture")
        if self.dataset.names != ["original", "synthetic"]:
            raise ValueError("Conditional-joint mixture must be original then synthetic")
        if self.dataset.counts != [1, 1] or len(self.dataset.schedule) != 2:
            raise ValueError("Conditional-joint training requires an exact 50/50 mixture")

        original_position = self.dataset.schedule.index(0)
        synthetic_position = self.dataset.schedule.index(1)
        draws = self.dataset.epoch_group_counts()
        if draws["original"] != draws["synthetic"]:
            raise ValueError("Conditional-joint epoch must expose equal mixture counts")
        synthetic_length = len(self.dataset.groups[1])
        local_relations = torch.as_tensor(
            self.conditional_relations, dtype=torch.long
        )
        relation_occurrences = []
        for offset in range(0, draws["synthetic"], synthetic_length):
            shifted = local_relations + offset
            active = shifted.max(dim=1).values < draws["synthetic"]
            relation_occurrences.append(
                shifted[active] * self.dataset.cycle_size + synthetic_position
            )
        relations = torch.cat(relation_occurrences, dim=0)
        if relations.numel() == 0:
            raise ValueError("Conditional-joint mixture exposes no complete relation")
        relation_order = torch.randperm(relations.size(0), generator=generator)
        train_relation_count = int(relations.size(0) * float(train_fraction))
        if train_relation_count <= 0 or train_relation_count >= relations.size(0):
            raise ValueError("Conditional-joint split leaves an empty partition")

        usable_originals = 2 * relations.size(0)
        if usable_originals > draws["original"]:
            raise ValueError("Conditional-joint relations exceed original mixture rows")
        original_occurrences = torch.randperm(
            draws["original"], generator=generator
        )[:usable_originals]
        original_indices = (
            original_occurrences * self.dataset.cycle_size + original_position
        )
        train_original_count = 2 * train_relation_count
        train = _ConditionalJointSubset(
            self,
            original_indices[:train_original_count],
            relations[relation_order[:train_relation_count]],
        )
        validation = _ConditionalJointSubset(
            self,
            original_indices[train_original_count:],
            relations[relation_order[train_relation_count:]],
        )
        return train, validation

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_trainer", None)
        return state


def _open_runtime_dataset(uri: str, **kwargs: Any):
    import stable_worldmodel as swm

    spec = parse_contextworld_dataset_uri(uri)
    identity = _describe_spec(spec)
    root = Path(identity["root"])
    contract = _bundle_contract(str(root))
    component = _component(contract, identity["component"])
    payload = _select_payload(
        component, split=identity["split"], payload_id=identity["payload_id"]
    )
    members = _payload_members(root, payload)

    num_steps = int(kwargs.get("num_steps", identity["history_length"] + 1))
    frameskip = int(kwargs.get("frameskip", identity["frameskip"]))
    if num_steps != int(identity["history_length"]) + 1:
        raise ValueError(
            f"Component {identity['component']} needs num_steps="
            f"{int(identity['history_length']) + 1}, observed={num_steps}"
        )
    if frameskip != int(identity["frameskip"]):
        raise ValueError(
            f"Component {identity['component']} needs frameskip="
            f"{identity['frameskip']}, observed={frameskip}"
        )
    keys = list(kwargs.get("keys_to_load") or MODEL_COLUMNS)
    if any(key not in MODEL_COLUMNS for key in keys):
        raise ValueError(
            "ContextWorld benchmark-component training exposes only pixels and action; "
            f"requested={keys}"
        )

    child_kwargs = {
        "num_steps": num_steps,
        "frameskip": frameskip,
        "transform": None,
        "keys_to_load": keys,
        "keys_to_cache": kwargs.get("keys_to_cache"),
    }
    adapter = identity["adapter"]
    leaves: list[Any] = []
    for member in members:
        if adapter == "stablewm_step_metadata_to_episode_table_v1":
            leaf = _ProjectedLanceSequence(
                member,
                num_steps=num_steps,
                frameskip=frameskip,
                keys_to_load=keys,
            )
        elif adapter == "cube_block_projection_to_sequence_v1":
            leaf = _CubeBlockedSequence(
                member, num_steps=num_steps, frameskip=frameskip
            )
        elif adapter is None:
            # Collection members already use the native episode/step schema,
            # but opening hundreds of them through LanceDB creates one
            # database catalog scan per table.  The direct Lance projection
            # has identical clip/action semantics and keeps startup linear in
            # the selected members.
            leaf = _ProjectedLanceSequence(
                member,
                num_steps=num_steps,
                frameskip=frameskip,
                keys_to_load=keys,
            )
        else:
            raise ValueError(f"Unsupported ContextWorld StableWM adapter: {adapter}")
        if int(leaf.get_dim("action")) != int(identity["action_dimension"]):
            raise ValueError(
                f"Component {identity['component']} raw action dimension mismatch"
            )
        leaves.append(leaf)

    conditional_joint = identity.get("conditional_joint")
    if conditional_joint is not None:
        if len(leaves) != 1:
            raise ValueError(
                "Conditional-joint training expects its "
                "single registered public Lance table"
            )
        synthetic = leaves[0]
        conditional_relations = _paired_episode_relations(synthetic)
    elif identity["component"] == "action_delay" and identity["payload_id"] == "full":
        delay_groups: dict[str, list[Any]] = {str(i): [] for i in range(5)}
        delay_groups["5_to_10"] = []
        for member, leaf in zip(members, leaves):
            match = _DELAY_PATTERN.search(member.name)
            if match is None:
                raise ValueError(f"Cannot derive action-delay group from {member.name}")
            delay = int(match.group("delay"))
            name = str(delay) if delay <= 4 else "5_to_10"
            if name not in delay_groups:
                raise ValueError(f"Unexpected action delay {delay} in {member.name}")
            delay_groups[name].append(leaf)
        if any(not values for values in delay_groups.values()):
            raise ValueError("ActionDelay full payload does not cover all six physical groups")
        grouped = {
            name: ScenarioBalancedDataset(values)
            for name, values in delay_groups.items()
        }
        group_weights = {name: 1.0 for name in grouped}
        synthetic = LogicalGroupDataset(
            grouped,
            group_weights,
            epoch_size=_default_epoch_size(grouped, group_weights),
        )
    else:
        synthetic = leaves[0] if len(leaves) == 1 else ScenarioBalancedDataset(leaves)
    if conditional_joint is None:
        conditional_relations = None

    weights = identity["weights"]
    original = None
    if weights["original"]:
        original = swm.data.load_dataset(
            str(identity["original_dataset"]), **child_kwargs
        )
        if int(original.get_dim("action")) != int(identity["action_dimension"]):
            raise ValueError("Original and ContextWorld raw action dimensions differ")
        top_groups = {"original": original, "synthetic": synthetic}
        top_weights = {
            "original": float(weights["original"]),
            "synthetic": float(weights["synthetic"]),
        }
        epoch_size = identity["epoch_size"] or _default_epoch_size(
            top_groups, top_weights
        )
        dataset = LogicalGroupDataset(
            top_groups, top_weights, epoch_size=int(epoch_size)
        )
        all_leaves = [original, *leaves]
    else:
        if identity["epoch_size"] is not None:
            dataset = LogicalGroupDataset(
                {"synthetic": synthetic},
                {"synthetic": 1.0},
                epoch_size=int(identity["epoch_size"]),
            )
        else:
            dataset = synthetic
        all_leaves = leaves

    normalizer_source = original
    if normalizer_source is None and identity["component"] == "door":
        # The default Door comparison recipe is a 50/50 original/synthetic
        # mixture.  Keep the registered original-data normalization available
        # when an explicit experiment overrides that recipe to synthetic-only.
        # Other components retain their existing runtime behavior unless a
        # future recipe explicitly adds the same contract-backed path.
        normalizer_source = _registered_action_normalizer_source(
            component,
            action_dim=int(identity["action_dimension"]),
        )

    runtime = _RuntimeDataset(
        dataset,
        all_leaves,
        # Component recipes fit action normalization on original data.  An
        # explicitly synthetic-only Door override uses the registered
        # original-data statistics above; it never needs the original H5 file
        # merely to scale its actions.
        normalizer_source=normalizer_source,
        conditional_relations=conditional_relations,
    )
    if kwargs.get("transform") is not None:
        runtime.transform = kwargs["transform"]
    return runtime


def register_stablewm_bundle_format() -> None:
    """Register the ContextWorld URI reader in the active StableWM process."""

    from stable_worldmodel.data.format import FORMATS, Format, register_format

    if FORMAT_NAME in FORMATS:
        return

    class ContextWorldBundleFormat(Format):
        name = FORMAT_NAME

        @classmethod
        def detect(cls, path) -> bool:
            return str(path).startswith(URI_PREFIX)

        @classmethod
        def open_reader(cls, path, **kwargs):
            return _open_runtime_dataset(str(path), **kwargs)

    register_format(ContextWorldBundleFormat)


__all__ = [
    "CONDITIONAL_JOINT_METHOD",
    "FORMAT_NAME",
    "URI_PREFIX",
    "build_contextworld_dataset_uri",
    "describe_contextworld_dataset",
    "parse_contextworld_dataset_uri",
    "register_stablewm_bundle_format",
    "resolve_contextworld_bundle",
    "resolve_contextworld_component",
    "resolve_contextworld_development_payload",
    "resolve_contextworld_development_payload_members",
]
