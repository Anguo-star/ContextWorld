"""Resolve the model a command line should evaluate.

ContextWorld's scoring boundary has always been model-independent: the
scorers talk to :class:`~contextworld.benchmarks.adapters.LatentWorldModelAdapter`,
and :func:`~contextworld.benchmarks.adapters.validate_adapter_protocol`
deliberately checks task geometry without constraining model family, latent
width or framework.  What was *not* open was the command line in front of it.
Every task CLI declared ``--adapter`` with ``choices=("lewm", "pldm")``, so a
third-party model could satisfy the entire scoring contract and still have no
way to be invoked.  This module removes that barrier without touching the
sealed adapter definitions.

A specification may name

* a built-in family (``lewm``, ``pldm``) supplied by the calling CLI;
* an import path, ``package.module:ClassName``, for a class already on
  ``sys.path``;
* an entry point registered by an installed distribution under the
  ``contextworld.adapters`` group.

Built-in names are resolved first and cannot be overridden by an installed
package.  That precedence is a correctness requirement, not a convenience:
``lewm`` and ``pldm`` identify the frozen reference baselines behind published
scoreboard rows, and a third-party distribution that could rebind those names
would silently change what a published number means.

External adapters are constructed through :class:`AdapterRequest` and the
``from_contextworld_request`` classmethod.  The built-in Stable-WorldModel
adapters predate that contract and are sealed, so they keep their existing
``from_checkpoint`` constructors and this module translates a request into the
appropriate call.  Nothing in ``adapters.py`` has to change for an external
model to run.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from contextworld.benchmarks.adapters import LatentWorldModelAdapter


ENTRY_POINT_GROUP = "contextworld.adapters"

# ``package.module:ClassName``.  A colon is required rather than accepting a
# trailing dotted attribute, so that an import path is never confused with a
# bare built-in or entry-point name.
_IMPORT_PATH_SEPARATOR = ":"


class AdapterResolutionError(ValueError):
    """A model specification could not be turned into an adapter class."""


class AdapterConstructionError(TypeError):
    """An adapter class was found but could not be constructed."""


@dataclass(frozen=True)
class AdapterRequest:
    """Everything a CLI knows about the model it was asked to evaluate.

    The fields describe the *evaluation*, not the implementation.  An external
    adapter is expected to read ``checkpoint`` and ``device`` and to ignore
    whatever does not apply to it; ``runtime`` deliberately carries the
    Stable-WorldModel repository pin as an opaque mapping rather than as named
    arguments, because it is meaningless to a model built on another stack.

    Action standardization arrives in one of two shapes because the benchmark
    tasks were frozen that way: four tasks carry a normalizer artifact and
    eight carry explicit per-dimension statistics.  Exactly one of the two is
    populated.
    """

    task: str
    checkpoint: Path
    device: str
    repo_root: Path
    action_normalizer: Path | None = None
    action_mean: Sequence[float] | None = None
    action_std: Sequence[float] | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_normalizer = self.action_normalizer is not None
        has_statistics = self.action_mean is not None or self.action_std is not None
        if has_normalizer and has_statistics:
            raise ValueError(
                f"{self.task}: an adapter request carries both a normalizer "
                "artifact and explicit action statistics; a task is frozen "
                "with one or the other"
            )
        if (self.action_mean is None) != (self.action_std is None):
            raise ValueError(
                f"{self.task}: action_mean and action_std must be supplied "
                "together"
            )


def _entry_points() -> dict[str, Any]:
    """Return ``name -> entry point`` for the adapter group, if any."""

    from importlib.metadata import entry_points

    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        found = entry_points().get(ENTRY_POINT_GROUP, ())
    return {entry.name: entry for entry in found}


def _import_from_path(spec: str) -> Any:
    module_name, _, attribute = spec.partition(_IMPORT_PATH_SEPARATOR)
    if not module_name or not attribute:
        raise AdapterResolutionError(
            f"{spec!r} is not a usable import path; expected "
            "'package.module:ClassName'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterResolutionError(
            f"could not import module {module_name!r} for adapter {spec!r}: "
            f"{exc}"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AdapterResolutionError(
            f"module {module_name!r} has no attribute {attribute!r}; "
            f"cannot resolve adapter {spec!r}"
        ) from exc


def _validate_adapter_class(candidate: Any, *, spec: str) -> type:
    if not isinstance(candidate, type):
        raise AdapterResolutionError(
            f"adapter {spec!r} resolved to {type(candidate).__name__}, not a "
            "class"
        )
    if not issubclass(candidate, LatentWorldModelAdapter):
        raise AdapterResolutionError(
            f"adapter {spec!r} resolved to {candidate.__name__}, which does "
            "not subclass LatentWorldModelAdapter; ContextWorld scorers "
            "require the encode_pixels/rollout_latents/protocol/metadata/"
            "frozen_state_hash contract"
        )
    missing = sorted(getattr(candidate, "__abstractmethods__", ()))
    if missing:
        raise AdapterResolutionError(
            f"adapter {spec!r} resolved to {candidate.__name__}, which is "
            f"still abstract; unimplemented members: {', '.join(missing)}"
        )
    return candidate


def resolve_adapter_class(
    spec: str, *, builtins: Mapping[str, type]
) -> type:
    """Turn a model specification into a concrete adapter class.

    Resolution order is built-in name, then import path, then entry point.
    """

    if not spec or not spec.strip():
        raise AdapterResolutionError("no adapter specification was supplied")
    spec = spec.strip()

    if spec in builtins:
        return _validate_adapter_class(builtins[spec], spec=spec)

    if _IMPORT_PATH_SEPARATOR in spec:
        return _validate_adapter_class(_import_from_path(spec), spec=spec)

    registered = _entry_points()
    if spec in registered:
        try:
            loaded = registered[spec].load()
        except Exception as exc:  # noqa: BLE001 - third-party import surface
            raise AdapterResolutionError(
                f"entry point {spec!r} in group {ENTRY_POINT_GROUP!r} failed "
                f"to load: {exc}"
            ) from exc
        return _validate_adapter_class(loaded, spec=spec)

    known = sorted(builtins)
    available = sorted(registered)
    raise AdapterResolutionError(
        f"unknown adapter {spec!r}. Built-in adapters for this task: "
        f"{', '.join(known) if known else '(none)'}. "
        f"Installed {ENTRY_POINT_GROUP!r} entry points: "
        f"{', '.join(available) if available else '(none)'}. "
        "To evaluate an external model, pass an import path such as "
        "'my_package.adapter:MyAdapter', or install a distribution that "
        f"registers a {ENTRY_POINT_GROUP!r} entry point."
    )


def _construct_from_checkpoint(
    adapter_class: type, request: AdapterRequest, *, spec: str
) -> LatentWorldModelAdapter:
    """Call a sealed built-in constructor with the shape it expects."""

    runtime = dict(request.runtime)
    common = {
        "repo_root": request.repo_root,
        "device": request.device,
        "stablewm_repo": runtime.get("stablewm_repo"),
        "stablewm_ref": runtime.get("stablewm_ref"),
    }
    if request.action_normalizer is not None:
        keywords = {"normalizer": request.action_normalizer, **common}
    elif request.action_mean is not None:
        keywords = {
            "action_mean": request.action_mean,
            "action_std": request.action_std,
            **common,
        }
    else:
        raise AdapterConstructionError(
            f"{request.task}: adapter {spec!r} uses the built-in "
            "from_checkpoint constructor, which needs either a normalizer "
            "artifact or explicit action statistics, and the request carried "
            "neither"
        )
    return adapter_class.from_checkpoint(request.checkpoint, **keywords)


def build_adapter(
    spec: str, *, builtins: Mapping[str, type], request: AdapterRequest
) -> LatentWorldModelAdapter:
    """Resolve ``spec`` and construct the adapter it names.

    A class that implements ``from_contextworld_request`` is constructed
    through it.  Anything else falls back to the built-in ``from_checkpoint``
    shape, which is how the sealed Stable-WorldModel adapters keep working
    unmodified.
    """

    adapter_class = resolve_adapter_class(spec, builtins=builtins)

    constructor = getattr(adapter_class, "from_contextworld_request", None)
    if callable(constructor):
        adapter = constructor(request)
    elif callable(getattr(adapter_class, "from_checkpoint", None)):
        adapter = _construct_from_checkpoint(adapter_class, request, spec=spec)
    else:
        raise AdapterConstructionError(
            f"adapter {spec!r} ({adapter_class.__name__}) exposes neither "
            "from_contextworld_request(request) nor from_checkpoint(...); "
            "ContextWorld cannot construct it"
        )

    if not isinstance(adapter, LatentWorldModelAdapter):
        raise AdapterConstructionError(
            f"adapter {spec!r} ({adapter_class.__name__}) constructed a "
            f"{type(adapter).__name__}, which is not a LatentWorldModelAdapter"
        )
    return adapter


def add_adapter_argument(
    parser: Any,
    *,
    builtins: Mapping[str, type],
    flag: str = "--adapter",
    default: str | None = "lewm",
    required: bool = False,
) -> None:
    """Declare a task CLI's model-selection flag.

    The flag deliberately carries no ``choices``.  Restricting it to the
    built-in families is what made external models unreachable, and argparse's
    rejection message could only ever list the two baselines.  Validation
    moves to :func:`resolve_adapter_class`, which reports the built-ins, the
    installed entry points, and how to pass an import path.
    """

    names = ", ".join(sorted(builtins))
    parser.add_argument(
        flag,
        default=default,
        required=required,
        metavar="ADAPTER",
        help=(
            f"Model to evaluate. Built-in: {names}. Also accepts an import "
            "path 'package.module:ClassName', or the name of an installed "
            f"{ENTRY_POINT_GROUP!r} entry point."
        ),
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "AdapterConstructionError",
    "AdapterRequest",
    "AdapterResolutionError",
    "add_adapter_argument",
    "build_adapter",
    "resolve_adapter_class",
]
