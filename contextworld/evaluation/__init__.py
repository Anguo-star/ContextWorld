"""Evaluation adapters for ContextWorld benchmark datasets."""

from importlib import import_module
from typing import Any

__all__ = [
    "TWOROOM_EVAL_ENV_ID",
    "register_tworoom_eval_env",
    "tworoom_eval_callables",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.tworoom"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
