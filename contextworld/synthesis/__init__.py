"""Composable scenario compilation, collection, and validation."""

from importlib import import_module
from typing import Any

__all__ = [
    "AtomRequest",
    "CompiledScenario",
    "ScenarioCompiler",
    "ScenarioRequest",
]


_LAZY_EXPORTS = {
    "ScenarioCompiler": ("compiler", "ScenarioCompiler"),
    "AtomRequest": ("models", "AtomRequest"),
    "CompiledScenario": ("models", "CompiledScenario"),
    "ScenarioRequest": ("models", "ScenarioRequest"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
