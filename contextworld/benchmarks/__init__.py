"""Public benchmark interfaces.

The first supported runtime is Stable-WorldModel LeWM.  The evaluator depends
on the adapter contract rather than on LeWM internals so additional projects
can be integrated without changing dataset or metric semantics.
"""

from .adapters import AdapterProtocol, SpeedICLModelAdapter
from .speed_icl_data import (
    DEFAULT_RELEASE_CONFIG,
    SpeedICLEvalBundle,
    SpeedICLEvalDataset,
    audit_speed_icl_release,
    build_speed_icl_training_data,
    export_speed_icl_artifacts,
    load_speed_icl_release,
)

__all__ = [
    "AdapterProtocol",
    "DEFAULT_RELEASE_CONFIG",
    "SpeedICLEvalBundle",
    "SpeedICLEvalDataset",
    "SpeedICLModelAdapter",
    "audit_speed_icl_release",
    "build_speed_icl_training_data",
    "export_speed_icl_artifacts",
    "load_speed_icl_release",
]
