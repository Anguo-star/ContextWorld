"""Public benchmark interfaces.

The first supported runtime is Stable-WorldModel LeWM.  The evaluator depends
on the adapter contract rather than on LeWM internals so additional projects
can be integrated without changing dataset or metric semantics.
"""

from .adapters import (
    AdapterProtocol,
    DoorICLModelAdapter,
    SpeedICLModelAdapter,
)
from .door_icl_data import (
    DEFAULT_DOOR_RELEASE_CONFIG,
    DoorICLEvalDataset,
    DoorICLEvalExample,
    audit_door_icl_release,
    door_icl_training_plan,
    export_door_icl_artifacts,
    load_door_icl_release,
)
from .speed_icl_data import (
    DEFAULT_RELEASE_CONFIG,
    SpeedICLEvalBundle,
    SpeedICLEvalDataset,
    audit_speed_icl_release,
    build_speed_icl_training_data,
    export_speed_icl_artifacts,
    load_speed_icl_release,
)
from .suite_data import (
    DEFAULT_SUITE_RELEASE_CONFIG,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_icl_suite_release,
)

__all__ = [
    "AdapterProtocol",
    "DEFAULT_DOOR_RELEASE_CONFIG",
    "DEFAULT_RELEASE_CONFIG",
    "DEFAULT_SUITE_RELEASE_CONFIG",
    "DoorICLEvalDataset",
    "DoorICLEvalExample",
    "DoorICLModelAdapter",
    "SpeedICLEvalBundle",
    "SpeedICLEvalDataset",
    "SpeedICLModelAdapter",
    "audit_door_icl_release",
    "audit_icl_suite_release",
    "audit_speed_icl_release",
    "build_speed_icl_training_data",
    "door_icl_training_plan",
    "export_door_icl_artifacts",
    "export_icl_suite_artifacts",
    "export_speed_icl_artifacts",
    "load_door_icl_release",
    "load_icl_suite_release",
    "load_speed_icl_release",
]
