"""Public benchmark interfaces.

The reference runtime supports Stable-WorldModel LeWM and PLDM checkpoints.
Evaluators depend on small adapter contracts rather than project internals, so
other world-model implementations can reuse the same data and scores.

Dataset readers are imported lazily.  This keeps the public adapter boundary
available from the core installation while preserving the existing package-
level names for users who install the evaluation extras.
"""

from importlib import import_module
from typing import Any

from .adapters import (
    ActionDelayICLModelAdapter,
    ActionStrengthICLModelAdapter,
    AdapterProtocol,
    ContactFrictionICLModelAdapter,
    CubeGraspRuleICLModelAdapter,
    LatentWorldModelAdapter,
    MotionDampingICLModelAdapter,
    PortalExitICLModelAdapter,
    ReacherArmMassICLModelAdapter,
    DoorICLModelAdapter,
    SpeedICLModelAdapter,
)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG": (
        "cube_grasp_rule_v4r1_icl_data",
        "DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG",
    ),
    "CubeGraspRuleV4R1ICLEvalDataset": (
        "cube_grasp_rule_v4r1_icl_data",
        "CubeGraspRuleV4R1ICLEvalDataset",
    ),
    "audit_cube_grasp_rule_v4r1_icl_release": (
        "cube_grasp_rule_v4r1_icl_data",
        "audit_cube_grasp_rule_v4r1_icl_release",
    ),
    "load_cube_grasp_rule_v4r1_icl_release": (
        "cube_grasp_rule_v4r1_icl_data",
        "load_cube_grasp_rule_v4r1_icl_release",
    ),
    "recompute_cube_grasp_rule_v4r1_public_reference": (
        "cube_grasp_rule_v4r1_icl_data",
        "recompute_cube_grasp_rule_v4r1_public_reference",
    ),
    "X0_POLICIES": ("causal_data_contract", "X0_POLICIES"),
    "audit_causal_data_contract": (
        "causal_data_contract",
        "audit_causal_data_contract",
    ),
    "DEFAULT_MOTION_DAMPING_RELEASE_CONFIG": (
        "motion_damping_icl_data",
        "DEFAULT_MOTION_DAMPING_RELEASE_CONFIG",
    ),
    "MotionDampingICLEvalDataset": (
        "motion_damping_icl_data",
        "MotionDampingICLEvalDataset",
    ),
    "audit_motion_damping_icl_release": (
        "motion_damping_icl_data",
        "audit_motion_damping_icl_release",
    ),
    "load_motion_damping_icl_release": (
        "motion_damping_icl_data",
        "load_motion_damping_icl_release",
    ),
    "DEFAULT_PORTAL_EXIT_RELEASE_CONFIG": (
        "portal_exit_icl_data",
        "DEFAULT_PORTAL_EXIT_RELEASE_CONFIG",
    ),
    "PortalExitICLEvalDataset": (
        "portal_exit_icl_data",
        "PortalExitICLEvalDataset",
    ),
    "audit_portal_exit_icl_release": (
        "portal_exit_icl_data",
        "audit_portal_exit_icl_release",
    ),
    "load_portal_exit_icl_release": (
        "portal_exit_icl_data",
        "load_portal_exit_icl_release",
    ),
    "DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG": (
        "reacher_arm_mass_icl_data",
        "DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG",
    ),
    "ReacherArmMassICLEvalDataset": (
        "reacher_arm_mass_icl_data",
        "ReacherArmMassICLEvalDataset",
    ),
    "audit_reacher_arm_mass_icl_release": (
        "reacher_arm_mass_icl_data",
        "audit_reacher_arm_mass_icl_release",
    ),
    "load_reacher_arm_mass_icl_release": (
        "reacher_arm_mass_icl_data",
        "load_reacher_arm_mass_icl_release",
    ),
    "resolve_reacher_initial_checkpoint": (
        "reacher_arm_mass_icl_data",
        "resolve_reacher_initial_checkpoint",
    ),
    "resolve_reacher_initial_checkpoint_config": (
        "reacher_arm_mass_icl_data",
        "resolve_reacher_initial_checkpoint_config",
    ),
    "resolve_reacher_original_h5": (
        "reacher_arm_mass_icl_data",
        "resolve_reacher_original_h5",
    ),
    "resolve_reacher_original_lance": (
        "reacher_arm_mass_icl_data",
        "resolve_reacher_original_lance",
    ),
    "DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG": (
        "contact_friction_icl_data",
        "DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG",
    ),
    "ContactFrictionICLEvalDataset": (
        "contact_friction_icl_data",
        "ContactFrictionICLEvalDataset",
    ),
    "audit_contact_friction_icl_release": (
        "contact_friction_icl_data",
        "audit_contact_friction_icl_release",
    ),
    "load_contact_friction_icl_release": (
        "contact_friction_icl_data",
        "load_contact_friction_icl_release",
    ),
    "DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG": (
        "action_strength_icl_data",
        "DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG",
    ),
    "ActionStrengthICLEvalDataset": (
        "action_strength_icl_data",
        "ActionStrengthICLEvalDataset",
    ),
    "action_strength_icl_evaluation_plans": (
        "action_strength_icl_data",
        "action_strength_icl_evaluation_plans",
    ),
    "action_strength_icl_training_plan": (
        "action_strength_icl_data",
        "action_strength_icl_training_plan",
    ),
    "audit_action_strength_icl_release": (
        "action_strength_icl_data",
        "audit_action_strength_icl_release",
    ),
    "load_action_strength_icl_release": (
        "action_strength_icl_data",
        "load_action_strength_icl_release",
    ),
    "DEFAULT_ACTION_DELAY_RELEASE_CONFIG": (
        "action_delay_icl_data",
        "DEFAULT_ACTION_DELAY_RELEASE_CONFIG",
    ),
    "ActionDelayICLEvalDataset": (
        "action_delay_icl_data",
        "ActionDelayICLEvalDataset",
    ),
    "action_delay_icl_training_plan": (
        "action_delay_icl_data",
        "action_delay_icl_training_plan",
    ),
    "audit_action_delay_icl_release": (
        "action_delay_icl_data",
        "audit_action_delay_icl_release",
    ),
    "load_action_delay_icl_release": (
        "action_delay_icl_data",
        "load_action_delay_icl_release",
    ),
    "DEFAULT_DOOR_RELEASE_CONFIG": (
        "door_icl_data",
        "DEFAULT_DOOR_RELEASE_CONFIG",
    ),
    "DoorICLEvalDataset": ("door_icl_data", "DoorICLEvalDataset"),
    "DoorICLEvalExample": ("door_icl_data", "DoorICLEvalExample"),
    "audit_door_icl_release": (
        "door_icl_data",
        "audit_door_icl_release",
    ),
    "door_icl_training_plan": (
        "door_icl_data",
        "door_icl_training_plan",
    ),
    "export_door_icl_artifacts": (
        "door_icl_data",
        "export_door_icl_artifacts",
    ),
    "load_door_icl_release": ("door_icl_data", "load_door_icl_release"),
    "DEFAULT_RELEASE_CONFIG": ("speed_icl_data", "DEFAULT_RELEASE_CONFIG"),
    "SpeedICLEvalBundle": ("speed_icl_data", "SpeedICLEvalBundle"),
    "SpeedICLEvalDataset": ("speed_icl_data", "SpeedICLEvalDataset"),
    "audit_speed_icl_release": (
        "speed_icl_data",
        "audit_speed_icl_release",
    ),
    "build_speed_icl_training_data": (
        "speed_icl_data",
        "build_speed_icl_training_data",
    ),
    "export_speed_icl_artifacts": (
        "speed_icl_data",
        "export_speed_icl_artifacts",
    ),
    "load_speed_icl_release": ("speed_icl_data", "load_speed_icl_release"),
    "DEFAULT_SUITE_RELEASE_CONFIG": (
        "suite_data",
        "DEFAULT_SUITE_RELEASE_CONFIG",
    ),
    "DEFAULT_SUITE_V2_RELEASE_CONFIG": (
        "suite_data",
        "DEFAULT_SUITE_V2_RELEASE_CONFIG",
    ),
    "audit_icl_suite_release": ("suite_data", "audit_icl_suite_release"),
    "export_icl_suite_artifacts": (
        "suite_data",
        "export_icl_suite_artifacts",
    ),
    "load_icl_suite_release": ("suite_data", "load_icl_suite_release"),
    "resolve_suite_v2_cli_default_config": (
        "suite_data",
        "resolve_suite_v2_cli_default_config",
    ),
}

_EVALUATION_DEPENDENCY_MODULES = {
    "PIL",
    "gymnasium",
    "h5py",
    "jsonschema",
    "lance",
    "pyarrow",
    "pymunk",
    "scipy",
}


def __getattr__(name: str) -> Any:
    """Load evaluation-only package exports on first access."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in _EVALUATION_DEPENDENCY_MODULES:
            raise ModuleNotFoundError(
                f"{name} requires the optional evaluation dependencies; "
                'install ContextWorld with `pip install "contextworld[eval]"`.'
            ) from exc
        raise
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "ActionDelayICLEvalDataset",
    "ActionDelayICLModelAdapter",
    "ActionStrengthICLEvalDataset",
    "ActionStrengthICLModelAdapter",
    "AdapterProtocol",
    "ContactFrictionICLEvalDataset",
    "ContactFrictionICLModelAdapter",
    "CubeGraspRuleICLModelAdapter",
    "CubeGraspRuleV4R1ICLEvalDataset",
    "LatentWorldModelAdapter",
    "MotionDampingICLEvalDataset",
    "MotionDampingICLModelAdapter",
    "PortalExitICLEvalDataset",
    "PortalExitICLModelAdapter",
    "ReacherArmMassICLEvalDataset",
    "ReacherArmMassICLModelAdapter",
    "DEFAULT_ACTION_DELAY_RELEASE_CONFIG",
    "DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG",
    "DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG",
    "DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG",
    "DEFAULT_MOTION_DAMPING_RELEASE_CONFIG",
    "DEFAULT_PORTAL_EXIT_RELEASE_CONFIG",
    "DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG",
    "DEFAULT_DOOR_RELEASE_CONFIG",
    "DEFAULT_RELEASE_CONFIG",
    "DEFAULT_SUITE_RELEASE_CONFIG",
    "DEFAULT_SUITE_V2_RELEASE_CONFIG",
    "DoorICLEvalDataset",
    "DoorICLEvalExample",
    "DoorICLModelAdapter",
    "SpeedICLEvalBundle",
    "SpeedICLEvalDataset",
    "SpeedICLModelAdapter",
    "X0_POLICIES",
    "action_delay_icl_training_plan",
    "action_strength_icl_training_plan",
    "action_strength_icl_evaluation_plans",
    "audit_action_delay_icl_release",
    "audit_action_strength_icl_release",
    "audit_causal_data_contract",
    "audit_contact_friction_icl_release",
    "audit_cube_grasp_rule_v4r1_icl_release",
    "audit_motion_damping_icl_release",
    "audit_portal_exit_icl_release",
    "audit_reacher_arm_mass_icl_release",
    "audit_door_icl_release",
    "audit_icl_suite_release",
    "audit_speed_icl_release",
    "build_speed_icl_training_data",
    "door_icl_training_plan",
    "export_door_icl_artifacts",
    "export_icl_suite_artifacts",
    "export_speed_icl_artifacts",
    "load_door_icl_release",
    "load_action_delay_icl_release",
    "load_action_strength_icl_release",
    "load_contact_friction_icl_release",
    "load_cube_grasp_rule_v4r1_icl_release",
    "load_motion_damping_icl_release",
    "load_portal_exit_icl_release",
    "load_reacher_arm_mass_icl_release",
    "resolve_reacher_initial_checkpoint",
    "resolve_reacher_initial_checkpoint_config",
    "resolve_reacher_original_h5",
    "resolve_reacher_original_lance",
    "load_icl_suite_release",
    "resolve_suite_v2_cli_default_config",
    "load_speed_icl_release",
    "recompute_cube_grasp_rule_v4r1_public_reference",
]
