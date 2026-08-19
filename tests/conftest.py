"""Suite-wide safeguards for legacy script modules with mutable globals.

Some archived Cube reference-training tests intentionally install Cube's
five-axis action contract into shared PushT script modules.  The production
Cube entrypoint runs in its own Python process; pytest does not.  Restore the
small dimension-only contract after every test so a historical Cube test
cannot change the interpretation of a later PushT payload.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


_SHARED_ACTION_DIMENSION_DEFAULTS = {
    # The archived script runners import these modules through the scripts
    # directory on ``sys.path``.  Other tests import the same source through
    # the ``scripts.`` package, so both module identities must be restored.
    "run_pusht_contact_friction_h3_train": {
        "ACTION_INPUT_DIM": 10,
    },
    "run_pusht_hidden_actuation_pilot": {
        "ACTION_DIM": 2,
        "ACTION_INPUT_DIM": 10,
    },
    "run_pusht_hidden_actuation_mixed": {
        "ACTION_INPUT_DIM": 10,
    },
    "scripts.run_pusht_contact_friction_h3_train": {
        "ACTION_INPUT_DIM": 10,
    },
    "scripts.run_pusht_hidden_actuation_pilot": {
        "ACTION_DIM": 2,
        "ACTION_INPUT_DIM": 10,
    },
    "scripts.run_pusht_hidden_actuation_mixed": {
        "ACTION_INPUT_DIM": 10,
    },
}


@pytest.fixture(autouse=True)
def _restore_import_search_path() -> Iterator[None]:
    """Keep checkout-specific import paths local to one test case."""

    before = list(sys.path)
    yield
    sys.path[:] = before


@pytest.fixture(autouse=True)
def _restore_shared_action_dimensions() -> Iterator[None]:
    """Keep Cube's private five-axis install local to one pytest case."""

    before = {
        (module_name, attribute): getattr(module, attribute)
        for module_name, attributes in _SHARED_ACTION_DIMENSION_DEFAULTS.items()
        if (module := sys.modules.get(module_name)) is not None
        for attribute in attributes
        if hasattr(module, attribute)
    }
    yield
    for module_name, defaults in _SHARED_ACTION_DIMENSION_DEFAULTS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute, default in defaults.items():
            setattr(module, attribute, before.get((module_name, attribute), default))
