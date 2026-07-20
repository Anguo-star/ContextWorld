from contextworld.synthesis.config import scenario_requests


def test_stratified_pool_is_deterministic_and_globally_separated() -> None:
    config = {
        "seed": 11,
        "scenario_generation_seed": 19,
        "scenario_value_pools": {
            "speed": {
                "sampler": "stratified_unique",
                "range": [2.5, 3.5],
                "count": 6,
                "shuffle": True,
            }
        },
        "scenario_sets": [
            {
                "name_prefix": "train",
                "split": "train",
                "atom": "agent_speed",
                "minimum_gap": 0.15,
                "values": {"pool": "speed", "start": 0, "count": 3},
            },
            {
                "name_prefix": "test",
                "split": "test",
                "atom": "agent_speed",
                "minimum_gap": 0.15,
                "values": {"pool": "speed", "start": 3, "count": 3},
            },
        ],
    }

    first = scenario_requests(config)
    second = scenario_requests(config)
    first_values = [float(request.atoms[0].value) for request in first]
    second_values = [float(request.atoms[0].value) for request in second]

    assert first_values == second_values
    assert len(set(first_values)) == 6
    assert min(
        abs(left - right)
        for index, left in enumerate(first_values)
        for right in first_values[index + 1 :]
    ) >= 0.15


def test_explicit_combination_set_preserves_values_and_atom_order() -> None:
    config = {
        "seed": 23,
        "scenario_sets": [
            {
                "name_prefix": "speed_door_train",
                "split": "train",
                "regime": "train_combinations",
                "episodes_per_scenario": 6,
                "seed_group": "paired",
                "combinations": [
                    {"door_position": 61, "agent_speed": 3.0},
                    {"agent_speed": 4.0, "door_position": 85},
                ],
            }
        ],
    }

    requests = scenario_requests(config)

    assert [request.name for request in requests] == [
        (
            "speed_door_train_000_"
            "agent_speed_v3_door_position_v61"
        ),
        (
            "speed_door_train_001_"
            "agent_speed_v4_door_position_v85"
        ),
    ]
    assert [
        [(atom.kind, atom.value) for atom in request.atoms]
        for request in requests
    ] == [
        [("agent_speed", 3.0), ("door_position", 61)],
        [("agent_speed", 4.0), ("door_position", 85)],
    ]
    assert all(request.episodes == 6 for request in requests)
    assert all(request.seed_group == "paired" for request in requests)


def test_combination_set_rejects_single_atom_entries() -> None:
    config = {
        "seed": 23,
        "scenario_sets": [
            {
                "name_prefix": "invalid",
                "split": "train",
                "combinations": [{"agent_speed": 3.0}],
            }
        ],
    }

    try:
        scenario_requests(config)
    except ValueError as exc:
        assert "at least two atom" in str(exc)
    else:
        raise AssertionError("Expected invalid combination table to fail")


def test_seed_groups_cross_every_generated_value() -> None:
    config = {
        "seed": 31,
        "scenario_sets": [
            {
                "name_prefix": "speed",
                "split": "train",
                "atom": "agent_speed",
                "episodes_per_scenario": 3,
                "seed_groups": ["block-a", "block-b"],
                "values": {"sampler": "fixed", "values": [3.0, 7.0]},
            }
        ],
    }

    requests = scenario_requests(config)

    assert len(requests) == 4
    assert {request.seed_group for request in requests} == {
        "block-a",
        "block-b",
    }
    assert {
        (request.seed_group, float(request.atoms[0].value))
        for request in requests
    } == {
        ("block-a", 3.0),
        ("block-a", 7.0),
        ("block-b", 3.0),
        ("block-b", 7.0),
    }
    assert len({request.name for request in requests}) == 4


def test_indexed_seed_groups_expand_compactly() -> None:
    config = {
        "seed": 31,
        "scenario_sets": [
            {
                "name_prefix": "speed",
                "split": "train",
                "atom": "agent_speed",
                "episodes_per_scenario": 4,
                "seed_groups": {
                    "sampler": "indexed",
                    "prefix": "speed-task",
                    "start": 3,
                    "count": 3,
                    "width": 3,
                },
                "values": {"sampler": "fixed", "values": [3.0, 5.0]},
            }
        ],
    }

    requests = scenario_requests(config)

    assert len(requests) == 6
    assert {request.seed_group for request in requests} == {
        "speed-task_003",
        "speed-task_004",
        "speed-task_005",
    }


def test_paired_cycle_assigns_each_seed_group_to_one_balanced_value() -> None:
    config = {
        "seed": 31,
        "scenario_sets": [
            {
                "name_prefix": "independent",
                "split": "train",
                "atom": "agent_speed",
                "assignment": "paired_cycle",
                "episodes_per_scenario": 2,
                "seed_groups": {
                    "sampler": "indexed",
                    "prefix": "independent",
                    "count": 6,
                },
                "values": {"sampler": "fixed", "values": [3.0, 5.0, 7.0]},
                "reset_constraints": {"target_room": "same"},
            }
        ],
    }

    requests = scenario_requests(config)

    assert len(requests) == 6
    assert len({request.seed_group for request in requests}) == 6
    assert [request.atoms[0].value for request in requests] == [
        3.0,
        5.0,
        7.0,
        3.0,
        5.0,
        7.0,
    ]
    assert all(request.reset_constraints == {"target_room": "same"} for request in requests)
