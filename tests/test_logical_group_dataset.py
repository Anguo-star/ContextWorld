from collections import Counter

from contextworld.training.groups import (
    ConcatenatedDataset,
    LogicalGroupDataset,
    ScenarioBalancedDataset,
)


class DummyDataset:
    def __init__(self, label: str, length: int) -> None:
        self.label = label
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.label, index


class BatchedDummyDataset(DummyDataset):
    def __init__(self, label: str, length: int) -> None:
        super().__init__(label, length)
        self.batch_calls = []

    def __getitems__(self, indices: list[int]):
        self.batch_calls.append(list(indices))
        return [(self.label, index) for index in indices]


def test_concatenated_dataset_preserves_order_and_batched_reads() -> None:
    left = BatchedDummyDataset("left", 2)
    right = BatchedDummyDataset("right", 3)
    dataset = ConcatenatedDataset([left, right])

    assert len(dataset) == 5
    assert dataset.__getitems__([4, 0, 2, 1]) == [
        ("right", 2),
        ("left", 0),
        ("right", 0),
        ("left", 1),
    ]
    assert left.batch_calls == [[0, 1]]
    assert right.batch_calls == [[2, 0]]


def test_scenarios_are_balanced_only_inside_their_group() -> None:
    speed = ScenarioBalancedDataset(
        [DummyDataset("speed_a", 2), DummyDataset("speed_b", 7)]
    )
    groups = LogicalGroupDataset(
        {"original": DummyDataset("original", 100), "speed": speed},
        {"original": 0.5, "speed": 0.5},
        epoch_size=40,
    )

    top_counts = Counter(groups.locate(index)[0] for index in range(len(groups)))
    samples = [groups[index] for index in range(len(groups))]
    scenario_counts = Counter(label for label, _ in samples if label.startswith("speed"))

    assert top_counts == {"original": 20, "speed": 20}
    assert scenario_counts == {"speed_a": 10, "speed_b": 10}


def test_number_of_scenarios_does_not_amplify_logical_group_weight() -> None:
    few = ScenarioBalancedDataset([DummyDataset("few", 3)])
    many = ScenarioBalancedDataset(
        [DummyDataset(f"many_{index}", index + 1) for index in range(12)]
    )
    groups = LogicalGroupDataset(
        {"original": DummyDataset("original", 200), "few": few, "many": many},
        {"original": 0.5, "few": 0.25, "many": 0.25},
        epoch_size=120,
    )

    assert groups.epoch_group_counts() == {
        "original": 60,
        "few": 30,
        "many": 30,
    }


def test_composed_float_weights_reduce_to_small_exact_cycle() -> None:
    groups = LogicalGroupDataset(
        {
            "original": DummyDataset("o", 10),
            "speed": DummyDataset("s", 10),
            "door": DummyDataset("d", 10),
            "composition": DummyDataset("c", 10),
        },
        {
            "original": 0.50,
            "speed": 0.1666666667,
            "door": 0.1666666667,
            "composition": 0.1666666666,
        },
        epoch_size=120,
    )

    assert groups.counts == [3, 1, 1, 1]
    assert groups.epoch_group_counts() == {
        "original": 60,
        "speed": 20,
        "door": 20,
        "composition": 20,
    }


def test_epoch_size_must_complete_weight_cycle() -> None:
    try:
        LogicalGroupDataset(
            {"a": DummyDataset("a", 1), "b": DummyDataset("b", 1)},
            {"a": 2 / 3, "b": 1 / 3},
            epoch_size=10,
        )
    except ValueError as exc:
        assert "weight cycle" in str(exc)
    else:
        raise AssertionError("Expected incomplete weight cycle to fail")


def test_batched_reads_are_forwarded_and_original_order_is_restored() -> None:
    speed_a = BatchedDummyDataset("speed_a", 2)
    speed_b = BatchedDummyDataset("speed_b", 3)
    original = BatchedDummyDataset("original", 20)
    speed = ScenarioBalancedDataset([speed_a, speed_b])
    groups = LogicalGroupDataset(
        {"original": original, "speed": speed},
        {"original": 0.5, "speed": 0.5},
        epoch_size=20,
    )

    indices = [7, 0, 11, 4, 3, 18]
    expected = [groups[index] for index in indices]
    original.batch_calls.clear()
    speed_a.batch_calls.clear()
    speed_b.batch_calls.clear()

    assert groups.__getitems__(indices) == expected
    assert len(original.batch_calls) == 1
    assert len(speed_a.batch_calls) <= 1
    assert len(speed_b.batch_calls) <= 1


def test_epoch_coverage_reports_draws_unique_fraction_and_reuse() -> None:
    groups = LogicalGroupDataset(
        {"original": DummyDataset("o", 10), "speed": DummyDataset("s", 3)},
        {"original": 0.5, "speed": 0.5},
        epoch_size=20,
    )

    assert groups.epoch_group_coverage() == {
        "original": {
            "draws": 10,
            "available_virtual_slots": 10,
            "unique_virtual_slots": 10,
            "unique_virtual_slot_fraction": 1.0,
            "mean_draws_per_virtual_slot": 1.0,
        },
        "speed": {
            "draws": 10,
            "available_virtual_slots": 3,
            "unique_virtual_slots": 3,
            "unique_virtual_slot_fraction": 1.0,
            "mean_draws_per_virtual_slot": 10 / 3,
        },
    }
