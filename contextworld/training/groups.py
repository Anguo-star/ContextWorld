from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import reduce
from typing import Any


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


def _weight_counts(weights: Sequence[float]) -> list[int]:
    if not weights or any(float(weight) <= 0 for weight in weights):
        raise ValueError(f"Logical-group weights must be positive: {weights}")
    fractions = [
        Fraction(str(float(weight))).limit_denominator(10_000)
        for weight in weights
    ]
    total = sum(fractions, start=Fraction(0, 1))
    normalized = [value / total for value in fractions]
    denominator = reduce(_lcm, (value.denominator for value in normalized), 1)
    counts = [
        value.numerator * (denominator // value.denominator)
        for value in normalized
    ]
    divisor = reduce(math.gcd, counts)
    return [value // divisor for value in counts]


def _smooth_schedule(counts: Sequence[int]) -> list[int]:
    """Interleave integer weights using smooth weighted round robin."""

    total = sum(counts)
    current = [0] * len(counts)
    schedule = []
    for _ in range(total):
        current = [value + weight for value, weight in zip(current, counts)]
        selected = max(range(len(counts)), key=lambda index: current[index])
        current[selected] -= total
        schedule.append(selected)
    return schedule


def _fetch_many(dataset: Any, indices: list[int]) -> list[Any]:
    """Use a child's batched read path when it exposes one."""

    getter = getattr(dataset, "__getitems__", None)
    if getter is not None:
        values = list(getter(indices))
    else:
        values = [dataset[index] for index in indices]
    if len(values) != len(indices):
        raise RuntimeError(
            "Batched dataset read returned the wrong number of samples: "
            f"requested={len(indices)}, returned={len(values)}"
        )
    return values


def _fetch_grouped(
    datasets: Sequence[Any], locations: Sequence[tuple[int, int]]
) -> list[Any]:
    """Batch child indices by dataset, then restore caller order."""

    pending: list[list[tuple[int, int]]] = [[] for _ in datasets]
    for output_index, (dataset_index, local_index) in enumerate(locations):
        pending[dataset_index].append((output_index, local_index))

    sentinel = object()
    output: list[Any] = [sentinel] * len(locations)
    for dataset_index, requests in enumerate(pending):
        if not requests:
            continue
        local_indices = [local_index for _, local_index in requests]
        values = _fetch_many(datasets[dataset_index], local_indices)
        for (output_index, _), value in zip(requests, values):
            output[output_index] = value
    if any(value is sentinel for value in output):
        raise RuntimeError("Grouped batch read did not fill every output position")
    return output


class ScenarioBalancedDataset:
    """Treat every scenario as one equal member inside a logical group.

    Shorter scenario datasets repeat modulo their number of training clips.
    This balancing happens *inside* a factor group and therefore cannot change
    the configured top-level original/speed/door/composition proportions.
    """

    def __init__(self, scenarios: Sequence[Any]) -> None:
        if not scenarios:
            raise ValueError("ScenarioBalancedDataset needs at least one scenario")
        lengths = [len(dataset) for dataset in scenarios]
        if any(length <= 0 for length in lengths):
            raise ValueError(
                "Every training scenario must provide at least one clip; "
                f"got lengths={lengths}"
            )
        self.scenarios = list(scenarios)
        self.lengths = lengths
        self.maximum_length = max(lengths)
        self._length = self.maximum_length * len(self.scenarios)

    def __len__(self) -> int:
        return self._length

    @property
    def column_names(self) -> list[str]:
        return list(getattr(self.scenarios[0], "column_names", []))

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        scenario_index = index % len(self.scenarios)
        round_index = index // len(self.scenarios)
        local_index = round_index % self.lengths[scenario_index]
        return scenario_index, local_index

    def __getitem__(self, index: int):
        scenario_index, local_index = self.locate(index)
        return self.scenarios[scenario_index][local_index]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        locations = [self.locate(int(index)) for index in indices]
        return _fetch_grouped(self.scenarios, locations)


class ConcatenatedDataset:
    """Concatenate child datasets while preserving their batched read paths."""

    def __init__(self, datasets: Sequence[Any]) -> None:
        if not datasets or any(len(dataset) <= 0 for dataset in datasets):
            raise ValueError("ConcatenatedDataset children must be non-empty")
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    @property
    def column_names(self) -> list[str]:
        return list(getattr(self.datasets[0], "column_names", []))

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous = (
            0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        )
        return dataset_index, index - previous

    def __getitem__(self, index: int):
        dataset_index, local_index = self.locate(index)
        return self.datasets[dataset_index][local_index]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        locations = [self.locate(int(index)) for index in indices]
        return _fetch_grouped(self.datasets, locations)


class LogicalGroupDataset:
    """Deterministic two-level mixture with exact per-epoch group weights.

    ``epoch_size`` is an explicit experiment budget, not the sum of child
    lengths.  It must be divisible by the smallest integer weight cycle so a
    complete epoch contains the requested group proportions exactly.  A
    DataLoader may shuffle these global indices without changing those counts.
    """

    def __init__(
        self,
        groups: Mapping[str, Any],
        weights: Mapping[str, float],
        *,
        epoch_size: int,
    ) -> None:
        if not groups:
            raise ValueError("LogicalGroupDataset needs at least one group")
        if set(groups) != set(weights):
            raise ValueError(
                f"Group/weight keys differ: groups={sorted(groups)}, "
                f"weights={sorted(weights)}"
            )
        self.names = list(groups)
        self.groups = [groups[name] for name in self.names]
        lengths = [len(group) for group in self.groups]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"Logical groups must be non-empty: {lengths}")

        self.counts = _weight_counts([weights[name] for name in self.names])
        self.schedule = _smooth_schedule(self.counts)
        self.cycle_size = len(self.schedule)
        self.epoch_size = int(epoch_size)
        if self.epoch_size <= 0 or self.epoch_size % self.cycle_size:
            raise ValueError(
                f"epoch_size={epoch_size} must be a positive multiple of "
                f"weight cycle {self.cycle_size}"
            )

        before = [0] * len(self.names)
        self._occurrence_before: list[int] = []
        for group_index in self.schedule:
            self._occurrence_before.append(before[group_index])
            before[group_index] += 1

    def __len__(self) -> int:
        return self.epoch_size

    @property
    def column_names(self) -> list[str]:
        return list(getattr(self.groups[0], "column_names", []))

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.counts)
        return {
            name: count / total for name, count in zip(self.names, self.counts)
        }

    def locate(self, index: int) -> tuple[str, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, position = divmod(index, self.cycle_size)
        group_index = self.schedule[position]
        occurrence = (
            cycle * self.counts[group_index]
            + self._occurrence_before[position]
        )
        local_index = occurrence % len(self.groups[group_index])
        return self.names[group_index], local_index

    def __getitem__(self, index: int):
        name, local_index = self.locate(index)
        group_index = self.names.index(name)
        return self.groups[group_index][local_index]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        locations = []
        for index in indices:
            name, local_index = self.locate(int(index))
            locations.append((self.names.index(name), local_index))
        return _fetch_grouped(self.groups, locations)

    def epoch_group_counts(self) -> dict[str, int]:
        cycles = self.epoch_size // self.cycle_size
        return {
            name: cycles * count
            for name, count in zip(self.names, self.counts)
        }

    def epoch_group_coverage(self) -> dict[str, dict[str, float | int]]:
        """Report deterministic virtual-slot coverage for one logical epoch."""

        draws = self.epoch_group_counts()
        output = {}
        for name, group in zip(self.names, self.groups):
            available = len(group)
            unique = min(draws[name], available)
            output[name] = {
                "draws": draws[name],
                "available_virtual_slots": available,
                "unique_virtual_slots": unique,
                "unique_virtual_slot_fraction": unique / available,
                "mean_draws_per_virtual_slot": draws[name] / available,
            }
        return output


class RelationBatchSampler:
    """DDP-safe flat batches with equal original and paired exposure.

    Each batch contains 64 unrelated original rows and 32 complete binary
    relations when ``batch_size=128``. Relations are sharded as units, so no
    train shuffle or distributed rank can separate their arms.
    """

    def __init__(
        self,
        singles: Sequence[int],
        relations: Sequence[Sequence[int]],
        *,
        batch_size: int,
        epoch_row_count: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        import torch

        self.singles = torch.as_tensor(singles, dtype=torch.long).clone()
        self.relations = torch.as_tensor(relations, dtype=torch.long).clone()
        self.batch_size = int(batch_size)
        self.epoch_row_count = int(epoch_row_count)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        # Lightning advances ``batch_sampler.sampler`` at every epoch.
        self.sampler = self
        if self.batch_size <= 0 or self.batch_size % 4:
            raise ValueError("relation batch size must be divisible by four")
        if self.relations.ndim != 2 or self.relations.size(1) != 2:
            raise ValueError("COJA relation batches require binary relations")
        if self.singles.numel() < self.batch_size // 2:
            raise ValueError("not enough original rows for one relation batch")
        if self.relations.size(0) < self.batch_size // 4:
            raise ValueError("not enough complete relations for one batch")
        if not 0 <= self.rank < self.world_size or self.world_size <= 0:
            raise ValueError("invalid distributed rank/world size")
        samples_per_rank = math.ceil(self.epoch_row_count / self.world_size)
        self._length = samples_per_rank // self.batch_size
        if self._length <= 0:
            raise ValueError("relation sampler has no complete batch")

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _cyclic_take(order, start: int, count: int):
        import torch

        positions = torch.arange(start, start + count) % order.numel()
        return order[positions]

    def __iter__(self):
        import torch

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        single_order = torch.randperm(
            self.singles.numel(), generator=generator
        )
        relation_order = torch.randperm(
            self.relations.size(0), generator=generator
        )
        singles_per_batch = self.batch_size // 2
        relations_per_batch = self.batch_size // 4
        for local_step in range(self._length):
            global_step = local_step * self.world_size + self.rank
            single_positions = self._cyclic_take(
                single_order,
                global_step * singles_per_batch,
                singles_per_batch,
            )
            relation_positions = self._cyclic_take(
                relation_order,
                global_step * relations_per_batch,
                relations_per_batch,
            )
            batch = self.singles[single_positions].tolist()
            batch.extend(
                self.relations[relation_positions].reshape(-1).tolist()
            )
            yield batch
