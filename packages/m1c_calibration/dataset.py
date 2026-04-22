"""Dataset loading and split helpers for M1c calibration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureBuilder
from .schema import M1CRow, ORDER_LABELS, order_target_from_label


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return rows


def load_order_examples(path: str | Path) -> list[M1CRow]:
    return [
        example
        for example in (M1CRow.from_mapping(row) for row in load_jsonl(path))
        if example.is_order_example
    ]


def split_examples(examples: Sequence[M1CRow]) -> dict[str, list[M1CRow]]:
    grouped = {"train": [], "validation": [], "test": [], "other": []}
    for example in examples:
        split = example.calibration_split.lower()
        if split not in {"train", "validation", "test"}:
            split = "other"
        grouped[split].append(example)
    return grouped


def resolve_splits(
    examples: Sequence[M1CRow],
    *,
    seed: int = 0,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, list[M1CRow]]:
    grouped = split_examples(examples)
    if grouped["train"] and grouped["validation"] and grouped["test"]:
        return grouped

    ordered = list(examples)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    if total <= 2:
        return {"train": ordered, "validation": [], "test": [], "other": []}

    n_train = max(1, int(round(total * train_fraction)))
    n_train = min(n_train, total - 2)
    n_validation = max(1, int(round(total * validation_fraction)))
    if n_train + n_validation >= total:
        n_validation = max(1, total - n_train - 1)

    train = ordered[:n_train]
    validation = ordered[n_train : n_train + n_validation]
    test = ordered[n_train + n_validation :]
    if not test:
        test = validation[-1:]
        validation = validation[:-1]

    return {"train": train, "validation": validation, "test": test, "other": grouped["other"]}


class M1CDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset over order-labeled M1c rows."""

    def __init__(self, examples: Sequence[M1CRow], feature_builder: FeatureBuilder) -> None:
        self.examples = list(examples)
        self.feature_builder = feature_builder
        self.features = feature_builder.transform_many(self.examples)
        labels: list[int] = []
        for example in self.examples:
            target = order_target_from_label(example.label)
            if target is None:
                raise ValueError(f"Example is not an order row: {example.label}")
            labels.append(target)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index]).float(),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

    def rows(self) -> list[M1CRow]:
        return list(self.examples)
