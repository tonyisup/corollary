"""Typed row wrappers and label helpers for M1c calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ORDER_LABELS: tuple[str, str] = ("world_before_self", "self_before_world")
ALL_LABELS: tuple[str, ...] = ("world_before_self", "self_before_world", "overlap", "silence", "unknown")
ORDER_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(ORDER_LABELS)}


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if label in {"world_before_self", "self_before_world", "overlap", "silence", "unknown"}:
        return label
    return "unknown"


def order_target_from_label(label: Any) -> int | None:
    normalized = normalize_label(label)
    if normalized not in ORDER_LABEL_TO_INDEX:
        return None
    return ORDER_LABEL_TO_INDEX[normalized]


def label_group(label: Any) -> str:
    normalized = normalize_label(label)
    if normalized in ORDER_LABELS:
        return "order"
    if normalized == "overlap":
        return "overlap"
    if normalized == "silence":
        return "silence"
    return "unknown"


@dataclass(frozen=True)
class M1CRow:
    """A normalized M1c record with a small typed surface."""

    raw: dict[str, Any]
    label: str
    calibration_split: str
    session_label: str
    setup_label: str
    mode: str
    self_stim: str
    world_stim: str
    seed: int | None = None
    experiment_block: str | None = None

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        label_key: str = "expected_label",
        split_key: str = "calibration_split",
    ) -> "M1CRow":
        raw = dict(row)
        label = normalize_label(raw.get(label_key, "unknown"))
        split = str(raw.get(split_key, "train") or "train")
        session_label = str(raw.get("session_label", "") or "")
        setup_label = str(raw.get("setup_label", "") or "")
        mode = str(raw.get("mode", "") or "")
        self_stim = str(raw.get("self_stim", "") or "")
        world_stim = str(raw.get("world_stim", "") or "")
        experiment_block = raw.get("experiment_block")
        if experiment_block is not None:
            experiment_block = str(experiment_block)

        seed = raw.get("seed")
        try:
            seed_int = int(seed) if seed not in (None, "") else None
        except (TypeError, ValueError):
            seed_int = None

        return cls(
            raw=raw,
            label=label,
            calibration_split=split,
            session_label=session_label,
            setup_label=setup_label,
            mode=mode,
            self_stim=self_stim,
            world_stim=world_stim,
            seed=seed_int,
            experiment_block=experiment_block,
        )

    @property
    def is_order_example(self) -> bool:
        return self.label in ORDER_LABELS

    @property
    def target_index(self) -> int | None:
        return order_target_from_label(self.label)

    @property
    def group(self) -> str:
        return label_group(self.label)
