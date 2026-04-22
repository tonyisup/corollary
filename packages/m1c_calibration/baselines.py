"""Reference thresholds and simple interpretable baselines for M1c."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .metrics import classification_metrics
from .schema import M1CRow, ORDER_LABELS, normalize_label

REFERENCE_PROVISIONAL_ORDER_THRESHOLD_DB = 2.62
REFERENCE_WORLD_EVIDENCE_GLOBAL_THRESHOLD_DB = 5.005
REFERENCE_WORLD_EVIDENCE_THRESHOLDS_BY_STIM: dict[str, float] = {
    "chirp": 5.025,
    "tone": 5.000,
    "vowel_ah": 1.580,
}


def _row_mapping(row: Mapping[str, Any] | M1CRow) -> Mapping[str, Any]:
    if isinstance(row, M1CRow):
        return row.raw
    return row


def _float_value(row: Mapping[str, Any] | M1CRow, key: str, default: float = 0.0) -> float:
    value = _row_mapping(row).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stimulus_value(row: Mapping[str, Any] | M1CRow) -> str:
    value = _row_mapping(row).get("world_stim", "unknown")
    token = str(value or "").strip()
    return token if token else "unknown"


def _true_labels(rows: Sequence[Mapping[str, Any] | M1CRow]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        label = normalize_label(_row_mapping(row).get("expected_label", "unknown"))
        labels.append(label)
    return labels


def _candidate_thresholds(rows: Sequence[Mapping[str, Any] | M1CRow], feature_key: str) -> list[float]:
    values = sorted({_float_value(row, feature_key) for row in rows})
    if not values:
        return [0.0]
    candidates = [values[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-6)
    return candidates


@dataclass(frozen=True)
class FixedThresholdRule:
    """A one-feature threshold that predicts the order label."""

    feature_key: str
    threshold: float
    positive_label: str = "world_before_self"
    negative_label: str = "self_before_world"
    direction: str = "<="

    def predict(self, row: Mapping[str, Any] | M1CRow) -> str:
        value = _float_value(row, self.feature_key)
        if self.direction not in {"<=", ">="}:
            raise ValueError(f"Unsupported direction: {self.direction}")
        keep_positive = value <= self.threshold if self.direction == "<=" else value >= self.threshold
        return self.positive_label if keep_positive else self.negative_label


@dataclass(frozen=True)
class StimulusAwareThresholdRule:
    """A per-stimulus threshold with a fallback for unseen stimuli."""

    feature_key: str
    thresholds_by_stim: dict[str, float]
    fallback_threshold: float
    positive_label: str = "world_before_self"
    negative_label: str = "self_before_world"
    direction: str = "<="

    def predict(self, row: Mapping[str, Any] | M1CRow) -> str:
        stim = _stimulus_value(row)
        threshold = self.thresholds_by_stim.get(stim, self.fallback_threshold)
        value = _float_value(row, self.feature_key)
        if self.direction not in {"<=", ">="}:
            raise ValueError(f"Unsupported direction: {self.direction}")
        keep_positive = value <= threshold if self.direction == "<=" else value >= threshold
        return self.positive_label if keep_positive else self.negative_label


def fit_best_threshold(
    rows: Sequence[Mapping[str, Any] | M1CRow],
    feature_key: str,
    *,
    direction: str = "<=",
) -> FixedThresholdRule:
    if not rows:
        return FixedThresholdRule(feature_key=feature_key, threshold=0.0, direction=direction)

    best_rule = FixedThresholdRule(feature_key=feature_key, threshold=_float_value(rows[0], feature_key), direction=direction)
    best_accuracy = -1.0
    for threshold in _candidate_thresholds(rows, feature_key):
        rule = FixedThresholdRule(feature_key=feature_key, threshold=threshold, direction=direction)
        predictions = [rule.predict(row) for row in rows]
        labels = _true_labels(rows)
        correct = sum(1 for pred, truth in zip(predictions, labels) if pred == truth)
        accuracy = correct / len(rows)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_rule = rule
    return best_rule


def fit_stimulus_aware_rule(
    rows: Sequence[Mapping[str, Any] | M1CRow],
    *,
    feature_key: str = "world_evidence_db",
) -> StimulusAwareThresholdRule:
    grouped: dict[str, list[Mapping[str, Any] | M1CRow]] = {}
    for row in rows:
        grouped.setdefault(_stimulus_value(row), []).append(row)

    thresholds_by_stim: dict[str, float] = {}
    for stim, stim_rows in grouped.items():
        rule = fit_best_threshold(stim_rows, feature_key)
        thresholds_by_stim[stim] = rule.threshold

    fallback_rule = fit_best_threshold(rows, feature_key)
    return StimulusAwareThresholdRule(
        feature_key=feature_key,
        thresholds_by_stim=thresholds_by_stim,
        fallback_threshold=fallback_rule.threshold,
    )


def reference_provisional_rule() -> FixedThresholdRule:
    return FixedThresholdRule(
        feature_key="self_evidence_db",
        threshold=REFERENCE_PROVISIONAL_ORDER_THRESHOLD_DB,
    )


def reference_stimulus_rule() -> StimulusAwareThresholdRule:
    return StimulusAwareThresholdRule(
        feature_key="world_evidence_db",
        thresholds_by_stim=dict(REFERENCE_WORLD_EVIDENCE_THRESHOLDS_BY_STIM),
        fallback_threshold=REFERENCE_WORLD_EVIDENCE_GLOBAL_THRESHOLD_DB,
    )


def evaluate_rule(
    rule: FixedThresholdRule | StimulusAwareThresholdRule,
    rows: Sequence[Mapping[str, Any] | M1CRow],
) -> dict[str, Any]:
    labels = _true_labels(rows)
    predictions = [rule.predict(row) for row in rows]
    metrics = classification_metrics(labels, predictions, labels=ORDER_LABELS)
    metrics["predictions"] = predictions
    return metrics
