"""Metrics for M1c calibration models and rules."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import M1CRow, ORDER_LABELS, normalize_label


def _row_mapping(row: Mapping[str, Any] | M1CRow) -> Mapping[str, Any]:
    if isinstance(row, M1CRow):
        return row.raw
    return row


def _safe_label(value: Any) -> str:
    return normalize_label(value)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = ORDER_LABELS,
) -> dict[str, Any]:
    labels = tuple(labels)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    total = len(y_true)

    for truth, pred in zip(y_true, y_pred):
        truth_idx = label_to_idx.get(_safe_label(truth))
        pred_idx = label_to_idx.get(_safe_label(pred))
        if truth_idx is None or pred_idx is None:
            continue
        confusion[truth_idx, pred_idx] += 1

    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_label: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for idx, label in enumerate(labels):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            recalls.append(recall)
            f1s.append(f1)

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "count": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "correct": correct,
        "confusion_matrix": confusion.tolist(),
        "per_label": per_label,
    }


def grouped_metrics(
    rows: Sequence[Mapping[str, Any] | M1CRow],
    predictions: Sequence[str],
    *,
    group_key: str,
    label_key: str = "expected_label",
    labels: Sequence[str] = ORDER_LABELS,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")

    grouped_true: dict[str, list[str]] = defaultdict(list)
    grouped_pred: dict[str, list[str]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        src = _row_mapping(row)
        group = str(src.get(group_key, "unknown") or "unknown")
        grouped_true[group].append(_safe_label(src.get(label_key, "unknown")))
        grouped_pred[group].append(_safe_label(pred))

    return {
        group: classification_metrics(true_labels, grouped_pred[group], labels=labels)
        for group, true_labels in grouped_true.items()
    }
