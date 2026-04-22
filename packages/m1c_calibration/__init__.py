"""M1c calibration utilities for Corollary.

This package turns labeled M1c runs into features, baselines, and a small
learned calibration model that can be compared against the existing rule-based
thresholds.
"""

from .baselines import (
    REFERENCE_PROVISIONAL_ORDER_THRESHOLD_DB,
    REFERENCE_WORLD_EVIDENCE_GLOBAL_THRESHOLD_DB,
    REFERENCE_WORLD_EVIDENCE_THRESHOLDS_BY_STIM,
    FixedThresholdRule,
    StimulusAwareThresholdRule,
    evaluate_rule,
    fit_best_threshold,
    fit_stimulus_aware_rule,
    reference_provisional_rule,
    reference_stimulus_rule,
)
from .dataset import M1CDataset, load_jsonl, load_order_examples, resolve_splits
from .features import DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERIC_FEATURES, FeatureBuilder
from .metrics import classification_metrics, grouped_metrics
from .model import CalibrationNet, CalibrationNetConfig
from .schema import M1CRow, ORDER_LABELS, normalize_label, order_target_from_label

__all__ = [
    "CalibrationNet",
    "CalibrationNetConfig",
    "DEFAULT_CATEGORICAL_FEATURES",
    "DEFAULT_NUMERIC_FEATURES",
    "FeatureBuilder",
    "FixedThresholdRule",
    "M1CDataset",
    "M1CRow",
    "ORDER_LABELS",
    "REFERENCE_PROVISIONAL_ORDER_THRESHOLD_DB",
    "REFERENCE_WORLD_EVIDENCE_GLOBAL_THRESHOLD_DB",
    "REFERENCE_WORLD_EVIDENCE_THRESHOLDS_BY_STIM",
    "StimulusAwareThresholdRule",
    "classification_metrics",
    "evaluate_rule",
    "fit_best_threshold",
    "fit_stimulus_aware_rule",
    "grouped_metrics",
    "load_jsonl",
    "load_order_examples",
    "normalize_label",
    "order_target_from_label",
    "reference_provisional_rule",
    "reference_stimulus_rule",
    "resolve_splits",
]
