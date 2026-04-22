"""Feature extraction for M1c calibration rows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import M1CRow

DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = (
    "amplitude",
    "duration_ms",
    "log_duration_ms",
    "gap_s",
    "lead_in_s",
    "tail_s",
    "overlap_ratio",
    "self_score",
    "world_score",
    "score_gap",
    "self_evidence_db",
    "world_evidence_db",
    "evidence_gap_db",
    "baseline_rms_db",
    "rms_db",
    "self_rms_db",
    "world_rms_db",
    "rms_gap_db",
    "expected_self_rms_db",
    "expected_world_rms_db",
    "expected_rms_gap_db",
    "self_error_ms",
    "self_shift_error_ms",
    "world_shift_error_ms",
    "self_shift_s",
    "world_shift_s",
    "shift_gap_s",
    "self_expected_shift_s",
    "world_expected_shift_s",
    "shift_error_gap_s",
    "self_vs_world_energy_db",
    "expected_self_delta_db",
    "expected_world_delta_db",
)

DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = ("mode", "self_stim", "world_stim")


def _row_mapping(row: Mapping[str, Any] | M1CRow) -> Mapping[str, Any]:
    if isinstance(row, M1CRow):
        return row.raw
    return row


def _safe_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(row: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = row.get(key, default)
    text = str(value or "").strip()
    return text if text else default


def _numeric_feature_map(row: Mapping[str, Any] | M1CRow) -> dict[str, float]:
    src = _row_mapping(row)
    sample_rate = max(_safe_float(src, "sample_rate", 16000.0), 1.0)

    def samples_to_ms(samples: float) -> float:
        return float(samples) * 1000.0 / sample_rate

    def samples_to_s(samples: float) -> float:
        return float(samples) / sample_rate

    duration_ms = _safe_float(src, "duration_ms")
    self_score = _safe_float(src, "self_score")
    world_score = _safe_float(src, "world_score")
    self_evidence_db = _safe_float(src, "self_evidence_db")
    world_evidence_db = _safe_float(src, "world_evidence_db")
    baseline_rms_db = _safe_float(src, "baseline_rms_db")
    rms_db = _safe_float(src, "rms_db")
    self_rms_db = _safe_float(src, "self_rms_db")
    world_rms_db = _safe_float(src, "world_rms_db")
    expected_self_rms_db = _safe_float(src, "expected_self_rms_db")
    expected_world_rms_db = _safe_float(src, "expected_world_rms_db")
    self_error_ms = _safe_float(src, "self_error_ms")
    self_shift_error_samples = _safe_float(src, "self_shift_error_samples")
    world_shift_error_samples = _safe_float(src, "world_shift_error_samples")
    self_shift_samples = _safe_float(src, "self_shift_samples")
    world_shift_samples = _safe_float(src, "world_shift_samples")
    self_expected_shift_samples = _safe_float(src, "self_expected_shift_samples")
    world_expected_shift_samples = _safe_float(src, "world_expected_shift_samples")

    return {
        "amplitude": _safe_float(src, "amplitude"),
        "duration_ms": duration_ms,
        "log_duration_ms": math.log1p(max(duration_ms, 0.0)),
        "gap_s": _safe_float(src, "gap_s"),
        "lead_in_s": _safe_float(src, "lead_in_s"),
        "tail_s": _safe_float(src, "tail_s"),
        "overlap_ratio": _safe_float(src, "overlap_ratio"),
        "self_score": self_score,
        "world_score": world_score,
        "score_gap": self_score - world_score,
        "self_evidence_db": self_evidence_db,
        "world_evidence_db": world_evidence_db,
        "evidence_gap_db": self_evidence_db - world_evidence_db,
        "baseline_rms_db": baseline_rms_db,
        "rms_db": rms_db,
        "self_rms_db": self_rms_db,
        "world_rms_db": world_rms_db,
        "rms_gap_db": self_rms_db - world_rms_db,
        "expected_self_rms_db": expected_self_rms_db,
        "expected_world_rms_db": expected_world_rms_db,
        "expected_rms_gap_db": expected_self_rms_db - expected_world_rms_db,
        "self_error_ms": self_error_ms,
        "self_shift_error_ms": samples_to_ms(self_shift_error_samples),
        "world_shift_error_ms": samples_to_ms(world_shift_error_samples),
        "self_shift_s": samples_to_s(self_shift_samples),
        "world_shift_s": samples_to_s(world_shift_samples),
        "shift_gap_s": samples_to_s(self_shift_samples - world_shift_samples),
        "self_expected_shift_s": samples_to_s(self_expected_shift_samples),
        "world_expected_shift_s": samples_to_s(world_expected_shift_samples),
        "shift_error_gap_s": samples_to_s(self_shift_error_samples - world_shift_error_samples),
        "self_vs_world_energy_db": self_rms_db - world_rms_db,
        "expected_self_delta_db": self_rms_db - expected_self_rms_db,
        "expected_world_delta_db": world_rms_db - expected_world_rms_db,
    }


@dataclass
class FeatureBuilder:
    """Standardize numeric M1c features and one-hot encode categorical fields."""

    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
    numeric_mean: dict[str, float] = field(default_factory=dict)
    numeric_std: dict[str, float] = field(default_factory=dict)
    categorical_vocab: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any] | M1CRow],
        *,
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
    ) -> "FeatureBuilder":
        numeric_features = tuple(numeric_features or DEFAULT_NUMERIC_FEATURES)
        categorical_features = tuple(categorical_features or DEFAULT_CATEGORICAL_FEATURES)
        if not rows:
            raise ValueError("FeatureBuilder.fit requires at least one row")

        numeric_mean: dict[str, float] = {}
        numeric_std: dict[str, float] = {}
        for feature_name in numeric_features:
            values = np.array([
                float(_numeric_feature_map(row).get(feature_name, 0.0))
                for row in rows
            ], dtype=np.float32)
            mean = float(np.mean(values)) if len(values) else 0.0
            std = float(np.std(values)) if len(values) else 1.0
            numeric_mean[feature_name] = mean
            numeric_std[feature_name] = std if std > 1e-8 else 1.0

        categorical_vocab: dict[str, tuple[str, ...]] = {}
        for feature_name in categorical_features:
            values = sorted({
                _safe_str(_row_mapping(row), feature_name, "unknown")
                for row in rows
            })
            if not values:
                values = ["unknown"]
            if "unknown" not in values:
                values = ["unknown", *values]
            categorical_vocab[feature_name] = tuple(values)

        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_mean=numeric_mean,
            numeric_std=numeric_std,
            categorical_vocab=categorical_vocab,
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_features)
        for feature_name in self.categorical_features:
            vocab = self.categorical_vocab.get(feature_name, ("unknown",))
            names.extend(f"{feature_name}={value}" for value in vocab)
        return names

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, row: Mapping[str, Any] | M1CRow) -> np.ndarray:
        numeric_map = _numeric_feature_map(row)
        values: list[float] = []
        for feature_name in self.numeric_features:
            raw_value = float(numeric_map.get(feature_name, 0.0))
            mean = self.numeric_mean.get(feature_name, 0.0)
            std = self.numeric_std.get(feature_name, 1.0) or 1.0
            values.append((raw_value - mean) / std)

        src = _row_mapping(row)
        for feature_name in self.categorical_features:
            vocab = self.categorical_vocab.get(feature_name, ("unknown",))
            token = _safe_str(src, feature_name, "unknown")
            if token not in vocab:
                token = "unknown"
            values.extend(1.0 if token == candidate else 0.0 for candidate in vocab)

        return np.asarray(values, dtype=np.float32)

    def transform_many(self, rows: Sequence[Mapping[str, Any] | M1CRow]) -> np.ndarray:
        if not rows:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack([self.transform(row) for row in rows], axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "numeric_mean": dict(self.numeric_mean),
            "numeric_std": dict(self.numeric_std),
            "categorical_vocab": {key: list(values) for key, values in self.categorical_vocab.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureBuilder":
        return cls(
            numeric_features=tuple(payload.get("numeric_features", DEFAULT_NUMERIC_FEATURES)),
            categorical_features=tuple(payload.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES)),
            numeric_mean={str(key): float(value) for key, value in dict(payload.get("numeric_mean", {})).items()},
            numeric_std={str(key): float(value) for key, value in dict(payload.get("numeric_std", {})).items()},
            categorical_vocab={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(payload.get("categorical_vocab", {})).items()
            },
        )
