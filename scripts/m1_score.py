#!/usr/bin/env python3
"""Stage-aware scoring for Corollary M1 experiments."""

from __future__ import annotations

from typing import Any


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_m1a(metrics: dict[str, Any], guardrails: dict[str, Any] | None = None) -> dict[str, Any]:
    guardrails = guardrails or {}
    self_f1 = _safe_float(metrics, "self_f1")
    false_self_rate = _safe_float(metrics, "false_self_rate")
    replay_false_self_rate = _safe_float(metrics, "replay_false_self_rate")
    onset_error_ms = _safe_float(metrics, "onset_error_ms")
    causal_self_score_mean = _safe_float(metrics, "causal_self_score_mean")
    combined_negative_score_mean = _safe_float(metrics, "combined_negative_score_mean")

    max_false_self = float(guardrails.get("max_false_self_rate", 0.05))
    max_replay_false_self_rate = float(guardrails.get("max_replay_false_self_rate", 0.10))
    max_onset_error_ms = float(guardrails.get("max_onset_error_ms", 50.0))
    min_causal_self_score_mean = float(guardrails.get("min_causal_self_score_mean", 0.45))
    max_combined_negative_score_mean = float(guardrails.get("max_combined_negative_score_mean", 0.45))

    score = self_f1
    score *= _clamp01(1.0 - (false_self_rate / max(max_false_self, 1e-9)) * 0.3)
    score *= _clamp01(1.0 - (replay_false_self_rate / max(max_replay_false_self_rate, 1e-9)) * 0.25)
    score *= _clamp01(1.0 - (onset_error_ms / max(max_onset_error_ms, 1e-9)) * 0.2)
    score *= _clamp01(causal_self_score_mean / max(min_causal_self_score_mean, 1e-9))
    score *= _clamp01(1.0 - (combined_negative_score_mean / max(max_combined_negative_score_mean, 1e-9)) * 0.4)

    failures: list[str] = []
    if false_self_rate > max_false_self:
        failures.append("false_self_rate")
    if replay_false_self_rate > max_replay_false_self_rate:
        failures.append("replay_false_self_rate")
    if onset_error_ms > max_onset_error_ms:
        failures.append("onset_error_ms")
    if causal_self_score_mean < min_causal_self_score_mean:
        failures.append("causal_self_score_mean")
    if combined_negative_score_mean > max_combined_negative_score_mean:
        failures.append("combined_negative_score_mean")

    return {
        "score": round(_clamp01(score), 6),
        "primary_metric": "self_f1",
        "guardrails_passed": not failures,
        "guardrail_failures": failures,
    }


def score_m1b(metrics: dict[str, Any], guardrails: dict[str, Any] | None = None) -> dict[str, Any]:
    guardrails = guardrails or {}
    attribution_gain = _safe_float(metrics, "attribution_gain")
    convergence_slope = _safe_float(metrics, "convergence_slope")
    prediction_error_final_db = _safe_float(metrics, "prediction_error_final_db")

    min_gain = float(guardrails.get("min_attribution_gain", 0.0))
    min_slope = float(guardrails.get("min_convergence_slope", 0.0))
    max_prediction_error_final_db = float(guardrails.get("max_prediction_error_final_db", -5.0))

    gain_term = _clamp01((attribution_gain + 0.05) / 0.25)
    slope_term = _clamp01((convergence_slope + 0.5) / 1.5)
    error_term = _clamp01((-prediction_error_final_db) / max(-max_prediction_error_final_db, 1e-9))
    score = 0.5 * gain_term + 0.3 * slope_term + 0.2 * error_term

    failures: list[str] = []
    if attribution_gain < min_gain:
        failures.append("attribution_gain")
    if convergence_slope < min_slope:
        failures.append("convergence_slope")
    if prediction_error_final_db > max_prediction_error_final_db:
        failures.append("prediction_error_final_db")

    return {
        "score": round(_clamp01(score), 6),
        "primary_metric": "attribution_gain",
        "guardrails_passed": not failures,
        "guardrail_failures": failures,
    }


def score_m1c(metrics: dict[str, Any], guardrails: dict[str, Any] | None = None) -> dict[str, Any]:
    guardrails = guardrails or {}
    overlap_f1 = _safe_float(metrics, "overlap_f1")
    world_to_self_confusion = _safe_float(metrics, "world_to_self_confusion")
    replay_false_self_rate = _safe_float(metrics, "replay_false_self_rate")

    max_world_to_self_confusion = float(guardrails.get("max_world_to_self_confusion", 0.10))
    max_replay_false_self_rate = float(guardrails.get("max_replay_false_self_rate", 0.05))

    score = overlap_f1
    score *= _clamp01(1.0 - (world_to_self_confusion / max(max_world_to_self_confusion, 1e-9)) * 0.5)
    score *= _clamp01(1.0 - (replay_false_self_rate / max(max_replay_false_self_rate, 1e-9)) * 0.5)

    failures: list[str] = []
    if world_to_self_confusion > max_world_to_self_confusion:
        failures.append("world_to_self_confusion")
    if replay_false_self_rate > max_replay_false_self_rate:
        failures.append("replay_false_self_rate")

    return {
        "score": round(_clamp01(score), 6),
        "primary_metric": "overlap_f1",
        "guardrails_passed": not failures,
        "guardrail_failures": failures,
    }


def score_stage(stage: str, metrics: dict[str, Any], guardrails: dict[str, Any] | None = None) -> dict[str, Any]:
    stage = stage.upper()
    if stage == "M1A":
        return score_m1a(metrics, guardrails)
    if stage == "M1B":
        return score_m1b(metrics, guardrails)
    if stage == "M1C":
        return score_m1c(metrics, guardrails)
    raise ValueError(f"Unsupported stage: {stage}")
