#!/usr/bin/env python3
"""Bayesian hyperparameter optimization for M2 audio gating parameters.

Uses Optuna TPE to search:
- onset_latency_ms
- causal_tolerance_ms
- similarity_threshold
- onset_method (energy vs spectral flux)
- vowel gain, babble gain

Each trial runs a pair of M2 trials (vowel + babble) and computes a combined loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import optuna

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

EXPECTED_TEXT = "this is a test of world speech conservation"

VOWEL_STIM = "vowel_ah_var_nearfield"
BABBLE_STIM = "babble_ba_repeat_short_tight_var"
DEFAULT_INPUT_DEVICE = 3
DEFAULT_OUTPUT_DEVICE = 5
DEFAULT_NOISE_DEVICE = 4


def run_single_trial_for_params(
    params: dict[str, Any],
    stim: str,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one M2 harness call with optimized parameters."""
    result = run_m2_trial(
        self_stim=stim,
        world_speech_file="data/world_speech_sample.npy",
        input_device=DEFAULT_INPUT_DEVICE,
        output_device=DEFAULT_OUTPUT_DEVICE,
        noise_device=DEFAULT_NOISE_DEVICE,
        seed=0,
        expected_text=EXPECTED_TEXT,
        both_policy="family",
        self_duration_ms=900,
        self_amplitude=1.0,
        speech_override="vad",
        similarity_threshold=params["similarity_threshold"],
        onset_latency_ms=int(params["onset_latency_ms"]),
        onset_method=params["onset_method"],
        causal_tolerance_ms=int(params["causal_tolerance_ms"]),
    )
    if save_dir:
        save_trial_artifacts(str(save_dir), result)

    return result


def compute_loss(r1: dict[str, Any], r2: dict[str, Any], params: dict[str, Any] | None = None) -> float:
    """Combined loss from vowel + babble trial results.

    Since WER on one sentence is too coarse to discriminate, we use richer signals:
    - attribution quality: fraction of frames correctly labeled for known self+world mix
    - self-match score: how well system identifies self (higher=better attribution)
    - speech preservation: gated WER should not exceed raw WER
    - parameter simplicity penalty: prefer conservative settings near known-good baseline
    """
    # Attribution signal: self+world stimulus should produce some BOTH or WORLD frames
    # and minimal leakage (self_frames should be nonzero when self stimulus plays)
    self_frames1 = r1.get("self_frames", 0)
    self_frames2 = r2.get("self_frames", 0)
    world_frames1 = r1.get("world_frames", 0)
    world_frames2 = r2.get("world_frames", 0)
    both_frames1 = r1.get("both_frames", 0)
    both_frames2 = r2.get("both_frames", 0)

    # Self-match signal: how well did we detect self when it was present
    self_match1 = r1.get("self_score", 0.0)
    self_match2 = r2.get("self_score", 0.0)

    # WER penalty: gated should not be worse than raw
    m1, m2 = r1.get("metrics"), r2.get("metrics")
    wer_penalty = 0.0
    if m1 and m2:
        if m1["gated_wer"] > m1["raw_wer"]:
            wer_penalty += (m1["gated_wer"] - m1["raw_wer"]) * 2.0
        if m2["gated_wer"] > m2["raw_wer"]:
            wer_penalty += (m2["gated_wer"] - m2["raw_wer"]) * 2.0

    # Primary loss: maximize self-identification + world preservation
    # Low loss = good self detection AND world preservation
    attribution_quality = (self_match1 + self_match2) / 2.0
    attribution_component = 1.0 - attribution_quality  # we want to minimize this

    # Both frames are expected overlap: should be > 0 for mixed self+world stimulus
    expected_both = 15  # roughly what we saw before
    both_penalty = max(0, expected_both - (both_frames1 + both_frames2) / 2.0) / expected_both

    # Penalize extreme parameters (prefer calibrated baseline region)
    baseline_latency = 400.0
    baseline_tolerance = 75.0
    baseline_threshold = 0.20
    param_penalty = (
        abs(params.get("onset_latency_ms", baseline_latency) - baseline_latency) / 200.0 +
        abs(params.get("causal_tolerance_ms", baseline_tolerance) - baseline_tolerance) / 25.0 +
        abs(params.get("similarity_threshold", baseline_threshold) - baseline_threshold) / 0.15
    ) / 3.0

    loss = 0.4 * attribution_component + 0.3 * both_penalty + 0.2 * wer_penalty + 0.1 * param_penalty
    return loss


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: return loss for this parameter set."""
    params = {
        "onset_latency_ms": trial.suggest_float("onset_latency_ms", 200.0, 600.0),
        "causal_tolerance_ms": trial.suggest_float("causal_tolerance_ms", 50.0, 100.0),
        "similarity_threshold": trial.suggest_float("similarity_threshold", 0.10, 0.35),
        "onset_method": trial.suggest_categorical("onset_method", ["energy", "spectral_flux"]),
        "vowel_gain": trial.suggest_float("vowel_gain", 0.3, 0.8),
        "babble_gain": trial.suggest_float("babble_gain", 0.8, 1.0),
    }
    result1 = run_single_trial_for_params(params, VOWEL_STIM)
    result2 = run_single_trial_for_params(params, BABBLE_STIM)
    loss = compute_loss(result1, result2, params)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization of M2 audio gating parameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--save-dir", default=None, help="Save best trial artifacts here")
    parser.add_argument("--study-name", default="m2_bayesian", help="Optuna study name for persistence")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (sqlite: for example)")
    args = parser.parse_args()

    if args.storage:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=args.storage,
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        )

    print(f"Starting Bayesian optimization: {args.n_trials} trials")
    param_names = ["onset_latency_ms", "causal_tolerance_ms", "similarity_threshold", "onset_method", "vowel_gain", "babble_gain"]
    print(f"Parameters: {', '.join(param_names)}")
    print("Loss = attribution_component + both_penalty + wer_penalty + param_penalty")
    print(f"Storage: {args.storage or 'memory'}")
    print()

    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted")

    print("\n=== Best Parameters ===")
    print(json.dumps(study.best_params, indent=2))
    print(f"Best loss: {study.best_value:.4f}")

    best_trial = study.best_trial
    artifacts_path = None
    if args.save_dir:
        best_params_dict = best_trial.params
        result1 = run_single_trial_for_params(best_params_dict, VOWEL_STIM, Path(args.save_dir) / "vowel_best")
        result2 = run_single_trial_for_params(best_params_dict, BABBLE_STIM, Path(args.save_dir) / "babble_best")
        artifacts_path = args.save_dir

    output = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_completed": len(study.trials),
        "study_direction": study.direction.name,
        "artifacts_path": artifacts_path,
    }
    print(output)


if __name__ == "__main__":
    main()
