#!/usr/bin/env python3
"""Bayesian hyperparameter optimization for M2 audio gating parameters.

Expanded version supporting:
- multi-seed averaging (avoid overfitting to one timing alignment)
- self_amplitude parameter (affects self-match confidence)
- self_duration_ms parameter (affects attribution window)

Each trial runs M2 trials across N seeds and averages the loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _run_for_seed(
    params: dict[str, Any],
    stim: str,
    seed: int,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one M2 harness call for a specific seed."""
    result = run_m2_trial(
        self_stim=stim,
        world_speech_file="data/world_speech_sample.npy",
        input_device=DEFAULT_INPUT_DEVICE,
        output_device=DEFAULT_OUTPUT_DEVICE,
        noise_device=DEFAULT_NOISE_DEVICE,
        seed=seed,
        expected_text=EXPECTED_TEXT,
        both_policy="family",
        family_gain_overrides={
            "vowel_ah_var_nearfield": params["vowel_gain"],
            "repeated_babble": params["babble_gain"],
        },
        self_duration_ms=int(params["self_duration_ms"]),
        self_amplitude=params["self_amplitude"],
        speech_override="vad",
        similarity_threshold=params["similarity_threshold"],
        onset_latency_ms=int(params["onset_latency_ms"]),
        onset_method=params["onset_method"],
        causal_tolerance_ms=int(params["causal_tolerance_ms"]),
    )
    if save_dir:
        save_trial_artifacts(str(save_dir), result)
    return result


def run_multi_seed(
    params: dict[str, Any],
    seeds: list[int],
    stim: str,
    save_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run M2 harness across multiple seeds and return all results."""
    results = []
    for seed in seeds:
        d = (save_dir / f"seed{seed}") if save_dir else None
        results.append(_run_for_seed(params, stim, seed, save_dir=d))
    return results


def compute_loss(results: list[dict[str, Any]], params: dict[str, Any] | None = None) -> float:
    """Combined loss averaged across all seeds and both stimuli.

    Loss = 0.4*attribution_component + 0.3*both_penalty + 0.2*wer_penalty + 0.1*param_penalty
    """
    total_loss = 0.0
    for r in results:
        self_match = r.get("self_score", 0.0)
        both = r.get("both_frames", 0)
        self_f = r.get("self_frames", 0)
        world_f = r.get("world_frames", 0)

        m = r.get("metrics")
        wer_penalty = 0.0
        if m:
            if m["gated_wer"] > m["raw_wer"]:
                wer_penalty = (m["gated_wer"] - m["raw_wer"]) * 2.0

        attribution_quality = self_match
        attribution_component = 1.0 - attribution_quality

        expected_both = 15
        both_penalty = max(0, expected_both - both) / expected_both if expected_both > 0 else 0.0

        baseline_latency = 400.0
        baseline_tolerance = 75.0
        baseline_threshold = 0.20
        param_penalty = 0.0
        if params:
            param_penalty = (
                abs(params.get("onset_latency_ms", baseline_latency) - baseline_latency) / 200.0 +
                abs(params.get("causal_tolerance_ms", baseline_tolerance) - baseline_tolerance) / 25.0 +
                abs(params.get("similarity_threshold", baseline_threshold) - baseline_threshold) / 0.15
            ) / 3.0

        loss = 0.4 * attribution_component + 0.3 * both_penalty + 0.2 * wer_penalty + 0.1 * param_penalty
        total_loss += loss

    return total_loss / len(results) if results else float("inf")


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: return multi-seed averaged loss."""
    seeds = trial.study.user_attrs.get("seeds", [0])
    save_dir = trial.study.user_attrs.get("save_dir", None)

    params = {
        "onset_latency_ms": trial.suggest_float("onset_latency_ms", 200.0, 600.0),
        "causal_tolerance_ms": trial.suggest_float("causal_tolerance_ms", 50.0, 100.0),
        "similarity_threshold": trial.suggest_float("similarity_threshold", 0.10, 0.35),
        "onset_method": trial.suggest_categorical("onset_method", ["energy", "spectral_flux"]),
        "vowel_gain": trial.suggest_float("vowel_gain", 0.3, 0.8),
        "babble_gain": trial.suggest_float("babble_gain", 0.8, 1.0),
        "self_amplitude": trial.suggest_float("self_amplitude", 0.5, 1.0),
        "self_duration_ms": trial.suggest_float("self_duration_ms", 500.0, 1500.0),
    }

    d = Path(save_dir) / f"trial{trial.number}" if save_dir else None

    vowel_results = run_multi_seed(params, seeds, VOWEL_STIM, save_dir=d / "vowel" if d else None)
    babble_results = run_multi_seed(params, seeds, BABBLE_STIM, save_dir=d / "babble" if d else None)

    all_results = vowel_results + babble_results
    loss = compute_loss(all_results, params)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization of M2 audio gating (multi-seed)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--save-dir", default=None, help="Save best trial artifacts")
    parser.add_argument("--study-name", default="m2_bayesian_v2", help="Optuna study name for persistence")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (sqlite: for example)")
    parser.add_argument("--seeds", default="0,1,2", help="Seeds to average over, comma-separated (e.g. 0,1,2)")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed details during trials")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",")]

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

    # Pass seeds to objective via study user_attrs
    study.set_user_attr("seeds", seeds)
    if args.save_dir:
        study.set_user_attr("save_dir", args.save_dir)

    n_per_trial = len(seeds) * 2  # vowel + babble
    total_runs = args.n_trials * n_per_trial
    print(f"Starting Bayesian optimization: {args.n_trials} trials")
    print(f"Seeds: {seeds} ({len(seeds)} per stimulus)")
    print(f"Runs per trial: {n_per_trial} ({n_per_trial} harness calls)")
    print(f"Estimated runs total: {total_runs} (~{total_runs * 6}s)")
    print(f"Parameters: onset_latency_ms, causal_tolerance_ms, similarity_threshold, onset_method, vowel_gain, babble_gain, self_amplitude, self_duration_ms")
    print("Loss = 0.4*attribution + 0.3*both + 0.2*wer + 0.1*param")
    print()

    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted")

    print("\n=== Best Parameters ===")
    print(json.dumps(study.best_params, indent=2))
    print(f"Best loss: {study.best_value:.4f}")
    print(f"Trials completed: {len(study.trials)}")

    best_trial = study.best_trial
    artifacts_path = None
    if args.save_dir:
        best_params_dict = best_trial.params
        vowel_best = run_multi_seed(best_params_dict, seeds, VOWEL_STIM, Path(args.save_dir) / "vowel_best")
        babble_best = run_multi_seed(best_params_dict, seeds, BABBLE_STIM, Path(args.save_dir) / "babble_best")
        artifacts_path = args.save_dir

    output = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_completed": len(study.trials),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "study_direction": study.direction.name,
        "artifacts_path": artifacts_path,
    }
    print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
