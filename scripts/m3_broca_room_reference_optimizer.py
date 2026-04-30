#!/usr/bin/env python3
"""Offline Bayesian optimization for Broca room-colored self scoring.

Optimizes room-reference scorer knobs over saved embodied Broca self-only
artifacts. This is intentionally offline: no hardware capture, no ASR loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m3_broca_room_reference_analysis import apply_room_filter, fit_room_filter, local_score

DEFAULT_TEXT = "I was speaking, then paused."


def load_artifacts(paths: list[str], template: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        observed = np.load(p / "observed.npy")
        rows.append({
            "path": str(p),
            "label": p.name,
            "summary": summary,
            "observed": observed,
            "expected_shift": int(summary["self_expected_shift_samples"]),
            "template": template,
        })
    return rows


def score_pair(train: dict[str, Any], test: dict[str, Any], *, taps: int, ridge: float, search_ms: float) -> dict[str, float]:
    train_dry_score, train_dry_shift = local_score(
        train["observed"], train["template"], train["expected_shift"], search_ms=search_ms
    )
    h = fit_room_filter(
        train["observed"], train["template"], train_dry_shift, taps=taps, ridge=ridge
    )
    room_ref = apply_room_filter(test["template"], h)
    test_dry_score, _ = local_score(
        test["observed"], test["template"], test["expected_shift"], search_ms=search_ms
    )
    test_room_score, _ = local_score(
        test["observed"], room_ref, test["expected_shift"], search_ms=search_ms
    )
    return {
        "train_dry_score": train_dry_score,
        "test_dry_score": test_dry_score,
        "test_room_score": test_room_score,
        "improvement": test_room_score - test_dry_score,
    }


def evaluate(rows: list[dict[str, Any]], *, taps: int, ridge: float, search_ms: float, threshold: float) -> dict[str, Any]:
    pair_rows = []
    for i, train in enumerate(rows):
        for j, test in enumerate(rows):
            if i == j:
                continue
            pair_rows.append({
                "train": train["label"],
                "test": test["label"],
                **score_pair(train, test, taps=taps, ridge=ridge, search_ms=search_ms),
            })

    improvements = [r["improvement"] for r in pair_rows]
    room_scores = [r["test_room_score"] for r in pair_rows]
    dry_scores = [r["test_dry_score"] for r in pair_rows]
    improved = sum(1 for x in improvements if x > 0)
    detected = sum(1 for x in room_scores if x >= threshold)
    dry_detected = sum(1 for x in dry_scores if x >= threshold)
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0
    p25_improvement = float(np.percentile(improvements, 25)) if improvements else 0.0
    mean_room = float(np.mean(room_scores)) if room_scores else 0.0

    # Loss: reward held-out improvement, robust lower quartile, and detections;
    # penalize configs that only win on average but regress many pairs.
    regression_rate = 1.0 - (improved / len(pair_rows) if pair_rows else 0.0)
    detection_gain = (detected - dry_detected) / len(pair_rows) if pair_rows else 0.0
    loss = -(
        0.45 * mean_improvement
        + 0.25 * p25_improvement
        + 0.20 * mean_room
        + 0.10 * detection_gain
    ) + 0.15 * regression_rate

    return {
        "loss": loss,
        "pairs": len(pair_rows),
        "improved_pairs": improved,
        "mean_improvement": mean_improvement,
        "p25_improvement": p25_improvement,
        "mean_room_score": mean_room,
        "detected_pairs": detected,
        "dry_detected_pairs": dry_detected,
        "pair_rows": pair_rows,
    }


def make_objective(rows: list[dict[str, Any]]):
    def objective(trial: optuna.Trial) -> float:
        taps = trial.suggest_categorical("taps", [32, 64, 128, 256])
        ridge = trial.suggest_float("ridge", 1e-5, 1e-1, log=True)
        search_ms = trial.suggest_float("search_ms", 100.0, 400.0)
        threshold = trial.suggest_float("self_present_threshold", 0.03, 0.30)
        metrics = evaluate(rows, taps=taps, ridge=ridge, search_ms=search_ms, threshold=threshold)
        trial.set_user_attr("metrics", {k: v for k, v in metrics.items() if k != "pair_rows"})
        return metrics["loss"]
    return objective


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Optimize Broca room-reference scorer offline")
    parser.add_argument("artifact_dirs", nargs="+", help="Saved Broca self calibration artifact dirs")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--study-name", default="m3_broca_room_reference")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--output", default="data/m3_broca_room_reference_optimizer_20260430.json")
    args = parser.parse_args()

    print(f"Synthesizing reference text: {args.text!r}")
    template = await synthesize(args.text)
    rows = load_artifacts(args.artifact_dirs, template)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(8, args.n_trials))
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        load_if_exists=bool(args.storage),
    )
    study.optimize(make_objective(rows), n_trials=args.n_trials, n_jobs=1, gc_after_trial=True)

    best = study.best_params
    best_metrics = evaluate(
        rows,
        taps=int(best["taps"]),
        ridge=float(best["ridge"]),
        search_ms=float(best["search_ms"]),
        threshold=float(best["self_present_threshold"]),
    )
    out = {
        "best_params": best,
        "best_value": study.best_value,
        "best_metrics": best_metrics,
        "n_trials": len(study.trials),
        "artifacts": [r["path"] for r in rows],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== BEST ===")
    print(json.dumps({"params": best, "value": study.best_value, **{k: v for k, v in best_metrics.items() if k != "pair_rows"}}, indent=2))
    print(f"Saved: {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
