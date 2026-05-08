#!/usr/bin/env python3
"""Causal room filter fitting: fit on one capture, score a different one.

This closes the oracle → causal gap by ensuring the filter is fitted on
different data than it scores.  In production, the filter would be fitted
at calibration time and applied at runtime to unseen captures.
"""

import asyncio
import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m3_broca_room_reference_analysis import fit_room_filter, apply_room_filter, local_score, dry_template_track

async def main():
    # Synthesize the template (same text used for all calibrations)
    template = await synthesize("I was speaking, then paused.")
    print(f"Template: {len(template)} samples")

    # Self-calibration trials: fit filter on one, score another
    cal_dir = Path("data/m3_broca_self_calibration_20260430_quick")
    trials = sorted([d for d in cal_dir.iterdir() if d.is_dir() and d.name.startswith("trial")])

    print(f"\n=== Causal filter fitting ({len(trials)} trials) ===")
    print("Fitting filter on each trial and scoring all others...\n")

    results = []
    for fit_trial in trials:
        fit_summary = json.loads((fit_trial / "summary.json").read_text())
        fit_observed = np.load(fit_trial / "observed.npy")
        fit_shift = int(fit_summary["self_expected_shift_samples"])

        # Fit room filter on this trial
        h = fit_room_filter(fit_observed, template, fit_shift, taps=64, ridge=0.000112)
        print(f"Filter fitted on {fit_trial.name}: sum={h.sum():.3f} max={h.max():.3f}")

        # Score all OTHER trials with this filter
        for score_trial in trials:
            if score_trial == fit_trial:
                continue
            score_summary = json.loads((score_trial / "summary.json").read_text())
            score_observed = np.load(score_trial / "observed.npy")
            score_shift = int(score_summary["self_expected_shift_samples"])

            # Apply room filter to template
            room_ref = apply_room_filter(template, h)

            # Score using local correlation around expected shift
            score, best_shift = local_score(score_observed, room_ref, score_shift, search_ms=256.4)

            # Also compute dry score for comparison
            dry_ref = dry_template_track(template, len(score_observed), score_shift)
            dry_score, _ = local_score(score_observed, dry_ref, score_shift, search_ms=256.4)

            results.append({
                "fit_trial": fit_trial.name,
                "score_trial": score_trial.name,
                "room_score": float(score),
                "dry_score": float(dry_score),
                "shift_error_ms": float(abs(best_shift - score_shift) * 1000 / 16000),
            })
            print(f"  → {score_trial.name}: room={score:.4f} dry={dry_score:.4f} shift_err={abs(best_shift - score_shift) * 1000 / 16000:.1f}ms")

    # Summary: how well does causal fitting work?
    print("\n=== Summary ===")
    room_scores = [r["room_score"] for r in results]
    dry_scores = [r["dry_score"] for r in results]
    print(f"Room scores: min={min(room_scores):.4f} max={max(room_scores):.4f} mean={np.mean(room_scores):.4f}")
    print(f"Dry scores:  min={min(dry_scores):.4f} max={max(dry_scores):.4f} mean={np.mean(dry_scores):.4f}")
    print(f"Room > Dry: {sum(1 for r, d in zip(room_scores, dry_scores) if r > d)}/{len(results)}")

    # Check if we can separate self from non-self using causal filter
    # The self-calibration trials should score higher than silence/world_only
    print("\n=== Separation test ===")
    # Score the fresh controls with each causal filter
    fresh_pos = [
        ("overlap_rep00", "data/m3_broca_controls_fresh/overlap_rep00"),
        ("overlap_rep01", "data/m3_broca_controls_fresh/overlap_rep01"),
        ("overlap_rep02", "data/m3_broca_controls_fresh/overlap_rep02"),
    ]
    fresh_neg = [
        ("silence_rep00", "data/m3_broca_controls_fresh/silence_rep00"),
        ("silence_rep01", "data/m3_broca_controls_fresh/silence_rep01"),
        ("world_only_rep00", "data/m3_broca_controls_fresh/world_only_rep00"),
        ("world_only_rep01", "data/m3_broca_controls_fresh/world_only_rep01"),
    ]

    for fit_trial in trials:
        fit_summary = json.loads((fit_trial / "summary.json").read_text())
        fit_observed = np.load(fit_trial / "observed.npy")
        fit_shift = int(fit_summary["self_expected_shift_samples"])
        h = fit_room_filter(fit_observed, template, fit_shift, taps=64, ridge=0.000112)
        room_ref = apply_room_filter(template, h)

        print(f"\nFilter from {fit_trial.name}:")
        for name, path in fresh_pos + fresh_neg:
            s = json.loads(Path(path, "summary.json").read_text())
            obs = np.load(Path(path) / "observed.npy")
            shift = int(s["self_expected_shift_samples"])
            score, _ = local_score(obs, room_ref, shift, search_ms=256.4)
            label = "POS" if (name, path) in fresh_pos else "NEG"
            print(f"  {label} {name}: {score:.4f}")

asyncio.run(main())
