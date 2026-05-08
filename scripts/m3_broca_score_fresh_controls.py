#!/usr/bin/env python3
"""Score fresh Broca controls using the pre-fitted room filter and compare
against the original calibration threshold."""

import asyncio
import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m3_broca_room_threshold_calibration import load_summary_observed, score_artifact

async def main():
    # Load the pre-fitted room filter from original calibration
    filter_path = "data/m3_broca_room_threshold_calibration_broca_controls_reexport.npy"
    h = np.load(filter_path)
    filters = [{"label": "original_calibration", "h": h}]
    print(f"Loaded room filter: {filter_path} ({len(h)} taps)")

    # Synthesize the template
    template = await synthesize("I was speaking, then paused.")
    print(f"Template: {len(template)} samples")

    # Load fresh controls
    pos_dirs = [
        "data/m3_broca_controls_fresh/overlap_rep00",
        "data/m3_broca_controls_fresh/overlap_rep01",
        "data/m3_broca_controls_fresh/overlap_rep02",
    ]
    neg_dirs = [
        "data/m3_broca_controls_fresh/silence_rep00",
        "data/m3_broca_controls_fresh/silence_rep01",
        "data/m3_broca_controls_fresh/silence_rep02",
        "data/m3_broca_controls_fresh/world_only_rep00",
        "data/m3_broca_controls_fresh/world_only_rep01",
        "data/m3_broca_controls_fresh/world_only_rep02",
    ]

    fallback_shift = int(np.median([
        json.loads(Path(d, "summary.json").read_text())["self_expected_shift_samples"]
        for d in pos_dirs
    ]))
    print(f"Fallback shift: {fallback_shift} samples")

    print("\n=== Scoring with ORIGINAL filter (0.208 threshold) ===")
    pos_scores = []
    for d in pos_dirs:
        p = load_summary_observed(d)
        s = score_artifact(p, filters, template, fallback_shift=fallback_shift, search_ms=256.4)
        pos_scores.append(s)
        print(f"  {s['label']}: room_max={s['max_room_score']:.4f} dry={s['dry_score']:.4f}")

    neg_scores = []
    for d in neg_dirs:
        p = load_summary_observed(d)
        s = score_artifact(p, filters, template, fallback_shift=fallback_shift, search_ms=256.4)
        neg_scores.append(s)
        print(f"  {s['label']}: room_max={s['max_room_score']:.4f} dry={s['dry_score']:.4f}")

    # Check separation at various thresholds
    print("\n=== Threshold analysis ===")
    for threshold in [0.01, 0.02, 0.04, 0.05, 0.10, 0.15, 0.208]:
        tp = sum(1 for s in pos_scores if s["max_room_score"] >= threshold)
        fp = sum(1 for s in neg_scores if s["max_room_score"] >= threshold)
        fn = len(pos_scores) - tp
        tn = len(neg_scores) - fp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / len(pos_scores) if pos_scores else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  thr={threshold:.3f}: TP={tp}/{len(pos_scores)} FP={fp}/{len(neg_scores)} prec={prec:.2f} rec={rec:.2f} f1={f1:.2f}")

asyncio.run(main())
