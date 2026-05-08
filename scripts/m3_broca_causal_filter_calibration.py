#!/usr/bin/env python3
"""Causal room filter calibration: fit on a dedicated self-playback capture,
then validate on separate positive/negative controls.

This is the production-causal path: the filter is fitted once at calibration
time and applied at runtime to unseen captures.
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
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts
from m3_broca_room_reference_analysis import fit_room_filter, apply_room_filter, local_score, dry_template_track

DEFAULT_TEXT = "I was speaking, then paused."


async def record_calibration_capture(
    *,
    text: str,
    input_device: int,
    output_device: int,
    seed: int = 0,
    save_dir: str | Path | None = None,
) -> dict:
    """Record a self-only calibration capture for room filter fitting."""
    self_pcm = await synthesize(text)
    result = run_m2_trial(
        self_stim="broca_tts_self",
        self_pcm_override=self_pcm,
        world_speech_file=None,
        input_device=input_device,
        output_device=output_device,
        noise_device=None,
        seed=seed,
        expected_text=text,
        both_policy="none",
        self_duration_ms=int(len(self_pcm) * 1000 / 16000),
        self_amplitude=1.0,
        speech_override="vad",
    )
    if save_dir:
        save_trial_artifacts(result, str(save_dir), seed=seed)
    return result


async def fit_causal_filter(
    cal_result: dict,
    template: np.ndarray,
    taps: int = 64,
    ridge: float = 0.000112,
) -> np.ndarray:
    """Fit room filter on a calibration capture."""
    observed = np.array(cal_result["observed_pcm"], dtype=np.float32)
    shift = int(cal_result.get("self_expected_shift_samples", 0))
    return fit_room_filter(observed, template, shift, taps=taps, ridge=ridge)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal room filter calibration")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.000112)
    parser.add_argument("--search-ms", type=float, default=256.4)
    parser.add_argument("--positive", action="append", default=[],
                        help="Positive control dirs (overlap captures)")
    parser.add_argument("--negative", action="append", default=[],
                        help="Negative control dirs (silence/world-only captures)")
    parser.add_argument("--calibration-dir", default=None,
                        help="Dir with existing calibration capture (skip recording)")
    parser.add_argument("--save-dir", default="data/m3_broca_causal_calibration",
                        help="Dir to save calibration capture and filter")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get calibration capture
    if args.calibration_dir:
        cal_dir = Path(args.calibration_dir)
        cal_summary = json.loads((cal_dir / "summary.json").read_text())
        cal_observed = np.load(cal_dir / "observed.npy")
        print(f"Using existing calibration capture: {cal_dir}")
    else:
        print("Recording calibration capture (self-only playback)...")
        cal_result = await record_calibration_capture(
            text=args.text,
            input_device=args.input_device,
            output_device=args.output_device,
            seed=0,
            save_dir=save_dir / "calibration_capture",
        )
        cal_observed = np.array(cal_result["observed_pcm"], dtype=np.float32)
        cal_summary = cal_result
        print(f"Calibration capture saved to {save_dir / 'calibration_capture'}")

    # Step 2: Synthesize template and fit filter
    template = await synthesize(args.text)
    shift = int(cal_summary.get("self_expected_shift_samples", 0))
    h = fit_room_filter(cal_observed, template, shift, taps=args.taps, ridge=args.ridge)
    print(f"Room filter fitted: {len(h)} taps, sum={h.sum():.3f}, max={h.max():.3f}")

    # Step 3: Export filter
    filter_path = save_dir / "room_filter_causal.npy"
    np.save(filter_path, h)
    print(f"Filter saved: {filter_path}")

    # Step 4: Validate on controls if provided
    if args.positive or args.negative:
        room_ref = apply_room_filter(template, h)
        fallback_shift = shift

        print("\n=== Validation on controls ===")
        pos_scores = []
        for p_dir in args.positive:
            p = json.loads(Path(p_dir, "summary.json").read_text())
            obs = np.load(Path(p_dir) / "observed.npy")
            s_shift = int(p.get("self_expected_shift_samples", fallback_shift))
            score, _ = local_score(obs, room_ref, s_shift, search_ms=args.search_ms)
            pos_scores.append(score)
            print(f"  POS {Path(p_dir).name}: {score:.4f}")

        neg_scores = []
        for n_dir in args.negative:
            n = json.loads(Path(n_dir, "summary.json").read_text())
            obs = np.load(Path(n_dir) / "observed.npy")
            n_shift = int(n.get("self_expected_shift_samples", fallback_shift))
            score, _ = local_score(obs, room_ref, n_shift, search_ms=args.search_ms)
            neg_scores.append(score)
            print(f"  NEG {Path(n_dir).name}: {score:.4f}")

        if pos_scores and neg_scores:
            min_pos = min(pos_scores)
            max_neg = max(neg_scores)
            if min_pos > max_neg:
                recommended = (min_pos + max_neg) / 2.0
                print(f"\nPerfect separation! Recommended threshold: {recommended:.4f}")
            else:
                print(f"\nOverlap: max_neg={max_neg:.4f} min_pos={min_pos:.4f}")
                recommended = (min_pos + max_neg) / 2.0
                print(f"Conservative threshold: {recommended:.4f}")

            # Export calibration metadata
            cal_meta = {
                "filter_path": str(filter_path),
                "recommended_threshold": recommended,
                "min_positive": min_pos,
                "max_negative": max_neg,
                "n_positives": len(pos_scores),
                "n_negatives": len(neg_scores),
                "taps": args.taps,
                "ridge": args.ridge,
                "search_ms": args.search_ms,
                "calibration_capture": str(cal_dir) if args.calibration_dir else str(save_dir / "calibration_capture"),
            }
            meta_path = save_dir / "room_filter_causal.json"
            meta_path.write_text(json.dumps(cal_meta, indent=2))
            print(f"Calibration metadata: {meta_path}")
    else:
        print("\nNo control dirs provided; skipping validation")


if __name__ == "__main__":
    asyncio.run(main())
