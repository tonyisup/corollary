#!/usr/bin/env python3
"""Validate multi-seed optimized M2 params vs manual baseline across seeds 0-2."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

EXPECTED_TEXT = "this is a test of world speech conservation"
VOWEL_STIM = "vowel_ah_var_nearfield"
BABBLE_STIM = "babble_ba_repeat_short_tight_var"
SEEDS = [0, 1, 2]


MANUAL_PARAMS = {
    "similarity_threshold": 0.20,
    "onset_latency_ms": 400,
    "onset_method": "energy",
    "causal_tolerance_ms": 75,
    "vowel_gain": 0.5,
    "babble_gain": 1.0,
    "self_amplitude": 1.0,
    "self_duration_ms": 900,
}

OPTIMIZED_PARAMS = {
    "similarity_threshold": 0.243,
    "onset_latency_ms": 406,
    "onset_method": "energy",
    "causal_tolerance_ms": 73,
    "vowel_gain": 0.602,
    "babble_gain": 0.968,
    "self_amplitude": 0.559,
    "self_duration_ms": 995,
}


def run_config(params: dict, stim: str, seed: int, save_dir: Path | None = None):
    return run_m2_trial(
        self_stim=stim,
        world_speech_file="data/world_speech_sample.npy",
        input_device=3,
        output_device=5,
        noise_device=4,
        seed=seed,
        expected_text=EXPECTED_TEXT,
        both_policy="family",
        self_duration_ms=int(params["self_duration_ms"]),
        self_amplitude=params["self_amplitude"],
        speech_override="vad",
        similarity_threshold=params["similarity_threshold"],
        onset_latency_ms=int(params["onset_latency_ms"]),
        onset_method=params["onset_method"],
        causal_tolerance_ms=int(params["causal_tolerance_ms"]),
    )


def main():
    results = {"manual": {}, "optimized": {}}

    for name, params in [("manual", MANUAL_PARAMS), ("optimized", OPTIMIZED_PARAMS)]:
        for stim in [VOWEL_STIM, BABBLE_STIM]:
            for seed in SEEDS:
                print(f"  {name}/{stim}/seed{seed}...")
                r = run_config(params, stim, seed)
                results[name][f"{stim}/seed{seed}"] = {
                    "self_match": r.get("self_score"),
                    "self_frames": r.get("self_frames"),
                    "both_frames": r.get("both_frames"),
                    "world_frames": r.get("world_frames"),
                    "speech_ratio": r.get("speech_ratio"),
                    "raw_wer": r["metrics"]["raw_wer"] if r.get("metrics") else None,
                    "gated_wer": r["metrics"]["gated_wer"] if r.get("metrics") else None,
                }

    out_path = Path("data/m2_multiseed_validation.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\n=== Summary ===")
    print(f"{'Config':<12} {'Self Match (mean)':<18} {'BOTH (mean)':<18} {'Gated WER (mean)':<18}")
    for name in ["manual", "optimized"]:
        vals = list(results[name].values())
        avg_sm = sum(v["self_match"] or 0.0 for v in vals) / len(vals)
        avg_both = sum(v["both_frames"] or 0 for v in vals) / len(vals)
        avg_wer = sum(v["gated_wer"] or 1.0 for v in vals) / len(vals)
        print(f"{name:<12} {avg_sm:<18.3f} {avg_both:<18.1f} {avg_wer:<18.3f}")


if __name__ == "__main__":
    main()
