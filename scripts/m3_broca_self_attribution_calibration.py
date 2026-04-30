#!/usr/bin/env python3
"""Broca self-only attribution calibration for embodied M2/M3.

Synthesizes one Broca utterance, plays it as the M2 self signal with world audio
zeroed, and sweeps timing/similarity settings to measure whether real TTS
speaker->mic capture is recognizable as self-present.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

DEFAULT_TEXT = "I was speaking, then paused."


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_methods(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    total = result["self_frames"] + result["both_frames"] + result["world_frames"]
    return {
        "self_score": result["self_score"],
        "self_frames": result["self_frames"],
        "both_frames": result["both_frames"],
        "world_frames": result["world_frames"],
        "self_present_frames": result["self_frames"] + result["both_frames"],
        "self_present_ratio": (result["self_frames"] + result["both_frames"]) / total if total else 0.0,
        "speech_ratio": result["speech_ratio"],
        "onset_error_ms": result["self_onset_error_ms"],
        "raw_text": result["raw_text"],
        "gated_text": result["gated_text"],
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Calibrate embodied Broca self attribution")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latencies-ms", default="300,400,500")
    parser.add_argument("--tolerances-ms", default="75,175")
    parser.add_argument("--similarity-thresholds", default="0.02,0.10")
    parser.add_argument("--onset-methods", default="energy,xcorr")
    parser.add_argument("--save-root", default="data/m3_broca_self_calibration_20260430")
    args = parser.parse_args()

    print(f"Synthesizing Broca self text: {args.text!r}")
    self_pcm = await synthesize(args.text)
    print(f"Broca PCM samples: {len(self_pcm)}")

    latencies = parse_floats(args.latencies_ms)
    tolerances = parse_floats(args.tolerances_ms)
    thresholds = parse_floats(args.similarity_thresholds)
    methods = parse_methods(args.onset_methods)

    root = Path(args.save_root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trial_idx = 0
    for method in methods:
        for latency in latencies:
            for tolerance in tolerances:
                for threshold in thresholds:
                    label = f"trial{trial_idx:03d}_{method}_lat{int(latency)}_tol{int(tolerance)}_thr{str(threshold).replace('.', 'p')}"
                    print(f"RUN {label}", flush=True)
                    result = run_m2_trial(
                        self_stim="broca_tts_self",
                        self_pcm_override=self_pcm,
                        world_speech_file="data/world_speech_sample.npy",
                        input_device=args.input_device,
                        output_device=args.output_device,
                        noise_device=args.noise_device,
                        seed=args.seed,
                        expected_text=args.text,
                        both_policy="manual",
                        both_gain=1.0,
                        self_duration_ms=int(len(self_pcm) * 1000 / 16000),
                        self_amplitude=1.0,
                        self_only=True,
                        speech_override="vad",
                        similarity_threshold=threshold,
                        onset_latency_ms=latency,
                        onset_method=method,
                        causal_tolerance_ms=tolerance,
                    )
                    save_dir = root / label
                    save_trial_artifacts(str(save_dir), result)
                    row: dict[str, Any] = {
                        "label": label,
                        "save_dir": str(save_dir),
                        "onset_method": method,
                        "onset_latency_ms": latency,
                        "causal_tolerance_ms": tolerance,
                        "similarity_threshold": threshold,
                    }
                    row.update(summarize(result))
                    rows.append(row)
                    trial_idx += 1

    ranked = sorted(
        rows,
        key=lambda r: (
            r["self_present_ratio"],
            -abs(r["onset_error_ms"]),
            r["self_score"],
        ),
        reverse=True,
    )
    out = {
        "text": args.text,
        "n_trials": len(rows),
        "rows": rows,
        "top": ranked[:10],
    }
    (root / "calibration_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== TOP CONFIGS ===")
    for r in ranked[:10]:
        print(
            f"{r['label']} present={r['self_present_ratio']:.3f} "
            f"self/both/world={r['self_frames']}/{r['both_frames']}/{r['world_frames']} "
            f"score={r['self_score']:.3f} onset_err={r['onset_error_ms']:.1f}"
        )
    print(f"Saved: {root / 'calibration_summary.json'}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
