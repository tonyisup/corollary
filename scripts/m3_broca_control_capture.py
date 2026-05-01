#!/usr/bin/env python3
"""Capture Broca-specific control artifacts for room-threshold calibration.

Collects the missing negative/diagnostic controls called out in docs:
- silence
- world_only
- overlap

The goal is to reuse the same Broca TTS reference text and device path used by the
embodied M2/M3 loop so threshold calibration is not relying on cross-stimulus
negatives alone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts
from m3_broca_room_threshold_calibration import DEFAULT_TEXT


def build_broca_control_cases(world_speech: str) -> list[dict[str, Any]]:
    return [
        {
            "name": "silence",
            "threshold_role": "negative",
            "world_speech": "__silence__",
            "world_only": True,
            "description": "No Broca playback and no external speech; same capture path.",
        },
        {
            "name": "world_only",
            "threshold_role": "negative",
            "world_speech": world_speech,
            "world_only": True,
            "description": "External speech without Broca self playback.",
        },
        {
            "name": "overlap",
            "threshold_role": "diagnostic",
            "world_speech": world_speech,
            "world_only": False,
            "description": "Broca self playback overlapped with external speech.",
        },
    ]


def artifact_dir_for_case(root: str | Path, case_name: str, repetition: int, total_repetitions: int) -> Path:
    width = max(2, len(str(total_repetitions)))
    return Path(root) / f"{case_name}_rep{repetition:0{width}d}"


def update_summary_with_control_metadata(
    summary_path: str | Path,
    *,
    case: dict[str, Any],
    broca_text: str,
    input_device: int,
    output_device: int,
    noise_device: int,
) -> None:
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update(
        {
            "control_case": case["name"],
            "threshold_role": case["threshold_role"],
            "control_description": case.get("description"),
            "broca_text": broca_text,
            "world_speech": case["world_speech"],
            "capture_devices": {
                "input_device": input_device,
                "output_device": output_device,
                "noise_device": noise_device,
            },
        }
    )
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def render_threshold_command(
    *,
    positives: list[Path],
    captured_cases: list[dict[str, Any]],
    output_path: Path,
) -> str:
    parts = ["./.venv/bin/python scripts/m3_broca_room_threshold_calibration.py"]
    for positive in positives:
        parts.extend(["--positive", str(positive)])
    for case in captured_cases:
        if case["threshold_role"] == "negative":
            parts.extend(["--negative", str(case["path"])])
    parts.extend(["--output", str(output_path)])
    return " ".join(parts)


def write_manifest(manifest_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Capture Broca-specific silence/world-only/overlap controls")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Broca TTS reference text")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--both-gain", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.20)
    parser.add_argument("--onset-latency-ms", type=float, default=300.0)
    parser.add_argument("--onset-method", choices=["xcorr", "energy", "spectral_flux", "score"], default="energy")
    parser.add_argument("--causal-tolerance-ms", type=float, default=175.0)
    parser.add_argument("--self-match-mode", choices=["global_xcorr", "local_xcorr", "room_local_oracle"], default="global_xcorr")
    parser.add_argument("--self-match-search-ms", type=float, default=256.4)
    parser.add_argument("--room-filter-taps", type=int, default=64)
    parser.add_argument("--room-filter-ridge", type=float, default=0.000112)
    parser.add_argument("--save-root", default="data/m3_broca_controls")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--positive", action="append", default=[], help="Optional positive/self artifact dir to render follow-up calibration command")
    parser.add_argument("--threshold-output", default="data/m3_broca_room_threshold_calibration_broca_controls.json")
    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")

    cases = build_broca_control_cases(world_speech=args.world_speech)
    self_pcm = await synthesize(args.text)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    captured_cases: list[dict[str, Any]] = []
    silence_world = np.zeros_like(np.load(args.world_speech))

    with tempfile.NamedTemporaryFile(prefix="broca_silence_", suffix=".npy", delete=False) as tmp:
        silence_path = Path(tmp.name)
    try:
        np.save(silence_path, silence_world)

        for repetition in range(args.repetitions):
            for case in cases:
                world_speech = str(silence_path) if case["world_speech"] == "__silence__" else case["world_speech"]
                artifact_dir = artifact_dir_for_case(save_root, case["name"], repetition, args.repetitions)
                print(f"\n=== Capturing {case['name']} rep {repetition + 1}/{args.repetitions} ===")
                result = run_m2_trial(
                    self_stim="broca_tts_self",
                    self_pcm_override=self_pcm,
                    world_speech_file=world_speech,
                    input_device=args.input_device,
                    output_device=args.output_device,
                    noise_device=args.noise_device,
                    seed=args.seed,
                    expected_text=None,
                    both_policy="manual",
                    both_gain=args.both_gain,
                    self_duration_ms=int(len(self_pcm) * 1000 / 16000),
                    self_amplitude=1.0,
                    speech_override="vad",
                    similarity_threshold=args.similarity_threshold,
                    onset_latency_ms=args.onset_latency_ms,
                    onset_method=args.onset_method,
                    causal_tolerance_ms=args.causal_tolerance_ms,
                    self_match_mode=args.self_match_mode,
                    self_match_search_ms=args.self_match_search_ms,
                    room_filter_taps=args.room_filter_taps,
                    room_filter_ridge=args.room_filter_ridge,
                    world_only=case["world_only"],
                )
                save_trial_artifacts(str(artifact_dir), result)
                update_summary_with_control_metadata(
                    artifact_dir / "summary.json",
                    case=case,
                    broca_text=args.text,
                    input_device=args.input_device,
                    output_device=args.output_device,
                    noise_device=args.noise_device,
                )
                captured_cases.append(
                    {
                        "name": case["name"],
                        "threshold_role": case["threshold_role"],
                        "path": str(artifact_dir),
                    }
                )
                print(f"Saved artifacts to: {artifact_dir}")

        manifest = {
            "broca_text": args.text,
            "world_speech": args.world_speech,
            "capture_devices": {
                "input_device": args.input_device,
                "output_device": args.output_device,
                "noise_device": args.noise_device,
            },
            "cases": captured_cases,
            "threshold_command": render_threshold_command(
                positives=[Path(p) for p in args.positive],
                captured_cases=captured_cases,
                output_path=Path(args.threshold_output),
            ) if args.positive else None,
        }
        manifest_path = save_root / "manifest.json"
        write_manifest(manifest_path, manifest)
        print(f"\nSaved manifest: {manifest_path}")
        if manifest["threshold_command"]:
            print("Suggested threshold calibration command:")
            print(manifest["threshold_command"])
        else:
            print("No --positive artifacts supplied; manifest saved without a threshold command.")
    finally:
        silence_path.unlink(missing_ok=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
