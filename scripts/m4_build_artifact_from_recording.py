#!/usr/bin/env python3
"""Build an M4 validation artifact from a pre-recorded human negative sample.

Usage:
  .venv/bin/python scripts/m4_build_artifact_from_recording.py \\
    --negative-audio data/m4_human_negative/human_negative.npy \\
    --output-dir data/m4_validation_human \\
    --input-device 3 \\
    --output-device 5

The negative audio should be a human speaker reading the negative validation text
through the same mic/speaker setup used for the self capture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m4_capture_validation import (
    ScenarioPlan,
    build_frame_records,
    write_scenario_artifact,
    _scenario_metadata,
    _make_plans,
)
from broca.tts import synthesize

SR = 16000
FRAME_MS = 30
DEFAULT_SELF_TEXT = "Hello, I am Corollary. This is my self voice for captured validation."
DEFAULT_NEGATIVE_TEXT = "Hello, I am a different speaker. This is a same-domain negative validation sample."


def build_artifact_from_recording(
    *,
    negative_audio: np.ndarray,
    output_dir: str | Path,
    self_text: str = DEFAULT_SELF_TEXT,
    negative_text: str = DEFAULT_NEGATIVE_TEXT,
    input_device: int = 3,
    output_device: int = 5,
    sr: int = SR,
) -> dict:
    """Build a replay-compatible M4 validation artifact with human-recorded negative."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Synthesize self voice (same as always)
    print("Synthesizing self voice...")
    self_pcm = np.asarray(asyncio.run(synthesize(self_text)), dtype=np.float32)

    # Use the pre-recorded human audio as the negative
    print(f"Using pre-recorded human negative: {len(negative_audio)} samples ({len(negative_audio)/sr:.1f}s)")
    negative_pcm = negative_audio.astype(np.float32)

    # Build scenario plans
    scenarios = []

    # Scenario 1: self_only
    plans = _make_plans(self_pcm, negative_pcm)
    for plan in plans:
        captured = plan.acoustic_pcm.astype(np.float32)
        records = build_frame_records(
            total_samples=len(captured),
            speech_start_sample=0,
            speech_end_sample=len(captured),
            speech_verdict=plan.speech_verdict,
            sr=sr,
        )
        metadata = _scenario_metadata(
            plan=plan,
            sr=sr,
            input_device=input_device,
            output_device=output_device,
            speech_start_sample=0,
            speech_end_sample=len(captured),
            dry_run=False,
        )
        metadata.update({
            "negative_source": "human_recording",
            "self_text": self_text,
            "negative_text": negative_text,
        })
        write_scenario_artifact(
            scenario_dir=output_path / plan.name,
            agent_pcm=plan.agent_pcm,
            captured_pcm=captured,
            frame_records=records,
            metadata=metadata,
        )
        scenarios.append(metadata)

    # Scenario 2: same-domain negative playback window (human)
    plan_human = ScenarioPlan(
        name="same_domain_negative_playback_window",
        agent_pcm=self_pcm,
        acoustic_pcm=negative_pcm,
        speech_verdict="WORLD",
        description="Human voice in agent playback opportunity; WORLD during speech window.",
    )
    captured = plan_human.acoustic_pcm.astype(np.float32)
    records = build_frame_records(
        total_samples=len(captured),
        speech_start_sample=0,
        speech_end_sample=len(captured),
        speech_verdict=plan_human.speech_verdict,
        sr=sr,
    )
    metadata = _scenario_metadata(
        plan=plan_human,
        sr=sr,
        input_device=input_device,
        output_device=output_device,
        speech_start_sample=0,
        speech_end_sample=len(captured),
        dry_run=False,
    )
    metadata.update({
        "negative_source": "human_recording",
        "self_text": self_text,
        "negative_text": negative_text,
    })
    write_scenario_artifact(
        scenario_dir=output_path / plan_human.name,
        agent_pcm=plan_human.agent_pcm,
        captured_pcm=captured,
        frame_records=records,
        metadata=metadata,
    )
    scenarios.append(metadata)

    summary = {
        "output_dir": str(output_path),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "dry_run": False,
        "negative_source": "human_recording",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build M4 validation artifact from a human-recorded negative sample"
    )
    parser.add_argument(
        "--negative-audio", required=True,
        help="Path to .npy file containing the human-recorded negative audio (16kHz float32)"
    )
    parser.add_argument("--output-dir", default="data/m4_validation_human")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--self-text", default=DEFAULT_SELF_TEXT)
    parser.add_argument("--negative-text", default=DEFAULT_NEGATIVE_TEXT)
    args = parser.parse_args()

    negative_audio = np.load(args.negative_audio).astype(np.float32)

    summary = build_artifact_from_recording(
        negative_audio=negative_audio,
        output_dir=args.output_dir,
        self_text=args.self_text,
        negative_text=args.negative_text,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nArtifact written to: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())