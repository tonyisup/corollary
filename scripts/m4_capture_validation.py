#!/usr/bin/env python3
"""Dedicated M4 captured-audio validation artifact builder.

This creates a small replay-compatible artifact for M4 speaker-identity work,
separate from M3 gated-playback artifacts. The first minimal battery contains:

- self_only: Corollary TTS played/captured; ground truth SELF during speech.
- same_domain_negative_playback_window: non-self TTS played/captured while the
  agent self template is the playback opportunity; ground truth WORLD.

The output layout is intentionally compatible with scripts/m4_f1_benchmark.py
--replay-dir: each scenario contains live_captured.npy, agent_pcm.npy, and
frame_records.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import DEFAULT_VOICE, synthesize  # noqa: E402
from m4_clean_capture import _build_playback_buffer  # noqa: E402

SR = 16000
FRAME_MS = 30
DEFAULT_SELF_TEXT = "Hello, I am Corollary. This is my self voice for captured validation."
DEFAULT_NEGATIVE_TEXT = "Hello, I am a different speaker. This is a same-domain negative validation sample."
DEFAULT_NEGATIVE_VOICE = "en-US-GuyNeural"


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    agent_pcm: np.ndarray
    acoustic_pcm: np.ndarray
    speech_verdict: str
    description: str


def build_frame_records(
    *,
    total_samples: int,
    speech_start_sample: int,
    speech_end_sample: int,
    speech_verdict: str,
    sr: int = SR,
    frame_ms: int = FRAME_MS,
) -> list[dict]:
    """Build frame-level ground-truth records for replay scoring.

    Frames whose start sample falls inside [speech_start_sample, speech_end_sample)
    receive `speech_verdict`; all other frames are SILENCE. This keeps the
    labels deterministic and compatible with m4_f1_benchmark.build_replay_cases().
    """
    frame_samples = int(sr * frame_ms / 1000)
    n_frames = total_samples // frame_samples
    records: list[dict] = []
    for frame_index in range(n_frames):
        start = frame_index * frame_samples
        end = start + frame_samples
        verdict = speech_verdict if speech_start_sample <= start < speech_end_sample else "SILENCE"
        records.append({
            "frame_index": frame_index,
            "timestamp_s": round(start / sr, 6),
            "start_sample": start,
            "end_sample": end,
            "verdict": verdict,
        })
    return records


def write_scenario_artifact(
    *,
    scenario_dir: Path,
    agent_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    frame_records: list[dict],
    metadata: dict,
) -> None:
    """Write one replay-compatible M4 validation scenario."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.save(scenario_dir / "agent_pcm.npy", np.asarray(agent_pcm, dtype=np.float32))
    np.save(scenario_dir / "live_captured.npy", np.asarray(captured_pcm, dtype=np.float32))
    (scenario_dir / "frame_records.json").write_text(json.dumps(frame_records, indent=2))
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _tone(seed: int, duration_s: float, sr: int = SR) -> np.ndarray:
    """Generate deterministic speech-like dry-run audio."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    f0 = 140 + (seed % 5) * 35
    pcm = np.zeros(n, dtype=np.float32)
    for h in range(1, 6):
        pcm += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    pcm *= 0.45 + 0.55 * np.sin(2 * np.pi * 3.5 * t) ** 2
    peak = float(np.max(np.abs(pcm)))
    if peak > 0:
        pcm = pcm / peak * 0.4
    return pcm.astype(np.float32)


def _make_plans(
    self_pcm: np.ndarray,
    negative_pcm: np.ndarray,
    negative_pcm_2: np.ndarray | None = None,
) -> list[ScenarioPlan]:
    plans = [
        ScenarioPlan(
            name="self_only",
            agent_pcm=self_pcm,
            acoustic_pcm=self_pcm,
            speech_verdict="SELF",
            description="Captured Corollary self voice; SELF during speech window.",
        ),
        ScenarioPlan(
            name="same_domain_negative_playback_window",
            agent_pcm=self_pcm,
            acoustic_pcm=negative_pcm,
            speech_verdict="WORLD",
            description="Captured non-self voice in agent playback opportunity; WORLD during speech window.",
        ),
    ]
    if negative_pcm_2 is not None:
        plans.append(ScenarioPlan(
            name="same_domain_negative_playback_window_2",
            agent_pcm=self_pcm,
            acoustic_pcm=negative_pcm_2,
            speech_verdict="WORLD",
            description="Captured second non-self voice in agent playback opportunity; WORLD during speech window.",
        ))
    return plans


def _scenario_metadata(
    *,
    plan: ScenarioPlan,
    sr: int,
    input_device: int | None,
    output_device: int | None,
    speech_start_sample: int,
    speech_end_sample: int,
    dry_run: bool,
) -> dict:
    return {
        "scenario": plan.name,
        "description": plan.description,
        "sr": sr,
        "input_device": input_device,
        "output_device": output_device,
        "speech_start_sample": speech_start_sample,
        "speech_end_sample": speech_end_sample,
        "speech_verdict": plan.speech_verdict,
        "dry_run": dry_run,
        "agent_samples": int(len(plan.agent_pcm)),
        "acoustic_samples": int(len(plan.acoustic_pcm)),
    }


def create_dry_run_artifact(
    *,
    output_dir: str | Path,
    sr: int = SR,
    duration_s: float = 1.2,
    include_variations: bool = True,
) -> dict:
    """Create a synthetic replay-compatible artifact without hardware.

    When include_variations is True, adds room-colored and device-EQ'd
    self/negative variants to exercise the same embedding-primary and
    template-match paths that real captured audio exercises.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    self_pcm = _tone(seed=42, duration_s=duration_s, sr=sr)
    negative_pcm = _tone(seed=99, duration_s=duration_s, sr=sr)
    negative_pcm_2 = _tone(seed=77, duration_s=duration_s, sr=sr) if include_variations else None

    all_plans = _make_plans(self_pcm, negative_pcm, negative_pcm_2)

    if include_variations:
        # Room-colored self: simple 1-tap "room" delay
        room_self = np.concatenate([np.zeros(2, dtype=np.float32), self_pcm[:-2] * 0.9])
        all_plans.append(ScenarioPlan(
            name="self_room_colored",
            agent_pcm=self_pcm,
            acoustic_pcm=room_self.astype(np.float32),
            speech_verdict="SELF",
            description="Synthetic room-colored self voice; SELF during speech window.",
        ))
        # Device-EQ'd self: simple high-pass emphasis
        eq_self = self_pcm.copy()
        eq_self[1:] -= 0.3 * self_pcm[:-1]
        all_plans.append(ScenarioPlan(
            name="self_device_eq",
            agent_pcm=self_pcm,
            acoustic_pcm=eq_self.astype(np.float32),
            speech_verdict="SELF",
            description="Synthetic device-EQ'd self voice; SELF during speech window.",
        ))
        # Room-colored negative
        room_neg = np.concatenate([np.zeros(2, dtype=np.float32), negative_pcm[:-2] * 0.9])
        all_plans.append(ScenarioPlan(
            name="same_domain_negative_playback_window_room",
            agent_pcm=self_pcm,
            acoustic_pcm=room_neg.astype(np.float32),
            speech_verdict="WORLD",
            description="Synthetic room-colored non-self voice; WORLD during speech window.",
        ))

    scenarios = []
    for plan in all_plans:
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
            input_device=None,
            output_device=None,
            speech_start_sample=0,
            speech_end_sample=len(captured),
            dry_run=True,
        )
        write_scenario_artifact(
            scenario_dir=output_path / plan.name,
            agent_pcm=plan.agent_pcm,
            captured_pcm=captured,
            frame_records=records,
            metadata=metadata,
        )
        scenarios.append(metadata)
    summary = {
        "output_dir": str(output_path),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "dry_run": True,
        "include_variations": include_variations,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _capture_playback(
    *,
    playback_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    sr: int = SR,
    lead_in_s: float = 0.3,
    tail_s: float = 1.0,
) -> tuple[np.ndarray, int, int]:
    """Play one PCM through the output device and capture mic input."""
    import sounddevice as sd

    play_buffer, lead_in = _build_playback_buffer(
        playback_pcm,
        sr=sr,
        lead_in_s=lead_in_s,
        tail_s=tail_s,
    )
    capture = np.zeros(len(play_buffer), dtype=np.float32)
    play_pos = [0]
    capture_pos = [0]

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Stream status: {status}")
        pp = play_pos[0]
        ep = min(pp + frames, len(play_buffer))
        outdata[:, 0] = 0
        if pp < len(play_buffer):
            n = ep - pp
            outdata[:n, 0] = play_buffer[pp:ep]
            play_pos[0] = ep
        cp = capture_pos[0]
        ce = min(cp + frames, len(capture))
        if cp < len(capture):
            n = ce - cp
            capture[cp:ce] = indata[:n, 0]
            capture_pos[0] = ce

    duration_s = len(play_buffer) / sr
    with sd.Stream(
        samplerate=sr,
        blocksize=int(sr * FRAME_MS / 1000),
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        time.sleep(duration_s + 0.25)
    speech_start = lead_in
    speech_end = lead_in + len(playback_pcm)
    return capture, speech_start, speech_end


async def create_live_artifact(
    *,
    output_dir: str | Path,
    input_device: int,
    output_device: int,
    self_text: str = DEFAULT_SELF_TEXT,
    negative_text: str = DEFAULT_NEGATIVE_TEXT,
    self_voice: str = DEFAULT_VOICE,
    negative_voice: str = DEFAULT_NEGATIVE_VOICE,
    sr: int = SR,
) -> dict:
    """Create a two-scenario live captured M4 validation artifact."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Synthesizing self voice...")
    self_pcm = np.asarray(await synthesize(self_text, voice=self_voice), dtype=np.float32)
    print("Synthesizing negative voice...")
    negative_pcm = np.asarray(await synthesize(negative_text, voice=negative_voice), dtype=np.float32)

    scenarios = []
    for plan in _make_plans(self_pcm, negative_pcm):
        print(f"\nRecording scenario: {plan.name}")
        print(f"  {plan.description}")
        captured, speech_start, speech_end = _capture_playback(
            playback_pcm=plan.acoustic_pcm,
            input_device=input_device,
            output_device=output_device,
            sr=sr,
        )
        rms = float(np.sqrt(np.mean(captured ** 2)))
        db = 20 * np.log10(max(rms, 1e-9))
        print(f"  Captured {len(captured)} samples, RMS={rms:.4f} ({db:.1f} dB)")
        records = build_frame_records(
            total_samples=len(captured),
            speech_start_sample=speech_start,
            speech_end_sample=speech_end,
            speech_verdict=plan.speech_verdict,
            sr=sr,
        )
        metadata = _scenario_metadata(
            plan=plan,
            sr=sr,
            input_device=input_device,
            output_device=output_device,
            speech_start_sample=speech_start,
            speech_end_sample=speech_end,
            dry_run=False,
        )
        metadata.update({
            "capture_samples": int(len(captured)),
            "capture_rms": round(rms, 6),
            "capture_db": round(db, 2),
            "self_text": self_text,
            "negative_text": negative_text,
            "self_voice": self_voice,
            "negative_voice": negative_voice,
        })
        # Store the expected self playback timeline, including lead-in/tail
        # silence, so replay scoring aligns agent_pcm with live_captured.npy.
        agent_reference, _ = _build_playback_buffer(plan.agent_pcm, sr=sr)
        write_scenario_artifact(
            scenario_dir=output_path / plan.name,
            agent_pcm=agent_reference,
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
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _default_output_dir() -> str:
    return f"data/m4_capture_validation_{time.strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create dedicated M4 captured-audio validation artifacts")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--self-text", default=DEFAULT_SELF_TEXT)
    parser.add_argument("--negative-text", default=DEFAULT_NEGATIVE_TEXT)
    parser.add_argument("--self-voice", default=DEFAULT_VOICE)
    parser.add_argument("--negative-voice", default=DEFAULT_NEGATIVE_VOICE)
    parser.add_argument("--dry-run", action="store_true", help="Write a synthetic artifact without hardware")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    if args.dry_run:
        summary = create_dry_run_artifact(output_dir=output_dir)
    else:
        summary = asyncio.run(create_live_artifact(
            output_dir=output_dir,
            input_device=args.input_device,
            output_device=args.output_device,
            self_text=args.self_text,
            negative_text=args.negative_text,
            self_voice=args.self_voice,
            negative_voice=args.negative_voice,
        ))
    print(json.dumps(summary, indent=2))
    print(f"\nArtifact written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
