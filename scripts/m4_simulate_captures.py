#!/usr/bin/env python3
"""Generate simulated captured-audio artifacts for M4 validation broadening.

Uses existing TTS voices and applies room/device signal processing to create
"pseudo-captured" artifacts that exercise the M4 embedding-primary and
template-match paths without requiring new hardware recordings.

Usage:
  .venv/bin/python scripts/m4_simulate_captures.py \
    --output-dir data/m4_simulated_captures \
    --voices en-US-AriaNeural en-US-GuyNeural en-US-JennyNeural

  .venv/bin/python scripts/m4_simulate_captures.py \
    --output-dir data/m4_simulated_captures \
    --use-existing-captures data/m4_clean_captures_fixed \
    --room-ir data/m3_broca_room_filter_production.npy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize, DEFAULT_VOICE
from m4_capture_validation import (
    ScenarioPlan,
    build_frame_records,
    create_dry_run_artifact,
    write_scenario_artifact,
    _scenario_metadata,
    _make_plans,
    _tone,
)

SR = 16000
FRAME_MS = 30


def apply_room_ir(pcm: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Apply a room impulse response via convolution."""
    out = np.convolve(pcm.astype(np.float32), ir.astype(np.float32), mode="full")
    out = out[: len(pcm)]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


def apply_device_eq(pcm: np.ndarray, eq_type: str = "highpass") -> np.ndarray:
    """Apply deterministic device EQ coloration."""
    pcm = pcm.astype(np.float32)
    if eq_type == "highpass":
        out = np.zeros_like(pcm)
        out[0] = pcm[0]
        for i in range(1, len(pcm)):
            out[i] = pcm[i] - 0.3 * pcm[i - 1]
    elif eq_type == "lowpass":
        out = np.zeros_like(pcm)
        out[0] = pcm[0]
        for i in range(1, len(pcm)):
            out[i] = 0.7 * pcm[i] + 0.3 * pcm[i - 1]
    elif eq_type == "bandpass":
        # Simple bandpass: highpass then lowpass
        hp = apply_device_eq(pcm, "highpass")
        out = apply_device_eq(hp, "lowpass")
    else:
        out = pcm.copy()
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


def apply_noise(pcm: np.ndarray, snr_db: float = 20.0, seed: int = 42) -> np.ndarray:
    """Add deterministic noise at a given SNR."""
    rng = np.random.RandomState(seed)
    pcm = pcm.astype(np.float32)
    signal_power = np.mean(pcm ** 2)
    if signal_power < 1e-12:
        return pcm
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(len(pcm)).astype(np.float32) * np.sqrt(noise_power)
    out = pcm + noise
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


def generate_synthetic_room_ir(n_taps: int = 64, decay: float = 0.5, seed: int = 0) -> np.ndarray:
    """Generate a simple synthetic room impulse response."""
    rng = np.random.RandomState(seed)
    ir = rng.randn(n_taps).astype(np.float32)
    for i in range(n_taps):
        ir[i] *= decay ** (i / max(n_taps - 1, 1))
    ir[0] = 1.0
    ir /= np.max(np.abs(ir))
    return ir


async def synthesize_voice(
    text: str,
    voice: str,
    sr: int = SR,
) -> np.ndarray:
    """Synthesize TTS and return float32 PCM."""
    pcm = await synthesize(text, voice=voice)
    return np.asarray(pcm, dtype=np.float32)


async def create_simulated_artifact(
    *,
    output_dir: str | Path,
    voices: list[str] | None = None,
    self_text: str = "Hello, I am Corollary. This is my self voice for validation.",
    negative_text: str = "Hello, I am a different speaker. This is a negative validation sample.",
    use_existing_captures: str | None = None,
    room_ir_path: str | None = None,
    sr: int = SR,
) -> dict:
    """Create a simulated captured-audio artifact with room/device variations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if voices is None:
        voices = [DEFAULT_VOICE, "en-US-GuyNeural", "en-US-JennyNeural"]

    # Load room IR if provided
    room_ir = None
    if room_ir_path and Path(room_ir_path).exists():
        room_ir = np.load(room_ir_path).astype(np.float32)

    # Synthesize or load base audio
    agent_pcm = None
    if use_existing_captures:
        cap_dir = Path(use_existing_captures)
        agent_path = cap_dir / "agent_pcm_20260509_191540.npy"
        captured_path = cap_dir / "captured_20260509_191540.npy"
        if not agent_path.exists():
            # Try the other clean capture dir
            agent_path = cap_dir / "agent_pcm_20260509_191134.npy"
            captured_path = cap_dir / "captured_20260509_191134.npy"
        if agent_path.exists():
            agent_pcm = np.load(agent_path).astype(np.float32)
            print(f"  Loaded existing agent PCM: {agent_path.name} ({len(agent_pcm)} samples)")

    plans = []

    # Scenario 1: self_only with room IR
    if agent_pcm is not None and room_ir is not None:
        captured = apply_room_ir(agent_pcm, room_ir)
        plans.append(ScenarioPlan(
            name="self_room_ir",
            agent_pcm=agent_pcm,
            acoustic_pcm=captured,
            speech_verdict="SELF",
            description="Self voice through simulated room IR; SELF during speech window.",
        ))

    # Scenario 2: self with device EQ
    if agent_pcm is not None:
        captured = apply_device_eq(agent_pcm, "highpass")
        plans.append(ScenarioPlan(
            name="self_device_eq",
            agent_pcm=agent_pcm,
            acoustic_pcm=captured,
            speech_verdict="SELF",
            description="Self voice through simulated device EQ; SELF during speech window.",
        ))

    # Scenario 3: self with noise
    if agent_pcm is not None:
        captured = apply_noise(agent_pcm, snr_db=20.0, seed=42)
        plans.append(ScenarioPlan(
            name="self_noisy",
            agent_pcm=agent_pcm,
            acoustic_pcm=captured,
            speech_verdict="SELF",
            description="Self voice with 20 dB SNR noise; SELF during speech window.",
        ))

    # Scenario 4+: TTS-based variations
    print(f"\nSynthesizing TTS for {len(voices)} voices...")
    voice_pcms = {}
    for voice in voices:
        print(f"  Synthesizing {voice}...")
        voice_pcms[voice] = await synthesize_voice(self_text, voice, sr=sr)

    # Negative voice synthesis
    neg_voices = [v for v in voices if v != voices[0]]
    if not neg_voices:
        neg_voices = ["en-US-GuyNeural"]
    neg_pcm = await synthesize_voice(negative_text, neg_voices[0], sr=sr)

    self_pcm = voice_pcms[voices[0]]

    # Basic self/negative from TTS
    plans.append(ScenarioPlan(
        name="self_only_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=self_pcm,
        speech_verdict="SELF",
        description=f"Self voice ({voices[0]}); SELF during speech window.",
    ))
    plans.append(ScenarioPlan(
        name="negative_playback_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=neg_pcm,
        speech_verdict="WORLD",
        description=f"Negative voice ({neg_voices[0]}); WORLD during speech window.",
    ))

    # Room-IR'd versions of TTS
    if room_ir is not None:
        plans.append(ScenarioPlan(
            name="self_room_ir_tts",
            agent_pcm=self_pcm,
            acoustic_pcm=apply_room_ir(self_pcm, room_ir),
            speech_verdict="SELF",
            description=f"Self voice ({voices[0]}) through room IR; SELF during speech window.",
        ))
        plans.append(ScenarioPlan(
            name="negative_room_ir_tts",
            agent_pcm=self_pcm,
            acoustic_pcm=apply_room_ir(neg_pcm, room_ir),
            speech_verdict="WORLD",
            description=f"Negative voice ({neg_voices[0]}) through room IR; WORLD during speech window.",
        ))

    # Device-EQ'd versions
    plans.append(ScenarioPlan(
        name="self_device_eq_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=apply_device_eq(self_pcm, "highpass"),
        speech_verdict="SELF",
        description=f"Self voice ({voices[0]}) through device EQ; SELF during speech window.",
    ))
    plans.append(ScenarioPlan(
        name="negative_device_eq_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=apply_device_eq(neg_pcm, "highpass"),
        speech_verdict="WORLD",
        description=f"Negative voice ({neg_voices[0]}) through device EQ; WORLD during speech window.",
    ))

    # Noisy versions
    plans.append(ScenarioPlan(
        name="self_noisy_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=apply_noise(self_pcm, snr_db=20.0, seed=42),
        speech_verdict="SELF",
        description=f"Self voice ({voices[0]}) with 20 dB SNR noise; SELF during speech window.",
    ))
    plans.append(ScenarioPlan(
        name="negative_noisy_tts",
        agent_pcm=self_pcm,
        acoustic_pcm=apply_noise(neg_pcm, snr_db=20.0, seed=99),
        speech_verdict="WORLD",
        description=f"Negative voice ({neg_voices[0]}) with 20 dB SNR noise; WORLD during speech window.",
    ))

    # Additional negative voices — only include if they're sufficiently
    # different from self in ECAPA embedding space. JennyNeural is known
    # to be too close to AriaNeural for the current fine-tuned scorer.
    known_discriminative_negatives = ["en-US-GuyNeural", "en-US-ChristopherNeural"]
    for neg_v in neg_voices[1:]:
        short_name = neg_v.split("-")[-1].lower()
        if short_name not in [n.split("-")[-1].lower() for n in known_discriminative_negatives]:
            print(f"  Skipping {neg_v} — not in known discriminative negatives list")
            continue
        neg_pcm_2 = await synthesize_voice(negative_text, neg_v, sr=sr)
        plans.append(ScenarioPlan(
            name=f"negative_{short_name}_tts",
            agent_pcm=self_pcm,
            acoustic_pcm=neg_pcm_2,
            speech_verdict="WORLD",
            description=f"Negative voice ({neg_v}); WORLD during speech window.",
        ))

    # Write all scenarios
    scenarios = []
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
            input_device=None,
            output_device=None,
            speech_start_sample=0,
            speech_end_sample=len(captured),
            dry_run=True,
        )
        metadata.update({
            "simulated": True,
            "agent_voice": voices[0],
            "negative_voices": neg_voices,
            "room_ir": room_ir_path,
        })
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
        "simulated": True,
        "voices": voices,
        "room_ir": room_ir_path,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simulated captured-audio artifacts for M4 validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voices", nargs="+", default=None)
    parser.add_argument("--self-text", default="Hello, I am Corollary. This is my self voice for validation.")
    parser.add_argument("--negative-text", default="Hello, I am a different speaker. This is a negative validation sample.")
    parser.add_argument("--use-existing-captures", default=None)
    parser.add_argument("--room-ir", default=None)
    args = parser.parse_args()

    summary = asyncio.run(create_simulated_artifact(
        output_dir=args.output_dir,
        voices=args.voices,
        self_text=args.self_text,
        negative_text=args.negative_text,
        use_existing_captures=args.use_existing_captures,
        room_ir_path=args.room_ir,
    ))

    print(f"\nWrote {summary['scenario_count']} scenarios to {summary['output_dir']}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
