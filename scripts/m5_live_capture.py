#!/usr/bin/env python3
"""Live multi-speaker capture for M5 validation.

Records clean self-captures and negative speaker captures through the
actual audio hardware, then builds a multi-speaker validation artifact
that exercises the full M4 + M5 pipeline.

Usage:
  .venv/bin/python scripts/m5_live_capture.py \\
    --input-device 3 --output-device 5 \\
    --self-text "Hello, I am Corollary. This is my voice." \\
    --negative-text "Hello, I am a different speaker." \\
    --negative-voice en-US-GuyNeural \\
    --output-dir data/m5_live_capture_multi
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
    write_scenario_artifact,
    _scenario_metadata,
    _make_plans,
)
from m4_clean_capture import _build_playback_buffer

SR = 16000


async def synthesize_and_capture(
    *,
    text: str,
    voice: str,
    input_device: int,
    output_device: int,
    sr: int = SR,
    output_dir: Path,
    scenario_name: str,
    agent_pcm: np.ndarray | None = None,
    description: str = "",
    negative_pcm: np.ndarray | None = None,
) -> dict:
    """Synthesize TTS, play through speaker, capture through mic."""
    import sounddevice as sd

    print(f"\n=== Recording: {scenario_name} ===")
    print(f"  Voice: {voice}")
    print(f"  Text: {text!r}")

    # Synthesize
    pcm = await synthesize(text, voice=voice)
    acoustic_pcm = np.asarray(pcm, dtype=np.float32)
    print(f"  Synthesized {len(acoustic_pcm)} samples ({len(acoustic_pcm)/sr:.1f}s)")

    # Build playback buffer with lead-in/tail
    play_buffer, lead_in = _build_playback_buffer(acoustic_pcm, sr=sr)
    capture_duration = len(play_buffer) / sr + 1.0
    capture_samples = int(capture_duration * sr)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)

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
        ce = min(cp + frames, len(mic_capture))
        if cp < len(mic_capture):
            n = ce - cp
            mic_capture[cp:ce] = indata[:n, 0]
            capture_pos[0] = ce

    print(f"  Playing and capturing... ({capture_duration:.1f}s)")
    with sd.Stream(
        samplerate=sr,
        blocksize=int(sr * 30 / 1000),
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        time.sleep(capture_duration + 0.5)

    rms = float(np.sqrt(np.mean(mic_capture ** 2)))
    db = 20 * np.log10(max(rms, 1e-9))
    print(f"  Captured {len(mic_capture)} samples, RMS={rms:.4f} ({db:.1f} dB)")

    if rms < 1e-4:
        print("  WARNING: Capture is nearly silent! Check audio devices.")

    # Trim silence from start/end
    # Find first frame above noise floor
    frame_samples = int(sr * 30 / 1000)
    threshold = rms * 0.1
    start_sample = 0
    for i in range(0, len(mic_capture) - frame_samples, frame_samples):
        frame_rms = np.sqrt(np.mean(mic_capture[i:i+frame_samples] ** 2))
        if frame_rms > threshold:
            start_sample = max(0, i - frame_samples)
            break

    end_sample = len(mic_capture)
    for i in range(len(mic_capture) - frame_samples, 0, -frame_samples):
        frame_rms = np.sqrt(np.mean(mic_capture[i:i+frame_samples] ** 2))
        if frame_rms > threshold:
            end_sample = min(len(mic_capture), i + frame_samples)
            break

    trimmed = mic_capture[start_sample:end_sample]

    # Build agent reference (for overlap, mix self + negative)
    if agent_pcm is not None:
        agent_reference = agent_pcm
    else:
        agent_reference = acoustic_pcm

    # Build frame records
    speech_start = lead_in
    speech_end = lead_in + len(acoustic_pcm)

    records = build_frame_records(
        total_samples=len(mic_capture),
        speech_start_sample=speech_start,
        speech_end_sample=speech_end,
        speech_verdict="SELF" if negative_pcm is None else "WORLD",
        sr=sr,
    )

    # Write artifact
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.save(scenario_dir / "agent_pcm.npy", agent_reference.astype(np.float32))
    np.save(scenario_dir / "live_captured.npy", mic_capture.astype(np.float32))

    metadata = {
        "scenario": scenario_name,
        "description": description,
        "sr": sr,
        "input_device": input_device,
        "output_device": output_device,
        "voice": voice,
        "text": text,
        "capture_samples": int(len(mic_capture)),
        "capture_rms": round(rms, 6),
        "capture_db": round(db, 2),
        "speech_start_sample": speech_start,
        "speech_end_sample": speech_end,
        "live_recording": True,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (scenario_dir / "frame_records.json").write_text(json.dumps(records, indent=2))
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata


async def create_overlap_capture(
    *,
    self_pcm: np.ndarray,
    negative_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    sr: int = SR,
    output_dir: Path,
) -> dict:
    """Create an overlap capture with self and negative playing simultaneously."""
    import sounddevice as sd

    print(f"\n=== Recording: overlap ===")
    print(f"  Mixing self + negative speakers")

    # Mix the two signals
    max_len = max(len(self_pcm), len(negative_pcm))
    mixed = np.zeros(max_len, dtype=np.float32)
    mixed[:len(self_pcm)] += self_pcm * 0.7
    mixed[:len(negative_pcm)] += negative_pcm * 0.7

    play_buffer, lead_in = _build_playback_buffer(mixed, sr=sr)
    capture_duration = len(play_buffer) / sr + 1.0
    capture_samples = int(capture_duration * sr)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)

    play_pos = [0]
    capture_pos = [0]

    def callback(indata, outdata, frames, time_info, status):
        pp = play_pos[0]
        ep = min(pp + frames, len(play_buffer))
        outdata[:, 0] = 0
        if pp < len(play_buffer):
            n = ep - pp
            outdata[:n, 0] = play_buffer[pp:ep]
            play_pos[0] = ep
        cp = capture_pos[0]
        ce = min(cp + frames, len(mic_capture))
        if cp < len(mic_capture):
            n = ce - cp
            mic_capture[cp:ce] = indata[:n, 0]
            capture_pos[0] = ce

    print(f"  Playing mixed audio and capturing... ({capture_duration:.1f}s)")
    with sd.Stream(
        samplerate=sr,
        blocksize=int(sr * 30 / 1000),
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        time.sleep(capture_duration + 0.5)

    rms = float(np.sqrt(np.mean(mic_capture ** 2)))
    db = 20 * np.log10(max(rms, 1e-9))
    print(f"  Captured {len(mic_capture)} samples, RMS={rms:.4f} ({db:.1f} dB)")

    # For overlap, both self and negative are playing
    speech_start = lead_in
    speech_end = lead_in + max_len

    records = build_frame_records(
        total_samples=len(mic_capture),
        speech_start_sample=speech_start,
        speech_end_sample=speech_end,
        speech_verdict="BOTH",
        sr=sr,
    )

    scenario_dir = output_dir / "overlap"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.save(scenario_dir / "agent_pcm.npy", self_pcm.astype(np.float32))
    np.save(scenario_dir / "live_captured.npy", mic_capture.astype(np.float32))

    metadata = {
        "scenario": "overlap",
        "description": "Self and negative speakers overlapping",
        "sr": sr,
        "input_device": input_device,
        "output_device": output_device,
        "capture_samples": int(len(mic_capture)),
        "capture_rms": round(rms, 6),
        "capture_db": round(db, 2),
        "live_recording": True,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (scenario_dir / "frame_records.json").write_text(json.dumps(records, indent=2))
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata


async def main(
    *,
    input_device: int,
    output_device: int,
    self_text: str,
    negative_text: str,
    self_voice: str,
    negative_voice: str,
    output_dir: str,
    sr: int = SR,
) -> dict:
    """Run the full multi-speaker capture pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M5 Multi-Speaker Live Capture")
    print("=" * 60)
    print(f"  Input device: {input_device}")
    print(f"  Output device: {output_device}")
    print(f"  Self voice: {self_voice}")
    print(f"  Negative voice: {negative_voice}")
    print(f"  Output dir: {output_dir}")

    # 1. Synthesize both voices
    print("\n[1/4] Synthesizing voices...")
    self_pcm = np.asarray(await synthesize(self_text, voice=self_voice), dtype=np.float32)
    negative_pcm = np.asarray(await synthesize(negative_text, voice=negative_voice), dtype=np.float32)
    print(f"  Self: {len(self_pcm)} samples")
    print(f"  Negative: {len(negative_pcm)} samples")

    # 2. Record self-only capture
    self_meta = await synthesize_and_capture(
        text=self_text,
        voice=self_voice,
        input_device=input_device,
        output_device=output_device,
        sr=sr,
        output_dir=output_path,
        scenario_name="self_live",
        description="Live self voice capture",
    )

    await asyncio.sleep(1.0)  # Pause between recordings

    # 3. Record negative-only capture
    neg_meta = await synthesize_and_capture(
        text=negative_text,
        voice=negative_voice,
        input_device=input_device,
        output_device=output_device,
        sr=sr,
        output_dir=output_path,
        scenario_name="negative_live",
        agent_pcm=self_pcm,  # Agent is still "self" — we're attributing the negative
        description="Live negative voice capture with self as agent",
        negative_pcm=negative_pcm,
    )

    await asyncio.sleep(1.0)

    # 4. Record overlap capture
    overlap_meta = await create_overlap_capture(
        self_pcm=self_pcm,
        negative_pcm=negative_pcm,
        input_device=input_device,
        output_device=output_device,
        sr=sr,
        output_dir=output_path,
    )

    # Write summary
    summary = {
        "output_dir": str(output_path),
        "scenarios": ["self_live", "negative_live", "overlap"],
        "self_voice": self_voice,
        "negative_voice": negative_voice,
        "self_capture": self_meta,
        "negative_capture": neg_meta,
        "overlap_capture": overlap_meta,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("Capture complete!")
    print(f"  Output: {output_dir}")
    print(f"  Scenarios: {summary['scenarios']}")
    print(f"  Self capture: {self_meta['capture_db']:.1f} dB")
    print(f"  Negative capture: {neg_meta['capture_db']:.1f} dB")
    print(f"  Overlap capture: {overlap_meta['capture_db']:.1f} dB")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 multi-speaker live capture")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--self-text", default="Hello, I am Corollary. This is my voice for multi-speaker validation.")
    parser.add_argument("--negative-text", default="Hello, I am a different speaker. This is a negative validation sample.")
    parser.add_argument("--self-voice", default=DEFAULT_VOICE)
    parser.add_argument("--negative-voice", default="en-US-GuyNeural")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"data/m5_live_capture_{time.strftime('%Y%m%d_%H%M%S')}"

    asyncio.run(main(
        input_device=args.input_device,
        output_device=args.output_device,
        self_text=args.self_text,
        negative_text=args.negative_text,
        self_voice=args.self_voice,
        negative_voice=args.negative_voice,
        output_dir=output_dir,
    ))
