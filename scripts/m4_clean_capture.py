#!/usr/bin/env python3
"""Clean self-capture for M4 room-colored self-embedding registration.

Plays a known TTS phrase through the output device, captures it through the mic,
and saves the captured PCM + agent PCM for M4 replay benchmark registration.

Usage:
    .venv/bin/python scripts/m4_clean_capture.py --input-device 3 --output-device 5
    .venv/bin/python scripts/m4_clean_capture.py --input-device 3 --output-device 5 --text "Hello, I am Corollary."
    .venv/bin/python scripts/m4_clean_capture.py --input-device 3 --output-device 5 --duration 10
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

from broca.tts import synthesize


DEFAULT_TEXT = "Hello, I am Corollary. This is a clean self-recording for room-colored embedding registration."
DEFAULT_OUTPUT_DIR = "data/m4_clean_captures"


async def capture_clean_self(
    *,
    text: str,
    input_device: int,
    output_device: int,
    sr: int = 16000,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Play TTS phrase and capture it cleanly through the mic loopback.

    Returns dict with capture metadata.
    """
    import sounddevice as sd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== M4 Clean Self-Capture ===")
    print(f"  Text: {text!r}")
    print(f"  Input device: {input_device}, Output device: {output_device}")

    # 1. Synthesize
    print("\n[1/3] Synthesizing TTS...")
    self_pcm = await synthesize(text)
    self_pcm = np.array(self_pcm, dtype=np.float32)
    duration_s = len(self_pcm) / sr
    print(f"  {len(self_pcm)} samples ({duration_s:.1f}s) at {sr} Hz")

    # 2. Play and capture via sounddevice duplex stream
    capture_duration = duration_s + 2.0  # extra padding
    capture_samples = int(capture_duration * sr)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)

    # The duplex stream runs at `sr`, and Broca synthesize() returns PCM at the
    # same rate. Do not resample the output buffer to a different nominal rate:
    # feeding 48k PCM into a 16k stream plays it at 1/3 speed.
    play_buffer, lead_in = _build_playback_buffer(self_pcm, sr=sr)
    total_play_samples = len(play_buffer)

    play_pos = [0]

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Stream status: {status}")
        # Output
        pp = play_pos[0]
        ep = min(pp + frames, total_play_samples)
        if pp < total_play_samples:
            n = ep - pp
            outdata[:n, 0] = play_buffer[pp:ep]
            if frames > n:
                outdata[n:, 0] = 0
            play_pos[0] = ep
        else:
            outdata[:, 0] = 0
        # Input
        cp = callback.capture_pos
        n_in = min(frames, capture_samples - cp)
        if n_in > 0:
            mic_capture[cp:cp + n_in] = indata[:n_in, 0]
            callback.capture_pos = cp + n_in

    callback.capture_pos = 0

    print(f"\n[2/3] Playing + capturing ({capture_duration:.1f}s)...")
    with sd.Stream(
        samplerate=sr,
        blocksize=int(sr * 0.03),  # 30ms blocks
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        await asyncio.sleep(capture_duration + 0.5)

    # Trim silent padding
    rms = np.sqrt(np.mean(mic_capture ** 2))
    threshold = rms * 0.01
    nonzero = np.where(np.abs(mic_capture) > threshold)[0]
    if len(nonzero) > 0:
        start_idx = max(0, nonzero[0] - int(0.1 * sr))
        end_idx = min(len(mic_capture), nonzero[-1] + int(0.5 * sr))
        mic_capture = mic_capture[start_idx:end_idx]

    actual_duration = len(mic_capture) / sr
    actual_rms = float(np.sqrt(np.mean(mic_capture ** 2)))
    actual_db = 20 * np.log10(max(actual_rms, 1e-9))
    print(f"  Captured {len(mic_capture)} samples ({actual_duration:.2f}s), RMS={actual_rms:.4f} ({actual_db:.1f} dB)")

    if actual_rms < 1e-4:
        print("  WARNING: Capture is near-silent. Check device connections and levels.")

    # 3. Save
    print(f"\n[3/3] Saving to {output_dir}/")
    ts = time.strftime("%Y%m%d_%H%M%S")

    agent_path = output_dir / f"agent_pcm_{ts}.npy"
    capture_path = output_dir / f"captured_{ts}.npy"
    meta_path = output_dir / f"meta_{ts}.json"

    np.save(agent_path, self_pcm.astype(np.float32))
    np.save(capture_path, mic_capture.astype(np.float32))

    meta = {
        "text": text,
        "agent_samples": len(self_pcm),
        "agent_duration_sr": duration_s,
        "capture_samples": len(mic_capture),
        "capture_duration_s": round(actual_duration, 3),
        "capture_rms": round(actual_rms, 6),
        "capture_db": round(actual_db, 2),
        "sr": sr,
        "input_device": input_device,
        "output_device": output_device,
        "timestamp": ts,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Agent PCM: {agent_path}")
    print(f"  Captured:  {capture_path}")
    print(f"  Meta:      {meta_path}")
    print("\nDone.")

    return meta


def _build_playback_buffer(
    self_pcm: np.ndarray,
    *,
    sr: int = 16000,
    lead_in_s: float = 0.3,
    tail_s: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Build the exact-rate playback buffer for clean capture.

    `self_pcm` is already at `sr` because `broca.tts.synthesize()` decodes to
    16 kHz. The returned buffer is also at `sr`, with leading/trailing silence.
    """
    self_pcm = np.asarray(self_pcm, dtype=np.float32)
    lead_in = int(lead_in_s * sr)
    total_play_samples = lead_in + len(self_pcm) + int(tail_s * sr)
    play_buffer = np.zeros(total_play_samples, dtype=np.float32)
    end = lead_in + len(self_pcm)
    play_buffer[lead_in:end] = self_pcm
    return play_buffer, lead_in


def main():
    parser = argparse.ArgumentParser(description="Clean self-capture for M4 room-colored embedding")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = asyncio.run(capture_clean_self(
        text=args.text,
        input_device=args.input_device,
        output_device=args.output_device,
        output_dir=args.output_dir,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
