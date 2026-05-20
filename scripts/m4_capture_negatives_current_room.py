#!/usr/bin/env python3
"""Capture negative TTS voices in the current room for M4 training.

Plays multiple TTS voices through the speaker and captures them through the mic.
Each voice is captured separately and saved with metadata.

Usage:
    .venv/bin/python scripts/m4_capture_negatives_current_room.py \\
        --input-device 1 --output-device 3 \\
        --output-dir data/m4_negatives_current_room \\
        --voices GuyNeural ChristopherNeural JennyNeural AriaNeural

    # Default: capture 6 diverse voices
    .venv/bin/python scripts/m4_capture_negatives_current_room.py \\
        --input-device 1 --output-device 3
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

# Default negative voices to capture (available edge-tts voices on this setup).
# DavisNeural and TonyNeural are not consistently available through edge-tts,
# so keep the default set to voices that are known to synthesize reliably here.
DEFAULT_VOICES = [
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
]

CAPTURE_TEXT = (
    "Hello, this is a test recording for speaker identification. "
    "The quick brown fox jumps over the lazy dog. "
    "I am speaking to help train a voice attribution system."
)


async def capture_voice(
    voice: str,
    text: str,
    input_device: int,
    output_device: int,
    sr: int = 16000,
) -> dict:
    """Play a TTS voice and capture it through the mic."""
    import sounddevice as sd

    print(f"\n  Synthesizing {voice}...")
    self_pcm = await synthesize(text, voice=voice)
    self_pcm = np.array(self_pcm, dtype=np.float32)
    duration_s = len(self_pcm) / sr
    print(f"    {len(self_pcm)} samples ({duration_s:.1f}s)")

    # Play and capture
    capture_duration = duration_s + 2.0
    capture_samples = int(capture_duration * sr)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)

    play_pos = [0]

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"    Stream status: {status}")
        pp = play_pos[0]
        total = len(self_pcm)
        if pp < total:
            n = min(frames, total - pp)
            outdata[:n, 0] = self_pcm[pp:pp + n]
            if frames > n:
                outdata[n:, 0] = 0
            play_pos[0] = pp + n
        else:
            outdata[:, 0] = 0
        cp = callback.capture_pos
        n_in = min(frames, capture_samples - cp)
        if n_in > 0:
            mic_capture[cp:cp + n_in] = indata[:n_in, 0]
            callback.capture_pos = cp + n_in

    callback.capture_pos = 0

    print(f"    Playing + capturing ({capture_duration:.1f}s)...")
    with sd.Stream(
        samplerate=sr,
        blocksize=int(sr * 0.03),
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
    print(f"    Captured {len(mic_capture)} samples ({actual_duration:.2f}s), RMS={actual_rms:.4f} ({actual_db:.1f} dB)")

    if actual_rms < 1e-4:
        print("    WARNING: Capture is near-silent!")

    return {
        "voice": voice,
        "text": text,
        "agent_pcm": self_pcm,
        "captured_pcm": mic_capture,
        "duration_s": round(actual_duration, 3),
        "rms": round(actual_rms, 6),
        "db": round(actual_db, 2),
    }


async def main():
    parser = argparse.ArgumentParser(description="Capture negative TTS voices in current room")
    parser.add_argument("--input-device", type=int, default=1,
                        help="Input device index (mic)")
    parser.add_argument("--output-device", type=int, default=3,
                        help="Output device index (speaker)")
    parser.add_argument("--output-dir", type=str, default="data/m4_negatives_current_room",
                        help="Output directory for captures")
    parser.add_argument("--voices", nargs="+", default=DEFAULT_VOICES,
                        help="TTS voices to capture")
    parser.add_argument("--text", type=str, default=CAPTURE_TEXT,
                        help="Text to speak for each voice")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M4 Negative Voice Capture — Current Room")
    print("=" * 60)
    print(f"  Input device:  {args.input_device} (mic)")
    print(f"  Output device: {args.output_device} (speaker)")
    print(f"  Voices:        {len(args.voices)}")
    print(f"  Output dir:    {output_dir}")
    print()

    ts = time.strftime("%Y%m%d_%H%M%S")
    results = []

    for i, voice in enumerate(args.voices):
        print(f"\n[{i+1}/{len(args.voices)}] Capturing {voice}...")
        try:
            result = await capture_voice(
                voice=voice,
                text=args.text,
                input_device=args.input_device,
                output_device=args.output_device,
                sr=args.sr,
            )

            # Save
            voice_safe = voice.replace("-", "_").lower()
            agent_path = output_dir / f"agent_pcm_{voice_safe}_{ts}.npy"
            capture_path = output_dir / f"captured_{voice_safe}_{ts}.npy"
            meta_path = output_dir / f"meta_{voice_safe}_{ts}.json"

            np.save(agent_path, result["agent_pcm"])
            np.save(capture_path, result["captured_pcm"])

            meta = {
                "voice": voice,
                "text": args.text,
                "agent_samples": len(result["agent_pcm"]),
                "capture_samples": len(result["captured_pcm"]),
                "capture_duration_s": result["duration_s"],
                "capture_rms": result["rms"],
                "capture_db": result["db"],
                "sr": args.sr,
                "input_device": args.input_device,
                "output_device": args.output_device,
                "timestamp": ts,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    Saved: {capture_path.name}")
            results.append(meta)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"voice": voice, "error": str(e)})

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary: {len([r for r in results if 'error' not in r])}/{len(args.voices)} captures successful")
    for r in results:
        if "error" in r:
            print(f"  FAIL: {r['voice']} — {r['error']}")
        else:
            print(f"  OK:   {r['voice']} — {r['capture_db']:.1f} dB, {r['capture_duration_s']:.1f}s")
    print()

    # Save summary
    summary_path = output_dir / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({"captures": results, "timestamp": ts}, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
