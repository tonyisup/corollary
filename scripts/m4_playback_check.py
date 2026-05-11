#!/usr/bin/env python3
"""Sanity check for M4 capture artifacts.

Plays back captured audio through the output device and prints diagnostic
metadata so you can verify a capture before committing to fine-tuning.

Usage:
  # Check a clean registration capture
  .venv/bin/python scripts/m4_playback_check.py \
    --capture data/m4_clean_captures_bl2021/captured_20260511_120000.npy \
    --agent-pcm data/m4_clean_captures_bl2021/agent_pcm_20260511_120000.npy \
    --output-device 5

  # Check a validation scenario
  .venv/bin/python scripts/m4_playback_check.py \
    --capture data/m4_capture_validation_bl2021_20260511/self_only/live_captured.npy \
    --agent-pcm data/m4_capture_validation_bl2021_20260511/self_only/agent_pcm.npy \
    --output-device 5

  # Check a human negative recording
  .venv/bin/python scripts/m4_playback_check.py \
    --capture data/m4_human_negative_bl2021/human_negative_speaker1_20260511.npy \
    --output-device 5

  # Just print metadata, don't play
  .venv/bin/python scripts/m4_playback_check.py \
    --capture data/m4_clean_captures_bl2021/captured_20260511_120000.npy \
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

SR = 16000


def describe_pcm(pcm: np.ndarray, label: str, sr: int = SR) -> dict:
    """Compute diagnostic stats for a PCM array."""
    pcm = pcm.astype(np.float32).flatten()
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    db = 20 * np.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(pcm)))
    duration_s = len(pcm) / sr

    # Simple energy-based VAD: find the speech-active region
    frame_len = int(sr * 0.025)  # 25ms frames
    hop = int(sr * 0.010)       # 10ms hop
    energies = []
    for i in range(0, len(pcm) - frame_len, hop):
        frame = pcm[i:i + frame_len]
        energies.append(float(np.sqrt(np.mean(frame ** 2))))

    if energies:
        max_energy = max(energies)
        threshold = max_energy * 0.05
        active_frames = [i for i, e in enumerate(energies) if e > threshold]
        if active_frames:
            speech_start_s = active_frames[0] * hop / sr
            speech_end_s = active_frames[-1] * hop / sr
        else:
            speech_start_s = None
            speech_end_s = None
    else:
        speech_start_s = None
        speech_end_s = None
        max_energy = 0.0

    stats = {
        "label": label,
        "samples": len(pcm),
        "duration_s": round(duration_s, 3),
        "rms": round(rms, 6),
        "rms_db": round(db, 2),
        "peak": round(peak, 6),
        "speech_start_s": round(speech_start_s, 3) if speech_start_s is not None else None,
        "speech_end_s": round(speech_end_s, 3) if speech_end_s is not None else None,
        "max_frame_energy": round(max_energy, 6),
    }
    return stats


def print_stats(stats: dict) -> None:
    """Pretty-print capture stats."""
    print(f"\n  [{stats['label']}]")
    print(f"    Samples:     {stats['samples']} ({stats['duration_s']}s)")
    print(f"    RMS:         {stats['rms']} ({stats['rms_db']} dB)")
    print(f"    Peak:        {stats['peak']}")
    if stats["speech_start_s"] is not None:
        print(f"    Speech region: {stats['speech_start_s']}s — {stats['speech_end_s']}s")
    else:
        print(f"    Speech region: NOT DETECTED (max frame energy: {stats['max_frame_energy']})")


def play_audio(pcm: np.ndarray, device: int, sr: int = SR) -> None:
    """Play mono float32 PCM through the output device."""
    pcm = pcm.astype(np.float32).flatten()
    # Clip to [-1, 1] to avoid distortion
    peak = np.max(np.abs(pcm))
    if peak > 1.0:
        print(f"  WARNING: peak={peak:.3f} > 1.0, clipping to avoid distortion")
        pcm = np.clip(pcm, -1.0, 1.0)

    print(f"  Playing {len(pcm)} samples ({len(pcm)/sr:.1f}s) on device {device}...")
    sd.play(pcm, samplerate=sr, device=device)
    sd.wait()
    print("  Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check for M4 capture artifacts")
    parser.add_argument("--capture", required=True, help="Path to captured audio (.npy)")
    parser.add_argument("--agent-pcm", default=None, help="Path to agent PCM (.npy), if available")
    parser.add_argument("--output-device", type=int, default=5, help="Output device index for playback")
    parser.add_argument("--dry-run", action="store_true", help="Print metadata only, don't play")
    parser.add_argument("--sr", type=int, default=SR, help="Sample rate (default: 16000)")
    args = parser.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"ERROR: capture file not found: {capture_path}")
        return 2

    print(f"=== M4 Capture Sanity Check ===")
    print(f"  Capture: {capture_path}")

    # Load and describe capture
    captured = np.load(str(capture_path))
    capture_stats = describe_pcm(captured, label="Captured (mic)", sr=args.sr)
    print_stats(capture_stats)

    # Warnings
    warnings = []
    if capture_stats["rms_db"] < -50:
        warnings.append("Capture is near-silent — check mic connection and gain")
    if capture_stats["rms_db"] > -10:
        warnings.append("Capture is very loud — risk of clipping/distortion")
    if capture_stats["peak"] > 0.99:
        warnings.append("Capture is clipping — reduce mic gain or increase distance")
    if capture_stats["speech_start_s"] is None:
        warnings.append("No speech region detected — capture may be silence")
    if capture_stats["speech_end_s"] is not None and capture_stats["speech_start_s"] is not None:
        speech_duration = capture_stats["speech_end_s"] - capture_stats["speech_start_s"]
        if speech_duration < 0.5:
            warnings.append(f"Speech region is very short ({speech_duration:.1f}s) — may be too brief for M4")

    if warnings:
        print(f"\n  ⚠ WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    # Load and describe agent PCM if available
    if args.agent_pcm:
        agent_path = Path(args.agent_pcm)
        if agent_path.exists():
            agent = np.load(str(agent_path))
            agent_stats = describe_pcm(agent, label="Agent (dry TTS)", sr=args.sr)
            print_stats(agent_stats)

            # Cross-check: agent should be louder than capture (dry vs room-colored)
            if agent_stats["rms_db"] < capture_stats["rms_db"] - 3:
                warnings.append("Agent PCM is quieter than capture — unusual, check sample rates match")
        else:
            print(f"\n  Agent PCM not found: {agent_path}")

    # Playback
    if not args.dry_run:
        print(f"\n--- Playback ---")
        print("  [You will hear the captured audio in 1 second...]")
        time.sleep(1)
        play_audio(captured, device=args.output_device, sr=args.sr)

        if args.agent_pcm and Path(args.agent_pcm).exists():
            agent = np.load(str(args.agent_pcm))
            print(f"\n  [Now playing the dry agent TTS for comparison...]")
            time.sleep(0.5)
            play_audio(agent, device=args.output_device, sr=args.sr)
    else:
        print(f"\n  (dry-run: skipping playback)")

    # Verdict
    is_ok = len(warnings) == 0
    print(f"\n{'=' * 40}")
    if is_ok:
        print("VERDICT: OK — capture looks healthy")
    else:
        print(f"VERDICT: {len(warnings)} warning(s) — review before proceeding")
    print(f"{'=' * 40}")

    return 0 if is_ok else 1


if __name__ == "__main__":
    sys.exit(main())
