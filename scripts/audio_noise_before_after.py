#!/usr/bin/env python3
"""Before/after noise comparison for the PC fan hypothesis.

Runs two silence captures back-to-back and produces a side-by-side comparison:
  1. Fan running (or whatever the current ambient condition is)
  2. Fan off (or modified condition)

Usage:
  .venv/bin/python scripts/audio_noise_before_after.py --input-device 1
  .venv/bin/python scripts/audio_noise_before_after.py --input-device 1 --duration 5 --output /tmp/fan_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "packages"))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

from scripts.audio_noise_profile import capture_audio, profile_noise, interpret, SR


def main() -> None:
    parser = argparse.ArgumentParser(description="Before/after noise comparison")
    parser.add_argument("--input-device", type=int, default=1, help="Mic input device index")
    parser.add_argument("--duration", type=float, default=4.0, help="Capture duration per condition in seconds")
    parser.add_argument("--output", type=str, default="/tmp/noise_before_after.json", help="JSON output path")
    args = parser.parse_args()

    print("=" * 60)
    print("Noise Before/After Comparison")
    print("=" * 60)
    print(f"  Input device: {args.input_device}")
    print(f"  Duration:     {args.duration}s per condition")
    print()

    # Condition A
    print("-" * 60)
    print("CONDITION A — Current state (e.g. fan ON)")
    print("-" * 60)
    input("  Press ENTER when ready to capture condition A...")
    audio_a = capture_audio(device=args.input_device, duration=args.duration, sample_rate=SR)
    stats_a = profile_noise(audio_a, sample_rate=SR)
    stats_a["input_device"] = args.input_device
    stats_a["interpretation"] = interpret(stats_a)

    print()
    print("  Condition A captured.")
    print(f"  RMS: {stats_a['rms_db']} dB | Median: {stats_a['frame_rms_db']['median']} dB")
    top_a = stats_a['spectral']['dominant_peaks_hz'][0] if stats_a['spectral']['dominant_peaks_hz'] else {'hz': '-', 'power_db': '-'}
    print(f"  Top peak: {top_a['hz']} Hz ({top_a['power_db']} dB)")
    print()

    # Condition B
    print("-" * 60)
    print("CONDITION B — Modified state (e.g. fan OFF)")
    print("-" * 60)
    input("  Change the condition now (e.g. turn off fan), then press ENTER...")
    audio_b = capture_audio(device=args.input_device, duration=args.duration, sample_rate=SR)
    stats_b = profile_noise(audio_b, sample_rate=SR)
    stats_b["input_device"] = args.input_device
    stats_b["interpretation"] = interpret(stats_b)

    print()
    print("  Condition B captured.")
    print(f"  RMS: {stats_b['rms_db']} dB | Median: {stats_b['frame_rms_db']['median']} dB")
    top_b = stats_b['spectral']['dominant_peaks_hz'][0] if stats_b['spectral']['dominant_peaks_hz'] else {'hz': '-', 'power_db': '-'}
    print(f"  Top peak: {top_b['hz']} Hz ({top_b['power_db']} dB)")
    print()

    # Comparison
    comparison = {
        "condition_a": stats_a,
        "condition_b": stats_b,
        "delta": {
            "rms_db": round(stats_b["rms_db"] - stats_a["rms_db"], 2),
            "median_frame_db": round(stats_b["frame_rms_db"]["median"] - stats_a["frame_rms_db"]["median"], 2),
            "flatness": round(stats_b["spectral"]["spectral_flatness"] - stats_a["spectral"]["spectral_flatness"], 4),
            "top_peak_hz_a": top_a["hz"],
            "top_peak_hz_b": top_b["hz"],
            "top_peak_db_a": top_a["power_db"],
            "top_peak_db_b": top_b["power_db"],
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(comparison, indent=2))

    d = comparison["delta"]
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  RMS change:          {d['rms_db']:+.2f}  dB  (B - A)")
    print(f"  Median frame change: {d['median_frame_db']:+.2f}  dB  (B - A)")
    print(f"  Flatness change:     {d['flatness']:+.4f}      (B - A)")
    print(f"  Top peak A:          {d['top_peak_hz_a']} Hz @ {d['top_peak_db_a']} dB")
    print(f"  Top peak B:          {d['top_peak_hz_b']} Hz @ {d['top_peak_db_b']} dB")
    print()

    if d["rms_db"] < -3:
        print("  >> Condition B is noticeably quieter — the removed source was a major noise contributor.")
    elif d["rms_db"] < -1:
        print("  >> Condition B is somewhat quieter — the removed source contributed some noise.")
    elif abs(d["rms_db"]) <= 1:
        print("  >> Conditions are similar in level — the removed source may not be the main noise contributor.")
    else:
        print("  >> Condition B is louder — something else changed.")

    if d["top_peak_hz_a"] != d["top_peak_hz_b"]:
        print(f"  >> Dominant peak shifted from {d['top_peak_hz_a']} Hz to {d['top_peak_hz_b']} Hz.")
        if abs(float(d["top_peak_hz_a"]) - float(d["top_peak_hz_b"])) > 500:
            print("     This is a large shift — the removed source had a strong tonal signature.")

    print(f"\n  Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
