#!/usr/bin/env python3
"""Capture and profile ambient/hardware noise on the live audio path.

Purpose:
- measure silence/noise floor on the current microphone path
- summarize frame-wise RMS distribution for threshold calibration
- identify dominant spectral peaks (e.g. 60 Hz hum, broadband hiss)
- save a JSON artifact for the current hardware/room setup

Examples:
  .venv/bin/python scripts/audio_noise_profile.py --input-device 1
  .venv/bin/python scripts/audio_noise_profile.py --input-device 1 --duration 5 --output /tmp/noise_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)


def capture_audio(device: int, duration: float, sample_rate: int = SR) -> np.ndarray:
    print(f"\nCapturing ambient audio for {duration:.1f}s from input device {device}...")
    print("Please remain silent; do not play TTS or world audio during capture.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def db(x: float) -> float:
    if x <= 0:
        return float("-inf")
    return 20.0 * np.log10(x)


def frame_audio(audio: np.ndarray, frame_samples: int = FRAME_SAMPLES) -> np.ndarray:
    n = len(audio) // frame_samples
    if n == 0:
        return np.zeros((0, frame_samples), dtype=np.float32)
    trimmed = audio[: n * frame_samples]
    return trimmed.reshape(n, frame_samples)


def spectral_profile(audio: np.ndarray, sample_rate: int = SR) -> dict:
    if len(audio) == 0:
        return {
            "dominant_peaks_hz": [],
            "band_rms_db": {},
            "spectral_flatness": None,
        }

    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2

    # Ignore DC for peak finding.
    valid = freqs >= 20.0
    freqs = freqs[valid]
    power = power[valid]

    peak_idx = np.argsort(power)[-8:][::-1]
    dominant_peaks = [
        {
            "hz": round(float(freqs[i]), 1),
            "power_db": round(db(float(np.sqrt(power[i]))), 2),
        }
        for i in peak_idx if power[i] > 0
    ]

    def band_rms(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return float("-inf")
        return db(float(np.sqrt(np.mean(power[mask]))))

    band_rms_db = {
        "20_120_hz": round(band_rms(20, 120), 2),
        "120_300_hz": round(band_rms(120, 300), 2),
        "300_1000_hz": round(band_rms(300, 1000), 2),
        "1000_4000_hz": round(band_rms(1000, 4000), 2),
        "4000_8000_hz": round(band_rms(4000, 8000), 2),
    }

    positive_power = power[power > 0]
    if len(positive_power) == 0:
        flatness = None
    else:
        geometric = float(np.exp(np.mean(np.log(positive_power))))
        arithmetic = float(np.mean(positive_power))
        flatness = geometric / arithmetic if arithmetic > 0 else None

    return {
        "dominant_peaks_hz": dominant_peaks,
        "band_rms_db": band_rms_db,
        "spectral_flatness": round(float(flatness), 4) if flatness is not None else None,
    }


def profile_noise(audio: np.ndarray, sample_rate: int = SR) -> dict:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    dc_offset = float(np.mean(audio)) if len(audio) else 0.0
    clipping_pct = float(np.mean(np.abs(audio) > 0.99) * 100.0) if len(audio) else 0.0

    frames = frame_audio(audio)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1)) if len(frames) else np.array([])
    frame_rms_db = np.array([db(float(x)) for x in frame_rms], dtype=np.float64) if len(frame_rms) else np.array([])

    spectral = spectral_profile(audio, sample_rate=sample_rate)

    return {
        "samples": int(len(audio)),
        "duration_s": round(len(audio) / sample_rate, 3),
        "sample_rate": sample_rate,
        "rms": round(rms, 6),
        "rms_db": round(db(rms), 2),
        "peak": round(peak, 6),
        "peak_db": round(db(peak), 2) if peak > 0 else float("-inf"),
        "dc_offset": round(dc_offset, 6),
        "clipping_pct": round(clipping_pct, 4),
        "frame_ms": FRAME_MS,
        "frame_count": int(len(frame_rms_db)),
        "frame_rms_db": {
            "min": round(float(np.min(frame_rms_db)), 2) if len(frame_rms_db) else None,
            "p10": round(float(np.percentile(frame_rms_db, 10)), 2) if len(frame_rms_db) else None,
            "median": round(float(np.median(frame_rms_db)), 2) if len(frame_rms_db) else None,
            "p90": round(float(np.percentile(frame_rms_db, 90)), 2) if len(frame_rms_db) else None,
            "max": round(float(np.max(frame_rms_db)), 2) if len(frame_rms_db) else None,
            "std": round(float(np.std(frame_rms_db)), 2) if len(frame_rms_db) else None,
        },
        "suggested_energy_threshold_db": round(float(np.median(frame_rms_db) + 15.0), 2) if len(frame_rms_db) else None,
        "spectral": spectral,
    }


def interpret(stats: dict) -> list[str]:
    notes: list[str] = []
    median_db = stats["frame_rms_db"]["median"]
    flatness = stats["spectral"]["spectral_flatness"]
    peaks = stats["spectral"]["dominant_peaks_hz"]

    if median_db is not None:
        if median_db > -45:
            notes.append("Noise floor is high for a silence capture; live speech gating may be contaminated without calibration.")
        elif median_db > -55:
            notes.append("Noise floor is moderate; dynamic energy-threshold calibration is recommended for live runs.")
        else:
            notes.append("Noise floor is relatively low; static is less likely to dominate gating.")

    if peaks:
        top_hz = peaks[0]["hz"]
        if 50 <= top_hz <= 70 or 110 <= top_hz <= 130:
            notes.append("Dominant low-frequency peak is consistent with mains hum / power contamination.")
        elif top_hz >= 3000:
            notes.append("Dominant high-frequency peak suggests hiss/static rather than hum.")

    if flatness is not None:
        if flatness > 0.3:
            notes.append("Spectrum is relatively flat; this looks more like broadband hiss/static than a narrow tonal contaminant.")
        elif flatness < 0.1:
            notes.append("Spectrum is peaky/tonal; investigate electrical hum, fan tones, or a specific hardware source.")

    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ambient/static noise on the current audio input path")
    parser.add_argument("--input-device", type=int, default=1, help="Mic input device index")
    parser.add_argument("--duration", type=float, default=4.0, help="Silence capture duration in seconds")
    parser.add_argument("--output", type=str, default="/tmp/audio_noise_profile.json", help="JSON output path")
    args = parser.parse_args()

    audio = capture_audio(device=args.input_device, duration=args.duration, sample_rate=SR)
    stats = profile_noise(audio, sample_rate=SR)
    stats["input_device"] = args.input_device
    stats["interpretation"] = interpret(stats)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(stats, indent=2))

    print("\n============================================================")
    print("Noise Profile")
    print("============================================================")
    print(f"Input device:                 {args.input_device}")
    print(f"Duration:                     {stats['duration_s']}s")
    print(f"RMS level:                    {stats['rms_db']} dB")
    print(f"Peak level:                   {stats['peak_db']} dB")
    print(f"Median frame floor:           {stats['frame_rms_db']['median']} dB")
    print(f"Suggested energy threshold:   {stats['suggested_energy_threshold_db']} dB")
    print(f"Spectral flatness:            {stats['spectral']['spectral_flatness']}")
    print("Dominant spectral peaks:")
    for peak in stats['spectral']['dominant_peaks_hz'][:5]:
        print(f"  - {peak['hz']} Hz ({peak['power_db']} dB)")
    if stats['interpretation']:
        print("Interpretation:")
        for note in stats['interpretation']:
            print(f"  - {note}")
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
