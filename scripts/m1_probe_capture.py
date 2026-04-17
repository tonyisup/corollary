#!/usr/bin/env python3
"""Hardware-facing M1A primitive probe playback + mic capture.

Purpose:
    Bridge synthetic M1A stimulus work into the first audible real-world path.

What it does:
    1. Generates a simple primitive probe (chirp, vowel, tone, click)
    2. Plays it through a real output device
    3. Records mic input during playback
    4. Reports timing/alignment and basic self-attribution-friendly stats

This is intentionally simpler than the full TTS loop. It is meant to be the
first hardware-facing M1A step, before ASR or lexical evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        factor = max(int(orig_sr / target_sr), 1)
        return audio[::factor].astype(np.float32)


def list_devices() -> None:
    devices = sd.query_devices()
    print("\nAvailable audio devices:\n")
    print(f"{'Index':<8} {'Name':<50} {'In':<4} {'Out':<4} {'Default SR'}")
    print("-" * 84)
    for i, dev in enumerate(devices):
        print(
            f"{i:<8} {dev['name']:<50} {dev['max_input_channels']:<4} "
            f"{dev['max_output_channels']:<4} {dev['default_samplerate']:.0f}"
        )
    print()


def normalized_xcorr(observed: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    if len(observed) < len(template):
        return 0.0, 0
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-9:
        return 0.0, 0
    best_score = -1.0
    best_shift = 0
    step = max(len(template) // 80, 1)
    for shift in range(0, len(observed) - len(template) + 1, step):
        window = observed[shift:shift + len(template)]
        denom = float(np.linalg.norm(window) * template_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def run_probe(
    stimulus: str,
    duration_ms: int,
    amplitude: float,
    input_device: int | None,
    output_device: int | None,
    lead_in_s: float,
    tail_s: float,
) -> dict[str, Any]:
    pcm = generate_stimulus(stimulus, duration_ms, amplitude)
    playback_duration = len(pcm) / SR

    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        native_sr = int(dev_info["default_samplerate"] or SR)
    else:
        native_sr = SR

    capture_duration = lead_in_s + playback_duration + tail_s
    total_samples = int(capture_duration * native_sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    print("\n" + "=" * 60)
    print("  COROLLARY — M1 Primitive Probe Capture")
    print("=" * 60)
    print(f"Stimulus: {stimulus}")
    print(f"Playback duration: {playback_duration:.3f}s")
    print(f"Capture duration:  {capture_duration:.3f}s")
    print(f"Input device:      {input_device if input_device is not None else 'default'}")
    print(f"Output device:     {output_device if output_device is not None else 'default'}")
    if native_sr != SR:
        print(f"Mic native SR:     {native_sr} Hz (will resample to {SR} Hz)")

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=native_sr,
        dtype="float32",
        blocksize=1024,
    )
    stream.start()

    lead_in_samples = int(lead_in_s * native_sr)
    pos = 0
    playback_started_at: float | None = None
    capture_started_at = time.perf_counter()

    while pos < total_samples:
        if playback_started_at is None and pos >= lead_in_samples:
            sd.play(pcm, samplerate=SR, device=output_device, blocking=False)
            playback_started_at = time.perf_counter()
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        audio[pos:pos + n] = data[:n, 0]
        pos += n

    stream.stop()
    stream.close()
    sd.wait()
    capture_finished_at = time.perf_counter()

    if native_sr != SR:
        audio = resample_audio(audio, native_sr, SR)

    observed = audio.astype(np.float32)
    score, shift = normalized_xcorr(observed, pcm)
    observed_rms = float(np.sqrt(np.mean(observed ** 2))) if len(observed) else 0.0
    observed_peak = float(np.max(np.abs(observed))) if len(observed) else 0.0

    expected_shift = int(lead_in_s * SR)
    onset_error_ms = abs(shift - expected_shift) * 1000.0 / SR

    return {
        "stimulus": stimulus,
        "playback_duration_s": round(playback_duration, 4),
        "capture_duration_s": round(capture_duration, 4),
        "capture_rms_db": round(db(observed_rms), 2),
        "capture_peak_db": round(db(observed_peak), 2),
        "match_score": round(score, 6),
        "expected_onset_samples": expected_shift,
        "observed_onset_samples": shift,
        "onset_error_ms": round(onset_error_ms, 4),
        "capture_elapsed_s": round(capture_finished_at - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-facing primitive M1A probe capture")
    parser.add_argument("--stimulus", choices=["chirp", "vowel_ah", "tone", "click"], default="chirp")
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    result = run_probe(
        stimulus=args.stimulus,
        duration_ms=args.duration_ms,
        amplitude=args.amplitude,
        input_device=args.input_device,
        output_device=args.output_device,
        lead_in_s=args.lead_in_s,
        tail_s=args.tail_s,
    )

    print("\nResults:\n")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nInterpretation:")
    if result["capture_rms_db"] < -50:
        print("  ⚠️  Very quiet capture. Check speaker volume, mic gain, and placement.")
    else:
        print("  ✅ Mic captured meaningful signal.")

    if result["match_score"] >= 0.5:
        print("  ✅ Strong probe-template match.")
    elif result["match_score"] >= 0.2:
        print("  ⚠️  Weak but visible probe-template match.")
    else:
        print("  ❌ Probe-template match is poor. Check routing and room setup.")

    print("  Next step: compare chirp vs vowel_ah on the same hardware path.")


if __name__ == "__main__":
    main()
