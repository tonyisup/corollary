#!/usr/bin/env python3
"""Auto-calibration bootstrap: fit room filter at startup.

Plays a known TTS phrase through the output device, captures it through the mic,
fits a room filter, and exports it for production use.

Usage:
    .venv/bin/python3 scripts/m3_auto_calibrate.py --input-device 3 --output-device 5
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

from broca.tts import synthesize
from m3_broca_room_reference_analysis import fit_room_filter, apply_room_filter, local_score, dry_template_track

DEFAULT_CAL_TEXT = "I was speaking, then paused."
DEFAULT_FILTER_PATH = "data/m3_broca_room_filter_production.npy"
DEFAULT_META_PATH = "data/m3_broca_room_filter_production.json"


async def calibrate(
    *,
    text: str,
    input_device: int,
    output_device: int,
    sr: int = 16000,
    tts_sr: int = 48000,
    taps: int = 64,
    ridge: float = 0.000112,
    search_ms: float = 256.4,
    filter_path: str | Path = DEFAULT_FILTER_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
    threshold_fraction: float = 0.5,
    min_threshold: float = 0.010,
    max_threshold: float = 0.200,
) -> dict:
    """Run auto-calibration bootstrap.

    Returns dict with calibration results.
    """
    import sounddevice as sd

    print("=== M3 Auto-Calibration Bootstrap ===")

    # 1. Synthesize calibration phrase
    print(f"\n[1/4] Synthesizing: {text!r}")
    self_pcm = await synthesize(text)
    self_pcm = np.array(self_pcm, dtype=np.float32)
    duration_s = len(self_pcm) / sr
    print(f"  {len(self_pcm)} samples ({duration_s:.1f}s)")

    # 2. Play and capture
    print(f"\n[2/4] Playing through device {output_device}, capturing from device {input_device}...")
    total_samples = int((duration_s + 1.0) * tts_sr)
    tts_start = int(0.5 * tts_sr)  # 0.5s lead-in
    tts_buffer = np.zeros(total_samples, dtype=np.float32)
    tts_end = min(tts_start + len(self_pcm) * tts_sr // sr, total_samples)
    # Resample self_pcm to tts_sr for playback
    from scipy.signal import resample as scipy_resample
    tts_output = scipy_resample(self_pcm, int(len(self_pcm) * tts_sr / sr)).astype(np.float32)
    actual_end = min(tts_start + len(tts_output), total_samples)
    tts_buffer[tts_start:actual_end] = tts_output[:actual_end - tts_start]

    capture_samples = int((duration_s + 1.5) * sr)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)
    samples_played = [0]
    samples_captured = [0]

    def audio_callback(indata, outdata, frames, time_info, status):
        if start := samples_played[0]:
            pass  # noqa: avoid unused warning
        sp = samples_played[0]
        ep = min(sp + frames, total_samples)
        if sp < total_samples:
            n = ep - sp
            outdata[:n, 0] = tts_buffer[sp:ep]
            if n < frames:
                outdata[n:, 0] = 0
        else:
            outdata[:] = 0
        samples_played[0] = ep

        cap_start = samples_captured[0]
        cap_end = min(cap_start + frames, capture_samples)
        if cap_start < capture_samples:
            n = cap_end - cap_start
            mic_capture[cap_start:cap_end] = indata[:n, 0]
            samples_captured[0] = cap_end

    try:
        with sd.Stream(
            samplerate=tts_sr,
            blocksize=int(480 * tts_sr / sr),
            dtype="float32",
            channels=1,
            callback=audio_callback,
            device=(input_device, output_device),
        ):
            await asyncio.sleep(duration_s + 1.0)
    except Exception as e:
        print(f"  ERROR: Audio stream failed: {e}")
        return {"success": False, "error": str(e)}

    # Trim capture
    mic_capture = mic_capture[:samples_captured[0]]
    print(f"  Captured {len(mic_capture)} samples")

    if np.max(np.abs(mic_capture)) < 1e-6:
        print("  ERROR: Capture is silent — check mic connection")
        return {"success": False, "error": "silent_capture"}

    # 3. Find expected shift (cross-correlation peak)
    print(f"\n[3/4] Fitting room filter...")
    # Use global xcorr to find the shift
    from m2_speech_preservation_harness import normalized_xcorr
    global_score, global_shift = normalized_xcorr(mic_capture, self_pcm)
    print(f"  Global xcorr: score={global_score:.4f} shift={global_shift} samples ({global_shift * 1000 / sr:.1f}ms)")

    # Fit room filter
    h = fit_room_filter(mic_capture, self_pcm, global_shift, taps=taps, ridge=ridge)

    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
        print("  ERROR: Filter has NaN/Inf values")
        return {"success": False, "error": "invalid_filter"}

    print(f"  Filter: {len(h)} taps, sum={h.sum():.4f}, max={h.max():.4f}")

    # 4. Score the calibration capture
    room_ref = apply_room_filter(self_pcm, h)
    room_score, best_shift = local_score(mic_capture, room_ref, global_shift, search_ms=search_ms)
    dry_score, _ = local_score(mic_capture, dry_template_track(self_pcm, len(mic_capture), global_shift), global_shift, search_ms=search_ms)

    print(f"\n[4/4] Validation:")
    print(f"  Room-local score: {room_score:.4f}")
    print(f"  Dry score: {dry_score:.4f}")
    print(f"  Improvement: {room_score / max(dry_score, 1e-6):.1f}x")

    if room_score < 0.01:
        print("  WARNING: Room-local score is very low — room filter may not be effective")
        print("  Using conservative threshold")

    # Compute threshold
    threshold = max(min_threshold, room_score * threshold_fraction)
    threshold = min(threshold, max_threshold)
    print(f"  Recommended threshold: {threshold:.4f} ({threshold_fraction:.0%} of self-score)")

    # 5. Export
    filter_path = Path(filter_path)
    meta_path = Path(meta_path)
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(filter_path, h)
    meta = {
        "calibration_text": text,
        "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "room_local_score": float(room_score),
        "dry_score": float(dry_score),
        "recommended_threshold": float(threshold),
        "threshold_fraction": threshold_fraction,
        "filter_taps": int(taps),
        "filter_ridge": float(ridge),
        "global_xcorr_score": float(global_score),
        "global_xcorr_shift": int(global_shift),
        "capture_samples": int(len(mic_capture)),
        "success": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  Filter exported: {filter_path}")
    print(f"  Metadata exported: {meta_path}")

    return {
        "success": True,
        "room_local_score": float(room_score),
        "threshold": float(threshold),
        "filter_path": str(filter_path),
    }


async def main():
    parser = argparse.ArgumentParser(description="M3 Auto-Calibration Bootstrap")
    parser.add_argument("--text", default=DEFAULT_CAL_TEXT, help="Calibration phrase")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.000112)
    parser.add_argument("--filter-path", default=DEFAULT_FILTER_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    args = parser.parse_args()

    result = await calibrate(
        text=args.text,
        input_device=args.input_device,
        output_device=args.output_device,
        taps=args.taps,
        ridge=args.ridge,
        filter_path=args.filter_path,
        meta_path=args.meta_path,
        threshold_fraction=args.threshold_fraction,
    )

    if result["success"]:
        print(f"\n=== Calibration complete ===")
        print(f"  Room-local score: {result['room_local_score']:.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Filter: {result['filter_path']}")
    else:
        print(f"\n=== Calibration FAILED ===")
        print(f"  Error: {result.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
