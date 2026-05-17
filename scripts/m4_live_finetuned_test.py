#!/usr/bin/env python3
"""M4 Live Test with Fine-Tuned Scorer.

Plays TTS through the speaker, captures on the mic, and runs the fine-tuned
scorer to check if the domain gap is closed on fresh live audio.

Usage:
    .venv/bin/python scripts/m4_live_finetuned_test.py \
        --input-device 1 \
        --output-device 2 \
        --scorer data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt \
        --duration 10.0 \
        --output /tmp/m4_live_finetuned_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from stg.speaker import SpeakerIdentifier, FineTunedSelfVoiceDetector
from broca.tts import synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 Live Test with Fine-Tuned Scorer")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=2)
    parser.add_argument("--scorer", required=True, help="Path to fine-tuned scorer.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--text", default="Hello, I am Corollary. This is a fresh live test with the fine-tuned scorer.")
    parser.add_argument("--output", default="/tmp/m4_live_finetuned_test.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Live Test — Fine-Tuned Scorer")
    print("=" * 60)

    # Load model
    print("\nLoading ECAPA model + fine-tuned scorer...")
    t0 = time.time()
    identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
    detector = FineTunedSelfVoiceDetector(identifier, threshold=args.threshold)
    detector.load_scorer(args.scorer)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Synthesize TTS
    print(f"\nSynthesizing TTS: '{args.text[:60]}...'")
    import asyncio
    tts_pcm = asyncio.run(synthesize(args.text, voice="en-US-AriaNeural"))
    tts_pcm = tts_pcm.astype(np.float32)
    print(f"  TTS: {len(tts_pcm)} samples ({len(tts_pcm)/16000:.1f}s)")

    # Register self from TTS (dry — no room coloring)
    # For live test, we register from the TTS itself since we don't have a clean capture
    mid = len(tts_pcm) // 2
    self_window = tts_pcm[mid:mid + 2 * 16000]
    detector.register_self(self_window)
    print(f"  Registered self from TTS (2s window)")

    # Play and capture
    sr = 16000
    capture_duration = args.duration
    print(f"\nPlaying TTS through device {args.output_device}, capturing on device {args.input_device}...")
    print(f"  Duration: {capture_duration}s")

    captured = sd.playrec(
        tts_pcm,
        samplerate=sr,
        channels=1,
        device=(args.input_device, args.output_device),
        blocking=True,
    )
    captured = captured.flatten().astype(np.float32)

    # Also capture silence baseline (last 2 seconds of capture, after TTS ends)
    tts_end_samples = len(tts_pcm)
    silence_start = min(tts_end_samples + int(0.5 * sr), len(captured) - int(2 * sr))
    silence_region = captured[silence_start:silence_start + int(2 * sr)]

    print(f"  Captured {len(captured)} samples ({len(captured)/sr:.1f}s)")
    print(f"  Capture RMS: {20*np.log10(np.sqrt(np.mean(captured**2)) + 1e-10):.1f} dB")

    # Analyze in 2s windows
    seg_samples = int(sr * 2.0)
    hop_samples = int(sr * 0.5)

    print("\n--- Analysis (2s windows, 0.5s hop) ---")
    results = []
    for start in range(0, len(captured) - seg_samples, hop_samples):
        seg = captured[start:start + seg_samples]
        seg_rms = 20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-10)
        is_self, conf = detector.is_self(seg)
        results.append({
            "start_s": round(start / sr, 1),
            "rms_db": round(seg_rms, 1),
            "is_self": is_self,
            "confidence": round(conf, 4),
        })

    # Print results
    speech_results = [r for r in results if r["rms_db"] > -50]
    silence_results = [r for r in results if r["rms_db"] <= -50]

    if speech_results:
        self_count = sum(1 for r in speech_results if r["is_self"])
        avg_conf = np.mean([r["confidence"] for r in speech_results])
        print(f"\n  Speech windows (RMS > -50 dB): {len(speech_results)}")
        print(f"    SELF: {self_count}/{len(speech_results)} ({100*self_count/len(speech_results):.0f}%)")
        print(f"    Avg confidence: {avg_conf:.4f}")
        for r in speech_results:
            label = "SELF" if r["is_self"] else "WORLD"
            print(f"      t={r['start_s']:5.1f}s  rms={r['rms_db']:6.1f}dB  {label}  conf={r['confidence']:.4f}")

    if silence_results:
        self_count_sil = sum(1 for r in silence_results if r["is_self"])
        print(f"\n  Silence windows (RMS <= -50 dB): {len(silence_results)}")
        print(f"    SELF: {self_count_sil}/{len(silence_results)} ({100*self_count_sil/len(silence_results):.0f}%)")

    # Summary
    if speech_results:
        self_pct = 100 * sum(1 for r in speech_results if r["is_self"]) / len(speech_results)
        avg_conf = np.mean([r["confidence"] for r in speech_results])

        print("\n" + "=" * 60)
        print("VERDICT")
        print(f"  Self-recall on live audio: {self_pct:.0f}%")
        print(f"  Avg confidence: {avg_conf:.4f}")

        if self_pct >= 80:
            print("\n  ✅ Domain gap CLOSED — fine-tuned scorer recognizes live self-voice")
        elif self_pct >= 50:
            print("\n  🟡 Partial — some self frames recognized, gap narrowing")
        else:
            print("\n  🔴 Domain gap persists — fine-tuned scorer doesn't generalize")
        print("=" * 60)

    # Save
    output = {
        "scorer": args.scorer,
        "threshold": args.threshold,
        "text": args.text,
        "capture_samples": len(captured),
        "capture_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(captured**2)) + 1e-10)), 1),
        "results": results,
        "summary": {
            "speech_windows": len(speech_results),
            "self_pct": round(self_pct, 1) if speech_results else 0,
            "avg_confidence": round(float(avg_conf), 4) if speech_results else 0,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
