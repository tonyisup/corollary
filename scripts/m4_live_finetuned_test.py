#!/usr/bin/env python3
"""M4 Live Test with fine-tuned scorer or prototype bank.

Plays TTS through the speaker, captures on the mic, and checks fresh live audio
against either the promoted fine-tuned scorer or the prototype-bank verifier.

Usage:
    .venv/bin/python scripts/m4_live_finetuned_test.py \
        --input-device 1 \
        --output-device 2 \
        --scorer data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt \
        --duration 10.0 \
        --output /tmp/m4_live_finetuned_test.json

    .venv/bin/python scripts/m4_live_finetuned_test.py \
        --input-device 1 \
        --output-device 2 \
        --bank data/m4_prototype_bank_round4_20260520/prototype_bank.npz \
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

from stg.speaker import (
    SpeakerIdentifier,
    FineTunedSelfVoiceDetector,
    GatedM4Arbitrator,
    PrototypeBankSelfVoiceDetector,
)
from broca.tts import synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 Live Test with fine-tuned scorer or prototype bank")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=2)
    parser.add_argument("--scorer", default=None, help="Path to fine-tuned scorer.pt")
    parser.add_argument("--bank", default=None, help="Path to prototype-bank .npz")
    parser.add_argument(
        "--m4-mode",
        choices=["auto", "bank", "scorer", "arbitration"],
        default="auto",
        help="Force M4 mode: auto, bank-only, scorer-only, or bank+scorer arbitration",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--text", default="Hello, I am Corollary. This is a fresh live test with the fine-tuned scorer.")
    parser.add_argument("--output", default="/tmp/m4_live_finetuned_test.json")
    args = parser.parse_args()

    if not args.scorer and not args.bank:
        raise SystemExit("Provide either --scorer or --bank")
    if args.m4_mode == "bank" and not args.bank:
        raise SystemExit("--m4-mode bank requires --bank")
    if args.m4_mode == "scorer" and not args.scorer:
        raise SystemExit("--m4-mode scorer requires --scorer")
    if args.m4_mode == "arbitration" and (not args.bank or not args.scorer):
        raise SystemExit("--m4-mode arbitration requires both --bank and --scorer")

    detector_kind = {
        "auto": "bank + scorer arbitration" if args.scorer and args.bank else "prototype bank" if args.bank else "fine-tuned scorer",
        "bank": "prototype bank",
        "scorer": "fine-tuned scorer",
        "arbitration": "bank + scorer arbitration",
    }[args.m4_mode]
    print("=" * 60)
    print(f"M4 Live Test — {detector_kind.title()}")
    print("=" * 60)
    effective_mode = {
        "auto": "arbitration" if args.bank and args.scorer else "bank" if args.bank else "scorer",
        "bank": "bank",
        "scorer": "scorer",
        "arbitration": "arbitration",
    }[args.m4_mode]
    print(f"  Effective M4 mode:   {effective_mode}")

    # Load model
    print(f"\nLoading ECAPA model + {detector_kind}...")
    t0 = time.time()
    identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
    bank_detector = None
    scorer_detector = None
    if args.bank and args.m4_mode in {"auto", "bank", "arbitration"}:
        bank_detector = PrototypeBankSelfVoiceDetector(identifier, threshold=args.threshold)
        bank_detector.load_bank(args.bank)
    if args.scorer and args.m4_mode in {"auto", "scorer", "arbitration"}:
        scorer_detector = FineTunedSelfVoiceDetector(identifier, threshold=args.threshold)
        scorer_detector.load_scorer(args.scorer)
    if args.m4_mode == "arbitration" or (args.m4_mode == "auto" and bank_detector is not None and scorer_detector is not None):
        detector = GatedM4Arbitrator(bank_detector=bank_detector, scorer_detector=scorer_detector, threshold=args.threshold)
    else:
        detector = bank_detector if bank_detector is not None else scorer_detector
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
    if hasattr(detector, "register_self"):
        detector.register_self(self_window)
    else:
        detector.add_self_prototype(self_window)
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
            "start_s": float(round(start / sr, 1)),
            "rms_db": float(round(seg_rms, 1)),
            "is_self": bool(is_self),
            "confidence": float(round(conf, 4)),
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
        "effective_m4_mode": effective_mode,
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
