#!/usr/bin/env python3
"""M4 Room-Colored Registration Test.

Compares M4 discrimination when registering with:
  1. Dry TTS (baseline)
  2. Room-colored TTS (via cerebellum forward model)

Tests both against SELF probes (same TTS voice) and WORLD probes (human speech).

Usage:
    .venv/bin/python scripts/m4_room_colored_test.py
"""

import sys, os, json, asyncio, numpy as np
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'packages'))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from broca.tts import synthesize
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
from cerebellum.room_coloring import predict_room_colored_self


async def main():
    print("=" * 70)
    print("M4 ROOM-COLORED REGISTRATION TEST")
    print("=" * 70)

    identifier = SpeakerIdentifier()
    identifier._load_model()
    model_type = "mock" if identifier._model == "mock" else "ecapa-tdnn"
    print(f"Model: {model_type}")

    # ── Find a trained cerebellum model ──
    data_dir = REPO_ROOT / "data"
    forward_models = sorted(data_dir.rglob("forward_model_*.json"))
    if not forward_models:
        print("ERROR: No trained forward models found in data/")
        sys.exit(1)

    # Use the most recent one
    cerebellum_path = str(forward_models[-1])
    print(f"Cerebellum model: {cerebellum_path}")

    # ── Synthesize TTS audio ──
    reg_text = "My name is Corollary. This is my reference voice speaking clearly."
    print(f"\nSynthesizing TTS: {reg_text!r}")
    dry_pcm = await synthesize(reg_text)
    print(f"  Dry PCM: {len(dry_pcm)} samples ({len(dry_pcm)/16000*1000:.0f}ms)")

    # Generate room-colored version
    print("Generating room-colored prediction...")
    room_colored_pcm = predict_room_colored_self(dry_pcm, cerebellum_path)
    print(f"  Room-colored PCM: {len(room_colored_pcm)} samples ({len(room_colored_pcm)/16000*1000:.0f}ms)")

    # ── Build detectors ──
    det_dry = SelfVoiceDetector(identifier, threshold=0.05)
    det_dry.register_self(dry_pcm)

    det_room = SelfVoiceDetector(identifier, threshold=0.05)
    det_room.register_self(room_colored_pcm)

    print(f"\nDetectors registered:")
    print(f"  Dry:       {len(dry_pcm)} samples")
    print(f"  Room-colored: {len(room_colored_pcm)} samples")

    # ── Build test probes ──
    self_texts = {
        "self_same": reg_text,
        "self_diff_1": "I am Corollary, an autonomous agent listening to my environment.",
        "self_diff_2": "The quick brown fox jumps over the lazy dog.",
        "self_short": "Hello.",
    }

    world_files = {
        "world_benchmark": "data/world_speech_benchmark.npy",
        "world_sample": "data/world_speech_sample.npy",
    }

    print("\nSynthesizing probe audio...")
    self_pcms = {}
    for name, text in self_texts.items():
        self_pcms[name] = await synthesize(text)
        print(f"  {name}: {len(self_pcms[name])} samples")

    world_pcms = {}
    for name, path in world_files.items():
        if os.path.exists(path):
            pcm = np.load(path).astype(np.float32)
            max_samples = 8 * 16000
            if len(pcm) > max_samples:
                pcm = pcm[:max_samples]
            world_pcms[name] = pcm
            print(f"  {name}: {len(pcm)} samples ({path})")

    # ── Test both detectors ──
    print()
    print("=" * 70)
    print("RESULTS: Dry TTS Registration")
    print("=" * 70)
    print(f"{'Probe':<25} | {'is_self':<7} | {'confidence':>10} | {'expected':<7} | {'pass'}")
    print("-" * 75)

    dry_results = []
    for name, pcm in {**self_pcms, **world_pcms}.items():
        is_self, conf = det_dry.is_self(pcm)
        expected = "SELF" if name.startswith("self_") else "WORLD"
        passed = (is_self and expected == "SELF") or (not is_self and expected == "WORLD")
        status = "  " if passed else "!!"
        print(f"{name:<25} | {str(is_self):<7} | {conf:>10.4f} | {expected:<7} | {status}")
        dry_results.append({"probe": name, "is_self": is_self, "confidence": conf, "expected": expected, "passed": passed})

    print()
    print("=" * 70)
    print("RESULTS: Room-Colored TTS Registration")
    print("=" * 70)
    print(f"{'Probe':<25} | {'is_self':<7} | {'confidence':>10} | {'expected':<7} | {'pass'}")
    print("-" * 75)

    room_results = []
    for name, pcm in {**self_pcms, **world_pcms}.items():
        is_self, conf = det_room.is_self(pcm)
        expected = "SELF" if name.startswith("self_") else "WORLD"
        passed = (is_self and expected == "SELF") or (not is_self and expected == "WORLD")
        status = "  " if passed else "!!"
        print(f"{name:<25} | {str(is_self):<7} | {conf:>10.4f} | {expected:<7} | {status}")
        room_results.append({"probe": name, "is_self": is_self, "confidence": conf, "expected": expected, "passed": passed})

    # ── Comparison ──
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    dry_correct = sum(1 for r in dry_results if r["passed"])
    room_correct = sum(1 for r in room_results if r["passed"])
    total = len(dry_results)

    print(f"Dry registration:      {dry_correct}/{total} correct ({dry_correct/total*100:.0f}%)")
    print(f"Room-colored registration: {room_correct}/{total} correct ({room_correct/total*100:.0f}%)")

    # Confidence comparison
    print(f"\n{'Probe':<25} | {'Dry Conf':>10} | {'Room Conf':>10} | {'Delta':>8}")
    print("-" * 60)
    for dr, rr in zip(dry_results, room_results):
        delta = rr["confidence"] - dr["confidence"]
        print(f"{dr['probe']:<25} | {dr['confidence']:>10.4f} | {rr['confidence']:>10.4f} | {delta:>+8.4f}")

    # Threshold analysis
    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for label, results in [("Dry", dry_results), ("Room-colored", room_results)]:
        self_confs = [r["confidence"] for r in results if r["expected"] == "SELF"]
        world_confs = [r["confidence"] for r in results if r["expected"] == "WORLD"]

        if self_confs and world_confs:
            gap = min(self_confs) - max(world_confs)
            print(f"\n{label}:")
            print(f"  SELF range:  [{min(self_confs):.4f}, {max(self_confs):.4f}]")
            print(f"  WORLD range: [{min(world_confs):.4f}, {max(world_confs):.4f}]")
            print(f"  Gap: {gap:+.4f}")
            if gap > 0:
                mid = (max(world_confs) + min(self_confs)) / 2
                print(f"  -> Clean separation! Midpoint threshold: {mid:.3f}")
                print(f"  -> Acceptable range: [{max(world_confs):.3f}, {min(self_confs):.3f}]")
            else:
                print(f"  -> OVERLAP — no clean threshold")

    # ── Save results ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_type,
        "cerebellum_model": cerebellum_path,
        "dry_results": dry_results,
        "room_results": room_results,
    }
    out_path = "data/m4_room_colored_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
