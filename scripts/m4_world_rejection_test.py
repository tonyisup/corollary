#!/usr/bin/env python3
"""M4 World Rejection Test — verify M4 rejects non-self (world) audio.

Uses existing world speech samples from the data directory as WORLD probes,
and Broca TTS as SELF. Tests the full discrimination pipeline.

Usage:
    .venv/bin/python scripts/m4_world_rejection_test.py
"""

import sys, os, json, asyncio, numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector


async def main():
    print("=" * 70)
    print("M4 WORLD REJECTION TEST")
    print("=" * 70)

    identifier = SpeakerIdentifier()
    identifier._load_model()
    model_type = "mock" if identifier._model == "mock" else "ecapa-tdnn"
    print(f"Model: {model_type}")
    print()

    # ── Register self with Broca TTS ──
    reg_text = "My name is Corollary. This is my reference voice speaking clearly."
    print(f"Registering self: {reg_text!r}")
    reg_pcm = await synthesize(reg_text)
    print(f"  -> {len(reg_pcm)} samples ({len(reg_pcm)/16000*1000:.0f}ms)")

    # ── Find world speech samples ──
    data_dir = Path("data")
    world_files = sorted(data_dir.glob("world_speech*.npy"))
    if not world_files:
        # Try to find any world audio
        world_files = sorted(data_dir.rglob("*world*.npy"))

    print(f"\nFound {len(world_files)} world speech file(s)")

    # Also check for world speech in calibration dirs
    world_calibration = data_dir / "m3_broca_self_calibration_20260430_quick"
    if world_calibration.exists():
        extra = list(world_calibration.rglob("world*.npy"))
        world_files.extend(extra)
        print(f"  + {len(extra)} from calibration dir")

    # ── Build test set ──
    tests = []

    # SELF probes (TTS)
    self_texts = {
        "self_same": reg_text,
        "self_diff_1": "I am Corollary, listening to my surroundings.",
        "self_diff_2": "The weather is pleasant today.",
        "self_short": "Hello.",
    }
    for name, text in self_texts.items():
        pcm = await synthesize(text)
        tests.append({"name": name, "pcm": pcm, "expected": "SELF", "text": text[:60]})

    # WORLD probes (from files)
    for wf in world_files[:5]:  # max 5 world samples
        try:
            pcm = np.load(wf)
            if pcm.dtype != np.float32:
                pcm = pcm.astype(np.float32)
            # Trim to reasonable length (max 8s)
            max_samples = 8 * 16000
            if len(pcm) > max_samples:
                pcm = pcm[:max_samples]
            tests.append({
                "name": f"world_{wf.stem}",
                "pcm": pcm,
                "expected": "WORLD",
                "text": str(wf.relative_to(data_dir)),
            })
        except Exception as e:
            print(f"  WARNING: Could not load {wf}: {e}")

    # Also add noise/silence as WORLD
    tests.append({
        "name": "world_silence",
        "pcm": np.zeros(16000, dtype=np.float32),
        "expected": "WORLD",
        "text": "silence",
    })
    tests.append({
        "name": "world_noise",
        "pcm": np.random.randn(16000).astype(np.float32) * 0.1,
        "expected": "WORLD",
        "text": "white noise",
    })

    print(f"\nTest set: {len(tests)} probes "
          f"({sum(1 for t in tests if t['expected']=='SELF')} SELF, "
          f"{sum(1 for t in tests if t['expected']=='WORLD')} WORLD)")

    # ── Run tests at multiple thresholds ──
    print()
    print("=" * 70)
    print("DISCRIMINATION RESULTS")
    print("=" * 70)

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]

    # Pre-compute all embeddings
    print("\nComputing embeddings...")
    for t in tests:
        t["embedding"] = identifier.get_embedding(t["pcm"])
    reg_embedding = identifier.get_embedding(reg_pcm)

    # Compute similarities to self
    for t in tests:
        n1 = np.linalg.norm(reg_embedding)
        n2 = np.linalg.norm(t["embedding"])
        t["similarity"] = float(np.dot(reg_embedding, t["embedding"]) / (n1 * n2 + 1e-10))

    # Print similarity table
    print(f"\n{'Probe':<30} | {'Expected':<7} | {'Similarity':>10} | {'Duration':>8}")
    print("-" * 65)
    for t in tests:
        dur_ms = len(t["pcm"]) / 16000 * 1000
        print(f"{t['name']:<30} | {t['expected']:<7} | {t['similarity']:>10.4f} | {dur_ms:>7.0f}ms")

    # Threshold sweep
    print()
    print(f"{'Threshold':>10} | {'Self OK':>7} | {'World OK':>8} | {'Accuracy':>9} | {'Status'}")
    print("-" * 60)

    best_th = None
    best_acc = 0

    for th in thresholds:
        self_ok = all(t["similarity"] >= th for t in tests if t["expected"] == "SELF")
        world_ok = all(t["similarity"] < th for t in tests if t["expected"] == "WORLD")
        correct = sum(1 for t in tests
                     if (t["expected"] == "SELF" and t["similarity"] >= th) or
                        (t["expected"] == "WORLD" and t["similarity"] < th))
        acc = correct / len(tests)

        if self_ok and world_ok:
            status = "PERFECT"
            if best_th is None:
                best_th = th
        elif self_ok:
            status = "SELF_OK"
            if best_th is None:
                best_th = th
        elif not self_ok:
            status = "SELF_FAIL"
        else:
            status = "MIXED"

        if acc > best_acc and self_ok:
            best_acc = acc

        marker = " <--" if status in ("PERFECT", "SELF_OK") and best_th == th else ""
        print(f"{th:>10.2f} | {str(self_ok):>7} | {str(world_ok):>8} | {acc:>8.1%} | {status}{marker}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    self_sims = [t["similarity"] for t in tests if t["expected"] == "SELF"]
    world_sims = [t["similarity"] for t in tests if t["expected"] == "WORLD"]

    if self_sims:
        print(f"SELF similarities:  min={min(self_sims):.4f}, max={max(self_sims):.4f}, "
              f"mean={np.mean(self_sims):.4f}")
    if world_sims:
        print(f"WORLD similarities: min={min(world_sims):.4f}, max={max(world_sims):.4f}, "
              f"mean={np.mean(world_sims):.4f}")

    if self_sims and world_sims:
        gap = min(self_sims) - max(world_sims)
        print(f"\nSeparation gap: {gap:+.4f}")
        if gap > 0:
            print(f"  -> Clean separation! Threshold range: [{max(world_sims):.3f}, {min(self_sims):.3f}]")
            mid = (max(world_sims) + min(self_sims)) / 2
            print(f"  -> Midpoint threshold: {mid:.3f}")
        else:
            print(f"  -> OVERLAP between SELF and WORLD distributions")
            print(f"  -> No single threshold achieves perfect separation")

    if best_th is not None:
        print(f"\nRECOMMENDED THRESHOLD: {best_th}")
    else:
        print(f"\nNo threshold achieved perfect separation.")
        if self_sims and world_sims:
            # Find best compromise
            for th in np.arange(0.0, 1.0, 0.01):
                self_acc = sum(1 for s in self_sims if s >= th) / len(self_sims)
                world_acc = sum(1 for s in world_sims if s < th) / len(world_sims)
                if self_acc == 1.0:
                    print(f"  Best threshold accepting all SELF: {th:.2f} (world acc: {world_acc:.0%})")
                    break

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_type,
        "recommended_threshold": best_th,
        "tests": [{"name": t["name"], "expected": t["expected"],
                    "similarity": t["similarity"], "text": t["text"]} for t in tests],
    }
    out_path = "data/m4_world_rejection_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
