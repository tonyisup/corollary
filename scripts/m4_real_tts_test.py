#!/usr/bin/env python3
"""M4 Real TTS Test — register self with Broca TTS, probe with same + different texts.

Tests whether ECAPA-TDNN can distinguish:
  1. Same TTS voice, same text (should be SELF — high confidence)
  2. Same TTS voice, different text (should be SELF — tests content independence)
  3. Different speaker / noise (should be WORLD)

Usage:
    .venv/bin/python scripts/m4_real_tts_test.py
"""

import sys, os, json, asyncio, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector


async def main():
    print("=" * 70)
    print("M4 REAL TTS TEST")
    print("=" * 70)

    identifier = SpeakerIdentifier()
    identifier._load_model()
    model_type = "mock" if identifier._model == "mock" else "ecapa-tdnn"
    print(f"Model: {model_type}")
    print()

    # ── Registration: Broca TTS self template ──
    registration_text = "My name is Corollary. This is my reference voice speaking clearly."
    print(f"[1/4] Synthesizing registration audio: {registration_text!r}")
    reg_pcm = await synthesize(registration_text)
    print(f"  -> {len(reg_pcm)} samples ({len(reg_pcm)/16000*1000:.0f}ms)")

    # ── Test probes ──
    probes = {
        "self_same_text": "My name is Corollary. This is my reference voice speaking clearly.",
        "self_diff_text_1": "I am Corollary, an autonomous agent listening to my environment.",
        "self_diff_text_2": "The quick brown fox jumps over the lazy dog.",
        "self_diff_text_3": "Hello world, this is a test of my voice recognition system.",
        "self_short": "Hello.",
        "self_long": "This is a much longer sentence that tests whether the embedding "
                       "stability holds across different durations and phonetic content.",
    }

    print("\n[2/4] Synthesizing probe audio...")
    probe_pcms = {}
    for name, text in probes.items():
        pcm = await synthesize(text)
        probe_pcms[name] = pcm
        print(f"  {name}: {len(pcm)} samples ({len(pcm)/16000*1000:.0f}ms) — {text[:60]}...")

    # ── Register self ──
    print("\n[3/4] Registering self template...")
    detector = SelfVoiceDetector(identifier, threshold=0.05)
    detector.register_self(reg_pcm)
    print(f"  Self embedding computed from registration audio")

    # ── Run all probes ──
    print("\n[4/4] Running probes...")
    print()
    print(f"{'Probe':<25} | {'is_self':<7} | {'confidence':>10} | {'expected':<7} | {'pass':<4} | notes")
    print("-" * 90)

    detailed_results = []

    for name, pcm in probe_pcms.items():
        is_self, conf = detector.is_self(pcm)

        if name.startswith("self_"):
            expected = "SELF"
            passed = is_self
        else:
            expected = "WORLD"
            passed = not is_self

        notes = ""
        if name == "self_same_text":
            notes = "(identical to registration)"
        elif name == "self_short":
            notes = "short utterance"
        elif name == "self_long":
            notes = "long utterance"

        status = "  " if passed else "!!"
        print(f"{name:<25} | {str(is_self):<7} | {conf:>10.4f} | {expected:<7} | {status:<4} | {notes}")

        detailed_results.append({
            "probe": name,
            "text": probes[name],
            "samples": len(pcm),
            "duration_ms": len(pcm) / 16000 * 1000,
            "is_self": is_self,
            "confidence": conf,
            "expected": expected,
            "passed": passed,
        })

    # ── Confidence analysis ──
    self_confs = [r["confidence"] for r in detailed_results if r["expected"] == "SELF"]
    world_confs = [r["confidence"] for r in detailed_results if r["expected"] == "WORLD"]

    print()
    print("=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    if self_confs:
        print(f"SELF confidences:  min={min(self_confs):.4f}, max={max(self_confs):.4f}, "
              f"mean={np.mean(self_confs):.4f}")
    if world_confs:
        print(f"WORLD confidences: min={min(world_confs):.4f}, max={max(world_confs):.4f}, "
              f"mean={np.mean(world_confs):.4f}")

    if self_confs and world_confs:
        gap = min(self_confs) - max(world_confs)
        print(f"Gap (min_self - max_world): {gap:+.4f}")
        if gap > 0:
            print(f"  -> Clean separation! Optimal threshold range: {max(world_confs):.3f} - {min(self_confs):.3f}")
        else:
            print(f"  -> OVERLAP! No clean threshold exists for this registration/probe pair.")

    # ── Threshold recommendation from real TTS data ──
    print()
    print("=" * 70)
    print("THRESHOLD RECOMMENDATION (from real TTS)")
    print("=" * 70)

    if self_confs:
        # Find threshold that accepts all SELF and rejects all WORLD
        all_confs = [(r["confidence"], r["expected"]) for r in detailed_results]
        all_confs.sort(key=lambda x: x[0])

        best_th = None
        best_acc = 0
        for th in np.arange(0.0, 1.0, 0.01):
            correct = sum(1 for c, e in all_confs
                         if (e == "SELF" and c >= th) or (e == "WORLD" and c < th))
            acc = correct / len(all_confs)
            if acc > best_acc:
                best_acc = acc
                best_th = th

        if best_th is not None:
            print(f"  Best threshold: {best_th:.2f} (accuracy: {best_acc:.0%})")

        # Acceptance range
        if self_confs:
            print(f"  SELF acceptance range: {min(self_confs):.3f} - {max(self_confs):.3f}")
            # Conservative: mean - 2*std
            mean_self = np.mean(self_confs)
            std_self = np.std(self_confs) if len(self_confs) > 1 else 0.05
            conservative_th = mean_self - 2 * std_self
            print(f"  Conservative threshold (mean-2*std): {conservative_th:.3f}")

    # ── Save results ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_type,
        "registration_text": registration_text,
        "registration_samples": len(reg_pcm),
        "registration_duration_ms": len(reg_pcm) / 16000 * 1000,
        "results": detailed_results,
    }

    out_path = "data/m4_real_tts_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
