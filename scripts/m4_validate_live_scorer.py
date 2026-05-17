#!/usr/bin/env python3
"""Quick validation: test fine-tuned scorer against live-domain captures.

Loads the fine-tuned scorer and runs it against the live self + negative captures
to measure whether the domain gap has closed.

Usage:
    .venv/bin/python scripts/m4_validate_live_scorer.py \
        --scorer data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt \
        --self-captures data/m4_live_domain_finetune_20260517/self_captures \
        --neg-captures data/m4_live_domain_finetune_20260517/neg_captures \
        --output /tmp/m4_validate_live_scorer.json
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

from stg.speaker import SpeakerIdentifier, FineTunedSelfVoiceDetector


def extract_segments(pcm: np.ndarray, sr: int = 16000, seg_dur: float = 2.0, hop: float = 1.0):
    """Extract overlapping segments from a long capture."""
    seg_samples = int(sr * seg_dur)
    hop_samples = int(sr * hop)
    segments = []
    for start in range(0, len(pcm) - seg_samples, hop_samples):
        segments.append(pcm[start:start + seg_samples])
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fine-tuned scorer on live captures")
    parser.add_argument("--scorer", required=True, help="Path to fine-tuned scorer.pt")
    parser.add_argument("--self-captures", required=True, help="Dir with self .npy captures")
    parser.add_argument("--neg-captures", required=True, help="Dir with negative .npy captures (or subdirs)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--output", default="/tmp/m4_validate_live_scorer.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Live-Domain Scorer Validation")
    print("=" * 60)

    # Load model
    print("\nLoading ECAPA model...")
    t0 = time.time()
    identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
    detector = FineTunedSelfVoiceDetector(identifier, threshold=args.threshold)
    detector.load_scorer(args.scorer)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load self captures
    self_dir = Path(args.self_captures)
    self_files = sorted(self_dir.glob("captured_*.npy"))
    if not self_files:
        self_files = sorted(self_dir.glob("*.npy"))
    print(f"\nSelf captures: {len(self_files)} files")

    # Load negative captures
    neg_dir = Path(args.neg_captures)
    neg_files = []
    # Check for subdirectories (each containing live_captured.npy)
    for subdir in sorted(neg_dir.iterdir()):
        if subdir.is_dir():
            neg_files.extend(sorted(subdir.glob("live_captured.npy")))
    if not neg_files:
        neg_files = sorted(neg_dir.glob("live_captured.npy"))
    if not neg_files:
        neg_files = sorted(neg_dir.glob("*.npy"))
    print(f"Negative captures: {len(neg_files)} files")

    # Register self from the first self capture
    first_self = np.load(self_files[0]).astype(np.float32)
    # Use a 2s window from the middle of the capture
    sr = 16000
    mid = len(first_self) // 2
    self_window = first_self[mid:mid + 2 * sr]
    detector.register_self(self_window)
    print(f"\nRegistered self from {self_files[0].name} (2s window from middle)")

    # Test against all self captures
    print("\n--- Self captures (should be SELF) ---")
    self_results = []
    for f in self_files:
        pcm = np.load(f).astype(np.float32)
        segments = extract_segments(pcm)
        scores = []
        for seg in segments:
            is_self, conf = detector.is_self(seg)
            scores.append({"is_self": is_self, "confidence": round(conf, 4)})
        avg_conf = np.mean([s["confidence"] for s in scores])
        pct_self = np.mean([s["is_self"] for s in scores]) * 100
        self_results.append({
            "file": f.name,
            "segments": len(scores),
            "avg_confidence": round(avg_conf, 4),
            "pct_self": round(pct_self, 1),
        })
        print(f"  {f.name}: {len(scores)} segs, avg_conf={avg_conf:.4f}, {pct_self:.0f}% SELF")

    # Test against all negative captures
    print("\n--- Negative captures (should be WORLD) ---")
    neg_results = []
    for f in neg_files:
        pcm = np.load(f).astype(np.float32)
        segments = extract_segments(pcm)
        scores = []
        for seg in segments:
            is_self, conf = detector.is_self(seg)
            scores.append({"is_self": is_self, "confidence": round(conf, 4)})
        avg_conf = np.mean([s["confidence"] for s in scores])
        pct_self = np.mean([s["is_self"] for s in scores]) * 100
        neg_results.append({
            "file": f.name,
            "segments": len(scores),
            "avg_confidence": round(avg_conf, 4),
            "pct_self": round(pct_self, 1),
        })
        print(f"  {f.name}: {len(scores)} segs, avg_conf={avg_conf:.4f}, {pct_self:.0f}% SELF")

    # Summary
    all_self_pct = np.mean([r["pct_self"] for r in self_results]) if self_results else 0
    all_neg_pct = np.mean([r["pct_self"] for r in neg_results]) if neg_results else 0
    self_recall = all_self_pct / 100.0
    fp_rate = all_neg_pct / 100.0
    precision = self_recall / (self_recall + fp_rate) if (self_recall + fp_rate) > 0 else 0
    f1 = 2 * precision * self_recall / (precision + self_recall) if (precision + self_recall) > 0 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Self-recall:  {all_self_pct:.1f}%")
    print(f"  FP rate:      {all_neg_pct:.1f}%")
    print(f"  Precision:    {precision:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Threshold:    {args.threshold}")

    if all_self_pct >= 90 and all_neg_pct <= 10:
        print("\n  ✅ Domain gap CLOSED — scorer generalizes to live domain")
    elif all_self_pct >= 70 and all_neg_pct <= 30:
        print("\n  🟡 Partial improvement — gap narrowed but not closed")
    else:
        print("\n  🔴 Domain gap persists — more training data or different approach needed")
    print("=" * 60)

    # Save results
    output = {
        "scorer": args.scorer,
        "threshold": args.threshold,
        "self_results": self_results,
        "neg_results": neg_results,
        "summary": {
            "self_recall_pct": round(all_self_pct, 1),
            "fp_rate_pct": round(all_neg_pct, 1),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
