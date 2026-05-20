#!/usr/bin/env python3
"""Validate an M4 prototype-bank self-voice detector on capture sets.

This compares a multi-prototype cosine-margin detector against the live-domain
and same-room validation captures.
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

from stg.speaker import SpeakerIdentifier, PrototypeBankSelfVoiceDetector

SR = 16000
SEG_DUR_S = 2.0
HOP_S = 1.0
SEG_SAMPLES = int(SR * SEG_DUR_S)
HOP_SAMPLES = int(SR * HOP_S)


def extract_segments(pcm: np.ndarray) -> list[np.ndarray]:
    segments = []
    for start in range(0, len(pcm) - SEG_SAMPLES, HOP_SAMPLES):
        segments.append(pcm[start : start + SEG_SAMPLES].astype(np.float32))
    return segments


def load_self_captures(directory: str) -> list[Path]:
    base = Path(directory)
    files = sorted(base.glob("captured_*.npy"))
    return files if files else sorted(base.glob("*.npy"))


def load_neg_captures(directory: str) -> list[Path]:
    base = Path(directory)
    files: list[Path] = []
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            files.extend(sorted(subdir.glob("live_captured.npy")))
    if not files:
        files = sorted(base.glob("live_captured.npy"))
    if not files:
        files = sorted(base.glob("*.npy"))
    return files


def build_detector(bank_path: str, backend: str) -> PrototypeBankSelfVoiceDetector:
    identifier = SpeakerIdentifier(use_real=True, backend=backend)
    detector = PrototypeBankSelfVoiceDetector(identifier)
    detector.load_bank(bank_path)
    threshold = float(detector.threshold)
    return detector


def validate(detector: PrototypeBankSelfVoiceDetector, self_files: list[Path], neg_files: list[Path], threshold: float):
    detector.threshold = threshold
    print(f"\n--- Self captures (should be SELF) ---")
    self_results = []
    for f in self_files:
        pcm = np.load(f).astype(np.float32)
        segments = extract_segments(pcm)
        scores = []
        for seg in segments:
            is_self, margin = detector.is_self(seg)
            scores.append({"is_self": is_self, "margin": round(margin, 4)})
        pct_self = np.mean([s["is_self"] for s in scores]) * 100 if scores else 0
        avg_margin = np.mean([s["margin"] for s in scores]) if scores else 0
        self_results.append({"file": f.name, "segments": len(scores), "pct_self": round(pct_self, 1), "avg_margin": round(avg_margin, 4)})
        print(f"  {f.name}: {len(scores)} segs, avg_margin={avg_margin:.4f}, {pct_self:.0f}% SELF")

    print(f"\n--- Negative captures (should be WORLD) ---")
    neg_results = []
    for f in neg_files:
        pcm = np.load(f).astype(np.float32)
        segments = extract_segments(pcm)
        scores = []
        for seg in segments:
            is_self, margin = detector.is_self(seg)
            scores.append({"is_self": is_self, "margin": round(margin, 4)})
        pct_self = np.mean([s["is_self"] for s in scores]) * 100 if scores else 0
        avg_margin = np.mean([s["margin"] for s in scores]) if scores else 0
        neg_results.append({"file": f.name, "segments": len(scores), "pct_self": round(pct_self, 1), "avg_margin": round(avg_margin, 4)})
        print(f"  {f.name}: {len(scores)} segs, avg_margin={avg_margin:.4f}, {pct_self:.0f}% SELF")

    self_recall = np.mean([r["pct_self"] for r in self_results]) / 100 if self_results else 0
    fp_rate = np.mean([r["pct_self"] for r in neg_results]) / 100 if neg_results else 0
    precision = self_recall / (self_recall + fp_rate) if (self_recall + fp_rate) > 0 else 0
    f1 = 2 * precision * self_recall / (precision + self_recall) if (precision + self_recall) > 0 else 0
    return self_results, neg_results, self_recall, fp_rate, precision, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an M4 prototype bank")
    parser.add_argument("--bank", required=True, help="Path to prototype bank .npz")
    parser.add_argument("--self-captures", required=True, help="Dir with self .npy captures")
    parser.add_argument("--neg-captures", required=True, help="Dir with negative .npy captures")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (defaults to bank suggestion)")
    parser.add_argument("--backend", default="ecapa", help="Speaker embedding backend")
    parser.add_argument("--output", default="/tmp/m4_validate_prototype_bank.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Prototype Bank Validation")
    print("=" * 60)

    t0 = time.time()
    detector = build_detector(args.bank, args.backend)
    bank = np.load(args.bank, allow_pickle=True)
    suggested = float(bank["threshold"][0]) if "threshold" in bank else 0.0
    threshold = suggested if args.threshold is None else args.threshold
    print(f"Loaded bank from {args.bank}")
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Self prototypes: {len(detector.self_bank)}")
    print(f"  Neg prototypes:  {len(detector.neg_bank)}")
    print(f"  Threshold:       {threshold}")

    self_files = load_self_captures(args.self_captures)
    neg_files = load_neg_captures(args.neg_captures)
    print(f"\nSelf captures: {len(self_files)} files")
    print(f"Negative captures: {len(neg_files)} files")

    self_results, neg_results, self_recall, fp_rate, precision, f1 = validate(detector, self_files, neg_files, threshold)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Self-recall:  {self_recall * 100:.1f}%")
    print(f"  FP rate:      {fp_rate * 100:.1f}%")
    print(f"  Precision:    {precision:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Threshold:    {threshold}")
    if self_recall >= 0.90 and fp_rate <= 0.10:
        print("\n  ✅ Domain gap CLOSED — prototype bank generalizes to live domain")
    elif self_recall >= 0.70 and fp_rate <= 0.30:
        print("\n  🟡 Partial improvement — gap narrowed but not closed")
    else:
        print("\n  🔴 Domain gap persists — more data or different strategy needed")
    print("=" * 60)

    output = {
        "bank": args.bank,
        "threshold": threshold,
        "self_results": self_results,
        "neg_results": neg_results,
        "summary": {
            "self_recall_pct": round(self_recall * 100, 1),
            "fp_rate_pct": round(fp_rate * 100, 1),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
