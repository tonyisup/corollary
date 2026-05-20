#!/usr/bin/env python3
"""Build a multi-prototype M4 self-voice bank from captures.

This is an alternative modeling strategy to the fine-tuned scorer: instead of
training a learned head on pairwise embeddings, we keep a bank of self and hard
negative prototypes and score probes by the cosine-margin between the two banks.

Usage:
    .venv/bin/python scripts/m4_build_prototype_bank.py \
        --self-captures /tmp/m4_diversity_round4_train/self \
        --neg-captures /tmp/m4_diversity_round4_train/other \
        --output /tmp/m4_prototype_bank.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def load_pcm_files(directory: str) -> list[np.ndarray]:
    return [np.load(p).astype(np.float32) for p in sorted(Path(directory).glob("*.npy"))]


def collect_embeddings(identifier: SpeakerIdentifier, pcms: list[np.ndarray]) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    for pcm in pcms:
        for seg in extract_segments(pcm):
            embeddings.append(identifier.get_embedding(seg))
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an M4 prototype bank")
    parser.add_argument("--self-captures", required=True, help="Dir with self .npy captures")
    parser.add_argument("--neg-captures", required=True, help="Dir with negative .npy captures")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--threshold", type=float, default=0.0, help="Suggested decision threshold")
    parser.add_argument("--backend", default="ecapa", help="Speaker embedding backend")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Prototype Bank Builder")
    print("=" * 60)

    identifier = SpeakerIdentifier(use_real=True, backend=args.backend)
    detector = PrototypeBankSelfVoiceDetector(identifier, threshold=args.threshold)

    self_pcms = load_pcm_files(args.self_captures)
    neg_pcms = load_pcm_files(args.neg_captures)
    print(f"Loaded {len(self_pcms)} self captures, {len(neg_pcms)} negative captures")

    self_bank = collect_embeddings(identifier, self_pcms)
    neg_bank = collect_embeddings(identifier, neg_pcms)
    detector.register_self_bank(np.stack(self_bank) if self_bank else np.zeros((0, 192), dtype=np.float32))
    detector.register_negative_bank(np.stack(neg_bank) if neg_bank else np.zeros((0, 192), dtype=np.float32))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        self_embeddings=np.stack(detector.self_bank) if detector.self_bank else np.zeros((0, 192), dtype=np.float32),
        neg_embeddings=np.stack(detector.neg_bank) if detector.neg_bank else np.zeros((0, 192), dtype=np.float32),
        threshold=np.array([args.threshold], dtype=np.float32),
        backend=np.array([args.backend]),
        segment_duration_s=np.array([SEG_DUR_S], dtype=np.float32),
        hop_s=np.array([HOP_S], dtype=np.float32),
    )

    meta = {
        "self_captures": len(self_pcms),
        "neg_captures": len(neg_pcms),
        "self_prototypes": len(detector.self_bank),
        "neg_prototypes": len(detector.neg_bank),
        "threshold": args.threshold,
        "backend": args.backend,
        "segment_duration_s": SEG_DUR_S,
        "hop_s": HOP_S,
        "output": str(out),
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    print(f"Saved bank to {out}")
    print(f"Self prototypes: {len(detector.self_bank)}")
    print(f"Neg prototypes:  {len(detector.neg_bank)}")
    print(f"Suggested threshold: {args.threshold}")


if __name__ == "__main__":
    main()
