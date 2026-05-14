#!/usr/bin/env python3
"""Generate a balanced M4 training dataset from existing captures.

Takes the existing self captures and human-negative captures, then:
1. Generates additional self-capture variants via TTS synthesis + room coloring
2. Balances classes using controlled augmentation (same factor for both)
3. Outputs a balanced dataset ready for fine-tuning

Usage:
    .venv/bin/python scripts/m4_generate_balanced_dataset.py \\
        --self-captures data/m4_clean_captures_bl2021 \\
        --other-captures data/m4_human_negative_bl2021 \\
        --output-dir data/m4_balanced_dataset \\
        --target-per-class 200 \\
        --aug-factor 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "packages"))

SR = 16000


def load_captures(directory: str) -> list[tuple[np.ndarray, dict]]:
    """Load all .npy captures from a directory."""
    dir_path = Path(directory)
    captures = []
    for npy_file in sorted(dir_path.glob("*.npy")):
        if "agent_pcm" in npy_file.name:
            continue
        pcm = np.load(str(npy_file)).astype(np.float32)
        # Load companion metadata
        meta_file = npy_file.with_suffix(".json")
        if not meta_file.exists():
            meta_file = npy_file.parent / f"meta_{npy_file.stem.split('_', 1)[1]}.json"
        meta = {}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        captures.append((pcm, meta))
    return captures


def generate_synthetic_self_captures(
    room_ir: np.ndarray,
    n_variants: int = 5,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate synthetic self-capture variants by playing different TTS-like signals through the room IR."""
    rng = np.random.default_rng(seed)
    variants = []

    for i in range(n_variants):
        # Generate speech-like signal: sum of modulated sinusoids
        duration_s = 7.0 + rng.uniform(-1.0, 1.0)  # 6-8 seconds
        n_samples = int(duration_s * SR)
        t = np.arange(n_samples, dtype=np.float32) / SR

        # Fundamental frequency with variation (simulating TTS)
        f0 = rng.uniform(120, 220)  # typical TTS range
        signal = np.zeros(n_samples, dtype=np.float32)

        # Add harmonics
        for harmonic in range(1, 6):
            amp = 1.0 / harmonic * rng.uniform(0.5, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)

        # Add amplitude envelope (simulating speech rhythm)
        envelope = np.ones(n_samples, dtype=np.float32)
        n_syllables = int(duration_s * 4)  # ~4 syllables per second
        for _ in range(n_syllables):
            pos = rng.integers(0, n_samples)
            width = rng.integers(int(0.1 * SR), int(0.3 * SR))
            start = max(0, pos - width // 2)
            end = min(n_samples, pos + width // 2)
            envelope[start:end] *= rng.uniform(0.1, 0.5)

        signal *= envelope
        signal = signal / max(np.max(np.abs(signal)), 1e-8) * 0.5

        # Convolve with room IR
        captured = fftconvolve(signal, room_ir, mode="full").astype(np.float32)
        # Add light noise
        noise = rng.normal(0, 0.001, size=len(captured)).astype(np.float32)
        captured += noise

        variants.append(captured)

    return variants


def extract_segments(
    pcm: np.ndarray,
    segment_duration_s: float = 2.0,
    hop_ratio: float = 0.5,
) -> list[np.ndarray]:
    """Extract overlapping segments from a capture."""
    seg_samples = int(SR * segment_duration_s)
    hop = int(seg_samples * hop_ratio)
    segments = []
    for start in range(0, len(pcm) - seg_samples, hop):
        seg = pcm[start:start + seg_samples].copy()
        # Skip near-silent segments
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 0.001:
            segments.append(seg)
    return segments


def balance_dataset(
    self_segments: list[np.ndarray],
    other_segments: list[np.ndarray],
    target_per_class: int,
    aug_factor: int,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Balance dataset by augmenting the minority class."""
    rng = np.random.RandomState(seed)

    # Apply augmentation
    def augment(seg: np.ndarray, n_aug: int) -> list[np.ndarray]:
        result = [seg]
        for _ in range(n_aug - 1):
            result.append(_augment_segment(seg, rng))
        return result

    self_augmented = []
    for seg in self_segments:
        self_augmented.extend(augment(seg, aug_factor))

    other_augmented = []
    for seg in other_segments:
        other_augmented.extend(augment(seg, aug_factor))

    # Balance: oversample minority to match majority
    if len(self_augmented) < len(other_augmented):
        # Oversample self
        indices = rng.choice(len(self_augmented), size=len(other_augmented), replace=True)
        self_balanced = [self_augmented[i] for i in indices]
        other_balanced = other_augmented
    else:
        # Oversample other
        indices = rng.choice(len(other_augmented), size=len(self_augmented), replace=True)
        other_balanced = [other_augmented[i] for i in indices]
        self_balanced = self_augmented

    # Trim to target
    if len(self_balanced) > target_per_class:
        indices = rng.choice(len(self_balanced), size=target_per_class, replace=False)
        self_balanced = [self_balanced[i] for i in sorted(indices)]
    if len(other_balanced) > target_per_class:
        indices = rng.choice(len(other_balanced), size=target_per_class, replace=False)
        other_balanced = [other_balanced[i] for i in sorted(indices)]

    return self_balanced, other_balanced


def _augment_segment(pcm: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply a random augmentation to a segment."""
    aug_type = rng.choice(['noise', 'stretch', 'reverb'])

    if aug_type == 'noise':
        snr_db = rng.uniform(15, 25)
        signal_power = np.mean(pcm ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.randn(len(pcm)) * np.sqrt(noise_power)
        return (pcm + noise).astype(np.float32)

    elif aug_type == 'stretch':
        # Simple time stretch via resampling
        factor = rng.uniform(0.9, 1.1)
        n_new = int(len(pcm) / factor)
        indices = np.linspace(0, len(pcm) - 1, n_new)
        stretched = np.interp(indices, np.arange(len(pcm)), pcm)
        # Pad or trim to original length
        if len(stretched) < len(pcm):
            stretched = np.pad(stretched, (0, len(pcm) - len(stretched)))
        else:
            stretched = stretched[:len(pcm)]
        return stretched.astype(np.float32)

    else:  # reverb
        # Simple synthetic reverb
        ir_len = rng.randint(100, 500)
        ir = np.zeros(ir_len, dtype=np.float32)
        ir[0] = 1.0
        for j in range(1, min(ir_len, 5)):
            ir[j] = rng.uniform(-0.3, 0.3) * (0.5 ** j)
        return fftconvolve(pcm, ir, mode="full")[:len(pcm)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced M4 training dataset")
    parser.add_argument("--self-captures", required=True, help="Dir with self capture .npy files")
    parser.add_argument("--other-captures", required=True, help="Dir with other-speaker .npy files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-per-class", type=int, default=200)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading self captures from {args.self_captures}")
    self_raw = load_captures(args.self_captures)
    print(f"  Loaded {len(self_raw)} self captures")

    print(f"Loading other captures from {args.other_captures}")
    other_raw = load_captures(args.other_captures)
    print(f"  Loaded {len(other_raw)} other captures")

    # Extract room IR from the first self capture (agent_pcm vs captured)
    agent_files = sorted(Path(args.self_captures).glob("agent_pcm_*.npy"))
    if agent_files:
        agent_pcm = np.load(str(agent_files[0])).astype(np.float32)
        captured_pcm = self_raw[0][0]
        # Estimate room IR via deconvolution
        room_ir = _estimate_room_ir(agent_pcm, captured_pcm)
        print(f"  Estimated room IR: {len(room_ir)} taps")

        # Generate synthetic self captures
        n_synthetic = max(0, 4 - len(self_raw))  # aim for at least 4 self captures
        if n_synthetic > 0:
            print(f"  Generating {n_synthetic} synthetic self captures")
            synthetic = generate_synthetic_self_captures(room_ir, n_variants=n_synthetic, seed=args.seed)
            for i, syn in enumerate(synthetic):
                meta = {"synthetic": True, "variant": i}
                self_raw.append((syn, meta))
    else:
        print("  WARNING: No agent_pcm files found, skipping synthetic generation")

    # Extract segments
    print("Extracting segments...")
    self_segments = []
    for pcm, meta in self_raw:
        segs = extract_segments(pcm)
        self_segments.extend(segs)

    other_segments = []
    for pcm, meta in other_raw:
        segs = extract_segments(pcm)
        other_segments.extend(segs)

    print(f"  Self segments: {len(self_segments)}")
    print(f"  Other segments: {len(other_segments)}")

    # Balance
    print(f"Balancing dataset (target={args.target_per_class} per class, aug_factor={args.aug_factor})...")
    self_balanced, other_balanced = balance_dataset(
        self_segments, other_segments,
        target_per_class=args.target_per_class,
        aug_factor=args.aug_factor,
        seed=args.seed,
    )

    print(f"  Final self: {len(self_balanced)}")
    print(f"  Final other: {len(other_balanced)}")

    # Save
    self_dir = Path(args.output_dir) / "self"
    other_dir = Path(args.output_dir) / "other"
    self_dir.mkdir(exist_ok=True)
    other_dir.mkdir(exist_ok=True)

    for i, seg in enumerate(self_balanced):
        np.save(str(self_dir / f"seg_{i:04d}.npy"), seg)
    for i, seg in enumerate(other_balanced):
        np.save(str(other_dir / f"seg_{i:04d}.npy"), seg)

    # Save metadata
    meta = {
        "self_count": len(self_balanced),
        "other_count": len(other_balanced),
        "self_raw_captures": len(self_raw),
        "other_raw_captures": len(other_raw),
        "aug_factor": args.aug_factor,
        "target_per_class": args.target_per_class,
        "seed": args.seed,
    }
    (Path(args.output_dir) / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {args.output_dir}")
    print("Done.")


def _estimate_room_ir(agent: np.ndarray, captured: np.ndarray, ir_length: int = 1600) -> np.ndarray:
    """Estimate room IR via frequency-domain deconvolution."""
    n = len(captured)
    agent_pad = np.zeros(n, dtype=np.float32)
    agent_pad[:len(agent)] = agent

    A = np.fft.rfft(agent_pad)
    C = np.fft.rfft(captured)

    # Wiener deconvolution
    snr = 10.0
    H = C * np.conj(A) / (np.abs(A) ** 2 + 1.0 / snr)
    h = np.fft.irfft(H, n=n)

    # Trim to ir_length and normalize
    ir = h[:ir_length].copy()
    ir = ir / max(np.max(np.abs(ir)), 1e-8)
    return ir.astype(np.float32)


if __name__ == "__main__":
    main()
