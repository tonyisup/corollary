#!/usr/bin/env python3
"""Train a balanced M4 scorer using the existing fine-tuning infrastructure.

Generates a balanced dataset from existing captures (including synthetic self
captures) and trains using the feature-based approach from m4_finetune_ecapa.py.

The key improvement over previous runs:
- Balanced classes: equal self and other segments
- Controlled augmentation: same factor for both classes
- Multiple synthetic self captures to increase positive diversity

Usage:
    .venv/bin/python scripts/m4_train_balanced_scorer.py \\
        --self-captures data/m4_clean_captures_bl2021 \\
        --other-captures data/m4_human_negative_bl2021 \\
        --output-dir data/m4_finetuned_ecapa_balanced_v1 \\
        --output /tmp/m4_balanced_train.json \\
        --aug-factor 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "packages"))

SR = 16000


def load_capture_pcm(directory: str) -> list[np.ndarray]:
    """Load all capture .npy files from a directory."""
    dir_path = Path(directory)
    pcms = []
    for npy_file in sorted(dir_path.glob("*.npy")):
        if "agent_pcm" in npy_file.name:
            continue
        pcm = np.load(str(npy_file)).astype(np.float32)
        pcms.append(pcm)
    return pcms


def estimate_room_ir(agent_pcm: np.ndarray, captured_pcm: np.ndarray, ir_length: int = 1600) -> np.ndarray:
    """Estimate room IR via frequency-domain deconvolution."""
    n = len(captured_pcm)
    agent_pad = np.zeros(n, dtype=np.float32)
    agent_pad[:len(agent_pcm)] = agent_pcm

    A = np.fft.rfft(agent_pad)
    C = np.fft.rfft(captured_pcm)

    snr = 10.0
    H = C * np.conj(A) / (np.abs(A) ** 2 + 1.0 / snr)
    h = np.fft.irfft(H, n=n)

    ir = h[:ir_length].copy()
    ir = ir / max(np.max(np.abs(ir)), 1e-8)
    return ir.astype(np.float32)


def generate_synthetic_self_captures(
    room_ir: np.ndarray,
    n_variants: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic (agent_pcm, captured) pairs."""
    rng = np.random.default_rng(seed)
    variants = []

    for i in range(n_variants):
        duration_s = 7.0 + rng.uniform(-1.0, 1.0)
        n_samples = int(duration_s * SR)
        t = np.arange(n_samples, dtype=np.float32) / SR

        f0 = rng.uniform(120, 220)
        signal = np.zeros(n_samples, dtype=np.float32)
        for harmonic in range(1, 6):
            amp = 1.0 / harmonic * rng.uniform(0.5, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)

        envelope = np.ones(n_samples, dtype=np.float32)
        n_syllables = int(duration_s * 4)
        for _ in range(n_syllables):
            pos = rng.integers(0, n_samples)
            width = rng.integers(int(0.1 * SR), int(0.3 * SR))
            start = max(0, pos - width // 2)
            end = min(n_samples, pos + width // 2)
            envelope[start:end] *= rng.uniform(0.1, 0.5)

        signal *= envelope
        signal = signal / max(np.max(np.abs(signal)), 1e-8) * 0.5

        captured = fftconvolve(signal, room_ir, mode="full").astype(np.float32)
        noise = rng.normal(0, 0.001, size=len(captured)).astype(np.float32)
        captured += noise

        variants.append((signal.astype(np.float32), captured))

    return variants


def prepare_balanced_training_dir(
    self_captures_dir: str,
    other_captures_dir: str,
    output_dir: str,
    aug_factor: int = 5,
    seed: int = 42,
) -> dict:
    """Prepare a balanced training directory for m4_finetune_ecapa.py.

    Creates a directory structure that the existing fine-tuning script can use:
      output_dir/
        self/       -> self capture .npy files (original + synthetic)
        other/      -> other-speaker capture .npy files
    """
    out_path = Path(output_dir)
    self_dir = out_path / "self"
    other_dir = out_path / "other"
    self_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    # Load existing captures
    self_pcms = load_capture_pcm(self_captures_dir)
    other_pcms = load_capture_pcm(other_captures_dir)

    print(f"Loaded {len(self_pcms)} self captures, {len(other_pcms)} other captures")

    # Estimate room IR from first self capture
    agent_files = sorted(Path(self_captures_dir).glob("agent_pcm_*.npy"))
    if agent_files:
        agent_pcm = np.load(str(agent_files[0])).astype(np.float32)
        room_ir = estimate_room_ir(agent_pcm, self_pcms[0])
        print(f"Estimated room IR: {len(room_ir)} taps")

        # Generate synthetic self captures to balance
        n_other = len(other_pcms)
        n_self = len(self_pcms)
        n_synthetic = max(0, n_other - n_self)
        if n_synthetic > 0:
            print(f"Generating {n_synthetic} synthetic self captures")
            synthetic = generate_synthetic_self_captures(room_ir, n_variants=n_synthetic, seed=seed)
            for i, (agent, captured) in enumerate(synthetic):
                self_pcms.append(captured)
                # Also save agent for reference
                np.save(str(self_dir / f"synthetic_agent_{i:03d}.npy"), agent)

    # Save all captures with names that m4_finetune_ecapa.py expects
    # Self: captured_*.npy (or agent_pcm_*.npy)
    # Other: human_negative_*.npy
    for i, pcm in enumerate(self_pcms):
        np.save(str(self_dir / f"captured_{i:03d}.npy"), pcm)
    for i, pcm in enumerate(other_pcms):
        np.save(str(other_dir / f"human_negative_{i:03d}.npy"), pcm)

    meta = {
        "n_self": len(self_pcms),
        "n_other": len(other_pcms),
        "aug_factor": aug_factor,
        "seed": seed,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(self_pcms)} self + {len(other_pcms)} other captures to {output_dir}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train balanced M4 scorer")
    parser.add_argument("--self-captures", required=True)
    parser.add_argument("--other-captures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", type=str, default="/tmp/m4_balanced_train.json")
    parser.add_argument("--aug-factor", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Balanced Scorer Training")
    print("=" * 60)

    # Step 1: Prepare balanced dataset
    print("\n[1/2] Preparing balanced training data...")
    meta = prepare_balanced_training_dir(
        args.self_captures, args.other_captures, args.output_dir,
        aug_factor=args.aug_factor, seed=args.seed,
    )

    # Step 2: Run fine-tuning using the existing script
    print("\n[2/2] Running feature-based fine-tuning...")
    train_dir = Path(args.output_dir)

    cmd = [
        sys.executable, "scripts/m4_finetune_ecapa.py",
        "--clean-captures", str(train_dir / "self"),
        "--other-speaker-dirs", str(train_dir / "other"),
        "--output-dir", str(Path(args.output_dir) / "fine_tuned"),
        "--output", args.output,
        "--num-epochs", str(args.num_epochs),
        "--lr", str(args.lr),
    ]

    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Training failed with exit code {result.returncode}")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        sys.exit(1)

    print(f"  Training complete in {elapsed:.1f}s")

    # Load and augment the training log
    log = {}
    if Path(args.output).exists():
        log = json.loads(Path(args.output).read_text())
    log["balanced"] = True
    log["n_self_captures"] = meta["n_self"]
    log["n_other_captures"] = meta["n_other"]
    log["total_training_time_s"] = round(elapsed, 1)
    Path(args.output).write_text(json.dumps(log, indent=2))

    print(f"\nResults saved to {args.output}")
    if "best_test_accuracy" in log:
        print(f"Best test accuracy: {log['best_test_accuracy']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
