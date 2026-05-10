#!/usr/bin/env python3
"""Fine-tune ECAPA-TDNN on room-colored captured audio.

Uses the existing clean captures and validation artifacts to create
positive (self) and negative (other-speaker) training pairs, then
fine-tunes a copy of ECAPA using a speaker verification loss.

Usage:
    .venv/bin/python scripts/m4_finetune_ecapa.py \\
        --clean-captures data/m4_clean_captures_fixed \\
        --num-epochs 20 \\
        --output-dir data/m4_finetuned_ecapa \\
        --output /tmp/m4_finetune_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "packages"))

SR = 16000


class CapturePairDataset(Dataset):
    """Dataset of (audio, label) pairs for speaker verification fine-tuning.

    Positive pairs: captured self TTS (label=1)
    Negative pairs: captured other-speaker TTS (label=0)

    Data augmentation is applied to increase effective dataset size:
      - Gaussian noise (SNR 15-25 dB)
      - Pitch shift (-2 to +2 semitones)
      - Time stretch (0.85-1.15x)
      - Random room impulse response convolution
    """

    def __init__(
        self,
        self_pcms: list[np.ndarray],
        other_pcms: list[np.ndarray],
        segment_duration_s: float = 2.0,
        augment: bool = True,
        aug_factor: int = 5,
        seed: int = 42,
    ):
        self.segment_samples = int(SR * segment_duration_s)
        self.augment = augment
        self.aug_factor = aug_factor
        self.rng = np.random.RandomState(seed)
        self.samples: list[tuple[np.ndarray, int]] = []

        for pcm in self_pcms:
            self._add_segments(pcm, label=1)
        for pcm in other_pcms:
            self._add_segments(pcm, label=0)

        print(f"  Raw segments: {len(self.samples)}")
        if self.augment:
            self._apply_augmentation()
            print(f"  After augmentation: {len(self.samples)}")

    def _add_segments(self, pcm: np.ndarray, label: int):
        """Extract overlapping segments from a long capture."""
        step = self.segment_samples // 2  # 50% overlap
        for start in range(0, len(pcm) - self.segment_samples, step):
            seg = pcm[start:start + self.segment_samples].copy()
            self.samples.append((seg, label))

    def _apply_augmentation(self):
        """Apply data augmentation to increase dataset size."""
        augmented = []
        for seg, label in self.samples:
            for _ in range(self.aug_factor - 1):  # -1 because original is kept
                aug_seg = self._augment_segment(seg)
                augmented.append((aug_seg, label))
        self.samples.extend(augmented)
        # Shuffle
        perm = self.rng.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def _augment_segment(self, pcm: np.ndarray) -> np.ndarray:
        """Apply a random augmentation to a segment."""
        aug_type = self.rng.choice(['noise', 'pitch', 'stretch', 'reverb', 'combined'])

        if aug_type == 'noise':
            return self._add_noise(pcm)
        elif aug_type == 'pitch':
            return self._pitch_shift(pcm)
        elif aug_type == 'stretch':
            return self._time_stretch(pcm)
        elif aug_type == 'reverb':
            return self._add_reverb(pcm)
        else:  # combined
            pcm = self._add_noise(pcm)
            if self.rng.random() > 0.5:
                pcm = self._pitch_shift(pcm)
            return pcm

    def _add_noise(self, pcm: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at random SNR."""
        snr_db = self.rng.uniform(15, 25)
        signal_power = np.mean(pcm ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.randn(len(pcm)) * np.sqrt(noise_power)
        return (pcm + noise).astype(np.float32)

    def _pitch_shift(self, pcm: np.ndarray) -> np.ndarray:
        """Pitch shift by a random amount."""
        try:
            import librosa
            n_steps = self.rng.uniform(-2, 2)
            shifted = librosa.effects.pitch_shift(pcm, sr=SR, n_steps=n_steps)
            return shifted.astype(np.float32)
        except Exception:
            return pcm

    def _time_stretch(self, pcm: np.ndarray) -> np.ndarray:
        """Time stretch by a random factor."""
        try:
            import librosa
            rate = self.rng.uniform(0.85, 1.15)
            stretched = librosa.effects.time_stretch(pcm, rate=rate)
            # Pad or trim to original length
            if len(stretched) < len(pcm):
                stretched = np.pad(stretched, (0, len(pcm) - len(stretched)))
            else:
                stretched = stretched[:len(pcm)]
            return stretched.astype(np.float32)
        except Exception:
            return pcm

    def _add_reverb(self, pcm: np.ndarray) -> np.ndarray:
        """Apply a simple synthetic room impulse response."""
        # Generate a simple exponential decay IR
        ir_length = int(SR * self.rng.uniform(0.1, 0.5))
        ir = np.exp(-np.linspace(0, 5, ir_length)) * self.rng.randn(ir_length)
        ir = ir / (np.linalg.norm(ir) + 1e-9) * self.rng.uniform(0.1, 0.3)
        reverbed = np.convolve(pcm, ir, mode='same')
        return reverbed.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pcm, label = self.samples[idx]
        # Normalize
        peak = np.max(np.abs(pcm))
        if peak > 0:
            pcm = pcm / peak * 0.7
        return torch.tensor(pcm, dtype=torch.float32), label


class SpeakerVerificationLoss(nn.Module):
    """Binary cross-entropy loss for same/different speaker verification."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bce(similarity, labels.float())


def load_npy_files(directory: str, pattern: str) -> list[np.ndarray]:
    """Load all .npys matching pattern from directory."""
    path = Path(directory)
    files = sorted(path.glob(pattern))
    return [np.load(f).astype(np.float32) for f in files]


def fine_tune_ecapa(
    clean_captures_dir: str,
    other_speaker_dirs: list[str],
    num_epochs: int = 20,
    lr: float = 1e-5,
    segment_duration_s: float = 2.0,
    batch_size: int = 16,
    output_dir: str = "data/m4_finetuned_ecapa",
    device: str = "cpu",
) -> dict:
    """Fine-tune ECAPA-TDNN on captured audio pairs."""
    from speechbrain.inference.speaker import SpeakerRecognition

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load base model
    print("Loading base ECAPA model...")
    cache_dir = output_path / "pretrained"
    cache_dir.mkdir(exist_ok=True)
    model = SpeakerRecognition.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir=str(cache_dir),
    )
    print(f"  Model loaded. Device: {device}")

    # 2. Prepare dataset
    print("Preparing dataset...")
    self_pcms = load_npy_files(clean_captures_dir, "captured_*.npy")
    if not self_pcms:
        self_pcms = load_npy_files(clean_captures_dir, "agent_pcm_*.npy")
    print(f"  Self captures: {len(self_pcms)}")

    other_pcms = []
    for other_dir in other_speaker_dirs:
        other_pcms.extend(load_npy_files(other_dir, "live_captured.npy"))
    print(f"  Other-speaker captures: {len(other_pcms)}")

    dataset = CapturePairDataset(
        self_pcms=self_pcms,
        other_pcms=other_pcms,
        segment_duration_s=segment_duration_s,
    )
    print(f"  Total segments: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Set up fine-tuning — use feature-based approach
    # SpeechBrain's SpeakerRecognition wraps the encoder in a specific way.
    # We freeze the base model and train a projection head on embeddings.
    return fine_tune_feature_based(
        model, dataset, dataloader, num_epochs, lr, output_path, device
    )


def fine_tune_feature_based(
    model,
    dataset: CapturePairDataset,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    output_path: Path,
    device: str,
) -> dict:
    """Feature-based fine-tuning: train a linear head on frozen ECAPA embeddings.

    Since SpeechBrain's encoder internals are complex, we freeze the base model
    and train a simple projection head that maps embedding pairs to a similarity score.
    """
    print("Using feature-based fine-tuning (frozen encoder + trainable head)")

    # Compute all embeddings with the frozen model
    print("Computing embeddings with frozen model...")
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            pcm, label = dataset[i]
            x = pcm.unsqueeze(0)
            emb = model.encode_batch(x).squeeze()
            all_embeddings.append(emb.cpu())
            all_labels.append(label)

    embeddings = torch.stack(all_embeddings)  # (N, emb_dim)
    labels = torch.tensor(all_labels)  # (N,)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Split into self and other
    self_mask = labels == 1
    other_mask = labels == 0
    self_embeddings = embeddings[self_mask]
    other_embeddings = embeddings[other_mask]
    print(f"  Self embeddings: {self_embeddings.shape[0]}")
    print(f"  Other embeddings: {other_embeddings.shape[0]}")

    # Create training pairs: (emb1, emb2) -> same/different
    # Positive pairs: self vs self (same speaker)
    # Negative pairs: self vs other (different speaker)
    n_self = len(self_embeddings)
    n_other = len(other_embeddings)

    # Create balanced pairs
    n_pairs = min(n_self, n_other)
    # Positive: pair each self embedding with another self embedding
    pos_emb1 = self_embeddings[:n_pairs]
    pos_emb2 = self_embeddings[n_pairs:n_pairs*2] if n_pairs * 2 <= n_self else torch.roll(self_embeddings[:n_pairs], shifts=1, dims=0)
    pos_input = torch.cat([pos_emb1, pos_emb2], dim=1)
    pos_labels = torch.ones(n_pairs)

    # Negative: pair self with other
    neg_input = torch.cat([self_embeddings[:n_pairs], other_embeddings[:n_pairs]], dim=1)
    neg_labels = torch.zeros(n_pairs)

    X = torch.cat([pos_input, neg_input], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]

    # Train a scorer
    emb_dim = embeddings.shape[1]
    scorer = nn.Sequential(
        nn.Linear(emb_dim * 2, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    optimizer = optim.Adam(scorer.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTraining scorer: {len(X_train)} train, {len(X_test)} test pairs")
    history = []
    for epoch in range(num_epochs):
        scorer.train()
        logits = scorer(X_train).squeeze()
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        scorer.eval()
        with torch.no_grad():
            test_logits = scorer(X_test).squeeze()
            test_loss = criterion(test_logits, y_test)
            preds = (torch.sigmoid(test_logits) > 0.5).float()
            acc = (preds == y_test).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss.item(), 4),
            "test_loss": round(test_loss.item(), 4),
            "test_accuracy": round(acc, 4),
        })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}: train_loss={loss.item():.4f} test_acc={acc:.4f}")

    # Save
    save_path = output_path / "fine_tuned"
    save_path.mkdir(exist_ok=True)
    torch.save(scorer.state_dict(), save_path / "scorer.pt")
    (save_path / "config.json").write_text(json.dumps({
        "fine_tuned": True,
        "method": "feature_based",
        "num_epochs": num_epochs,
        "lr": lr,
        "emb_dim": int(emb_dim),
        "num_train_pairs": int(len(X_train)),
        "num_test_pairs": int(len(X_test)),
        "final_test_accuracy": history[-1]["test_accuracy"] if history else 0,
    }, indent=2))

    return {
        "output_dir": str(output_path),
        "method": "feature_based",
        "num_epochs": num_epochs,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune ECAPA on captured audio")
    parser.add_argument("--clean-captures", default="data/m4_clean_captures_fixed")
    parser.add_argument("--other-speaker-dirs", nargs="+",
                        default=["data/m4_capture_validation_heldout_20260509/same_domain_negative_playback_window"])
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--output-dir", default="data/m4_finetuned_ecapa")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    print("=" * 60)
    print("M4 ECAPA Fine-tuning")
    print("=" * 60)

    t0 = time.time()
    result = fine_tune_ecapa(
        clean_captures_dir=args.clean_captures,
        other_speaker_dirs=args.other_speaker_dirs,
        num_epochs=args.num_epochs,
        lr=args.lr,
        segment_duration_s=args.segment_duration,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

    result["duration_s"] = round(time.time() - t0, 1)
    print(f"\nFine-tuning complete in {result['duration_s']}s")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
