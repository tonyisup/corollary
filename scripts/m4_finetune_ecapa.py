#!/usr/bin/env python3
"""Fine-tune ECAPA-TDNN on room-colored captured audio.

Uses the existing clean captures and validation artifacts to create
positive (self) and negative (other-speaker) training pairs, then
fine-tunes a copy of ECAPA using a speaker verification loss.

Two modes:
  --full-fine-tune     Unfreeze last N SE blocks + scorer, train end-to-end
  (default)            Feature-based: frozen encoder, train MLP scorer on embeddings

Usage:
    # Feature-based (fast, small data)
    .venv/bin/python scripts/m4_finetune_ecapa.py \\
        --clean-captures data/m4_clean_captures_bl2021 \\
        --other-speaker-dirs data/capture_neg/ data/human_neg/ \\
        --output-dir data/m4_finetuned_ecapa \\
        --output /tmp/log.json

    # Full fine-tune (unfreezes encoder layers)
    .venv/bin/python scripts/m4_finetune_ecapa.py \\
        --clean-captures data/m4_clean_captures_bl2021 \\
        --other-speaker-dirs data/capture_neg/ data/human_neg/ \\
        --full-fine-tune --unfreeze-blocks 2 \\
        --num-epochs 50 --lr 1e-5 \\
        --output-dir data/m4_finetuned_ecapa_full \\
        --output /tmp/log.json
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
            if len(stretched) < len(pcm):
                stretched = np.pad(stretched, (0, len(pcm) - len(stretched)))
            else:
                stretched = stretched[:len(pcm)]
            return stretched.astype(np.float32)
        except Exception:
            return pcm

    def _add_reverb(self, pcm: np.ndarray) -> np.ndarray:
        """Apply a simple synthetic room impulse response."""
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


def load_npy_files(directory: str, pattern: str) -> list[np.ndarray]:
    """Load all .npys matching pattern from directory."""
    path = Path(directory)
    files = sorted(path.glob(pattern))
    return [np.load(f).astype(np.float32) for f in files]


def build_pair_indices(labels: torch.Tensor, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build balanced positive/negative pair indices.

    Returns (pair_idx1, pair_idx2, pair_labels) where:
      - Positive pair: two self segments (label=1)
      - Negative pair: self + other segment (label=0)
    """
    self_idx = (labels == 1).nonzero(as_tuple=True)[0]
    other_idx = (labels == 0).nonzero(as_tuple=True)[0]

    n_self = len(self_idx)
    n_other = len(other_idx)
    n_pairs = min(n_self, n_other)

    rng = torch.Generator().manual_seed(seed)

    # Positive: pair each self with another (rolled) self
    pos_i1 = self_idx[:n_pairs]
    pos_i2 = self_idx[torch.arange(n_pairs) - 1] if n_pairs <= n_self else torch.roll(pos_i1, 1)

    # Negative: pair self with other
    neg_i1 = self_idx[:n_pairs]
    neg_i2 = other_idx[:n_pairs]

    i1 = torch.cat([pos_i1, neg_i1])
    i2 = torch.cat([pos_i2, neg_i2])
    pair_labels = torch.cat([torch.ones(n_pairs), torch.zeros(n_pairs)])

    # Shuffle
    perm = torch.randperm(len(i1), generator=rng)
    return i1[perm], i2[perm], pair_labels[perm]


def compute_embeddings(encoder, dataset: CapturePairDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ECAPA embeddings for all segments in the dataset.

    Args:
        encoder: Either a callable (pcm_batch -> embeddings) or a module
                 with .eval() and .encode_batch().
    """
    if hasattr(encoder, 'eval'):
        encoder.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            pcm, label = dataset[i]
            x = pcm.unsqueeze(0)
            if callable(encoder) and not hasattr(encoder, 'encode_batch'):
                emb = encoder(x).squeeze()
            else:
                emb = encoder.encode_batch(x).squeeze()
            all_embeddings.append(emb.cpu())
            all_labels.append(label)

    return torch.stack(all_embeddings), torch.tensor(all_labels)


def build_scorer(emb_dim: int) -> nn.Sequential:
    """Build the scorer MLP head."""
    return nn.Sequential(
        nn.Linear(emb_dim * 2, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


# ── Feature-Based Fine-Tuning ──────────────────────────────────────────────


def fine_tune_feature_based(
    model,
    dataset: CapturePairDataset,
    output_path: Path,
    num_epochs: int,
    lr: float,
) -> dict:
    """Feature-based fine-tuning: train an MLP head on frozen ECAPA embeddings."""
    print("Using feature-based fine-tuning (frozen encoder + trainable head)")

    print("Computing embeddings with frozen model...")
    embeddings, labels = compute_embeddings(model.encode_batch, dataset)
    print(f"  Embeddings shape: {embeddings.shape}")

    self_count = (labels == 1).sum().item()
    other_count = (labels == 0).sum().item()
    print(f"  Self embeddings: {self_count}")
    print(f"  Other embeddings: {other_count}")

    # Build pairs once (embeddings are frozen, so pairs don't change)
    pair_i1, pair_i2, pair_labels = build_pair_indices(labels)
    emb_dim = embeddings.shape[1]

    X = torch.cat([embeddings[pair_i1], embeddings[pair_i2]], dim=1)
    y = pair_labels.float()

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scorer = build_scorer(emb_dim)
    optimizer = optim.Adam(scorer.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining scorer: {len(X_train)} train, {len(X_test)} test pairs")
    history = _train_scorer_loop(scorer, X_train, y_train, X_test, y_test,
                                  optimizer, criterion, num_epochs)

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
        "num_self": self_count,
        "num_other": other_count,
        "final_test_accuracy": history[-1]["test_accuracy"] if history else 0,
    }, indent=2))

    return {
        "output_dir": str(output_path),
        "method": "feature_based",
        "num_epochs": num_epochs,
        "num_train_pairs": int(len(X_train)),
        "num_test_pairs": int(len(X_test)),
        "num_self": self_count,
        "num_other": other_count,
        "history": history,
    }


def _train_scorer_loop(
    scorer, X_train, y_train, X_test, y_test,
    optimizer, criterion, num_epochs,
) -> list[dict]:
    """Core scorer training loop. Used by both feature-based and full fine-tune."""
    history = []
    for epoch in range(num_epochs):
        scorer.train()
        logits = scorer(X_train).squeeze()
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}: train_loss={loss.item():.4f} test_acc={acc:.4f}")

    return history


# ── Full Fine-Tuning (Unfrozen Encoder Layers) ─────────────────────────────


def fine_tune_full(
    model,
    dataset: CapturePairDataset,
    output_path: Path,
    num_epochs: int = 50,
    lr: float = 1e-5,
    unfreeze_blocks: int = 2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    train_batch_size: int = 8,
) -> dict:
    """Full fine-tuning: unfreeze last N SE blocks + train scorer end-to-end.

    Uses mini-batch training so gradients flow through the unfrozen encoder
    layers into the scorer head. Embeddings are computed on-the-fly per batch.
    """
    print(f"Full fine-tuning: unfreezing last {unfreeze_blocks} ECAPA block(s)")

    emb_model = model.mods.embedding_model
    total_blocks = len(emb_model.blocks)

    # Freeze all blocks, then unfreeze the last N
    for param in emb_model.parameters():
        param.requires_grad = False

    unfrozen_block_indices = list(range(total_blocks - unfreeze_blocks, total_blocks))
    for idx in unfrozen_block_indices:
        for param in emb_model.blocks[idx].parameters():
            param.requires_grad = True

    unfrozen_params = sum(p.numel() for p in emb_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in emb_model.parameters())
    print(f"  Unfrozen: {unfrozen_params:,} / {total_params:,} params ({100*unfrozen_params/total_params:.1f}%)")
    print(f"  Unfrozen blocks: {unfrozen_block_indices} (0-indexed)")

    # Pre-compute mel features once (the expensive step) — but run full emb_model
    # forward per batch. This avoids manually reconstructing the ECAPA pipeline.
    print("Pre-computing mel features for all segments...")
    emb_model.eval()
    with torch.no_grad():
        all_pcms = torch.stack([dataset[i][0].clone().detach() for i in range(len(dataset))])
        feats = model.mods.compute_features(all_pcms)  # [N, time, n_mels]
        wav_lens = torch.ones(len(all_pcms))
        del all_pcms
    print(f"  Features: {feats.shape}")

    # Get labels from frozen model for building fixed train/test splits
    with torch.no_grad():
        _, labels_check = compute_embeddings(model, dataset)

    # Only unfreeze fc so projection adapts to fine-tuned block outputs.
    # mfa, asp, asp_bn stay frozen — they have ~12M params combined and
    # unfreezing them risks overfitting with our small dataset.
    if hasattr(emb_model, 'fc'):
        for param in emb_model.fc.parameters():
            param.requires_grad = True

    total_unfrozen = sum(p.numel() for p in emb_model.parameters() if p.requires_grad)
    print(f"  Total unfrozen (encoder): {total_unfrozen:,} / {total_params:,} params ({100*total_unfrozen/total_params:.1f}%)")

    pair_i1, pair_i2, pair_labels = build_pair_indices(labels_check)
    split = int(0.8 * len(pair_i1))
    train_i1, test_i1 = pair_i1[:split], pair_i1[split:]
    train_i2, test_i2 = pair_i2[:split], pair_i2[split:]
    train_y, test_y = pair_labels[:split], pair_labels[split:]

    emb_dim = 192
    scorer = build_scorer(emb_dim)

    trainable_params = list(scorer.parameters()) + [p for p in emb_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training: {len(train_i1)} train pairs, {len(test_i1)} test pairs")
    print(f"  lr={lr}, wd={weight_decay}, epochs={num_epochs}, batch_size={train_batch_size}")

    def _embed_batch(indices):
        """Run full ECAPA forward on pre-computed features for given indices."""
        # feats is (N, T, C), emb_model.forward does transpose internally
        # Output from forward is (B, 1, D) — squeeze the extra dim
        return emb_model(feats[indices]).squeeze(1)  # (batch, 192)

    history = []
    for epoch in range(num_epochs):
        emb_model.train()
        scorer.train()
        perm = torch.randperm(len(train_i1))
        total_loss, n_batches = 0.0, 0

        for b_start in range(0, len(perm), train_batch_size):
            b_idx = perm[b_start:b_start + train_batch_size]
            i1b, i2b, yb = train_i1[b_idx], train_i2[b_idx], train_y[b_idx].float()

            unique_idx = torch.unique(torch.cat([i1b, i2b]))
            embs_map = _embed_batch(unique_idx)
            idx_lookup = {uid.item(): embs_map[j] for j, uid in enumerate(unique_idx)}

            emb1 = torch.stack([idx_lookup[i1b[i].item()] for i in range(len(b_idx))])
            emb2 = torch.stack([idx_lookup[i2b[i].item()] for i in range(len(b_idx))])
            X_batch = torch.cat([emb1, emb2], dim=1)
            logits = scorer(X_batch).squeeze()
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Eval
        emb_model.eval()
        scorer.eval()
        with torch.no_grad():
            eval_embs = _embed_batch(torch.arange(len(feats)))
            te1 = eval_embs[test_i1]
            te2 = eval_embs[test_i2]
            tlogits = scorer(torch.cat([te1, te2], dim=1)).squeeze()
            tloss = criterion(tlogits, test_y.float())
            acc = ((torch.sigmoid(tlogits) > 0.5).float() == test_y.float()).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "test_loss": round(tloss.item(), 4),
            "test_accuracy": round(acc, 4),
        })
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}: train_loss={avg_loss:.4f} test_acc={acc:.4f}")

    # Save both the encoder and scorer
    save_path = output_path / "fine_tuned"
    save_path.mkdir(exist_ok=True)

    # Save encoder (just the unfrozen blocks + full model state for reproducibility)
    torch.save({
        "embedding_model_state": emb_model.state_dict(),
        "unfrozen_blocks": unfrozen_block_indices,
    }, save_path / "encoder.pt")
    torch.save(scorer.state_dict(), save_path / "scorer.pt")

    (save_path / "config.json").write_text(json.dumps({
        "fine_tuned": True,
        "method": "full_fine_tune",
        "unfrozen_blocks": unfrozen_block_indices,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "emb_dim": emb_dim,
        "num_train_pairs": int(len(train_i1)),
        "num_test_pairs": int(len(test_i1)),
        "unfrozen_params": unfrozen_params,
        "total_embedding_params": total_params,
        "final_test_accuracy": history[-1]["test_accuracy"] if history else 0,
    }, indent=2))

    return {
        "output_dir": str(output_path),
        "method": "full_fine_tune",
        "unfrozen_blocks": unfrozen_block_indices,
        "num_epochs": num_epochs,
        "num_train_pairs": int(len(train_i1)),
        "num_test_pairs": int(len(test_i1)),
        "unfrozen_params": unfrozen_params,
        "total_embedding_params": total_params,
        "history": history,
    }


# ── Orchestration ──────────────────────────────────────────────────────────


def fine_tune_ecapa(
    clean_captures_dir: str,
    other_speaker_dirs: list[str],
    num_epochs: int = 20,
    lr: float = 1e-4,
    segment_duration_s: float = 2.0,
    batch_size: int = 16,
    output_dir: str = "data/m4_finetuned_ecapa",
    device: str = "cpu",
    full_fine_tune: bool = False,
    unfreeze_blocks: int = 2,
    weight_decay: float = 1e-4,
) -> dict:
    """Fine-tune ECAPA-TDNN on captured audio pairs.

    Args:
        full_fine_tune: If True, unfreeze encoder layers and train end-to-end.
        unfreeze_blocks: Number of SE blocks to unfreeze (counted from the end).
        weight_decay: L2 regularization (full fine-tune only).
    """
    from speechbrain.inference.speaker import SpeakerRecognition

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading base ECAPA model...")
    cache_dir = output_path / "pretrained"
    cache_dir.mkdir(exist_ok=True)
    model = SpeakerRecognition.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir=str(cache_dir),
    )
    print(f"  Model loaded. Device: {device}")

    print("Preparing dataset...")
    self_pcms = load_npy_files(clean_captures_dir, "captured_*.npy")
    if not self_pcms:
        self_pcms = load_npy_files(clean_captures_dir, "agent_pcm_*.npy")
    print(f"  Self captures: {len(self_pcms)}")

    other_pcms = []
    for other_dir in other_speaker_dirs:
        pcms = load_npy_files(other_dir, "live_captured.npy")
        if not pcms:
            pcms = load_npy_files(other_dir, "human_negative_*.npy")
        other_pcms.extend(pcms)
    print(f"  Other-speaker captures: {len(other_pcms)}")

    dataset = CapturePairDataset(
        self_pcms=self_pcms,
        other_pcms=other_pcms,
        segment_duration_s=segment_duration_s,
    )
    print(f"  Total segments: {len(dataset)}")

    if full_fine_tune:
        return fine_tune_full(
            model, dataset, output_path,
            num_epochs=num_epochs,
            lr=lr,
            unfreeze_blocks=unfreeze_blocks,
            weight_decay=weight_decay,
            device=device,
            train_batch_size=batch_size,
        )
    else:
        return fine_tune_feature_based(
            model, dataset, output_path,
            num_epochs=num_epochs,
            lr=lr,
        )


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
    parser.add_argument("--full-fine-tune", action="store_true",
                        help="Unfreeze encoder layers and train end-to-end")
    parser.add_argument("--unfreeze-blocks", type=int, default=2,
                        help="Number of SE blocks to unfreeze (from the end, default: 2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight decay for full fine-tune only")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 ECAPA Fine-tuning")
    if args.full_fine_tune:
        print(f"  Mode: full (unfreeze last {args.unfreeze_blocks} blocks)")
    else:
        print("  Mode: feature-based (frozen encoder)")
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
        full_fine_tune=args.full_fine_tune,
        unfreeze_blocks=args.unfreeze_blocks,
        weight_decay=args.weight_decay,
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
