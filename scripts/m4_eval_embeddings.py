#!/usr/bin/env python3
"""Evaluate embedding models on captured-audio M4 validation artifacts.

Compares ECAPA-TDNN, wav2vec2-base, and hubert-base on the task of
distinguishing self-voice from other-speaker audio in captured (room-colored) TTS.

Usage:
    .venv/bin/python scripts/m4_eval_embeddings.py \\
        --capture-dir data/m4_capture_validation_heldout_20260509 \\
        --register-from-clean data/m4_clean_captures_fixed \\
        --output /tmp/m4_embedding_eval.json
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

SR = 16000


class EmbeddingModel:
    """Unified interface for speaker embedding models."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.model = None
        self.sr = SR

    def load(self):
        raise NotImplementedError

    def get_embedding(self, pcm: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm < 1e-9:
            return 0.0
        return float(dot / norm)


class EmbeddingECAPA(EmbeddingModel):
    def __init__(self, **kwargs):
        super().__init__("ecapa-tdnn", **kwargs)

    def load(self):
        from stg.speaker import SpeakerIdentifier
        self.identifier = SpeakerIdentifier(use_real=True)
        self.identifier._load_model()

    def get_embedding(self, pcm: np.ndarray) -> np.ndarray:
        return self.identifier.get_embedding(pcm, sr=self.sr)


class EmbeddingWav2Vec2(EmbeddingModel):
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", **kwargs):
        super().__init__(f"wav2vec2-{model_name.split('/')[-1]}", **kwargs)
        self.model_name = model_name

    def load(self):
        from transformers import Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.eval()
        if self.device == "mps":
            self.model = self.model.to("mps")

    def get_embedding(self, pcm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
            if self.device == "mps":
                x = x.to("mps")
            output = self.model(x)
            emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return emb.astype(np.float32)


class EmbeddingHubert(EmbeddingModel):
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", **kwargs):
        super().__init__(f"hubert-{model_name.split('/')[-1]}", **kwargs)
        self.model_name = model_name

    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        if self.device == "mps":
            self.model = self.model.to("mps")

    def get_embedding(self, pcm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
            if self.device == "mps":
                x = x.to("mps")
            output = self.model(x)
            emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return emb.astype(np.float32)


def load_capture(capture_dir: str) -> dict:
    """Load a capture validation artifact."""
    path = Path(capture_dir)
    scenarios = {}
    for scenario_dir in sorted(path.iterdir()):
        if not scenario_dir.is_dir():
            continue
        audio = np.load(scenario_dir / "live_captured.npy").astype(np.float32)
        agent = np.load(scenario_dir / "agent_pcm.npy").astype(np.float32)
        with open(scenario_dir / "frame_records.json") as f:
            records = json.load(f)
        gt_verdicts = [r.get("verdict", "SILENCE") for r in records]
        scenarios[scenario_dir.name] = {
            "audio": audio,
            "agent_pcm": agent,
            "gt_verdicts": gt_verdicts,
        }
    return scenarios


def register_self_embedding(model: EmbeddingModel, clean_capture_dir: str) -> np.ndarray:
    """Compute self embedding from a clean capture."""
    cap_path = Path(clean_capture_dir)
    agent_files = sorted(cap_path.glob("agent_pcm_*.npy"))
    if not agent_files:
        return None
    agent_pcm = np.load(agent_files[-1]).astype(np.float32)
    # Use first 1 second for registration
    seg = agent_pcm[:min(len(agent_pcm), SR)]
    return model.get_embedding(seg)


def evaluate_model(
    model: EmbeddingModel,
    scenarios: dict,
    self_embedding: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """Evaluate an embedding model on self vs world classification."""
    results = {}
    for name, scenario in scenarios.items():
        audio = scenario["audio"]
        gt_verdicts = scenario["gt_verdicts"]
        frame_samples = int(SR * 30 / 1000)  # 30ms frames

        # Accumulate audio and compute embeddings at 300ms intervals
        buffer = np.array([], dtype=np.float32)
        predictions = []

        for fi, gt in enumerate(gt_verdicts):
            s_start = fi * frame_samples
            s_end = s_start + frame_samples
            if s_end > len(audio):
                break
            frame = audio[s_start:s_end]
            buffer = np.concatenate([buffer, frame])

            # Classify when buffer >= 300ms
            if len(buffer) >= int(SR * 0.3):
                try:
                    emb = model.get_embedding(buffer)
                    sim = model.similarity(emb, self_embedding)
                    predicted = "SELF" if sim > threshold else "WORLD"
                except Exception:
                    predicted = "WORLD"
                predictions.append((gt, predicted))
                # Keep last 100ms for overlap
                buffer = buffer[-int(SR * 0.1):]
            else:
                predictions.append((gt, "PENDING"))

        # Compute metrics (exclude PENDING)
        valid = [(g, p) for g, p in predictions if p != "PENDING"]
        tp = sum(1 for g, p in valid if p == "SELF" and g == "SELF")
        fp = sum(1 for g, p in valid if p == "SELF" and g != "SELF")
        fn = sum(1 for g, p in valid if p != "SELF" and g == "SELF")
        tn = sum(1 for g, p in valid if p != "SELF" and g != "SELF")

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        results[name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_valid": len(valid),
        }

    return results


def sweep_threshold(
    model: EmbeddingModel,
    scenarios: dict,
    self_embedding: np.ndarray,
    thresholds: list[float] = None,
) -> list[dict]:
    """Sweep classification thresholds and report F1."""
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(-10, 21)]

    results = []
    for thresh in thresholds:
        r = evaluate_model(model, scenarios, self_embedding, thresh)
        # Aggregate across scenarios
        total_tp = sum(v["tp"] for v in r.values())
        total_fp = sum(v["fp"] for v in r.values())
        total_fn = sum(v["fn"] for v in r.values())
        f1 = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
        results.append({"threshold": thresh, "f1": round(f1, 4), "tp": total_tp, "fp": total_fp, "fn": total_fn})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate embedding models for M4")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--register-from-clean", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--models", nargs="+", default=["ecapa", "wav2vec2", "hubert"])
    args = parser.parse_args()

    print(f"Loading capture from: {args.capture_dir}")
    scenarios = load_capture(args.capture_dir)
    print(f"Scenarios: {list(scenarios.keys())}")

    model_classes = {
        "ecapa": lambda: EmbeddingECAPA(device=args.device),
        "wav2vec2": lambda: EmbeddingWav2Vec2(device=args.device),
        "hubert": lambda: EmbeddingHubert(device=args.device),
    }

    all_results = {}

    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}")
        print(f"{'='*60}")

        model = model_classes[model_key]()
        t0 = time.time()
        model.load()
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Register self embedding
        if args.register_from_clean:
            self_emb = register_self_embedding(model, args.register_from_clean)
            if self_emb is None:
                print("  WARNING: No clean capture found, using first scenario agent_pcm")
                first_scenario = next(iter(scenarios.values()))
                self_emb = model.get_embedding(first_scenario["agent_pcm"][:SR])
        else:
            first_scenario = next(iter(scenarios.values()))
            self_emb = model.get_embedding(first_scenario["agent_pcm"][:SR])

        print(f"  Self embedding norm: {np.linalg.norm(self_emb):.4f}")

        # Sweep thresholds
        print("  Sweeping thresholds...")
        sweep = sweep_threshold(model, scenarios, self_emb)

        best = max(sweep, key=lambda x: x["f1"])
        print(f"  Best F1={best['f1']:.4f} at threshold={best['threshold']:.2f}")
        print(f"    TP={best['tp']} FP={best['fp']} FN={best['fn']}")

        # Detailed results at best threshold
        detailed = evaluate_model(model, scenarios, self_emb, best["threshold"])
        for scenario_name, metrics in detailed.items():
            print(f"  {scenario_name}: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

        all_results[model_key] = {
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "sweep": sweep,
            "detailed": detailed,
        }

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_key, results in all_results.items():
        print(f"  {model_key:>12s}: best F1={results['best_f1']:.4f} @ threshold={results['best_threshold']:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
