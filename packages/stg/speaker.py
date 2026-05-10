"""STG — Speaker Identification & Embeddings.

M4: Extract speaker embeddings (ECAPA-TDNN via SpeechBrain) to distinguish
self-voice from others. Falls back to mock if SpeechBrain not available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SpeakerIdentifier:
    """Speaker Identification unit.

    Computes speaker embeddings and compares them to known templates.
    Supports multiple backends:
      - "ecapa": SpeechBrain ECAPA-TDNN (192-dim, default)
      - "wav2vec2": facebook/wav2vec2-base-960h (768-dim)
      - "hubert": facebook/hubert-base-ls960 (768-dim)
      - "mock": deterministic hash-based mock
    """

    def __init__(self, use_real: bool = True, backend: str = "ecapa") -> None:
        self._model = None
        self._use_real = use_real
        self._backend = backend
        self._torch_model = None

    def _load_model(self):
        if self._model is not None:
            return
        if not self._use_real or self._backend == "mock":
            self._model = "mock"
            return

        if self._backend == "ecapa":
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                cache_dir = Path(os.path.expanduser(
                    '~/.cache/speechbrain/spkrec-ecapa-voxceleb'
                ))
                self._model = SpeakerRecognition.from_hparams(
                    source='speechbrain/spkrec-ecapa-voxceleb',
                    savedir=str(cache_dir),
                )
            except Exception as e:
                print(f"WARNING: Could not load ECAPA model: {e}. Using mock embeddings.")
                self._model = "mock"

        elif self._backend in ("wav2vec2", "hubert"):
            try:
                from transformers import Wav2Vec2Model, AutoModel
                if self._backend == "wav2vec2":
                    self._torch_model = Wav2Vec2Model.from_pretrained(
                        "facebook/wav2vec2-base-960h"
                    )
                else:
                    self._torch_model = AutoModel.from_pretrained(
                        "facebook/hubert-base-ls960"
                    )
                self._torch_model.eval()
                self._model = self._backend
            except Exception as e:
                print(f"WARNING: Could not load {self._backend}: {e}. Using mock.")
                self._model = "mock"

        else:
            print(f"WARNING: Unknown backend '{self._backend}'. Using mock.")
            self._model = "mock"

    def get_embedding(self, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Compute speaker embedding for a chunk of audio.

        Returns: embedding vector (192-dim for ECAPA, 768-dim for wav2vec2/hubert).
        """
        self._load_model()

        if self._model == "mock":
            content_hash = int(np.sum(np.abs(pcm)) * 1000) % 2**32
            np.random.seed(content_hash)
            return np.random.randn(192).astype(np.float32)

        if self._backend in ("wav2vec2", "hubert") and self._torch_model is not None:
            pcm = pcm.astype(np.float32)
            peak = np.max(np.abs(pcm))
            if peak > 0:
                pcm = pcm / peak * 0.7
            with torch.no_grad():
                x = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
                output = self._torch_model(x)
                emb = output.last_hidden_state.mean(dim=1).squeeze().numpy()
            return emb.astype(np.float32)

        # ECAPA path
        pcm = pcm.astype(np.float32)
        peak = np.max(np.abs(pcm))
        if peak > 0:
            pcm = pcm / peak * 0.7
        with torch.no_grad():
            emb = self._model.encode_batch(
                torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
            ).squeeze().numpy()
        return emb.astype(np.float32)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm < 1e-9:
            return 0.0
        return float(dot / norm)


class SelfVoiceDetector:
    """Specialized detector for Corollary's own voice.

    Uses ECAPA-TDNN embeddings to distinguish self from world during speech.
    This enables hard SELF attribution in the ThalamicAttributor even when
    VAD detects speech (overriding the conservative M2 BOTH policy).
    """

    def __init__(self, identifier: SpeakerIdentifier, threshold: float = 0.4) -> None:
        self.identifier = identifier
        self.self_embedding: Optional[np.ndarray] = None
        self.threshold = threshold  # Cosine similarity threshold for self

    def register_self(self, pcm: np.ndarray) -> None:
        """Register Corollary's own voice template."""
        self.self_embedding = self.identifier.get_embedding(pcm)

    def register_self_embedding(self, embedding: np.ndarray) -> None:
        """Register Corollary's own voice from a pre-computed embedding."""
        self.self_embedding = embedding.astype(np.float32)

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely Corollary's own voice.

        Returns (is_self, confidence) where confidence is the cosine similarity
        between the input embedding and the registered self template.
        """
        if self.self_embedding is None:
            return False, 0.0

        emb = self.identifier.get_embedding(pcm)
        sim = self.identifier.similarity(emb, self.self_embedding)
        return (sim >= self.threshold), sim

    def is_world(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely NOT Corollary's own voice."""
        is_self, sim = self.is_self(pcm)
        return (not is_self), (1.0 - sim)


class FineTunedSelfVoiceDetector:
    """Self-voice detector using a fine-tuned scorer on frozen ECAPA embeddings.

    This wraps a frozen SpeakerIdentifier (ECAPA) with a trained projection head
    that maps (self_emb, input_emb) pairs to a same/different speaker score.
    The scorer is trained on room-colored captured audio to improve discrimination
    in that domain.

    Usage:
        identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
        detector = FineTunedSelfVoiceDetector(identifier)
        detector.load_scorer("data/m4_finetuned_ecapa/fine_tuned/scorer.pt")
        detector.register_self(clean_pcm)
        is_self, conf = detector.is_self(test_pcm)
    """

    def __init__(self, identifier: SpeakerIdentifier, threshold: float = 0.5) -> None:
        self.identifier = identifier
        self.self_embedding: Optional[np.ndarray] = None
        self.threshold = threshold
        self.scorer: Optional[torch.nn.Module] = None
        self._scorer_loaded = False

    def load_scorer(self, scorer_path: str) -> None:
        """Load a trained scorer from disk."""
        import torch.nn as nn
        emb_dim = 192  # ECAPA embedding dimension
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.scorer.load_state_dict(torch.load(scorer_path, map_location="cpu"))
        self.scorer.eval()
        self._scorer_loaded = True
        print(f"Loaded fine-tuned scorer from {scorer_path}")

    def register_self(self, pcm: np.ndarray) -> None:
        """Register Corollary's own voice template."""
        self.self_embedding = self.identifier.get_embedding(pcm)

    def register_self_embedding(self, embedding: np.ndarray) -> None:
        """Register Corollary's own voice from a pre-computed embedding."""
        self.self_embedding = embedding.astype(np.float32)

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely Corollary's own voice.

        Uses the fine-tuned scorer if available, otherwise falls back to
        cosine similarity.
        """
        if self.self_embedding is None:
            return False, 0.0

        emb = self.identifier.get_embedding(pcm)

        if self._scorer_loaded and self.scorer is not None:
            import torch
            with torch.no_grad():
                x = torch.cat([
                    torch.tensor(self.self_embedding, dtype=torch.float32),
                    torch.tensor(emb, dtype=torch.float32),
                ]).unsqueeze(0)
                logit = self.scorer(x).squeeze().item()
                prob = float(torch.sigmoid(torch.tensor(logit)))
            return (prob >= self.threshold), prob

        # Fallback to cosine similarity
        sim = self.identifier.similarity(emb, self.self_embedding)
        return (sim >= self.threshold), sim

    def is_world(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely NOT Corollary's own voice."""
        is_self, conf = self.is_self(pcm)
        return (not is_self), (1.0 - conf)
