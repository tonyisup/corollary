"""STG — Speaker Identification & Embeddings.

M4: Extract speaker embeddings (ECAPA-TDNN via SpeechBrain) to distinguish
self-voice from others. Falls back to mock if SpeechBrain not available.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


class SpeakerIdentifier:
    """Speaker Identification unit.

    Computes speaker embeddings and compares them to known templates.
    Uses SpeechBrain ECAPA-TDNN (192-dim embeddings).
    """

    def __init__(self, use_real: bool = True) -> None:
        self._model = None
        self._use_real = use_real

    def _load_model(self):
        if self._model is not None:
            return
        if not self._use_real:
            self._model = "mock"
            return
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            self._model = SpeakerRecognition.from_hparams(
                source='speechbrain/spkrec-ecapa-voxceleb',
                savedir='~/.cache/speechbrain/spkrec-ecapa-voxceleb',
            )
        except Exception as e:
            print(f"WARNING: Could not load ECAPA model: {e}. Using mock embeddings.")
            self._model = "mock"

    def get_embedding(self, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Compute speaker embedding for a chunk of audio.

        Returns: 192-dim embedding vector (ECAPA-TDNN).
        """
        self._load_model()

        if self._model == "mock":
            # Deterministic mock based on audio content
            content_hash = int(np.sum(np.abs(pcm)) * 1000) % 2**32
            np.random.seed(content_hash)
            return np.random.randn(192).astype(np.float32)

        # Normalize
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
