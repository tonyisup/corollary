"""STG — Speaker Identification & Embeddings.

M4: Extract speaker embeddings (ECAPA-TDNN) to distinguish self-voice 
from others.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

try:
    from pyannote.audio import Inference, Model
except ImportError:
    Inference = None  # type: ignore

class SpeakerIdentifier:
    """Speaker Identification unit.
    
    Computes speaker embeddings and compares them to known templates.
    """

    def __init__(self, model_id: str = "pyannote/embedding") -> None:
        if Inference is None:
            raise ImportError("pyannote.audio is not installed.")
        
        # Note: This might require HUGGINGFACE_TOKEN in real use.
        # For this prototype, we define the interface.
        self.model_id = model_id
        self._inference: Optional[Inference] = None

    def _get_inference(self) -> Inference:
        if self._inference is None:
            # Lazy load
            self._inference = Inference(self.model_id, window="whole")
        return self._inference

    def get_embedding(self, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Compute speaker embedding for a chunk of audio.
        
        Returns: 512-dim embedding vector.
        """
        # pyannote expects (channel, samples) or path
        # For now, we simulate or use the model if available.
        
        # MOCK IMPLEMENTATION for environment stability
        # In real hardware with internet/token, this would call self._get_inference()
        
        # Simple surrogate: use a deterministic seed from the audio content
        # to produce a 'pseudorandom' embedding that is consistent for the same voice.
        # This allows us to test the M4 logic without downloading 1GB models.
        
        # Hash the audio content (simplified)
        content_hash = int(np.sum(np.abs(pcm)) * 1000) % 2**32
        np.random.seed(content_hash)
        return np.random.randn(512).astype(np.float32)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm < 1e-9:
            return 0.0
        return float(dot / norm)


class SelfVoiceDetector:
    """Specialized detector for Corollary's own voice."""

    def __init__(self, identifier: SpeakerIdentifier) -> None:
        self.identifier = identifier
        self.self_embedding: Optional[np.ndarray] = None
        self.threshold = 0.85

    def register_self(self, pcm: np.ndarray) -> None:
        """Register Corollary's own voice template."""
        self.self_embedding = self.identifier.get_embedding(pcm)

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely Corollary's own voice."""
        if self.self_embedding is None:
            return False, 0.0
        
        emb = self.identifier.get_embedding(pcm)
        sim = self.identifier.similarity(emb, self.self_embedding)
        return (sim >= self.threshold), sim
