"""STG — Speaker Identification & Embeddings.

M4: Extract speaker embeddings (ECAPA-TDNN via SpeechBrain) to distinguish
self-voice from others. Falls back to mock if SpeechBrain not available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


class PrototypeBankSelfVoiceDetector:
    """Self-voice detector using a bank of self and hard-negative prototypes.

    This strategy replaces the single-template + learned head setup with a
    multi-prototype verifier. It scores a probe by the margin between its
    strongest match to the self bank and its strongest match to the negative
    bank. In practice, this makes the detector less brittle to session-to-session
    self variation while still letting hard negatives pull the score down.
    """

    def __init__(self, identifier: SpeakerIdentifier, threshold: float = 0.0) -> None:
        self.identifier = identifier
        self.threshold = threshold
        self.self_bank: list[np.ndarray] = []
        self.neg_bank: list[np.ndarray] = []

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr[None, :]
        return arr.astype(np.float32)

    def register_self_bank(self, embeddings: list[np.ndarray] | np.ndarray) -> None:
        """Register one or more self prototypes.

        The inputs may be raw PCM segments or pre-computed embeddings. If the
        vectors look like embeddings (shape [..., 192]), they are stored directly;
        otherwise they are embedded with the configured SpeakerIdentifier.
        """
        self.self_bank = self._prepare_bank(embeddings)

    def register_negative_bank(self, embeddings: list[np.ndarray] | np.ndarray) -> None:
        """Register one or more negative prototypes."""
        self.neg_bank = self._prepare_bank(embeddings)

    def add_self_prototype(self, prototype: np.ndarray) -> None:
        """Append a single self prototype to the bank."""
        self.self_bank.extend(self._prepare_bank(prototype))

    def add_negative_prototype(self, prototype: np.ndarray) -> None:
        """Append a single negative prototype to the bank."""
        self.neg_bank.extend(self._prepare_bank(prototype))

    def save_bank(self, bank_path: str) -> None:
        """Persist the prototype bank to disk as an .npz file."""
        self._save_bank(bank_path)

    def load_bank(self, bank_path: str) -> None:
        """Load a prototype bank from disk."""
        bank = np.load(bank_path, allow_pickle=True)
        self.self_bank = [row.astype(np.float32) for row in np.asarray(bank["self_embeddings"], dtype=np.float32)]
        self.neg_bank = [row.astype(np.float32) for row in np.asarray(bank["neg_embeddings"], dtype=np.float32)]
        if "threshold" in bank:
            self.threshold = float(np.asarray(bank["threshold"]).reshape(-1)[0])

    def _save_bank(self, bank_path: str) -> None:
        np.savez(
            bank_path,
            self_embeddings=np.stack(self.self_bank) if self.self_bank else np.zeros((0, 192), dtype=np.float32),
            neg_embeddings=np.stack(self.neg_bank) if self.neg_bank else np.zeros((0, 192), dtype=np.float32),
            threshold=np.array([self.threshold], dtype=np.float32),
        )

    def _prepare_bank(self, items: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        if isinstance(items, np.ndarray):
            items = [items]
        bank: list[np.ndarray] = []
        for item in items:
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim == 1 and arr.shape[0] == 192:
                bank.append(arr.astype(np.float32))
                continue
            if arr.ndim == 2 and arr.shape[1] == 192:
                bank.extend([row.astype(np.float32) for row in arr])
                continue
            emb = self.identifier.get_embedding(arr)
            bank.append(emb.astype(np.float32))
        return bank

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return dot / denom

    def score(self, pcm: np.ndarray) -> float:
        """Return the self-versus-negative margin score."""
        emb = self.identifier.get_embedding(pcm)
        self_score = max((self._cosine(emb, proto) for proto in self.self_bank), default=-1.0)
        neg_score = max((self._cosine(emb, proto) for proto in self.neg_bank), default=-1.0)
        return self_score - neg_score

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely Corollary's own voice."""
        margin = self.score(pcm)
        return (margin >= self.threshold), margin

    def is_world(self, pcm: np.ndarray) -> tuple[bool, float]:
        """Check if the given audio is likely NOT Corollary's own voice."""
        is_self, margin = self.is_self(pcm)
        return (not is_self), (-margin)


@dataclass
class M4ArbitrationResult:
    """Outcome of dual M4 arbitration between the bank and scorer."""

    final_is_self: bool
    final_confidence: float
    primary_source: str
    bank_is_self: Optional[bool]
    bank_confidence: Optional[float]
    scorer_is_self: Optional[bool]
    scorer_confidence: Optional[float]
    disagreed: bool
    degraded_mode: bool

    def to_metadata_dict(self) -> dict:
        return {
            "m4_primary_source": self.primary_source,
            "m4_bank_is_self": self.bank_is_self,
            "m4_bank_confidence": self.bank_confidence,
            "m4_scorer_is_self": self.scorer_is_self,
            "m4_scorer_confidence": self.scorer_confidence,
            "m4_disagreed": self.disagreed,
            "m4_degraded_mode": self.degraded_mode,
            "m4_final_is_self": self.final_is_self,
            "m4_final_confidence": self.final_confidence,
        }


class GatedM4Arbitrator:
    """Conservative dual-path M4 detector.

    The prototype bank is the primary live gate and the fine-tuned scorer acts
    as a cross-check. Agreement is required to return SELF; disagreement is
    logged and treated conservatively.
    """

    def __init__(
        self,
        bank_detector: Optional[object] = None,
        scorer_detector: Optional[object] = None,
        threshold: float = 0.5,
    ) -> None:
        self.bank_detector = bank_detector
        self.scorer_detector = scorer_detector
        self.threshold = threshold
        self.last_result: Optional[M4ArbitrationResult] = None

    def register_self(self, pcm: np.ndarray) -> None:
        if self.bank_detector is not None:
            if hasattr(self.bank_detector, "add_self_prototype"):
                self.bank_detector.add_self_prototype(pcm)
            elif hasattr(self.bank_detector, "register_self"):
                self.bank_detector.register_self(pcm)

        if self.scorer_detector is not None and hasattr(self.scorer_detector, "register_self"):
            self.scorer_detector.register_self(pcm)

    def add_self_prototype(self, prototype: np.ndarray) -> None:
        if self.bank_detector is not None and hasattr(self.bank_detector, "add_self_prototype"):
            self.bank_detector.add_self_prototype(prototype)
        elif self.bank_detector is not None and hasattr(self.bank_detector, "register_self"):
            self.bank_detector.register_self(prototype)

    def load_bank(self, bank_path: str) -> None:
        if self.bank_detector is not None and hasattr(self.bank_detector, "load_bank"):
            self.bank_detector.load_bank(bank_path)

    def load_scorer(self, scorer_path: str) -> None:
        if self.scorer_detector is not None and hasattr(self.scorer_detector, "load_scorer"):
            self.scorer_detector.load_scorer(scorer_path)

    @staticmethod
    def _normalize_detector_result(result) -> tuple[Optional[bool], Optional[float]]:
        if result is None:
            return None, None
        is_self, confidence = result
        return (
            bool(is_self) if is_self is not None else None,
            float(confidence) if confidence is not None else None,
        )

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        bank_is_self, bank_confidence = (None, None)
        scorer_is_self, scorer_confidence = (None, None)

        if self.bank_detector is not None and hasattr(self.bank_detector, "is_self"):
            bank_is_self, bank_confidence = self._normalize_detector_result(
                self.bank_detector.is_self(pcm)
            )

        if self.scorer_detector is not None and hasattr(self.scorer_detector, "is_self"):
            scorer_is_self, scorer_confidence = self._normalize_detector_result(
                self.scorer_detector.is_self(pcm)
            )

        available_bank = bank_is_self is not None
        available_scorer = scorer_is_self is not None
        if not available_bank and not available_scorer:
            self.last_result = M4ArbitrationResult(
                final_is_self=False,
                final_confidence=0.0,
                primary_source="fallback",
                bank_is_self=None,
                bank_confidence=None,
                scorer_is_self=None,
                scorer_confidence=None,
                disagreed=False,
                degraded_mode=True,
            )
            return False, 0.0

        if available_bank and available_scorer:
            disagreed = bank_is_self != scorer_is_self
            if bank_is_self and scorer_is_self:
                final_is_self = True
                final_confidence = float(bank_confidence or 0.0)
                primary_source = "bank"
                degraded_mode = False
            elif (not bank_is_self) and (not scorer_is_self):
                final_is_self = False
                final_confidence = max(bank_confidence or 0.0, scorer_confidence or 0.0)
                primary_source = "bank"
                degraded_mode = False
            else:
                final_is_self = False
                final_confidence = min(
                    [c for c in (bank_confidence, scorer_confidence) if c is not None],
                    default=0.0,
                )
                primary_source = "bank" if bank_is_self else "scorer"
                degraded_mode = True
        else:
            final_is_self = bool(bank_is_self if available_bank else scorer_is_self)
            final_confidence = float(bank_confidence if available_bank else scorer_confidence or 0.0)
            primary_source = "bank" if available_bank else "scorer"
            disagreed = False
            degraded_mode = False

        self.last_result = M4ArbitrationResult(
            final_is_self=final_is_self,
            final_confidence=float(final_confidence),
            primary_source=primary_source,
            bank_is_self=bank_is_self,
            bank_confidence=bank_confidence,
            scorer_is_self=scorer_is_self,
            scorer_confidence=scorer_confidence,
            disagreed=disagreed,
            degraded_mode=degraded_mode,
        )
        return final_is_self, float(final_confidence)

    def is_world(self, pcm: np.ndarray) -> tuple[bool, float]:
        is_self, confidence = self.is_self(pcm)
        return (not is_self), (1.0 - confidence)
