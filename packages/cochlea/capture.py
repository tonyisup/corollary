"""Cochlea — audio DSP pipeline.

Converts raw PCM to log-mel features.  In Phase 1+ also wraps
WebRTC AEC3 for real-time echo cancellation.
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import Tuple

# ── Global audio constants ────────────────────────────────────────────────
SR = 16000          # sample rate, Hz
HOP = 160           # hop length  (10 ms at 16 kHz)
WIN = 400           # window length (25 ms at 16 kHz)
N_MELS = 80


def pcm_to_mels(audio: np.ndarray) -> np.ndarray:
    """Convert float32 PCM (1D array) → log-mel spectrogram (T × N_MELS)."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (time, n_mels)


class EfferenceBuffer:
    """Rolling ring-buffer for efference-copy PCM.

    Stores the last `max_ms` of TTS output so the Cerebellum can
    compute predicted mel features on demand.
    """

    def __init__(self, max_ms: int = 3000) -> None:
        self.max_len = int(SR * max_ms / 1000)

    def append(self, pcm: np.ndarray) -> None:
        """Append PCM samples (overwrites oldest when full)."""
        raise NotImplementedError("TODO: implement ring-buffer")

    @property
    def audio(self) -> np.ndarray:
        """Return entire buffer as float32 array."""
        raise NotImplementedError("TODO: implement ring-buffer")
