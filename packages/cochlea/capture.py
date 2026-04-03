"""Cochlea — audio DSP pipeline.

Converts raw PCM to log-mel features.  In Phase 1+ also wraps
WebRTC AEC3 for real-time echo cancellation.
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import List, Tuple

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
        self._buf: np.ndarray = np.zeros(self.max_len, dtype=np.float32)
        self._write_pos: int = 0
        self._filled: int = 0  # how many samples have been written total

    def append(self, pcm: np.ndarray) -> None:
        """Append PCM samples (overwrites oldest when full)."""
        n = len(pcm)
        if n == 0:
            return

        # Handle wrap-around in chunks
        start_idx = self._write_pos
        space_before_wrap = self.max_len - start_idx
        if n <= space_before_wrap:
            self._buf[start_idx:start_idx + n] = pcm
        else:
            # Write what fits before wrap
            self._buf[start_idx:] = pcm[:space_before_wrap]
            # Write remainder from the beginning
            remaining = n - space_before_wrap
            self._buf[:remaining] = pcm[space_before_wrap:]

        self._write_pos = (self._write_pos + n) % self.max_len
        self._filled = min(self._filled + n, self.max_len)

    @property
    def audio(self) -> np.ndarray:
        """Return entire buffer as float32 array.

        Returns all `self._filled` samples in correct temporal order,
        oldest first.
        """
        if not self._filled:
            return np.zeros(0, dtype=np.float32)

        if self._filled < self.max_len:
            # Buffer never wrapped — data is contiguous from 0
            return self._buf[:self._filled].copy()

        # Buffer has wrapped — data starts at write_pos
        result = np.empty(self.max_len, dtype=np.float32)
        end = self._write_pos
        result[:self.max_len - end] = self._buf[end:]
        result[self.max_len - end:] = self._buf[:end]
        return result

    @property
    def is_full(self) -> bool:
        """Whether the buffer has been filled at least once."""
        return self._filled >= self.max_len

    def clear(self) -> None:
        """Reset the buffer to empty."""
        self._buf.fill(0)
        self._write_pos = 0
        self._filled = 0
