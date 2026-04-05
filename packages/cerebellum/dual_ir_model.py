"""Cerebellum — Dual-IR Forward Model.

Learn the room impulse response for a stereo speaker setup.
Two speakers play the SAME monaural signal, but each has its own
acoustic path to the microphone.

The captured audio is the sum of both paths:
    captured(t) = convolve(TTS, IR_left) + convolve(TTS, IR_right)

When TTS is the same on both channels (mono), mathematically:
    captured(t) = convolve(TTS, IR_left + IR_right)

So the composite IR is what the NLMS actually learns.
But modeling it as two paths explicitly:
  - Captures the physical structure (two sources, one mic)
  - Prepares for future stereo TTS (different signals per speaker)
  - Allows per-speaker trust tracking (e.g. "left speaker moved")

Usage:
    cerebellum = DualIRBabblingCerebellum()
    error = cerebellum.update_stereo(left_pcm, right_pcm, captured_pcm)
    error = cerebellum.update_mono(mono_pcm, captured_pcm)  # Convenience
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from cochlea.capture import pcm_to_mels, SR
from cerebellum.trust_model import TrustTracker, TrustState
from typing import List, Optional


class DualIRBabblingCerebellum:
    """Forward model with separate IRs for two speaker paths.

    Each IR models one acoustic path: speaker → mic.
    In mono mode (same signal on both speakers), the NLMS learns
    the composite IR (IR_left + IR_right) distributed across both
    IR arrays.

    In stereo mode (different signals), the NLMS learns each IR
    independently.
    """

    def __init__(
        self,
        ir_length_ms: int = 300,
        sample_rate: int = SR,
        nlms_mu: float = 0.25,
        nlms_epsilon: float = 1e-6,
    ) -> None:
        self.sample_rate = sample_rate
        self.ir_length = int(sample_rate * ir_length_ms / 1000)
        self.nlms_mu = nlms_mu
        self.nlms_epsilon = nlms_epsilon

        # Two separate room IRs (initialized as delta = "no room")
        self.ir_left: np.ndarray = np.zeros(self.ir_length, dtype=np.float32)
        self.ir_left[0] = 1.0
        self.ir_right: np.ndarray = np.zeros(self.ir_length, dtype=np.float32)
        self.ir_right[0] = 1.0

        # Learning state
        self.total_updates: int = 0
        self.prediction_errors: List[float] = []

        # Trust: composite trust across both IRs
        self.trust = TrustTracker()

    def _nlms_update_stereo(
        self,
        left: np.ndarray,
        right: np.ndarray,
        captured: np.ndarray,
    ) -> np.ndarray:
        """NLMS update for dual-path stereo.

        Each path has its own IR, updated based on its contribution
        to the total prediction error.

        In mono mode (left ≈ right), both IRs share the gradient equally
        (mu/2 each) so the composite update stays within NLMS convergence.
        In stereo mode, each IR learns its own path independently.
        """
        N = self.ir_length
        mu = self.nlms_mu
        eps = self.nlms_epsilon

        # Detect mono: if left and right are approximately equal
        mono = (np.abs(left - right).sum() < 0.001 * np.abs(left).sum())

        max_len = min(min(len(left), len(right)), len(captured))
        left = left[:max_len]
        right = right[:max_len]
        captured = captured[:max_len]

        # Pad inputs for look-back
        pad_left = np.zeros(N - 1, dtype=np.float32)
        pad_right = np.zeros(N - 1, dtype=np.float32)
        left_ext = np.concatenate([pad_left, left, pad_left])
        right_ext = np.concatenate([pad_right, right, pad_right])

        # Prediction from both paths
        pred_left = fftconvolve(left, self.ir_left, mode="full")[:max_len].astype(np.float32)
        pred_right = fftconvolve(right, self.ir_right, mode="full")[:max_len].astype(np.float32)
        prediction = pred_left + pred_right

        error = captured - prediction

        # Update both IRs
        for t in range(max_len):
            e = error[t]
            if abs(e) < 1e-10:
                continue

            # Path input vectors
            x_left = left_ext[t:t + N][::-1]
            x_right = right_ext[t:t + N][::-1]

            x_power_left = float(np.dot(x_left, x_left))
            x_power_right = float(np.dot(x_right, x_right))

            # In mono mode, split equally (50/50) and dampen to prevent
            # the pair from diverging
            mu_l = mu_r = mu / 4.0  # Quarter step per path to keep combined at mu/2

            if x_power_left > eps:
                self.ir_left += mu_l * e * x_left / (x_power_left + eps)
            if x_power_right > eps:
                self.ir_right += mu_r * e * x_right / (x_power_right + eps)

            # In mono mode: gently pull the two IRs toward their average
            # This prevents underdetermined drift while still allowing
            # the composite to learn
            if mono:
                avg = (self.ir_left + self.ir_right) / 2.0
                coupling = 0.001  # very gentle pull
                self.ir_left += coupling * (avg - self.ir_left)
                self.ir_right += coupling * (avg - self.ir_right)

        return error

    def update_stereo(
        self,
        left_pcm: np.ndarray,
        right_pcm: np.ndarray,
        captured_pcm: np.ndarray,
    ) -> float:
        """Update both IRs from stereo playback capture."""
        known_left = self._normalize(left_pcm)
        known_right = self._normalize(right_pcm)
        captured = self._normalize(captured_pcm)
        error = self._nlms_update_stereo(known_left, known_right, captured)
        self.total_updates += 1
        self.prediction_errors.append(float(np.sqrt(np.mean(error ** 2))))
        error_db = 20 * np.log10(max(self.prediction_errors[-1], 1e-10))
        if error_db < -30:
            self.trust.confirm(error_db)
        elif error_db > -10:
            self.trust.fail()
        else:
            self.trust.confirm(error_db)
        return self.prediction_errors[-1]

    def update_mono(self, mono_pcm: np.ndarray, captured_pcm: np.ndarray) -> float:
        """Both speakers play the same mono signal."""
        return self.update_stereo(mono_pcm, mono_pcm, captured_pcm)

    def predict_mel(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the mel spectrogram the mic should hear."""
        pred_pcm = self.predict_pcm(efference_pcm)
        return pcm_to_mels(pred_pcm)

    def predict_pcm(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the raw PCM the mic should receive."""
        pred_left = fftconvolve(efference_pcm, self.ir_left, mode="full")
        pred_right = fftconvolve(efference_pcm, self.ir_right, mode="full")
        combined = pred_left + pred_right
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined = combined / peak * 0.9
        return combined.astype(np.float32)

    def get_composite_ir(self) -> np.ndarray:
        """Return the composite IR (left + right)."""
        return (self.ir_left + self.ir_right).copy()

    def trust_status(self) -> dict:
        return self.trust.get_status()

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalize(pcm: np.ndarray) -> np.ndarray:
        energy = np.sqrt(np.sum(pcm ** 2))
        if energy < 1e-10:
            return pcm
        return pcm / energy
