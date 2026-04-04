"""Cerebellum — Adaptive Forward Model.

Learn the room impulse response through babbling (self-generated audio),
not by being told. The system:
    1. Emits a known sound (chirp, vowel, utterance)
    2. Captures the room-colored echo through the microphone
    3. Compares prediction (clean efference copy) with observation
    4. Updates its internal room IR estimate via adaptive filtering
    5. Uses the learned IR for future forward-model predictions

This mirrors how infants learn room acoustics: babble, listen, update.
No chirp-deconvolution shortcuts. Pure experience-driven learning.

Usage:
    from cerebellum.adaptive_model import BabblingCerebellum

    cerebellum = BabblingCerebellum(ir_length_ms=300)

    # One babble loop: known PCM out, captured PCM back
    cerebellum.update(known_pcm, captured_pcm)

    # Predict what the mic will hear for a new utterance
    predicted_mel = cerebellum.predict_mel(new_pcm)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from cochlea.capture import pcm_to_mels, SR
from typing import List, Tuple


class BabblingCerebellum:
    """Forward model that learns the room impulse response through babbling.

    Maintains an estimated room_ir that is updated incrementally using
    normalized least mean squares (NLMS) on the prediction error.

    The learned IR is then used to convolve efference copies for
    forward-model predictions that account for real room acoustics.
    """

    def __init__(
        self,
        ir_length_ms: int = 300,
        sample_rate: int = SR,
        learning_rate: float = 0.1,
        nlms_mu: float = 0.05,
        nlms_epsilon: float = 1e-6,
    ) -> None:
        self.sample_rate = sample_rate
        self.ir_length = int(sample_rate * ir_length_ms / 1000)
        self.learning_rate = learning_rate

        # NLMS adaptive filter parameters
        self.nlms_mu = nlms_mu        # step size (0 < mu < 2 for convergence)
        self.nlms_epsilon = nlms_epsilon  # numerical stability

        # Estimated room impulse response (initialized as a delta — "no room")
        self.room_ir: np.ndarray = np.zeros(self.ir_length, dtype=np.float32)
        self.room_ir[0] = 1.0  # start: identity (dry signal)

        # Learning state
        self.total_updates: int = 0
        self.prediction_errors: List[float] = []
        self.convergence_history: List[float] = []

    def update(
        self,
        known_pcm: np.ndarray,
        captured_pcm: np.ndarray,
    ) -> float:
        """Update the estimated room IR by minimizing prediction error.

        This uses NLMS (Normalized Least Mean Squares) adaptive filtering
        to learn the room IR from one babbling pair.

        Parameters
        ----------
        known_pcm : np.ndarray
            The known input signal (what was sent to speakers), float32.
        captured_pcm : np.ndarray
            What the microphone captured, float32.

        Returns
        -------
        float
            RMS prediction error for this update.
        """
        # Normalize inputs
        known = self._normalize(known_pcm)
        captured = self._normalize(captured_pcm)

        # Align timing via cross-correlation
        # This is the biologically honest way to handle the delay between
        # when a sound is produced and when it's heard
        delay, captured = self._align_signals(known, captured)

        # NLMS update
        error = self._nlms_update(known, captured)

        self.total_updates += 1
        self.prediction_errors.append(float(np.sqrt(np.mean(error ** 2))))
        self.convergence_history.append(self.prediction_errors[-1])

        return self.prediction_errors[-1]

    def predict_mel(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict what the cochlea will hear given an efference copy.

        Convolves the efference copy with the learned room IR,
        then converts to mel spectrogram.

        Parameters
        ----------
        efference_pcm : np.ndarray
            Float32 PCM audio (efference copy from Broca).

        Returns
        -------
        np.ndarray
            Predicted mel spectrogram, shape (T, 80), in dB.
        """
        # Convolve with learned room IR
        roomed_pcm = fftconvolve(efference_pcm, self.room_ir, mode="full").astype(
            np.float32
        )

        # Normalize to match expected level
        peak = np.max(np.abs(roomed_pcm))
        if peak > 0:
            roomed_pcm = roomed_pcm / peak * 0.9

        return pcm_to_mels(roomed_pcm)

    def predict_pcm(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the raw PCM the cochlea will receive.

        Parameters
        ----------
        efference_pcm : np.ndarray
            Float32 PCM audio (efference copy from Broca).

        Returns
        -------
        np.ndarray
            Predicted room-colored PCM audio.
        """
        roomed = fftconvolve(efference_pcm, self.room_ir, mode="full").astype(
            np.float32
        )
        return roomed / max(np.max(np.abs(roomed)), 1e-10) * 0.9

    def is_ready(self, min_updates: int = 3) -> bool:
        """Check if the forward model has been updated enough to be useful.

        Parameters
        ----------
        min_updates : int
            Minimum number of babbling updates before the model is considered
            ready.
        """
        return self.total_updates >= min_updates

    def get_learned_ir(self) -> np.ndarray:
        """Return a copy of the current learned room IR."""
        return self.room_ir.copy()

    def get_convergence(self) -> dict:
        """Return convergence statistics."""
        if not self.prediction_errors:
            return {"updates": 0, "current_error_db": None, "trend": None}

        errors = self.prediction_errors
        current = errors[-1]
        trend = "improving" if len(errors) >= 2 and errors[-1] < errors[-2] else "stable"

        return {
            "updates": self.total_updates,
            "current_error": current,
            "current_error_db": 20 * np.log10(max(current, 1e-10)),
            "mean_error": float(np.mean(errors)),
            "min_error": float(np.min(errors)),
            "trend": trend,
        }

    # ── Private helpers ─────────────────────────────────────────────────

    def _normalize(self, pcm: np.ndarray) -> np.ndarray:
        """Normalize PCM to unit energy."""
        energy = np.sqrt(np.sum(pcm ** 2))
        if energy < 1e-10:
            return pcm
        return pcm / energy

    def _align_signals(
        self,
        known: np.ndarray,
        captured: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """Find and correct timing offset between known and captured signals.

        Uses cross-correlation to find the delay at which the known signal
        best matches the captured signal. This handles the sound propagation
        delay between speaker and microphone.

        Returns
        -------
        Tuple[int, np.ndarray]
            (delay_samples, aligned_captured)
        """
        # Cross-correlate to find delay
        correlation = np.correlate(
            captured,
            known,
            mode="full",
        )
        peak_idx = np.argmax(np.abs(correlation))
        delay = peak_idx - len(known) + 1

        # Trim captured signal to align with known signal
        max_len = min(len(captured) - abs(delay), len(known))
        if delay >= 0:
            captured = captured[delay:delay + max_len]
        else:
            captured = captured[:max_len]
        known = known[:max_len]

        return delay, captured

    def _nlms_update(self, known: np.ndarray, captured: np.ndarray) -> np.ndarray:
        """NLMS adaptive filter update.

        Estimates the system impulse response by minimizing the mean
        squared error between the predicted output (known * estimated_ir)
        and the actual captured output.

        Parameters
        ----------
        known : np.ndarray
            Normalized input signal (what was played).
        captured : np.ndarray
            Normalized output signal (what was captured).

        Returns
        -------
        np.ndarray
            Prediction error vector.
        """
        # Pad known signal for convolution
        output_len = len(known) + self.ir_length - 1
        known_padded = np.zeros(output_len, dtype=np.float32)
        known_padded[:len(known)] = known

        # Current prediction: known * estimated_ir
        prediction = fftconvolve(known, self.room_ir, mode="full").astype(np.float32)

        # Trim to match captured length
        pred_len = min(len(prediction), len(captured))
        prediction = prediction[:pred_len]
        captured = captured[:pred_len]

        # Prediction error
        error = captured - prediction

        # NLMS update: adjust IR in the direction of the gradient
        # w(n+1) = w(n) + mu * e(n) * x(n) / (||x(n)||^2 + epsilon)
        # For each filter tap, the gradient is the input signal at that lag
        for tap in range(self.ir_length):
            # Input at this tap (delayed version of known signal)
            x_tap = known_padded[:len(known) - self.ir_length + 1 + tap]
            e_tap = error
            min_len = min(len(e_tap), len(x_tap))
            x_tap = x_tap[:min_len]
            e_tap = e_tap[:min_len]

            # Input power for normalization
            input_power = np.dot(x_tap, x_tap)
            if input_power < self.nlms_epsilon:
                continue

            # Compute gradient: <e, x_tap> / ||x_tap||^2 = mean(e * x_tap) / mean(x_tap^2)
            gradient = float(
                self.nlms_mu * np.dot(e_tap, x_tap) / (input_power + self.nlms_epsilon)
            )

            # Update this tap
            self.room_ir[tap] += gradient

        return error
