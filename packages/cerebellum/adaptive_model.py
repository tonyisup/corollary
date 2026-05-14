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
from cerebellum.trust_model import TrustTracker, TrustState, DEFAULT_IR_DIR
from typing import Any, List, Tuple, Optional
import json
import os
import time


class BabblingCerebellum:
    """Forward model that learns the room impulse response through babbling.

    Maintains an estimated room_ir that is updated incrementally using
    normalized least mean squares (NLMS) on the prediction error.

    The learned IR is then used to convolve efference copies for
    forward-model predictions that account for real room acoustics.

    Trust model: tracks how much the cerebellum should trust its learned IR.
    Trust grows when predictions match reality, decays when they don't.
    When trust falls below a threshold, the cerebellum signals that it
    needs to rebabble.
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

        # Trust model — tracks confidence in the learned IR
        self.trust = TrustTracker()
        self._ir_save_path: Optional[str] = None

        # M7 phase 2: drift alarm for adaptive learning-rate control
        self._drift_alarm: Optional[Any] = None

        # M7 phase 3: multi-hypothesis IR bank
        self._ir_bank: Optional[Any] = None

        # M7 phase 4: device profile bank
        self._device_profile_bank: Optional[Any] = None

    def set_drift_alarm(self, alarm: Any) -> None:
        """Attach a ``DriftAlarm`` to modulate learning rate on room drift."""
        self._drift_alarm = alarm

    def set_ir_bank(self, bank: Any) -> None:
        """Attach an ``IRBank`` for multi-hypothesis room modeling."""
        self._ir_bank = bank

    def set_device_profile_bank(self, bank: Any) -> None:
        """Attach a ``DeviceProfileBank`` for orientation-dependent profiles."""
        self._device_profile_bank = bank

    @property
    def _active_ir(self) -> np.ndarray:
        """Return the best IR — from bank if attached, else the single IR."""
        if self._ir_bank is not None:
            return self._ir_bank.best_hypothesis.ir
        return self.room_ir

    def predict_pcm(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the raw PCM the cochlea will receive."""
        roomed = fftconvolve(efference_pcm, self._active_ir, mode="full").astype(
            np.float32
        )
        return roomed

    def save_model(self, path: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        """Persist the learned IR and trust state to disk.

        Parameters
        ----------
        path : str, optional
            File path. Uses default timestamp-named file if not specified.
        metadata : dict, optional
            Extra metadata to store alongside the model.

        Returns
        -------
        str
            Path to the saved file.
        """
        if path is None:
            ts = int(time.time())
            path = os.path.join(DEFAULT_IR_DIR, f"room_ir_{ts}.json")
        os.makedirs(os.path.dirname(path) or DEFAULT_IR_DIR, exist_ok=True)

        data = {
            "ir": self.room_ir.tolist(),
            "trust_state": self.trust.state.to_dict(),
            "metadata": metadata or {},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        self._ir_save_path = path
        return path

    @classmethod
    def load_model(cls, path: str, **kwargs) -> "BabblingCerebellum":
        """Restore a cerebellum from a saved IR file.

        Parameters
        ----------
        path : str
            Path to a previously saved room IR JSON file.
        **kwargs
            Extra constructor overrides (e.g. nlms_mu).

        Returns
        -------
        BabblingCerebellum
            Instance with the learned IR and restored trust state.
        """
        with open(path, "r") as f:
            data = json.load(f)

        ir = np.array(data["ir"], dtype=np.float32)
        trust_state = TrustState.from_dict(data.get("trust_state", {}))

        # Infer IR length from saved array
        ir_length_ms = int(len(ir) * 1000 / SR)
        model = cls(ir_length_ms=ir_length_ms, **kwargs)
        model.room_ir = ir
        model.trust.state = trust_state
        model.total_updates = trust_state.confirmations
        model._ir_save_path = path
        return model

    def trust_status(self) -> dict:
        """Return current trust diagnostics."""
        return self.trust.get_status()

    def update(
        self,
        known_pcm: np.ndarray,
        captured_pcm: np.ndarray,
        *,
        allow_high_trust_gate: bool = True,
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
        # Preserve amplitude ratio for proper IR learning
        # The cerebellum needs to know: "how loud is the signal going in vs coming back?"
        known_rms = float(np.sqrt(np.mean(known_pcm ** 2)))
        captured_rms = float(np.sqrt(np.mean(captured_pcm ** 2)))
        amplitude_ratio = captured_rms / max(known_rms, 1e-10)
        
        # Normalize for numerical stability but track the scale
        known = self._normalize(known_pcm)
        captured = self._normalize(captured_pcm)

        # Align timing via cross-correlation
        delay, captured_aligned, known_aligned = self._align_signals(known, captured)

        predicted = fftconvolve(known_aligned, self.room_ir, mode="full").astype(np.float32)[: len(captured_aligned)]
        error_before = captured_aligned - predicted
        error_rms_before = float(np.sqrt(np.mean(error_before ** 2)))
        error_db_before = 20 * np.log10(max(error_rms_before, 1e-10))

        prediction_norm = float(np.linalg.norm(predicted))
        captured_norm = float(np.linalg.norm(captured_aligned))
        denom = max(prediction_norm * captured_norm, 1e-10)
        prediction_similarity = float(np.dot(captured_aligned, predicted) / denom)

        should_gate_update = False
        if allow_high_trust_gate and self.trust.state.trust >= 0.7:
            avg_error_db = float(self.trust.state.avg_error_db)
            if (
                error_db_before > avg_error_db + 10.0
                or error_db_before > -5.0
                or prediction_similarity < 0.3
            ):
                should_gate_update = True

        if should_gate_update:
            error = error_before
        else:
            error = self._nlms_update(known_aligned, captured_aligned)
            self.total_updates += 1

        error_rms = float(np.sqrt(np.mean(error ** 2)))
        error_db = 20 * np.log10(max(error_rms, 1e-10))
        self.prediction_errors.append(error_rms)
        self.convergence_history.append(error_rms)

        # Update trust: use prediction error as signal
        # Error < -30 dB → good prediction → trust grows
        # Error > -10 dB → poor prediction → trust decays
        if error_db < -30:
            self.trust.confirm(error_db)
        elif error_db > -10:
            self.trust.fail()
        else:
            # In between: partial confirmation
            self.trust.confirm(error_db)

        # M7 phase 2: feed error to drift alarm
        if self._drift_alarm is not None:
            self._drift_alarm.record_error(error_db)

        return self.prediction_errors[-1]

    def update_trust_gated(
        self,
        known_pcm: np.ndarray,
        captured_pcm: np.ndarray,
        *,
        verdict: str,
        m4_confidence: Optional[float] = None,
        self_match_score: Optional[float] = None,
        adapt_confidence_threshold: float = 0.80,
        adapt_match_threshold: float = 0.20,
    ) -> dict:
        """Conditionally update NLMS from externally trusted attribution.

        M7 phase 1 adds the missing *when to adapt* gate on top of the
        existing NLMS implementation. The room model may learn only from
        frames that the thalamus classified as SELF and that also pass M4
        confidence plus room-local self-match thresholds. WORLD/BOTH/SILENCE
        frames hold the filter steady to avoid learning another speaker or
        overlap contamination.

        This method is intentionally a thin wrapper around :meth:`update` so
        existing babbling calibration behavior remains unchanged.
        """
        normalized_verdict = str(verdict).upper()
        if normalized_verdict != "SELF":
            return {
                "adapted": False,
                "reason": "verdict_not_self",
                "error": None,
                "updates": self.total_updates,
            }

        if m4_confidence is None or m4_confidence < adapt_confidence_threshold:
            return {
                "adapted": False,
                "reason": "m4_confidence_below_threshold",
                "error": None,
                "updates": self.total_updates,
            }

        if self_match_score is None or self_match_score < adapt_match_threshold:
            return {
                "adapted": False,
                "reason": "self_match_below_threshold",
                "error": None,
                "updates": self.total_updates,
            }

        error = self.update(known_pcm, captured_pcm, allow_high_trust_gate=False)
        return {
            "adapted": True,
            "reason": "trusted_self",
            "error": error,
            "updates": self.total_updates,
        }

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
        # Convolve with learned room IR (or best bank hypothesis)
        roomed_pcm = fftconvolve(efference_pcm, self._active_ir, mode="full").astype(
            np.float32
        )

        # Normalize
        peak = np.max(np.abs(roomed_pcm))
        if peak > 0:
            roomed_pcm = roomed_pcm / peak * 0.9

        return pcm_to_mels(roomed_pcm)

    def predict_mel_aligned(
        self, efference_pcm: np.ndarray, target_length_frames: int
    ) -> np.ndarray:
        """Predict mel spectrogram aligned to match captured audio's frame count.

        Computes the predicted mel, then centers it within the target frame
        count by padding with silence on both sides proportional to the
        speaker-to-mic delay learned in the room IR.

        Parameters
        ----------
        efference_pcm : np.ndarray
            Float32 PCM audio (efference copy from Broca).
        target_length_frames : int
            Number of mel frames in the observed (captured) signal.

        Returns
        -------
        np.ndarray
            Aligned predicted mel spectrogram, shape (target_length_frames, 80).
        """
        # Get the raw predicted mel (will be longer than input due to IR convolution)
        pred_mel = self.predict_mel(efference_pcm)
        n_pred = pred_mel.shape[0]

        if n_pred == target_length_frames:
            return pred_mel

        # Find delay in learned IR (first significant peak)
        ir_threshold = 0.1 * np.max(np.abs(self.room_ir))
        peaks = np.where(np.abs(self.room_ir) > ir_threshold)[0]
        delay_samples = int(peaks[0]) if len(peaks) > 0 else 0
        delay_frames = delay_samples // 160  # approximate frame size

        # Center the prediction in the target frame count
        pad_total = max(0, target_length_frames - n_pred)
        pad_before = min(delay_frames, pad_total)
        pad_after = pad_total - pad_before

        if target_length_frames > n_pred:
            padded = np.zeros((target_length_frames, pred_mel.shape[1]), dtype=np.float32)
            padded[pad_before:pad_before+n_pred] = pred_mel
            return padded
        else:
            # Truncate and shift
            start = max(0, delay_frames - (n_pred - target_length_frames) // 2)
            return pred_mel[start:start+target_length_frames]

    def predict_pcm(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the raw PCM the cochlea will receive.

        Uses the active IR (from bank or single).
        """
        roomed = fftconvolve(efference_pcm, self._active_ir, mode="full").astype(
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
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Find and correct timing offset between known and captured signals.

        Uses cross-correlation to find the delay at which the known signal
        best matches the captured signal. This handles the sound propagation
        delay between speaker and microphone.

        Returns
        -------
        Tuple[int, np.ndarray, np.ndarray]
            (delay_samples, aligned_captured, aligned_known)
        """
        # Cross-correlate to find delay
        correlation = np.correlate(
            captured,
            known,
            mode="full",
        )
        peak_idx = np.argmax(np.abs(correlation))
        delay = peak_idx - len(known) + 1

        # Trim both signals to align
        max_len = min(len(captured) - abs(delay), len(known))
        if delay >= 0:
            captured = captured[delay:delay + max_len]
        else:
            captured = captured[:max_len]
        known = known[:max_len]

        return delay, captured, known

    def _nlms_update(self, known: np.ndarray, captured: np.ndarray) -> np.ndarray:
        """NLMS adaptive filter update — sample-by-sample.

        Standard NLMS processes samples sequentially, updating the
        filter coefficients at each time step. This is the algorithm
        actually used in real-time adaptive filtering (AEC, echo cancellation).

        The filter models: captured(t) = sum_n [IR(n) * known(t-n)]
        At each sample t, we predict, compute error, and update all IR taps.

        Parameters
        ----------
        known : np.ndarray
            Input signal (what was played to speakers), normalized.
        captured : np.ndarray
            Captured signal (what the mic heard), normalized.

        Returns
        -------
        np.ndarray
            Final prediction error segment.
        """
        N = self.ir_length
        mu = self.nlms_mu
        if self._drift_alarm is not None:
            mu *= self._drift_alarm.adaptation_rate_multiplier
        eps = self.nlms_epsilon

        # Align lengths
        max_len = min(len(known), len(captured))
        known = known[:max_len]
        captured = captured[:max_len]

        # Pad known with zeros at both ends (for look-back and look-ahead)
        # We need N-1 samples of pre-padding and N-1 samples of post-padding
        padding = np.zeros(N - 1, dtype=np.float32)
        extended = np.concatenate([padding, known, padding])

        errors = np.zeros(max_len, dtype=np.float32)

        # Process sample by sample
        for t in range(max_len):
            # Input vector: [known(t), known(t-1), ..., known(t-N+1)]
            # In extended, known[0] starts at index N-1
            # So known(t) → extended[N-1+t], known(t-n) → extended[N-1+t-n]
            # Slice from index t to t+N in extended, then reverse
            x = extended[t : t + N][::-1]

            # Prediction: h^T * x
            y_hat = float(np.dot(x, self.room_ir))

            # Error
            e = captured[t] - y_hat
            errors[t] = e

            # NLMS update: h = h + mu * e * x / (x^T x + eps)
            x_power = float(np.dot(x, x))
            if x_power > eps:
                self.room_ir += mu * e * x / (x_power + eps)

        return errors
