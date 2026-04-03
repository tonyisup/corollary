"""Thalamus — gating/masking.

M0 extend: spectral subtraction gate with SVSR computation.
Uses efference copy (predicted mel) to attenuate self-voice in observed mel.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ── Global constants ──────────────────────────────────────────────────────
# SVSR threshold in dB: above this, residual is considered "world voice"
DEFAULT_SVSR_THRESHOLD = 20.0

# Floor for subtraction: don't subtract below this fraction of observed energy
SUBTRACTION_FLOOR = 0.01


def spectral_subtraction(
    observed_mel: np.ndarray,
    predicted_mel: np.ndarray,
    floor: float = SUBTRACTION_FLOOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract predicted (efference) mel from observed mel.

    Parameters
    ----------
    observed_mel : np.ndarray
        Mel spectrogram from STG, shape (T, F).
    predicted_mel : np.ndarray
        Predicted mel from cerebellum forward model, shape (T', F).
        Will be padded or trimmed to match observed_mel's time dimension.
    floor : float
        Minimum fraction of observed energy to keep (prevents over-subtraction).

    Returns
    -------
    residual : np.ndarray
        Residual mel after subtraction, same shape as observed_mel.
    mask : np.ndarray
        Boolean mask where residual > floor * observed, shape (T, F).
        True = likely world-voice, False = likely self-voice.
    """
    # Match time dimensions
    t_obs, t_pred = observed_mel.shape[0], predicted_mel.shape[0]
    if t_pred < t_obs:
        # Pad prediction with silence
        padded = np.zeros_like(observed_mel)
        padded[:t_pred] = predicted_mel
        predicted_mel = padded
    elif t_pred > t_obs:
        # Truncate prediction
        predicted_mel = predicted_mel[:t_obs]

    # Convert dB to linear for subtraction (mel is typically in dB)
    # mel_db -> linear: 10^(mel/10)
    obs_linear = np.power(10, observed_mel.astype(np.float64) / 10.0)
    pred_linear = np.power(10, predicted_mel.astype(np.float64) / 10.0)

    # Spectral subtraction (don't go below floor)
    residual_linear = np.maximum(
        obs_linear - pred_linear,
        floor * obs_linear,
    )

    # Convert back to dB
    residual_db = 10.0 * np.log10(np.maximum(residual_linear, 1e-10))

    # Mask: where residual is significant
    mask = residual_linear > (floor * obs_linear)

    return residual_db.astype(np.float32), mask


def compute_svsr(
    observed_mel: np.ndarray,
    predicted_mel: np.ndarray,
    residual_mel: np.ndarray,
) -> float:
    """Compute Self-Voice Suppression Ratio in dB.

    SVSR measures how effectively the efference copy suppresses self-voice:
        SVSR = 10 * log10(sum(observed^2) / sum(residual^2))

    Higher = better suppression. >= 20 dB is the M0 target.

    Parameters
    ----------
    observed_mel : np.ndarray
        Observed mel spectrogram (dB), shape (T, F).
    predicted_mel : np.ndarray
        Predicted mel from efference copy (dB), shape (T, F).
    residual_mel : np.ndarray
        Residual after spectral subtraction (dB), shape (T, F).

    Returns
    -------
    float
        SVSR in dB.
    """
    # Use linear domain for energy computation
    obs_linear = np.power(10, observed_mel.astype(np.float64) / 10.0)
    res_linear = np.power(10, residual_mel.astype(np.float64) / 10.0)

    obs_energy = np.sum(obs_linear ** 2)
    res_energy = np.sum(res_linear ** 2)

    if res_energy < 1e-30:
        return float("inf")  # Complete suppression

    return float(10.0 * np.log10(obs_energy / res_energy))


class ThalamicGate:
    """The thalamus: gates sensory input to conscious processing.

    In M0 this is a simple spectral subtraction gate.
    In M2+ this incorporates real-time AEC and cerebellar forward model.
    """

    def __init__(
        self,
        svsr_threshold: float = DEFAULT_SVSR_THRESHOLD,
        subtraction_floor: float = SUBTRACTION_FLOOR,
    ) -> None:
        self.svsr_threshold = svsr_threshold
        self.subtraction_floor = subtraction_floor

    def gate(
        self,
        observed_mel: np.ndarray,
        predicted_mel: np.ndarray,
    ) -> dict:
        """Gate observed audio against predicted efference copy.

        Parameters
        ----------
        observed_mel : np.ndarray
            Mel spectrogram from STG, shape (T, F).
        predicted_mel : np.ndarray
            Predicted mel from cerebellum, shape (T, F).

        Returns
        -------
        dict with keys:
            residual : np.ndarray — residual mel after gating
            mask : np.ndarray — True=world-voice
            svsr : float — SVSR in dB
            is_world : bool — True if SVSR > threshold
        """
        residual, mask = spectral_subtraction(
            observed_mel, predicted_mel, floor=self.subtraction_floor
        )
        svsr = compute_svsr(observed_mel, predicted_mel, residual)

        return {
            "residual": residual,
            "mask": mask,
            "svsr": svsr,
            # Low SVSR = poor self-suppression = likely world voice
            "is_world": svsr < self.svsr_threshold,
        }
