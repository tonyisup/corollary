"""Thalamus — corollary discharge gate.

Aligns efference-copy mel features to incoming mic mel features,
then subtracts the self-predicted component to suppress the agent's
own voice before it reaches the ASR.
"""

from __future__ import annotations

import numpy as np


def estimate_lag(
    mic_mel: np.ndarray,
    eff_mel: np.ndarray,
    max_lag: int = 40,
) -> int:
    """Estimate temporal lag between mic and efference mel.

    Uses frame-wise cosine similarity across a range of lags.

    Returns
    -------
    lag : int
        Positive means mic is ahead of eff; negative means eff is ahead.
    """
    best_off: int = 0
    best_sim: float = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = mic_mel[lag:]
            b = eff_mel[: len(mic_mel) - lag]
        else:
            a = mic_mel[: len(mic_mel) + lag]
            b = eff_mel[-lag:]

        if len(a) < 5:
            continue

        num = (a * b).sum()
        den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
        sim = num / den

        if sim > best_sim:
            best_sim = sim
            best_off = lag

    return best_off


def spectral_gate(
    mic_mel: np.ndarray,
    eff_mel: np.ndarray,
    lag: int,
    strength: float = 0.7,
) -> np.ndarray:
    """Subtract efference prediction from mic mel to suppress self-voice.

    Parameters
    ----------
    mic_mel : (T, 80)
        Log-mel spectrogram from the microphone.
    eff_mel : (T', 80)
        Log-mel spectogram predicted from TTS output.
    lag : int
        Temporal offset from `estimate_lag`.
    strength : float
        Attenuation factor (0 = no suppression, 1 = full cancel).

    Returns
    -------
    gated_mel : (T, 80)
        Self-suppressed log-mel spectrogram.
    """
    gated = mic_mel.copy()

    if lag >= 0:
        b = eff_mel[: len(mic_mel) - lag]
        gated[lag:] -= strength * b
    else:
        b = eff_mel[-lag:]
        gated[: len(gated) + lag] -= strength * b

    # Clamp to reasonable floor
    floor = mic_mel.min() - 3.0
    return np.maximum(gated, floor)
