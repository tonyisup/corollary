"""Cerebellum — forward model.

Predicts what the cochlea will hear *before* the TTS audio leaves
the speaker, so the thalamus can proactively attenuate self-signal.

Phase 0 (M0): simple mel-of-self via direct librosa conversion.
Phase 1+: learns room acoustics through babbling (BabblingCerebellum).
Phase 2+: learns a predictive model that accounts for speaker colorization.
"""

from __future__ import annotations

import numpy as np
from cochlea.capture import pcm_to_mels


def predict_mel(eff_pcm: np.ndarray) -> np.ndarray:
    """Generate predicted mel spectrogram from efference-copy PCM.

    Parameters
    ----------
    eff_pcm : np.ndarray
        float32 PCM audio from TTS output (or simulated playback).

    Returns
    -------
    np.ndarray
        Log-mel spectrogram, shape (T, 80).
    """
    return pcm_to_mels(eff_pcm)
