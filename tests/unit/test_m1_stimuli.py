import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m1_tone_attribution import generate_stimulus_with_metadata


def dominant_frequency_hz(pcm: np.ndarray, sr: int = 16_000) -> float:
    spectrum = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sr)
    return float(freqs[int(np.argmax(spectrum))])


def test_speech_band_chirp_metadata_and_audio():
    result = generate_stimulus_with_metadata("speech_band_chirp", 350, 0.7, seed=0)

    assert result["chirp_band_hz"] == [150.0, 2500.0]
    assert result["pcm"].dtype == np.float32
    assert np.max(np.abs(result["pcm"])) <= 0.7001


def test_low_chirp_stays_in_low_mid_band():
    result = generate_stimulus_with_metadata("low_chirp", 350, 0.7, seed=0)

    assert result["chirp_band_hz"] == [150.0, 1200.0]
    # The strongest FFT bin for a swept chirp is implementation/window dependent,
    # but it should remain inside the requested low/mid calibration band.
    assert 100.0 <= dominant_frequency_hz(result["pcm"]) <= 1300.0
