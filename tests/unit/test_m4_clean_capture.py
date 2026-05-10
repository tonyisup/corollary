"""Tests for M4 clean self-capture playback preparation."""

import importlib.util
import os
import sys

import numpy as np

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "m4_clean_capture.py"
)
_spec = importlib.util.spec_from_file_location("m4_clean_capture", SCRIPT_PATH)
m4_clean_capture = importlib.util.module_from_spec(_spec)
sys.modules["m4_clean_capture"] = m4_clean_capture
_spec.loader.exec_module(m4_clean_capture)


def test_playback_buffer_preserves_16khz_tts_duration():
    """Clean capture should not feed 48k-resampled PCM into a 16k stream."""
    sr = 16000
    self_pcm = np.ones(sr, dtype=np.float32)

    play_buffer, lead_in_samples = m4_clean_capture._build_playback_buffer(
        self_pcm,
        sr=sr,
        lead_in_s=0.3,
        tail_s=1.0,
    )

    assert lead_in_samples == int(0.3 * sr)
    assert len(play_buffer) == int((1.0 + 0.3 + 1.0) * sr)
    nonzero = np.flatnonzero(play_buffer)
    assert nonzero[0] == lead_in_samples
    assert nonzero[-1] == lead_in_samples + len(self_pcm) - 1
