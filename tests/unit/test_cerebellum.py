"""Tests for the Cerebellum forward model."""

import numpy as np
import pytest

from cerebellum.forward_model import predict_mel


class TestPredictMel:
    def test_silence_produces_mel(self):
        """Silence PCM should produce a valid mel spectrogram with very low energy."""
        silence = np.zeros(16000, dtype=np.float32)
        mel = predict_mel(silence)
        assert mel.ndim == 2
        assert mel.shape[1] == 80  # n_mels
        assert mel.dtype == np.float32
        # Silence mel should be very low (zeros or near -inf dB)
        assert np.mean(np.abs(mel)) < 1.0

    def test_random_audio_produces_mel(self):
        """Random PCM should produce a valid (non-trivial) mel spectrogram."""
        audio = np.random.randn(32000).astype(np.float32) * 0.1
        mel = predict_mel(audio)
        assert mel.ndim == 2
        assert mel.shape[1] == 80
        assert mel.shape[0] > 0

    def test_output_shape_scales_with_input_duration(self):
        """Longer audio should produce more time frames."""
        short = np.sin(np.linspace(0, 440 * 2 * np.pi, 8000)).astype(np.float32) * 0.1
        long = np.sin(np.linspace(0, 440 * 2 * np.pi, 32000)).astype(np.float32) * 0.1
        mel_short = predict_mel(short)
        mel_long = predict_mel(long)
        assert mel_short.shape[1] == mel_long.shape[1]  # Same n_mels
        assert mel_long.shape[0] > mel_short.shape[0]  # More frames

    def test_deterministic_output(self):
        """Same input should always produce the same mel spectrogram."""
        audio = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)).astype(np.float32) * 0.1
        mel1 = predict_mel(audio)
        mel2 = predict_mel(audio)
        np.testing.assert_array_equal(mel1, mel2)

    def test_sine_wave_has_energy(self):
        """A 440 Hz sine wave should produce a mel spectrogram with meaningful energy."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
        mel = predict_mel(tone)
        # Should have significantly more energy than silence
        silence_mel = predict_mel(np.zeros(sr, dtype=np.float32))
        # Tone mel should have non-trivial values (silence mel is all zeros)
        assert np.mean(np.abs(mel)) > np.mean(np.abs(silence_mel)) + 1.0

    def test_minimal_input_produces_mel(self):
        """Even a tiny input should produce at least one time frame."""
        tiny = np.zeros(512, dtype=np.float32)
        mel = predict_mel(tiny)
        assert mel.ndim == 2
        assert mel.shape[0] >= 1
        assert mel.shape[1] == 80
