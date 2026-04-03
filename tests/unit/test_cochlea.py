"""Tests for the Cochlea audio DSP pipeline."""

import numpy as np
import pytest

from cochlea.capture import EfferenceBuffer, pcm_to_mels, SR, HOP, N_MELS


# --- EfferenceBuffer Tests ---

class TestEfferenceBuffer:
    def test_empty_buffer(self):
        buf = EfferenceBuffer(max_ms=100)
        audio = buf.audio
        assert len(audio) == 0
        assert audio.dtype == np.float32
        assert not buf.is_full

    def test_append_and_retrieve(self):
        buf = EfferenceBuffer(max_ms=1000)
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        buf.append(samples)
        retrieved = buf.audio
        np.testing.assert_array_almost_equal(retrieved, samples)
        assert len(retrieved) == 3

    def test_append_empty_array(self):
        buf = EfferenceBuffer(max_ms=100)
        buf.append(np.zeros(0, dtype=np.float32))
        assert len(buf.audio) == 0

    def test_multiple_appends(self):
        buf = EfferenceBuffer(max_ms=1000)
        buf.append(np.array([1.0, 2.0], dtype=np.float32))
        buf.append(np.array([3.0, 4.0], dtype=np.float32))
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(buf.audio, expected)

    def test_wrap_around(self):
        # Verify wrap-around keeps all samples in correct chronological order
        buf = EfferenceBuffer(max_ms=100)
        max_len = buf.max_len

        first_chunk = np.ones(max_len - 10, dtype=np.float32)
        second_chunk = np.full(20, 2.0, dtype=np.float32)
        buf.append(first_chunk)
        buf.append(second_chunk)

        result = buf.audio
        assert len(result) == max_len
        assert buf.is_full
        # second_chunk overwrites oldest data and wraps around
        # The 20 samples of second_chunk end up at the temporal "end"
        np.testing.assert_array_almost_equal(
            result[max_len - 20:], np.full(20, 2.0, dtype=np.float32)
        )
        # Earlier samples are still 1.0 from first_chunk
        assert result[0] == 1.0

    def test_is_full_flag(self):
        buf = EfferenceBuffer(max_ms=100)
        assert not buf.is_full

        # Write exactly max_len samples
        data = np.ones(buf.max_len, dtype=np.float32)
        buf.append(data)
        assert buf.is_full

    def test_clear(self):
        buf = EfferenceBuffer(max_ms=1000)
        buf.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        buf.clear()
        assert len(buf.audio) == 0
        assert not buf.is_full

    def test_overflow_keeps_newest(self):
        """When buffer is full, oldest data should be overwritten."""
        buf = EfferenceBuffer(max_ms=100)
        max_len = buf.max_len

        # Fill completely with 1.0
        buf.append(np.ones(max_len, dtype=np.float32))
        # Append 100 new samples of 2.0 — should overwrite oldest 100
        new_data = np.full(100, 2.0, dtype=np.float32)
        buf.append(new_data)

        result = buf.audio
        assert len(result) == max_len
        # Newest 100 should be 2.0
        np.testing.assert_array_almost_equal(result[-100:], np.full(100, 2.0, dtype=np.float32))
        # Oldest max_len-100 should still be 1.0
        np.testing.assert_array_almost_equal(
            result[:max_len - 100],
            np.full(max_len - 100, 1.0, dtype=np.float32),
        )


# --- pcm_to_mels Tests ---

class TestPcmToMels:
    def test_output_shape(self):
        # 1 second of silence at 16kHz
        audio = np.zeros(SR, dtype=np.float32)
        mels = pcm_to_mels(audio)
        assert mels.ndim == 2
        assert mels.shape[1] == N_MELS
        # Time dimension depends on audio length
        assert mels.shape[0] > 0

    def test_sine_wave_produces_mels(self):
        t = np.arange(SR, dtype=np.float32) / SR
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz
        mels = pcm_to_mels(audio)
        assert mels.shape[1] == N_MELS
        # 440 Hz should show energy somewhere in the mel bands
        assert np.any(mels > -80)  # Not all silence
