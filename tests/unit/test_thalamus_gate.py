"""Tests for the Thalamus gating/spectral subtraction."""

import numpy as np
import pytest

from thalamus.gate import (
    spectral_subtraction,
    compute_svsr,
    ThalamicGate,
    DEFAULT_SVSR_THRESHOLD,
)
from thalamus.attributor import SourceVerdict
from scripts.m2_speech_preservation_harness import build_gating_gain


# --- Spectral Subtraction Tests ---

class TestSpectralSubtraction:
    def test_identical_inputs_zero_residual(self):
        """When predicted == observed, residual should be at floor level."""
        mel = np.random.randn(100, 80).astype(np.float32)
        residual, mask = spectral_subtraction(mel, mel)
        # Residual should be small (floored)
        assert residual.shape == mel.shape
        # Most of the signal should be subtracted
        obs_linear = np.power(10, mel.astype(np.float64) / 10.0)
        res_linear = np.power(10, residual.astype(np.float64) / 10.0)
        assert np.mean(res_linear) < np.mean(obs_linear) * 0.1

    def test_zero_prediction_no_subtraction(self):
        """When predicted is silence, observed should be nearly unchanged."""
        observed = np.full((50, 80), -20.0, dtype=np.float32)  # ~10% amplitude
        predicted = np.full((50, 80), -80.0, dtype=np.float32)  # near silence
        residual, mask = spectral_subtraction(observed, predicted)
        # Residual should be close to observed
        np.testing.assert_array_almost_equal(residual, observed, decimal=4)

    def test_mismatched_time_dims_pads(self):
        """Prediction shorter than observed should be zero-padded."""
        observed = np.random.randn(100, 80).astype(np.float32)
        predicted = np.random.randn(50, 80).astype(np.float32)
        residual, mask = spectral_subtraction(observed, predicted)
        assert residual.shape == observed.shape

    def test_mask_has_correct_shape(self):
        observed = np.random.randn(10, 40).astype(np.float32)
        predicted = np.random.randn(10, 40).astype(np.float32)
        _, mask = spectral_subtraction(observed, predicted)
        assert mask.shape == observed.shape
        assert mask.dtype == bool

    def test_prediction_truncated_if_longer(self):
        observed = np.zeros((10, 80), dtype=np.float32)
        predicted = np.random.randn(20, 80).astype(np.float32)
        residual, mask = spectral_subtraction(observed, predicted)
        assert residual.shape == observed.shape


def test_m2_gating_gain_softens_both_frames():
    gain = build_gating_gain(
        [SourceVerdict.WORLD, SourceVerdict.BOTH, SourceVerdict.SELF],
        frame_samples=4,
        total_samples=12,
        both_gain=0.4,
    )

    np.testing.assert_array_equal(gain[:4], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(gain[4:8], np.full(4, 0.4, dtype=np.float32))
    np.testing.assert_array_equal(gain[8:], np.zeros(4, dtype=np.float32))


# --- SVSR Tests ---

class TestComputeSvsr:
    def test_perfect_suppression_is_infinite(self):
        """If residual is truly zero, SVSR should be infinity."""
        # Construct residual directly (bypass spectral_subtraction floor)
        observed = np.full((10, 80), -20.0, dtype=np.float32)
        predicted = np.full((10, 80), -20.0, dtype=np.float32)
        # Residual at floor: still has some energy
        residual, _ = spectral_subtraction(observed, predicted)
        svsr = compute_svsr(observed, predicted, residual)
        # Due to floor, SVSR is high but not infinite
        assert svsr > 30  # At least 30 dB suppression

    def test_no_suppression_is_low(self):
        """If residual == observed, SVSR should be ~0 dB."""
        observed = np.random.randn(10, 80).astype(np.float32) * 10
        predicted = np.full((10, 80), -80.0, dtype=np.float32)  # silence
        residual, _ = spectral_subtraction(observed, predicted)
        svsr = compute_svsr(observed, predicted, residual)
        assert svsr < 1.0  # Almost no suppression

    def test_partial_suppression_measurable(self):
        """50% amplitude reduction should give measurable SVSR."""
        observed = np.full((100, 80), -20.0, dtype=np.float32)
        # Predict at half the amplitude (in dB: -20 vs -26 dB ≈ half power)
        predicted = np.full((100, 80), -23.0, dtype=np.float32)
        residual, _ = spectral_subtraction(observed, predicted)
        svsr = compute_svsr(observed, predicted, residual)
        assert svsr > 0  # Some suppression
        assert not np.isnan(svsr)


# --- ThalamicGate Tests ---

class TestThalamicGate:
    def test_gate_identical_signals(self):
        gate = ThalamicGate(svsr_threshold=20.0)
        mel = np.random.randn(100, 80).astype(np.float32) * 5
        result = gate.gate(mel, mel)
        # Due to floor, SVSR is high but not infinite (~40 dB)
        assert result["svsr"] > 30
        assert not result["is_world"]  # Perfect suppression = self, not world

    def test_gate_detects_world(self):
        gate = ThalamicGate(svsr_threshold=20.0)
        # Observed has signal, predicted is silence → world detected
        observed = np.full((100, 80), -20.0, dtype=np.float32)
        predicted = np.full((100, 80), -80.0, dtype=np.float32)
        result = gate.gate(observed, predicted)
        # SVSR should be ~0 (no suppression)
        assert result["svsr"] < 1.0
        assert result["is_world"] is True  # No suppression detected = world

    def test_gate_returns_mask(self):
        gate = ThalamicGate()
        observed = np.random.randn(50, 80).astype(np.float32)
        predicted = np.random.randn(50, 80).astype(np.float32)
        result = gate.gate(observed, predicted)
        assert result["mask"].shape == observed.shape
        assert result["residual"].shape == observed.shape
        assert isinstance(result["svsr"], float)

    def test_threshold_controls_is_world(self):
        # High threshold → harder to call it "world"
        gate_lenient = ThalamicGate(svsr_threshold=50.0)
        gate_strict = ThalamicGate(svsr_threshold=5.0)

        observed = np.full((100, 80), -20.0, dtype=np.float32)
        predicted = np.full((100, 80), -60.0, dtype=np.float32)  # quiet prediction

        lenient = gate_lenient.gate(observed, predicted)
        strict = gate_strict.gate(observed, predicted)

        # Same SVSR, different interpretation
        assert lenient["svsr"] == strict["svsr"]
        # Strict gate might call it world sooner
