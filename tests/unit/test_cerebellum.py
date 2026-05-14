"""Tests for the Cerebellum adaptive model — babbling, persistence, trust."""

import json
import os
import tempfile

import numpy as np
import pytest

from scipy.signal import fftconvolve

from cochlea.capture import SR
from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.trust_model import (
    TrustTracker,
    TrustState,
    save_ir,
    load_ir,
    list_saved_models,
    DEFAULT_IR_DIR
)


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture()
def short_room_ir():
    """A small room IR for tests (100 ms, with early reflections)."""
    ir = np.zeros(int(0.1 * SR), dtype=np.float32)
    ir[0] = 1.0
    ir[16] = 0.35
    ir[48] = 0.15
    return ir


@pytest.fixture()
def known_signal():
    """A speech-like known signal for babbling."""
    t = np.linspace(0, 1.0, int(1.0 * SR), dtype=np.float32)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    return sig / (np.max(np.abs(sig)) * 1.1)


def make_interfered_capture(known_signal: np.ndarray, room_ir: np.ndarray, noise_scale: float = 2.0) -> np.ndarray:
    """Return a room-colored capture with strong added interference."""
    captured = fftconvolve(known_signal, room_ir, mode="full").astype(np.float32)
    noise = np.random.randn(*captured.shape).astype(np.float32) * noise_scale
    return captured + noise


# ── BabblingCerebellum Tests ─────────────────────────────────────────────


class TestBabblingCerebellumInit:
    def test_default_init(self):
        cerebellum = BabblingCerebellum()
        assert cerebellum.sample_rate == SR
        assert cerebellum.ir_length == int(0.3 * SR)
        assert len(cerebellum.room_ir) == cerebellum.ir_length
        assert cerebellum.room_ir[0] == 1.0
        assert cerebellum.total_updates == 0

    def test_custom_init(self):
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        assert cerebellum.ir_length == int(0.1 * SR)
        assert cerebellum.nlms_mu == 0.1


class TestBabblingCerebellumUpdate:
    def test_update_low_error(self, known_signal, short_room_ir):
        """NLMS should converge to low error on a clean signal."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        for _ in range(20):
            err = cerebellum.update(known_signal, captured)
        assert cerebellum.total_updates == 20
        assert err < 0.01

    def test_update_increases_total_updates(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum()
        assert cerebellum.total_updates == 0
        cerebellum.update(known_signal, captured)
        assert cerebellum.total_updates == 1

    def test_update_updates_trust(self, known_signal, short_room_ir):
        """Each update should affect trust status."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        initial = cerebellum.trust_status()
        for _ in range(10):
            cerebellum.update(known_signal, captured)
        final = cerebellum.trust_status()
        assert final["trust"] > initial["trust"]

    def test_update_learns_under_interference_when_not_yet_trusted(self, known_signal, short_room_ir):
        """Low-trust models should still keep learning, even from rough captures."""
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        captured = make_interfered_capture(known_signal, short_room_ir)

        initial_updates = cerebellum.total_updates
        cerebellum.update(known_signal, captured)

        assert cerebellum.total_updates == initial_updates + 1

    def test_update_skips_learning_under_high_trust_interference(self, known_signal, short_room_ir):
        """High-trust models should freeze updates on likely interference blocks."""
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        cerebellum.trust.state.trust = 0.9
        cerebellum.trust.state.avg_error_db = -30.0

        captured = make_interfered_capture(known_signal, short_room_ir)
        initial_updates = cerebellum.total_updates
        initial_ir = cerebellum.room_ir.copy()

        cerebellum.update(known_signal, captured)

        assert cerebellum.total_updates == initial_updates
        np.testing.assert_array_equal(cerebellum.room_ir, initial_ir)


class TestBabblingCerebellumTrustGatedUpdate:
    def test_trust_gated_update_adapts_on_confident_self(self, known_signal, short_room_ir):
        """M7 phase 1: confident SELF frames are allowed to update NLMS."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        initial_updates = cerebellum.total_updates
        initial_ir = cerebellum.room_ir.copy()

        result = cerebellum.update_trust_gated(
            known_signal,
            captured,
            verdict="SELF",
            m4_confidence=0.92,
            self_match_score=0.4,
            adapt_confidence_threshold=0.8,
            adapt_match_threshold=0.2,
        )

        assert result["adapted"] is True
        assert result["reason"] == "trusted_self"
        assert cerebellum.total_updates == initial_updates + 1
        assert not np.allclose(cerebellum.room_ir, initial_ir)
        assert result["error"] is not None

    @pytest.mark.parametrize("verdict", ["WORLD", "BOTH", "SILENCE"])
    def test_trust_gated_update_holds_on_non_self_verdicts(self, known_signal, short_room_ir, verdict):
        """Never adapt the room filter on WORLD/BOTH/SILENCE frames."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        initial_updates = cerebellum.total_updates
        initial_ir = cerebellum.room_ir.copy()

        result = cerebellum.update_trust_gated(
            known_signal,
            captured,
            verdict=verdict,
            m4_confidence=0.99,
            self_match_score=0.99,
        )

        assert result["adapted"] is False
        assert result["reason"] == "verdict_not_self"
        assert cerebellum.total_updates == initial_updates
        np.testing.assert_array_equal(cerebellum.room_ir, initial_ir)

    def test_trust_gated_update_overrides_high_trust_interference_gate(self, known_signal, short_room_ir):
        """Trusted SELF evidence should adapt even when the old high-trust gate would freeze."""
        shifted_room_ir = np.zeros_like(short_room_ir)
        shifted_room_ir[0] = 1.0
        shifted_room_ir[128] = 0.85
        shifted_room_ir[511] = -0.35
        captured = fftconvolve(known_signal, shifted_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        cerebellum.trust.state.trust = 0.9
        cerebellum.trust.state.avg_error_db = -40.0
        initial_updates = cerebellum.total_updates

        result = cerebellum.update_trust_gated(
            known_signal,
            captured,
            verdict="SELF",
            m4_confidence=0.95,
            self_match_score=0.8,
        )

        assert result["adapted"] is True
        assert result["reason"] == "trusted_self"
        assert cerebellum.total_updates == initial_updates + 1

    def test_trust_gated_update_holds_on_low_m4_confidence(self, known_signal, short_room_ir):
        """SELF verdict alone is insufficient; M4 confidence must pass the gate."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        initial_ir = cerebellum.room_ir.copy()

        result = cerebellum.update_trust_gated(
            known_signal,
            captured,
            verdict="SELF",
            m4_confidence=0.2,
            self_match_score=0.9,
            adapt_confidence_threshold=0.8,
        )

        assert result["adapted"] is False
        assert result["reason"] == "m4_confidence_below_threshold"
        assert cerebellum.total_updates == 0
        np.testing.assert_array_equal(cerebellum.room_ir, initial_ir)

    def test_trust_gated_update_holds_on_low_self_match(self, known_signal, short_room_ir):
        """M4 confidence and room-local self-match jointly gate adaptation."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        initial_ir = cerebellum.room_ir.copy()

        result = cerebellum.update_trust_gated(
            known_signal,
            captured,
            verdict="SELF",
            m4_confidence=0.95,
            self_match_score=0.05,
            adapt_match_threshold=0.2,
        )

        assert result["adapted"] is False
        assert result["reason"] == "self_match_below_threshold"
        assert cerebellum.total_updates == 0
        np.testing.assert_array_equal(cerebellum.room_ir, initial_ir)


class TestBabblingCerebellumPredict:
    def test_predict_mel_shape(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        for _ in range(10):
            cerebellum.update(known_signal, captured)
        mel = cerebellum.predict_mel(known_signal)
        assert mel.ndim == 2
        assert mel.shape[1] == 80
        assert mel.shape[0] > 0

    def test_predict_pcm(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        for _ in range(10):
            cerebellum.update(known_signal, captured)
        pcm = cerebellum.predict_pcm(known_signal)
        assert pcm.ndim == 1
        assert len(pcm) > len(known_signal)


class TestBabblingCerebellumPersistence:
    def test_save_and_load(self, known_signal, short_room_ir):
        """Save and load should preserve the IR and trust state."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        for _ in range(10):
            cerebellum.update(known_signal, captured)
        original_ir = cerebellum.room_ir.copy()
        original_trust = cerebellum.trust_status()["trust"]
        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        cerebellum.save_model(path=path)
        # Load
        loaded = BabblingCerebellum.load_model(path)
        np.testing.assert_array_almost_equal(loaded.room_ir, original_ir, decimal=5)
        assert loaded.trust_status()["trust"] == original_trust
        os.unlink(path)


# ── TrustTracker Tests ──────────────────────────────────────────────────


class TestTrustTrackerConfirm:
    def test_good_error_increases_trust(self):
        tracker = TrustTracker()
        assert tracker.state.trust == 0.5
        action = tracker.confirm(-50.0)
        assert action in ("confirmed", "stable")
        assert tracker.state.trust > 0.5

    def test_bad_error_records_failure(self):
        tracker = TrustTracker()
        action = tracker.fail()
        assert action == "ok"
        assert tracker.state.failures == 1
        assert tracker.state.failure_streak == 1
        assert tracker.state.trust < 0.5

    def test_recovery_from_low_trust(self):
        tracker = TrustTracker()
        # Decay trust to near zero
        tracker.decay(10000)
        assert tracker.state.trust < 0.2
        action = tracker.confirm(-50.0)
        assert action == "recovering"

    def test_stable_at_high_trust(self):
        tracker = TrustTracker()
        # Push trust high
        for _ in range(100):
            tracker.confirm(-50.0)
        assert tracker.state.trust >= 0.9
        action = tracker.confirm(-50.0)
        assert action == "stable"


class TestTrustTrackerDecay:
    def test_decay_reduces_trust(self):
        tracker = TrustTracker()
        tracker.confirm(-50.0)
        assert tracker.state.trust > 0.5
        tracker.decay(seconds_elapsed=10.0)
        assert tracker.state.trust < 0.6

    def test_decay_does_not_go_negative(self):
        tracker = TrustTracker()
        tracker.decay(seconds_elapsed=100000)
        assert tracker.state.trust >= 0.0


class TestTrustStreak:
    def test_failure_streak_increases(self):
        tracker = TrustTracker()
        for _ in range(5):
            tracker.fail()
        assert tracker.state.failure_streak == 5

    def test_streak_triggers_stale(self):
        tracker = TrustTracker(streak_limit=3)
        for _ in range(3):
            action = tracker.fail()
        assert action == "stale"

    def test_streak_reset_on_confirm(self):
        tracker = TrustTracker()
        for _ in range(3):
            tracker.fail()
        assert tracker.state.failure_streak == 3
        tracker.confirm(-50.0)
        assert tracker.state.failure_streak == 0


class TestTrustShouldRebabble:
    def test_default_not_stale(self):
        tracker = TrustTracker(rebabble_threshold=0.30)
        assert not tracker.should_rebabble()

    def test_stale_after_decay(self):
        tracker = TrustTracker(rebabble_threshold=0.30)
        tracker.decay(seconds_elapsed=1000)
        assert tracker.should_rebabble()


class TestTrustStateSerialization:
    def test_roundtrip(self):
        state = TrustState(trust=0.85, confirmations=42, failure_streak=2)
        d = state.to_dict()
        restored = TrustState.from_dict(d)
        assert restored.trust == 0.85
        assert restored.confirmations == 42
        assert restored.failure_streak == 2


class TestIRPersistence:
    def test_save_and_load_ir(self, short_room_ir):
        tracker = TrustTracker()
        tracker.confirm(-50.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_ir(short_room_ir, tracker, path=path)
        loaded_ir, loaded_tracker = load_ir(path)
        np.testing.assert_array_almost_equal(loaded_ir, short_room_ir, decimal=5)
        assert loaded_tracker.state.trust > 0.5
        os.unlink(path)


class TestListSavedModels:
    def test_empty_dir(self, tmp_path):
        models = list_saved_models(str(tmp_path))
        assert models == []

    def test_lists_models(self, short_room_ir):
        tracker = TrustTracker()
        tracker.confirm(-50.0)
        # Create a temp dir with a saved model
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "room_ir_test.json")
        save_ir(short_room_ir, tracker, path=path)
        models = list_saved_models(tmp)
        assert len(models) == 1
        assert models[0]["file"] == "room_ir_test.json"
        assert models[0]["ir_length"] == len(short_room_ir)
        os.unlink(path)
        os.rmdir(tmp)
