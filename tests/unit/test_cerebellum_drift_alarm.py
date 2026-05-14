"""Tests for M7 phase 2: SelfModel drift alarm."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.drift_alarm import DriftAlarm, DriftStatus


SR = 16000


@pytest.fixture()
def short_room_ir():
    ir = np.zeros(int(0.1 * SR), dtype=np.float32)
    ir[0] = 1.0
    ir[16] = 0.35
    ir[48] = 0.15
    return ir


@pytest.fixture()
def known_signal():
    t = np.linspace(0, 1.0, int(1.0 * SR), dtype=np.float32)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    return sig / (np.max(np.abs(sig)) * 1.1)


class TestDriftAlarm:
    def test_stable_error_keeps_nominal_status(self):
        alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0)
        for _ in range(15):
            alarm.record_error(-35.0)
        assert alarm.status == DriftStatus.NOMINAL
        assert alarm.should_spike_adaptation is False
        assert alarm.should_rebabble is False

    def test_error_spike_triggers_adaptation_boost(self):
        alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0)
        for _ in range(10):
            alarm.record_error(-35.0)
        # 5 frames at -20 dB: recent avg = (-35*5 + -20*5)/10 = -27.5, degradation = 7.5 dB
        for _ in range(5):
            alarm.record_error(-20.0)
        assert alarm.status == DriftStatus.DEGRADED
        assert alarm.should_spike_adaptation is True
        assert alarm.should_rebabble is False

    def test_sustained_degradation_triggers_rebabble(self):
        alarm = DriftAlarm(window_frames=5, spike_threshold_db=6.0, rebabble_after_spikes=3)
        for _ in range(5):
            alarm.record_error(-35.0)
        for _ in range(3):
            for _ in range(3):
                alarm.record_error(-20.0)
            alarm.record_error(-35.0)
        assert alarm.status == DriftStatus.STALE
        assert alarm.should_rebabble is True

    def test_recovery_clears_drift(self):
        alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0)
        for _ in range(10):
            alarm.record_error(-35.0)
        for _ in range(5):
            alarm.record_error(-20.0)
        assert alarm.status == DriftStatus.DEGRADED
        for _ in range(10):
            alarm.record_error(-35.0)
        assert alarm.status == DriftStatus.NOMINAL
        assert alarm.should_spike_adaptation is False

    def test_adaptation_rate_multiplier_scales_with_drift(self):
        alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0, max_rate_multiplier=4.0)
        for _ in range(10):
            alarm.record_error(-35.0)
        assert alarm.adaptation_rate_multiplier == 1.0
        for _ in range(5):
            alarm.record_error(-20.0)
        assert alarm.adaptation_rate_multiplier > 1.0
        assert alarm.adaptation_rate_multiplier <= 4.0

    def test_serialization_roundtrip(self):
        alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0)
        for _ in range(5):
            alarm.record_error(-30.0)
        data = alarm.to_dict()
        assert data["status"] == "nominal"
        assert data["window_frames"] == 10
        restored = DriftAlarm.from_dict(data)
        assert restored.status == alarm.status
        assert restored.window_frames == alarm.window_frames


class TestCerebellumDriftIntegration:
    def test_cerebellum_uses_drift_alarm_to_boost_learning_rate(self, known_signal, short_room_ir):
        """When drift is detected, the effective NLMS learning rate should increase."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        alarm = DriftAlarm(window_frames=5, spike_threshold_db=6.0)
        cerebellum.set_drift_alarm(alarm)

        # Bootstrap: stable room A — update() feeds the alarm internally
        for _ in range(10):
            cerebellum.update(known_signal, captured)

        # Simulate room drift: inject high-error frames
        shifted_ir = np.zeros_like(short_room_ir)
        shifted_ir[0] = 1.0
        shifted_ir[128] = 0.85
        shifted_ir[511] = -0.35
        drifted_capture = fftconvolve(known_signal, shifted_ir, mode="full").astype(np.float32)

        for _ in range(5):
            cerebellum.update(known_signal, drifted_capture)

        # After drift detection, the alarm's multiplier should be > 1
        assert alarm.adaptation_rate_multiplier > 1.0

    def test_cerebellum_triggers_rebabble_on_stale_drift(self, known_signal, short_room_ir):
        """Sustained drift should set the rebabble flag."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        alarm = DriftAlarm(window_frames=5, spike_threshold_db=6.0, rebabble_after_spikes=2)
        cerebellum.set_drift_alarm(alarm)

        for _ in range(10):
            cerebellum.update(known_signal, captured)

        shifted_ir = np.zeros_like(short_room_ir)
        shifted_ir[0] = 1.0
        shifted_ir[128] = 0.85
        drifted_capture = fftconvolve(known_signal, shifted_ir, mode="full").astype(np.float32)

        for _ in range(2):
            for _ in range(5):
                cerebellum.update(known_signal, drifted_capture)

        assert alarm.should_rebabble is True
