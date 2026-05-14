"""Tests for M7 phase 4: Mobile device-profile layer."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.device_profile import (
    DeviceOrientation,
    DeviceProfile,
    DeviceProfileBank,
    OrientationSensor,
)


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


class TestDeviceOrientation:
    def test_canonical_orientations(self):
        assert DeviceOrientation.PORTRAIT.value == "portrait"
        assert DeviceOrientation.LANDSCAPE.value == "landscape"
        assert DeviceOrientation.POCKET.value == "pocket"
        assert DeviceOrientation.FACE_DOWN.value == "face_down"
        assert DeviceOrientation.ON_TABLE.value == "on_table"
        assert DeviceOrientation.UNKNOWN.value == "unknown"

    def test_from_string(self):
        assert DeviceOrientation.from_string("portrait") == DeviceOrientation.PORTRAIT
        assert DeviceOrientation.from_string("landscape") == DeviceOrientation.LANDSCAPE
        assert DeviceOrientation.from_string("pocket") == DeviceOrientation.POCKET
        assert DeviceOrientation.from_string("face_down") == DeviceOrientation.FACE_DOWN
        assert DeviceOrientation.from_string("on_table") == DeviceOrientation.ON_TABLE
        assert DeviceOrientation.from_string("bogus") == DeviceOrientation.UNKNOWN


class TestDeviceProfile:
    def test_profile_initializes_per_orientation(self):
        prof = DeviceProfile(DeviceOrientation.PORTRAIT, ir_length=1600)
        assert prof.orientation == DeviceOrientation.PORTRAIT
        assert prof.ir_length == 1600
        assert prof.guess_count == 0

    def test_profile_guess_ir(self):
        prof = DeviceProfile(DeviceOrientation.PORTRAIT, ir_length=1600)
        prof.guess_ir(portrait_tilt_deg=45.0)
        assert prof.guess_count == 1
        assert prof.ir[0] == 1.0  # always has direct path

    def test_profile_serialization(self):
        prof = DeviceProfile(DeviceOrientation.LANDSCAPE, ir_length=800)
        prof.guess_ir(landscape_tilt_deg=90.0)
        data = prof.to_dict()
        assert data["orientation"] == "landscape"
        assert data["ir_length"] == 800
        restored = DeviceProfile.from_dict(data)
        assert restored.orientation == DeviceOrientation.LANDSCAPE
        assert restored.ir_length == 800


class TestOrientationSensor:
    def test_mock_sensor_returns_configured_orientation(self):
        sensor = OrientationSensor(mock_orientation=DeviceOrientation.PORTRAIT)
        assert sensor.current_orientation() == DeviceOrientation.PORTRAIT

    def test_mock_sensor_can_change(self):
        sensor = OrientationSensor(mock_orientation=DeviceOrientation.PORTRAIT)
        sensor.set_mock_orientation(DeviceOrientation.LANDSCAPE)
        assert sensor.current_orientation() == DeviceOrientation.LANDSCAPE

    def test_mock_sensor_stable_frames(self):
        sensor = OrientationSensor(mock_orientation=DeviceOrientation.PORTRAIT)
        assert sensor.orientation_stable_frames == 0
        sensor.set_mock_orientation(DeviceOrientation.LANDSCAPE)
        assert sensor.orientation_stable_frames == 1


class TestDeviceProfileBank:
    def test_bank_starts_with_unknown_profile(self):
        bank = DeviceProfileBank(ir_length=1600)
        assert bank.active_orientation == DeviceOrientation.UNKNOWN
        assert bank.num_profiles >= 1

    def test_bank_switches_profile_on_orientation_change(self):
        bank = DeviceProfileBank(ir_length=1600)
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        assert bank.active_orientation == DeviceOrientation.PORTRAIT
        bank.update_orientation(DeviceOrientation.LANDSCAPE)
        assert bank.active_orientation == DeviceOrientation.LANDSCAPE

    def test_bank_creates_new_profile_for_new_orientation(self):
        bank = DeviceProfileBank(ir_length=1600)
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        bank.update_orientation(DeviceOrientation.LANDSCAPE)
        bank.update_orientation(DeviceOrientation.POCKET)
        assert bank.num_profiles == 4  # unknown + 3 orientations

    def test_bank_reuses_existing_profile(self):
        bank = DeviceProfileBank(ir_length=1600)
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        bank.update_orientation(DeviceOrientation.LANDSCAPE)
        bank.update_orientation(DeviceOrientation.PORTRAIT)  # back to portrait
        assert bank.num_profiles == 3  # unknown + portrait + landscape

    def test_bank_get_active_ir(self):
        bank = DeviceProfileBank(ir_length=1600)
        ir = bank.get_active_ir()
        assert ir is not None
        assert len(ir) == 1600

    def test_bank_adapts_active_profile(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        initial_ir = bank.get_active_ir().copy()
        for _ in range(5):
            bank.adapt(known_signal, captured)
        final_ir = bank.get_active_ir()
        # IR should have changed from initial
        assert not np.allclose(initial_ir, final_ir)

    def test_bank_serialization_roundtrip(self):
        bank = DeviceProfileBank(ir_length=1600)
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        bank.update_orientation(DeviceOrientation.LANDSCAPE)
        data = bank.to_dict()
        restored = DeviceProfileBank.from_dict(data)
        assert restored.num_profiles == bank.num_profiles
        assert restored.active_orientation == bank.active_orientation


class TestCerebellumDeviceProfileIntegration:
    def test_cerebellum_accepts_device_profile_bank(self, known_signal, short_room_ir):
        """BabblingCerebellum with device profile bank should adapt per orientation."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)
        cerebellum.set_device_profile_bank(bank)

        # Adapt in portrait mode
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        for _ in range(5):
            cerebellum.update(known_signal, captured)

        assert bank.active_orientation == DeviceOrientation.PORTRAIT
        assert bank.num_profiles >= 2  # unknown + portrait

    def test_cerebellum_switches_profile_on_orientation(self, known_signal, short_room_ir):
        """When orientation changes, the cerebellum should use a different profile."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)
        cerebellum.set_device_profile_bank(bank)

        # Adapt in portrait
        bank.update_orientation(DeviceOrientation.PORTRAIT)
        for _ in range(5):
            cerebellum.update(known_signal, captured)
        portrait_ir = bank.get_active_ir().copy()

        # Switch to landscape and adapt
        bank.update_orientation(DeviceOrientation.LANDSCAPE)
        for _ in range(5):
            cerebellum.update(known_signal, captured)
        landscape_ir = bank.get_active_ir()

        # Portrait and landscape IRs should differ
        portrait_profile = bank.get_profile(DeviceOrientation.PORTRAIT)
        landscape_profile = bank.get_profile(DeviceOrientation.LANDSCAPE)
        assert portrait_profile is not None
        assert landscape_profile is not None
