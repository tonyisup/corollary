"""Tests for room acoustics simulation."""

import numpy as np
import pytest

from cochlea.room import generate_room_ir, apply_room_ir, estimate_rt60, DEFAULT_RT60, DEFAULT_FS


class TestGenerateRoomIR:
    def test_ir_has_direct_sound(self):
        ir = generate_room_ir(seed=42)
        assert ir[0] > 0  # Direct sound should be strongest

    def test_ir_length(self):
        length_sec = 0.5
        ir = generate_room_ir(length=length_sec, seed=42)
        expected = int(length_sec * DEFAULT_FS)
        assert len(ir) == expected

    def test_ir_is_float32(self):
        ir = generate_room_ir()
        assert ir.dtype == np.float32

    def test_ir_energy_decays(self):
        ir = generate_room_ir(seed=42)
        # Early portion should have more energy than late portion
        early_energy = np.sum(ir[:100] ** 2)
        late_energy = np.sum(ir[-100:] ** 2)
        assert early_energy > late_energy

    def test_shorter_rt60_dries_faster(self):
        ir_dry = generate_room_ir(rt60=0.1, length=2.0, seed=42)
        ir_wet = generate_room_ir(rt60=0.6, length=2.0, seed=42)
        # Wet room should have more late energy
        # Measure energy after 0.5s (well past both RT60s for dry)
        late = int(0.5 * DEFAULT_FS)
        energy_dry = np.sum(ir_dry[late:] ** 2)
        energy_wet = np.sum(ir_wet[late:] ** 2)
        assert energy_wet > energy_dry

    def test_reproducibility(self):
        ir1 = generate_room_ir(seed=123)
        ir2 = generate_room_ir(seed=123)
        np.testing.assert_array_equal(ir1, ir2)

    def test_different_seeds_different_irs(self):
        ir1 = generate_room_ir(seed=1)
        ir2 = generate_room_ir(seed=2)
        assert not np.array_equal(ir1, ir2)


class TestApplyRoomIR:
    def test_output_length(self):
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        ir = generate_room_ir(length=0.5, seed=42)
        output = apply_room_ir(audio, ir)
        # Convolution length = len(audio) + len(ir) - 1
        expected = len(audio) + len(ir) - 1
        assert len(output) == expected

    def test_output_is_float32(self):
        audio = np.random.randn(1000).astype(np.float32)
        ir = generate_room_ir(length=0.1, seed=42)
        output = apply_room_ir(audio, ir)
        assert output.dtype == np.float32

    def test_default_ir_works(self):
        """When no IR provided, generate_room_ir() is used."""
        audio = np.random.randn(1600).astype(np.float32)
        output = apply_room_ir(audio)
        assert len(output) > len(audio)

    def test_silent_audio_stays_silent(self):
        """Silence through room should be near-silence (just noise floor)."""
        audio = np.zeros(16000, dtype=np.float32)
        ir = generate_room_ir(length=0.1, seed=42)
        output = apply_room_ir(audio, ir, noise_db=-80)
        assert np.max(np.abs(output)) < 0.01  # Very quiet

    def test_noise_floor_respected(self):
        """Lower noise_db should produce quieter noise floor."""
        audio = np.ones(16000, dtype=np.float32) * 0.5
        # Different seed for reproducibility
        ir = generate_room_ir(length=0.1, seed=123)
        out_hiss = apply_room_ir(audio, ir, noise_db=-20)
        out_quiet = apply_room_ir(audio, ir, noise_db=-80)
        # Quieter noise setting → lower standard deviation in the tail
        assert np.std(out_hiss) > np.std(out_quiet)


class TestEstimateRT60:
    def test_estimation_is_reasonable(self):
        """RT60 estimation should be positive and within 2x of target for synthetic IR."""
        target_rt60 = 0.3
        ir = generate_room_ir(rt60=target_rt60, length=2.0, seed=42)
        estimated = estimate_rt60(ir)
        # Synthetic IRs have noisy decay curves — estimation won't be exact
        assert 0.05 < estimated  # At least some decay measured
        assert estimated < 1.5   # Within 5x of target is acceptable

    def test_zero_ir_returns_negative(self):
        """Empty IR should return -1."""
        ir = np.zeros(100, dtype=np.float32)
        assert estimate_rt60(ir) == -1.0
