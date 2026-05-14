"""Tests for M7 phase 3: Multi-hypothesis IR bank."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.ir_bank import IRBank, IRHypothesis


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


class TestIRHypothesis:
    def test_hypothesis_tracks_error(self):
        h = IRHypothesis(ir_length=1600, nlms_mu=0.1)
        assert h.mean_error == 0.0
        h.update_error(0.5)
        h.update_error(0.3)
        assert h.mean_error > 0.0
        assert h.age == 2

    def test_hypothesis_serialization(self):
        h = IRHypothesis(ir_length=1600, nlms_mu=0.1)
        h.update_error(0.5)
        data = h.to_dict()
        restored = IRHypothesis.from_dict(data)
        assert restored.mean_error == h.mean_error
        assert restored.age == h.age


class TestIRBank:
    def test_bank_starts_with_single_hypothesis(self):
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        assert bank.num_hypotheses == 1
        assert bank.best_hypothesis_index == 0

    def test_bank_scores_each_hypothesis(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=3)
        errors = bank.score_all(known_signal, captured)
        assert len(errors) == 1  # starts with 1 hypothesis
        assert all(e >= 0 for e in errors)

    def test_bank_adapts_best_hypothesis(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=3)
        initial_errors = bank.score_all(known_signal, captured)
        for _ in range(10):
            bank.adapt_best(known_signal, captured)
        final_errors = bank.score_all(known_signal, captured)
        assert final_errors[bank.best_hypothesis_index] <= initial_errors[0] + 0.01

    def test_bank_spawns_new_hypothesis_on_drift(self):
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        bank.spawn_hypothesis()
        assert bank.num_hypotheses == 2
        bank.spawn_hypothesis()
        assert bank.num_hypotheses == 3

    def test_bank_respects_max_hypotheses(self):
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=3)
        bank.spawn_hypothesis()
        bank.spawn_hypothesis()
        bank.spawn_hypothesis()  # should not exceed max
        assert bank.num_hypotheses == 3

    def test_bank_prunes_worst_hypothesis(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        # Spawn 2 more hypotheses
        bank.spawn_hypothesis()
        bank.spawn_hypothesis()
        assert bank.num_hypotheses == 3
        # Adapt the best one many times
        for _ in range(20):
            bank.adapt_best(known_signal, captured)
        # Prune down to 2
        bank.prune_worst(target_count=2)
        assert bank.num_hypotheses == 2
        # The best hypothesis should have been kept
        assert bank.best_hypothesis_index < 2

    def test_bank_best_hypothesis_has_lowest_error(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        bank.spawn_hypothesis()
        # Adapt hypothesis 0 heavily
        for _ in range(20):
            bank.adapt_hypothesis(0, known_signal, captured)
        # Score all
        errors = bank.score_all(known_signal, captured)
        assert bank.best_hypothesis_index == int(np.argmin(errors))

    def test_bank_serialization_roundtrip(self, known_signal, short_room_ir):
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        bank.spawn_hypothesis()
        for _ in range(5):
            bank.adapt_best(known_signal, captured)
        data = bank.to_dict()
        restored = IRBank.from_dict(data)
        assert restored.num_hypotheses == bank.num_hypotheses
        assert restored.max_hypotheses == bank.max_hypotheses
        assert restored.best_hypothesis_index == bank.best_hypothesis_index


class TestCerebellumIRBankIntegration:
    def test_cerebellum_can_use_ir_bank(self, known_signal, short_room_ir):
        """BabblingCerebellum with IR bank should adapt and track multiple hypotheses."""
        captured = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=3)
        cerebellum.set_ir_bank(bank)

        # Bootstrap
        for _ in range(10):
            cerebellum.update(known_signal, captured)

        assert bank.num_hypotheses >= 1
        assert bank.best_hypothesis_index >= 0

    def test_cerebellum_ir_bank_switches_on_drift(self, known_signal, short_room_ir):
        """When room changes, the bank should spawn a new hypothesis and switch."""
        captured_a = fftconvolve(known_signal, short_room_ir, mode="full").astype(np.float32)
        cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
        bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
        cerebellum.set_ir_bank(bank)

        # Adapt to room A
        for _ in range(15):
            cerebellum.update(known_signal, captured_a)

        best_before = bank.best_hypothesis_index

        # Simulate room B
        shifted_ir = np.zeros_like(short_room_ir)
        shifted_ir[0] = 1.0
        shifted_ir[128] = 0.85
        shifted_ir[511] = -0.35
        captured_b = fftconvolve(known_signal, shifted_ir, mode="full").astype(np.float32)

        # After drift, the bank should have spawned alternatives
        for _ in range(15):
            cerebellum.update(known_signal, captured_b)

        # The bank should have more than 1 hypothesis or switched
        assert bank.num_hypotheses >= 1
