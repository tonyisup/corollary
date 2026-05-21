from __future__ import annotations

import numpy as np

from stg.speaker import GatedM4Arbitrator


class DummyDetector:
    def __init__(self, is_self: bool, confidence: float) -> None:
        self.is_self_value = is_self
        self.confidence_value = confidence
        self.register_calls = 0
        self.self_templates = []

    def is_self(self, pcm: np.ndarray):
        return self.is_self_value, self.confidence_value

    def add_self_prototype(self, pcm: np.ndarray) -> None:
        self.self_templates.append(np.asarray(pcm, dtype=np.float32).copy())

    def register_self(self, pcm: np.ndarray) -> None:
        self.register_calls += 1
        self.self_templates.append(np.asarray(pcm, dtype=np.float32).copy())


class TestGatedM4Arbitrator:
    def test_agreement_returns_self_and_records_primary_source(self):
        bank = DummyDetector(True, 0.18)
        scorer = DummyDetector(True, 0.99)
        arbiter = GatedM4Arbitrator(bank, scorer, threshold=0.5)

        is_self, confidence = arbiter.is_self(np.array([1.0, 0.0], dtype=np.float32))

        assert is_self is True
        assert confidence == 0.18
        assert arbiter.last_result is not None
        assert arbiter.last_result.primary_source == "bank"
        assert arbiter.last_result.disagreed is False
        assert arbiter.last_result.degraded_mode is False

    def test_disagreement_is_conservative_and_logged(self):
        bank = DummyDetector(True, 0.17)
        scorer = DummyDetector(False, 0.99)
        arbiter = GatedM4Arbitrator(bank, scorer, threshold=0.5)

        is_self, confidence = arbiter.is_self(np.array([1.0, 0.0], dtype=np.float32))

        assert is_self is False
        assert confidence == 0.17
        assert arbiter.last_result is not None
        assert arbiter.last_result.disagreed is True
        assert arbiter.last_result.degraded_mode is True
        assert arbiter.last_result.primary_source == "bank"
        md = arbiter.last_result.to_metadata_dict()
        assert md["m4_bank_is_self"] is True
        assert md["m4_scorer_is_self"] is False

    def test_register_self_reaches_both_detectors(self):
        bank = DummyDetector(True, 0.17)
        scorer = DummyDetector(True, 0.99)
        arbiter = GatedM4Arbitrator(bank, scorer, threshold=0.5)

        arbiter.register_self(np.array([0.1, 0.2], dtype=np.float32))

        assert bank.self_templates
        assert scorer.register_calls == 1
        assert scorer.self_templates
