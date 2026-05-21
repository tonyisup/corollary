from __future__ import annotations

import pytest

from pfc.streaming_runtime import _select_m4_detector
from stg.speaker import GatedM4Arbitrator


class DummyDetector:
    pass


class TestM4ModeSelection:
    def test_auto_with_both_paths_uses_arbitration(self):
        bank = DummyDetector()
        scorer = DummyDetector()

        detector, label = _select_m4_detector(
            mode="auto",
            bank_detector=bank,
            scorer_detector=scorer,
            threshold=0.5,
        )

        assert isinstance(detector, GatedM4Arbitrator)
        assert label == "bank + scorer arbitration"

    def test_bank_mode_forces_bank_only(self):
        bank = DummyDetector()
        scorer = DummyDetector()

        detector, label = _select_m4_detector(
            mode="bank",
            bank_detector=bank,
            scorer_detector=scorer,
            threshold=0.5,
        )

        assert detector is bank
        assert label == "prototype bank"

    def test_scorer_mode_forces_scorer_only(self):
        bank = DummyDetector()
        scorer = DummyDetector()

        detector, label = _select_m4_detector(
            mode="scorer",
            bank_detector=bank,
            scorer_detector=scorer,
            threshold=0.5,
        )

        assert detector is scorer
        assert label == "fine-tuned scorer"

    def test_arbitration_mode_requires_both_paths(self):
        with pytest.raises(ValueError, match="requires both --m4-bank and --m4-scorer"):
            _select_m4_detector(
                mode="arbitration",
                bank_detector=DummyDetector(),
                scorer_detector=None,
                threshold=0.5,
            )

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid M4 mode"):
            _select_m4_detector(
                mode="nope",
                bank_detector=DummyDetector(),
                scorer_detector=DummyDetector(),
                threshold=0.5,
            )
