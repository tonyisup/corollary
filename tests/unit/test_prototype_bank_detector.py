from __future__ import annotations

import numpy as np
import pytest

from stg.speaker import PrototypeBankSelfVoiceDetector


class DummyIdentifier:
    def get_embedding(self, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
        return np.asarray(pcm, dtype=np.float32)


class TestPrototypeBankSelfVoiceDetector:
    def test_margin_scores_self_above_negative(self):
        detector = PrototypeBankSelfVoiceDetector(DummyIdentifier(), threshold=-0.05)
        detector.register_self_bank([
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
        ])
        detector.register_negative_bank([
            np.array([0.0, 1.0], dtype=np.float32),
        ])

        is_self, margin = detector.is_self(np.array([1.0, 0.0], dtype=np.float32))
        assert is_self is True
        assert margin > 0.0

    def test_margin_scores_negative_below_threshold(self):
        detector = PrototypeBankSelfVoiceDetector(DummyIdentifier(), threshold=-0.05)
        detector.register_self_bank([
            np.array([1.0, 0.0], dtype=np.float32),
        ])
        detector.register_negative_bank([
            np.array([0.0, 1.0], dtype=np.float32),
        ])

        is_self, margin = detector.is_self(np.array([0.0, 1.0], dtype=np.float32))
        assert is_self is False
        assert margin < 0.0

    def test_bank_round_trip(self, tmp_path):
        detector = PrototypeBankSelfVoiceDetector(DummyIdentifier(), threshold=-0.05)
        detector.register_self_bank([
            np.array([1.0, 0.0], dtype=np.float32),
        ])
        detector.register_negative_bank([
            np.array([0.0, 1.0], dtype=np.float32),
        ])

        bank_path = tmp_path / "bank.npz"
        detector.save_bank(str(bank_path))

        restored = PrototypeBankSelfVoiceDetector(DummyIdentifier())
        restored.load_bank(str(bank_path))

        assert len(restored.self_bank) == 1
        assert len(restored.neg_bank) == 1
        assert restored.threshold == pytest.approx(-0.05)
        is_self, margin = restored.is_self(np.array([1.0, 0.0], dtype=np.float32))
        assert is_self is True
        assert margin > 0.0
