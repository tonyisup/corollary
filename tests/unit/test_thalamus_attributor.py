import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import ThalamicAttributor, SourceVerdict


def test_attributor_silence():
    attributor = ThalamicAttributor()
    # rms_db below threshold (-55.0 default)
    res = attributor.attribute(match_score=0.9, observed_onset_samples=100, expected_onset_samples=100, rms_db=-60.0, is_playing=True)
    assert res.verdict == SourceVerdict.SILENCE


def test_attributor_self():
    attributor = ThalamicAttributor()
    # High match, timely, playing
    res = attributor.attribute(match_score=0.8, observed_onset_samples=800, expected_onset_samples=800, rms_db=-30.0, is_playing=True)
    assert res.verdict == SourceVerdict.SELF
    assert res.confidence > 0.7


def test_attributor_speech_overlap_is_both():
    attributor = ThalamicAttributor()
    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
    )
    assert res.verdict == SourceVerdict.BOTH


def test_attributor_world_replay():
    attributor = ThalamicAttributor()
    # High match, but NOT playing
    res = attributor.attribute(match_score=0.8, observed_onset_samples=800, expected_onset_samples=None, rms_db=-30.0, is_playing=False)
    assert res.verdict == SourceVerdict.WORLD


def test_attributor_both_masking():
    attributor = ThalamicAttributor()
    # Playing, but poor match or wrong timing
    res = attributor.attribute(match_score=0.2, observed_onset_samples=1200, expected_onset_samples=800, rms_db=-30.0, is_playing=True)
    assert res.verdict == SourceVerdict.BOTH


def test_attributor_untimely_self():
    attributor = ThalamicAttributor()
    # High match, but wrong timing (e.g., late reflection mistaken for self)
    res = attributor.attribute(match_score=0.8, observed_onset_samples=1200, expected_onset_samples=800, rms_db=-30.0, is_playing=True)
    # Causal score will be 0.0 because error is (1200-800)/16 = 25ms, which is >= tolerance 25ms
    assert res.verdict == SourceVerdict.BOTH


class DummyDetector:
    def __init__(self, is_self: bool, confidence: float) -> None:
        self._is_self = is_self
        self._confidence = confidence

    def is_self(self, pcm: np.ndarray) -> tuple[bool, float]:
        return self._is_self, self._confidence


def test_attributor_m4_override_promotes_self_under_speech() -> None:
    attributor = ThalamicAttributor(
        m4_detector=DummyDetector(True, 0.9),
        m4_self_threshold=0.4,
        sr=16000,
    )
    frame = np.ones(16000 // 30, dtype=np.float32)

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=np.tile(frame, 20),
    )

    assert res.verdict == SourceVerdict.SELF
    assert res.metadata is not None
    assert res.metadata["m4_confidence"] == 0.9


def test_attributor_m4_falls_back_to_both_when_not_self() -> None:
    attributor = ThalamicAttributor(
        m4_detector=DummyDetector(False, 0.2),
        m4_self_threshold=0.4,
        sr=16000,
    )
    frame = np.ones(16000 // 30, dtype=np.float32)

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=np.tile(frame, 20),
    )

    assert res.verdict == SourceVerdict.BOTH
