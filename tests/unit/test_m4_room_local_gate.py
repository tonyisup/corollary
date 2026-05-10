"""Tests for room-local M4 gate in ThalamicAttributor."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import ThalamicAttributor, SourceVerdict
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector


def _make_identifier():
    identifier = SpeakerIdentifier(use_real=False)
    identifier._load_model()
    return identifier


class FakeRoomColDetector:
    """Fake M4 detector that always accepts with high confidence."""

    def register_self(self, pcm):
        pass

    def is_self(self, pcm):
        return True, 0.95


def _make_attributor(*, room_filter=None, room_local_threshold=0.04, template_match_threshold=0.20):
    return ThalamicAttributor(
        similarity_threshold=0.04,
        energy_threshold_db=-50.0,
        causal_tolerance_ms=100.0,
        sr=16000,
        m4_detector=FakeRoomColDetector(),
        m4_self_threshold=0.10,
        m4_template_match_threshold=template_match_threshold,
        m4_min_buffer_ms=0.0,
        room_filter=room_filter,
        room_local_threshold=room_local_threshold,
    )


def test_room_local_gate_allows_self_when_template_fails():
    """When ECAPA accepts and room-local score is high, SELF should fire even if raw template is low."""
    sr = 16000
    n_taps = 64
    # Create a room filter that boosts correlation
    room_filter = np.zeros(n_taps, dtype=np.float32)
    room_filter[0] = 1.0
    room_filter[10] = 0.3

    attributor = _make_attributor(room_filter=room_filter, room_local_threshold=0.04)

    # Create captured audio that has low raw correlation but high room-colored correlation
    frame_audio = np.random.RandomState(42).randn(int(sr * 0.03)).astype(np.float32) * 0.1
    agent_pcm = np.random.RandomState(43).randn(int(sr * 0.03)).astype(np.float32) * 0.1

    # Pre-fill the M4 buffer so it's "accepted"
    attributor._m4_buffer = np.random.RandomState(44).randn(int(sr * 0.5)).astype(np.float32) * 0.1

    result = attributor.attribute(
        match_score=0.01,  # Very low raw template score
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=agent_pcm,
        agent_start_sample=0,
    )

    meta = result.metadata or {}
    assert meta.get("m4_outcome") == "accepted", f"Expected accepted, got {meta.get('m4_outcome')}"
    assert result.verdict == SourceVerdict.SELF, f"Expected SELF, got {result.verdict}"


def test_negative_template_correlation_blocks_when_ecapa_accepts_no_room_filter():
    """Without room_filter, ECAPA confidence alone should not drive SELF with anti-correlated template."""
    attributor = _make_attributor(room_filter=None, room_local_threshold=0.5)
    # No room_filter → strict gate applies

    sr = 16000
    frame_audio = np.random.RandomState(42).randn(int(sr * 0.03)).astype(np.float32) * 0.01
    agent_pcm = np.random.RandomState(43).randn(int(sr * 1.0)).astype(np.float32) * 0.01
    attributor._m4_buffer = np.random.RandomState(44).randn(int(sr * 0.5)).astype(np.float32) * 0.1

    result = attributor.attribute(
        match_score=-0.5,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=agent_pcm,
        agent_start_sample=0,
    )

    meta = result.metadata or {}
    assert meta.get("m4_outcome") == "accepted"
    # Without room_filter, negative template correlation should block SELF
    assert result.verdict != SourceVerdict.SELF


def test_captured_audio_allows_self_with_negative_template_and_high_ecapa_conf():
    """With room_filter (captured audio), high ECAPA confidence can drive SELF despite negative template."""
    sr = 16000
    room_filter = np.zeros(64, dtype=np.float32)
    room_filter[0] = 1.0

    attributor = _make_attributor(room_filter=room_filter, room_local_threshold=0.5)
    attributor._m4_buffer = np.random.RandomState(44).randn(int(sr * 0.5)).astype(np.float32) * 0.1

    frame_audio = np.random.RandomState(42).randn(int(sr * 0.03)).astype(np.float32) * 0.01
    agent_pcm = np.random.RandomState(43).randn(int(sr * 1.0)).astype(np.float32) * 0.01

    result = attributor.attribute(
        match_score=-0.5,  # Negative template correlation
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=agent_pcm,
        agent_start_sample=0,
    )

    meta = result.metadata or {}
    assert meta.get("m4_outcome") == "accepted"
    # With room_filter, combined score (0.7 * 0.95 + 0.3 * 0.0 = 0.665) >= 0.10 → SELF
    assert result.verdict == SourceVerdict.SELF


def test_raw_template_match_still_works_without_room_filter():
    """When room_filter is None, raw template match should still gate normally."""
    attributor = _make_attributor(room_filter=None, room_local_threshold=0.04)

    frame_audio = np.random.RandomState(42).randn(int(16000 * 0.03)).astype(np.float32) * 0.1
    agent_pcm = frame_audio.copy()  # Perfect correlation

    attributor._m4_buffer = np.random.RandomState(44).randn(int(16000 * 0.5)).astype(np.float32) * 0.1

    result = attributor.attribute(
        match_score=0.95,  # High raw template score
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=agent_pcm,
        agent_start_sample=0,
    )

    meta = result.metadata or {}
    assert meta.get("m4_outcome") == "accepted"
    assert result.verdict == SourceVerdict.SELF


def test_m4_gate_metadata_reports_both_scores():
    """The metadata should report both template_match_passed and room_local_passed."""
    sr = 16000
    room_filter = np.zeros(64, dtype=np.float32)
    room_filter[0] = 1.0
    room_filter[10] = 0.3

    attributor = _make_attributor(room_filter=room_filter, room_local_threshold=0.04, template_match_threshold=0.5)

    frame_audio = np.random.RandomState(42).randn(int(sr * 0.03)).astype(np.float32) * 0.1
    agent_pcm = np.random.RandomState(43).randn(int(sr * 0.03)).astype(np.float32) * 0.1

    attributor._m4_buffer = np.random.RandomState(44).randn(int(sr * 0.5)).astype(np.float32) * 0.1

    result = attributor.attribute(
        match_score=0.01,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=agent_pcm,
        agent_start_sample=0,
    )

    meta = result.metadata or {}
    assert "m4_template_match_passed" in meta
    assert "room_local_passed" in meta
