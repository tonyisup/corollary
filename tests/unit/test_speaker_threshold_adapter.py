"""Tests for SpeakerThresholdAdapter — per-speaker adaptive M4 threshold."""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter, SpeakerThresholdProfile


class TestSpeakerThresholdProfile:
    def test_initial_state(self):
        p = SpeakerThresholdProfile(speaker_id="alice")
        assert p.count == 0
        assert p.mean_confidence == 0.0
        assert p.max_confidence == 0.0
        assert p.min_confidence == 1.0
        assert not p.is_registered

    def test_single_update(self):
        p = SpeakerThresholdProfile(speaker_id="alice", warm_threshold=1)
        p.update(0.5)
        assert p.count == 1
        assert p.mean_confidence == 0.5
        assert p.max_confidence == 0.5
        assert p.min_confidence == 0.5
        assert p.is_registered  # warm_threshold=1

    def test_multiple_updates(self):
        p = SpeakerThresholdProfile(speaker_id="alice", warm_threshold=3)
        p.update(0.3)
        p.update(0.5)
        assert not p.is_registered  # count=2 < warm_threshold=3
        p.update(0.4)
        assert p.is_registered  # count=3 >= warm_threshold=3
        assert p.mean_confidence == 0.4
        assert p.max_confidence == 0.5
        assert p.min_confidence == 0.3

    def test_ema(self):
        p = SpeakerThresholdProfile(speaker_id="alice", warm_threshold=1)
        p.update(0.0, alpha=0.5)
        p.update(1.0, alpha=0.5)
        # EMA: 0.0 -> 0.5*1.0 + 0.5*0.0 = 0.5
        assert p.ema_confidence == 0.5


class TestSpeakerThresholdAdapter:
    def test_default_threshold_for_unknown_speaker(self):
        adapter = SpeakerThresholdAdapter(base_threshold=0.10)
        assert adapter.get_threshold("unknown") == 0.10

    def test_base_threshold_for_self(self):
        adapter = SpeakerThresholdAdapter(base_threshold=0.10)
        assert adapter.get_threshold("self") == 0.10

    def test_adaptive_threshold_after_warm(self):
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=3,
        )
        # Simulate 3 observations from speaker "alice"
        for conf in [0.15, 0.20, 0.18]:
            adapter.update("alice", conf)
        # max_conf=0.20 + margin=0.05 = 0.25, max(base=0.10, 0.25) = 0.25
        # But wait — adaptive only raises, and 0.25 > 0.10, so threshold = 0.25
        assert adapter.get_threshold("alice") == 0.25

    def test_adaptive_threshold_raises_for_overlapping_speaker(self):
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=2,
        )
        # Speaker whose confidence overlaps with self region
        adapter.update("bob", 0.35)
        adapter.update("bob", 0.40)
        # max_conf=0.40 + margin=0.05 = 0.45, max(base=0.10, 0.45) = 0.45
        assert adapter.get_threshold("bob") == 0.45

    def test_adaptive_threshold_for_high_confidence_speaker(self):
        """When a speaker's confidence exceeds base_threshold + margin."""
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=2,
            max_threshold=0.50,
        )
        # Speaker with very high confidence (near self region)
        adapter.update("charlie", 0.42)
        adapter.update("charlie", 0.45)
        # max_conf=0.45 + margin=0.05 = 0.50, capped at max_threshold=0.50
        # max(base=0.10, 0.50) = 0.50
        assert adapter.get_threshold("charlie") == 0.50

    def test_should_accept_self(self):
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=2,
        )
        # Unknown speaker: uses base threshold
        assert adapter.should_accept_self("unknown", 0.15)
        assert not adapter.should_accept_self("unknown", 0.05)

    def test_update_and_get_profile(self):
        adapter = SpeakerThresholdAdapter(warm_threshold=2)
        adapter.update("alice", 0.3)
        adapter.update("alice", 0.5)
        profile = adapter.get_profile("alice")
        assert profile is not None
        assert profile.count == 2
        assert profile.mean_confidence == 0.4
        assert profile.is_registered

    def test_summary(self):
        adapter = SpeakerThresholdAdapter(warm_threshold=1)
        adapter.register_self()
        adapter.update_self(0.6)
        adapter.update("alice", 0.3)
        summary = adapter.summary()
        assert "self" in summary
        assert "alice" in summary
        assert summary["self"]["count"] == 1
        assert summary["alice"]["count"] == 1

    def test_max_threshold_cap(self):
        """Threshold should never exceed max_threshold."""
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=1,
            max_threshold=0.30,
        )
        # Speaker with very high confidence
        adapter.update("dave", 0.90)
        # max_conf=0.90 + margin=0.05 = 0.95, capped at max_threshold=0.30
        # max(base=0.10, 0.30) = 0.30
        assert adapter.get_threshold("dave") == 0.30

    def test_unknown_speaker_defaults_to_base(self):
        adapter = SpeakerThresholdAdapter(base_threshold=0.15)
        assert adapter.get_threshold("nonexistent") == 0.15

    def test_multiple_speakers_independent(self):
        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=2,
        )
        # Alice: low confidence speaker (below base threshold)
        adapter.update("alice", 0.05)
        adapter.update("alice", 0.08)
        # max_conf=0.08 + margin=0.05 = 0.13, max(base=0.10, 0.13) = 0.13
        assert adapter.get_threshold("alice") == 0.13

        # Bob: higher confidence speaker
        adapter.update("bob", 0.20)
        adapter.update("bob", 0.25)
        # max_conf=0.25 + margin=0.05 = 0.30, max(base=0.10, 0.30) = 0.30
        assert adapter.get_threshold("bob") == 0.30
