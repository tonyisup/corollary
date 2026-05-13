"""Test: synthetic streaming with per-speaker adaptive threshold.

Validates that the synthetic path in run_synthetic exercises the
SpeakerThresholdAdapter, so the full M4 adaptive threshold pipeline
can be tested without hardware.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from thalamus.attributor import SourceVerdict
from pfc.streaming_runtime import (
    StreamingConfig, StreamingRuntime, SyntheticAttributionFrame,
)
from pfc.self_model import SelfModel
from sag.graph import SemanticAbstractionGraph


class TestSyntheticAdaptiveThreshold:
    def test_synthetic_adapts_to_world_speaker(self):
        """Synthetic WORLD frames with high M4 conf → adapter raises threshold."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
        )
        runtime = StreamingRuntime(config)

        # Simulate WORLD frames with high M4 confidence (like speaker 3)
        frames = [
            SyntheticAttributionFrame(
                time_s=0.0, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-30.0, m4_outcome="accepted", m4_confidence=0.25,
                speaker_id="speaker_alice",
            ),
            SyntheticAttributionFrame(
                time_s=0.5, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-32.0, m4_outcome="accepted", m4_confidence=0.28,
                speaker_id="speaker_alice",
            ),
            SyntheticAttributionFrame(
                time_s=1.0, verdict=SourceVerdict.SELF, is_speech=True,
                is_in_motor_window=True, energy_db=-25.0,
                m4_outcome="accepted", m4_confidence=0.55,
                speaker_id="self",
            ),
        ]

        logs = runtime.run_synthetic(frames)
        assert len(logs) == 3

        # Adapter should have learned speaker_alice's profile
        threshold = adapter.get_threshold("speaker_alice")
        assert threshold > 0.10, f"Threshold should be raised, got {threshold}"

        # SELF should still use base threshold
        self_threshold = adapter.get_threshold("self")
        assert self_threshold == 0.10

    def test_synthetic_self_model_receives_speaker_ids(self):
        """SelfModel processes frame logs with per-speaker IDs from synthetic run."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        sag = SemanticAbstractionGraph()
        self_model = SelfModel()
        self_model.initialize_sag(sag)

        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
            self_model=self_model,
        )
        runtime = StreamingRuntime(config)

        frames = [
            SyntheticAttributionFrame(
                time_s=0.0, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-30.0, m4_outcome="accepted", m4_confidence=0.20,
                speaker_id="speaker_alice",
            ),
            SyntheticAttributionFrame(
                time_s=0.5, verdict=SourceVerdict.SELF, is_speech=True,
                is_in_motor_window=True, energy_db=-25.0,
                m4_outcome="accepted", m4_confidence=0.50,
                speaker_id="self",
            ),
            SyntheticAttributionFrame(
                time_s=1.0, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-28.0, m4_outcome="accepted", m4_confidence=0.15,
                speaker_id="speaker_bob",
            ),
        ]

        logs = runtime.run_synthetic(frames)
        assert len(logs) == 3

        # Verify frame logs have speaker_id
        assert logs[0].speaker_id == "speaker_alice"
        # SELF frame → current_speaker_id set to None
        assert logs[1].speaker_id is None
        assert logs[2].speaker_id == "speaker_bob"

        # SelfModel should have extracted per-speaker IDs
        frame_dicts = [runtime._log_to_dict(l) for l in logs]
        speakers = self_model._extract_known_speakers(frame_dicts)
        assert "speaker_alice" in speakers
        assert "speaker_bob" in speakers
        # Frame 1 (SELF) has speaker_id=None, so it falls back to verdict-based "self"
        assert "self" in speakers

    def test_synthetic_without_adapter_unchanged(self):
        """Without adapter, synthetic behavior is unchanged."""
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
        )
        runtime = StreamingRuntime(config)

        frames = [
            SyntheticAttributionFrame(
                time_s=0.0, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-30.0, m4_confidence=0.25, speaker_id="speaker_alice",
            ),
            SyntheticAttributionFrame(
                time_s=0.5, verdict=SourceVerdict.SELF, is_speech=True,
                is_in_motor_window=True, energy_db=-25.0,
                m4_confidence=0.55, speaker_id="self",
            ),
        ]

        logs = runtime.run_synthetic(frames)
        assert len(logs) == 2
        # Without adapter, speaker_id in logs should be None
        assert logs[0].speaker_id is None
        assert logs[1].speaker_id is None

    def test_synthetic_adapter_summary(self):
        """After synthetic run, adapter summary shows learned profiles."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
        )
        runtime = StreamingRuntime(config)

        frames = [
            SyntheticAttributionFrame(
                time_s=0.0, verdict=SourceVerdict.WORLD, is_speech=True,
                energy_db=-30.0, m4_confidence=0.20, speaker_id="speaker_alice",
            ),
            SyntheticAttributionFrame(
                time_s=0.5, verdict=SourceVerdict.SELF, is_speech=True,
                is_in_motor_window=True, energy_db=-25.0,
                m4_confidence=0.50, speaker_id="self",
            ),
        ]

        runtime.run_synthetic(frames)

        summary = adapter.summary()
        assert "speaker_alice" in summary
        assert "self" in summary
        assert summary["self"]["mean_conf"] > summary["speaker_alice"]["mean_conf"]
