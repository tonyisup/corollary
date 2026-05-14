"""Test: streaming runtime with speaker identification + adaptive threshold.

Exercises the full pipeline: audio → VAD → attribution → speaker ID →
adaptive threshold. Uses synthetic frames to verify the integration works
end-to-end without requiring real audio or ECAPA models.
"""

import sys
import os
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from pfc.streaming_runtime import StreamingConfig, StreamingRuntime, SyntheticAttributionFrame
from thalamus.attributor import SourceVerdict


class TestStreamingSpeakerAdaptiveIntegration:
    def test_runtime_creates_adapter_and_identifier(self):
        """When speaker_threshold_adapter is provided, runtime creates speaker ID infrastructure."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)  # 1s silence
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
        )
        runtime = StreamingRuntime(config)
        assert runtime._speaker_threshold_adapter is adapter
        # Speaker identifier may be None if ECAPA model isn't available,
        # but the adapter should always be set
        assert runtime._current_speaker_id is None

    def test_runtime_without_adapter_has_no_speaker_id(self):
        """Without adapter, no speaker identification infrastructure."""
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
        )
        runtime = StreamingRuntime(config)
        assert runtime._speaker_threshold_adapter is None
        assert runtime._speaker_world_model is None
        assert runtime._current_speaker_id is None

    def test_synthetic_frames_with_speaker_id(self):
        """Synthetic frames can carry speaker_id for per-speaker testing."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
        )
        runtime = StreamingRuntime(config)

        # Create synthetic frames with speaker_id
        frames = [
            SyntheticAttributionFrame(
                time_s=0.0,
                verdict=SourceVerdict.WORLD,
                is_speech=True,
                energy_db=-30.0,
                speaker_id="speaker_1",
            ),
            SyntheticAttributionFrame(
                time_s=0.5,
                verdict=SourceVerdict.SELF,
                is_speech=True,
                is_in_motor_window=True,
                energy_db=-25.0,
                speaker_id="self",
            ),
        ]

        # run_synthetic doesn't call _update_speaker_id (it bypasses _process_frame),
        # but it should still work without errors
        logs = runtime.run_synthetic(frames)
        assert len(logs) == 2

    def test_adapter_learns_from_replay_confidence_pattern(self):
        """Simulate the adapter learning from a sequence of WORLD then SELF frames.

        This mirrors what happens in the real pipeline: WORLD frames arrive first,
        the adapter raises the threshold, then SELF frames should still pass.
        """
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)

        # Simulate WORLD frames with varying confidence (like speaker 3)
        world_confs = [0.18, 0.22, 0.25, 0.20, 0.23]
        for conf in world_confs:
            adapter.update("world_speaker", conf)

        # After seeing WORLD frames, threshold should be raised
        threshold = adapter.get_threshold("world_speaker")
        assert threshold > 0.10, f"Threshold should be raised above base, got {threshold}"

        # All WORLD frames should be rejected
        for conf in world_confs:
            assert conf < threshold, f"WORLD conf {conf} should be below threshold {threshold}"

        # SELF frames (using "self" profile) should still use base threshold
        self_threshold = adapter.get_threshold("self")
        assert self_threshold == 0.10

        # Typical self confidence (0.3-0.8 range) should pass
        for self_conf in [0.30, 0.45, 0.60, 0.75]:
            assert self_conf >= self_threshold, f"SELF conf {self_conf} should pass base threshold"

    def test_adapter_summary_after_learning(self):
        """Verify adapter summary shows learned profiles."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05, warm_threshold=3)

        for conf in [0.15, 0.20, 0.18]:
            adapter.update("world_speaker", conf)
        for conf in [0.45, 0.50, 0.55]:
            adapter.update("self", conf)

        summary = adapter.summary()
        assert "self" in summary
        assert "world_speaker" in summary
        assert summary["self"]["mean_conf"] > summary["world_speaker"]["mean_conf"]
        assert summary["world_speaker"]["registered"] is True

    def test_update_speaker_id_ignores_both_before_human_registration(self):
        """BOTH frames must not seed cold-start conversational registration."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
            enable_conversational_registration=True,
        )
        runtime = StreamingRuntime(config)
        runtime._speaker_id_interval_frames = 1

        class FakeIdentifier:
            def get_embedding(self, pcm):
                return np.array([1.0, 0.0], dtype=np.float32)

        class FakeRegistry:
            known_speakers = ["self"]

            def best_match(self, embedding):
                return (None, 0.0)

            def identify_speaker(self, embedding):
                return (None, 0.0)

        runtime._speaker_identifier = FakeIdentifier()
        runtime._speaker_world_model = SimpleNamespace(registry=FakeRegistry())

        big_frame = np.ones(int(0.5 * runtime.sr), dtype=np.float32)

        registered_id = runtime._update_speaker_id(
            big_frame,
            SimpleNamespace(verdict=SourceVerdict.BOTH),
        )
        assert registered_id is None
        assert runtime._speaker_id_buffer is None
        assert runtime._current_speaker_id is None
        assert runtime._last_speaker_best_match_id is None

    def test_update_speaker_id_accumulates_both_after_human_registration(self):
        """Known speakers should remain identifiable during BOTH/overlap frames."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
        )
        runtime = StreamingRuntime(config)
        runtime._speaker_id_interval_frames = 1
        runtime._registration_complete = True

        class FakeIdentifier:
            def get_embedding(self, pcm):
                return np.array([1.0, 0.0], dtype=np.float32)

        class FakeRegistry:
            known_speakers = ["self", "speaker_overlap"]

            def best_match(self, embedding):
                return ("speaker_overlap", 0.8)

            def identify_speaker(self, embedding):
                return ("speaker_overlap", 0.8)

        runtime._speaker_identifier = FakeIdentifier()
        runtime._speaker_world_model = SimpleNamespace(registry=FakeRegistry())

        big_frame = np.ones(int(0.5 * runtime.sr), dtype=np.float32)

        runtime._update_speaker_id(
            big_frame,
            SimpleNamespace(verdict=SourceVerdict.BOTH),
        )
        assert runtime._speaker_id_buffer is not None
        assert len(runtime._speaker_id_buffer) >= int(0.5 * runtime.sr)
        assert runtime._current_speaker_id == "speaker_overlap"
        assert runtime._last_speaker_best_match_id == "speaker_overlap"
        assert runtime._last_speaker_match_passed is True
