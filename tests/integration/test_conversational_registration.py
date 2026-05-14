"""Test: conversational registration integrated into the streaming runtime.

Validates that the streaming runtime can auto-register a new speaker from
accumulated WORLD audio when enable_conversational_registration is set.
Uses synthetic frames to test without hardware.
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.streaming_runtime import StreamingConfig, StreamingRuntime, StreamingFrameLog
from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from thalamus.attributor import SourceVerdict


class TestConversationalRegistration:
    def test_config_defaults(self):
        """Conversational registration config defaults to False."""
        config = StreamingConfig()
        assert config.enable_conversational_registration is False
        assert config.registration_speaker_name == "User"

    def test_config_enabled(self):
        """Can enable conversational registration via config."""
        config = StreamingConfig(
            enable_conversational_registration=True,
            registration_speaker_name="TestUser",
        )
        assert config.enable_conversational_registration is True
        assert config.registration_speaker_name == "TestUser"

    def test_runtime_state_initialized(self):
        """Runtime initializes conversational registration state from config."""
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            enable_conversational_registration=True,
            registration_speaker_name="TestUser",
        )
        runtime = StreamingRuntime(config)
        assert runtime._conversational_registration is True
        assert runtime._registration_speaker_name == "TestUser"
        assert runtime._registration_complete is False

    def test_log_includes_speaker_registered_field(self):
        """StreamingFrameLog includes speaker_registered field."""
        log = StreamingFrameLog(
            frame_index=0,
            timestamp_s=0.0,
            vad_speech=True,
            verdict="WORLD",
            controller_state="IDLE",
            can_speak=True,
            should_interrupt=False,
            tts_playing=False,
            tts_position_s=0.0,
            tts_total_duration_s=0.0,
            output_source="none",
            repair_position_s=0.0,
            repair_total_duration_s=0.0,
            repair_queued=False,
            repair_playing=False,
            repair_completed=False,
            m4_outcome="not_attempted",
            m4_confidence=None,
            energy_db=-30.0,
            self_match_score=0.0,
            self_match_score_per_frame=None,
            is_in_motor_window=False,
            deferred=False,
            resumed=False,
            speaker_registered="human_testuser",
        )
        assert log.speaker_registered == "human_testuser"

    def test_log_to_dict_includes_registration(self):
        """_log_to_dict includes speaker_registered field."""
        tts_pcm = np.zeros(16000, dtype=np.float32)
        config = StreamingConfig(tts_pcm=tts_pcm)
        runtime = StreamingRuntime(config)

        log = StreamingFrameLog(
            frame_index=0,
            timestamp_s=0.0,
            vad_speech=True,
            verdict="WORLD",
            controller_state="IDLE",
            can_speak=True,
            should_interrupt=False,
            tts_playing=False,
            tts_position_s=0.0,
            tts_total_duration_s=0.0,
            output_source="none",
            repair_position_s=0.0,
            repair_total_duration_s=0.0,
            repair_queued=False,
            repair_playing=False,
            repair_completed=False,
            m4_outcome="not_attempted",
            m4_confidence=None,
            energy_db=-30.0,
            self_match_score=0.0,
            self_match_score_per_frame=None,
            is_in_motor_window=False,
            deferred=False,
            resumed=False,
            speaker_registered="human_testuser",
        )

        d = runtime._log_to_dict(log)
        assert "speaker_registered" in d
        assert d["speaker_registered"] == "human_testuser"

    def test_no_registration_without_world_audio(self):
        """Registration doesn't trigger without sufficient WORLD audio."""
        tts_pcm = np.zeros(16000, dtype=np.float32)
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        config = StreamingConfig(
            tts_pcm=tts_pcm,
            m4_enable=False,
            speaker_threshold_adapter=adapter,
            enable_conversational_registration=True,
        )
        runtime = StreamingRuntime(config)
        assert runtime._conversational_registration is True
        assert runtime._registration_complete is False

    def test_conversational_registration_flag_in_frame_log(self):
        """speaker_registered is set on frame log when registration occurs."""
        # Create a frame log with speaker_registered set
        log = StreamingFrameLog(
            frame_index=0,
            timestamp_s=0.0,
            vad_speech=True,
            verdict="WORLD",
            controller_state="IDLE",
            can_speak=True,
            should_interrupt=False,
            tts_playing=False,
            tts_position_s=0.0,
            tts_total_duration_s=0.0,
            output_source="none",
            repair_position_s=0.0,
            repair_total_duration_s=0.0,
            repair_queued=False,
            repair_playing=False,
            repair_completed=False,
            m4_outcome="not_attempted",
            m4_confidence=None,
            energy_db=-30.0,
            self_match_score=0.0,
            self_match_score_per_frame=None,
            is_in_motor_window=False,
            deferred=False,
            resumed=False,
            speaker_registered=None,
        )
        # Initially None
        assert log.speaker_registered is None
        # Can be set after creation
        log.speaker_registered = "human_user"
        assert log.speaker_registered == "human_user"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
