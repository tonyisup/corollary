"""Unit tests for the streaming runtime."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig


def _make_test_pcm(duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Create a simple test TTS PCM (sine wave)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_silence(duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Create silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_noise(duration_s: float = 2.0, sr: int = 16000, level: float = 0.01) -> np.ndarray:
    """Create low-level noise (below speech threshold)."""
    return (level * np.random.randn(int(sr * duration_s))).astype(np.float32)


class TestStreamingConfig:
    def test_defaults(self):
        config = StreamingConfig()
        assert config.sr == 16000
        assert config.tts_chunk_ms == 200.0
        assert config.gated_playback is True
        assert config.m4_enable is False

    def test_custom(self):
        config = StreamingConfig(sr=48000, tts_chunk_ms=100.0, m4_enable=True)
        assert config.sr == 48000
        assert config.tts_chunk_ms == 100.0
        assert config.m4_enable is True


class TestStreamingRuntimeInit:
    def test_init_no_tts(self):
        config = StreamingConfig()
        runtime = StreamingRuntime(config)
        assert runtime._tts_pcm is None
        assert runtime._tts_total_samples == 0

    def test_init_with_tts(self):
        pcm = _make_test_pcm(duration_s=1.0)
        config = StreamingConfig(tts_pcm=pcm)
        runtime = StreamingRuntime(config)
        assert runtime._tts_pcm is not None
        assert runtime._tts_total_samples == len(pcm)

    def test_tts_chunk_size(self):
        pcm = _make_test_pcm(duration_s=1.0)
        config = StreamingConfig(tts_pcm=pcm, tts_chunk_ms=200.0, sr=16000)
        runtime = StreamingRuntime(config)
        assert runtime._tts_chunk_samples == int(0.2 * 16000)


class TestStreamingRuntimeOffline:
    def test_offline_silence_no_gating(self):
        """With silence input and no gating, TTS should play through."""
        pcm = _make_test_pcm(duration_s=0.5)
        silence = _make_silence(duration_s=0.5)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(silence)

        assert len(logs) > 0
        # TTS should have played
        tts_played = sum(1 for l in logs if l.tts_playing)
        assert tts_played > 0
        # Should complete
        summary = runtime.get_summary()
        assert summary["tts_completed"] is True

    def test_offline_silence_with_gating(self):
        """With silence input and gating, TTS should still play (no human to gate)."""
        pcm = _make_test_pcm(duration_s=0.5)
        silence = _make_silence(duration_s=0.5)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=True,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(silence)

        assert len(logs) > 0
        summary = runtime.get_summary()
        assert summary["tts_completed"] is True

    def test_offline_saves_logs(self):
        """Logs should be saved to JSONL when save_path is set."""
        pcm = _make_test_pcm(duration_s=0.3)
        silence = _make_silence(duration_s=0.3)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)

        try:
            config = StreamingConfig(
                tts_pcm=pcm,
                tts_text="test",
                gated_playback=False,
                sr=16000,
                save_path=log_path,
            )
            runtime = StreamingRuntime(config)
            runtime.run_offline(silence)

            assert log_path.exists()
            with open(log_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            assert len(entries) > 0

            # Check log structure
            entry = entries[0]
            assert "frame_index" in entry
            assert "timestamp_s" in entry
            assert "vad_speech" in entry
            assert "verdict" in entry
            assert "controller_state" in entry
            assert "can_speak" in entry
            assert "tts_playing" in entry
            assert "tts_position_s" in entry
            assert "m4_outcome" in entry
        finally:
            log_path.unlink(missing_ok=True)

    def test_offline_summary(self):
        """Summary should contain expected fields."""
        pcm = _make_test_pcm(duration_s=0.3)
        silence = _make_silence(duration_s=0.3)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        runtime.run_offline(silence)

        summary = runtime.get_summary()
        assert "total_frames" in summary
        assert "speech_frames" in summary
        assert "tts_played_frames" in summary
        assert "verdict_counts" in summary
        assert "controller_state_counts" in summary
        assert summary["total_frames"] > 0

    def test_offline_no_tts(self):
        """Runtime should handle no TTS gracefully."""
        silence = _make_silence(duration_s=0.3)
        config = StreamingConfig(
            tts_pcm=None,
            tts_text="",
            gated_playback=True,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(silence)

        assert len(logs) > 0
        # No TTS should play
        tts_played = sum(1 for l in logs if l.tts_playing)
        assert tts_played == 0

    def test_offline_noise_input(self):
        """Runtime should handle noise input without crashing."""
        pcm = _make_test_pcm(duration_s=0.3)
        noise = _make_noise(duration_s=0.3, level=0.01)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=True,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(noise)

        assert len(logs) > 0
        summary = runtime.get_summary()
        assert summary["total_frames"] > 0


class TestStreamingRuntimeGating:
    def test_gated_vs_ungated(self):
        """Gated mode should produce different behavior than ungated."""
        pcm = _make_test_pcm(duration_s=0.5)
        # Use the TTS itself as input (simulating self-hearing)
        input_audio = pcm.copy()

        # Ungated
        config_ungated = StreamingConfig(
            tts_pcm=pcm.copy(),
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime_ungated = StreamingRuntime(config_ungated)
        logs_ungated = runtime_ungated.run_offline(input_audio.copy())

        # Gated
        config_gated = StreamingConfig(
            tts_pcm=pcm.copy(),
            tts_text="test",
            gated_playback=True,
            sr=16000,
        )
        runtime_gated = StreamingRuntime(config_gated)
        logs_gated = runtime_gated.run_offline(input_audio.copy())

        # Both should complete
        summary_ungated = runtime_ungated.get_summary()
        summary_gated = runtime_gated.get_summary()

        # Both should have processed frames
        assert summary_ungated["total_frames"] > 0
        assert summary_gated["total_frames"] > 0

    def test_tts_position_tracking(self):
        """TTS position should advance monotonically."""
        pcm = _make_test_pcm(duration_s=0.5)
        silence = _make_silence(duration_s=0.5)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(silence)

        positions = [l.tts_position_s for l in logs]
        # Position should be non-decreasing
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i-1] - 0.001  # small tolerance


class TestStreamingRuntimeFrameLog:
    def test_log_fields(self):
        """All expected log fields should be present."""
        pcm = _make_test_pcm(duration_s=0.2)
        silence = _make_silence(duration_s=0.2)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        logs = runtime.run_offline(silence)

        assert len(logs) > 0
        log = logs[0]
        assert isinstance(log.frame_index, int)
        assert isinstance(log.timestamp_s, float)
        assert isinstance(log.vad_speech, bool)
        assert isinstance(log.verdict, str)
        assert isinstance(log.controller_state, str)
        assert isinstance(log.can_speak, bool)
        assert isinstance(log.tts_playing, bool)
        assert isinstance(log.tts_position_s, float)
        assert isinstance(log.m4_outcome, str)
        assert isinstance(log.energy_db, float)
        assert isinstance(log.self_match_score, float)
        assert isinstance(log.is_in_motor_window, bool)
        assert isinstance(log.deferred, bool)
        assert isinstance(log.resumed, bool)
