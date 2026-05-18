"""Unit tests for the streaming runtime."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from thalamus.attributor import SourceVerdict
from pfc.streaming_runtime import StreamingRuntime, StreamingConfig, SyntheticAttributionFrame


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
            assert "output_source" in entry
            assert "repair_position_s" in entry
            assert "repair_playing" in entry
            assert "repair_completed" in entry
            assert "collision_repair_text" in entry
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


class TestStreamingRuntimeSyntheticValidation:
    def test_synthetic_pre_agent_defer_holds_and_resumes_tts(self):
        """If human is already speaking at request time, TTS should hold then resume."""
        pcm = _make_test_pcm(duration_s=0.6)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test defer",
            gated_playback=True,
            output_request_s=0.30,
            listen_threshold_ms=90.0,
            cooldown_ms=120.0,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        frames = []
        for i in range(30):
            t = i * 0.03
            if t < 0.60:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.WORLD,
                    is_speech=True,
                    energy_db=-25.0,
                ))
            else:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.SILENCE,
                    is_speech=False,
                    energy_db=-70.0,
                ))

        logs = runtime.run_synthetic(frames)
        summary = runtime.get_summary()

        assert summary["deferred_events"] >= 1
        assert summary["resumed_events"] >= 1
        assert summary["tts_played_frames"] > 0
        first_defer = next(l for l in logs if l.deferred)
        assert first_defer.tts_position_s == 0.0
        assert any(l.resumed and l.tts_playing for l in logs)

    def test_synthetic_overlap_interrupt_holds_position_and_resumes(self):
        """Sustained BOTH-overlap should interrupt, hold TTS position, then resume."""
        pcm = _make_test_pcm(duration_s=0.8)
        config = StreamingConfig(
            tts_pcm=pcm,
            repair_pcm=_make_test_pcm(duration_s=0.12),
            tts_text="test interrupt",
            gated_playback=True,
            output_request_s=0.0,
            both_external_min_frames=2,
            both_external_energy_ratio=1.2,
            cooldown_ms=120.0,
            sr=16000,
        )
        runtime = StreamingRuntime(config)
        frames = []
        for i in range(70):
            t = i * 0.03
            if i < 5:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.SELF,
                    is_speech=False,
                    is_in_motor_window=True,
                    self_match_score=0.9,
                    self_match_score_per_frame=1.0,
                    causal_score=1.0,
                ))
            elif i < 10:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.BOTH,
                    is_speech=True,
                    is_in_motor_window=True,
                    self_match_score=0.9,
                    self_match_score_per_frame=2.0,
                    causal_score=1.0,
                ))
            elif i < 16:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.SILENCE,
                    is_speech=False,
                    is_in_motor_window=False,
                    self_match_score_per_frame=0.0,
                    energy_db=-70.0,
                ))
            else:
                frames.append(SyntheticAttributionFrame(
                    time_s=t,
                    verdict=SourceVerdict.SELF,
                    is_speech=False,
                    is_in_motor_window=True,
                    self_match_score=0.9,
                    self_match_score_per_frame=1.0,
                    causal_score=1.0,
                ))

        logs = runtime.run_synthetic(frames)
        summary = runtime.get_summary()

        assert summary["interrupt_events"] >= 1
        assert summary["collision_repair_events"] == 1
        assert summary["repair_played_frames"] > 0
        assert summary["repair_audio_completed"] is True
        assert summary["resumed_events"] >= 1
        assert summary["tts_played_frames"] > 0
        assert summary["tts_completed"] is True
        interrupted = [l for l in logs if l.should_interrupt]
        assert interrupted
        assert [l.collision_repair_text for l in interrupted] == ["Sorry, go ahead."]
        repair_logs = [l for l in logs if l.output_source == "repair"]
        assert repair_logs
        tts_after_repair = [l for l in logs if l.output_source == "tts" and l.timestamp_s > repair_logs[-1].timestamp_s]
        assert tts_after_repair
        assert all(l.controller_state != "LISTEN" for l in repair_logs)
        original_positions_during_repair = [l.tts_position_s for l in repair_logs]
        assert max(original_positions_during_repair) == min(original_positions_during_repair)
        held_positions = [l.tts_position_s for l in logs if l.controller_state == "LISTEN"]
        assert held_positions
        assert max(held_positions) - min(held_positions) <= 0.031
        assert any(l.resumed and l.tts_playing for l in logs)


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

    def test_live_processing_does_not_mute_mic_during_tts(self):
        """Self-listening path must still analyze mic audio while TTS is playing."""
        pcm = _make_test_pcm(duration_s=0.5)
        mic_speech = _make_test_pcm(duration_s=0.39)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
            energy_threshold_db=-55.0,
        )
        runtime = StreamingRuntime(config)
        runtime._start_time = 0.0
        runtime._logs = []

        runtime._process_frame(mic_speech[:6240])

        assert runtime._logs
        assert any(log.vad_speech for log in runtime._logs)
        assert any(log.energy_db > -55.0 for log in runtime._logs)

    def test_realtime_output_render_does_not_run_attribution(self):
        """Realtime playback rendering must stay independent of heavy analysis/M4."""
        pcm = _make_test_pcm(duration_s=0.5)
        config = StreamingConfig(
            tts_pcm=pcm,
            tts_text="test",
            gated_playback=False,
            sr=16000,
        )
        runtime = StreamingRuntime(config)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("attribution must not run in realtime render path")

        runtime._attributor.attribute = fail_if_called

        chunk = runtime._render_output_frame(1920, timestamp_s=0.0)

        assert chunk.shape == (1920,)
        assert np.sqrt(np.mean(chunk ** 2)) > 0.01
        assert runtime._tts_position == 1920

    def test_live_callback_timestamps_use_sample_clock(self):
        """Live callback timestamps should be monotonic and independent of wall-clock jitter."""
        runtime = StreamingRuntime(StreamingConfig(tts_pcm=_make_test_pcm(duration_s=1.0), sr=16000))
        runtime._reset_live_diagnostics()

        t0 = runtime._claim_live_callback_timestamp(1920)
        t1 = runtime._claim_live_callback_timestamp(1920)
        t2 = runtime._claim_live_callback_timestamp(1920)

        assert t0 == 0.0
        assert t1 == 1920 / 16000
        assert t2 == 3840 / 16000
        assert runtime._live_callback_count == 3

    def test_summary_includes_live_analysis_diagnostics(self):
        """Summaries should expose whether async mic analysis dropped data."""
        pcm = _make_test_pcm(duration_s=0.5)
        silence = _make_silence(duration_s=0.5)
        runtime = StreamingRuntime(StreamingConfig(tts_pcm=pcm, gated_playback=False, sr=16000))
        runtime.run_offline(silence)
        runtime._live_callback_count = 4
        runtime._live_analysis_blocks_enqueued = 3
        runtime._live_analysis_blocks_dropped = 1
        runtime._live_analysis_queue_max_depth = 2
        runtime._live_analysis_worker_errors = ["boom"]

        summary = runtime.get_summary()

        assert summary["live_callback_count"] == 4
        assert summary["analysis_blocks_enqueued"] == 3
        assert summary["analysis_blocks_dropped"] == 1
        assert summary["analysis_queue_max_depth"] == 2
        assert summary["analysis_worker_errors"] == ["boom"]


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
        assert log.output_source in {"none", "tts", "repair"}
        assert isinstance(log.repair_position_s, float)
        assert isinstance(log.repair_total_duration_s, float)
        assert isinstance(log.repair_queued, bool)
        assert isinstance(log.repair_playing, bool)
        assert isinstance(log.repair_completed, bool)
        assert isinstance(log.m4_outcome, str)
        assert isinstance(log.energy_db, float)
        assert isinstance(log.self_match_score, float)
        assert isinstance(log.is_in_motor_window, bool)
        assert isinstance(log.deferred, bool)
        assert isinstance(log.resumed, bool)
        assert log.collision_repair_text is None or isinstance(log.collision_repair_text, str)


class TestStreamingAcceptanceScript:
    def test_streaming_acceptance_script_passes(self):
        """The deterministic streaming acceptance script should pass on tracked artifacts."""
        import json
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "scripts/m3_streaming_acceptance.py",
                "--run-dir",
                "data/m3_gated_playback_test",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/juicebox/Projects/corollary",
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        verdict = json.loads(result.stdout)
        assert verdict["pass"] is True
        assert verdict["acceptance_criteria"]["synthetic_pre_agent_defer_passed"] is True
        assert verdict["acceptance_criteria"]["synthetic_overlap_interrupt_resume_passed"] is True
        assert verdict["acceptance_criteria"]["synthetic_collision_repair_passed"] is True
        assert verdict["acceptance_criteria"]["synthetic_repair_audio_passed"] is True
        assert verdict["acceptance_criteria"]["offline_audio_replay_smoke_passed"] is True


def test_hardware_repair_acceptance_script():
    """Run the hardware-repair acceptance wrapper on the tracked live artifact."""
    import subprocess
    import sys
    result = subprocess.run(
        [
            sys.executable,
            "scripts/m3_streaming_repair_hardware_acceptance.py",
            "--run-dir",
            "data/m3_streaming_repair_live_20260509_072237",
            "--output",
            "/tmp/test_hw_repair_verdict.json",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/juicebox/Projects/corollary",
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    verdict = json.loads(Path("/tmp/test_hw_repair_verdict.json").read_text())
    assert verdict["pass"] is True
    assert verdict["criteria"]["interrupt_once"] is True
    assert verdict["criteria"]["collision_repair_once"] is True
    assert verdict["criteria"]["repair_audio_completed"] is True
    assert verdict["criteria"]["repair_audio_played"] is True
    assert verdict["criteria"]["repair_not_during_listen"] is True
    assert verdict["criteria"]["original_tts_held_during_repair"] is True
    assert verdict["criteria"]["tts_resumed_after_repair"] is True
    assert verdict["criteria"]["tts_completed"] is True
    assert verdict["criteria"]["final_idle"] is True


def test_interruption_latency_synthetic():
    """Interruption should be detected within a small number of VAD frames.

    With both_external_min_frames=3, the controller should transition to LISTEN
    within 3-4 frames of sustained BOTH overlap (high energy ratio).
    """
    frames = []
    frame_idx = 0

    # 10 frames of SELF (Corollary speaking alone)
    for i in range(10):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.SELF,
            is_speech=False,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=1.0,
            causal_score=1.0,
        ))
        frame_idx += 1

    # 10 frames of BOTH with high energy (human overlapping)
    first_both_idx = frame_idx
    for i in range(10):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.BOTH,
            is_speech=True,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=2.0,  # Above default 1.5 threshold
            causal_score=1.0,
        ))
        frame_idx += 1

    # 10 frames of SELF (resumes after human stops)
    for i in range(10):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.SELF,
            is_speech=False,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=1.0,
            causal_score=1.0,
        ))
        frame_idx += 1

    config = StreamingConfig(
        tts_pcm=np.zeros(16000 * 5, dtype=np.float32),
        tts_text="latency test",
        gated_playback=True,
        callback_blocksize_ms=120.0,
        sr=16000,
        m4_enable=False,
        both_external_min_frames=3,
        both_external_energy_ratio=1.5,
        cooldown_ms=120.0,
    )
    runtime = StreamingRuntime(config)
    logs = runtime.run_synthetic(frames)

    # Find first LISTEN
    first_listen_idx = None
    for i, log in enumerate(logs):
        if log.controller_state == "LISTEN":
            first_listen_idx = i
            break

    assert first_listen_idx is not None, "Controller should have entered LISTEN"

    # Latency should be <= both_external_min_frames + 1 (for the trigger frame)
    # i.e., at most 4 frames from first BOTH to first LISTEN
    latency_frames = first_listen_idx - first_both_idx
    assert latency_frames <= 4, (
        f"Interruption latency {latency_frames} frames exceeds expected max 4 "
        f"(first_both={first_both_idx}, first_listen={first_listen_idx})"
    )

    # Verify interrupt was flagged
    interrupted = [l for l in logs if l.should_interrupt]
    assert len(interrupted) >= 1

    # Verify system resumed after LISTEN
    summary = runtime.get_summary()
    assert summary["resumed_events"] >= 1
