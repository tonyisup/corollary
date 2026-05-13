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
    assert res.metadata["m4_attempted"] is True
    assert res.metadata["m4_outcome"] == "accepted"
    assert res.metadata["m4_buffer_length"] > 0


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
    assert res.metadata is not None
    assert res.metadata["m4_attempted"] is True
    assert res.metadata["m4_outcome"] == "below_threshold"
    assert res.metadata["m4_confidence"] == 0.2


def test_attributor_m4_exception_returns_both() -> None:
    """Test that M4 exceptions result in BOTH with outcome='exception'."""
    class FailingDetector:
        def is_self(self, pcm):
            raise RuntimeError("M4 failed")

    attributor = ThalamicAttributor(
        m4_detector=FailingDetector(),
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
    assert res.metadata is not None
    assert res.metadata["m4_attempted"] is True
    assert res.metadata["m4_outcome"] == "exception"
    assert res.metadata["m4_confidence"] is None


def test_attributor_m4_buffer_too_short() -> None:
    """Test that M4 is not attempted when buffer is too short."""
    detector = DummyDetector(True, 0.9)
    attributor = ThalamicAttributor(
        m4_detector=detector,
        m4_self_threshold=0.4,
        sr=16000,
        m4_min_buffer_ms=500,  # Need 500ms
    )
    frame = np.ones(16000 // 30, dtype=np.float32)  # ~33ms frame

    # First call - buffer too short
    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,  # Only one frame, not enough
    )

    assert res.verdict == SourceVerdict.BOTH
    assert res.metadata is not None
    assert res.metadata["m4_attempted"] is False
    assert res.metadata["m4_outcome"] == "buffer_too_short"
    assert res.metadata["m4_buffer_length"] < 500 * 16000 / 1000


def test_attributor_m4_not_attempted_no_detector() -> None:
    """Test that M4 is not attempted when no detector is provided."""
    attributor = ThalamicAttributor(
        m4_detector=None,
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
    assert res.metadata is not None
    assert res.metadata["m4_attempted"] is False
    assert res.metadata["m4_outcome"] == "not_attempted"
    assert res.metadata["m4_confidence"] is None


def test_attributor_m4_buffer_accumulation_and_gating():
    """Test that M4 buffer accumulates audio and only invokes M4 when sufficient."""
    import numpy as np

    # Create a dummy detector that tracks if it was called
    class CallTrackingDetector:
        def __init__(self, is_self_return=True, confidence_return=0.9):
            self.is_self_called = False
            self.is_self_return = is_self_return
            self.confidence_return = confidence_return

        def is_self(self, pcm: np.ndarray):
            self.is_self_called = True
            return self.is_self_return, self.confidence_return

    # Use a small buffer threshold so that two frames are enough.
    # Frame size is 30ms = 16000//30 samples.
    frame_size = 16000 // 30  # ~533 samples
    # We want min_buffer_ms such that two frames (60ms) are enough.
    # Let's set m4_min_buffer_ms = 60.0 (requires two frames)
    attributor = ThalamicAttributor(
        m4_detector=CallTrackingDetector(is_self_return=True, confidence_return=0.9),
        m4_self_threshold=0.4,
        sr=16000,
        m4_min_buffer_ms=60.0,  # Need 60ms = ~1066 samples at 16kHz
        m4_window_ms=1000,
    )

    frame = np.ones(frame_size, dtype=np.float32)

    # Initially, buffer should be None
    assert attributor._m4_buffer is None

    # Add first frame - buffer should exist but not enough for M4 (one frame = 30ms < 60ms)
    res1 = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
    )
    # Should fall back to BOTH because buffer not full enough
    assert res1.verdict == SourceVerdict.BOTH
    assert attributor._m4_buffer is not None
    assert len(attributor._m4_buffer) == frame_size
    assert not attributor.m4_detector.is_self_called  # M4 not called yet

    # Add second frame - now we should have enough (two frames = 60ms)
    res2 = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
    )
    # Should now use M4 and return SELF (since detector returns True)
    assert res2.verdict == SourceVerdict.SELF
    assert attributor.m4_detector.is_self_called
    # Buffer should contain two frames (maybe trimmed if exceeded window, but window is 1000ms)
    assert len(attributor._m4_buffer) == frame_size * 2

    # Test that old frames are dropped when exceeding window
    # Window is 1000ms, min_buffer is 60ms, we can add many frames before trimming.
    # Add enough frames to exceed window (1000ms) and ensure trimming occurs.
    window_samples = int(1000 * 16000 / 1000)  # 16000 samples
    # Currently we have 2 frames. Add frames until we exceed window.
    frames_to_add = (window_samples // frame_size) + 1  # enough to go over
    for _ in range(frames_to_add):
        attributor.attribute(
            match_score=0.8,
            observed_onset_samples=800,
            expected_onset_samples=800,
            rms_db=-30.0,
            is_playing=True,
            is_speech=True,
            frame_audio=frame,
        )
    # Buffer should be trimmed to window size
    assert len(attributor._m4_buffer) <= window_samples
    # And we should have called M4 multiple times (at least for the frames after threshold)
    # The detector's is_self_called will be True (it stays True after first call)
    # We just check that it was called at least once.
    assert attributor.m4_detector.is_self_called


def test_attributor_m4_min_buffer_ms_default():
    """Test that the default m4_min_buffer_ms is set correctly."""
    # Default should be 500.0 ms
    attributor = ThalamicAttributor()
    assert attributor.m4_min_buffer_samples == int(500.0 * 16000 / 1000)


def test_attributor_m4_buffer_reset_conditions():
    """Test that buffer is properly managed under different conditions."""
    import numpy as np

    class SimpleDetector:
        def is_self(self, pcm):
            return True, 0.8

    attributor = ThalamicAttributor(
        m4_detector=SimpleDetector(),
        m4_self_threshold=0.4,
        sr=16000,
        m4_min_buffer_ms=300,  # 300ms
        m4_window_ms=1000,
    )

    frame = np.ones(16000 // 30, dtype=np.float32)  # 30ms frame

    # Test 1: Buffer accumulates when playing and speech
    attributor.attribute(
        match_score=0.8, observed_onset_samples=800, expected_onset_samples=800,
        rms_db=-30.0, is_playing=True, is_speech=True, frame_audio=frame
    )
    assert attributor._m4_buffer is not None
    len_before = len(attributor._m4_buffer)

    # Next call with is_speech=False should not accumulate to buffer
    attributor.attribute(
        match_score=0.8, observed_onset_samples=800, expected_onset_samples=800,
        rms_db=-30.0, is_playing=True, is_speech=False, frame_audio=frame
    )
    # Buffer should not have grown
    assert len(attributor._m4_buffer) == len_before

    # Test 2: Buffer does not accumulate when not playing
    attributor.attribute(
        match_score=0.8, observed_onset_samples=800, expected_onset_samples=800,
        rms_db=-30.0, is_playing=False, is_speech=True, frame_audio=frame
    )
    assert len(attributor._m4_buffer) == len_before  # Still same length

def test_per_frame_energy_ratio_ignores_quiet_expected_self_frame() -> None:
    attributor = ThalamicAttributor(per_frame_self_min_rms_db=-50.0)
    frame_audio = np.ones(480, dtype=np.float32) * 0.1  # observed speech/noise
    quiet_self = np.ones(480, dtype=np.float32) * 1e-4  # -80 dB expected self

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=quiet_self,
        agent_start_sample=0,
    )

    assert res.self_match_score_per_frame == 0.0


def test_per_frame_energy_ratio_keeps_loud_expected_self_frame() -> None:
    attributor = ThalamicAttributor(per_frame_self_min_rms_db=-50.0)
    frame_audio = np.ones(480, dtype=np.float32) * 0.2
    loud_self = np.ones(480, dtype=np.float32) * 0.1  # -20 dB expected self

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame_audio,
        self_pcm=loud_self,
        agent_start_sample=0,
    )

    assert res.self_match_score_per_frame is not None
    assert 1.9 < res.self_match_score_per_frame < 2.1


def test_room_local_score_computed_when_filter_provided() -> None:
    attributor = ThalamicAttributor(
        room_filter=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        room_local_threshold=0.208,
    )
    frame = np.ones(480, dtype=np.float32) * 0.1
    self_pcm = np.ones(480 * 10, dtype=np.float32) * 0.1

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
        self_pcm=self_pcm,
        agent_start_sample=0,
    )

    assert res.room_local_score is not None
    assert res.room_local_score > 0.0


def test_room_local_score_none_without_filter() -> None:
    attributor = ThalamicAttributor()
    frame = np.ones(480, dtype=np.float32) * 0.1
    self_pcm = np.ones(480 * 10, dtype=np.float32) * 0.1

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
        self_pcm=self_pcm,
        agent_start_sample=0,
    )

    assert res.room_local_score is None


def test_room_local_self_override_boosts_verdict() -> None:
    """When room-local score exceeds threshold and dry match is weak,
    the verdict should shift from BOTH toward SELF."""
    # Identity room filter (no coloring)
    attributor = ThalamicAttributor(
        similarity_threshold=0.5,
        room_filter=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        room_local_threshold=0.208,
    )
    frame = np.ones(480, dtype=np.float32) * 0.1
    self_pcm = np.ones(480 * 10, dtype=np.float32) * 0.1

    res = attributor.attribute(
        match_score=0.3,  # below similarity_threshold
        observed_onset_samples=0,
        expected_onset_samples=0,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
        self_pcm=self_pcm,
        agent_start_sample=0,
    )

    # Room-local score should be high (identical signals)
    assert res.room_local_score is not None
    assert res.room_local_score >= 0.208
    # Verdict should be SELF due to room-local override
    assert res.verdict == SourceVerdict.SELF


def test_room_local_score_with_exported_filter() -> None:
    """Integration test: load the exported room filter and verify it produces
    a room_local_score when given a matching self capture."""
    from pathlib import Path

    np.random.seed(42)

    filter_path = "data/m3_broca_room_threshold_calibration_broca_controls_reexport.npy"
    h = np.load(filter_path)

    attributor = ThalamicAttributor(
        room_filter=h,
        room_local_threshold=0.208,
    )

    # Create a realistic self PCM (sine wave + noise, like TTS)
    frame_len = 480
    t = np.arange(frame_len * 10, dtype=np.float32) / 16000.0
    self_pcm = (np.sin(2 * np.pi * 440.0 * t) * 0.05).astype(np.float32)
    self_pcm += np.random.randn(len(t)).astype(np.float32) * 0.005

    # Make the observed frame a room-colored version of the self PCM
    # by convolving with the room filter (simulating what the mic hears)
    room_colored = np.convolve(self_pcm, h, mode="same").astype(np.float32)
    frame = room_colored[frame_len:frame_len * 2].copy()

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=frame_len,
        expected_onset_samples=frame_len,
        rms_db=-20.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
        self_pcm=self_pcm,
        agent_start_sample=0,
    )

    # Room-local score should be computed and positive for room-colored self
    assert res.room_local_score is not None
    assert res.room_local_score > 0.0, f"self capture room_local_score={res.room_local_score}"


def test_attributor_m4_unknown_fallback_does_not_poison_adaptive_threshold() -> None:
    """Unknown/live fallback should use base threshold and not ratchet adapter.

    Regression for the live BL-2021USB path: when speaker_id is None, room-colored
    self playback was being funneled into a single "unknown" profile. The adapter
    then raised that profile's threshold above the current frame confidence,
    making M4 acceptance impossible in live runs.
    """
    from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter

    adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
    attributor = ThalamicAttributor(
        m4_detector=DummyDetector(True, 0.18),
        m4_self_threshold=0.10,
        speaker_threshold_adapter=adapter,
        sr=16000,
    )
    frame = np.ones(16000 // 30, dtype=np.float32)

    res = attributor.attribute(
        match_score=0.30,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=np.tile(frame, 20),
        speaker_id=None,
    )

    assert res.metadata is not None
    assert res.metadata["m4_attempted"] is True
    assert res.metadata["m4_confidence"] == 0.18
    assert res.metadata["m4_outcome"] == "accepted"
    assert res.verdict == SourceVerdict.SELF
    assert adapter.get_profile("unknown") is None
