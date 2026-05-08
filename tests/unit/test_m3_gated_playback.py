import numpy as np

from scripts.m3_live_e2e_harness import (
    build_controller_gated_tts_buffer,
    plan_controller_gated_tts_segments,
)


def test_defer_plan_starts_tts_after_preexisting_human_speech() -> None:
    segments = plan_controller_gated_tts_segments(
        agent_start_s=0.5,
        agent_duration_s=1.0,
        human_start_s=0.0,
        human_end_s=1.2,
        expected_behavior="defer",
        resume_delay_s=0.3,
    )

    assert len(segments) == 1
    assert segments[0]["src_start_s"] == 0.0
    assert segments[0]["dst_start_s"] == 1.5
    assert segments[0]["duration_s"] == 1.0


def test_interrupt_plan_mutes_overlap_and_resumes_remaining_tts_after_human() -> None:
    segments = plan_controller_gated_tts_segments(
        agent_start_s=0.5,
        agent_duration_s=2.0,
        human_start_s=1.1,
        human_end_s=1.7,
        expected_behavior="interrupt_defer_resume",
        resume_delay_s=0.2,
    )

    assert segments == [
        {"src_start_s": 0.0, "dst_start_s": 0.5, "duration_s": 0.6},
        {"src_start_s": 0.6, "dst_start_s": 1.9, "duration_s": 1.4},
    ]


def test_defer_plan_preserves_original_controller_request_time() -> None:
    segments = plan_controller_gated_tts_segments(
        agent_start_s=0.5,
        agent_duration_s=1.0,
        human_start_s=0.0,
        human_end_s=1.2,
        expected_behavior="defer",
        resume_delay_s=0.3,
    )

    assert segments[0]["dst_start_s"] > 0.5
    # The physical playback is delayed, but the controller request still
    # belongs at the original agent_start_s so pre-agent defer can be scored.
    from scripts.m3_live_e2e_harness import controller_request_time_s

    assert controller_request_time_s("defer", original_agent_start_s=0.5) == 0.5


def test_gated_tts_buffer_contains_silence_during_blocked_overlap_then_resumes() -> None:
    sr = 10
    tts = np.arange(1, 21, dtype=np.float32)
    segments = plan_controller_gated_tts_segments(
        agent_start_s=0.5,
        agent_duration_s=2.0,
        human_start_s=1.1,
        human_end_s=1.7,
        expected_behavior="interrupt_defer_resume",
        resume_delay_s=0.2,
    )

    buf = build_controller_gated_tts_buffer(tts, sr=sr, total_samples=35, segments=segments)

    assert np.all(buf[:5] == 0)
    np.testing.assert_array_equal(buf[5:11], tts[:6])
    assert np.all(buf[11:19] == 0)
    np.testing.assert_array_equal(buf[19:33], tts[6:20])
