from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_for_speak_interruption, frames_from_m2_summary
from pfc.m2_turn_bridge import run_turn_taking_bridge
from thalamus.attributor import SourceVerdict


def test_frames_from_both_speech_summary_defer_and_resume() -> None:
    summary = {"self_frames": 0, "both_frames": 30, "world_frames": 100, "speech_ratio": 0.4}
    frames = frames_from_m2_summary(summary, start_s=1.0, frame_step_s=0.15, cooldown_gap_s=0.5)

    assert frames[0].attribution.verdict == SourceVerdict.BOTH
    assert frames[0].attribution.is_speech is True
    assert frames[-1].attribution.verdict == SourceVerdict.SILENCE

    result = run_turn_taking_bridge(
        frames,
        "queued response",
        controller=AgentiveController(listen_threshold_ms=100, cooldown_ms=200),
    )
    assert result.accepted_initial_output is False
    assert result.resumed_output == "queued response"


def test_frames_from_self_only_summary_do_not_defer() -> None:
    summary = {"self_frames": 30, "both_frames": 0, "world_frames": 100, "speech_ratio": 0.0}
    frames = frames_from_m2_summary(summary, start_s=1.0)

    assert frames[0].attribution.verdict == SourceVerdict.SELF

    result = run_turn_taking_bridge(
        frames,
        "speak now",
        controller=AgentiveController(listen_threshold_ms=100, cooldown_ms=200),
        request_at_s=1.0,
    )
    assert result.accepted_initial_output is True
    assert result.resumed_output is None


def test_artifact_backed_speak_interruption_replay() -> None:
    frames = frames_for_speak_interruption(
        "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0",
        "data/m2_policy/frame_timeline_validation_20260429_140648/vowel_seed0",
    )

    assert frames[0].attribution.verdict == SourceVerdict.SELF
    assert frames[1].attribution.verdict in {SourceVerdict.WORLD, SourceVerdict.BOTH}
    assert frames[1].attribution.is_speech is True
    assert frames[-1].attribution.verdict == SourceVerdict.SILENCE

    result = run_turn_taking_bridge(
        frames,
        "resume me",
        controller=AgentiveController(listen_threshold_ms=100, cooldown_ms=200),
        request_at_s=frames[0].time_s,
    )
    assert result.accepted_initial_output is True
    assert result.decisions[1].should_interrupt is True
    assert result.resumed_output == "resume me"
