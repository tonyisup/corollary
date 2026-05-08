from types import SimpleNamespace

import numpy as np

from cochlea.vad import VAD_FRAME_SAMPLES
from scripts.m3_live_e2e_harness import LiveScenario, compute_live_metrics, run_attribution_pipeline
from thalamus.attributor import ThalamicAttributor


def decision(should_interrupt: bool = False):
    return SimpleNamespace(should_interrupt=should_interrupt)


def both_motor_frame(gt_human_speaking: bool | None = None) -> dict:
    frame = {"verdict": "BOTH", "is_speech": True, "is_in_motor_window": True}
    if gt_human_speaking is not None:
        frame["gt_human_speaking"] = gt_human_speaking
    return frame


def world_after_frame() -> dict:
    return {"verdict": "WORLD", "is_speech": True, "is_in_motor_window": False}


def test_interrupt_resume_scenario_scores_overlap_interrupt_and_resume() -> None:
    scenario = LiveScenario(
        name="human_early",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="interrupt_defer_resume",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[both_motor_frame()],
        decisions=[decision(should_interrupt=True)],
        accepted_initial_output=True,
        resumed_output="agent",
        transitions=["IDLE", "SPEAK", "LISTEN", "IDLE"],
        latency_ms=1.0,
    )

    assert metrics.true_interrupts == 1
    assert metrics.false_interrupts == 0
    assert metrics.missed_interrupts == 0
    assert metrics.successful_resumes == 1
    assert metrics.total_resumes_expected == 1
    assert metrics.summary()["resume_rate"] == "1/1"


def test_respond_scenario_does_not_count_post_agent_world_as_missed_interrupt() -> None:
    scenario = LiveScenario(
        name="human_after",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="respond",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[world_after_frame()],
        decisions=[decision(should_interrupt=False)],
        accepted_initial_output=True,
        resumed_output=None,
        transitions=["IDLE", "SPEAK", "IDLE"],
        latency_ms=1.0,
    )

    assert metrics.true_interrupts == 0
    assert metrics.false_interrupts == 0
    assert metrics.missed_interrupts == 0
    assert metrics.successful_resumes == 0
    assert metrics.total_resumes_expected == 0
    assert metrics.summary()["resume_rate"] == "n/a"


def test_respond_scenario_interrupt_is_not_scored_as_successful_resume() -> None:
    scenario = LiveScenario(
        name="human_after",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="respond",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[world_after_frame()],
        decisions=[decision(should_interrupt=True)],
        accepted_initial_output=True,
        resumed_output="agent",
        transitions=["IDLE", "SPEAK", "LISTEN", "IDLE"],
        latency_ms=1.0,
    )

    assert metrics.true_interrupts == 0
    assert metrics.false_interrupts == 1
    assert metrics.successful_resumes == 0
    assert metrics.total_resumes_expected == 0
    assert "Expected post-agent respond/listen behavior" in metrics.notes[0]
    assert "Unexpected resumed output" in metrics.notes[1]


def test_defer_scenario_accepting_initial_output_is_false_defer() -> None:
    scenario = LiveScenario(
        name="human_before",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="defer",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[both_motor_frame()],
        decisions=[decision(should_interrupt=False)],
        accepted_initial_output=True,
        resumed_output=None,
        transitions=["IDLE", "SPEAK"],
        latency_ms=1.0,
    )

    assert metrics.false_defers == 1
    assert metrics.total_resumes_expected == 0
    assert metrics.summary()["resume_rate"] == "n/a"


def test_defer_scenario_ignores_interrupts_after_successful_initial_defer() -> None:
    scenario = LiveScenario(
        name="human_before",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="defer",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[both_motor_frame(gt_human_speaking=True)],
        decisions=[decision(should_interrupt=True)],
        accepted_initial_output=False,
        resumed_output="agent",
        transitions=["IDLE", "LISTEN", "IDLE", "SPEAK", "LISTEN"],
        latency_ms=1.0,
    )

    assert metrics.false_defers == 0
    assert metrics.false_interrupts == 0
    assert metrics.successful_resumes == 0
    assert metrics.total_resumes_expected == 0
    assert "fixed-playback harness" in metrics.notes[0]


class NoSpeechVAD:
    def process_frame(self, frame):
        return False


def test_run_attribution_pipeline_offsets_motor_window_by_agent_start_s() -> None:
    agent_pcm = np.ones(VAD_FRAME_SAMPLES * 4, dtype=np.float32) * 0.1
    captured = np.zeros(VAD_FRAME_SAMPLES * 8, dtype=np.float32)
    attributor = ThalamicAttributor(energy_threshold_db=-90.0)

    records = run_attribution_pipeline(
        captured,
        agent_pcm,
        attributor,
        NoSpeechVAD(),
        agent_start_s=(2 * VAD_FRAME_SAMPLES / 16000),
    )

    assert records[0]["is_in_motor_window"] is False
    assert records[1]["is_in_motor_window"] is False
    assert records[2]["is_in_motor_window"] is True
    assert records[5]["is_in_motor_window"] is True
    assert records[6]["is_in_motor_window"] is False


def test_annotate_metric_mode_marks_gated_metrics_as_suppression_not_reflex() -> None:
    from scripts.m3_live_e2e_harness import annotate_metric_mode

    summary = {
        "missed_interrupts": 3,
        "true_interrupts": 0,
        "resume_rate": "0/3",
        "notes": [],
    }

    annotated = annotate_metric_mode(summary, gated_playback=True)

    assert annotated["metric_mode"] == "gated_suppression"
    assert annotated["reflex_metrics_are_diagnostic"] is True
    assert "suppression analyzer" in annotated["notes"][0]


def test_annotate_metric_mode_keeps_ungated_metrics_as_reflex() -> None:
    from scripts.m3_live_e2e_harness import annotate_metric_mode

    annotated = annotate_metric_mode({"notes": []}, gated_playback=False)

    assert annotated["metric_mode"] == "reflex_interrupt_resume"
    assert annotated["reflex_metrics_are_diagnostic"] is False
    assert annotated["notes"] == []


def test_interrupt_resume_uses_scheduled_human_ground_truth_not_both_verdict() -> None:
    scenario = LiveScenario(
        name="human_late",
        captured_audio_file="",
        agent_pcm_file="",
        agent_text="agent",
        expected_behavior="interrupt_defer_resume",
    )

    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=[both_motor_frame(gt_human_speaking=False)],
        decisions=[decision(should_interrupt=True)],
        accepted_initial_output=True,
        resumed_output="agent",
        transitions=["IDLE", "SPEAK", "LISTEN", "IDLE"],
        latency_ms=1.0,
    )

    assert metrics.true_interrupts == 0
    assert metrics.false_interrupts == 1
    assert metrics.missed_interrupts == 0
