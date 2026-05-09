import json
from pathlib import Path

import numpy as np

from scripts.m3_gated_playback_analyze import analyze_run, compute_window_rms_db


def write_scenario(root: Path, name: str, metrics: dict, audio: np.ndarray, frame_records: list[dict]) -> None:
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "metrics.json").write_text(json.dumps(metrics))
    (scenario_dir / "frame_records.json").write_text(json.dumps(frame_records))
    np.save(scenario_dir / "live_captured.npy", audio.astype(np.float32))


def test_compute_window_rms_db_returns_ordered_energy_levels() -> None:
    audio = np.zeros(100, dtype=np.float32)
    audio[10:30] = 0.1
    audio[50:70] = 0.001

    loud = compute_window_rms_db(audio, sr=100, start_s=0.1, end_s=0.3)
    quiet = compute_window_rms_db(audio, sr=100, start_s=0.5, end_s=0.7)
    silent = compute_window_rms_db(audio, sr=100, start_s=0.8, end_s=0.9)

    assert loud > quiet > silent
    assert round(loud, 1) == -20.0


def test_analyze_run_summarizes_gated_segments_and_overlap_suppression(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 3, dtype=np.float32)
    audio[int(0.5 * sr):int(1.0 * sr)] = 0.1
    audio[int(1.9 * sr):int(2.5 * sr)] = 0.1
    # Simulated human/blocked window has only low room noise.
    audio[int(1.1 * sr):int(1.7 * sr)] += 0.001

    write_scenario(
        tmp_path,
        "human_early",
        {
            "scenario": "human_early",
            "gated_playback": True,
            "expected_behavior": "interrupt_defer_resume",
            "false_interrupts": 0,
            "false_defers": 0,
            "missed_interrupts": 0,
            "true_interrupts": 1,
            "resume_rate": "1/1",
            "tts_segments": [
                {"src_start_s": 0.0, "dst_start_s": 0.5, "duration_s": 0.6},
                {"src_start_s": 0.6, "dst_start_s": 1.9, "duration_s": 1.4},
            ],
        },
        audio,
        [
            {"start_s": 0.9, "gt_human_speaking": False, "is_in_motor_window": True},
            {"start_s": 1.2, "gt_human_speaking": True, "is_in_motor_window": True},
            {"start_s": 2.0, "gt_human_speaking": False, "is_in_motor_window": True},
        ],
    )

    summary = analyze_run(tmp_path, sr=sr)

    assert summary["scenario_count"] == 1
    assert summary["totals"] == {
        "false_interrupts": 0,
        "false_defers": 0,
        "missed_interrupts": 0,
        "true_interrupts": 1,
    }
    scenario = summary["scenarios"][0]
    assert scenario["scenario"] == "human_early"
    assert scenario["gated_playback"] is True
    assert scenario["tts_segment_count"] == 2
    assert scenario["blocked_window_count"] == 1
    assert scenario["segment_overlap_passed"] is True
    assert scenario["suppression_passed"] is True


def test_analyze_run_validates_defer_segment_after_human_turn(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[int(1.5 * sr):int(2.2 * sr)] = 0.1

    write_scenario(
        tmp_path,
        "human_before",
        {
            "scenario": "human_before",
            "expected_behavior": "defer",
            "gated_playback": True,
            "false_interrupts": 0,
            "false_defers": 0,
            "missed_interrupts": 0,
            "true_interrupts": 0,
            "resume_rate": "n/a",
            "tts_segments": [
                {"src_start_s": 0.0, "dst_start_s": 1.5, "duration_s": 0.7},
            ],
        },
        audio,
        [
            {"start_s": 0.0, "gt_human_speaking": True, "is_in_motor_window": False},
            {"start_s": 1.1, "gt_human_speaking": True, "is_in_motor_window": False},
            {"start_s": 1.6, "gt_human_speaking": False, "is_in_motor_window": True},
        ],
    )

    summary = analyze_run(tmp_path, sr=sr)
    scenario = summary["scenarios"][0]

    assert scenario["defer_passed"] is True
    assert scenario["suppression_passed"] is True


def test_analyze_run_flags_missing_gated_playback(tmp_path: Path) -> None:
    write_scenario(
        tmp_path,
        "human_before",
        {
            "scenario": "human_before",
            "gated_playback": False,
            "false_interrupts": 0,
            "false_defers": 1,
            "missed_interrupts": 0,
            "true_interrupts": 0,
            "resume_rate": "n/a",
            "tts_segments": [],
        },
        np.zeros(16000, dtype=np.float32),
        [],
    )

    summary = analyze_run(tmp_path, sr=16000)

    assert summary["scenarios"][0]["suppression_passed"] is False
    assert "gated_playback is false" in summary["scenarios"][0]["notes"]


def test_acceptance_script_passes_on_saved_artifact() -> None:
    """The acceptance script should pass on the tracked gated playback artifact."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/m3_gated_playback_acceptance.py",
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
    assert verdict["all_suppression_passed"] is True
    assert verdict["acceptance_criteria"]["all_gated_playback"] is True
