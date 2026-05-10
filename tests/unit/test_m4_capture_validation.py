"""Tests for dedicated M4 captured-audio validation artifacts."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "m4_capture_validation.py"
)
_spec = importlib.util.spec_from_file_location("m4_capture_validation", SCRIPT_PATH)
m4_capture_validation = importlib.util.module_from_spec(_spec)
sys.modules["m4_capture_validation"] = m4_capture_validation
_spec.loader.exec_module(m4_capture_validation)


def test_frame_records_label_only_speech_window_as_target_verdict():
    sr = 16000
    records = m4_capture_validation.build_frame_records(
        total_samples=sr,
        speech_start_sample=int(0.3 * sr),
        speech_end_sample=int(0.7 * sr),
        speech_verdict="SELF",
        sr=sr,
        frame_ms=100,
    )

    verdicts = [r["verdict"] for r in records]
    assert verdicts == ["SILENCE", "SILENCE", "SILENCE", "SELF", "SELF", "SELF", "SELF", "SILENCE", "SILENCE", "SILENCE"]
    assert all("timestamp_s" in r and "frame_index" in r for r in records)


def test_write_scenario_artifact_is_replay_compatible(tmp_path):
    sr = 16000
    scenario_dir = tmp_path / "self_only"
    agent_pcm = np.ones(sr, dtype=np.float32) * 0.1
    captured_pcm = np.ones(sr, dtype=np.float32) * 0.05
    records = m4_capture_validation.build_frame_records(
        total_samples=sr,
        speech_start_sample=0,
        speech_end_sample=sr,
        speech_verdict="SELF",
        sr=sr,
        frame_ms=30,
    )

    m4_capture_validation.write_scenario_artifact(
        scenario_dir=scenario_dir,
        agent_pcm=agent_pcm,
        captured_pcm=captured_pcm,
        frame_records=records,
        metadata={"scenario": "self_only", "sr": sr},
    )

    assert np.load(scenario_dir / "agent_pcm.npy").dtype == np.float32
    assert np.load(scenario_dir / "live_captured.npy").dtype == np.float32
    assert (scenario_dir / "frame_records.json").exists()
    assert json.loads((scenario_dir / "metadata.json").read_text())["scenario"] == "self_only"


def test_dry_run_creates_two_minimal_scenarios(tmp_path):
    summary = m4_capture_validation.create_dry_run_artifact(
        output_dir=tmp_path,
        sr=16000,
        duration_s=0.6,
        include_variations=False,
    )

    assert summary["scenario_count"] == 2
    assert (tmp_path / "self_only" / "live_captured.npy").exists()
    assert (tmp_path / "same_domain_negative_playback_window" / "frame_records.json").exists()
    negative_records = json.loads(
        (tmp_path / "same_domain_negative_playback_window" / "frame_records.json").read_text()
    )
    assert {r["verdict"] for r in negative_records} == {"WORLD"}


def test_dry_run_with_variations_adds_room_and_device_scenarios(tmp_path):
    summary = m4_capture_validation.create_dry_run_artifact(
        output_dir=tmp_path,
        sr=16000,
        duration_s=0.6,
        include_variations=True,
    )

    assert summary["scenario_count"] == 6
    assert summary["include_variations"] is True
    scenario_names = {s["scenario"] for s in summary["scenarios"]}
    assert "self_room_colored" in scenario_names
    assert "self_device_eq" in scenario_names
    assert "same_domain_negative_playback_window_room" in scenario_names
    assert "same_domain_negative_playback_window_2" in scenario_names
