"""Tests for per-frame M4 replay analyzer."""

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "m4_replay_analyze.py"
)
import importlib.util
_spec = importlib.util.spec_from_file_location("m4_replay_analyze", SCRIPT_PATH)
m4_replay_analyze = importlib.util.module_from_spec(_spec)
sys.modules["m4_replay_analyze"] = m4_replay_analyze
_spec.loader.exec_module(m4_replay_analyze)

SR = 16000


def _make_synthetic_case(tmp_path, name, verdict, n_frames=10):
    frame_samples = int(SR * 30 / 1000)
    total = n_frames * frame_samples
    audio = np.random.RandomState(42).randn(total).astype(np.float32) * 0.1
    agent = np.random.RandomState(43).randn(total).astype(np.float32) * 0.1
    scenario_dir = tmp_path / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.save(scenario_dir / "live_captured.npy", audio)
    np.save(scenario_dir / "agent_pcm.npy", agent)
    records = []
    for fi in range(n_frames):
        s = fi * frame_samples
        records.append({
            "frame_index": fi,
            "timestamp_s": round(s / SR, 4),
            "start_sample": s,
            "end_sample": s + frame_samples,
            "verdict": verdict if fi > 2 and fi < n_frames - 2 else "SILENCE",
        })
    (scenario_dir / "frame_records.json").write_text(json.dumps(records))
    (scenario_dir / "metadata.json").write_text(json.dumps({"scenario": name, "sr": SR}))
    return scenario_dir


def test_analyze_replay_returns_per_frame_records(tmp_path):
    _make_synthetic_case(tmp_path, "self_only", "SELF")
    result = m4_replay_analyze.analyze_replay(
        replay_dir=str(tmp_path),
        real_ecapa=False,
        m4_template_match_threshold=0.0,
    )
    assert len(result["results"]) == 1
    frames = result["results"][0]["frames"]
    assert len(frames) == 10
    assert all("m4_outcome" in f and "match_score" in f for f in frames)


def test_sweep_thresholds_returns_all_thresholds(tmp_path):
    _make_synthetic_case(tmp_path, "self_only", "SELF")
    thresholds = [0.0, 0.1, 0.2]
    result = m4_replay_analyze.sweep_thresholds(
        replay_dir=str(tmp_path),
        thresholds=thresholds,
        real_ecapa=False,
    )
    assert len(result["thresholds"]) == 3
    assert [t["threshold"] for t in result["thresholds"]] == thresholds


def test_summary_counts_match_frames(tmp_path):
    _make_synthetic_case(tmp_path, "test_scenario", "SELF", n_frames=20)
    result = m4_replay_analyze.analyze_replay(
        replay_dir=str(tmp_path),
        real_ecapa=False,
        m4_template_match_threshold=0.0,
    )
    summary = result["results"][0]["summary"]
    assert summary["total_frames"] == 20
    assert summary["self_gt_frames"] + summary["neg_gt_frames"] <= summary["scored_frames"]


def test_summary_scores_only_speech_active_frames():
    frames = [
        {
            "gt_verdict": "SELF",
            "final_verdict": "SELF",
            "m4_outcome": "accepted",
            "m4_attempted": True,
            "m4_template_match_passed": True,
            "energy_db": -25.0,
        },
        {
            "gt_verdict": "SELF",
            "final_verdict": "SILENCE",
            "m4_outcome": "not_attempted",
            "m4_attempted": False,
            "m4_template_match_passed": False,
            "energy_db": -60.0,
        },
    ]

    summary = m4_replay_analyze._summarize_case("self_with_pause", frames)

    assert summary["total_frames"] == 2
    assert summary["scored_frames"] == 1
    assert summary["tp"] == 1
    assert summary["fn"] == 0
