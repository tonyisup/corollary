"""Tests for deterministic M7 trust-gated room adaptation replay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


from scripts.m7_trust_gated_replay import run_replay


def test_m7_replay_improves_room_b_error_and_holds_non_self():
    result = run_replay(seed=7, adaptation_cycles=8)

    summary = result["summary"]
    assert summary["passed"] is True
    assert summary["trusted_self_updates"] > 0
    assert summary["held_world_updates"] == 0
    assert summary["held_both_updates"] == 0
    assert summary["room_b_post_error"] < summary["room_b_pre_error"]
    assert summary["improvement_ratio"] >= 0.20


def test_m7_replay_cli_writes_json(tmp_path: Path):
    output = tmp_path / "m7_replay.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/m7_trust_gated_replay.py",
            "--seed",
            "11",
            "--adaptation-cycles",
            "4",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["summary"]["passed"] is True
    assert data["summary"]["trusted_self_updates"] > 0
    assert "M7 trust-gated replay" in result.stdout
