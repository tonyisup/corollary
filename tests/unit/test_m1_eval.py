from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from m1_eval_runner import EvalConfig, run_eval
from m1_search import generate_m1c_configs


def test_m1c_eval_runner_returns_scored_result() -> None:
    result = run_eval(EvalConfig(stage="M1C", scenario="mixed_source_overlap", stimulus="chirp", iterations=6))

    assert result["stage"] == "M1C"
    assert result["primary_metric"] == "overlap_f1"
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["metrics"]["overlap_f1"] <= 1.0
    assert 0.0 <= result["metrics"]["world_to_self_confusion"] <= 1.0
    assert 0.0 <= result["metrics"]["replay_false_self_rate"] <= 1.0


def test_m1c_search_configs_emit_expected_stage_and_scenario() -> None:
    config = next(generate_m1c_configs())

    assert config["stage"] == "M1C"
    assert config["scenario"] == "mixed_source_overlap"
    assert config["stimulus"] in {"chirp", "tone", "vowel_ah"}
