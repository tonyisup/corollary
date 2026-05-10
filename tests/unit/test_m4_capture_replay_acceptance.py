"""Tests for deterministic M4 captured replay acceptance."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "m4_capture_replay_acceptance.py"
)
import importlib.util

_spec = importlib.util.spec_from_file_location("m4_capture_replay_acceptance", SCRIPT_PATH)
m4_capture_replay_acceptance = importlib.util.module_from_spec(_spec)
sys.modules["m4_capture_replay_acceptance"] = m4_capture_replay_acceptance
_spec.loader.exec_module(m4_capture_replay_acceptance)


def _result(*, aggregate_f1=0.966, eligible_f1=1.0, neg_fp=0, self_f1=0.966):
    return SimpleNamespace(
        aggregate_f1=aggregate_f1,
        aggregate_eligible_f1=eligible_f1,
        cases=[
            {
                "name": "replay_same_domain_negative_playback_window",
                "fp": neg_fp,
                "eligible_fp": 0,
                "f1": 0.0,
            },
            {
                "name": "replay_self_only",
                "fp": 0,
                "eligible_fp": 0,
                "f1": self_f1,
            },
        ],
    )


def test_verdict_passes_when_eligible_and_negative_fp_gates_pass():
    verdict = m4_capture_replay_acceptance.build_verdict(
        _result(),
        replay_dir="data/m4_capture_validation_heldout_20260509",
        register_from_clean="data/m4_clean_captures_fixed",
        fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    assert verdict["pass"] is True
    assert verdict["acceptance_criteria"]["eligible_f1_passed"] is True
    assert verdict["acceptance_criteria"]["negative_fp_passed"] is True


def test_verdict_fails_when_same_domain_negative_has_false_positives():
    verdict = m4_capture_replay_acceptance.build_verdict(
        _result(neg_fp=1),
        replay_dir="capture",
        register_from_clean="clean",
        fine_tuned_scorer="scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    assert verdict["pass"] is False
    assert verdict["acceptance_criteria"]["negative_fp_passed"] is False


def test_verdict_requires_expected_replay_cases():
    result = SimpleNamespace(aggregate_f1=1.0, aggregate_eligible_f1=1.0, cases=[])

    verdict = m4_capture_replay_acceptance.build_verdict(
        result,
        replay_dir="capture",
        register_from_clean="clean",
        fine_tuned_scorer="scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    assert verdict["pass"] is False
    assert verdict["acceptance_criteria"]["required_cases_present"] is False


def test_verdict_accepts_tts_suffixed_case_names():
    """Simulated artifacts use names like 'self_only_tts' and 'negative_playback_tts'."""
    result = SimpleNamespace(
        aggregate_f1=0.96,
        aggregate_eligible_f1=0.99,
        cases=[
            {"name": "replay_self_only_tts", "fp": 0, "eligible_fp": 0, "f1": 0.96},
            {"name": "replay_same_domain_negative_playback_window_tts", "fp": 0, "eligible_fp": 0, "f1": 0.0},
        ],
    )

    verdict = m4_capture_replay_acceptance.build_verdict(
        result,
        replay_dir="simulated",
        register_from_clean="clean",
        fine_tuned_scorer="scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    assert verdict["acceptance_criteria"]["required_cases_present"] is True
    assert verdict["pass"] is True


def test_verdict_sums_negative_fps_across_all_negative_scenarios():
    """When there are multiple negative scenarios, FPs are summed."""
    result = SimpleNamespace(
        aggregate_f1=0.96,
        aggregate_eligible_f1=0.99,
        cases=[
            {"name": "replay_self_only", "fp": 0, "eligible_fp": 0, "f1": 0.96},
            {"name": "replay_same_domain_negative_playback_window_tts", "fp": 3, "eligible_fp": 1, "f1": 0.0},
            {"name": "replay_same_domain_negative_playback_window_room", "fp": 2, "eligible_fp": 0, "f1": 0.0},
        ],
    )

    verdict = m4_capture_replay_acceptance.build_verdict(
        result,
        replay_dir="simulated",
        register_from_clean="clean",
        fine_tuned_scorer="scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    assert verdict["negative_fp"] == 5
    assert verdict["negative_eligible_fp"] == 1
    assert verdict["pass"] is False


def test_verdict_eligible_f1_is_primary_gate():
    """Eligible-frame F1 is the primary gate; aggregate F1 is diagnostic."""
    # High eligible F1 but low aggregate F1 (many negative scenarios)
    result = SimpleNamespace(
        aggregate_f1=0.88,
        aggregate_eligible_f1=0.96,
        cases=[
            {"name": "replay_self_only", "fp": 0, "eligible_fp": 0, "f1": 0.96},
            {"name": "replay_same_domain_negative_playback_window", "fp": 0, "eligible_fp": 0, "f1": 0.0},
        ],
    )

    verdict = m4_capture_replay_acceptance.build_verdict(
        result,
        replay_dir="capture",
        register_from_clean="clean",
        fine_tuned_scorer="scorer.pt",
        threshold=0.7,
        min_aggregate_f1=0.95,
        min_eligible_f1=0.95,
        max_negative_fp=0,
    )

    # Passes because eligible F1 >= 0.95 and negative_fp == 0
    assert verdict["pass"] is True
    # But aggregate F1 is below threshold (diagnostic)
    assert verdict["acceptance_criteria"]["aggregate_f1_passed"] is False
