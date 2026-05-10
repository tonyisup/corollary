"""Tests for multi-artifact M4 captured replay acceptance."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "m4_multi_artifact_acceptance.py"
)
import importlib.util

_spec = importlib.util.spec_from_file_location("m4_multi_artifact_acceptance", SCRIPT_PATH)
m4_multi_artifact_acceptance = importlib.util.module_from_spec(_spec)
sys.modules["m4_multi_artifact_acceptance"] = m4_multi_artifact_acceptance
_spec.loader.exec_module(m4_multi_artifact_acceptance)


def _mock_verdict(artifact_dir, passed=True, f1=0.966):
    return {
        "artifact_dir": artifact_dir,
        "pass": passed,
        "aggregate_f1": f1,
        "aggregate_eligible_f1": 1.0,
        "negative_fp": 0,
        "acceptance_criteria": {
            "required_cases_present": True,
            "aggregate_f1_passed": passed,
            "eligible_f1_passed": True,
            "negative_fp_passed": True,
            "negative_eligible_fp_passed": True,
        },
        "cases": [
            {"name": "replay_self_only", "fp": 0, "eligible_fp": 0},
            {"name": "replay_same_domain_negative_playback_window", "fp": 0, "eligible_fp": 0},
        ],
    }


def test_multi_artifact_all_pass():
    """When all artifacts pass, the aggregate verdict passes."""
    with patch.object(
        m4_multi_artifact_acceptance, "run_acceptance", side_effect=lambda **kw: _mock_verdict(kw["replay_dir"], passed=True)
    ):
        verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
            artifact_dirs=["data/m4_capture_validation_heldout_20260509"],
            register_from_clean="data/m4_clean_captures_fixed",
            fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
            threshold=0.70,
        )

    assert verdict["pass"] is True
    assert verdict["all_passed"] is True
    assert verdict["artifact_count"] == 1


def test_multi_artifact_one_fails():
    """When one artifact fails, the aggregate verdict fails."""
    call_count = [0]
    def mock_run(**kw):
        call_count[0] += 1
        passed = call_count[0] == 1
        return _mock_verdict(kw["replay_dir"], passed=passed, f1=0.966 if passed else 0.5)

    with patch.object(
        m4_multi_artifact_acceptance, "run_acceptance", side_effect=mock_run
    ):
        verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
            artifact_dirs=[
                "data/m4_capture_validation_heldout_20260509",
                "data/m4_capture_validation_room2",
            ],
            register_from_clean="data/m4_clean_captures_fixed",
            fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
            threshold=0.70,
        )

    assert verdict["pass"] is False
    assert verdict["all_passed"] is False
    assert verdict["artifact_count"] == 2


def test_multi_artifact_missing_dir_reports_error():
    """A missing artifact directory is reported as an error."""
    verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
        artifact_dirs=["data/nonexistent_artifact"],
        register_from_clean="data/m4_clean_captures_fixed",
        fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
        threshold=0.70,
    )

    assert verdict["pass"] is False
    assert verdict["any_errors"] is True
    assert "error" in verdict["results"][0]


def test_multi_artifact_includes_synthetic_when_requested(tmp_path):
    """include_synthetic adds a synthetic battery to the artifact list."""
    with patch.object(
        m4_multi_artifact_acceptance, "run_acceptance", side_effect=lambda **kw: _mock_verdict(kw["replay_dir"], passed=True)
    ):
        verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
            artifact_dirs=["data/m4_capture_validation_heldout_20260509"],
            register_from_clean="data/m4_clean_captures_fixed",
            fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
            threshold=0.70,
            include_synthetic=True,
        )

    # Real artifact count is 1; synthetic is in synthetic_results
    assert verdict["artifact_count"] == 1
    assert len(verdict["synthetic_results"]) == 1
    assert verdict["synthetic_results"][0]["simulated"] is True
    assert verdict["synthetic_results"][0]["gate_exempt"] is True
    # Overall pass depends only on real artifacts
    assert verdict["pass"] is True


def test_multi_artifact_reports_per_artifact_f1():
    """Each artifact result includes its aggregate F1."""
    with patch.object(
        m4_multi_artifact_acceptance, "run_acceptance", side_effect=lambda **kw: _mock_verdict(kw["replay_dir"], passed=True, f1=0.97)
    ):
        verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
            artifact_dirs=["data/m4_capture_validation_heldout_20260509"],
            register_from_clean="data/m4_clean_captures_fixed",
            fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
            threshold=0.70,
        )

    assert len(verdict["results"]) == 1
    assert "aggregate_f1" in verdict["results"][0]
    assert "aggregate_eligible_f1" in verdict["results"][0]
    assert "negative_fp" in verdict["results"][0]


def test_multi_artifact_empty_list():
    """An empty artifact list should pass vacuously."""
    verdict = m4_multi_artifact_acceptance.run_multi_artifact_acceptance(
        artifact_dirs=[],
        register_from_clean="data/m4_clean_captures_fixed",
        fine_tuned_scorer="data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt",
        threshold=0.70,
    )

    assert verdict["artifact_count"] == 0
    assert verdict["all_passed"] is True
    assert verdict["pass"] is True
