import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m3_broca_control_capture import (
    artifact_dir_for_case,
    build_broca_control_cases,
    render_threshold_command,
    update_summary_with_control_metadata,
)


def test_build_broca_control_cases_assigns_threshold_roles() -> None:
    cases = build_broca_control_cases(world_speech="data/world_speech_sample.npy")

    assert [case["name"] for case in cases] == ["silence", "world_only", "overlap"]
    assert [case["threshold_role"] for case in cases] == ["negative", "negative", "diagnostic"]
    assert cases[0]["world_only"] is True
    assert cases[1]["world_only"] is True
    assert cases[2]["world_only"] is False


def test_artifact_dir_for_case_zero_pads_repetitions() -> None:
    path = artifact_dir_for_case("data/broca_controls", "world_only", 2, 12)

    assert path == Path("data/broca_controls/world_only_rep02")


def test_update_summary_with_control_metadata_persists_fields(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"self_score": 0.1}), encoding="utf-8")

    case = {
        "name": "world_only",
        "threshold_role": "negative",
        "world_speech": "data/world_speech_sample.npy",
        "world_only": True,
    }
    update_summary_with_control_metadata(
        summary_path,
        case=case,
        broca_text="I was speaking, then paused.",
        input_device=2,
        output_device=3,
        noise_device=4,
    )

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["control_case"] == "world_only"
    assert saved["threshold_role"] == "negative"
    assert saved["broca_text"] == "I was speaking, then paused."
    assert saved["capture_devices"] == {"input_device": 2, "output_device": 3, "noise_device": 4}


def test_render_threshold_command_uses_negative_cases_only() -> None:
    command = render_threshold_command(
        positives=[Path("data/pos_a"), Path("data/pos_b")],
        captured_cases=[
            {"path": "data/controls/silence_rep0", "threshold_role": "negative"},
            {"path": "data/controls/world_only_rep0", "threshold_role": "negative"},
            {"path": "data/controls/overlap_rep0", "threshold_role": "diagnostic"},
        ],
        output_path=Path("data/out.json"),
    )

    assert "--positive data/pos_a --positive data/pos_b" in command
    assert "--negative data/controls/silence_rep0 --negative data/controls/world_only_rep0" in command
    assert "overlap_rep0" not in command
    assert command.endswith("--output data/out.json")
