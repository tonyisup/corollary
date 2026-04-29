from pathlib import Path

from pfc.m2_artifact_replay import frames_from_artifact, frames_from_frame_records
from thalamus.attributor import SourceVerdict


def test_frames_from_frame_records_preserves_world_verdict() -> None:
    records = [
        {"start_s": 0.0, "verdict": "WORLD", "is_speech": True, "rms_db": -30.0},
        {"start_s": 0.03, "verdict": "WORLD", "is_speech": True, "rms_db": -31.0},
    ]
    frames = frames_from_frame_records(records)
    assert frames[0].attribution.verdict == SourceVerdict.WORLD
    assert frames[1].attribution.verdict == SourceVerdict.WORLD
    assert frames[-1].attribution.verdict == SourceVerdict.SILENCE


def test_frames_from_artifact_prefers_frame_verdicts(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        '{"self_frames": 0, "both_frames": 2, "world_frames": 0, "speech_ratio": 1.0}',
        encoding="utf-8",
    )
    (tmp_path / "frame_verdicts.jsonl").write_text(
        '{"start_s": 0.0, "verdict": "WORLD", "is_speech": true, "rms_db": -30.0}\n',
        encoding="utf-8",
    )
    frames = frames_from_artifact(tmp_path)
    assert frames[0].attribution.verdict == SourceVerdict.WORLD
