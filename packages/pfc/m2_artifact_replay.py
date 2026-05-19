"""Replay saved M2 artifacts as M3 turn-taking frame sequences."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from thalamus.attributor import AttributionResult, SourceVerdict

from .m2_turn_bridge import M2Frame


def attribution(verdict: SourceVerdict, is_speech: bool, energy_db: float = -30.0,
                is_in_motor_window: bool | None = None) -> AttributionResult:
    if is_in_motor_window is None:
        is_in_motor_window = verdict in {SourceVerdict.SELF, SourceVerdict.BOTH}
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=0.0,
        is_in_motor_window=is_in_motor_window,
        causal_score=0.0,
        energy_db=energy_db,
        is_speech=is_speech,
    )


def load_m2_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / p
        if candidate.exists():
            p = candidate
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def frames_from_frame_records(
    records: list[dict[str, Any]],
    *,
    max_frames: int = 12,
    cooldown_gap_s: float = 0.35,
) -> list[M2Frame]:
    """Convert saved per-frame M2 verdict records into M3 frames."""
    frames: list[M2Frame] = []
    # Prefer frames that should matter for M3 turn-taking: speech-bearing
    # WORLD/BOTH frames. Fall back to SELF/non-speech verdicts only for controls.
    agentive = [
        row for row in records
        if row.get("is_speech") and row.get("verdict") in {"WORLD", "BOTH"}
    ]
    non_speech_self = [row for row in records if row.get("verdict") == "SELF"]
    other_interesting = [
        row for row in records
        if row.get("verdict") in {"WORLD", "BOTH", "SELF"} or row.get("is_speech")
    ]
    interesting = agentive or non_speech_self or other_interesting
    source = interesting[:max_frames] if interesting else records[:1]
    for row in source:
        verdict = SourceVerdict[str(row.get("verdict", "SILENCE"))]
        frames.append(M2Frame(
            float(row.get("start_s", 0.0)),
            attribution(verdict, bool(row.get("is_speech", False)), float(row.get("rms_db", -30.0))),
        ))
    if frames:
        frames.append(M2Frame(
            frames[-1].time_s + cooldown_gap_s,
            attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0),
        ))
    return frames


def load_frame_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "frame_verdicts.jsonl"
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frames_from_artifact(path: str | Path) -> list[M2Frame]:
    """Load an M2 artifact directory using frame timeline when available."""
    p = Path(path)
    if p.is_dir() and (p / "frame_verdicts.jsonl").exists():
        return frames_from_frame_records(load_frame_records(p))
    return frames_from_m2_summary(load_m2_summary(p))


def frames_for_speak_interruption(
    self_artifact: str | Path,
    overlap_artifact: str | Path,
    *,
    start_s: float = 1.0,
    frame_step_s: float = 0.15,
    cooldown_gap_s: float = 0.35,
) -> list[M2Frame]:
    """Build an artifact-backed M3 SPEAK-interruption sequence.

    This composes two saved M2 artifacts into the minimal sequence needed to
    prove the M3 lifecycle path: first a `SELF` frame that represents active
    agent speech, then speech-bearing `WORLD`/`BOTH` evidence from an overlap
    artifact, then silence/cooldown.
    """
    self_frames = [
        frame for frame in frames_from_artifact(self_artifact)
        if frame.attribution.verdict == SourceVerdict.SELF
    ]
    if not self_frames:
        raise ValueError(f"No SELF frame available in {self_artifact}")

    overlap_frames = [
        frame for frame in frames_from_artifact(overlap_artifact)
        if frame.attribution.is_speech and frame.attribution.verdict in {SourceVerdict.WORLD, SourceVerdict.BOTH}
    ]
    if not overlap_frames:
        raise ValueError(f"No speech-bearing WORLD/BOTH frame available in {overlap_artifact}")

    return [
        M2Frame(start_s, self_frames[0].attribution),
        M2Frame(start_s + frame_step_s, overlap_frames[0].attribution),
        M2Frame(
            start_s + frame_step_s + cooldown_gap_s,
            attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0),
        ),
    ]


def frames_from_m2_summary(
    summary: dict[str, Any],
    *,
    start_s: float = 1.0,
    frame_step_s: float = 0.15,
    cooldown_gap_s: float = 0.35,
) -> list[M2Frame]:
    """Convert aggregate M2 summary counts into a compact M3 frame sequence.

    This is intentionally coarse: M2 summaries currently save aggregate frame
    counts, not the full verdict timeline. The replay therefore synthesizes a
    short representative sequence preserving whether agentive speech-like WORLD
    or BOTH evidence existed.
    """
    both_frames = int(summary.get("both_frames", 0))
    world_frames = int(summary.get("world_frames", 0))
    self_frames = int(summary.get("self_frames", 0))
    speech_ratio = float(summary.get("speech_ratio", 0.0))
    is_speech = speech_ratio > 0.0

    frames: list[M2Frame] = []
    t = start_s
    if both_frames > 0 and is_speech:
        frames.append(M2Frame(t, attribution(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)))
        frames.append(M2Frame(t + frame_step_s, attribution(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)))
        t += frame_step_s
    elif world_frames > 0 and is_speech:
        frames.append(M2Frame(t, attribution(SourceVerdict.WORLD, is_speech=True)))
        frames.append(M2Frame(t + frame_step_s, attribution(SourceVerdict.WORLD, is_speech=True)))
        t += frame_step_s
    elif self_frames > 0:
        frames.append(M2Frame(t, attribution(SourceVerdict.SELF, is_speech=False)))
    else:
        frames.append(M2Frame(t, attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0)))

    frames.append(M2Frame(t + cooldown_gap_s, attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0)))
    return frames
