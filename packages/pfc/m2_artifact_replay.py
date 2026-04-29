"""Replay saved M2 artifacts as M3 turn-taking frame sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thalamus.attributor import AttributionResult, SourceVerdict

from .m2_turn_bridge import M2Frame


def attribution(verdict: SourceVerdict, is_speech: bool, energy_db: float = -30.0) -> AttributionResult:
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=0.0,
        is_in_motor_window=verdict in {SourceVerdict.SELF, SourceVerdict.BOTH},
        causal_score=0.0,
        energy_db=energy_db,
        is_speech=is_speech,
    )


def load_m2_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


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
        frames.append(M2Frame(t, attribution(SourceVerdict.BOTH, is_speech=True)))
        frames.append(M2Frame(t + frame_step_s, attribution(SourceVerdict.BOTH, is_speech=True)))
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
