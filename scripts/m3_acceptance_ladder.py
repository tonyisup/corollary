#!/usr/bin/env python3
"""Run the current M3 acceptance ladder over synthetic and artifact-backed frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import attribution, frames_for_speak_interruption
from pfc.m2_turn_bridge import M2Frame, run_turn_taking_bridge
from thalamus.attributor import SourceVerdict

DEFAULT_SELF_ARTIFACT = "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0"
DEFAULT_OVERLAP_ARTIFACT = "data/m2_policy/frame_timeline_validation_20260429_140648/vowel_seed0"


def controller() -> AgentiveController:
    return AgentiveController(listen_threshold_ms=100, cooldown_ms=200)


def row(name: str, passed: bool, result: Any, frames: list[M2Frame], note: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "note": note,
        "accepted_initial_output": result.accepted_initial_output,
        "resumed_output": result.resumed_output,
        "states": [decision.state.name for decision in result.decisions],
        "interrupts": [decision.should_interrupt for decision in result.decisions],
        "verdicts": [frame.attribution.verdict.name for frame in frames],
        "speech": [frame.attribution.is_speech for frame in frames],
    }


def run_ladder(self_artifact: str, overlap_artifact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    defer_frames = [
        M2Frame(1.00, attribution(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.15, attribution(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.50, attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0)),
    ]
    defer = run_turn_taking_bridge(defer_frames, "after you", controller=controller())
    rows.append(row(
        "defer_on_agentive_overlap",
        defer.accepted_initial_output is False and defer.resumed_output == "after you" and "LISTEN" in [d.state.name for d in defer.decisions],
        defer,
        defer_frames,
        "Agent should defer when output is requested during external/overlap speech.",
    ))

    self_only_frames = [
        M2Frame(1.00, attribution(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.15, attribution(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.50, attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0)),
    ]
    self_only = run_turn_taking_bridge(self_only_frames, "speaking", controller=controller(), request_at_s=1.0)
    rows.append(row(
        "speak_on_self_only",
        self_only.accepted_initial_output is True and self_only.resumed_output is None and not any(d.should_interrupt for d in self_only.decisions),
        self_only,
        self_only_frames,
        "Self-only activity should not be treated as an external turn.",
    ))

    synthetic_interrupt_frames = [
        M2Frame(1.00, attribution(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.15, attribution(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.50, attribution(SourceVerdict.SILENCE, is_speech=False, energy_db=-70.0)),
    ]
    synthetic_interrupt = run_turn_taking_bridge(
        synthetic_interrupt_frames,
        "pause and resume",
        controller=controller(),
        request_at_s=1.0,
    )
    rows.append(row(
        "interrupt_active_speech_on_overlap",
        synthetic_interrupt.accepted_initial_output is True
        and any(d.should_interrupt for d in synthetic_interrupt.decisions)
        and synthetic_interrupt.resumed_output == "pause and resume",
        synthetic_interrupt,
        synthetic_interrupt_frames,
        "Active speech should interrupt on speech-bearing BOTH and resume after cooldown.",
    ))

    artifact_frames = frames_for_speak_interruption(self_artifact, overlap_artifact)
    artifact_interrupt = run_turn_taking_bridge(
        artifact_frames,
        "artifact resume",
        controller=controller(),
        request_at_s=artifact_frames[0].time_s,
    )
    rows.append(row(
        "artifact_backed_interrupt",
        artifact_interrupt.accepted_initial_output is True
        and any(d.should_interrupt for d in artifact_interrupt.decisions)
        and artifact_interrupt.resumed_output == "artifact resume",
        artifact_interrupt,
        artifact_frames,
        "Saved M2 artifacts should drive the same SPEAK interruption path.",
    ))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run current M3 acceptance ladder")
    parser.add_argument("--self-artifact", default=DEFAULT_SELF_ARTIFACT)
    parser.add_argument("--overlap-artifact", default=DEFAULT_OVERLAP_ARTIFACT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = run_ladder(args.self_artifact, args.overlap_artifact)
    passed = all(r["passed"] for r in rows)

    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status} {r['name']}: states={r['states']} interrupts={r['interrupts']} resumed={r['resumed_output']!r}")

    out = {"passed": passed, "rows": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
