#!/usr/bin/env python3
"""M3 proto-turn-taking demo using synthetic M2 attribution frames."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.attributor import AttributionResult, SourceVerdict
from pfc.agentive_controller import AgentiveController
from pfc.m2_turn_bridge import M2Frame, run_turn_taking_bridge


def attribution(verdict: SourceVerdict, is_speech: bool) -> AttributionResult:
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=0.2,
        is_in_motor_window=verdict in {SourceVerdict.SELF, SourceVerdict.BOTH},
        causal_score=0.0,
        energy_db=-30.0 if verdict != SourceVerdict.SILENCE else -70.0,
        is_speech=is_speech,
    )


def print_result(title: str, frames: list[M2Frame], result) -> None:
    print(f"\n--- {title} ---")
    for frame, decision in zip(frames, result.decisions):
        print(
            f"t={frame.time_s:.2f}s verdict={frame.attribution.verdict.name} "
            f"speech={frame.attribution.is_speech} state={decision.state.name} "
            f"can_speak={decision.can_speak} interrupt={decision.should_interrupt}"
        )
    print(f"accepted_initial_output={result.accepted_initial_output}")
    print(f"resumed_output={result.resumed_output!r}")


def main() -> None:
    defer_frames = [
        M2Frame(1.00, attribution(SourceVerdict.BOTH, is_speech=True)),
        M2Frame(1.15, attribution(SourceVerdict.BOTH, is_speech=True)),
        M2Frame(1.50, attribution(SourceVerdict.SILENCE, is_speech=False)),
    ]
    defer_result = run_turn_taking_bridge(
        defer_frames,
        "I can respond now.",
        controller=AgentiveController(listen_threshold_ms=100, cooldown_ms=200),
    )
    print_result("M3 defers before speaking", defer_frames, defer_result)

    interrupt_frames = [
        M2Frame(2.00, attribution(SourceVerdict.SELF, is_speech=False)),
        M2Frame(2.15, attribution(SourceVerdict.BOTH, is_speech=True)),
        M2Frame(2.50, attribution(SourceVerdict.SILENCE, is_speech=False)),
    ]
    interrupt_result = run_turn_taking_bridge(
        interrupt_frames,
        "I was speaking, then paused.",
        controller=AgentiveController(listen_threshold_ms=100, cooldown_ms=200),
        request_at_s=2.0,
    )
    print_result("M3 interrupts active speech", interrupt_frames, interrupt_result)


if __name__ == "__main__":
    main()
