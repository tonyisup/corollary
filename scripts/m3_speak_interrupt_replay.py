#!/usr/bin/env python3
"""Replay saved M2 artifacts through the M3 SPEAK-interruption path."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_for_speak_interruption
from pfc.m2_turn_bridge import run_turn_taking_bridge

DEFAULT_SELF_ARTIFACT = "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0"
DEFAULT_OVERLAP_ARTIFACT = "data/m2_policy/frame_timeline_validation_20260429_140648/vowel_seed0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay artifact-backed M3 SPEAK interruption")
    parser.add_argument("--self-artifact", default=DEFAULT_SELF_ARTIFACT)
    parser.add_argument("--overlap-artifact", default=DEFAULT_OVERLAP_ARTIFACT)
    parser.add_argument("--text", default="I was speaking, then paused.")
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    args = parser.parse_args()

    frames = frames_for_speak_interruption(args.self_artifact, args.overlap_artifact)
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
    )
    result = run_turn_taking_bridge(
        frames,
        args.text,
        controller=controller,
        request_at_s=frames[0].time_s,
    )

    print(f"self_artifact={args.self_artifact}")
    print(f"overlap_artifact={args.overlap_artifact}")
    for frame, decision in zip(frames, result.decisions):
        print(
            f"t={frame.time_s:.2f}s verdict={frame.attribution.verdict.name} "
            f"speech={frame.attribution.is_speech} state={decision.state.name} "
            f"can_speak={decision.can_speak} interrupt={decision.should_interrupt}"
        )
    print(f"accepted_initial_output={result.accepted_initial_output}")
    print(f"resumed_output={result.resumed_output!r}")


if __name__ == "__main__":
    main()
