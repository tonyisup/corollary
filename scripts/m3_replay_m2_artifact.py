#!/usr/bin/env python3
"""Replay a saved M2 summary artifact through the M3 controller."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_from_artifact, load_m2_summary
from pfc.m2_turn_bridge import run_turn_taking_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay M2 artifact into M3 turn-taking")
    parser.add_argument("summary", help="Path to M2 summary.json or artifact directory")
    parser.add_argument("--text", default="I can respond now.")
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    args = parser.parse_args()

    summary = load_m2_summary(args.summary)
    frames = frames_from_artifact(args.summary)
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
    )
    result = run_turn_taking_bridge(frames, args.text, controller=controller)

    print(f"artifact={args.summary}")
    print(
        "summary="
        f"self/both/world={summary.get('self_frames')}/{summary.get('both_frames')}/{summary.get('world_frames')} "
        f"speech_ratio={summary.get('speech_ratio')}"
    )
    for frame, decision in zip(frames, result.decisions):
        print(
            f"t={frame.time_s:.2f}s verdict={frame.attribution.verdict.name} "
            f"speech={frame.attribution.is_speech} state={decision.state.name} "
            f"can_speak={decision.can_speak}"
        )
    print(f"accepted_initial_output={result.accepted_initial_output}")
    print(f"resumed_output={result.resumed_output!r}")


if __name__ == "__main__":
    main()
