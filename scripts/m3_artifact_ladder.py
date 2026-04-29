#!/usr/bin/env python3
"""Run a graded M3 replay ladder over saved M2 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_from_artifact, load_m2_summary
from pfc.m2_turn_bridge import run_turn_taking_bridge


DEFAULT_ARTIFACTS = [
    ("clean_vowel_overlap", "data/m2_policy/family_policy_validation_20260429_130530/vowel_ah_var_nearfield_seed0"),
    ("babble_overlap_stress", "data/m2_policy/family_policy_validation_20260429_130530/babble_ba_repeat_short_tight_var_seed0"),
    ("world_only_control", "data/m2_baseline/control_world-only_seed0_in3_out5_noise4_20260429_091828"),
    ("self_only_control", "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0"),
]


def replay(name: str, artifact: str, text: str) -> dict[str, Any]:
    summary = load_m2_summary(artifact)
    frames = frames_from_artifact(artifact)
    controller = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    result = run_turn_taking_bridge(frames, text, controller=controller)
    return {
        "name": name,
        "artifact": artifact,
        "self_frames": summary.get("self_frames"),
        "both_frames": summary.get("both_frames"),
        "world_frames": summary.get("world_frames"),
        "speech_ratio": summary.get("speech_ratio"),
        "accepted_initial_output": result.accepted_initial_output,
        "deferred_output": result.deferred_output,
        "resumed_output": result.resumed_output,
        "states": [decision.state.name for decision in result.decisions],
        "can_speak": [decision.can_speak for decision in result.decisions],
        "verdicts": [frame.attribution.verdict.name for frame in frames],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 replay ladder over M2 artifacts")
    parser.add_argument("--text", default="I can respond now.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    for name, artifact in DEFAULT_ARTIFACTS:
        if not Path(artifact).exists():
            print(f"SKIP {name}: missing {artifact}")
            continue
        row = replay(name, artifact, args.text)
        rows.append(row)
        print(
            f"{name}: verdicts={row['verdicts']} states={row['states']} "
            f"accepted={row['accepted_initial_output']} resumed={row['resumed_output']!r}"
        )

    out = {"rows": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
