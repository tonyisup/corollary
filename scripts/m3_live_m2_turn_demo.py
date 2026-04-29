#!/usr/bin/env python3
"""Run one M2 trial and immediately feed its frame timeline into M3."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_from_frame_records
from pfc.m2_turn_bridge import run_turn_taking_bridge
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

EXPECTED_TEXT = "this is a test of world speech conservation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-ish M2 to M3 turn-taking demo")
    parser.add_argument("--self-stim", default="vowel_ah_var_nearfield")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-duration-ms", type=int, default=900)
    parser.add_argument("--self-amplitude", type=float, default=1.0)
    parser.add_argument("--expected-text", default=EXPECTED_TEXT)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--requested-output", default="I can respond now.")
    args = parser.parse_args()

    result = run_m2_trial(
        self_stim=args.self_stim,
        world_speech_file=args.world_speech,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device,
        seed=args.seed,
        expected_text=args.expected_text,
        both_policy="family",
        self_duration_ms=args.self_duration_ms,
        self_amplitude=args.self_amplitude,
        speech_override="vad",
        similarity_threshold=0.20,
        onset_latency_ms=400,
        onset_method="energy",
        causal_tolerance_ms=75,
    )
    if args.save_dir:
        save_trial_artifacts(args.save_dir, result)
        print(f"Saved artifacts to: {args.save_dir}")

    frames = frames_from_frame_records(result["frame_records"])
    controller = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    bridge = run_turn_taking_bridge(frames, args.requested_output, controller=controller)

    print("\n--- M3 Live M2 Turn Demo ---")
    print(f"M2 policy: {result['both_policy_reason']} mode={result['both_mode']} gain={result['both_gain']}")
    print(f"M2 frames self/both/world: {result['self_frames']}/{result['both_frames']}/{result['world_frames']}")
    if result["metrics"]:
        print(f"M2 WER raw/gated: {result['metrics']['raw_wer']:.3f}/{result['metrics']['gated_wer']:.3f}")
    print(f"M3 accepted_initial_output={bridge.accepted_initial_output}")
    print(f"M3 resumed_output={bridge.resumed_output!r}")
    print("M3 states:", " -> ".join(decision.state.name for decision in bridge.decisions))


if __name__ == "__main__":
    main()
