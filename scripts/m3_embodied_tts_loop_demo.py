#!/usr/bin/env python3
"""M3 embodied TTS loop demo: Broca audio -> M2 capture -> M3 controller.

This is the first embodied bridge beyond replay/live-ish frame injection. The
agent synthesizes the exact requested utterance with Broca, uses that PCM as the
M2 self reference while playing it through the output device, captures mic/world
audio, then feeds the resulting M2 frame timeline into the M3 SPEAK lifecycle.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_from_frame_records
from pfc.m2_turn_bridge import run_turn_taking_bridge
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

WORLD_TEXT = "this is a test of world speech conservation"
DEFAULT_AGENT_TEXT = "I was speaking, then paused."

ROOM_LOCAL_OPERATING_DEFAULTS = {
    "similarity_threshold": 0.05,
    "onset_latency_ms": 300.0,
    "onset_method": "score",
    "causal_tolerance_ms": 175.0,
    "self_match_mode": "room_local_oracle",
    "self_match_search_ms": 256.4,
    "room_filter_taps": 64,
    "room_filter_ridge": 0.000112,
}

LEGACY_DEFAULTS = {
    "similarity_threshold": 0.20,
    "onset_latency_ms": 400.0,
    "onset_method": "energy",
    "causal_tolerance_ms": 75.0,
    "self_match_mode": "global_xcorr",
    "self_match_search_ms": 250.0,
    "room_filter_taps": 64,
    "room_filter_ridge": 1e-4,
}


def apply_broca_room_local_operating_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = ROOM_LOCAL_OPERATING_DEFAULTS if args.broca_room_local_operating_point else LEGACY_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Embodied Broca->M2->M3 TTS loop demo")
    parser.add_argument("--agent-text", default=DEFAULT_AGENT_TEXT)
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--expected-world-text", default=WORLD_TEXT)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--both-gain", type=float, default=0.5)
    parser.add_argument("--broca-room-local-operating-point", action="store_true", help="Apply current experimental Broca room-local operating defaults (threshold 0.10, score onset, 300ms latency, 175ms tolerance, room_local_oracle scorer)")
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--onset-latency-ms", type=float, default=None)
    parser.add_argument("--onset-method", choices=["xcorr", "energy", "spectral_flux", "score"], default=None)
    parser.add_argument("--causal-tolerance-ms", type=float, default=None)
    parser.add_argument("--self-match-mode", choices=["global_xcorr", "local_xcorr", "room_local_oracle"], default=None)
    parser.add_argument("--self-match-search-ms", type=float, default=None)
    parser.add_argument("--room-filter-taps", type=int, default=None)
    parser.add_argument("--room-filter-ridge", type=float, default=None)
    parser.add_argument("--request-at-s", type=float, default=0.0)
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    parser.add_argument("--save-dir", default=None)
    # M4 self-voice detection
    parser.add_argument("--m4-enable", action="store_true", help="Enable M4 speaker embedding detector")
    parser.add_argument("--m4-min-buffer-ms", type=float, default=60.0, help="M4 minimum buffer ms")
    parser.add_argument("--m4-self-threshold", type=float, default=0.50, help="M4 self-decision threshold")
    parser.add_argument("--cerebellum-model", default=None, help="Path to trained BabblingCerebellum JSON for room-colored self registration")

    args = apply_broca_room_local_operating_defaults(parser.parse_args())

    print("\n--- M3 Embodied TTS Loop Demo ---")
    print(f"Synthesizing agent text: {args.agent_text!r}")
    self_pcm = await synthesize(args.agent_text)
    print(f"Broca PCM samples: {len(self_pcm)}")

    result = run_m2_trial(
        self_stim="broca_tts_self",
        self_pcm_override=self_pcm,
        world_speech_file=args.world_speech,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device,
        seed=args.seed,
        expected_text=args.expected_world_text,
        both_policy="manual",
        both_gain=args.both_gain,
        self_duration_ms=int(len(self_pcm) * 1000 / 16000),
        self_amplitude=1.0,
        speech_override="vad",
        similarity_threshold=args.similarity_threshold,
        onset_latency_ms=args.onset_latency_ms,
        onset_method=args.onset_method,
        causal_tolerance_ms=args.causal_tolerance_ms,
        self_match_mode=args.self_match_mode,
        self_match_search_ms=args.self_match_search_ms,
        room_filter_taps=args.room_filter_taps,
        room_filter_ridge=args.room_filter_ridge,
        # M4 self-voice detection
        m4_enable=args.m4_enable,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
        m4_self_threshold=args.m4_self_threshold,
        cerebellum_model=args.cerebellum_model,
    )
    if args.save_dir:
        save_trial_artifacts(args.save_dir, result)
        print(f"Saved artifacts to: {args.save_dir}")

    frames = frames_from_frame_records(result["frame_records"])
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
    )
    bridge = run_turn_taking_bridge(
        frames,
        args.agent_text,
        controller=controller,
        request_at_s=args.request_at_s,
    )

    print(f"M2 frames self/both/world: {result['self_frames']}/{result['both_frames']}/{result['world_frames']}")
    print(f"M2 self_match_mode={result['self_match_mode']} search_ms={result['self_match_search_ms']}")
    print(f"M2 self_score={result['self_score']:.3f} dry_global={result['dry_global_self_score']:.3f} dry_local={result['dry_local_self_score']:.3f} room_local={result['room_local_self_score']} onset_error_ms={result['self_onset_error_ms']:.1f}")
    if result["metrics"]:
        print(f"M2 world WER raw/gated: {result['metrics']['raw_wer']:.3f}/{result['metrics']['gated_wer']:.3f}")
    print(f"M3 request_at_s={args.request_at_s}")
    print(f"M3 accepted_initial_output={bridge.accepted_initial_output}")
    print(f"M3 resumed_output={bridge.resumed_output!r}")
    print("M3 states:", " -> ".join(decision.state.name for decision in bridge.decisions))
    print("M3 interrupts:", [decision.should_interrupt for decision in bridge.decisions])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
