#!/usr/bin/env python3
"""M3 Back-and-Forth Embodied Loop Demo.

Multi-turn conversation: Broca → room capture → M2 attribution → M3 controller → Broca response.
Measures turn-taking metrics: response latency, overlap rate, interruption recovery.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from pfc.agentive_controller import AgentiveController
from pfc.m2_turn_bridge import run_turn_taking_bridge
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts

WORLD_TEXT = "this is a test of world speech conservation"

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


@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    turn_num: int
    agent_text: str
    agent_pcm_samples: int = 0
    world_detected: bool = False
    listen_start_s: float | None = None
    response_start_s: float | None = None
    response_latency_ms: float | None = None
    interrupted: bool = False
    recovered: bool = False
    m2_self_frames: int = 0
    m2_both_frames: int = 0
    m2_world_frames: int = 0
    asr_transcript: str = ""  # ASR result from this turn


@dataclass
class ConversationMetrics:
    """Overall conversation metrics."""
    turns: list[TurnMetrics] = field(default_factory=list)
    total_time_s: float = 0.0
    avg_response_latency_ms: float = 0.0
    overlap_count: int = 0
    interruption_recovery_rate: float = 0.0

    def compute(self):
        latencies = [t.response_latency_ms for t in self.turns if t.response_latency_ms is not None]
        if latencies:
            self.avg_response_latency_ms = sum(latencies) / len(latencies)
        recoveries = [t for t in self.turns if t.interrupted and t.recovered]
        interrupted = [t for t in self.turns if t.interrupted]
        if interrupted:
            self.interruption_recovery_rate = len(recoveries) / len(interrupted)


async def run_turn(
    turn_num: int,
    agent_text: str,
    world_speech_file: str,
    args: argparse.Namespace,
    controller: AgentiveController,
    prev_turn_end_s: float = 0.0,
) -> TurnMetrics:
    """Run a single turn: synthesize → play → capture → attribute → decide."""
    metrics = TurnMetrics(turn_num=turn_num, agent_text=agent_text)
    
    print(f"\n{'='*60}")
    print(f"TURN {turn_num}: Agent says: {agent_text!r}")
    print(f"{'='*60}")
    
    # Synthesize agent response
    print(f"Synthesizing agent text: {agent_text!r}")
    self_pcm = await synthesize(agent_text)
    metrics.agent_pcm_samples = len(self_pcm)
    print(f"Broca PCM samples: {len(self_pcm)}")
    
    # Run M2 trial (plays self_pcm, captures mic)
    print(f"\nRunning M2 attribution trial for turn {turn_num}...")
    result = run_m2_trial(
        self_stim="broca_tts_self",
        self_pcm_override=self_pcm,
        world_speech_file=world_speech_file,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device,
        seed=args.seed + turn_num,  # Different seed per turn
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
        # M4 self-voice detection (optional)
        m4_enable=args.m4_enable,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
        m4_self_threshold=args.m4_self_threshold,
    )
    
    metrics.m2_self_frames = result['self_frames']
    metrics.m2_both_frames = result['both_frames']
    metrics.m2_world_frames = result['world_frames']
    
    print(f"M2 frames self/both/world: {metrics.m2_self_frames}/{metrics.m2_both_frames}/{metrics.m2_world_frames}")
    print(f"M2 self_score={result['self_score']:.3f}")
    
    # Capture ASR transcript for contingent responses
    metrics.asr_transcript = result.get('gated_text', result.get('raw_text', ''))
    
    # Feed M2 frames into M3 controller
    from pfc.m2_artifact_replay import frames_from_frame_records
    frames = frames_from_frame_records(result["frame_records"])
    
    turn_start_time = time.time()
    bridge = run_turn_taking_bridge(
        frames,
        agent_text,
        controller=controller,
        request_at_s=args.request_at_s,
    )
    
    metrics.world_detected = any(f.attribution.verdict.name == "WORLD" for f in frames)
    metrics.interrupted = bridge.accepted_initial_output and any(d.should_interrupt for d in bridge.decisions)
    
    # Calculate response latency (use M2 frame timing)
    world_frames = [f for f in frames if f.attribution.verdict.name == "WORLD"]
    if world_frames and bridge.accepted_initial_output:
        first_world_s = world_frames[0].time_s
        # Response starts after cooldown from first world detection
        metrics.listen_start_s = first_world_s
        metrics.response_start_s = first_world_s + (args.cooldown_ms / 1000.0)
        # Latency = time from world detected to agent's next turn start
        # (simplified: use frame timing)
        metrics.response_latency_ms = (metrics.response_start_s - metrics.listen_start_s) * 1000.0
    
    print(f"M3 accepted_initial_output={bridge.accepted_initial_output}")
    print(f"M3 states: {' -> '.join(d.state.name for d in bridge.decisions)}")
    print(f"M3 interrupts: {[d.should_interrupt for d in bridge.decisions]}")
    
    if args.save_dir:
        turn_dir = os.path.join(args.save_dir, f"turn_{turn_num:02d}")
        save_trial_artifacts(turn_dir, result)
        # Save turn metrics
        with open(os.path.join(turn_dir, "turn_metrics.json"), "w") as f:
            json.dump({
                "turn_num": metrics.turn_num,
                "agent_text": metrics.agent_text,
                "world_detected": metrics.world_detected,
                "response_latency_ms": metrics.response_latency_ms,
                "interrupted": metrics.interrupted,
                "m2_frames": f"{metrics.m2_self_frames}/{metrics.m2_both_frames}/{metrics.m2_world_frames}",
            }, f, indent=2)
        print(f"Saved turn artifacts to: {turn_dir}")
    
    return metrics


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="M3 Back-and-Forth Embodied Loop Demo")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns (default: 3)")
    parser.add_argument("--agent-text", default="I'm Corollary.")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--expected-world-text", default=WORLD_TEXT)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--both-gain", type=float, default=0.5)
    parser.add_argument("--broca-room-local-operating-point", action="store_true", 
                        help="Apply current experimental Broca room-local operating defaults")
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
    parser.add_argument("--m4-self-threshold", type=float, default=0.10, help="M4 self-decision threshold")
    
    args = apply_broca_room_local_operating_defaults(parser.parse_args())
    
    print("\n" + "="*60)
    print("M3 BACK-AND-FORTH EMBODIED LOOP DEMO")
    print("="*60)
    print(f"Turns: {args.turns}")
    print(f"M4 enabled: {args.m4_enable}")
    
    # Initialize M3 controller (persists across turns)
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
    )
    
    conversation_metrics = ConversationMetrics()
    start_time = time.time()
    prev_turn_end_s = 0.0
    prev_asr = ''  # Track previous turn's ASR transcript
    prev_interrupted = False  # Track if previous turn was interrupted
    prev_agent_text = ''  # Track previous turn's agent text
    
    for turn_num in range(1, args.turns + 1):
        # For turn 1, use provided agent_text. For subsequent turns, generate contingent response
        if turn_num == 1:
            agent_text = args.agent_text
        elif prev_interrupted:
            # Recovery: repeat the interrupted message
            agent_text = f"As I said: {prev_agent_text[:40]}"
        else:
            # Contingent response based on ASR transcript
            if prev_asr:
                # Use ASR transcript in response (truncate to avoid timing issues)
                truncated = prev_asr[:50] if len(prev_asr) > 50 else prev_asr
                agent_text = f"You said: {truncated}"
            else:
                agent_text = f"Turn {turn_num}: Heard you!"
        
        metrics = await run_turn(
            turn_num=turn_num,
            agent_text=agent_text,
            world_speech_file=args.world_speech,
            args=args,
            controller=controller,
            prev_turn_end_s=prev_turn_end_s,
        )
        conversation_metrics.turns.append(metrics)
        prev_turn_end_s = time.time()  # Simplified: use wall-clock time
        prev_asr = metrics.asr_transcript  # Update for next turn's contingent response
        
        # Update interruption tracking
        if prev_interrupted and not metrics.interrupted:
            metrics.recovered = True  # Successfully recovered
        prev_interrupted = metrics.interrupted
        prev_agent_text = metrics.agent_text
        
        # Brief pause between turns
        if turn_num < args.turns:
            print(f"\nWaiting 2s before next turn...")
            await asyncio.sleep(2.0)
    
    # Compute overall metrics
    conversation_metrics.total_time_s = time.time() - start_time
    conversation_metrics.compute()
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSATION METRICS SUMMARY")
    print("="*60)
    print(f"Total turns: {len(conversation_metrics.turns)}")
    print(f"Total time: {conversation_metrics.total_time_s:.1f}s")
    print(f"Avg response latency: {conversation_metrics.avg_response_latency_ms:.0f}ms")
    print(f"Interruption recovery rate: {conversation_metrics.interruption_recovery_rate:.1%}")
    
    # Save conversation metrics
    if args.save_dir:
        metrics_path = os.path.join(args.save_dir, "conversation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "total_turns": len(conversation_metrics.turns),
                "total_time_s": conversation_metrics.total_time_s,
                "avg_response_latency_ms": conversation_metrics.avg_response_latency_ms,
                "interruption_recovery_rate": conversation_metrics.interruption_recovery_rate,
                "turns": [
                    {
                        "turn": t.turn_num,
                        "agent_text": t.agent_text,
                        "world_detected": t.world_detected,
                        "response_latency_ms": t.response_latency_ms,
                        "interrupted": t.interrupted,
                        "recovered": t.recovered,
                        "m2_frames": f"{t.m2_self_frames}/{t.m2_both_frames}/{t.m2_world_frames}",
                    }
                    for t in conversation_metrics.turns
                ]
            }, f, indent=2)
        print(f"\nSaved conversation metrics to: {metrics_path}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
