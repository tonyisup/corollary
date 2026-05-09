#!/usr/bin/env python3
"""M3 Streaming Demo — production streaming callback integration.

Runs the real-time streaming runtime on pre-recorded audio (no hardware needed)
or on live audio hardware.

Usage:
  # Offline replay (deterministic, no hardware)
  .venv/bin/python scripts/m3_streaming_demo.py \\
    --mode offline \\
    --input-audio data/m3_interrupt_v3-mic_adjusted/human_early/captured.npy \\
    --agent-pcm data/m3_interrupt_v3-mic_adjusted/human_early/agent_pcm.npy \\
    --output /tmp/streaming_demo_log.jsonl

  # Live hardware mode
  .venv/bin/python scripts/m3_streaming_demo.py \\
    --mode live \\
    --input-device 3 \\
    --output-device 5 \\
    --duration 10.0
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig


def load_audio_from_scenario(scenario_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load captured audio and agent PCM for a known scenario."""
    base = Path(f"data/m3_interrupt_v3-mic_adjusted/{scenario_name}")
    captured = np.load(base / "captured.npy").astype(np.float32)
    agent_pcm = np.load(base / "agent_pcm.npy").astype(np.float32)
    return captured, agent_pcm


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 Streaming Demo")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Run mode: offline (pre-recorded audio) or live (hardware)",
    )
    parser.add_argument(
        "--scenario",
        choices=["human_early", "human_late", "human_before", "human_after", "overlap_self_first"],
        default="human_early",
        help="Scenario to load in offline mode",
    )
    parser.add_argument("--input-audio", help="Path to input audio .npy (overrides --scenario)")
    parser.add_argument("--agent-pcm", help="Path to agent PCM .npy (overrides --scenario)")
    parser.add_argument("--agent-text", default="I was speaking, then paused.")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in live mode (seconds)")
    parser.add_argument("--tts-chunk-ms", type=float, default=200.0)
    parser.add_argument("--gated-playback", action="store_true", default=True, help="Enable real-time TTS gating")
    parser.add_argument("--no-gated-playback", action="store_false", dest="gated_playback", help="Disable gating (always play)")
    parser.add_argument("--m4-enable", action="store_true", default=False, help="Enable M4 speaker-embedding detector")
    parser.add_argument("--room-filter", default=None, help="Path to room filter .npy")
    parser.add_argument("--output", "-o", default="/tmp/streaming_demo_log.jsonl", help="Output JSONL log path")
    parser.add_argument("--summary", default="/tmp/streaming_demo_summary.json", help="Output summary JSON path")

    args = parser.parse_args()

    # Load TTS audio
    if args.mode == "offline":
        if args.input_audio and args.agent_pcm:
            input_audio = np.load(args.input_audio).astype(np.float32)
            agent_pcm = np.load(args.agent_pcm).astype(np.float32)
        else:
            input_audio, agent_pcm = load_audio_from_scenario(args.scenario)

        print(f"Loaded audio: {len(input_audio)} samples ({len(input_audio)/16000:.2f}s)")
        print(f"Agent PCM:  {len(agent_pcm)} samples ({len(agent_pcm)/16000:.2f}s)")
    else:
        agent_pcm = None
        input_audio = None

    config = StreamingConfig(
        tts_pcm=agent_pcm,
        tts_text=args.agent_text,
        input_device=args.input_device,
        output_device=args.output_device,
        tts_chunk_ms=args.tts_chunk_ms,
        m4_enable=args.m4_enable,
        gated_playback=args.gated_playback,
        room_filter_path=args.room_filter,
        save_path=Path(args.output),
    )

    runtime = StreamingRuntime(config)

    if args.mode == "offline":
        logs = runtime.run_offline(input_audio)
    else:
        logs = runtime.run(duration_s=args.duration)

    # Summary
    summary = runtime.get_summary()
    print("\n--- Streaming Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
