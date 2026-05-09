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
import asyncio
import json
import sys
import os
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig
from broca.tts import synthesize


SR = 16000

def load_audio_from_scenario(scenario_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load captured audio and agent PCM for a known scenario."""
    base = Path(f"data/m3_interrupt_v3-mic_adjusted/{scenario_name}")
    captured = np.load(base / "captured.npy").astype(np.float32)
    agent_pcm = np.load(base / "agent_pcm.npy").astype(np.float32)
    return captured, agent_pcm


def synthesize_sync(text: str) -> np.ndarray:
    """Pre-render Broca TTS before opening the realtime audio callback."""
    return asyncio.run(synthesize(text)).astype(np.float32)


def play_array_blocking(data: np.ndarray, device: int, samplerate: int = SR) -> None:
    """Play mono float32 audio on a specific output device."""
    import sounddevice as sd

    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(device=device, channels=data.shape[1], samplerate=samplerate, dtype="float32") as stream:
        pos = 0
        block = 1024
        while pos < len(data):
            end = min(pos + block, len(data))
            stream.write(data[pos:end])
            pos = end


def start_delayed_playback(data: np.ndarray, device: int, delay_s: float) -> threading.Thread:
    """Start a delayed playback thread for deterministic live overlap validation."""
    def _worker() -> None:
        time.sleep(max(0.0, delay_s))
        play_array_blocking(data, device=device, samplerate=SR)

    thread = threading.Thread(target=_worker, name="m3-human-stimulus", daemon=True)
    thread.start()
    return thread


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
    parser.add_argument("--repair-pcm", help="Path to pre-rendered repair PCM .npy")
    parser.add_argument("--agent-text", default="I was speaking, then paused.")
    parser.add_argument("--repair-text", default="Sorry, go ahead.")
    parser.add_argument("--human-text", default="Excuse me, I need to say something here.")
    parser.add_argument("--human-output-device", type=int, default=4)
    parser.add_argument("--human-delay-s", type=float, default=0.45)
    parser.add_argument("--play-human-stimulus", action="store_true", help="In live mode, play a delayed human stimulus on --human-output-device")
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

    # Load or pre-render TTS audio. Live mode must pre-render before opening the
    # audio callback; never synthesize inside the realtime path.
    repair_pcm = None
    human_pcm = None
    if args.repair_pcm:
        repair_pcm = np.load(args.repair_pcm).astype(np.float32)
        print(f"Repair PCM: {len(repair_pcm)} samples ({len(repair_pcm)/SR:.2f}s) from {args.repair_pcm}")
    elif args.repair_text:
        print(f"Synthesizing repair cue: {args.repair_text!r}")
        repair_pcm = synthesize_sync(args.repair_text)
        print(f"Repair PCM: {len(repair_pcm)} samples ({len(repair_pcm)/SR:.2f}s)")

    if args.mode == "offline":
        if args.input_audio and args.agent_pcm:
            input_audio = np.load(args.input_audio).astype(np.float32)
            agent_pcm = np.load(args.agent_pcm).astype(np.float32)
        else:
            input_audio, agent_pcm = load_audio_from_scenario(args.scenario)

        print(f"Loaded audio: {len(input_audio)} samples ({len(input_audio)/SR:.2f}s)")
        print(f"Agent PCM:  {len(agent_pcm)} samples ({len(agent_pcm)/SR:.2f}s)")
    else:
        input_audio = None
        if args.agent_pcm:
            agent_pcm = np.load(args.agent_pcm).astype(np.float32)
            print(f"Agent PCM:  {len(agent_pcm)} samples ({len(agent_pcm)/SR:.2f}s) from {args.agent_pcm}")
        else:
            print(f"Synthesizing agent utterance: {args.agent_text!r}")
            agent_pcm = synthesize_sync(args.agent_text)
            print(f"Agent PCM:  {len(agent_pcm)} samples ({len(agent_pcm)/SR:.2f}s)")
        if args.play_human_stimulus:
            print(f"Synthesizing human stimulus: {args.human_text!r}")
            human_pcm = synthesize_sync(args.human_text)
            print(f"Human PCM:  {len(human_pcm)} samples ({len(human_pcm)/SR:.2f}s)")

    config = StreamingConfig(
        tts_pcm=agent_pcm,
        repair_pcm=repair_pcm,
        tts_text=args.agent_text,
        collision_repair_text=args.repair_text,
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
        human_thread = None
        if human_pcm is not None:
            print(
                f"Starting delayed human stimulus on device {args.human_output_device} "
                f"after {args.human_delay_s:.2f}s"
            )
            human_thread = start_delayed_playback(
                human_pcm,
                device=args.human_output_device,
                delay_s=args.human_delay_s,
            )
        logs = runtime.run(duration_s=args.duration)
        if human_thread is not None:
            human_thread.join(timeout=1.0)

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
