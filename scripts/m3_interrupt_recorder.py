#!/usr/bin/env python3
"""M3 Interruption Scenario Recorder.

Records controlled human-speech-overlaps-Corollary scenarios for turn-taking
evaluation. Each scenario plays a TTS utterance while a human speaks at a
specified time. Saves raw mic audio, playback timing, and ground truth labels.

Usage:
    .venv/bin/python scripts/m3_interrupt_recorder.py --scenarios all --output-dir data/m3_interrupt_scenarios
    .venv/bin/python scripts/m3_interrupt_recorder.py --scenarios 1,3,5 --output-dir data/m3_interrupt_scenarios
    .venv/bin/python scripts/m3_interrupt_recorder.py --list-devices

Scenario definitions:
    1. human_before:     Human speaks before Corollary starts
    2. human_after:      Human speaks after Corollary finishes
    3. human_early:      Human speaks during first half of Corollary
    4. human_late:       Human speaks during second half of Corollary
    5. brief_noise:      Brief human noise during Corollary
    6. human_double:     Human speaks, stops, speaks again
    7. self_only:        Corollary speaks alone (control)
    8. world_only:       Human speaks alone (control)
    9. overlap_human_first: Human starts, Corollary overlaps
    10. overlap_self_first: Corollary starts, human overlaps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. pip install sounddevice")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from cochlea.capture import SR


# ── Scenario definitions ──

SCENARIO_DEFS = {
    "human_before": {
        "description": "Human speaks before Corollary starts",
        "agent_text": "I'm Corollary. I was about to say something important.",
        "human_offset_s": -1.5,  # Human speaks 1.5s before agent starts
        "human_duration_s": 1.0,
        "expected_behavior": "defer",
    },
    "human_after": {
        "description": "Human speaks after Corollary finishes",
        "agent_text": "I'm Corollary. I have finished my thought.",
        "human_offset_s": 2.0,  # Human speaks 2s after agent starts
        "human_duration_s": 1.0,
        "expected_behavior": "respond",
    },
    "human_early": {
        "description": "Human speaks during first half of Corollary",
        "agent_text": "I'm Corollary. I was about to say something important and now I'm being interrupted early.",
        "human_offset_s": 0.5,
        "human_duration_s": 1.0,
        "expected_behavior": "interrupt_defer_resume",
    },
    "human_late": {
        "description": "Human speaks during second half of Corollary",
        "agent_text": "I'm Corollary. I was about to say something important and now I'm being interrupted late in my utterance.",
        "human_offset_s": 1.5,
        "human_duration_s": 1.0,
        "expected_behavior": "interrupt_defer_resume",
    },
    "brief_noise": {
        "description": "Brief human noise during Corollary",
        "agent_text": "I'm Corollary. I am speaking continuously without interruption.",
        "human_offset_s": 1.0,
        "human_duration_s": 0.1,  # 100ms — single frame
        "expected_behavior": "no_interrupt",
    },
    "human_double": {
        "description": "Human speaks, stops, speaks again",
        "agent_text": "I'm Corollary. I am trying to complete my utterance but keep getting interrupted.",
        "human_offset_s": 0.5,
        "human_duration_s": 0.5,
        "human_second_offset_s": 2.0,
        "human_second_duration_s": 0.5,
        "expected_behavior": "interrupt_defer_resume_interrupt",
    },
    "self_only": {
        "description": "Corollary speaks alone (control)",
        "agent_text": "I'm Corollary. I am speaking alone with no human interruption.",
        "human_offset_s": None,  # No human speech
        "human_duration_s": 0,
        "expected_behavior": "complete",
    },
    "world_only": {
        "description": "Human speaks alone (control)",
        "agent_text": "",  # No agent speech
        "human_offset_s": 0.5,
        "human_duration_s": 2.0,
        "expected_behavior": "defer",
    },
    "overlap_human_first": {
        "description": "Human starts speaking, Corollary overlaps",
        "agent_text": "I'm Corollary. I'm joining the conversation.",
        "human_offset_s": 0.0,  # Human starts at t=0
        "human_duration_s": 2.0,
        "agent_offset_s": 1.0,  # Agent starts 1s after human
        "expected_behavior": "defer_then_respond",
    },
    "overlap_self_first": {
        "description": "Corollary starts, human overlaps",
        "agent_text": "I'm Corollary. I started speaking and now someone is interrupting me.",
        "human_offset_s": 1.0,  # Human starts 1s after agent
        "human_duration_s": 1.5,
        "expected_behavior": "interrupt_defer_resume",
    },
}


def play_array_blocking(pcm: np.ndarray, device: int, samplerate: int) -> None:
    """Play audio array blocking."""
    sd.play(pcm, samplerate=samplerate, device=device)
    sd.wait()


async def synthesize_text(text: str) -> np.ndarray:
    """Synthesize text to PCM."""
    if not text:
        return np.zeros(0, dtype=np.float32)
    return await synthesize(text)


def record_scenario(
    scenario_name: str,
    scenario_def: dict,
    input_device: int,
    output_device: int,
    output_dir: Path,
    agent_pcm: np.ndarray | None = None,
) -> dict:
    """Record a single interruption scenario."""

    desc = scenario_def["description"]
    agent_text = scenario_def["agent_text"]
    human_offset_s = scenario_def["human_offset_s"]
    human_duration_s = scenario_def["human_duration_s"]
    expected = scenario_def["expected_behavior"]

    # Synthesize agent audio if not provided
    if agent_pcm is None and agent_text:
        print(f"  Synthesizing: {agent_text[:60]}...")
        agent_pcm = asyncio.run(synthesize_text(agent_text))
    elif not agent_text:
        agent_pcm = np.zeros(0, dtype=np.float32)

    agent_duration_s = len(agent_pcm) / SR if len(agent_pcm) > 0 else 0

    # Timing
    lead_in_s = 0.5
    tail_s = 1.0
    agent_start_s = lead_in_s

    # Handle agent_offset_s for overlap scenarios
    if "agent_offset_s" in scenario_def:
        agent_start_s = lead_in_s + scenario_def["agent_offset_s"]

    # Human timing (relative to recording start)
    human_start_s = lead_in_s + human_offset_s if human_offset_s is not None else None

    # Second human burst for double scenario
    human_second = None
    if "human_second_offset_s" in scenario_def:
        human_second = {
            "start_s": lead_in_s + scenario_def["human_second_offset_s"],
            "duration_s": scenario_def["human_second_duration_s"],
        }

    # Total duration
    total_duration = lead_in_s + max(
        agent_start_s + agent_duration_s if len(agent_pcm) > 0 else 0,
        (human_start_s + human_duration_s) if human_start_s is not None else 0,
        (human_second["start_s"] + human_second["duration_s"]) if human_second else 0,
    ) + tail_s

    # Add padding
    total_duration += 0.5

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)

    # Build output mix (agent audio at correct time)
    total_samples = int(total_duration * output_sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    if len(agent_pcm) > 0:
        # Resample agent PCM to output SR if needed
        agent_resampled = agent_pcm
        if abs(output_sr - SR) > 1:
            import librosa
            agent_resampled = librosa.resample(agent_pcm, orig_sr=SR, target_sr=output_sr)

        agent_start_idx = int(agent_start_s * output_sr)
        agent_end_idx = min(total_samples, agent_start_idx + len(agent_resampled))
        output_audio[agent_start_idx:agent_end_idx] = agent_resampled[:agent_end_idx - agent_start_idx]

    # Build ground truth labels (at input SR)
    total_input_samples = int(total_duration * input_sr)
    gt_human_speaking = np.zeros(total_input_samples, dtype=bool)
    gt_agent_playing = np.zeros(total_input_samples, dtype=bool)
    gt_agent_should_speak = np.zeros(total_input_samples, dtype=bool)

    # Agent playing region
    if len(agent_pcm) > 0:
        agent_play_start_idx = int(agent_start_s * input_sr)
        agent_play_end_idx = int((agent_start_s + agent_duration_s) * input_sr)
        gt_agent_playing[agent_play_start_idx:agent_play_end_idx] = True

    # Human speaking region(s)
    if human_start_s is not None:
        human_start_idx = int(human_start_s * input_sr)
        human_end_idx = int((human_start_s + human_duration_s) * input_sr)
        gt_human_speaking[human_start_idx:human_end_idx] = True

    if human_second:
        hs_start_idx = int(human_second["start_s"] * input_sr)
        hs_end_idx = int((human_second["start_s"] + human_second["duration_s"]) * input_sr)
        gt_human_speaking[hs_start_idx:hs_end_idx] = True

    # Ground truth: when should agent be speaking?
    # Agent should speak when: agent is playing AND human is not speaking
    gt_agent_should_speak = gt_agent_playing & ~gt_human_speaking

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Description: {desc}")
    print(f"Expected: {expected}")
    print(f"Agent audio: {agent_duration_s:.2f}s at t={agent_start_s:.2f}s")
    if human_start_s is not None:
        print(f"Human speech: {human_duration_s:.2f}s at t={human_start_s:.2f}s")
    if human_second:
        print(f"Human 2nd: {human_second['duration_s']:.2f}s at t={human_second['start_s']:.2f}s")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"\n*** SPEAK WHEN INDICATED ***")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # Start recording and playback
    captured = np.zeros(total_input_samples, dtype=np.float32)
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()

    # Start playback in a thread
    playback_thread = threading.Thread(target=play_array_blocking, args=(output_audio, output_device, output_sr))
    playback_thread.start()

    # Record
    pos = 0
    while pos < total_input_samples:
        data, _ = stream.read(min(1024, total_input_samples - pos))
        n = min(len(data), total_input_samples - pos)
        captured[pos:pos + n] = data[:n, 0]
        pos += n

    playback_thread.join()
    stream.stop()
    stream.close()

    # Save artifacts
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Save audio
    np.save(str(scenario_dir / "captured.npy"), captured)
    if len(agent_pcm) > 0:
        np.save(str(scenario_dir / "agent_pcm.npy"), agent_pcm)

    # Save ground truth
    np.save(str(scenario_dir / "gt_human_speaking.npy"), gt_human_speaking)
    np.save(str(scenario_dir / "gt_agent_playing.npy"), gt_agent_playing)
    np.save(str(scenario_dir / "gt_agent_should_speak.npy"), gt_agent_should_speak)

    # Save metadata
    metadata = {
        "scenario_name": scenario_name,
        "description": desc,
        "expected_behavior": expected,
        "agent_text": agent_text,
        "agent_start_s": agent_start_s,
        "agent_duration_s": agent_duration_s,
        "agent_samples": len(agent_pcm),
        "human_start_s": human_start_s,
        "human_duration_s": human_duration_s,
        "human_second": human_second,
        "total_duration_s": total_duration,
        "input_sr": input_sr,
        "output_sr": output_sr,
        "lead_in_s": lead_in_s,
        "tail_s": tail_s,
        "recorded_at": datetime.now().isoformat(),
    }
    with open(scenario_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Summary stats
    captured_rms = float(np.sqrt(np.mean(captured**2)))
    human_region_rms = float(np.sqrt(np.mean(captured[gt_human_speaking]**2))) if np.any(gt_human_speaking) else 0
    agent_region_rms = float(np.sqrt(np.mean(captured[gt_agent_playing]**2))) if np.any(gt_agent_playing) else 0

    print(f"  Captured RMS: {captured_rms:.4f}")
    print(f"  Human region RMS: {human_region_rms:.4f}")
    print(f"  Agent region RMS: {agent_region_rms:.4f}")
    print(f"  Saved to: {scenario_dir}")

    return {
        "scenario": scenario_name,
        "dir": str(scenario_dir),
        "captured_rms": captured_rms,
        "human_region_rms": human_region_rms,
        "agent_region_rms": agent_region_rms,
    }


def main():
    parser = argparse.ArgumentParser(description="M3 Interruption Scenario Recorder")
    parser.add_argument("--scenarios", default="all",
                        help="Comma-separated scenario names, or 'all'")
    parser.add_argument("--output-dir", default="data/m3_interrupt_scenarios")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:\n")
        devices = sd.query_devices()
        print(f"{'Index':<8} {'Name':<50} {'In':<4} {'Out':<4} {'SR'}")
        print("-" * 80)
        for i, dev in enumerate(devices):
            print(f"{i:<8} {dev['name']:<50} {dev['max_input_channels']:<4} "
                  f"{dev['max_output_channels']:<4} {dev['default_samplerate']}")
        return

    if args.list_scenarios:
        print("\nAvailable scenarios:\n")
        for name, defn in SCENARIO_DEFS.items():
            print(f"  {name:<25} {defn['description']}")
            print(f"  {'':25} Expected: {defn['expected_behavior']}")
        return

    # Select scenarios
    if args.scenarios == "all":
        selected = list(SCENARIO_DEFS.keys())
    else:
        selected = [s.strip() for s in args.scenarios.split(",")]
        for s in selected:
            if s not in SCENARIO_DEFS:
                print(f"ERROR: Unknown scenario '{s}'. Use --list-scenarios to see available.")
                sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M3 INTERRUPTION SCENARIO RECORDER")
    print("=" * 60)
    print(f"Scenarios: {len(selected)}")
    print(f"Output: {output_dir}")
    print(f"Input device: {args.input_device}")
    print(f"Output device: {args.output_device}")

    # Pre-synthesize all agent audio
    print("\nPre-synthesizing agent audio...")
    agent_pcms = {}
    for name in selected:
        text = SCENARIO_DEFS[name]["agent_text"]
        if text and text not in agent_pcms:
            print(f"  {text[:60]}...")
            agent_pcms[text] = asyncio.run(synthesize_text(text))

    # Record each scenario
    results = []
    for i, name in enumerate(selected):
        defn = SCENARIO_DEFS[name]
        agent_pcm = agent_pcms.get(defn["agent_text"]) if defn["agent_text"] else np.zeros(0, dtype=np.float32)

        result = record_scenario(
            scenario_name=name,
            scenario_def=defn,
            input_device=args.input_device,
            output_device=args.output_device,
            output_dir=output_dir,
            agent_pcm=agent_pcm,
        )
        results.append(result)

        if i < len(selected) - 1:
            print(f"\n--- Next: {selected[i+1]} ---")
            time.sleep(2)

    # Save summary
    summary = {
        "recorded_at": datetime.now().isoformat(),
        "input_device": args.input_device,
        "output_device": args.output_device,
        "scenarios": results,
    }
    with open(output_dir / "recording_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RECORDING COMPLETE — {len(results)} scenarios saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
