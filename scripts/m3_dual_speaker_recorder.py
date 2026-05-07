#!/usr/bin/env python3
"""M3 Dual-Speaker Interruption Recorder.

Records interruption scenarios with TTS through the main speaker and
human speech through an external speaker. This separates the acoustic
sources, making the overlap detection problem more realistic.

Usage:
    python scripts/m3_dual_speaker_recorder.py --list-scenarios
    python scripts/m3_dual_speaker_recorder.py --scenario human_early
    python scripts/m3_dual_speaker_recorder.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
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
from cochlea.vad import VAD_FRAME_SAMPLES


# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "human_early": {
        "agent_text": "I was speaking, then paused.",
        "human_text": "Excuse me, I have a question.",
        "human_delay_s": 0.8,  # Human starts 0.8s into agent utterance
        "description": "Human interrupts mid-utterance",
    },
    "human_late": {
        "agent_text": "I was speaking, then paused.",
        "human_text": "Sorry, could you repeat that?",
        "human_delay_s": 1.8,
        "description": "Human interrupts near end of utterance",
    },
    "human_before": {
        "agent_text": "I was speaking, then paused.",
        "human_text": "Hello, can I ask something?",
        "human_delay_s": -1.0,  # Human speaks before agent starts
        "description": "Human speaks before agent starts",
    },
    "human_after": {
        "agent_text": "I was speaking, then paused.",
        "human_text": "Okay, thanks for explaining.",
        "human_delay_s": 3.0,  # Human speaks after agent finishes
        "description": "Human speaks after agent finishes",
    },
    "overlap_self_first": {
        "agent_text": "I'm Corollary. I started speaking first.",
        "human_text": "Hold on, I need to say something.",
        "human_delay_s": 0.5,
        "description": "Human and agent overlap from the start",
    },
}


async def record_scenario(
    scenario_name: str,
    scenario: dict,
    input_device: int,
    tts_output_device: int,
    human_output_device: int,
    output_dir: Path,
    seed: int = 0,
) -> None:
    """Record a single scenario with dual-speaker output."""
    
    agent_text = scenario["agent_text"]
    human_text = scenario["human_text"]
    human_delay_s = scenario["human_delay_s"]
    
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Description: {scenario['description']}")
    print(f"Agent text: {agent_text!r}")
    print(f"Human text: {human_text!r}")
    print(f"Human delay: {human_delay_s}s")
    print(f"TTS output: device {tts_output_device}")
    print(f"Human output: device {human_output_device}")
    print(f"Mic input: device {input_device}")
    print(f"{'='*60}")
    
    # Synthesize TTS
    print(f"\nSynthesizing agent TTS...")
    agent_pcm = await synthesize(agent_text)
    agent_samples = len(agent_pcm)
    agent_duration_s = agent_samples / SR
    print(f"  Agent PCM: {agent_samples} samples ({agent_duration_s:.2f}s)")
    
    # Synthesize human speech (using same TTS for now — in practice, you'd
    # pre-record human speech or use a different voice)
    print(f"Synthesizing human speech...")
    human_pcm = await synthesize(human_text)
    human_samples = len(human_pcm)
    human_duration_s = human_samples / SR
    print(f"  Human PCM: {human_samples} samples ({human_duration_s:.2f}s)")
    
    # Calculate timing
    agent_start_s = 0.5  # Lead-in before agent starts
    human_start_s = agent_start_s + human_delay_s if human_delay_s >= 0 else human_delay_s
    total_duration_s = max(
        agent_start_s + agent_duration_s,
        human_start_s + human_duration_s if human_start_s >= 0 else human_duration_s
    ) + 1.5  # Tail
    
    # Get sample rates
    input_sr = int(sd.query_devices(input_device)["default_samplerate"])
    tts_sr = int(sd.query_devices(tts_output_device)["default_samplerate"])
    human_sr = int(sd.query_devices(human_output_device)["default_samplerate"])
    
    # Resample TTS to output rate
    if tts_sr != SR:
        from scipy.signal import resample as scipy_resample
        tts_output = scipy_resample(agent_pcm, int(len(agent_pcm) * tts_sr / SR)).astype(np.float32)
    else:
        tts_output = agent_pcm.copy()
    
    # Resample human to output rate
    if human_sr != SR:
        from scipy.signal import resample as scipy_resample
        human_output = scipy_resample(human_pcm, int(len(human_pcm) * human_sr / SR)).astype(np.float32)
    else:
        human_output = human_pcm.copy()
    
    # Build output buffers
    total_samples = int(total_duration_s * input_sr)
    mic_capture = np.zeros(total_samples, dtype=np.float32)
    
    # We'll use a single callback that outputs to both devices
    # This requires opening two output streams and one input stream
    # For simplicity, we'll mix both outputs into a single stereo stream
    # and use a virtual aggregate device
    
    # Actually, the simplest approach: use two separate output streams
    # and synchronize them manually
    
    print(f"\nRecording {total_duration_s:.1f}s...")
    print(f"  Agent starts at t={agent_start_s:.1f}s")
    print(f"  Human starts at t={human_start_s:.1f}s")
    print(f"  Press Ctrl+C to cancel")
    
    # For now, use a simpler approach: mix both outputs into a single
    # output stream. This doesn't give us the acoustic separation we want,
    # but it's a start.
    
    # TODO: Implement true dual-output using aggregate device or
    # separate streams
    
    print(f"\n  [TODO: Dual-output recording not yet implemented]")
    print(f"  For now, use the single-speaker recorder and manually")
    print(f"  play human speech through the external speaker.")
    
    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scenario_name": scenario_name,
        "description": scenario["description"],
        "agent_text": agent_text,
        "human_text": human_text,
        "human_delay_s": human_delay_s,
        "agent_start_s": agent_start_s,
        "human_start_s": human_start_s,
        "total_duration_s": total_duration_s,
        "input_sr": input_sr,
        "tts_sr": tts_sr,
        "human_sr": human_sr,
        "tts_output_device": tts_output_device,
        "human_output_device": human_output_device,
        "input_device": input_device,
        "recorded_at": datetime.now().isoformat(),
        "dual_speaker": True,
        "status": "not_implemented",
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved metadata to: {output_dir / 'metadata.json'}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="M3 Dual-Speaker Interruption Recorder")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a single scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--input-device", type=int, default=3, help="Mic input device")
    parser.add_argument("--tts-output-device", type=int, default=4, help="TTS output device (main speaker)")
    parser.add_argument("--human-output-device", type=int, default=0, help="Human speech output device (external speaker)")
    parser.add_argument("--output-dir", default="data/m3_interrupt_v4-dual_speaker", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for name, sc in SCENARIOS.items():
            print(f"  {name:25s} — {sc['description']}")
        return
    
    if not args.scenario and not args.all:
        parser.print_help()
        return
    
    scenarios_to_run = list(SCENARIOS.items()) if args.all else [(args.scenario, SCENARIOS[args.scenario])]
    output_dir = Path(args.output_dir)
    
    for name, scenario in scenarios_to_run:
        await record_scenario(
            name, scenario,
            args.input_device,
            args.tts_output_device,
            args.human_output_device,
            output_dir / name,
            args.seed,
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
