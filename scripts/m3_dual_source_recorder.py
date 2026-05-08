#!/usr/bin/env python3
"""M3 Dual-Source Recorder.

Records interruption scenarios with TTS through the main speaker and
human speech from a live mic. This creates a dataset where the TTS and
human speech have genuinely different acoustic signatures at the mic,
matching real-world conversation geometry.

Usage:
    python scripts/m3_dual_source_recorder.py --list-scenarios
    python scripts/m3_dual_source_recorder.py --scenario human_early
    python scripts/m3_dual_source_recorder.py --all
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


def _expected_behavior(delay_s: float) -> str:
    if delay_s < 0:
        return "defer"
    elif delay_s > 2.0:
        return "respond"
    else:
        return "interrupt_defer_resume"


# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "human_early": {
        "agent_text": "I was speaking, then paused.",
        "human_prompt": "Speak now (human early)",
        "human_delay_s": 0.8,
        "description": "Human interrupts mid-utterance",
    },
    "human_late": {
        "agent_text": "I was speaking, then paused.",
        "human_prompt": "Speak now (human late)",
        "human_delay_s": 1.8,
        "description": "Human interrupts near end of utterance",
    },
    "human_before": {
        "agent_text": "I was speaking, then paused.",
        "human_prompt": "Speak now (before agent)",
        "human_delay_s": -1.0,
        "description": "Human speaks before agent starts",
    },
    "human_after": {
        "agent_text": "I was speaking, then paused.",
        "human_prompt": "Speak now (after agent)",
        "human_delay_s": 3.0,
        "description": "Human speaks after agent finishes",
    },
    "overlap_self_first": {
        "agent_text": "I'm Corollary. I started speaking first.",
        "human_prompt": "Speak now (overlap)",
        "human_delay_s": 0.5,
        "description": "Human and agent overlap from the start",
    },
}


async def record_scenario(
    scenario_name: str,
    scenario: dict,
    input_device: int,
    output_device: int,
    output_dir: Path,
    seed: int = 0,
    lead_in_s: float = 0.5,
    tail_s: float = 1.0,
    monitor: bool = True,
) -> None:
    """Record a single scenario.
    
    Plays TTS through the output device. Records mic input.
    Human speaks at the prompted time (live).
    """
    
    agent_text = scenario["agent_text"]
    human_delay_s = scenario["human_delay_s"]
    
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Description: {scenario['description']}")
    print(f"Agent text: {agent_text!r}")
    print(f"Human delay: {human_delay_s}s (relative to agent start)")
    print(f"Mic input: device {input_device}")
    print(f"TTS output: device {output_device}")
    print(f"{'='*60}")
    
    # Synthesize TTS
    print(f"\nSynthesizing agent TTS...")
    agent_pcm = await synthesize(agent_text)
    agent_samples = len(agent_pcm)
    agent_duration_s = agent_samples / SR
    print(f"  Agent PCM: {agent_samples} samples ({agent_duration_s:.2f}s)")
    
    # Timing
    agent_start_s = lead_in_s
    total_duration_s = agent_start_s + agent_duration_s + tail_s
    
    # Get sample rates
    input_sr = int(sd.query_devices(input_device)["default_samplerate"])
    output_sr = int(sd.query_devices(output_device)["default_samplerate"])
    
    # Resample TTS to output rate
    if output_sr != SR:
        from scipy.signal import resample as scipy_resample
        tts_output = scipy_resample(
            agent_pcm,
            int(len(agent_pcm) * output_sr / SR)
        ).astype(np.float32)
    else:
        tts_output = agent_pcm.copy()
    
    # Build output buffer (silence + TTS + silence)
    total_output_samples = int(total_duration_s * output_sr)
    output_buffer = np.zeros(total_output_samples, dtype=np.float32)
    tts_start_sample = int(agent_start_s * output_sr)
    tts_end_sample = min(tts_start_sample + len(tts_output), total_output_samples)
    output_buffer[tts_start_sample:tts_end_sample] = tts_output[:tts_end_sample - tts_start_sample]
    
    # Capture buffer
    total_input_samples = int(total_duration_s * input_sr)
    mic_capture = np.zeros(total_input_samples, dtype=np.float32)
    
    # Ground truth tracking
    gt_agent_playing = np.zeros(total_input_samples, dtype=bool)
    gt_human_speaking = np.zeros(total_input_samples, dtype=bool)
    
    # Mark agent playing region (at input sample rate)
    agent_start_input = int(agent_start_s * input_sr)
    agent_end_input = min(agent_start_input + int(agent_duration_s * input_sr), total_input_samples)
    gt_agent_playing[agent_start_input:agent_end_input] = True
    
    # Mark expected human speaking region
    human_start_s = agent_start_s + human_delay_s
    if human_start_s >= 0:
        human_start_input = int(human_start_s * input_sr)
        human_end_input = min(human_start_input + int(2.0 * input_sr), total_input_samples)  # ~2s window
        gt_human_speaking[human_start_input:human_end_input] = True
    
    samples_played = [0]
    samples_captured = [0]
    
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Audio status: {status}")
        
        # Output TTS
        start = samples_played[0]
        end = min(start + frames, len(output_buffer))
        if start < len(output_buffer):
            n = end - start
            outdata[:n, 0] = output_buffer[start:end]
            if n < frames:
                outdata[n:, 0] = 0
        else:
            outdata[:] = 0
        
        # Capture mic
        cap_start = samples_captured[0]
        cap_end = min(cap_start + frames, total_input_samples)
        if cap_start < total_input_samples:
            n = cap_end - cap_start
            mic_capture[cap_start:cap_end] = indata[:n, 0]
            samples_captured[0] = cap_end
        
        samples_played[0] = end
    
    # ── Visual timeline ──
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"  {scenario['description']}")
    print(f"  Agent: {agent_text!r}")
    print()
    
    # Build timeline
    timeline_s = int(total_duration_s) + 1
    scale = 5  # chars per second
    
    # Speak windows (relative to recording start)
    speak_windows = []
    if human_delay_s >= 0:
        speak_start = agent_start_s + human_delay_s
    else:
        speak_start = human_delay_s
    speak_end = speak_start + 2.0  # ~2s window
    speak_windows.append((speak_start, speak_end))
    
    # Agent window
    agent_window = (agent_start_s, agent_start_s + agent_duration_s)
    
    # Print timeline header
    print("  TIMELINE:")
    print(f"  {'':>6s} ", end="")
    for t in range(timeline_s + 1):
        print(f"{t:>5d}", end="")
    print()
    
    # Speak line
    print(f"  {'SPEAK':>6s} ", end="")
    speak_line = [' '] * (timeline_s * scale + 5)
    for ws, we in speak_windows:
        ws_idx = int(ws * scale)
        we_idx = int(we * scale)
        for i in range(max(0, ws_idx), min(len(speak_line), we_idx)):
            speak_line[i] = '█'
    print(''.join(speak_line))
    
    # Agent line
    print(f"  {'COROLLARY':>6s} ", end="")
    cor_line = [' '] * (timeline_s * scale + 5)
    cs_idx = int(agent_window[0] * scale)
    ce_idx = int(agent_window[1] * scale)
    for i in range(max(0, cs_idx), min(len(cor_line), ce_idx)):
        cor_line[i] = '▓'
    print(''.join(cor_line))
    
    print()
    print(f"  {'███':>3s} = you speak    {'▓▓▓':>3s} = Corollary speaks")
    print()
    
    # Pre-countdown
    print("  Get ready...")
    await asyncio.sleep(1)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        await asyncio.sleep(1)
    
    print(f"\n  >>> RECORDING! <<<")
    print(f"  (speak during the ███ window)")
    
    t0 = time.time()
    with sd.Stream(
        samplerate=input_sr,
        blocksize=int(VAD_FRAME_SAMPLES * input_sr / SR),
        dtype="float32",
        channels=1,
        callback=audio_callback,
        device=(input_device, output_device),
    ):
        await asyncio.sleep(total_duration_s + 0.3)
    
    elapsed = time.time() - t0
    
    # Trim capture
    mic_capture = mic_capture[:samples_captured[0]]
    gt_agent_playing = gt_agent_playing[:samples_captured[0]]
    gt_human_speaking = gt_human_speaking[:samples_captured[0]]
    
    # Resample capture to SR for attribution
    if input_sr != SR:
        from scipy.signal import resample as scipy_resample
        mic_capture_16k = scipy_resample(
            mic_capture,
            int(len(mic_capture) * SR / input_sr)
        ).astype(np.float32)
        gt_agent_16k = scipy_resample(
            gt_agent_playing.astype(np.float32),
            int(len(gt_agent_playing) * SR / input_sr)
        ).astype(bool)
        gt_human_16k = scipy_resample(
            gt_human_speaking.astype(np.float32),
            int(len(gt_human_speaking) * SR / input_sr)
        ).astype(bool)
    else:
        mic_capture_16k = mic_capture
        gt_agent_16k = gt_agent_playing
        gt_human_16k = gt_human_speaking
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "captured.npy", mic_capture_16k)
    np.save(output_dir / "agent_pcm.npy", agent_pcm)
    np.save(output_dir / "gt_agent_playing.npy", gt_agent_16k)
    np.save(output_dir / "gt_agent_should_speak.npy", gt_agent_16k)  # Same as playing for our purposes
    np.save(output_dir / "gt_human_speaking.npy", gt_human_16k)
    
    # Detect actual human speech from capture (for verification)
    from cochlea.vad import SmoothedVAD
    vad = SmoothedVAD()
    human_frames = 0
    for i in range(0, len(mic_capture_16k) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES):
        frame = mic_capture_16k[i:i + VAD_FRAME_SAMPLES]
        if vad.process_frame(frame):
            human_frames += 1
    
    metadata = {
        "scenario_name": scenario_name,
        "description": scenario["description"],
        "agent_text": agent_text,
        "agent_start_s": agent_start_s,
        "agent_duration_s": agent_duration_s,
        "human_delay_s": human_delay_s,
        "total_duration_s": total_duration_s,
        "input_sr": input_sr,
        "output_sr": output_sr,
        "input_device": input_device,
        "output_device": output_device,
        "recorded_at": datetime.now().isoformat(),
        "dual_source": True,
        "human_frames_detected": human_frames,
        "quality_ok": human_frames > 10,
        "expected_behavior": _expected_behavior(human_delay_s),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Capture complete: {len(mic_capture_16k)} samples ({len(mic_capture_16k)/SR:.2f}s)")
    print(f"  Human speech frames detected: {human_frames}")
    print(f"  Quality: {'OK' if human_frames > 10 else 'LOW - check recording'}")
    print(f"  Saved to: {output_dir}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="M3 Dual-Source Recorder")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a single scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--input-device", type=int, default=3, help="Mic input device")
    parser.add_argument("--output-device", type=int, default=5, help="TTS output device (main speaker)")
    parser.add_argument("--output-dir", default="data/m3_interrupt_v4-dual_source", help="Output directory")
    parser.add_argument("--lead-in-s", type=float, default=0.5, help="Lead-in before agent starts")
    parser.add_argument("--tail-s", type=float, default=1.0, help="Tail after agent finishes")
    parser.add_argument("--redo", action="store_true", help="Re-record existing scenarios")
    
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
        scenario_dir = output_dir / name
        if scenario_dir.exists() and not args.redo:
            print(f"\nSkipping {name} (already exists, use --redo to re-record)")
            continue
        
        try:
            await record_scenario(
                name, scenario,
                args.input_device,
                args.output_device,
                scenario_dir,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
            )
        except KeyboardInterrupt:
            print(f"\n  Cancelled {name}")
            continue
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
