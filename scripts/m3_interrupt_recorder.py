#!/usr/bin/env python3
"""M3 Interruption Scenario Recorder v2.

Records controlled human-speech-overlaps-Corollary scenarios with clear
visual cues for when to speak.

Usage:
    .venv/bin/python scripts/m3_interrupt_recorder.py --scenarios all
    .venv/bin/python scripts/m3_interrupt_recorder.py --scenarios human_early,human_late
    .venv/bin/python scripts/m3_interrupt_recorder.py --list-devices
    .venv/bin/python scripts/m3_interrupt_recorder.py --list-scenarios

Before running:
    1. Set your mic close to your mouth
    2. Set your speaker.volume to a comfortable level
    3. Run --list-devices to confirm input/output device numbers

Recording process:
    - Each scenario starts with "========== SPEAK NOW ==========" (green) or
      ".......... stay silent .........." (dim)
    - You see exactly when to speak and when to stay quiet
    - If you mess up, pick redo at the prompt after each scenario, or use
      --redo to re-record specific scenarios from the command line
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
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
# Times are relative to recording start. lead_in_s=0.5s before first event.

SCENARIO_DEFS = {
    "self_only": {
        "description": "Corollary speaks alone — no human speech",
        "agent_text": "I'm Corollary. I am speaking alone.",
        "speak_windows": [],  # No human speech expected
        "expected_behavior": "complete",
        "comment": "Just let Corollary speak. You stay silent.",
    },
    "world_only": {
        "description": "Human speaks alone — no Corollary",
        "agent_text": "",
        "speak_windows": [(0.5, 2.0)],  # Speak from 0.5s to 2.0s
        "expected_behavior": "defer",
        "comment": "Speak immediately after the beep. Say something like 'Hello, I'm here.'",
    },
    "human_before": {
        "description": "Human speaks before Corollary",
        "agent_text": "I'm Corollary. Was someone talking?",
        "speak_windows": [(0.5, 1.5)],  # Speak before agent
        "expected_behavior": "defer",
        "comment": "Speak IMMEDIATELY when you see the cue. Corollary will start after you.",
    },
    "human_after": {
        "description": "Human speaks after Corollary finishes",
        "agent_text": "I'm Corollary. I have finished.",
        "speak_windows": [(3.5, 4.5)],  # Speak after agent finishes (~3s)
        "expected_behavior": "respond",
        "comment": "Stay silent while Corollary speaks. Then respond when you see the cue.",
    },
    "human_early": {
        "description": "Human interrupts Corollary early",
        "agent_text": "I'm Corollary. I was about to say something important that will take a while.",
        "speak_windows": [(1.5, 2.5)],  # Interrupt during first half
        "expected_behavior": "interrupt_defer_resume",
        "comment": "Interrupt Corollary EARLY, about 1 second after they start.",
    },
    "human_late": {
        "description": "Human interrupts Corollary late",
        "agent_text": "I'm Corollary. I have more to say and will keep talking for a bit longer.",
        "speak_windows": [(3.0, 4.0)],  # Interrupt during second half
        "expected_behavior": "interrupt_defer_resume",
        "comment": "Interrupt Corollary LATE, well into their utterance.",
    },
    "brief_noise": {
        "description": "Brief noise during Corollary (should not interrupt)",
        "agent_text": "I'm Corollary. I am speaking continuously.",
        "speak_windows": [(2.0, 2.15)],  # Very brief — ~150ms
        "expected_behavior": "no_interrupt",
        "comment": "Make a very SHORT noise — like 'ah' — about 150ms. Just a cough or quick syllable.",
    },
    "human_double": {
        "description": "Human interrupts twice",
        "agent_text": "I'm Corollary. I'm trying to finish my thought.",
        "speak_windows": [(1.5, 2.0), (3.5, 4.0)],  # Two interruptions
        "expected_behavior": "interrupt_defer_resume_interrupt",
        "comment": "Interrupt twice: once early, once later. Each time say something short.",
    },
    "overlap_human_first": {
        "description": "Human starts first, Corollary joins",
        "agent_text": "I'm Corollary. I'm joining in.",
        "speak_windows": [(0.5, 2.5)],  # Human starts first
        "agent_delay_s": 1.5,  # Agent starts 1.5s after recording start
        "expected_behavior": "defer_then_respond",
        "comment": "Start speaking immediately. Corollary will join while you're talking.",
    },
    "overlap_self_first": {
        "description": "Corollary starts, human interrupts",
        "agent_text": "I'm Corollary. I started speaking first.",
        "speak_windows": [(2.0, 3.0)],  # Human interrupts
        "expected_behavior": "interrupt_defer_resume",
        "comment": "Let Corollary start, then interrupt them about 1.5s in.",
    },
}


def color(text, code):
    """Add ANSI color."""
    return f"\033[{code}m{text}\033[0m"


def red(text): return color(text, "91")
def green(text): return color(text, "92")
def yellow(text): return color(text, "93")
def dim(text): return color(text, "2")
def bold(text): return color(text, "1")


def play_array_blocking(pcm: np.ndarray, device: int, samplerate: int) -> None:
    sd.play(pcm, samplerate=samplerate, device=device)
    sd.wait()


async def synthesize_text(text: str) -> np.ndarray:
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
    """Record a single interruption scenario with clear visual cues."""

    desc = scenario_def["description"]
    agent_text = scenario_def["agent_text"]
    speak_windows = scenario_def["speak_windows"]
    expected = scenario_def["expected_behavior"]
    comment = scenario_def["comment"]
    agent_delay_s = scenario_def.get("agent_delay_s", 0.0)

    # Synthesize agent audio
    if agent_pcm is None and agent_text:
        print(f"  Synthesizing: {agent_text[:60]}...")
        agent_pcm = asyncio.run(synthesize_text(agent_text))
    elif not agent_text:
        agent_pcm = np.zeros(0, dtype=np.float32)

    agent_duration_s = len(agent_pcm) / SR if len(agent_pcm) > 0 else 0

    # Timing
    lead_in_s = 0.5
    tail_s = 1.0
    agent_start_s = lead_in_s + agent_delay_s

    # Calculate total duration from events
    all_end_times = [agent_start_s + agent_duration_s, agent_start_s + agent_duration_s + 1.0]
    for start, end in speak_windows:
        rec_start = lead_in_s + start
        all_end_times.append(rec_start + end - start)
    total_duration = max(all_end_times) + tail_s + 0.5

    # Device SR
    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)

    # Build output mix
    total_samples = int(total_duration * output_sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    if len(agent_pcm) > 0:
        agent_resampled = agent_pcm
        if abs(output_sr - SR) > 1:
            import librosa
            agent_resampled = librosa.resample(agent_pcm, orig_sr=SR, target_sr=output_sr)
        start_idx = int(agent_start_s * output_sr)
        end_idx = min(total_samples, start_idx + len(agent_resampled))
        output_audio[start_idx:end_idx] = agent_resampled[:end_idx - start_idx]

    # Ground truth labels
    total_input_samples = int(total_duration * input_sr)
    gt_human = np.zeros(total_input_samples, dtype=bool)
    gt_agent = np.zeros(total_input_samples, dtype=bool)
    gt_should = np.zeros(total_input_samples, dtype=bool)

    if len(agent_pcm) > 0:
        a_start = int(agent_start_s * input_sr)
        a_end = int((agent_start_s + agent_duration_s) * input_sr)
        gt_agent[a_start:a_end] = True

    for speak_start, speak_end in speak_windows:
        s_start = int((lead_in_s + speak_start) * input_sr)
        s_end = int((lead_in_s + speak_end) * input_sr)
        gt_human[s_start:s_end] = True

    gt_should = gt_agent & ~gt_human

    # ── Display cue timeline ──
    print(f"\n{'='*60}")
    print(bold(f"Scenario: {scenario_name}"))
    print(f"  {desc}")
    print(f"  Expected: {expected}")
    print(f"  {dim(comment)}")
    print()

    # Show timeline
    timeline_s = total_duration
    print("  TIMELINE:")
    print(f"  {'':>6s} ", end="")
    for t in range(int(timeline_s) + 1):
        print(f"{t:>5d}", end="")
    print()

    print(f"  {'SPEAK':>6s} ", end="")
    scale = 5  # 5 chars per second
    speak_line = [' '] * (int(timeline_s) * scale + 5)
    for ws, we in speak_windows:
        ws_idx = int((lead_in_s + ws) * scale)
        we_idx = int((lead_in_s + we) * scale)
        for i in range(max(0, ws_idx), min(len(speak_line), we_idx)):
            speak_line[i] = '█'
    print(''.join(speak_line))

    if agent_text:
        print(f"  {'COROLLARY':>6s} ", end="")
        cor_line = [' '] * (int(timeline_s) * scale + 5)
        cs_idx = int(agent_start_s * scale)
        ce_idx = int((agent_start_s + agent_duration_s) * scale)
        for i in range(max(0, cs_idx), min(len(cor_line), ce_idx)):
            cor_line[i] = '▓'
        print(''.join(cor_line))

    print()
    print(f"  {green('███')} = you speak    {dim('▓▓▓')} = Corollary speaks")
    print()

    # Pre-countdown
    print(dim("  Get ready..."))
    time.sleep(1)

    # Countdown
    for i in range(3, 0, -1):
        print(f"  {yellow(str(i))}...")
        time.sleep(1)

    print(bold(green("  >>> RECORDING <<<")))
    print()

    # Start recording and playback
    captured = np.zeros(total_input_samples, dtype=np.float32)
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr,
                           dtype="float32", blocksize=1024)
    stream.start()

    playback_thread = threading.Thread(
        target=play_array_blocking, args=(output_audio, output_device, output_sr)
    )
    playback_thread.start()

    # Show real-time cues during recording
    start_time = time.time()
    cue_thread = threading.Thread(
        target=_show_cues,
        args=(speak_windows, agent_start_s, agent_duration_s, total_duration, lead_in_s),
        daemon=True,
    )
    cue_thread.start()

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

    print()
    print(dim("  Recording complete."))

    # Save artifacts
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(scenario_dir / "captured.npy"), captured)
    if len(agent_pcm) > 0:
        np.save(str(scenario_dir / "agent_pcm.npy"), agent_pcm)
    np.save(str(scenario_dir / "gt_human_speaking.npy"), gt_human)
    np.save(str(scenario_dir / "gt_agent_playing.npy"), gt_agent)
    np.save(str(scenario_dir / "gt_agent_should_speak.npy"), gt_should)

    metadata = {
        "scenario_name": scenario_name,
        "description": desc,
        "expected_behavior": expected,
        "agent_text": agent_text,
        "agent_start_s": agent_start_s,
        "agent_duration_s": agent_duration_s,
        "agent_samples": len(agent_pcm),
        "speak_windows": speak_windows,
        "total_duration_s": total_duration,
        "input_sr": input_sr,
        "output_sr": output_sr,
        "lead_in_s": lead_in_s,
        "tail_s": tail_s,
        "recorded_at": datetime.now().isoformat(),
    }
    with open(scenario_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Quality check
    rms = float(np.sqrt(np.mean(captured**2)))
    human_rms = float(np.sqrt(np.mean(captured[gt_human]**2))) if np.any(gt_human) else 0
    agent_rms = float(np.sqrt(np.mean(captured[gt_agent]**2))) if np.any(gt_agent) else 0

    print(f"  RMS: {rms:.4f} | Human region: {human_rms:.4f} | Agent region: {agent_rms:.4f}")

    # Warn if human region is too quiet
    if speak_windows and human_rms < 0.01:
        print(red("  ⚠ WARNING: Human speaking region is very quiet!"))
        print(dim("    You may not have spoken, or the mic may be too quiet."))
        print(dim("    Consider re-recording this scenario with --redo"))
    elif speak_windows and human_rms > 0.01:
        print(green(f"  ✓ Human speech detected (RMS={human_rms:.4f})"))

    print(f"  Saved: {scenario_dir}")

    return {
        "scenario": scenario_name,
        "dir": str(scenario_dir),
        "captured_rms": rms,
        "human_region_rms": human_rms,
        "agent_region_rms": agent_rms,
        "human_speaking_ms": float(np.sum(gt_human)) / input_sr * 1000,
        "quality_ok": human_rms > 0.01 if speak_windows else True,
    }


def _show_cues(speak_windows, agent_start_s, agent_duration_s, total_duration, lead_in_s):
    """Show real-time SPEAK/SILENT cues during recording."""
    start = time.time()
    last_cue = None

    while True:
        elapsed = time.time() - start
        if elapsed > total_duration:
            break

        rec_time = elapsed  # Time since recording started

        # Check if in a speak window
        speaking = False
        for ws, we in speak_windows:
            w_start = lead_in_s + ws
            w_end = lead_in_s + we
            if w_start <= rec_time <= w_end:
                speaking = True
                break

        # Determine current agent state
        agent_active = agent_start_s <= rec_time <= agent_start_s + agent_duration_s

        # Build cue
        if speaking:
            cue = green("  ██████████  SPEAK NOW  ██████████")
        elif agent_active:
            cue = dim("  ..........  listening  ..........")
        else:
            cue = dim("  ..........  stay quiet ..........")

        if cue != last_cue:
            # Clear line and print
            print(f"\r{cue}", end="", flush=True)
            last_cue = cue

        time.sleep(0.05)

    print()


def prompt_post_scenario(current: str, next_scenario: str | None) -> str:
    """Return 'next', 'redo_discard', or 'redo_keep'."""
    if next_scenario:
        next_desc = f"next scenario ({next_scenario})"
    else:
        next_desc = "finish (save summary and exit)"

    while True:
        print()
        print(dim("  ── What next? ──"))
        print(f"    {bold('1')} — Redo {current}: discard this take (delete saved files, record again)")
        print(f"    {bold('2')} — Redo {current}: keep this take (archive folder, then record again)")
        print(f"    {bold('3')} — Continue to {next_desc}")
        raw = input(f"\n  Enter 1, 2, or 3 [default 3]: ").strip().lower()

        if raw in ("", "3", "n", "next"):
            return "next"
        if raw in ("1", "d", "discard", "redo-discard"):
            return "redo_discard"
        if raw in ("2", "k", "keep", "redo-keep"):
            return "redo_keep"
        print(yellow("  Please enter 1, 2, or 3 (empty = continue)."))


def main():
    parser = argparse.ArgumentParser(description="M3 Interruption Scenario Recorder v2")
    parser.add_argument("--scenarios", default="all",
                        help="Comma-separated names, or 'all'")
    parser.add_argument("--output-dir", default="data/m3_interrupt_v2")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--redo", nargs="*", metavar="SCENARIO",
                        help="Re-record specific scenarios")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:\n")
        for i, dev in enumerate(sd.query_devices()):
            print(f"  {i:<4} {dev['name']:<45} in={dev['max_input_channels']} "
                  f"out={dev['max_output_channels']} sr={dev['default_samplerate']}")
        return

    if args.list_scenarios:
        print("\nAvailable scenarios:\n")
        for name, d in SCENARIO_DEFS.items():
            windows = ", ".join(f"{s}-{e}s" for s, e in d["speak_windows"]) if d["speak_windows"] else "none"
            print(f"  {name:<22} speak={windows}")
            print(f"  {'':22} {d['description']}")
            print(f"  {'':22} {d['comment']}")
            print()
        return

    if args.scenarios == "all":
        selected = list(SCENARIO_DEFS.keys())
    else:
        selected = [s.strip() for s in args.scenarios.split(",")]

    if args.redo:
        selected = args.redo

    for s in selected:
        if s not in SCENARIO_DEFS:
            print(f"ERROR: Unknown scenario '{s}'")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(bold("M3 INTERRUPTION SCENARIO RECORDER v2"))
    print("=" * 60)
    print(f"Scenarios: {len(selected)}")
    print(f"Output:    {output_dir}")
    print(f"Input:     device {args.input_device}")
    print(f"Output:    device {args.output_device}")
    print()
    print("During recording you will see:")
    print(f"  {green('██████████  SPEAK NOW  ██████████')}  — speak during these windows")
    print(f"  {dim('..........  listening  ..........')}  — Corollary is talking, stay quiet")
    print(f"  {dim('..........  stay quiet ..........')}  — silence, wait for next cue")
    print()

    # Pre-synthesize agent audio
    print("Pre-synthesizing agent audio...")
    agent_pcms = {}
    for name in selected:
        text = SCENARIO_DEFS[name]["agent_text"]
        if text and text not in agent_pcms:
            print(f"  {text[:60]}...")
            agent_pcms[text] = asyncio.run(synthesize_text(text))

    # Record each scenario (with optional on-the-spot redo)
    results: list[dict] = []
    i = 0
    while i < len(selected):
        name = selected[i]
        defn = SCENARIO_DEFS[name]
        agent_pcm = agent_pcms.get(defn["agent_text"]) if defn["agent_text"] else np.zeros(0, dtype=np.float32)

        while True:
            result = record_scenario(
                scenario_name=name,
                scenario_def=defn,
                input_device=args.input_device,
                output_device=args.output_device,
                output_dir=output_dir,
                agent_pcm=agent_pcm,
            )
            results.append(result)

            next_name = selected[i + 1] if i + 1 < len(selected) else None
            choice = prompt_post_scenario(name, next_name)

            if choice == "next":
                i += 1
                break

            results.pop()
            scenario_dir = output_dir / name

            if choice == "redo_discard":
                if scenario_dir.exists():
                    shutil.rmtree(scenario_dir)
                print(dim(f"  Discarded data under {scenario_dir}; re-recording {name}..."))
            else:  # redo_keep
                if scenario_dir.exists():
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    archive_dir = output_dir / f"{name}_archived_{stamp}"
                    shutil.move(str(scenario_dir), str(archive_dir))
                    print(dim(f"  Archived previous take to {archive_dir}; re-recording {name}..."))
                else:
                    print(dim(f"  No existing folder for {name}; re-recording..."))

    # Summary
    summary = {
        "recorded_at": datetime.now().isoformat(),
        "input_device": args.input_device,
        "output_device": args.output_device,
        "scenarios": results,
    }
    with open(output_dir / "recording_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok = sum(1 for r in results if r["quality_ok"])
    print(f"\n{'='*60}")
    print(f"RECORDING COMPLETE — {ok}/{len(results)} scenarios with good audio")
    print(f"{'='*60}")

    if ok < len(results):
        failed = [r["scenario"] for r in results if not r["quality_ok"]]
        print(red(f"\nRe-record these (human speech not detected): {', '.join(failed)}"))
        print(f"  Command: ./.venv/bin/python scripts/m3_interrupt_recorder.py --redo {' '.join(failed)}")


if __name__ == "__main__":
    main()
