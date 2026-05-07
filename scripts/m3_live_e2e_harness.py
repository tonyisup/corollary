#!/usr/bin/env python3
"""M3 Live E2E Validation Harness.

Replays pre-recorded captured audio through the speaker and re-captures it
through the mic, running the full attribution + controller pipeline in real-time.
Compares live results against the offline analysis of the same recording.

This tests:
- Room acoustics (speaker → room → mic)
- Real-time capture latency and jitter
- Attribution accuracy with live audio
- Controller state transitions under real conditions

Usage:
    python scripts/m3_live_e2e_harness.py --scenario human_early
    python scripts/m3_live_e2e_harness.py --all
    python scripts/m3_live_e2e_harness.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from cochlea.capture import SR, CaptureStream
from cochlea.vad import VAD_FRAME_SAMPLES, SmoothedVAD
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState
from pfc.m2_artifact_replay import frames_from_frame_records
from pfc.m2_turn_bridge import run_turn_taking_bridge
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts


# ── Scenario definitions ──────────────────────────────────────────────────────

@dataclass
class LiveScenario:
    name: str
    captured_audio_file: str       # Path to .npy file with pre-recorded captured audio
    agent_pcm_file: str            # Path to .npy file with agent TTS
    agent_text: str
    description: str = ""


DATA_DIR = Path("data/m3_interrupt_v2")

SCENARIOS: dict[str, LiveScenario] = {
    "human_early": LiveScenario(
        name="human_early",
        captured_audio_file="data/m3_interrupt_v2/human_early/captured.npy",
        agent_pcm_file="data/m3_interrupt_v2/human_early/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        description="Human starts mid-utterance",
    ),
    "human_late": LiveScenario(
        name="human_late",
        captured_audio_file="data/m3_interrupt_v2/human_late/captured.npy",
        agent_pcm_file="data/m3_interrupt_v2/human_late/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        description="Human starts near end of utterance",
    ),
    "human_before": LiveScenario(
        name="human_before",
        captured_audio_file="data/m3_interrupt_v2/human_before/captured.npy",
        agent_pcm_file="data/m3_interrupt_v2/human_before/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        description="Human speaks before agent starts",
    ),
    "human_after": LiveScenario(
        name="human_after",
        captured_audio_file="data/m3_interrupt_v2/human_after/captured.npy",
        agent_pcm_file="data/m3_interrupt_v2/human_after/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        description="Human speaks after agent finishes",
    ),
    "overlap_self_first": LiveScenario(
        name="overlap_self_first",
        captured_audio_file="data/m3_interrupt_v2/overlap_self_first/captured.npy",
        agent_pcm_file="data/m3_interrupt_v2/overlap_self_first/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        description="Human and agent overlap from the start",
    ),
}


# ── Live replay + capture ────────────────────────────────────────────────────

async def run_live_replay_capture(
    scenario: LiveScenario,
    input_device: int,
    output_device: int,
) -> dict:
    """Replay pre-recorded captured audio through speaker, re-capture through mic.
    
    This simulates the real conversation: the pre-recorded audio contains both
    agent TTS and human speech as captured by the mic. Replaying it through the
    speaker and re-capturing tests the full room acoustics chain.
    
    Returns dict with live_captured, agent_pcm, replay_audio.
    """
    import sounddevice as sd
    
    # Load pre-recorded audio
    replay_audio = np.load(scenario.captured_audio_file).astype(np.float32)
    agent_pcm = np.load(scenario.agent_pcm_file).astype(np.float32)
    
    duration_s = len(replay_audio) / SR + 1.0  # Extra second for tail
    total_samples = int(duration_s * SR)
    live_captured = np.zeros(total_samples, dtype=np.float32)
    
    print(f"  Replay audio: {len(replay_audio)} samples ({len(replay_audio)/SR:.2f}s)")
    print(f"  Agent PCM:    {len(agent_pcm)} samples ({len(agent_pcm)/SR:.2f}s)")
    print(f"  Duration:     {duration_s:.1f}s")
    
    capture_queue = asyncio.Queue()
    samples_played = 0
    
    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal samples_played
        if status:
            print(f"  Audio status: {status}")
        
        # Output: replay the pre-recorded audio
        start = samples_played
        end = min(start + frames, len(replay_audio))
        if start < len(replay_audio):
            outdata[:end-start, 0] = replay_audio[start:end]
            if end - start < frames:
                outdata[end-start:, 0] = 0
        else:
            outdata[:] = 0
        
        # Input: capture the live mic signal
        capture_queue.put_nowait(indata[:, 0].copy())
        samples_played += frames
    
    print(f"  Starting live replay+capture (input={input_device}, output={output_device})...")
    t0 = time.time()
    
    with sd.Stream(
        samplerate=SR,
        blocksize=VAD_FRAME_SAMPLES,
        dtype="float32",
        channels=1,
        callback=audio_callback,
        device=(input_device, output_device),
    ):
        await asyncio.sleep(duration_s + 0.5)
    
    elapsed = time.time() - t0
    
    # Collect captured audio
    captured_frames = []
    while not capture_queue.empty():
        captured_frames.append(capture_queue.get_nowait())
    
    if captured_frames:
        live_captured = np.concatenate(captured_frames)[:total_samples]
    
    print(f"  Capture complete: {len(live_captured)} samples in {elapsed:.1f}s")
    
    return {
        "live_captured": live_captured,
        "agent_pcm": agent_pcm,
        "replay_audio": replay_audio,
        "capture_time_s": elapsed,
    }


# ── Attribution pipeline ─────────────────────────────────────────────────────

def run_attribution_pipeline(
    captured_audio: np.ndarray,
    agent_pcm: np.ndarray,
    attributor: ThalamicAttributor,
    vad: SmoothedVAD,
) -> list[dict]:
    """Run the full attribution pipeline on captured audio.
    
    Returns a list of frame records compatible with frames_from_frame_records().
    """
    frame_records = []
    frame_step = VAD_FRAME_SAMPLES
    num_frames = max(0, (len(captured_audio) - frame_step) // frame_step)
    
    # Compute global self score
    from scripts.m2_speech_preservation_harness import normalized_xcorr
    global_score, _ = normalized_xcorr(captured_audio, agent_pcm)
    
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_step
        if end > len(captured_audio):
            break
        frame_audio = captured_audio[start:end]
        
        # VAD
        is_speech = vad.process(frame_audio)
        
        # RMS
        rms = np.sqrt(np.mean(frame_audio ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        
        # Self-match score
        is_playing = end <= len(agent_pcm)
        match_score = global_score if is_playing else 0.0
        
        # Attribution
        result = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=start,
            expected_onset_samples=0,
            rms_db=rms_db,
            is_playing=is_playing,
            self_pcm=agent_pcm,
            agent_start_sample=0,
            frame_audio=frame_audio,
        )
        
        frame_records.append({
            "frame_index": i,
            "time_s": start / SR,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "self_match_score": result.self_match_score,
            "self_match_score_per_frame": result.self_match_score_per_frame,
            "is_in_motor_window": result.is_in_motor_window,
            "causal_score": result.causal_score,
            "energy_db": result.energy_db,
            "is_speech": is_speech,
            "rms_db": rms_db,
        })
    
    return frame_records


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class LiveMetrics:
    scenario: str
    total_frames: int = 0
    false_interrupts: int = 0
    false_defers: int = 0
    missed_interrupts: int = 0
    true_interrupts: int = 0
    successful_resumes: int = 0
    total_resumes_expected: int = 0
    state_transitions: list = field(default_factory=list)
    latency_ms: float = 0.0
    notes: list = field(default_factory=list)
    
    def summary(self) -> dict:
        return {
            "scenario": self.scenario,
            "total_frames": self.total_frames,
            "false_interrupts": self.false_interrupts,
            "false_defers": self.false_defers,
            "missed_interrupts": self.missed_interrupts,
            "true_interrupts": self.true_interrupts,
            "successful_resumes": self.successful_resumes,
            "total_resumes_expected": self.total_resumes_expected,
            "resume_rate": f"{self.successful_resumes}/{self.total_resumes_expected}",
            "state_transitions": self.state_transitions,
            "latency_ms": round(self.latency_ms, 1),
            "notes": self.notes,
        }


# ── Main harness ──────────────────────────────────────────────────────────────

async def run_scenario(scenario: LiveScenario, args: argparse.Namespace) -> dict:
    """Run a single live E2E scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"{'='*60}")
    
    # 1. Run live replay + capture
    print(f"\n[1/3] Running live replay + capture...")
    t0 = time.time()
    capture_result = await run_live_replay_capture(
        scenario, args.input_device, args.output_device
    )
    capture_time = (time.time() - t0) * 1000
    
    live_captured = capture_result["live_captured"]
    agent_pcm = capture_result["agent_pcm"]
    
    # 2. Run attribution pipeline
    print(f"\n[2/3] Running attribution pipeline...")
    attributor = ThalamicAttributor(
        energy_threshold_db=-55.0,
        similarity_threshold=0.05,
        onset_latency_ms=300.0,
        causal_tolerance_ms=175.0,
    )
    vad = SmoothedVAD()
    
    t0 = time.time()
    frame_records = run_attribution_pipeline(live_captured, agent_pcm, attributor, vad)
    attr_time = (time.time() - t0) * 1000
    
    print(f"  Attribution time: {attr_time:.0f}ms")
    print(f"  Frames: {len(frame_records)}")
    
    # Count verdicts
    verdict_counts = {}
    for fr in frame_records:
        v = fr["verdict"].name if hasattr(fr["verdict"], "name") else str(fr["verdict"])
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"  Verdicts: {verdict_counts}")
    
    # 3. Run controller
    print(f"\n[3/3] Running controller...")
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
    )
    
    frames = frames_from_frame_records(frame_records)
    t0 = time.time()
    bridge = run_turn_taking_bridge(
        frames,
        scenario.agent_text,
        controller=controller,
        request_at_s=0.0,
    )
    ctrl_time = (time.time() - t0) * 1000
    
    print(f"  Controller time: {ctrl_time:.0f}ms")
    print(f"  Accepted initial: {bridge.accepted_initial_output}")
    print(f"  Resumed: {bridge.resumed_output}")
    
    # State transitions
    states = [d.state.name for d in bridge.decisions]
    transitions = []
    prev = None
    for s in states:
        if s != prev:
            transitions.append(s)
            prev = s
    print(f"  States: {' -> '.join(transitions[:15])}")
    
    # Compute metrics
    metrics = LiveMetrics(scenario=scenario.name)
    metrics.total_frames = len(bridge.decisions)
    metrics.state_transitions = transitions
    metrics.latency_ms = capture_time + attr_time + ctrl_time
    
    # Count interrupts
    for i, d in enumerate(bridge.decisions):
        if i < len(frame_records):
            fr = frame_records[i]
            gt_human = fr.get("verdict") in (SourceVerdict.WORLD, SourceVerdict.BOTH) and fr.get("is_speech", False)
            gt_agent = fr.get("is_in_motor_window", False)
            
            if d.should_interrupt and not gt_human:
                metrics.false_interrupts += 1
            elif d.should_interrupt and gt_human:
                metrics.true_interrupts += 1
            elif not d.should_interrupt and gt_human and gt_agent:
                metrics.missed_interrupts += 1
    
    # Resume
    metrics.successful_resumes = 1 if bridge.resumed_output else 0
    metrics.total_resumes_expected = 1
    
    print(f"\n{'='*60}")
    print(f"Results for {scenario.name}:")
    summary = metrics.summary()
    for k, v in summary.items():
        if k not in ("state_transitions", "notes"):
            print(f"  {k}: {v}")
    print(f"{'='*60}")
    
    # Save artifacts
    if args.save_dir:
        save_path = Path(args.save_dir) / scenario.name
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / "live_captured.npy", live_captured)
        np.save(save_path / "agent_pcm.npy", agent_pcm)
        
        serializable_records = []
        for fr in frame_records:
            sr = dict(fr)
            if hasattr(sr["verdict"], "name"):
                sr["verdict"] = sr["verdict"].name
            serializable_records.append(sr)
        
        with open(save_path / "frame_records.json", "w") as f:
            json.dump(serializable_records, f, indent=2, default=str)
        
        with open(save_path / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n  Saved artifacts to: {save_path}")
    
    return summary


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="M3 Live E2E Validation Harness")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a single scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    parser.add_argument("--save-dir", default=None, help="Directory to save artifacts")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable scenarios:")
        for name, sc in SCENARIOS.items():
            print(f"  {name:25s} — {sc.description}")
        return
    
    if not args.scenario and not args.all:
        parser.print_help()
        return
    
    scenarios_to_run = list(SCENARIOS.values()) if args.all else [SCENARIOS[args.scenario]]
    
    all_metrics = []
    for scenario in scenarios_to_run:
        try:
            metrics = await run_scenario(scenario, args)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nERROR in scenario {scenario.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if len(all_metrics) > 1:
        print(f"\n\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        total_fi = sum(m.get("false_interrupts", 0) for m in all_metrics)
        total_fd = sum(m.get("false_defers", 0) for m in all_metrics)
        total_mi = sum(m.get("missed_interrupts", 0) for m in all_metrics)
        total_ti = sum(m.get("true_interrupts", 0) for m in all_metrics)
        total_sr = sum(m.get("successful_resumes", 0) for m in all_metrics)
        total_tr = sum(m.get("total_resumes_expected", 0) for m in all_metrics)
        
        print(f"  False interrupts:  {total_fi}")
        print(f"  False defers:      {total_fd}")
        print(f"  Missed interrupts: {total_mi}")
        print(f"  True interrupts:   {total_ti}")
        print(f"  Resume rate:       {total_sr}/{total_tr}")
        print(f"{'='*60}")
    
    if args.save_dir:
        overall_path = Path(args.save_dir) / "live_e2e_results.json"
        with open(overall_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nSaved overall results to: {overall_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
