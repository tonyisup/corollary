#!/usr/bin/env python3
"""M7 Live Drift Validation Harness.

Validates the full M7 pipeline (trust-gated NLMS + drift alarm + IR bank +
device profile) on live hardware with a real room change.

Protocol:
1. Bootstrap: Play TTS, capture on mic, fit initial room IR
2. Adapt: Run trust-gated NLMS adaptation for N frames
3. Drift: User changes room (moves mic, changes orientation, etc.)
4. Recover: System should detect drift via DriftAlarm and adapt

Usage:
    .venv/bin/python scripts/m7_live_drift_validation.py \
        --input-device 1 \
        --output-device 2 \
        --duration 15.0 \
        --output tmp/m7_live_drift_validation.json

The script will prompt you when to change the room setup.
"""

from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.streaming_runtime import StreamingConfig, StreamingRuntime
from pfc.self_model import SelfModel
from sag.graph import SemanticAbstractionGraph
from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.drift_alarm import DriftAlarm
from cerebellum.ir_bank import IRBank
from cerebellum.device_profile import DeviceProfileBank, DeviceOrientation
from broca.tts import synthesize as _synthesize_tts


def make_tts_pcm(text: str = "Hello, I am Corollary. This is a live drift validation test.", duration_s: float = 7.0) -> np.ndarray:
    """Generate TTS audio for drift validation.

    Uses real TTS synthesis (AriaNeural) instead of synthetic multi-harmonic
    signal to avoid crackling/popping from amplitude modulation artifacts.
    """
    import asyncio

    # Generate TTS — loop the text if needed to fill duration
    loop_text = text + " " + text + " " + text  # ~3x coverage
    pcm = asyncio.run(_synthesize_tts(loop_text, voice="en-US-AriaNeural"))
    pcm = pcm.astype(np.float32)
    target_samples = int(duration_s * 16000)
    if len(pcm) < target_samples:
        # Loop to fill
        repeats = (target_samples // len(pcm)) + 1
        pcm = np.tile(pcm, repeats)
    return pcm[:target_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Live Drift Validation")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=2)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Total duration in seconds (default: 30)")
    parser.add_argument("--room-change-s", type=float, default=None,
                        help="Time in seconds to prompt room change (default: duration/2)")
    parser.add_argument("--output", type=str, default="/tmp/m7_live_drift.json")
    args = parser.parse_args()

    room_change_s = args.room_change_s if args.room_change_s else args.duration / 2

    print("=" * 60)
    print("M7 Live Drift Validation — Room Change Protocol")
    print("=" * 60)
    print(f"  Input device:    {args.input_device} (BL-2021USB)")
    print(f"  Output device:   {args.output_device} (Mac mini Speakers)")
    print(f"  Total duration:  {args.duration}s")
    print(f"  Room change at:  {room_change_s}s")
    print(f"  M4 scorer:       fine-tuned (live domain)")
    print(f"  Room filter:     production (white noise calibrated)")
    print()

    # Create TTS audio
    tts_pcm = make_tts_pcm(duration_s=args.duration)
    print(f"  TTS audio: {len(tts_pcm)} samples ({len(tts_pcm)/16000:.1f}s)")

    # Set up M7 components
    cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
    drift_alarm = DriftAlarm(window_frames=20, spike_threshold_db=6.0)
    ir_bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
    device_bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)

    # Attach to cerebellum
    cerebellum.set_drift_alarm(drift_alarm)
    cerebellum.set_ir_bank(ir_bank)
    cerebellum.set_device_profile_bank(device_bank)

    # Set up SelfModel
    sag = SemanticAbstractionGraph()
    self_model = SelfModel()
    self_model.initialize_sag(sag)

    # Configure streaming runtime
    config = StreamingConfig(
        tts_pcm=tts_pcm,
        tts_text="M7 drift validation",
        input_device=args.input_device,
        output_device=args.output_device,
        m4_enable=True,
        m4_scorer_path="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt",
        m4_self_threshold=0.50,
        gated_playback=False,
        room_filter_path="data/m3_broca_room_filter_production.npy",
        room_local_threshold=0.0747,
        enable_room_filter_adaptation=True,
        room_adapter=cerebellum,
        room_adapt_confidence_threshold=0.10,
        room_adapt_match_threshold=0.10,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)
    runtime._self_model = self_model

    # ── Phase 1: Baseline ──
    print(f"\n[Phase 1] Baseline — keep mic in current position")
    print(f"  Starting in 3 seconds...")
    time.sleep(3)

    # Run with room change prompt injected into the callback
    room_change_done = [False]
    baseline_adaptations = [0]
    post_change_adaptations = [0]

    print(f"\n  >>> RUNNING — keep mic still for {room_change_s}s <<<")
    print(f"  >>> Then MOVE MIC to different room/position <<<\n")

    t_start = time.time()
    try:
        logs = runtime.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        logs = runtime._logs

    elapsed = time.time() - t_start

    # Split logs into pre/post room change
    pre_change = [l for l in logs if l.timestamp_s < room_change_s]
    post_change = [l for l in logs if l.timestamp_s >= room_change_s]

    print(f"\n  Completed: {len(logs)} frames in {elapsed:.1f}s")
    print(f"  Pre-change:  {len(pre_change)} frames (0–{room_change_s}s)")
    print(f"  Post-change: {len(post_change)} frames ({room_change_s}s–{elapsed:.1f}s)")

    # ── Phase 2: Analysis ──
    print(f"\n[Phase 2] Analysis...")

    def analyze_phase(phase_logs, phase_name):
        if not phase_logs:
            print(f"\n  {phase_name}: no data")
            return
        adapted = sum(1 for l in phase_logs if l.room_adapted)
        verdicts = {}
        for l in phase_logs:
            v = l.verdict
            verdicts[v] = verdicts.get(v, 0) + 1
        print(f"\n  {phase_name} ({len(phase_logs)} frames):")
        print(f"    Adapted: {adapted}")
        print(f"    Verdicts: {verdicts}")

    analyze_phase(pre_change, "Pre-change (baseline)")
    analyze_phase(post_change, "Post-change (drift)")

    # Drift alarm stats
    print(f"\n  Drift alarm:")
    if drift_alarm._baseline_set:
        print(f"    Baseline: {drift_alarm._baseline_error_db:.2f} dB")
        print(f"    Status:   {drift_alarm.status.value}")
        print(f"    Spikes:   {drift_alarm._spike_count}")
        print(f"    Rate mult: {drift_alarm.adaptation_rate_multiplier:.2f}x")
    else:
        print(f"    Baseline not set (insufficient SELF frames)")

    print(f"  IR bank: {ir_bank.num_hypotheses} hypotheses")
    print(f"  Device profiles: {device_bank.num_profiles}")

    # Overall verdict distribution
    all_verdicts = {}
    for l in logs:
        v = l.verdict
        all_verdicts[v] = all_verdicts.get(v, 0) + 1
    print(f"  Overall verdicts: {all_verdicts}")

    # Save results
    total_adapted = sum(1 for l in logs if l.room_adapted)
    results = {
        "protocol": "room_change",
        "total_duration_s": round(elapsed, 1),
        "room_change_s": room_change_s,
        "total_frames": len(logs),
        "adapted_frames": total_adapted,
        "pre_change_frames": len(pre_change),
        "post_change_frames": len(post_change),
        "pre_change_adapted": sum(1 for l in pre_change if l.room_adapted),
        "post_change_adapted": sum(1 for l in post_change if l.room_adapted),
        "drift_alarm_status": drift_alarm.status.value if drift_alarm._baseline_set else "not_set",
        "drift_alarm_baseline_db": drift_alarm._baseline_error_db if drift_alarm._baseline_set else None,
        "drift_alarm_spike_count": drift_alarm._spike_count,
        "drift_alarm_multiplier": drift_alarm.adaptation_rate_multiplier,
        "ir_bank_hypotheses": ir_bank.num_hypotheses,
        "device_profiles": device_bank.num_profiles,
        "verdict_counts": all_verdicts,
    }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
