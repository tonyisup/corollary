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
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration per phase in seconds (default: 15)")
    parser.add_argument("--phases", type=int, default=2,
                        help="Number of phases (default: 2 = baseline + post-change)")
    parser.add_argument("--transition-s", type=float, default=10.0,
                        help="Seconds between phases for room change (default: 10)")
    parser.add_argument("--output", type=str, default="/tmp/m7_live_drift.json")
    args = parser.parse_args()

    n_phases = max(2, args.phases)
    phase_duration = args.duration
    transition_s = args.transition_s

    print("=" * 60)
    print(f"M7 Live Drift Validation — {n_phases}-Phase Room Change Protocol")
    print("=" * 60)
    print(f"  Input device:    {args.input_device} (BL-2021USB)")
    print(f"  Output device:   {args.output_device} (Mac mini Speakers)")
    print(f"  Phases:          {n_phases}")
    print(f"  Phase duration:  {phase_duration}s each")
    print(f"  Transition:      {transition_s}s between phases")
    print(f"  M4 scorer:       fine-tuned (live domain)")
    print(f"  Room filter:     production (white noise calibrated)")
    print()

    # Timing
    # Create TTS audio (cover all phases)
    total_duration = phase_duration * n_phases
    tts_pcm = make_tts_pcm(duration_s=total_duration)
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

    # Disable file saving during runs; we'll save combined logs at the end
    config.save_path = None

    # ── Countdown ──
    print(f"\n  Starting in 3 seconds...")
    time.sleep(3)

    # ── Run N phases ──
    all_logs = []
    phase_results = []
    t_global_start = time.time()

    for phase_idx in range(n_phases):
        phase_name = "Baseline" if phase_idx == 0 else f"Phase {phase_idx + 1}"
        print(f"\n{'='*60}")
        print(f"  [{phase_name}] — Running for {phase_duration}s")
        if phase_idx == 0:
            print(f"  Keep mic in current position")
        else:
            print(f"  Mic should be in new position")
        print(f"{'='*60}")

        t_phase_start = time.time()
        try:
            runtime.run(duration_s=phase_duration)
        except KeyboardInterrupt:
            print("\n  Interrupted by user")
        phase_logs = list(runtime._logs)
        t_phase_end = time.time()

        # Adjust timestamps to be global
        offset = t_phase_start - t_global_start
        for l in phase_logs:
            l.timestamp_s += offset

        all_logs.extend(phase_logs)
        phase_summary = runtime.get_summary()
        phase_results.append({
            "name": phase_name,
            "frames": len(phase_logs),
            "duration_s": round(t_phase_end - t_phase_start, 1),
            "adapted": sum(1 for l in phase_logs if l.room_adapted),
            "verdicts": dict(__import__('collections').Counter(l.verdict for l in phase_logs)),
            "live_callback_count": phase_summary.get("live_callback_count", 0),
            "analysis_blocks_enqueued": phase_summary.get("analysis_blocks_enqueued", 0),
            "analysis_blocks_dropped": phase_summary.get("analysis_blocks_dropped", 0),
            "analysis_queue_max_depth": phase_summary.get("analysis_queue_max_depth", 0),
            "analysis_worker_errors": phase_summary.get("analysis_worker_errors", []),
        })

        print(f"  {phase_name}: {len(phase_logs)} frames, {phase_results[-1]['adapted']} adapted")

        # Transition between phases (except after last)
        if phase_idx < n_phases - 1:
            print(f"\n{'='*60}")
            print(f"  >>> MOVE MIC TO DIFFERENT ROOM NOW <<<")
            print(f"  >>> Waiting {transition_s}s... <<<")
            print(f"{'='*60}")
            # Audible beep
            try:
                import os
                for _ in range(3):
                    os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null')
                    time.sleep(0.5)
            except Exception:
                pass
            time.sleep(transition_s)
            # Reset drift alarm for new phase
            drift_alarm.reset_baseline()

    elapsed_total = time.time() - t_global_start

    print(f"\n  Total: {len(all_logs)} frames in {elapsed_total:.1f}s")

    # ── Analysis ──
    print(f"\n[Analysis]")

    for pr in phase_results:
        print(f"\n  {pr['name']} ({pr['frames']} frames, {pr['duration_s']}s):")
        print(f"    Adapted: {pr['adapted']}")
        print(f"    Verdicts: {pr['verdicts']}")

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
    all_verdicts = dict(__import__('collections').Counter(l.verdict for l in all_logs))
    print(f"  Overall verdicts: {all_verdicts}")

    # Save results
    total_adapted = sum(1 for l in all_logs if l.room_adapted)
    total_analysis_blocks_enqueued = sum(pr.get("analysis_blocks_enqueued", 0) for pr in phase_results)
    total_analysis_blocks_dropped = sum(pr.get("analysis_blocks_dropped", 0) for pr in phase_results)
    all_worker_errors = [
        err
        for pr in phase_results
        for err in pr.get("analysis_worker_errors", [])
    ]
    results = {
        "protocol": "room_change_multi",
        "n_phases": n_phases,
        "phase_duration_s": phase_duration,
        "transition_s": transition_s,
        "total_duration_s": round(elapsed_total, 1),
        "total_frames": len(all_logs),
        "adapted_frames": total_adapted,
        "phases": phase_results,
        "drift_alarm_status": drift_alarm.status.value if drift_alarm._baseline_set else "not_set",
        "drift_alarm_baseline_db": drift_alarm._baseline_error_db if drift_alarm._baseline_set else None,
        "drift_alarm_spike_count": drift_alarm._spike_count,
        "drift_alarm_multiplier": drift_alarm.adaptation_rate_multiplier,
        "ir_bank_hypotheses": ir_bank.num_hypotheses,
        "device_profiles": device_bank.num_profiles,
        "verdict_counts": all_verdicts,
        "analysis_blocks_enqueued": total_analysis_blocks_enqueued,
        "analysis_blocks_dropped": total_analysis_blocks_dropped,
        "analysis_queue_max_depth": max(
            (pr.get("analysis_queue_max_depth", 0) for pr in phase_results),
            default=0,
        ),
        "analysis_worker_errors": all_worker_errors,
    }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {args.output}")

    # Save combined JSONL
    jsonl_path = Path(args.output).with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for l in all_logs:
            f.write(json.dumps(runtime._log_to_dict(l)) + "\n")
    print(f"  Frame logs saved to {jsonl_path}")
    print("Done.")


if __name__ == "__main__":
    main()
