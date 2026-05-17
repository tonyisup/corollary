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


def make_tts_pcm(text: str = "Hello, I am Corollary. This is a live drift validation test.", duration_s: float = 7.0) -> np.ndarray:
    """Generate a simple TTS-like signal for testing."""
    SR = 16000
    n_samples = int(duration_s * SR)
    t = np.arange(n_samples, dtype=np.float32) / SR

    # Multi-harmonic signal simulating TTS
    f0 = 180.0
    signal = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 6):
        amp = 1.0 / h * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))  # amplitude modulation
        signal += amp * np.sin(2 * np.pi * f0 * h * t)

    # Envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    signal *= envelope
    signal = signal / max(np.max(np.abs(signal)), 1e-8) * 0.5

    return signal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Live Drift Validation")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=str, default="/tmp/m7_live_drift.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M7 Live Drift Validation")
    print("=" * 60)
    print(f"  Input device:  {args.input_device}")
    print(f"  Output device: {args.output_device}")
    print(f"  Duration:      {args.duration}s")
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
        gated_playback=False,  # Always play TTS
        enable_room_filter_adaptation=True,
        room_adapter=cerebellum,
        room_adapt_confidence_threshold=0.30,  # Adapt on confident SELF frames
        room_adapt_match_threshold=0.10,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)
    runtime._self_model = self_model

    # Phase 1: Bootstrap
    print("\n[Phase 1] Bootstrap — playing TTS and capturing...")
    print("  Starting in 3 seconds...")
    time.sleep(3)

    # Run the streaming loop
    print(f"  Running for {args.duration}s...")
    print("  >>> CHANGE THE ROOM SETUP NOW (move mic, change orientation, etc.) <<<")
    print()

    try:
        logs = runtime.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        logs = runtime._logs

    print(f"\n  Collected {len(logs)} frame logs")

    # Phase 2: Analyze results
    print("\n[Phase 2] Analysis...")

    # Room adaptation stats
    adapted_frames = sum(1 for l in logs if l.room_adapted)
    total_frames = len(logs)
    print(f"  Total frames: {total_frames}")
    print(f"  Room-adapted frames: {adapted_frames}")

    # Drift alarm stats
    if drift_alarm._baseline_set:
        print(f"  Drift alarm baseline: {drift_alarm._baseline_error_db:.2f} dB")
        print(f"  Drift alarm status: {drift_alarm.status.value}")
        print(f"  Spike count: {drift_alarm._spike_count}")
        print(f"  Adaptation rate multiplier: {drift_alarm.adaptation_rate_multiplier:.2f}")
    else:
        print("  Drift alarm: baseline not set (insufficient data)")

    # IR bank stats
    print(f"  IR bank hypotheses: {ir_bank.num_hypotheses}")
    print(f"  Best hypothesis index: {ir_bank.best_hypothesis_index}")
    print(f"  Best hypothesis age: {ir_bank.best_hypothesis.age}")

    # Device profile stats
    print(f"  Device profiles: {device_bank.num_profiles}")
    print(f"  Active orientation: {device_bank.active_orientation.value}")

    # Verdict distribution
    verdict_counts = {}
    for l in logs:
        v = l.verdict
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"  Verdict distribution: {verdict_counts}")

    # Save results
    results = {
        "total_frames": total_frames,
        "adapted_frames": adapted_frames,
        "drift_alarm_status": drift_alarm.status.value if drift_alarm._baseline_set else "not_set",
        "drift_alarm_baseline_db": drift_alarm._baseline_error_db if drift_alarm._baseline_set else None,
        "drift_alarm_spike_count": drift_alarm._spike_count,
        "drift_alarm_multiplier": drift_alarm.adaptation_rate_multiplier,
        "ir_bank_hypotheses": ir_bank.num_hypotheses,
        "ir_bank_best_index": ir_bank.best_hypothesis_index,
        "device_profiles": device_bank.num_profiles,
        "device_active_orientation": device_bank.active_orientation.value,
        "verdict_counts": verdict_counts,
    }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
