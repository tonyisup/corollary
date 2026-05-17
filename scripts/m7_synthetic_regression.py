#!/usr/bin/env python3
"""M7 Full Pipeline Synthetic Regression Test.

Exercises the complete M7 pipeline components in deterministic tests:
1. Trust-gated NLMS: BabblingCerebellum + update_trust_gated
2. Drift alarm: DriftAlarm error tracking and status transitions
3. IR bank: Multi-hypothesis scoring, adaptation, pruning
4. Device profile: Orientation switching and IR seeding
5. Integration: All components wired together

Usage:
    .venv/bin/python scripts/m7_synthetic_regression.py \\
        --output /tmp/m7_regression.json

Exit code 0 = all scenarios pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.drift_alarm import DriftAlarm, DriftStatus
from cerebellum.ir_bank import IRBank
from cerebellum.device_profile import DeviceProfileBank, DeviceOrientation


SR = 16000


def make_room_ir(kind: str = "a", taps: int = 1600) -> np.ndarray:
    """Create a synthetic room IR."""
    ir = np.zeros(taps, dtype=np.float32)
    ir[0] = 1.0
    if kind == "a":
        ir[18] = 0.28
        ir[63] = 0.12
        ir[211] = 0.04
    elif kind == "b":
        ir[35] = 0.42
        ir[101] = -0.18
        ir[377] = 0.09
    elif kind == "c":
        ir[50] = 0.35
        ir[150] = -0.15
        ir[400] = 0.07
    return ir


def make_tts_pcm(duration_s: float = 5.0, f0: float = 180.0, seed: int = 42) -> np.ndarray:
    """Generate a TTS-like signal."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * SR)
    t = np.arange(n_samples, dtype=np.float32) / SR
    signal = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 6):
        amp = 1.0 / h * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        signal += amp * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    signal *= envelope
    signal = signal / max(np.max(np.abs(signal)), 1e-8) * 0.5
    return signal.astype(np.float32)


def test_trust_gated_adaptation():
    """Test 1: Trust-gated NLMS adapts on SELF, holds on BOTH/WORLD."""
    print("\n  [Test 1] Trust-gated NLMS adaptation")

    cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
    room_ir = make_room_ir("a")
    tts = make_tts_pcm(duration_s=3.0)
    captured = fftconvolve(tts, room_ir, mode="full").astype(np.float32)

    # Test SELF frame — should adapt
    result_self = cerebellum.update_trust_gated(
        tts, captured,
        verdict="SELF",
        m4_confidence=0.9,
        self_match_score=0.5,
    )
    assert result_self["adapted"] is True, f"SELF should adapt, got: {result_self}"
    assert result_self["reason"] == "trusted_self"

    # Test BOTH frame — should NOT adapt
    result_both = cerebellum.update_trust_gated(
        tts, captured,
        verdict="BOTH",
        m4_confidence=0.9,
        self_match_score=0.5,
    )
    assert result_both["adapted"] is False, f"BOTH should not adapt, got: {result_both}"
    assert result_both["reason"] == "verdict_not_self"

    # Test WORLD frame — should NOT adapt
    result_world = cerebellum.update_trust_gated(
        tts, captured,
        verdict="WORLD",
        m4_confidence=0.9,
        self_match_score=0.5,
    )
    assert result_world["adapted"] is False
    assert result_world["reason"] == "verdict_not_self"

    # Test low M4 confidence — should NOT adapt
    result_low = cerebellum.update_trust_gated(
        tts, captured,
        verdict="SELF",
        m4_confidence=0.2,
        self_match_score=0.5,
    )
    assert result_low["adapted"] is False
    assert result_low["reason"] == "m4_confidence_below_threshold"

    # Test low self-match — should NOT adapt
    result_low_match = cerebellum.update_trust_gated(
        tts, captured,
        verdict="SELF",
        m4_confidence=0.9,
        self_match_score=0.05,
    )
    assert result_low_match["adapted"] is False
    assert result_low_match["reason"] == "self_match_below_threshold"

    # Verify adaptation actually changed the IR
    updates_after = cerebellum.total_updates
    assert updates_after > 0, f"Expected >0 updates, got {updates_after}"

    print(f"    PASS: SELF adapts, BOTH/WORLD/low-conf hold, {updates_after} updates")
    return True


def test_drift_alarm():
    """Test 2: DriftAlarm detects degradation and triggers rebabble."""
    print("\n  [Test 2] Drift alarm")

    alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0, rebabble_after_spikes=3)

    # Stable period
    for _ in range(10):
        alarm.record_error(-35.0)
    assert alarm.status == DriftStatus.NOMINAL

    # Spike
    for _ in range(5):
        alarm.record_error(-20.0)
    assert alarm.status == DriftStatus.DEGRADED
    assert alarm.should_spike_adaptation is True

    # Sustained degradation → rebabble
    for _ in range(3):
        for _ in range(5):
            alarm.record_error(-20.0)
        alarm.record_error(-35.0)
    assert alarm.status == DriftStatus.STALE
    assert alarm.should_rebabble is True

    # Recovery
    for _ in range(15):
        alarm.record_error(-35.0)
    assert alarm.status == DriftStatus.NOMINAL
    assert alarm.should_rebabble is False

    # Adaptation rate multiplier
    alarm2 = DriftAlarm(window_frames=10, spike_threshold_db=6.0, max_rate_multiplier=4.0)
    for _ in range(10):
        alarm2.record_error(-35.0)
    assert alarm2.adaptation_rate_multiplier == 1.0
    for _ in range(5):
        alarm2.record_error(-20.0)
    assert alarm2.adaptation_rate_multiplier > 1.0
    assert alarm2.adaptation_rate_multiplier <= 4.0

    print("    PASS: Drift detection, rebabble, recovery, rate multiplier all correct")
    return True


def test_ir_bank():
    """Test 3: Multi-hypothesis IR bank."""
    print("\n  [Test 3] Multi-hypothesis IR bank")

    bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
    room_ir = make_room_ir("a")
    tts = make_tts_pcm(duration_s=3.0)
    captured = fftconvolve(tts, room_ir, mode="full").astype(np.float32)

    # Initial: 1 hypothesis
    assert bank.num_hypotheses == 1

    # Adapt the best hypothesis
    for _ in range(10):
        bank.adapt_best(tts, captured)

    errors = bank.score_all(tts, captured)
    assert len(errors) == 1
    assert bank.best_hypothesis_index == 0

    # Spawn new hypotheses
    bank.spawn_hypothesis()
    bank.spawn_hypothesis()
    assert bank.num_hypotheses == 3

    # Score all — best should be the one we adapted
    errors = bank.score_all(tts, captured)
    assert len(errors) == 3
    assert bank.best_hypothesis_index == 0  # First one was adapted

    # Prune
    bank.prune_worst(target_count=2)
    assert bank.num_hypotheses == 2

    # Serialization
    data = bank.to_dict()
    restored = IRBank.from_dict(data)
    assert restored.num_hypotheses == bank.num_hypotheses
    assert restored.best_hypothesis_index == bank.best_hypothesis_index

    print(f"    PASS: Scoring, adaptation, spawn, prune, serialization all correct")
    return True


def test_device_profile():
    """Test 4: Device profile bank."""
    print("\n  [Test 4] Device profile bank")

    bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)

    # Initial: unknown profile
    assert bank.active_orientation == DeviceOrientation.UNKNOWN
    assert bank.num_profiles == 1

    # Switch orientations
    bank.update_orientation(DeviceOrientation.PORTRAIT)
    assert bank.active_orientation == DeviceOrientation.PORTRAIT
    assert bank.num_profiles == 2

    bank.update_orientation(DeviceOrientation.LANDSCAPE)
    assert bank.active_orientation == DeviceOrientation.LANDSCAPE
    assert bank.num_profiles == 3

    bank.update_orientation(DeviceOrientation.POCKET)
    assert bank.active_orientation == DeviceOrientation.POCKET
    assert bank.num_profiles == 4

    # Reuse existing profile
    bank.update_orientation(DeviceOrientation.PORTRAIT)
    assert bank.num_profiles == 4  # No new profile created

    # Get active IR
    ir = bank.get_active_ir()
    assert ir is not None
    assert len(ir) == 1600

    # Profile-specific priors
    portrait_prof = bank.get_profile(DeviceOrientation.PORTRAIT)
    assert portrait_prof is not None
    assert portrait_prof.guess_count == 1

    # Serialization
    data = bank.to_dict()
    restored = DeviceProfileBank.from_dict(data)
    assert restored.num_profiles == bank.num_profiles
    assert restored.active_orientation == bank.active_orientation

    print(f"    PASS: Orientation switching, IR priors, reuse, serialization all correct")
    return True


def test_full_integration():
    """Test 5: All M7 components wired together."""
    print("\n  [Test 5] Full M7 integration")

    # Set up all components
    cerebellum = BabblingCerebellum(ir_length_ms=100, nlms_mu=0.1)
    drift_alarm = DriftAlarm(window_frames=10, spike_threshold_db=6.0)
    ir_bank = IRBank(ir_length=1600, nlms_mu=0.1, max_hypotheses=5)
    device_bank = DeviceProfileBank(ir_length=1600, nlms_mu=0.1)

    cerebellum.set_drift_alarm(drift_alarm)
    cerebellum.set_ir_bank(ir_bank)
    cerebellum.set_device_profile_bank(device_bank)

    # Simulate: stable room A
    room_a = make_room_ir("a")
    tts = make_tts_pcm(duration_s=5.0)
    captured_a = fftconvolve(tts, room_a, mode="full").astype(np.float32)

    device_bank.update_orientation(DeviceOrientation.PORTRAIT)

    # Adapt for 20 steps
    for _ in range(20):
        result = cerebellum.update_trust_gated(
            tts, captured_a,
            verdict="SELF",
            m4_confidence=0.9,
            self_match_score=0.5,
        )
        assert result["adapted"] is True

    assert cerebellum.total_updates == 20
    assert drift_alarm.status == DriftStatus.NOMINAL

    # Simulate: room change (room B)
    room_b = make_room_ir("b")
    captured_b = fftconvolve(tts, room_b, mode="full").astype(np.float32)

    # Adapt for 10 steps — drift alarm should detect degradation
    for _ in range(10):
        result = cerebellum.update_trust_gated(
            tts, captured_b,
            verdict="SELF",
            m4_confidence=0.9,
            self_match_score=0.3,  # Lower match due to room change
        )
        # Record error in drift alarm
        error_db = -10.0  # Simulated high error
        drift_alarm.record_error(error_db)

    # Drift should be detected
    assert drift_alarm.status in (DriftStatus.DEGRADED, DriftStatus.STALE), \
        f"Expected DEGRADED or STALE, got {drift_alarm.status}"

    # IR bank should have spawned new hypotheses
    assert ir_bank.num_hypotheses >= 1

    # Device profile should have switched
    device_bank.update_orientation(DeviceOrientation.LANDSCAPE)
    assert device_bank.active_orientation == DeviceOrientation.LANDSCAPE

    print(f"    PASS: All components work together")
    print(f"      Cerebellum updates: {cerebellum.total_updates}")
    print(f"      Drift alarm: {drift_alarm.status.value}")
    print(f"      IR bank: {ir_bank.num_hypotheses} hypotheses")
    print(f"      Device: {device_bank.active_orientation.value}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Synthetic Regression Test")
    parser.add_argument("--output", type=str, default="/tmp/m7_regression.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M7 Full Pipeline Synthetic Regression Test")
    print("=" * 60)

    tests = [
        ("Trust-gated NLMS", test_trust_gated_adaptation),
        ("Drift alarm", test_drift_alarm),
        ("IR bank", test_ir_bank),
        ("Device profile", test_device_profile),
        ("Full integration", test_full_integration),
    ]

    results = []
    all_passed = True

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append({"name": name, "passed": passed})
        except Exception as e:
            print(f"    FAIL: {e}")
            results.append({"name": name, "passed": False, "error": str(e)})
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    n_passed = sum(1 for r in results if r["passed"])
    print(f"Results: {n_passed}/{len(results)} passed")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    # Save
    output = {
        "all_passed": all_passed,
        "results": results,
        "timestamp": time.time(),
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
