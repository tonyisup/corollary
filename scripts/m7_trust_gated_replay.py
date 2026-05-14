#!/usr/bin/env python3
"""M7 deterministic trust-gated room adaptation replay.

This script validates the phase-1 M7 invariant without hardware:
- learn an initial room A forward model
- simulate a room B acoustic drift
- present a mixed sequence of trusted SELF, WORLD, and BOTH frames
- allow NLMS updates only on trusted SELF frames
- verify room-B prediction error improves while WORLD/BOTH produce zero updates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cerebellum.adaptive_model import BabblingCerebellum


SR = 16000


def _room_ir(kind: str, taps: int = 1600) -> np.ndarray:
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
    else:
        raise ValueError(f"unknown room IR kind: {kind}")
    return ir


def _speech_like(seed: int, duration_s: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SR), dtype=np.float32) / SR
    freqs = rng.choice(np.array([160, 210, 280, 360, 520, 720, 950], dtype=np.float32), size=4, replace=False)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    sig = sum(np.sin(2.0 * np.pi * f * t + p) for f, p in zip(freqs, phases))
    sig = sig.astype(np.float32)
    # Slow amplitude envelope gives the adaptive filter varied but stable input.
    env = 0.45 + 0.55 * np.sin(np.pi * np.linspace(0.0, 1.0, len(sig), dtype=np.float32))
    sig = sig * env
    sig = sig / max(float(np.max(np.abs(sig))), 1e-8)
    return (0.5 * sig).astype(np.float32)


def _capture(clean: np.ndarray, ir: np.ndarray, *, noise_scale: float = 0.002, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = fftconvolve(clean, ir, mode="full").astype(np.float32)
    y += rng.normal(0.0, noise_scale, size=len(y)).astype(np.float32)
    return y


def _prediction_mse(cerebellum: BabblingCerebellum, clean: np.ndarray, captured: np.ndarray) -> float:
    pred = cerebellum.predict_pcm(clean).astype(np.float32)
    n = min(len(pred), len(captured))
    return float(np.mean((captured[:n] - pred[:n]) ** 2))


def _run_room_a_bootstrap(cerebellum: BabblingCerebellum, room_a: np.ndarray, seed: int) -> None:
    for i in range(10):
        clean = _speech_like(seed + i)
        captured = _capture(clean, room_a, seed=seed + 100 + i)
        cerebellum.update(clean, captured)


def run_replay(
    *,
    seed: int = 13,
    adaptation_cycles: int = 8,
    min_improvement_ratio: float = 0.20,
) -> dict[str, Any]:
    room_a = _room_ir("a")
    room_b = _room_ir("b")
    cerebellum = BabblingCerebellum(ir_length_ms=100, sample_rate=SR, nlms_mu=0.1)

    _run_room_a_bootstrap(cerebellum, room_a, seed)

    heldout = _speech_like(seed + 1000)
    heldout_capture_b = _capture(heldout, room_b, seed=seed + 2000)
    room_b_pre_error = _prediction_mse(cerebellum, heldout, heldout_capture_b)

    events: list[dict[str, Any]] = []
    trusted_self_updates = 0
    held_world_updates = 0
    held_both_updates = 0

    for cycle in range(adaptation_cycles):
        clean = _speech_like(seed + 3000 + cycle)
        captured_self_b = _capture(clean, room_b, seed=seed + 4000 + cycle)

        before = cerebellum.total_updates
        self_result = cerebellum.update_trust_gated(
            clean,
            captured_self_b,
            verdict="SELF",
            m4_confidence=0.93,
            self_match_score=0.55,
            adapt_confidence_threshold=0.8,
            adapt_match_threshold=0.2,
        )
        after = cerebellum.total_updates
        trusted_self_updates += after - before
        events.append({"cycle": cycle, "kind": "SELF", **self_result})

        world = _speech_like(seed + 5000 + cycle)
        captured_world_b = _capture(world, room_b, seed=seed + 6000 + cycle)
        before = cerebellum.total_updates
        world_result = cerebellum.update_trust_gated(
            clean,
            captured_world_b,
            verdict="WORLD",
            m4_confidence=0.99,
            self_match_score=0.99,
        )
        after = cerebellum.total_updates
        held_world_updates += after - before
        events.append({"cycle": cycle, "kind": "WORLD", **world_result})

        # BOTH contains self plus another speaker. The verdict, not the high
        # scores, must keep adaptation frozen.
        both_len = min(len(captured_self_b), len(captured_world_b))
        captured_both = captured_self_b[:both_len] + 0.7 * captured_world_b[:both_len]
        before = cerebellum.total_updates
        both_result = cerebellum.update_trust_gated(
            clean,
            captured_both.astype(np.float32),
            verdict="BOTH",
            m4_confidence=0.99,
            self_match_score=0.99,
        )
        after = cerebellum.total_updates
        held_both_updates += after - before
        events.append({"cycle": cycle, "kind": "BOTH", **both_result})

    room_b_post_error = _prediction_mse(cerebellum, heldout, heldout_capture_b)
    improvement_ratio = (room_b_pre_error - room_b_post_error) / max(room_b_pre_error, 1e-12)
    passed = (
        trusted_self_updates > 0
        and held_world_updates == 0
        and held_both_updates == 0
        and improvement_ratio >= min_improvement_ratio
    )

    return {
        "config": {
            "seed": seed,
            "adaptation_cycles": adaptation_cycles,
            "min_improvement_ratio": min_improvement_ratio,
            "sample_rate": SR,
        },
        "summary": {
            "passed": bool(passed),
            "room_b_pre_error": room_b_pre_error,
            "room_b_post_error": room_b_post_error,
            "improvement_ratio": improvement_ratio,
            "trusted_self_updates": int(trusted_self_updates),
            "held_world_updates": int(held_world_updates),
            "held_both_updates": int(held_both_updates),
            "total_updates": int(cerebellum.total_updates),
        },
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 deterministic trust-gated room adaptation replay")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--adaptation-cycles", type=int, default=8)
    parser.add_argument("--min-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("/tmp/m7_trust_gated_replay.json"))
    args = parser.parse_args()

    result = run_replay(
        seed=args.seed,
        adaptation_cycles=args.adaptation_cycles,
        min_improvement_ratio=args.min_improvement_ratio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    summary = result["summary"]
    print("M7 trust-gated replay")
    print(f"  passed: {summary['passed']}")
    print(f"  room_b_pre_error:  {summary['room_b_pre_error']:.8f}")
    print(f"  room_b_post_error: {summary['room_b_post_error']:.8f}")
    print(f"  improvement_ratio: {summary['improvement_ratio']:.3f}")
    print(f"  trusted_self_updates: {summary['trusted_self_updates']}")
    print(f"  held_world_updates:   {summary['held_world_updates']}")
    print(f"  held_both_updates:    {summary['held_both_updates']}")
    print(f"  wrote: {args.output}")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
