#!/usr/bin/env python3
"""Run one bounded M1 evaluation and emit a JSON result.

Initial implementation focus:
- M1B synthetic babbling / room-learning evaluation

The scaffold is stage-aware, but only M1B has a concrete synthetic runner today.
M1A and M1C can be wired in once their experiment harnesses exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from cerebellum.adaptive_model import BabblingCerebellum
from thalamus.gate import compute_svsr, spectral_subtraction
from cochlea.capture import pcm_to_mels, SR

sys.path.insert(0, os.path.dirname(__file__))
from m1_score import score_stage
from m1_tone_attribution import M1AConfig, evaluate_m1a
from m1_mixed_attribution import M1CConfig, evaluate_m1c


@dataclass
class EvalConfig:
    stage: str
    scenario: str
    iterations: int = 10
    ir_length_ms: int = 300
    nlms_mu: float = 0.25
    noise_db: float = -35.0
    playback_amplitude: float = 0.7
    shift_min_frames: int = -10
    shift_max_frames: int = 40
    stimulus: str = "tone"
    stimulus_duration_ms: int = 300
    room_rt60: float = 0.3
    attribution_threshold: float = 0.55
    event_threshold: float = 0.08
    onset_jitter_ms: int = 8
    transformed_world_gain: float = 0.85
    replay_gain: float = 0.9
    hard_negative_noise_db_offset: float = 5.0
    hard_negative_amplitude_jitter: float = 0.15
    causal_tolerance_ms: float = 20.0
    causal_weight: float = 0.5
    similarity_weight: float = 0.5
    stimulus_seed: int = 0


DEFAULT_GUARDRAILS: dict[str, dict[str, Any]] = {
    "M1A": {
        "max_false_self_rate": 0.05,
        "max_replay_false_self_rate": 0.10,
        "max_onset_error_ms": 50.0,
        "min_causal_self_score_mean": 0.45,
        "max_combined_negative_score_mean": 0.45,
    },
    "M1B": {
        "min_attribution_gain": 0.0,
        "min_convergence_slope": 0.0,
        "max_prediction_error_final_db": -5.0,
    },
    "M1C": {
        "max_world_to_self_confusion": 0.10,
        "max_replay_false_self_rate": 0.05,
    },
}


def generate_stimulus(kind: str, duration_ms: int, amplitude: float) -> np.ndarray:
    n = max(1, int(SR * duration_ms / 1000))
    t = np.arange(n, dtype=np.float32) / SR
    ramp_len = max(1, int(0.01 * SR))
    ramp = np.ones(n, dtype=np.float32)
    if 2 * ramp_len < n:
        in_ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        out_ramp = np.linspace(1.0, 0.0, ramp_len, dtype=np.float32)
        ramp[:ramp_len] = in_ramp
        ramp[-ramp_len:] = out_ramp

    if kind == "tone":
        pcm = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    elif kind == "click":
        pcm = np.zeros(n, dtype=np.float32)
        click_positions = [0, n // 3, (2 * n) // 3]
        for pos in click_positions:
            pcm[pos:pos + 4] = 1.0
    elif kind == "chirp":
        f0, f1 = 300.0, 1800.0
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / max(t[-1], 1e-6))
        pcm = np.sin(phase).astype(np.float32)
    else:
        raise ValueError(f"Unsupported stimulus kind: {kind}")

    pcm *= ramp
    peak = float(np.max(np.abs(pcm)))
    if peak > 0:
        pcm = pcm / peak
    return (pcm * amplitude).astype(np.float32)


def generate_room_ir(rt60: float) -> np.ndarray:
    length = max(64, int(rt60 * SR))
    ir = np.zeros(length, dtype=np.float32)
    ir[0] = 1.0
    reflections = [(16, 0.4), (48, 0.25), (96, 0.15), (200, 0.08)]
    for idx, gain in reflections:
        if idx < length:
            ir[idx] += gain
    start = int(0.03 * SR)
    for i in range(start, length):
        decay = np.exp(-6.91 * max(i - start, 0) / max(length - start, 1))
        ir[i] += np.random.randn() * decay * 0.08
    energy = np.sum(ir ** 2)
    if energy > 0:
        ir /= np.sqrt(energy)
    return ir


def add_noise(audio: np.ndarray, noise_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    noise_amp = rms * (10 ** (noise_db / 20.0)) if noise_db > -120 else 0.0
    noise = rng.standard_normal(len(audio)).astype(np.float32) * noise_amp
    return (audio + noise).astype(np.float32)


def best_svsr(observed_mel: np.ndarray, predicted_mel: np.ndarray, shift_min: int, shift_max: int) -> float:
    t_obs = observed_mel.shape[0]
    t_pred = predicted_mel.shape[0]
    best = 0.0
    for shift in range(shift_min, shift_max + 1):
        if shift >= 0:
            shifted = np.concatenate([
                np.tile(predicted_mel[:1], (shift, 1)),
                predicted_mel,
            ], axis=0)[:t_obs]
        else:
            shifted = predicted_mel[-shift:][:t_obs] if -shift < t_pred else np.zeros_like(predicted_mel[:t_obs])
            if len(shifted) < t_obs:
                shifted = np.concatenate([
                    shifted,
                    np.zeros((t_obs - len(shifted), predicted_mel.shape[1]), dtype=np.float32),
                ])
        obs_trim = observed_mel[:len(shifted)]
        residual, _ = spectral_subtraction(obs_trim, shifted)
        svsr = compute_svsr(obs_trim, shifted, residual)
        if svsr > best:
            best = float(svsr)
    return best


def evaluate_m1b(config: EvalConfig) -> dict[str, Any]:
    known_pcm = generate_stimulus(config.stimulus, config.stimulus_duration_ms, config.playback_amplitude)
    true_ir = generate_room_ir(config.room_rt60)
    captured_clean = fftconvolve(known_pcm, true_ir, mode="full").astype(np.float32)
    captured = add_noise(captured_clean, config.noise_db, seed=42)

    cerebellum = BabblingCerebellum(ir_length_ms=config.ir_length_ms, nlms_mu=config.nlms_mu)

    svsr_history: list[float] = []
    error_history_db: list[float] = []
    trust_history: list[float] = []

    observed_mel = pcm_to_mels(captured)
    for _ in range(config.iterations):
        cerebellum.update(known_pcm, captured)
        predicted_mel = cerebellum.predict_mel(known_pcm)
        svsr_history.append(best_svsr(observed_mel, predicted_mel, config.shift_min_frames, config.shift_max_frames))
        conv = cerebellum.get_convergence()
        error_history_db.append(float(conv["current_error_db"]))
        trust_history.append(float(cerebellum.trust_status()["trust"]))

    first_svsr = svsr_history[0] if svsr_history else 0.0
    final_svsr = svsr_history[-1] if svsr_history else 0.0
    attribution_gain = final_svsr - first_svsr
    slope = attribution_gain / max(config.iterations - 1, 1)

    metrics = {
        "svsr_initial_db": round(first_svsr, 4),
        "svsr_final_db": round(final_svsr, 4),
        "attribution_gain": round(attribution_gain / 20.0, 6),
        "convergence_slope": round(slope / 20.0, 6),
        "prediction_error_final_db": round(error_history_db[-1] if error_history_db else 0.0, 4),
        "prediction_error_initial_db": round(error_history_db[0] if error_history_db else 0.0, 4),
        "trust_final": round(trust_history[-1] if trust_history else 0.0, 6),
        "iterations": config.iterations,
    }
    return metrics


def run_eval(config: EvalConfig) -> dict[str, Any]:
    stage = config.stage.upper()
    if stage == "M1B":
        metrics = evaluate_m1b(config)
    elif stage == "M1A":
        metrics = evaluate_m1a(M1AConfig(
            stimulus=config.stimulus,
            stimulus_duration_ms=config.stimulus_duration_ms,
            playback_amplitude=config.playback_amplitude,
            room_rt60=config.room_rt60,
            noise_db=config.noise_db,
            shift_min_frames=config.shift_min_frames,
            shift_max_frames=config.shift_max_frames,
            attribution_threshold=config.attribution_threshold,
            event_threshold=config.event_threshold,
            iterations=config.iterations,
            onset_jitter_ms=config.onset_jitter_ms,
            transformed_world_gain=config.transformed_world_gain,
            replay_gain=config.replay_gain,
            hard_negative_noise_db_offset=config.hard_negative_noise_db_offset,
            hard_negative_amplitude_jitter=config.hard_negative_amplitude_jitter,
            causal_tolerance_ms=config.causal_tolerance_ms,
            causal_weight=config.causal_weight,
            similarity_weight=config.similarity_weight,
            stimulus_seed=config.stimulus_seed,
        ))
    elif stage == "M1C":
        metrics = evaluate_m1c(M1CConfig(
            stimulus=config.stimulus,
            stimulus_duration_ms=config.stimulus_duration_ms,
            playback_amplitude=config.playback_amplitude,
            room_rt60=config.room_rt60,
            noise_db=config.noise_db,
            iterations=config.iterations,
            attribution_threshold=config.attribution_threshold,
            transformed_world_gain=config.transformed_world_gain,
            replay_gain=config.replay_gain,
        ))
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    scoring = score_stage(stage, metrics, DEFAULT_GUARDRAILS.get(stage))
    return {
        "stage": stage,
        "scenario": config.scenario,
        "config": asdict(config),
        "metrics": metrics,
        **scoring,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one M1 evaluation and emit JSON")
    parser.add_argument("--stage", default="M1B")
    parser.add_argument("--scenario", default="fixed_room_convergence")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ir-length-ms", type=int, default=300)
    parser.add_argument("--nlms-mu", type=float, default=0.25)
    parser.add_argument("--noise-db", type=float, default=-35.0)
    parser.add_argument("--playback-amplitude", type=float, default=0.7)
    parser.add_argument("--shift-min-frames", type=int, default=-10)
    parser.add_argument("--shift-max-frames", type=int, default=40)
    parser.add_argument("--stimulus", default="tone", choices=["tone", "click", "chirp", "vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield"])
    parser.add_argument("--stimulus-duration-ms", type=int, default=300)
    parser.add_argument("--room-rt60", type=float, default=0.3)
    parser.add_argument("--attribution-threshold", type=float, default=0.55)
    parser.add_argument("--event-threshold", type=float, default=0.08)
    parser.add_argument("--onset-jitter-ms", type=int, default=8)
    parser.add_argument("--transformed-world-gain", type=float, default=0.85)
    parser.add_argument("--replay-gain", type=float, default=0.9)
    parser.add_argument("--hard-negative-noise-db-offset", type=float, default=5.0)
    parser.add_argument("--hard-negative-amplitude-jitter", type=float, default=0.15)
    parser.add_argument("--causal-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--causal-weight", type=float, default=0.5)
    parser.add_argument("--similarity-weight", type=float, default=0.5)
    parser.add_argument("--stimulus-seed", type=int, default=0)
    parser.add_argument("--output", default="-")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        stage=args.stage,
        scenario=args.scenario,
        iterations=args.iterations,
        ir_length_ms=args.ir_length_ms,
        nlms_mu=args.nlms_mu,
        noise_db=args.noise_db,
        playback_amplitude=args.playback_amplitude,
        shift_min_frames=args.shift_min_frames,
        shift_max_frames=args.shift_max_frames,
        stimulus=args.stimulus,
        stimulus_duration_ms=args.stimulus_duration_ms,
        room_rt60=args.room_rt60,
        attribution_threshold=args.attribution_threshold,
        event_threshold=args.event_threshold,
        onset_jitter_ms=args.onset_jitter_ms,
        transformed_world_gain=args.transformed_world_gain,
        replay_gain=args.replay_gain,
        hard_negative_noise_db_offset=args.hard_negative_noise_db_offset,
        hard_negative_amplitude_jitter=args.hard_negative_amplitude_jitter,
        causal_tolerance_ms=args.causal_tolerance_ms,
        causal_weight=args.causal_weight,
        similarity_weight=args.similarity_weight,
        stimulus_seed=args.stimulus_seed,
    )

    started_at = time.time()
    result = run_eval(config)
    result["run_started_at"] = started_at
    result["run_finished_at"] = time.time()
    result["run_duration_s"] = round(result["run_finished_at"] - started_at, 4)

    payload = json.dumps(result, sort_keys=True)
    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
