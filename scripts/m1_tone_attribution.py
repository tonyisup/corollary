#!/usr/bin/env python3
"""Synthetic M1A self-sound attribution harness.

This is intentionally pre-linguistic and simple:
- generate a known self stimulus
- simulate room coloration + delay
- compare against negative-control world stimuli / silence
- score self/not-self attribution and onset alignment
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR


@dataclass
class M1AConfig:
    stimulus: str = "tone"
    stimulus_duration_ms: int = 300
    playback_amplitude: float = 0.7
    room_rt60: float = 0.3
    noise_db: float = -35.0
    shift_min_frames: int = -10
    shift_max_frames: int = 40
    attribution_threshold: float = 0.55
    event_threshold: float = 0.08
    iterations: int = 10


def generate_stimulus(kind: str, duration_ms: int, amplitude: float) -> np.ndarray:
    n = max(1, int(SR * duration_ms / 1000))
    t = np.arange(n, dtype=np.float32) / SR
    ramp_len = max(1, int(0.01 * SR))
    ramp = np.ones(n, dtype=np.float32)
    if 2 * ramp_len < n:
        ramp[:ramp_len] = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        ramp[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len, dtype=np.float32)

    if kind == "tone":
        pcm = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    elif kind == "click":
        pcm = np.zeros(n, dtype=np.float32)
        click_positions = [0, n // 3, (2 * n) // 3]
        for pos in click_positions:
            pcm[pos:min(n, pos + 4)] = 1.0
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


WORLD_VARIANTS = {
    "tone": "chirp",
    "chirp": "click",
    "click": "tone",
}


def generate_room_ir(rt60: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
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
        ir[i] += rng.standard_normal() * decay * 0.08
    energy = np.sum(ir ** 2)
    if energy > 0:
        ir /= np.sqrt(energy)
    return ir


def add_noise(audio: np.ndarray, noise_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    noise_amp = rms * (10 ** (noise_db / 20.0)) if noise_db > -120 else 0.0
    return (audio + rng.standard_normal(len(audio)).astype(np.float32) * noise_amp).astype(np.float32)


def delayed_room_capture(pcm: np.ndarray, ir: np.ndarray, delay_samples: int, noise_db: float, seed: int) -> np.ndarray:
    roomed = fftconvolve(pcm, ir, mode="full").astype(np.float32)
    captured = np.concatenate([np.zeros(delay_samples, dtype=np.float32), roomed])
    return add_noise(captured, noise_db, seed=seed)


def event_present(audio: np.ndarray, threshold: float) -> bool:
    if len(audio) == 0:
        return False
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    return peak >= threshold or rms >= threshold * 0.5


def best_match_score(observed: np.ndarray, template: np.ndarray, shift_min_samples: int, shift_max_samples: int) -> tuple[float, int]:
    template_energy = float(np.linalg.norm(template))
    if template_energy < 1e-9 or len(observed) < len(template):
        return 0.0, 0

    best_score = -1.0
    best_shift = 0
    base_len = len(template)
    for shift in range(max(0, shift_min_samples), max(shift_max_samples + 1, 1)):
        if shift + base_len > len(observed):
            break
        window = observed[shift:shift + base_len]
        denom = float(np.linalg.norm(window) * template_energy)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_m1a(config: M1AConfig) -> dict[str, Any]:
    self_stimulus = generate_stimulus(config.stimulus, config.stimulus_duration_ms, config.playback_amplitude)
    world_kind = WORLD_VARIANTS.get(config.stimulus, "chirp")
    world_stimulus = generate_stimulus(world_kind, config.stimulus_duration_ms, config.playback_amplitude)
    room_ir = generate_room_ir(config.room_rt60, seed=7)

    shift_min_samples = max(0, config.shift_min_frames * 160)
    shift_max_samples = max(shift_min_samples + 1, config.shift_max_frames * 160)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    onset_errors_ms: list[float] = []

    tp = fp = tn = fn = 0

    for i in range(config.iterations):
        delay_ms = 20 + 5 * (i % 5)
        delay_samples = int(delay_ms * SR / 1000)

        captured_self = delayed_room_capture(self_stimulus, room_ir, delay_samples, config.noise_db, seed=100 + i)
        score_self, shift_self = best_match_score(captured_self, self_stimulus, shift_min_samples, shift_max_samples)
        predicted_self = event_present(captured_self, config.event_threshold) and score_self >= config.attribution_threshold
        positive_scores.append(score_self)
        onset_errors_ms.append(abs(shift_self - delay_samples) * 1000.0 / SR)
        if predicted_self:
            tp += 1
        else:
            fn += 1

        captured_world = delayed_room_capture(world_stimulus, room_ir, delay_samples, config.noise_db, seed=200 + i)
        score_world, _ = best_match_score(captured_world, self_stimulus, shift_min_samples, shift_max_samples)
        predicted_world_as_self = event_present(captured_world, config.event_threshold) and score_world >= config.attribution_threshold
        negative_scores.append(score_world)
        if predicted_world_as_self:
            fp += 1
        else:
            tn += 1

        silence = add_noise(np.zeros_like(captured_self), config.noise_db, seed=300 + i)
        score_silence, _ = best_match_score(silence, self_stimulus, shift_min_samples, shift_max_samples)
        predicted_silence_as_self = event_present(silence, config.event_threshold) and score_silence >= config.attribution_threshold
        negative_scores.append(score_silence)
        if predicted_silence_as_self:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    false_self_rate = fp / max(fp + tn, 1)
    self_f1 = compute_f1(tp, fp, fn)

    return {
        "self_f1": round(self_f1, 6),
        "self_precision": round(tp / max(tp + fp, 1), 6),
        "self_recall": round(tp / max(tp + fn, 1), 6),
        "self_accuracy": round(accuracy, 6),
        "false_self_rate": round(false_self_rate, 6),
        "onset_error_ms": round(float(np.mean(onset_errors_ms)) if onset_errors_ms else 0.0, 4),
        "self_score_mean": round(float(np.mean(positive_scores)) if positive_scores else 0.0, 6),
        "world_score_mean": round(float(np.mean(negative_scores)) if negative_scores else 0.0, 6),
        "positive_trials": config.iterations,
        "negative_trials": config.iterations * 2,
    }
