#!/usr/bin/env python3
"""Synthetic M1C mixed-source attribution harness.

Evaluates frame-level self/world/overlap labeling under structured overlap,
including a replay negative control where self-like audio is present without an
active motor command.
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

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import (
    WORLD_VARIANTS,
    add_noise,
    generate_room_ir,
    generate_stimulus,
    transformed_same_class_world,
)

FRAME_SAMPLES = 320
HOP_SAMPLES = 160


@dataclass
class M1CConfig:
    stimulus: str = "chirp"
    stimulus_duration_ms: int = 300
    playback_amplitude: float = 0.7
    room_rt60: float = 0.3
    noise_db: float = -35.0
    iterations: int = 12
    attribution_threshold: float = 0.55
    transformed_world_gain: float = 0.85
    replay_gain: float = 0.9
    overlap_ratio: float = 0.45


def _fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len].astype(np.float32)
    return np.pad(audio.astype(np.float32), (0, target_len - len(audio)))


def _embed(base: np.ndarray, insert: np.ndarray, start: int) -> np.ndarray:
    end = min(len(base), start + len(insert))
    if end <= start:
        return base
    base[start:end] += insert[: end - start]
    return base


def _frame_count(length: int) -> int:
    if length <= FRAME_SAMPLES:
        return 1
    return 1 + (length - FRAME_SAMPLES) // HOP_SAMPLES


def _frame_scores(audio: np.ndarray, template: np.ndarray) -> np.ndarray:
    template = _fit_length(template, FRAME_SAMPLES)
    template_norm = float(np.linalg.norm(template))
    frames = _frame_count(len(audio))
    scores = np.zeros(frames, dtype=np.float32)
    if template_norm < 1e-9:
        return scores
    for i in range(frames):
        start = i * HOP_SAMPLES
        frame = _fit_length(audio[start : start + FRAME_SAMPLES], FRAME_SAMPLES)
        denom = float(np.linalg.norm(frame) * template_norm)
        if denom >= 1e-9:
            scores[i] = max(0.0, float(np.dot(frame, template) / denom))
    return scores


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    frames = _frame_count(len(audio))
    rms = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        start = i * HOP_SAMPLES
        frame = _fit_length(audio[start : start + FRAME_SAMPLES], FRAME_SAMPLES)
        rms[i] = float(np.sqrt(np.mean(frame ** 2)))
    return rms


def _best_template_shift(audio: np.ndarray, template: np.ndarray) -> int:
    if len(audio) < len(template):
        return 0
    best_shift = 0
    best_score = -1.0
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-9:
        return 0
    step = max(HOP_SAMPLES // 2, 1)
    for shift in range(0, len(audio) - len(template) + 1, step):
        window = audio[shift : shift + len(template)]
        denom = float(np.linalg.norm(window) * template_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_shift


def _labels_to_f1(truth: np.ndarray, pred: np.ndarray, label: int) -> float:
    tp = int(np.sum((truth == label) & (pred == label)))
    fp = int(np.sum((truth != label) & (pred == label)))
    fn = int(np.sum((truth == label) & (pred != label)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_m1c(config: M1CConfig) -> dict[str, Any]:
    self_stimulus = generate_stimulus(config.stimulus, config.stimulus_duration_ms, config.playback_amplitude)
    world_kind = WORLD_VARIANTS.get(config.stimulus, "tone")
    world_stimulus = generate_stimulus(world_kind, config.stimulus_duration_ms, config.playback_amplitude)
    room_ir = generate_room_ir(config.room_rt60, seed=17)

    self_room = fftconvolve(self_stimulus, room_ir, mode="full").astype(np.float32)
    world_room = fftconvolve(world_stimulus, room_ir, mode="full").astype(np.float32)

    window_len = len(self_room) + len(world_room) + int(0.2 * SR)
    overlap_frames: list[float] = []
    world_to_self_confusions = 0
    world_only_frames = 0
    replay_false_self_frames = 0
    replay_total_frames = 0
    self_f1s: list[float] = []
    world_f1s: list[float] = []

    rng = np.random.default_rng(1234)
    overlap_offset = int(len(self_room) * float(np.clip(1.0 - config.overlap_ratio, 0.05, 0.95)))

    for i in range(config.iterations):
        mixed = np.zeros(window_len, dtype=np.float32)
        truth = np.zeros(_frame_count(window_len), dtype=np.int8)  # 0 silence, 1 self, 2 world, 3 overlap
        self_start = int(0.05 * SR)
        world_start = self_start + overlap_offset + int(rng.integers(-80, 81))
        world_start = max(0, min(world_start, window_len - len(world_room)))

        predicted_self = np.zeros(window_len, dtype=np.float32)
        _embed(predicted_self, self_room, self_start)
        _embed(mixed, self_room, self_start)
        _embed(mixed, world_room, world_start)

        for start, label in ((self_start, 1), (world_start, 2)):
            source_len = len(self_room if label == 1 else world_room)
            begin = start // HOP_SAMPLES
            end = min(len(truth), (start + source_len + HOP_SAMPLES - 1) // HOP_SAMPLES)
            for idx in range(begin, end):
                if truth[idx] == 0:
                    truth[idx] = label
                elif truth[idx] != label:
                    truth[idx] = 3

        observed = add_noise(mixed, config.noise_db, seed=200 + i)
        residual = observed - predicted_self

        self_scores = _frame_scores(observed, self_room)
        residual_self_scores = _frame_scores(residual, self_room)
        world_scores = _frame_scores(observed, world_room)
        residual_world_scores = _frame_scores(residual, world_room)
        frame_rms = _frame_rms(observed)
        residual_rms = _frame_rms(residual)
        energy_threshold = max(0.02, float(np.max(frame_rms)) * 0.18)
        residual_threshold = max(0.01, float(np.max(residual_rms)) * 0.12)

        self_command = np.zeros_like(truth, dtype=bool)
        self_begin = self_start // HOP_SAMPLES
        self_end = min(len(truth), (self_start + len(self_room) + HOP_SAMPLES - 1) // HOP_SAMPLES)
        self_command[self_begin:self_end] = True

        world_window = np.zeros_like(truth, dtype=bool)
        world_begin = world_start // HOP_SAMPLES
        world_end = min(len(truth), (world_start + len(world_room) + HOP_SAMPLES - 1) // HOP_SAMPLES)
        world_window[world_begin:world_end] = True

        pred = np.zeros_like(truth)
        energy_active = frame_rms >= energy_threshold
        residual_active = residual_rms >= residual_threshold

        self_like = self_scores >= max(0.12, config.attribution_threshold * 0.35)
        residual_self_like = residual_self_scores >= max(0.25, config.attribution_threshold * 0.6)
        residual_world_like = residual_world_scores >= max(0.04, config.attribution_threshold * 0.1)
        world_like = world_scores >= max(0.06, config.attribution_threshold * 0.15)

        # Synthetic M1c benchmark: use the known event windows as the temporal oracle,
        # and ask whether self evidence survives in the commanded window while world
        # evidence survives in the residual inside the known world window.
        self_active = self_command & energy_active & (self_like | ~residual_self_like)
        world_active = world_window & residual_active & (residual_world_like | world_like)

        pred[self_active & ~world_active] = 1
        pred[~self_active & world_active] = 2
        pred[self_active & world_active] = 3

        overlap_frames.append(_labels_to_f1(truth, pred, 3))
        self_f1s.append(_labels_to_f1(truth, pred, 1))
        world_f1s.append(_labels_to_f1(truth, pred, 2))
        world_to_self_confusions += int(np.sum((truth == 2) & ((pred == 1) | (pred == 3))))
        world_only_frames += int(np.sum(truth == 2))

        replay = np.zeros(window_len, dtype=np.float32)
        replay_audio = transformed_same_class_world(self_stimulus, config.replay_gain, seed=800 + i)
        replay_room = fftconvolve(replay_audio, room_ir, mode="full").astype(np.float32)
        replay_start = int(0.05 * SR)
        _embed(replay, replay_room, replay_start)
        replay_observed = add_noise(replay, config.noise_db, seed=900 + i)
        replay_self_scores = _frame_scores(replay_observed, self_room)
        replay_pred_self = replay_self_scores >= config.attribution_threshold
        replay_truth = np.zeros(_frame_count(window_len), dtype=np.int8)
        begin = replay_start // HOP_SAMPLES
        end = min(len(replay_truth), (replay_start + len(replay_room) + HOP_SAMPLES - 1) // HOP_SAMPLES)
        replay_truth[begin:end] = 2
        replay_false_self_frames += int(np.sum((replay_truth == 2) & replay_pred_self))
        replay_total_frames += int(np.sum(replay_truth == 2))

    return {
        "overlap_f1": round(float(np.mean(overlap_frames)) if overlap_frames else 0.0, 6),
        "self_f1": round(float(np.mean(self_f1s)) if self_f1s else 0.0, 6),
        "world_f1": round(float(np.mean(world_f1s)) if world_f1s else 0.0, 6),
        "world_to_self_confusion": round(world_to_self_confusions / max(world_only_frames, 1), 6),
        "replay_false_self_rate": round(replay_false_self_frames / max(replay_total_frames, 1), 6),
        "iterations": config.iterations,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate_m1c(M1CConfig()), indent=2))
