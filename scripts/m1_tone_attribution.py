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
from scipy.signal import fftconvolve, resample

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
    onset_jitter_ms: int = 8
    transformed_world_gain: float = 0.85
    replay_gain: float = 0.9
    hard_negative_noise_db_offset: float = 5.0
    hard_negative_amplitude_jitter: float = 0.15
    causal_tolerance_ms: float = 20.0
    causal_weight: float = 0.5
    similarity_weight: float = 0.5
    stimulus_seed: int = 0


def build_ramp(n: int, attack_ms: float = 10.0, release_ms: float | None = None) -> np.ndarray:
    release_ms = attack_ms if release_ms is None else release_ms
    attack_len = max(1, int((attack_ms / 1000.0) * SR))
    release_len = max(1, int((release_ms / 1000.0) * SR))
    ramp = np.ones(n, dtype=np.float32)
    if attack_len + release_len < n:
        ramp[:attack_len] = np.linspace(0.0, 1.0, attack_len, dtype=np.float32)
        ramp[-release_len:] = np.linspace(1.0, 0.0, release_len, dtype=np.float32)
    return ramp


def generate_stimulus(kind: str, duration_ms: int, amplitude: float, seed: int | None = None) -> np.ndarray:
    return generate_stimulus_with_metadata(kind, duration_ms, amplitude, seed=seed)["pcm"]


def generate_stimulus_with_metadata(
    kind: str,
    duration_ms: int,
    amplitude: float,
    seed: int | None = None,
) -> dict[str, Any]:
    resolved_seed = 0 if seed is None else int(seed)
    rng = np.random.default_rng(resolved_seed)

    actual_duration_ms = int(duration_ms)
    actual_amplitude = float(amplitude)
    attack_ms = 10.0
    release_ms = 10.0
    f0_hz = None
    envelope_lfo_hz = None
    formants: list[tuple[float, float]] | None = None
    chirp_band_hz: tuple[float, float] | None = None

    if kind == "vowel_ah_var":
        actual_duration_ms = int(rng.integers(max(80, duration_ms - 80), duration_ms + 121))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.9, 1.1), 0.05, 1.0))
        attack_ms = float(rng.uniform(6.0, 18.0))
        release_ms = float(rng.uniform(8.0, 24.0))
        f0_hz = float(rng.uniform(120.0, 170.0))
        envelope_lfo_hz = float(rng.uniform(2.0, 4.5))
        formants = [
            (float(rng.uniform(720.0, 900.0)), 1.0),
            (float(rng.uniform(1080.0, 1320.0)), 0.35),
            (float(rng.uniform(2300.0, 2700.0)), 0.15),
        ]
    elif kind == "vowel_ah_var_nearfield":
        actual_duration_ms = int(rng.integers(max(170, duration_ms - 85), duration_ms + 21))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.98, 1.08), 0.05, 1.0))
        attack_ms = float(rng.uniform(6.8, 9.6))
        release_ms = float(rng.uniform(8.0, 12.0))
        f0_hz = float(rng.uniform(148.0, 163.0))
        envelope_lfo_hz = float(rng.uniform(3.4, 4.5))
        formants = [
            (float(rng.uniform(770.0, 855.0)), 1.0),
            (float(rng.uniform(1140.0, 1265.0)), 0.35),
            (float(rng.uniform(2380.0, 2570.0)), 0.15),
        ]
    elif kind == "vowel_ah_var_nearfield_room2":
        actual_duration_ms = int(rng.integers(max(230, duration_ms - 70), duration_ms + 16))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.99, 1.07), 0.05, 1.0))
        attack_ms = float(rng.uniform(8.0, 9.2))
        release_ms = float(rng.uniform(8.0, 9.3))
        f0_hz = float(rng.uniform(149.0, 157.0))
        envelope_lfo_hz = float(rng.uniform(3.7, 4.45))
        formants = [
            (float(rng.uniform(770.0, 840.0)), 1.0),
            (float(rng.uniform(1145.0, 1245.0)), 0.35),
            (float(rng.uniform(2385.0, 2560.0)), 0.15),
        ]
    elif kind == "babble_ba_var":
        actual_duration_ms = int(rng.integers(max(170, duration_ms - 60), duration_ms + 61))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.95, 1.08), 0.05, 1.0))
        attack_ms = float(rng.uniform(6.0, 12.0))
        release_ms = float(rng.uniform(8.0, 14.0))
        f0_hz = float(rng.uniform(135.0, 165.0))
        envelope_lfo_hz = float(rng.uniform(3.0, 4.6))
        formants = [
            (float(rng.uniform(700.0, 880.0)), 1.0),
            (float(rng.uniform(1050.0, 1280.0)), 0.4),
            (float(rng.uniform(2200.0, 2550.0)), 0.18),
        ]
    elif kind in {"babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var"}:
        if kind == "babble_ba_repeat_short_var":
            actual_duration_ms = int(rng.integers(max(360, duration_ms - 90), duration_ms + 91))
        elif kind == "babble_ba_repeat_few_var":
            actual_duration_ms = int(rng.integers(max(420, duration_ms - 100), duration_ms + 111))
        elif kind == "babble_ba_repeat_wide_var":
            actual_duration_ms = int(rng.integers(max(620, duration_ms - 140), duration_ms + 161))
        elif kind == "babble_ba_repeat_short_tight_var":
            actual_duration_ms = int(rng.integers(max(360, duration_ms - 40), duration_ms + 21))
        else:
            actual_duration_ms = int(rng.integers(max(540, duration_ms - 120), duration_ms + 141))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.97, 1.04), 0.05, 1.0))
        attack_ms = float(rng.uniform(5.0, 10.0))
        release_ms = float(rng.uniform(10.0, 18.0))
        if kind == "babble_ba_repeat_short_tight_var":
            f0_hz = float(rng.uniform(148.0, 160.0))
            envelope_lfo_hz = float(rng.uniform(4.1, 4.7))
        else:
            f0_hz = float(rng.uniform(135.0, 165.0))
            envelope_lfo_hz = float(rng.uniform(3.0, 4.8))
        formants = [
            (float(rng.uniform(700.0, 880.0)), 1.0),
            (float(rng.uniform(1050.0, 1280.0)), 0.4),
            (float(rng.uniform(2200.0, 2550.0)), 0.18),
        ]
    elif kind == "babble_da_var":
        actual_duration_ms = int(rng.integers(max(170, duration_ms - 60), duration_ms + 61))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.95, 1.08), 0.05, 1.0))
        attack_ms = float(rng.uniform(5.0, 11.0))
        release_ms = float(rng.uniform(8.0, 14.0))
        f0_hz = float(rng.uniform(138.0, 170.0))
        envelope_lfo_hz = float(rng.uniform(3.0, 4.8))
        formants = [
            (float(rng.uniform(820.0, 980.0)), 1.0),
            (float(rng.uniform(1300.0, 1650.0)), 0.42),
            (float(rng.uniform(2400.0, 2900.0)), 0.18),
        ]
    elif kind == "babble_ma_var":
        actual_duration_ms = int(rng.integers(max(180, duration_ms - 70), duration_ms + 71))
        actual_amplitude = float(np.clip(amplitude * rng.uniform(0.94, 1.06), 0.05, 1.0))
        attack_ms = float(rng.uniform(7.0, 13.0))
        release_ms = float(rng.uniform(9.0, 15.0))
        f0_hz = float(rng.uniform(132.0, 162.0))
        envelope_lfo_hz = float(rng.uniform(2.8, 4.4))
        formants = [
            (float(rng.uniform(260.0, 420.0)), 1.0),
            (float(rng.uniform(1100.0, 1450.0)), 0.45),
            (float(rng.uniform(2200.0, 2550.0)), 0.16),
        ]

    n = max(1, int(SR * actual_duration_ms / 1000))
    t = np.arange(n, dtype=np.float32) / SR
    ramp = build_ramp(n, attack_ms=attack_ms, release_ms=release_ms)

    if kind == "tone":
        pcm = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    elif kind == "click":
        pcm = np.zeros(n, dtype=np.float32)
        click_positions = [0, n // 3, (2 * n) // 3]
        for pos in click_positions:
            pcm[pos:min(n, pos + 4)] = 1.0
    elif kind in {"chirp", "speech_band_chirp", "low_chirp"}:
        # Cardioid mic rejection is frequency-dependent and high-frequency
        # geometry can be dominated by reflections/body diffraction. Keep the
        # existing chirp unchanged for backwards compatibility, and add M1
        # calibration probes biased toward the lower/mid speech bands.
        if kind == "speech_band_chirp":
            f0, f1 = 150.0, 2500.0
        elif kind == "low_chirp":
            f0, f1 = 150.0, 1200.0
        else:
            f0, f1 = 300.0, 1800.0
        chirp_band_hz = (f0, f1)
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / max(t[-1], 1e-6))
        pcm = np.sin(phase).astype(np.float32)
    elif kind in {"vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield", "vowel_ah_var_nearfield_room2", "babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var", "babble_da_var", "babble_ma_var"}:
        if f0_hz is None:
            f0_hz = 140.0
        if envelope_lfo_hz is None:
            envelope_lfo_hz = 3.0
        if formants is None:
            formants = [(800.0, 1.0), (1200.0, 0.35), (2500.0, 0.15)]
        vowel = np.zeros_like(t, dtype=np.float32)
        for freq, formant_amp in formants:
            vowel += formant_amp * np.sin(2 * np.pi * freq * t)
        vowel *= (0.6 + 0.4 * np.sin(2 * np.pi * f0_hz * t)).astype(np.float32)
        vowel *= (0.9 + 0.1 * np.sin(2 * np.pi * envelope_lfo_hz * t)).astype(np.float32)
        if kind in {"babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var", "babble_da_var", "babble_ma_var"}:
            if kind in {"babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var"}:
                burst_ms = float(rng.uniform(10.0, 22.0))
                closure_ms = float(rng.uniform(8.0, 18.0))
                burst_strength = 0.45
                formant_blend = 1.0
                vowel_onset_gap_samples = max(1, int(0.002 * SR))
                excitation = rng.standard_normal(n).astype(np.float32)
            elif kind == "babble_da_var":
                burst_ms = float(rng.uniform(6.0, 16.0))
                closure_ms = float(rng.uniform(4.0, 12.0))
                burst_strength = 0.30
                formant_blend = 1.15
                vowel_onset_gap_samples = max(1, int(0.002 * SR))
                excitation = rng.standard_normal(n).astype(np.float32)
            else:
                burst_ms = float(rng.uniform(0.0, 6.0))
                closure_ms = float(rng.uniform(10.0, 24.0))
                burst_strength = 0.08
                formant_blend = 0.9
                vowel_onset_gap_samples = 0
                excitation = rng.standard_normal(n).astype(np.float32)

            def build_single_babble(segment_len: int, segment_vowel: np.ndarray) -> tuple[np.ndarray, int]:
                burst_len = max(1, int(SR * burst_ms / 1000.0))
                closure_len = max(1, int(SR * closure_ms / 1000.0))
                burst_env = np.zeros(segment_len, dtype=np.float32)
                start = min(closure_len, segment_len - 1)
                end = min(start + burst_len, segment_len)
                if end > start:
                    burst_env[start:end] = np.linspace(1.0, 0.0, end - start, dtype=np.float32)
                burst = excitation[:segment_len] * burst_env * burst_strength
                vowel_onset = min(end + vowel_onset_gap_samples, segment_len)
                segment_pcm = burst.copy()
                if kind == "babble_ma_var":
                    nasal_env = np.ones_like(segment_vowel, dtype=np.float32)
                    nasal_env[:min(vowel_onset or 1, len(nasal_env))] *= 0.6
                    vowel_body = segment_vowel * nasal_env * formant_blend
                else:
                    vowel_body = segment_vowel * formant_blend
                if vowel_onset < segment_len:
                    vowel_tail = vowel_body[: segment_len - vowel_onset]
                    vowel_ramp = np.linspace(0.0, 1.0, len(vowel_tail), dtype=np.float32)
                    segment_pcm[vowel_onset:] += vowel_tail * np.minimum(vowel_ramp * 2.5, 1.0)
                return segment_pcm, vowel_onset

            if kind in {"babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var"}:
                if kind == "babble_ba_repeat_short_var":
                    syllable_count = int(rng.integers(3, 5))
                    gap_ms = float(rng.uniform(28.0, 48.0))
                elif kind == "babble_ba_repeat_few_var":
                    syllable_count = int(rng.integers(2, 4))
                    gap_ms = float(rng.uniform(45.0, 75.0))
                elif kind == "babble_ba_repeat_wide_var":
                    syllable_count = int(rng.integers(4, 6))
                    gap_ms = float(rng.uniform(85.0, 130.0))
                elif kind == "babble_ba_repeat_short_tight_var":
                    syllable_count = int(rng.integers(3, 4))
                    gap_ms = float(rng.uniform(24.0, 36.0))
                else:
                    syllable_count = int(rng.integers(4, 7))
                    gap_ms = float(rng.uniform(35.0, 70.0))
                gap_len = max(1, int(SR * gap_ms / 1000.0))
                segment_len = max(1, int((n - gap_len * max(syllable_count - 1, 0)) / syllable_count))
                pcm = np.zeros(n, dtype=np.float32)
                cursor = 0
                for _ in range(syllable_count):
                    remaining = n - cursor
                    if remaining <= 0:
                        break
                    this_len = min(segment_len, remaining)
                    seg_vowel = vowel[:this_len]
                    seg_pcm, _ = build_single_babble(this_len, seg_vowel)
                    pcm[cursor:cursor + this_len] += seg_pcm
                    cursor += this_len + gap_len
                attack_ms = max(attack_ms, burst_ms + closure_ms)
                release_ms = max(release_ms, gap_ms / 2.0)
            else:
                pcm, _ = build_single_babble(n, vowel)
                attack_ms = max(attack_ms, burst_ms + closure_ms)
        else:
            pcm = vowel
    else:
        raise ValueError(f"Unsupported stimulus kind: {kind}")

    pcm *= ramp
    peak = float(np.max(np.abs(pcm)))
    if peak > 0:
        pcm = pcm / peak
    pcm = (pcm * actual_amplitude).astype(np.float32)

    return {
        "kind": kind,
        "family": "babble_ba_repeat" if kind in {"babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var"} else ("babble_ba" if kind == "babble_ba_var" else ("babble_da" if kind == "babble_da_var" else ("babble_ma" if kind == "babble_ma_var" else ("vowel_ah" if kind in {"vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield", "vowel_ah_var_nearfield_room2"} else kind)))),
        "seed": resolved_seed,
        "duration_ms": actual_duration_ms,
        "amplitude": round(actual_amplitude, 6),
        "attack_ms": round(attack_ms, 4),
        "release_ms": round(release_ms, 4),
        "f0_hz": None if f0_hz is None else round(f0_hz, 4),
        "envelope_lfo_hz": None if envelope_lfo_hz is None else round(envelope_lfo_hz, 4),
        "formants": None if formants is None else [
            {"hz": round(freq, 4), "amp": round(formant_amp, 4)} for freq, formant_amp in formants
        ],
        "chirp_band_hz": None if chirp_band_hz is None else [round(chirp_band_hz[0], 4), round(chirp_band_hz[1], 4)],
        "pcm": pcm,
    }


WORLD_VARIANTS = {
    "tone": "chirp",
    "chirp": "click",
    "speech_band_chirp": "low_chirp",
    "low_chirp": "speech_band_chirp",
    "click": "tone",
    "vowel_ah": "chirp",
    "vowel_ah_var": "chirp",
    "vowel_ah_var_nearfield": "chirp",
    "vowel_ah_var_nearfield_room2": "chirp",
    "babble_ba_var": "chirp",
    "babble_ba_repeat_var": "chirp",
    "babble_ba_repeat_short_var": "chirp",
    "babble_ba_repeat_few_var": "chirp",
    "babble_ba_repeat_wide_var": "chirp",
    "babble_ba_repeat_short_tight_var": "chirp",
    "babble_da_var": "chirp",
    "babble_ma_var": "chirp",
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


def time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    if len(audio) < 2 or abs(factor - 1.0) < 1e-6:
        return audio.astype(np.float32)
    target_len = max(1, int(len(audio) * factor))
    stretched = resample(audio, target_len).astype(np.float32)
    if target_len > len(audio):
        return stretched[:len(audio)]
    if target_len < len(audio):
        return np.pad(stretched, (0, len(audio) - target_len))
    return stretched


def transformed_same_class_world(audio: np.ndarray, gain: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stretched = time_stretch(audio, float(rng.uniform(0.92, 1.08)))
    rolled = np.roll(stretched, int(rng.integers(8, 64)))
    polarity = -1.0 if rng.random() < 0.5 else 1.0
    transformed = rolled * gain * polarity
    peak = float(np.max(np.abs(transformed)))
    if peak > 1e-9:
        transformed = transformed / peak * min(0.98, gain)
    return transformed.astype(np.float32)


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


def estimate_event_onset(audio: np.ndarray, threshold: float, stimulus_kind: str) -> int:
    if len(audio) == 0:
        return 0

    magnitude = np.abs(audio)
    peak = float(np.max(magnitude))
    if peak < 1e-9:
        return 0

    if stimulus_kind == "click":
        transient = np.abs(np.diff(audio, prepend=audio[0]))
        transient_peak = float(np.max(transient))
        transient_threshold = max(threshold * 0.5, transient_peak * 0.25)
        idx = np.where(transient >= transient_threshold)[0]
        return int(idx[0]) if len(idx) else 0

    if stimulus_kind in {"tone", "vowel_ah"}:
        window = max(8, SR // 500)
        kernel = np.ones(window, dtype=np.float32) / window
        envelope = np.convolve(magnitude, kernel, mode="same")
        env_peak = float(np.max(envelope))
        env_threshold = max(threshold * 0.75, env_peak * 0.15)
        idx = np.where(envelope >= env_threshold)[0]
        return int(idx[0]) if len(idx) else 0

    onset_threshold = max(threshold, peak * 0.2)
    idx = np.where(magnitude >= onset_threshold)[0]
    return int(idx[0]) if len(idx) else 0


def causal_alignment_score(observed_shift: int, expected_shift: int, tolerance_ms: float) -> float:
    error_ms = abs(observed_shift - expected_shift) * 1000.0 / SR
    return max(0.0, 1.0 - (error_ms / max(tolerance_ms, 1e-6)))


def combined_attribution_score(
    similarity_score: float,
    observed_shift: int,
    expected_shift: int | None,
    causal_tolerance_ms: float,
    causal_weight: float,
    similarity_weight: float,
) -> tuple[float, float]:
    causal_score = 0.0 if expected_shift is None else causal_alignment_score(observed_shift, expected_shift, causal_tolerance_ms)
    total_weight = max(causal_weight + similarity_weight, 1e-9)
    combined = (
        similarity_weight * similarity_score + causal_weight * causal_score
    ) / total_weight
    return combined, causal_score


def evaluate_m1a(config: M1AConfig) -> dict[str, Any]:
    self_meta = generate_stimulus_with_metadata(
        config.stimulus,
        config.stimulus_duration_ms,
        config.playback_amplitude,
        seed=config.stimulus_seed,
    )
    self_stimulus = self_meta["pcm"]
    world_kind = WORLD_VARIANTS.get(config.stimulus, "chirp")
    world_meta = generate_stimulus_with_metadata(
        world_kind,
        config.stimulus_duration_ms,
        config.playback_amplitude,
        seed=config.stimulus_seed + 1,
    )
    world_stimulus = world_meta["pcm"]
    room_ir = generate_room_ir(config.room_rt60, seed=7)

    shift_min_samples = max(0, config.shift_min_frames * 160)
    shift_max_samples = max(shift_min_samples + 1, config.shift_max_frames * 160)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    onset_errors_ms: list[float] = []
    same_class_scores: list[float] = []
    replay_scores: list[float] = []
    silence_scores: list[float] = []
    cross_class_scores: list[float] = []
    causal_scores_self: list[float] = []
    causal_scores_negative: list[float] = []
    combined_scores_self: list[float] = []
    combined_scores_negative: list[float] = []

    tp = fp = tn = fn = 0
    replay_fp = 0
    same_class_fp = 0

    rng = np.random.default_rng(999)

    for i in range(config.iterations):
        base_delay_ms = 20 + 5 * (i % 5)
        jitter_ms = int(rng.integers(-config.onset_jitter_ms, config.onset_jitter_ms + 1))
        delay_ms = max(0, base_delay_ms + jitter_ms)
        delay_samples = int(delay_ms * SR / 1000)

        amplitude_scale = float(1.0 + rng.uniform(-config.hard_negative_amplitude_jitter, config.hard_negative_amplitude_jitter))
        trial_self = (self_stimulus * amplitude_scale).astype(np.float32)
        peak = float(np.max(np.abs(trial_self)))
        if peak > 1.0:
            trial_self = (trial_self / peak).astype(np.float32)

        captured_self = delayed_room_capture(trial_self, room_ir, delay_samples, config.noise_db, seed=100 + i)
        score_self, shift_self = best_match_score(captured_self, self_stimulus, shift_min_samples, shift_max_samples)
        onset_self = estimate_event_onset(captured_self, config.event_threshold, config.stimulus)
        combined_self, causal_self = combined_attribution_score(
            score_self,
            onset_self,
            delay_samples,
            config.causal_tolerance_ms,
            config.causal_weight,
            config.similarity_weight,
        )
        predicted_self = event_present(captured_self, config.event_threshold) and combined_self >= config.attribution_threshold
        positive_scores.append(score_self)
        causal_scores_self.append(causal_self)
        combined_scores_self.append(combined_self)
        onset_errors_ms.append(abs(onset_self - delay_samples) * 1000.0 / SR)
        if predicted_self:
            tp += 1
        else:
            fn += 1

        cross_noise_db = config.noise_db + rng.uniform(0.0, config.hard_negative_noise_db_offset)
        captured_world = delayed_room_capture(world_stimulus, room_ir, delay_samples, cross_noise_db, seed=200 + i)
        score_world, _ = best_match_score(captured_world, self_stimulus, shift_min_samples, shift_max_samples)
        onset_world = estimate_event_onset(captured_world, config.event_threshold, config.stimulus)
        combined_world, causal_world = combined_attribution_score(
            score_world,
            onset_world,
            None,
            config.causal_tolerance_ms,
            config.causal_weight,
            config.similarity_weight,
        )
        predicted_world_as_self = event_present(captured_world, config.event_threshold) and combined_world >= config.attribution_threshold
        negative_scores.append(score_world)
        cross_class_scores.append(score_world)
        causal_scores_negative.append(causal_world)
        combined_scores_negative.append(combined_world)
        if predicted_world_as_self:
            fp += 1
        else:
            tn += 1

        transformed_world = transformed_same_class_world(self_stimulus, config.transformed_world_gain, seed=400 + i)
        captured_same_class = delayed_room_capture(
            transformed_world,
            room_ir,
            delay_samples,
            config.noise_db + rng.uniform(-2.0, config.hard_negative_noise_db_offset),
            seed=500 + i,
        )
        score_same_class, _ = best_match_score(captured_same_class, self_stimulus, shift_min_samples, shift_max_samples)
        onset_same_class = estimate_event_onset(captured_same_class, config.event_threshold, config.stimulus)
        combined_same_class, causal_same_class = combined_attribution_score(
            score_same_class,
            onset_same_class,
            None,
            config.causal_tolerance_ms,
            config.causal_weight,
            config.similarity_weight,
        )
        predicted_same_class_as_self = event_present(captured_same_class, config.event_threshold) and combined_same_class >= config.attribution_threshold
        negative_scores.append(score_same_class)
        same_class_scores.append(score_same_class)
        causal_scores_negative.append(causal_same_class)
        combined_scores_negative.append(combined_same_class)
        if predicted_same_class_as_self:
            fp += 1
            same_class_fp += 1
        else:
            tn += 1

        replay_delay_ms = max(0, delay_ms + int(rng.integers(-config.onset_jitter_ms, config.onset_jitter_ms + 1)))
        replay_delay_samples = int(replay_delay_ms * SR / 1000)
        replay_audio = (self_stimulus * config.replay_gain).astype(np.float32)
        captured_replay = delayed_room_capture(
            replay_audio,
            room_ir,
            replay_delay_samples,
            config.noise_db + rng.uniform(-2.0, config.hard_negative_noise_db_offset),
            seed=600 + i,
        )
        score_replay, _ = best_match_score(captured_replay, self_stimulus, shift_min_samples, shift_max_samples)
        onset_replay = estimate_event_onset(captured_replay, config.event_threshold, config.stimulus)
        combined_replay, causal_replay = combined_attribution_score(
            score_replay,
            onset_replay,
            None,
            config.causal_tolerance_ms,
            config.causal_weight,
            config.similarity_weight,
        )
        predicted_replay_as_self = event_present(captured_replay, config.event_threshold) and combined_replay >= config.attribution_threshold
        negative_scores.append(score_replay)
        replay_scores.append(score_replay)
        causal_scores_negative.append(causal_replay)
        combined_scores_negative.append(combined_replay)
        if predicted_replay_as_self:
            fp += 1
            replay_fp += 1
        else:
            tn += 1

        silence = add_noise(np.zeros_like(captured_self), config.noise_db + rng.uniform(0.0, config.hard_negative_noise_db_offset), seed=300 + i)
        score_silence, _ = best_match_score(silence, self_stimulus, shift_min_samples, shift_max_samples)
        onset_silence = estimate_event_onset(silence, config.event_threshold, config.stimulus)
        combined_silence, causal_silence = combined_attribution_score(
            score_silence,
            onset_silence,
            None,
            config.causal_tolerance_ms,
            config.causal_weight,
            config.similarity_weight,
        )
        predicted_silence_as_self = event_present(silence, config.event_threshold) and combined_silence >= config.attribution_threshold
        negative_scores.append(score_silence)
        silence_scores.append(score_silence)
        causal_scores_negative.append(causal_silence)
        combined_scores_negative.append(combined_silence)
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
        "replay_false_self_rate": round(replay_fp / max(config.iterations, 1), 6),
        "same_class_false_self_rate": round(same_class_fp / max(config.iterations, 1), 6),
        "onset_error_ms": round(float(np.mean(onset_errors_ms)) if onset_errors_ms else 0.0, 4),
        "self_score_mean": round(float(np.mean(positive_scores)) if positive_scores else 0.0, 6),
        "world_score_mean": round(float(np.mean(negative_scores)) if negative_scores else 0.0, 6),
        "cross_class_score_mean": round(float(np.mean(cross_class_scores)) if cross_class_scores else 0.0, 6),
        "same_class_score_mean": round(float(np.mean(same_class_scores)) if same_class_scores else 0.0, 6),
        "replay_score_mean": round(float(np.mean(replay_scores)) if replay_scores else 0.0, 6),
        "silence_score_mean": round(float(np.mean(silence_scores)) if silence_scores else 0.0, 6),
        "causal_self_score_mean": round(float(np.mean(causal_scores_self)) if causal_scores_self else 0.0, 6),
        "causal_negative_score_mean": round(float(np.mean(causal_scores_negative)) if causal_scores_negative else 0.0, 6),
        "combined_self_score_mean": round(float(np.mean(combined_scores_self)) if combined_scores_self else 0.0, 6),
        "combined_negative_score_mean": round(float(np.mean(combined_scores_negative)) if combined_scores_negative else 0.0, 6),
        "positive_trials": config.iterations,
        "negative_trials": config.iterations * 4,
        "stimulus_family": self_meta["family"],
        "stimulus_seed": self_meta["seed"],
        "stimulus_duration_realized_ms": self_meta["duration_ms"],
        "stimulus_amplitude_realized": self_meta["amplitude"],
        "stimulus_attack_ms": self_meta["attack_ms"],
        "stimulus_release_ms": self_meta["release_ms"],
        "stimulus_f0_hz": self_meta["f0_hz"],
        "stimulus_envelope_lfo_hz": self_meta["envelope_lfo_hz"],
    }
