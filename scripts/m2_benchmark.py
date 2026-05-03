from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    import torch
    _HAS_EMBED = True
except Exception:
    _HAS_EMBED = False

#!/usr/bin/env python3
"""M2 Benchmark — Controlled speech-preservation measurement.

Problem with the prior harness: self audio (built-in speaker, cm from mic)
overwhelms world speech (headphones on desk), making WER=1.0 regardless of
gating strategy. The benchmark was measuring the wrong thing — it could not
distinguish between "gating helps" and "gating doesn't matter because the
capture is already destroyed."

This script fixes the benchmark by:

1. Systematically sweeping self-amplitude to find the regime where world
   speech is genuinely recoverable (raw WER <= ~0.3).
2. Reporting self/world SNR at the mic as a first-class metric.
3. Using a longer, phonetically diverse reference text for ASR discrimination.
4. Separating acquisition from processing (capture-once, replay-offline).
5. Providing a calibration mode to find the operating point before running
   the full benchmark.

Usage:
  # Step 1: Calibrate — find the self-amplitude where raw WER starts to degrade
  ./.venv/bin/python scripts/m2_benchmark.py --mode calibrate \\
      --self-stim vowel_ah_var_nearfield \\
      --input-device 3 --output-device 5 --noise-device 4

  # Step 2: Run the benchmark at a chosen amplitude
  ./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \\
      --self-stim vowel_ah_var_nearfield \\
      --self-amplitude 0.15 \\
      --input-device 3 --output-device 5 --noise-device 4 \\
      --save-root data/m2_benchmark

  # Step 3: Offline replay — evaluate multiple gating strategies against
  #         the same captures (no hardware needed)
  ./.venv/bin/python scripts/m2_benchmark.py --mode offline \\
      --capture-root data/m2_benchmark/calibration_20260501_120000/captures
"""


import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from cerebellum.room_coloring import (
    align_template_to_observed_onset,
    estimate_onset_samples,
    predict_room_colored_self,
)
from stg.speaker import SelfVoiceDetector, SpeakerIdentifier
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from stg.asr import transcribe

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata


# ---------------------------------------------------------------------------
# Reference text — longer and more phonetically diverse than the old
# "this is a test of world speech conservation" to give Whisper more
# discriminative power.
# ---------------------------------------------------------------------------
REFERENCE_TEXT = (
    "the quick brown fox jumps over the lazy dog and then "
    "runs through the forest to find a quiet place near the river"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein(seq1: list[str], seq2: list[str]) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            cost = 0 if a == b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def word_retention_rate(reference: str, hypothesis: str) -> float:
    """Fraction of reference words that appear in the hypothesis (order-independent)."""
    ref_words = set(normalize_text(reference).split())
    hyp_words = set(normalize_text(hypothesis).split())
    if not ref_words:
        return 1.0
    return len(ref_words & hyp_words) / len(ref_words)


def error_breakdown(reference: str, hypothesis: str) -> dict:
    """Compute deletion, substitution, and insertion counts.

    Uses Levenshtein alignment to classify each edit operation.
    Returns dict with deletions, substitutions, insertions, and total_errors.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words:
        return {"deletions": 0, "substitutions": 0, "insertions": len(hyp_words), "total": len(hyp_words)}
    if not hyp_words:
        return {"deletions": len(ref_words), "substitutions": 0, "insertions": 0, "total": len(ref_words)}

    # Build full Levenshtein matrix for backtracking
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    # Backtrack to classify operations
    deletions = 0
    substitutions = 0
    insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1  # match
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return {
        "deletions": deletions,
        "substitutions": substitutions,
        "insertions": insertions,
        "total": deletions + substitutions + insertions,
    }


def energy_retained_in_window(
    observed: np.ndarray,
    processed: np.ndarray,
    window_start_s: float,
    window_end_s: float,
    sr: int = 16000,
) -> float:
    """Fraction of world-speech energy retained after processing.

    Computes the ratio of processed energy to observed energy within a
    specific time window (e.g., the overlap region). 1.0 = full preservation,
    0.0 = complete suppression.
    """
    start_idx = int(window_start_s * sr)
    end_idx = int(window_end_s * sr)
    start_idx = max(0, start_idx)
    end_idx = min(len(observed), end_idx)

    if end_idx <= start_idx:
        return 0.0

    obs_energy = np.sum(observed[start_idx:end_idx] ** 2)
    proc_energy = np.sum(processed[start_idx:end_idx] ** 2)

    if obs_energy < 1e-20:
        return 1.0  # No energy to preserve
    return float(proc_energy / obs_energy)


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def play_array_blocking(data: np.ndarray, device: int, samplerate: float) -> None:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(
        device=device, channels=data.shape[1],
        samplerate=samplerate, dtype="float32",
    ) as stream:
        stream.write(np.asarray(data, dtype=np.float32))


def normalized_xcorr(observed: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    if len(observed) < len(template):
        return 0.0, 0
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-9:
        return 0.0, 0
    best_score = -1.0
    best_shift = 0
    step = max(len(template) // 80, 1)
    for shift in range(0, len(observed) - len(template) + 1, step):
        window = observed[shift:shift + len(template)]
        denom = float(np.linalg.norm(window) * template_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def local_xcorr(
    observed: np.ndarray, template: np.ndarray,
    expected_shift: int, search_ms: float = 250.0,
) -> tuple[float, int]:
    if len(observed) < len(template):
        return 0.0, expected_shift
    # z-normalize template
    t = template.astype(np.float64)
    t = t - np.mean(t)
    t_std = np.std(t)
    if t_std < 1e-9:
        return 0.0, expected_shift
    t = t / t_std
    radius = int(search_ms * SR / 1000.0)
    start = max(0, expected_shift - radius)
    stop = min(len(observed) - len(template), expected_shift + radius)
    if stop <= start:
        return 0.0, expected_shift
    best_score = -1.0
    best_shift = start
    step = max(len(template) // 160, 1)
    for shift in range(start, stop + 1, step):
        w = observed[shift:shift + len(template)].astype(np.float64)
        w = w - np.mean(w)
        w_std = np.std(w)
        if w_std < 1e-9:
            score = 0.0
        else:
            w = w / w_std
            score = float(np.clip(np.dot(w, t) / len(t), -1.0, 1.0))
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift




def embedding_similarity(
    observed: np.ndarray, template: np.ndarray,
    expected_shift: int, search_ms: float = 250.0,
    model=None, feature_extractor=None,
) -> Tuple[float, int]:
    """
    Cosine similarity between Wav2Vec2 embeddings of template and
    windows in observed.
    """
    if not _HAS_EMBED:
        return 0.0, expected_shift
    if model is None or feature_extractor is None:
        return 0.0, expected_shift
    if len(observed) < len(template):
        return 0.0, expected_shift

    sr = 16000
    radius = int(search_ms * sr / 1000.0)
    start = max(0, expected_shift - radius)
    stop = min(len(observed) - len(template), expected_shift + radius)
    if stop <= start:
        return 0.0, expected_shift

    # embed template (single)
    with torch.no_grad():
        t_in = feature_extractor(template, sampling_rate=sr, return_tensors="pt")
        t_out = model(**t_in).last_hidden_state  # (1, T, D)
        t_emb = t_out.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
        t_emb = t_emb / (np.linalg.norm(t_emb) + 1e-12)

    best_score = -1.0
    best_shift = start
    step = max(len(template) // 160, 1)

    for shift in range(start, stop + 1, step):
        w = observed[shift:shift + len(template)]
        with torch.no_grad():
            w_in = feature_extractor(w, sampling_rate=sr, return_tensors="pt")
            w_out = model(**w_in).last_hidden_state
            w_emb = w_out.mean(dim=1).squeeze(0).cpu().numpy()
            w_emb = w_emb / (np.linalg.norm(w_emb) + 1e-12)
        score = float(np.clip(np.dot(w_emb, t_emb), -1.0, 1.0))
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift

def energy_onset(
    observed: np.ndarray, start_hint_samples: int,
    search_before_ms: float = 0.0, search_after_ms: float = 800.0,
    frame_samples: int = VAD_FRAME_SAMPLES, threshold_ratio: float = 0.20,
) -> int:
    start = max(0, start_hint_samples - int(search_before_ms * SR / 1000.0))
    stop = min(len(observed), start_hint_samples + int(search_after_ms * SR / 1000.0))
    if stop <= start + frame_samples:
        return start_hint_samples
    frames: list[tuple[int, float]] = []
    for pos in range(start, stop - frame_samples + 1, frame_samples):
        frame = observed[pos:pos + frame_samples]
        frames.append((pos, float(np.sqrt(np.mean(frame ** 2)))))
    if not frames:
        return start_hint_samples
    floor = float(np.median([r for _, r in frames[: max(1, min(5, len(frames)))]]))
    peak = max(r for _, r in frames)
    threshold = floor + (peak - floor) * float(np.clip(threshold_ratio, 0.0, 1.0))
    for pos, rms in frames:
        if rms >= threshold:
            return pos
    return max(frames, key=lambda x: x[1])[0]


def build_gating_gain(
    verdicts: list[SourceVerdict], frame_samples: int,
    total_samples: int, both_gain: float,
) -> np.ndarray:
    """Build gain mask based on verdicts.
    
    Policy:
    - SELF: gain=1.0 (let my voice through, no suppression)
    - BOTH: gain=both_gain (suppress echoes)  
    - WORLD: gain=1.0 (let world speech through)
    - SILENCE: gain=1.0 (nothing to suppress)
    """
    gain = np.ones(total_samples, dtype=np.float32)
    both_gain = float(np.clip(both_gain, 0.0, 1.0))
    for i, verdict in enumerate(verdicts):
        start = i * frame_samples
        end = min(start + frame_samples, total_samples)
        if verdict == SourceVerdict.SELF:
            gain[start:end] = 1.0  # Let SELF through (no suppression)
        elif verdict == SourceVerdict.BOTH:
            gain[start:end] = both_gain  # Suppress echoes
        # WORLD and SILENCE: gain stays 1.0 (no change needed)
    return gain


def subtract_self_template(
    observed: np.ndarray, self_pcm: np.ndarray,
    verdicts: list[SourceVerdict], frame_samples: int,
    self_shift: int, both_subtract_gain: float,
    template_onset_samples: int = 0,
) -> np.ndarray:
    out = observed.astype(np.float32).copy()
    if len(self_pcm) == 0:
        return out
    template = align_template_to_observed_onset(
        self_pcm,
        total_samples=len(out),
        observed_onset_samples=self_shift,
        template_onset_samples=template_onset_samples,
    )
    both_mask = np.zeros(len(out), dtype=bool)
    for i, verdict in enumerate(verdicts):
        if verdict == SourceVerdict.BOTH:
            a = i * frame_samples
            b = min(a + frame_samples, len(out))
            both_mask[a:b] = True
    if not np.any(both_mask):
        return out
    t = template[both_mask]
    y = out[both_mask]
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return out
    scale = float(np.dot(y, t) / denom)
    scale = float(np.clip(scale, 0.0, 2.0)) * float(np.clip(both_subtract_gain, 0.0, 1.0))
    out[both_mask] = y - scale * t
    return out


def summarize_offline_results(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate offline replay rows by strategy.

    Offline replay is most useful when it compares processing strategies against
    the same captures. Keep the per-capture rows, but also emit a compact
    strategy-level view so we can tell whether a strategy is consistently helping
    or just winning one noisy seed.
    """
    aggregate: dict[str, Any] = {}
    strategy_names = sorted({
        strategy["name"]
        for capture in all_results
        for strategy in capture.get("strategies", [])
    })
    for name in strategy_names:
        rows = [
            strategy
            for capture in all_results
            for strategy in capture.get("strategies", [])
            if strategy["name"] == name
        ]
        deltas = [float(row["wer_delta"]) for row in rows]
        if not deltas:
            continue
        retentions = [float(row.get("retention", 0)) for row in rows]
        energies = [float(row.get("energy_retained", 0)) for row in rows]
        deletions = [int(row.get("errors", {}).get("deletions", 0)) for row in rows]
        substitutions = [int(row.get("errors", {}).get("substitutions", 0)) for row in rows]
        insertions = [int(row.get("errors", {}).get("insertions", 0)) for row in rows]
        aggregate[name] = {
            "trials": len(deltas),
            "mean_raw_wer": float(np.mean([row["raw_wer"] for row in rows])),
            "mean_gated_wer": float(np.mean([row["gated_wer"] for row in rows])),
            "mean_wer_delta": float(np.mean(deltas)),
            "std_wer_delta": float(np.std(deltas)),
            "improved": sum(1 for delta in deltas if delta < -0.01),
            "neutral": sum(1 for delta in deltas if -0.01 <= delta <= 0.01),
            "regressed": sum(1 for delta in deltas if delta > 0.01),
            "mean_retention": float(np.mean(retentions)),
            "mean_energy_retained": float(np.mean(energies)),
            "mean_deletions": float(np.mean(deletions)),
            "mean_substitutions": float(np.mean(substitutions)),
            "mean_insertions": float(np.mean(insertions)),
        }
    return aggregate


def strategy_definitions_for_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    strategies = [
        {"name": "preserve", "both_mode": "gain", "both_gain": 1.0, "template_source": "dry"},
        {"name": "gain_0p5", "both_mode": "gain", "both_gain": 0.5, "template_source": "dry"},
        {"name": "subtract_0p1", "both_mode": "subtract", "both_subtract_gain": 0.1, "template_source": "dry"},
        {"name": "subtract_0p25", "both_mode": "subtract", "both_subtract_gain": 0.25, "template_source": "dry"},
        {"name": "subtract_0p5", "both_mode": "subtract", "both_subtract_gain": 0.5, "template_source": "dry"},
        {"name": "subtract_0p75", "both_mode": "subtract", "both_subtract_gain": 0.75, "template_source": "dry"},
        {"name": "subtract_1p0", "both_mode": "subtract", "both_subtract_gain": 1.0, "template_source": "dry"},
    ]
    if getattr(args, "cerebellum_model", None):
        strategies.extend([
            {"name": "predicted_subtract_0p25", "both_mode": "subtract", "both_subtract_gain": 0.25, "template_source": "predicted"},
            {"name": "predicted_subtract_0p5", "both_mode": "subtract", "both_subtract_gain": 0.5, "template_source": "predicted"},
        ])
    return strategies


def build_m4_detector_for_self_pcm(
    self_pcm: np.ndarray,
    args: argparse.Namespace,
) -> tuple[SelfVoiceDetector | None, dict[str, Any]]:
    if not getattr(args, "m4_enable", False):
        return None, {"enabled": False}

    registration_pcm = np.asarray(self_pcm, dtype=np.float32)
    registration_mode = "dry_self"
    if getattr(args, "cerebellum_model", None):
        registration_pcm = predict_room_colored_self(registration_pcm, args.cerebellum_model)
        registration_mode = "room_colored_prediction"

    identifier = SpeakerIdentifier(use_real=not getattr(args, "m4_mock", False))
    # Registration threshold: noise filter for SelfVoiceDetector; real decision is in ThalamicAttributor
    # Keep it low (0.05) so the detector doesn't pre-filter — ThalamicAttributor's m4_self_threshold is the real gate
    registration_threshold = min(getattr(args, "m4_registration_threshold", 0.4), 0.05)
    detector = SelfVoiceDetector(identifier, threshold=registration_threshold)
    detector.register_self(registration_pcm)
    return detector, {
        "enabled": True,
        "registration_mode": registration_mode,
        "self_template_samples": int(len(registration_pcm)),
        "registration_threshold": float(getattr(args, "m4_registration_threshold", 0.4)),
        "self_decision_threshold": float(getattr(args, "m4_self_threshold", 0.4)),
        "min_buffer_ms": float(getattr(args, "m4_min_buffer_ms", 500)),
        "mock_embeddings": bool(getattr(args, "m4_mock", False)),
    }


# ---------------------------------------------------------------------------
# Signal-level analysis
# ---------------------------------------------------------------------------

def estimate_self_world_snr(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    self_shift_samples: int,
    world_start_samples: int,
    world_end_samples: int,
) -> dict[str, float]:
    """Estimate self and world signal levels in the captured audio."""
    self_rms = 0.0
    self_db = -100.0

    # Self region: where self PCM is expected
    self_start = max(0, self_shift_samples)
    self_end = min(len(observed), self_shift_samples + len(self_pcm))
    if self_end > self_start:
        self_region = observed[self_start:self_end]
        if len(self_region) > 0:
            rms = float(np.sqrt(np.mean(self_region ** 2)))
            self_rms = rms
            self_db = db(rms)

    # World region: the full world speech window
    w_start = max(0, world_start_samples)
    w_end = min(len(observed), world_end_samples)
    if w_end > w_start:
        world_region = observed[w_start:w_end]
        world_rms = float(np.sqrt(np.mean(world_region ** 2))) if len(world_region) > 0 else 0.0
        world_db = db(world_rms)
    else:
        world_rms = 0.0
        world_db = -100.0

    # Overlap: intersection of self and world
    ov_start = max(self_start, w_start)
    ov_end = min(self_end, w_end)
    if ov_end > ov_start:
        overlap_region = observed[ov_start:ov_end]
        overlap_rms = float(np.sqrt(np.mean(overlap_region ** 2))) if len(overlap_region) > 0 else 0.0
        overlap_db = db(overlap_rms)
    else:
        overlap_rms = 0.0
        overlap_db = -100.0

    # World-only: world region minus overlap (approximate)
    if w_start < ov_start and ov_start > w_start:
        world_only = observed[w_start:min(ov_start, w_end)]
    elif w_end > ov_end and ov_end > w_start:
        world_only = observed[max(ov_end, w_start):w_end]
    else:
        world_only = np.zeros(1)
    world_only_rms = float(np.sqrt(np.mean(world_only ** 2))) if len(world_only) > 0 else 0.0
    world_only_db = db(world_only_rms)

    sw_ratio_db = db(self_rms / max(world_only_rms, 1e-10))

    return {
        "self_rms": self_rms,
        "self_db": self_db,
        "world_rms": world_rms,
        "world_db": world_db,
        "overlap_rms": overlap_rms,
        "overlap_db": overlap_db,
        "world_only_rms": world_only_rms,
        "world_only_db": world_only_db,
        "self_to_world_ratio_db": sw_ratio_db,
    }


# ---------------------------------------------------------------------------
# Core capture + process trial
# ---------------------------------------------------------------------------

def run_capture(
    self_pcm: np.ndarray,
    world_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    noise_device: int,
    lead_in_s: float = 0.5,
    tail_s: float = 1.5,
) -> tuple[np.ndarray, float, float, float]:
    """Run a hardware capture trial. Returns (observed, self_start_s, world_start_s, total_duration)."""
    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR

    world_start_s = lead_in_s
    # Center self within the world speech window
    self_start_s = world_start_s + (world_duration / 2.0) - (self_duration / 2.0)
    total_duration = max(world_start_s + world_duration, self_start_s + self_duration) + tail_s

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    def get_padded_pcm(pcm: np.ndarray, start_s: float, total_s: float, target_sr: float) -> np.ndarray:
        padded = np.zeros(int(total_s * target_sr), dtype=np.float32)
        start_idx = int(start_s * target_sr)
        if abs(target_sr - SR) > 1:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=SR, target_sr=int(target_sr))
        end_idx = min(len(padded), start_idx + len(pcm))
        padded[start_idx:end_idx] = pcm[: end_idx - start_idx]
        return padded

    self_padded = get_padded_pcm(self_pcm, self_start_s, total_duration, output_sr)
    world_padded = get_padded_pcm(world_pcm, world_start_s, total_duration, noise_sr)

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    stream = sd.InputStream(
        device=input_device, channels=1, samplerate=input_sr,
        dtype="float32", blocksize=1024,
    )
    stream.start()

    self_thread = threading.Thread(
        target=play_array_blocking, args=(self_padded, output_device, output_sr))
    world_thread = threading.Thread(
        target=play_array_blocking, args=(world_padded, noise_device, noise_sr))

    self_thread.start()
    world_thread.start()

    total_samples = int(total_duration * input_sr)
    captured_audio = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured_audio[pos:pos + n] = data[:n, 0]
        pos += n

    self_thread.join()
    world_thread.join()
    stream.stop()
    stream.close()

    if input_sr != SR:
        import librosa
        observed = librosa.resample(captured_audio, orig_sr=input_sr, target_sr=SR)
    else:
        observed = captured_audio

    return observed, self_start_s, world_start_s, total_duration


def process_trial(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    self_start_s: float,
    world_start_s: float,
    world_duration_s: float,
    expected_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run attribution, gating, and ASR on a captured observation."""

    self_expected_shift = int((self_start_s + (args.onset_latency_ms + args.calibrated_offset_ms) / 1000.0) * SR)
    world_start_samples = int(world_start_s * SR)
    world_end_samples = int((world_start_s + world_duration_s) * SR)

    # Self-match scoring
    if len(self_pcm) > 1:
        dry_global_score, xcorr_shift = normalized_xcorr(observed, self_pcm)
        dry_local_score, dry_local_shift = local_xcorr(
            observed, self_pcm, self_expected_shift,
            search_ms=args.self_match_search_ms,
        )
    else:
        dry_global_score = 0.0
        xcorr_shift = self_expected_shift
        dry_local_score = 0.0
        dry_local_shift = self_expected_shift

    if args.self_match_mode == "global_xcorr":
        self_score = dry_global_score
        score_shift = xcorr_shift
    elif args.self_match_mode == "local_xcorr":
        self_score = dry_local_score
        score_shift = dry_local_shift
    elif args.self_match_mode == "embedding":
        # compute embedding similarity once per trial
        if not hasattr(process_trial, "_embed_model"):
            if _HAS_EMBED:
                from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
                process_trial._embed_model = Wav2Vec2Model.from_pretrained(args.embedding_model)
                process_trial._embed_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.embedding_model)
            else:
                process_trial._embed_model = None
        if getattr(process_trial, "_embed_model", None) is not None:
            sim, _ = embedding_similarity(
                observed, self_pcm,
                self_expected_shift,
                search_ms=args.self_match_search_ms,
                model=process_trial._embed_model,
                feature_extractor=process_trial._embed_fe,
            )
            self_score = sim
        else:
            self_score = dry_local_score  # fallback
        score_shift = self_expected_shift
    else:
        raise ValueError(f"Unknown self_match_mode: {args.self_match_mode}")

    if args.onset_method == "energy":
        self_shift = energy_onset(observed, self_expected_shift)
    elif args.onset_method == "score":
        self_shift = score_shift
    else:
        self_shift = xcorr_shift

    # Attribution
    m4_detector, m4_info = build_m4_detector_for_self_pcm(self_pcm, args)
    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=args.similarity_threshold,
        causal_tolerance_ms=args.causal_tolerance_ms,
        m4_detector=m4_detector,
        m4_self_threshold=args.m4_self_threshold,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
    )
    vad = SmoothedVAD()

    num_frames = len(observed) // VAD_FRAME_SAMPLES
    verdicts: list[SourceVerdict] = []
    frame_records: list[dict[str, Any]] = []
    speech_frames = 0

    for i in range(num_frames):
        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES
        frame = observed[start:end]

        vad_speech = vad.process_frame(frame)
        if args.speech_override == "true":
            is_speech = True
        elif args.speech_override == "false":
            is_speech = False
        else:
            is_speech = vad_speech
        if is_speech:
            speech_frames += 1

        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = db(rms)
        is_playing = (start >= self_expected_shift) and (end <= self_expected_shift + len(self_pcm))

        if args.self_match_mode == "embedding" and is_playing:
            # compute embedding similarity once for expected window
            if not hasattr(process_trial, "_embed_model"):
                if _HAS_EMBED:
                    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
                    process_trial._embed_model = Wav2Vec2Model.from_pretrained(args.embedding_model)
                    process_trial._embed_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.embedding_model)
                else:
                    process_trial._embed_model = None
            if getattr(process_trial, "_embed_model", None) is not None:
                sim, _ = embedding_similarity(
                    observed, self_pcm,
                    self_expected_shift,
                    search_ms=args.self_match_search_ms,
                    model=process_trial._embed_model,
                    feature_extractor=process_trial._embed_fe,
                )
                match_score = sim
            else:
                match_score = self_score
        else:
            match_score = self_score if is_playing else 0.0
        res = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=self_shift,
            expected_onset_samples=self_expected_shift if is_playing else None,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
            frame_audio=frame,
        )
        verdicts.append(res.verdict)
        frame_records.append({
            "index": i,
            "start_s": start / SR,
            "end_s": end / SR,
            "verdict": res.verdict.name,
            "is_speech": bool(is_speech),
            "rms_db": rms_db,
            "match_score": float(self_score if is_playing else 0.0),
            "is_playing": bool(is_playing),
            "causal_score": float(res.causal_score),
            "onset_error_ms": res.onset_error_ms,
            "m4_attempted": bool(res.metadata and res.metadata.get("m4_attempted", False)),
            "m4_confidence": None if not res.metadata else res.metadata.get("m4_confidence"),
            "m4_buffer_length": None if not res.metadata else res.metadata.get("m4_buffer_length"),
            "m4_outcome": None if not res.metadata else res.metadata.get("m4_outcome"),
        })

    # Gating
    gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), args.both_gain)
    mask = gain > 0.0
    if args.both_mode == "subtract":
        processed = subtract_self_template(
            observed, self_pcm, verdicts, VAD_FRAME_SAMPLES,
            self_shift, args.both_subtract_gain,
        )
        asr_mask = mask.astype(np.float32)
    else:
        processed = observed
        asr_mask = gain

    # ASR
    print("  Running ASR on raw audio...")
    raw_res = transcribe(observed, model_size="base")
    print(f"  Raw ASR: \"{raw_res.text}\"")

    print("  Running ASR on gated audio...")
    gated_res = transcribe(processed, mask=asr_mask, model_size="base")
    print(f"  Gated ASR: \"{gated_res.text}\"")

    raw_wer = word_error_rate(expected_text, raw_res.text)
    gated_wer = word_error_rate(expected_text, gated_res.text)

    # Signal analysis
    snr = estimate_self_world_snr(
        observed, self_pcm, self_shift,
        world_start_samples, world_end_samples,
    )

    return {
        "raw_text": raw_res.text,
        "gated_text": gated_res.text,
        "raw_wer": raw_wer,
        "gated_wer": gated_wer,
        "wer_delta": gated_wer - raw_wer,
        "self_score": self_score,
        "speech_ratio": speech_frames / num_frames,
        "self_frames": sum(1 for v in verdicts if v == SourceVerdict.SELF),
        "both_frames": sum(1 for v in verdicts if v == SourceVerdict.BOTH),
        "world_frames": sum(1 for v in verdicts if v == SourceVerdict.WORLD),
        "snr": snr,
        "frame_records": frame_records,
        "observed": observed,
        "mask": mask,
        "gain": gain,
        "processed": processed,
        "verdicts": verdicts,
        "self_pcm": self_pcm,
        "self_shift": self_shift,
        "self_expected_shift": self_expected_shift,
        "m4": m4_info,
        "m4_triggered_frames": sum(1 for row in frame_records if row.get("m4_confidence") is not None),
    }


# ---------------------------------------------------------------------------
# Calibration mode
# ---------------------------------------------------------------------------

def run_calibration(args: argparse.Namespace) -> str:
    """Sweep self-amplitude to find the operating point where world speech
    starts to become unrecoverable. Returns the save directory path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_root, f"calibration_{ts}")
    os.makedirs(save_root, exist_ok=True)

    # Load world speech
    if args.world_speech.endswith(".npy"):
        world_pcm = np.load(args.world_speech)
    else:
        print(f"ERROR: world speech file not found: {args.world_speech}")
        sys.exit(1)

    # Amplitude sweep
    amplitudes = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]
    results = []

    print("=" * 60)
    print("M2 BENCHMARK — Calibration Mode")
    print(f"  Self stimulus: {args.self_stim}")
    print(f"  Self duration: {args.self_duration_ms}ms")
    print(f"  Reference text: \"{args.expected_text}\"")
    print(f"  World speech: {args.world_speech} ({len(world_pcm)/SR:.2f}s)")
    print("=" * 60)

    for amp in amplitudes:
        print(f"\n--- Amplitude {amp:.2f} ---")
        if amp == 0.0:
            # World-only control
            self_pcm = np.zeros(1, dtype=np.float32)
        else:
            meta = generate_stimulus_with_metadata(
                args.self_stim, args.self_duration_ms, amp, seed=0,
            )
            self_pcm = meta["pcm"]

        try:
            observed, self_start_s, world_start_s, total_dur = run_capture(
                self_pcm, world_pcm,
                args.input_device, args.output_device, args.noise_device,
            )
        except Exception as e:
            print(f"  CAPTURE ERROR: {e}")
            results.append({"amplitude": amp, "error": str(e)})
            continue

        trial = process_trial(
            observed, self_pcm, self_start_s, world_start_s,
            len(world_pcm) / SR, args.expected_text, args,
        )

        # Save artifacts
        amp_dir = os.path.join(save_root, f"amp_{amp:.2f}")
        os.makedirs(amp_dir, exist_ok=True)
        np.save(os.path.join(amp_dir, "observed.npy"), observed)
        np.save(os.path.join(amp_dir, "gating_mask.npy"), trial["mask"].astype(np.uint8))
        np.save(os.path.join(amp_dir, "gating_gain.npy"), trial["gain"].astype(np.float32))
        np.save(os.path.join(amp_dir, "processed.npy"), trial["processed"].astype(np.float32))
        # Save self PCM and shift for offline subtraction replay
        # (trial["self_pcm"] is the generated PCM; trial["self_shift"] is the onset)
        np.save(os.path.join(amp_dir, "self_pcm.npy"), trial.get("self_pcm", np.zeros(1, dtype=np.float32)))
        with open(os.path.join(amp_dir, "frame_verdicts.jsonl"), "w") as f:
            for row in trial["frame_records"]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        summary = {
            "amplitude": amp,
            "raw_wer": trial["raw_wer"],
            "gated_wer": trial["gated_wer"],
            "wer_delta": trial["wer_delta"],
            "self_score": trial["self_score"],
            "self_frames": trial["self_frames"],
            "both_frames": trial["both_frames"],
            "world_frames": trial["world_frames"],
            "m4": trial["m4"],
            "m4_triggered_frames": trial["m4_triggered_frames"],
            "self_shift_samples": int(trial["self_shift"]),
            "self_expected_shift_samples": int(trial["self_expected_shift"]),
            "snr": trial["snr"],
        }
        with open(os.path.join(amp_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results.append(summary)

        print(f"  Raw WER:   {trial['raw_wer']:.3f}")
        print(f"  Gated WER: {trial['gated_wer']:.3f}")
        print(f"  WER delta: {trial['wer_delta']:+.3f}")
        print(f"  Self RMS:  {trial['snr']['self_rms']:.4f} ({trial['snr']['self_db']:.1f} dB)")
        print(f"  World RMS: {trial['snr']['world_only_rms']:.4f} ({trial['snr']['world_only_db']:.1f} dB)")
        print(f"  S/W ratio: {trial['snr']['self_to_world_ratio_db']:.1f} dB")
        print(f"  Verdicts:  SELF={trial['self_frames']} BOTH={trial['both_frames']} WORLD={trial['world_frames']}")

    # Save aggregate
    with open(os.path.join(save_root, "calibration_summary.json"), "w") as f:
        json.dump({
            "config": {
                "self_stim": args.self_stim,
                "self_duration_ms": args.self_duration_ms,
                "expected_text": args.expected_text,
                "world_speech": args.world_speech,
                "similarity_threshold": args.similarity_threshold,
                "onset_latency_ms": args.onset_latency_ms,
                "calibrated_offset_ms": args.calibrated_offset_ms,
                "onset_method": args.onset_method,
                "causal_tolerance_ms": args.causal_tolerance_ms,
                "both_gain": args.both_gain,
                "both_mode": args.both_mode,
                "m4_min_buffer_ms": args.m4_min_buffer_ms,
                "m4_self_threshold": args.m4_self_threshold,
                "m4_registration_threshold": args.m4_registration_threshold,
            },
            "results": results,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print(f"{'Amplitude':>10} {'Raw WER':>9} {'Gated WER':>10} {'Delta':>7} {'S/W dB':>8} {'Verdicts':>20}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['amplitude']:>10.2f} ERROR: {r['error']}")
            continue
        verdicts = f"S{r['self_frames']}/B{r['both_frames']}/W{r['world_frames']}"
        print(
            f"{r['amplitude']:>10.2f} "
            f"{r['raw_wer']:>9.3f} "
            f"{r['gated_wer']:>10.3f} "
            f"{r['wer_delta']:>+7.3f} "
            f"{r['snr']['self_to_world_ratio_db']:>8.1f} "
            f"{verdicts:>20}"
        )
    print("=" * 60)
    print(f"\nArtifacts saved to: {save_root}")
    print(f"Aggregate summary:  {save_root}/calibration_summary.json")
    print()
    print("Next step: choose an amplitude where raw WER is between ~0.1 and ~0.3")
    print("and run the full benchmark with multiple seeds:")
    print(f"  ./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \\")
    print(f"    --self-stim {args.self_stim} --self-amplitude <chosen> \\")
    print(f"    --input-device {args.input_device} --output-device {args.output_device} --noise-device {args.noise_device} \\")
    print(f"    --save-root {args.save_root}")

    return save_root


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> str:
    """Run the full benchmark at a fixed amplitude with multiple seeds."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_root, f"benchmark_{ts}")
    os.makedirs(save_root, exist_ok=True)

    if args.world_speech.endswith(".npy"):
        world_pcm = np.load(args.world_speech)
    else:
        print(f"ERROR: world speech file not found: {args.world_speech}")
        sys.exit(1)

    seeds = list(range(args.num_seeds))
    results = []

    print("=" * 60)
    print("M2 BENCHMARK — Full Run")
    print(f"  Self stimulus: {args.self_stim}")
    print(f"  Self amplitude: {args.self_amplitude}")
    print(f"  Self duration: {args.self_duration_ms}ms")
    print(f"  Seeds: {seeds}")
    print(f"  Reference text: \"{args.expected_text}\"")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        meta = generate_stimulus_with_metadata(
            args.self_stim, args.self_duration_ms, args.self_amplitude, seed=seed,
        )
        self_pcm = meta["pcm"]

        try:
            observed, self_start_s, world_start_s, total_dur = run_capture(
                self_pcm, world_pcm,
                args.input_device, args.output_device, args.noise_device,
            )
        except Exception as e:
            print(f"  CAPTURE ERROR: {e}")
            results.append({"seed": seed, "error": str(e)})
            continue

        trial = process_trial(
            observed, self_pcm, self_start_s, world_start_s,
            len(world_pcm) / SR, args.expected_text, args,
        )

        # Save artifacts
        seed_dir = os.path.join(save_root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        np.save(os.path.join(seed_dir, "observed.npy"), observed)
        np.save(os.path.join(seed_dir, "gating_mask.npy"), trial["mask"].astype(np.uint8))
        np.save(os.path.join(seed_dir, "gating_gain.npy"), trial["gain"].astype(np.float32))
        np.save(os.path.join(seed_dir, "processed.npy"), trial["processed"].astype(np.float32))
        np.save(os.path.join(seed_dir, "self_pcm.npy"), trial["self_pcm"])
        with open(os.path.join(seed_dir, "frame_verdicts.jsonl"), "w") as f:
            for row in trial["frame_records"]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        summary = {
            "seed": seed,
            "raw_wer": trial["raw_wer"],
            "gated_wer": trial["gated_wer"],
            "wer_delta": trial["wer_delta"],
            "self_score": trial["self_score"],
            "self_frames": trial["self_frames"],
            "both_frames": trial["both_frames"],
            "world_frames": trial["world_frames"],
            "m4": trial["m4"],
            "m4_triggered_frames": trial["m4_triggered_frames"],
            "self_shift_samples": int(trial["self_shift"]),
            "self_expected_shift_samples": int(trial["self_expected_shift"]),
            "snr": trial["snr"],
        }
        with open(os.path.join(seed_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results.append(summary)

        print(f"  Raw WER:   {trial['raw_wer']:.3f}")
        print(f"  Gated WER: {trial['gated_wer']:.3f}")
        print(f"  WER delta: {trial['wer_delta']:+.3f}")

    # Aggregate
    valid = [r for r in results if "error" not in r]
    if valid:
        agg = {
            "mean_raw_wer": float(np.mean([r["raw_wer"] for r in valid])),
            "mean_gated_wer": float(np.mean([r["gated_wer"] for r in valid])),
            "mean_wer_delta": float(np.mean([r["wer_delta"] for r in valid])),
            "improved": sum(1 for r in valid if r["wer_delta"] < 0),
            "neutral": sum(1 for r in valid if r["wer_delta"] == 0),
            "regressed": sum(1 for r in valid if r["wer_delta"] > 0),
            "mean_self_world_db": float(np.mean([r["snr"]["self_to_world_ratio_db"] for r in valid])),
        }
    else:
        agg = {"error": "no valid results"}

    with open(os.path.join(save_root, "benchmark_summary.json"), "w") as f:
        json.dump({
            "config": {
                "self_stim": args.self_stim,
                "self_amplitude": args.self_amplitude,
                "self_duration_ms": args.self_duration_ms,
                "num_seeds": args.num_seeds,
                "expected_text": args.expected_text,
                "world_speech": args.world_speech,
                "similarity_threshold": args.similarity_threshold,
                "onset_latency_ms": args.onset_latency_ms,
                "calibrated_offset_ms": args.calibrated_offset_ms,
                "onset_method": args.onset_method,
                "causal_tolerance_ms": args.causal_tolerance_ms,
                "both_gain": args.both_gain,
                "both_mode": args.both_mode,
                "m4_min_buffer_ms": args.m4_min_buffer_ms,
                "m4_self_threshold": args.m4_self_threshold,
                "m4_registration_threshold": args.m4_registration_threshold,
            },
            "aggregate": agg,
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print(f"  Mean raw WER:   {agg['mean_raw_wer']:.3f}")
    print(f"  Mean gated WER: {agg['mean_gated_wer']:.3f}")
    print(f"  Mean WER delta: {agg['mean_wer_delta']:+.3f}")
    print(f"  Improved/Neutral/Regressed: {agg['improved']}/{agg['neutral']}/{agg['regressed']}")
    print(f"  Mean S/W ratio: {agg['mean_self_world_db']:.1f} dB")
    print("=" * 60)
    print(f"\nArtifacts saved to: {save_root}")

    return save_root


# ---------------------------------------------------------------------------
# Offline replay mode
# ---------------------------------------------------------------------------

def run_offline(args: argparse.Namespace) -> str:
    """Replay multiple gating strategies against saved captures."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_root, f"offline_{ts}")
    os.makedirs(save_root, exist_ok=True)

    # Find all capture directories under capture_root
    capture_dirs = sorted([
        os.path.join(args.capture_root, d)
        for d in os.listdir(args.capture_root)
        if os.path.isdir(os.path.join(args.capture_root, d))
        and os.path.exists(os.path.join(args.capture_root, d, "observed.npy"))
    ])

    if not capture_dirs:
        print(f"ERROR: no capture directories found under {args.capture_root}")
        sys.exit(1)

    print(f"Found {len(capture_dirs)} captures")

    strategies = strategy_definitions_for_args(args)
    all_results = []

    for cap_dir in capture_dirs:
        cap_name = os.path.basename(cap_dir)
        print(f"\n--- {cap_name} ---")

        observed = np.load(os.path.join(cap_dir, "observed.npy"))
        with open(os.path.join(cap_dir, "summary.json")) as f:
            cap_summary = json.load(f)

        verdicts = []
        frame_records = []
        if os.path.exists(os.path.join(cap_dir, "frame_verdicts.jsonl")):
            with open(os.path.join(cap_dir, "frame_verdicts.jsonl")) as f:
                for line in f:
                    row = json.loads(line)
                    verdicts.append(SourceVerdict[row["verdict"]])
                    frame_records.append(row)

        self_pcm = None
        self_shift = cap_summary.get("self_shift_samples")
        if os.path.exists(os.path.join(cap_dir, "self_pcm.npy")):
            self_pcm = np.load(os.path.join(cap_dir, "self_pcm.npy"))
            if self_shift is None:
                for fr in frame_records:
                    if fr.get("is_playing", False):
                        self_shift = int(fr["start_s"] * SR)
                        break

        predicted_self_pcm = None
        predicted_self_onset = 0
        if self_pcm is not None and getattr(args, "cerebellum_model", None):
            predicted_self_pcm = predict_room_colored_self(self_pcm, args.cerebellum_model)
            predicted_self_onset = estimate_onset_samples(predicted_self_pcm)

        cap_results = {"capture": cap_name, "strategies": []}

        for strat in strategies:
            print(f"  Strategy: {strat['name']}")

            if not verdicts:
                print("    SKIP: no frame verdicts found")
                continue

            gain = build_gating_gain(
                verdicts, VAD_FRAME_SAMPLES, len(observed),
                strat.get("both_gain", 1.0),
            )
            mask = gain > 0.0

            if strat["both_mode"] == "subtract":
                template_source = strat.get("template_source", "dry")
                if template_source == "predicted":
                    if predicted_self_pcm is None or self_shift is None:
                        print("    SKIP: predicted self template not available")
                        continue
                    subtract_pcm = predicted_self_pcm
                    template_onset_samples = predicted_self_onset
                else:
                    if self_pcm is None or self_shift is None:
                        print("    SKIP: self_pcm not available in capture")
                        continue
                    subtract_pcm = self_pcm
                    template_onset_samples = 0
                processed = subtract_self_template(
                    observed, subtract_pcm, verdicts,
                    VAD_FRAME_SAMPLES, self_shift,
                    strat.get("both_subtract_gain", 0.25),
                    template_onset_samples=template_onset_samples,
                )
                asr_mask = mask.astype(np.float32)
            else:
                processed = observed
                asr_mask = gain

            raw_res = transcribe(observed, model_size="base")
            gated_res = transcribe(processed, mask=asr_mask, model_size="base")

            raw_text = raw_res.text
            gated_text = gated_res.text
            raw_wer = word_error_rate(args.expected_text, raw_text)
            gated_wer = word_error_rate(args.expected_text, gated_text)

            retention = word_retention_rate(args.expected_text, gated_text)
            errors = error_breakdown(args.expected_text, gated_text)

            overlap_start = None
            overlap_end = None
            for fr in frame_records:
                if fr.get("verdict") == "BOTH":
                    if overlap_start is None:
                        overlap_start = fr["start_s"]
                    overlap_end = fr["end_s"]
            energy_retained = 0.0
            if overlap_start is not None and overlap_end is not None:
                energy_retained = energy_retained_in_window(
                    observed, processed, overlap_start, overlap_end, SR
                )

            print(f"    Raw WER:   {raw_wer:.3f}  (\"{raw_text[:50]}...\")")
            print(f"    Gated WER: {gated_wer:.3f}  (\"{gated_text[:50]}...\")")
            print(f"    Delta: {gated_wer - raw_wer:+.3f}  Retention: {retention:.2f}  Energy: {energy_retained:.2f}")
            print(f"    Errors: D={errors['deletions']} S={errors['substitutions']} I={errors['insertions']}")

            cap_results["strategies"].append({
                "name": strat["name"],
                "raw_wer": raw_wer,
                "gated_wer": gated_wer,
                "wer_delta": gated_wer - raw_wer,
                "retention": retention,
                "errors": errors,
                "energy_retained": energy_retained,
            })

        all_results.append(cap_results)

    aggregate = summarize_offline_results(all_results)
    with open(os.path.join(save_root, "offline_summary.json"), "w") as f:
        json.dump({"results": all_results, "strategies": aggregate}, f, indent=2)

    if aggregate:
        print("\nStrategy aggregate:")
        for name, row in aggregate.items():
            print(
                f"  {name}: mean delta {row['mean_wer_delta']:+.3f} "
                f"({row['improved']}/{row['neutral']}/{row['regressed']} I/N/R)"
            )
    print(f"\nOffline results saved to: {save_root}")
    return save_root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="M2 Benchmark — Controlled speech-preservation measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["calibrate", "benchmark", "offline"], required=True,
        help="calibrate: sweep self-amplitude to find operating point; "
             "benchmark: run multi-seed benchmark at fixed amplitude; "
             "offline: replay strategies against saved captures",
    )
    parser.add_argument("--self-stim", default="vowel_ah_var_nearfield")
    parser.add_argument("--self-amplitude", type=float, default=0.15)
    parser.add_argument("--self-duration-ms", type=int, default=350)
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--expected-text", default=REFERENCE_TEXT)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--save-root", default="data/m2_benchmark")
    parser.add_argument("--num-seeds", type=int, default=5)

    # Attribution params (defaults from calibrated M2 settings)
    parser.add_argument("--similarity-threshold", type=float, default=0.20)
    parser.add_argument("--onset-latency-ms", type=float, default=400.0)
    parser.add_argument(
        "--calibrated-offset-ms", type=float, default=0.0,
        help="Additional offset (ms) from calibration to add to onset_latency_ms",
    )
    parser.add_argument("--onset-method", choices=["xcorr", "energy", "score"], default="energy")
    parser.add_argument("--causal-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--self-match-mode", choices=["global_xcorr", "local_xcorr", "embedding"], default="global_xcorr")
    parser.add_argument("--embedding-model", default="facebook/wav2vec2-base")
    parser.add_argument("--self-match-search-ms", type=float, default=250.0)

    # Gating params
    parser.add_argument("--both-gain", type=float, default=0.5)
    parser.add_argument("--both-mode", choices=["gain", "subtract"], default="gain")
    parser.add_argument("--both-subtract-gain", type=float, default=0.25)

    parser.add_argument("--speech-override", choices=["vad", "true", "false"], default="vad")

    # Offline mode
    parser.add_argument("--capture-root", default=None, help="Root directory containing capture subdirectories")
    parser.add_argument(
        "--cerebellum-model",
        default=None,
        help="Optional saved BabblingCerebellum JSON. When provided, offline mode also evaluates room-colored predicted-self subtraction.",
    )

    # M4 / self-voice override
    parser.add_argument("--m4-enable", action="store_true", help="Enable M4 self-voice override inside the attribution loop")
    parser.add_argument("--m4-self-threshold", type=float, default=0.4, help="Decision threshold applied by ThalamicAttributor to M4 similarity")
    parser.add_argument("--m4-registration-threshold", type=float, default=0.4, help="Threshold stored on the SelfVoiceDetector during template registration")
    parser.add_argument("--m4-mock", action="store_true", help="Force mock speaker embeddings for deterministic smoke tests")
    parser.add_argument("--m4-min-buffer-ms", type=int, default=500, help="Minimum buffer size (ms) for M4 accumulation before classification")

    args = parser.parse_args()

    if args.mode == "calibrate":
        run_calibration(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "offline":
        if not args.capture_root:
            # Try to find the most recent calibration/benchmark
            if os.path.exists(args.save_root):
                subdirs = sorted(os.listdir(args.save_root))
                if subdirs:
                    args.capture_root = os.path.join(args.save_root, subdirs[-1])
                    print(f"Using capture root: {args.capture_root}")
                else:
                    print(f"ERROR: --capture-root required for offline mode")
                    sys.exit(1)
        run_offline(args)


if __name__ == "__main__":
    main()