#!/usr/bin/env python3
"""M2 — Speech-Aware Self/World Separation Harness.

Goal: Suppress "Self" audio while preserving "World" speech.

Execution:
1. Play Self-Babble (Device 4).
2. Play World-Speech (Device 3).
3. Capture and use ThalamicAttributor + VAD to label segments.
4. Run Whisper ASR on both raw and gated audio to measure preservation.
5. Optionally score against a reference transcript and save artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

# Add corollary packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from stg.asr import transcribe
from m2_policy.policy import select_family_policy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from m1_tone_attribution import generate_stimulus_with_metadata


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def normalize_text(text: str) -> str:
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


def play_array_blocking(data: np.ndarray, device: int, samplerate: float) -> None:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(device=device, channels=data.shape[1], samplerate=samplerate, dtype="float32") as stream:
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
    observed: np.ndarray,
    template: np.ndarray,
    expected_shift: int,
    search_ms: float = 250.0,
) -> tuple[float, int]:
    """Score a template near the expected motor-window shift."""
    if len(observed) < len(template):
        return 0.0, expected_shift
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-9:
        return 0.0, expected_shift
    radius = int(search_ms * SR / 1000.0)
    start = max(0, expected_shift - radius)
    stop = min(len(observed) - len(template), expected_shift + radius)
    if stop <= start:
        return 0.0, expected_shift
    best_score = -1.0
    best_shift = start
    step = max(len(template) // 160, 1)
    for shift in range(start, stop + 1, step):
        window = observed[shift:shift + len(template)]
        denom = float(np.linalg.norm(window) * template_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def fit_room_filter(
    observed: np.ndarray,
    template: np.ndarray,
    shift: int,
    taps: int = 64,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Fit a small FIR mapping dry self PCM to observed mic capture."""
    dry = np.zeros(len(observed), dtype=np.float32)
    start = max(0, int(shift))
    src_start = 0 if shift >= 0 else -int(shift)
    n = min(len(template) - src_start, len(observed) - start)
    if n > 0:
        dry[start:start + n] = template[src_start:src_start + n]
    fit_start = max(taps - 1, start + taps - 1)
    fit_stop = min(len(observed), start + n)
    if fit_stop <= fit_start + taps * 2:
        return np.r_[1.0, np.zeros(taps - 1, dtype=np.float64)]
    idx = np.arange(fit_start, fit_stop)
    if len(idx) > 6000:
        idx = idx[np.linspace(0, len(idx) - 1, 6000).astype(int)]
    x = np.stack([dry[idx - k] for k in range(taps)], axis=1).astype(np.float64)
    y = observed[idx].astype(np.float64)
    xtx = x.T @ x + ridge * np.eye(taps)
    xty = x.T @ y
    try:
        h = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.r_[1.0, np.zeros(taps - 1, dtype=np.float64)]
    return np.clip(h, -2.0, 2.0)


def apply_room_filter(template: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(template, h, mode="full").astype(np.float32)


def build_gating_gain(
    verdicts: list[SourceVerdict],
    frame_samples: int,
    total_samples: int,
    both_gain: float,
) -> np.ndarray:
    """Build a sample-level gain vector from frame verdicts."""
    gain = np.ones(total_samples, dtype=np.float32)
    both_gain = float(np.clip(both_gain, 0.0, 1.0))

    for i, verdict in enumerate(verdicts):
        start = i * frame_samples
        end = min(start + frame_samples, total_samples)
        if verdict == SourceVerdict.SELF:
            gain[start:end] = 0.0
        elif verdict == SourceVerdict.BOTH:
            gain[start:end] = both_gain

    return gain




def subtract_self_template(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    verdicts: list[SourceVerdict],
    frame_samples: int,
    self_shift: int,
    both_subtract_gain: float,
) -> np.ndarray:
    """Subtract aligned self-template only inside BOTH frames."""
    out = observed.astype(np.float32).copy()
    if len(self_pcm) == 0:
        return out

    template = np.zeros_like(out)
    start = max(0, int(self_shift))
    src_start = 0
    if self_shift < 0:
        src_start = -int(self_shift)
        start = 0
    n = min(len(self_pcm) - src_start, len(out) - start)
    if n <= 0:
        return out

    # Fit a single conservative scale using BOTH-window samples only.
    both_mask = np.zeros(len(out), dtype=bool)
    for i, verdict in enumerate(verdicts):
        if verdict == SourceVerdict.BOTH:
            a = i * frame_samples
            b = min(a + frame_samples, len(out))
            both_mask[a:b] = True
    if not np.any(both_mask):
        return out

    template[start:start+n] = self_pcm[src_start:src_start+n]
    t = template[both_mask]
    y = out[both_mask]
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return out
    scale = float(np.dot(y, t) / denom)
    scale = float(np.clip(scale, 0.0, 2.0)) * float(np.clip(both_subtract_gain, 0.0, 1.0))
    out[both_mask] = y - scale * t
    return out

def energy_onset(
    observed: np.ndarray,
    start_hint_samples: int,
    search_before_ms: float = 0.0,
    search_after_ms: float = 800.0,
    frame_samples: int = VAD_FRAME_SAMPLES,
    threshold_ratio: float = 0.20,
) -> int:
    """Estimate onset as first local frame above a fraction of local peak RMS."""
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

def run_m2_trial(
    self_stim: str,
    world_speech_file: str,
    input_device: int,
    output_device: int,
    noise_device: int,
    lead_in_s: float = 0.5,
    tail_s: float = 1.0,
    seed: int = 0,
    expected_text: str | None = None,
    both_gain: float = 0.5,
    both_mode: str = "gain",
    both_subtract_gain: float = 0.5,
    both_policy: str = "manual",
    family_gain_overrides: dict[str, float] | None = None,
    self_duration_ms: int = 350,
    self_amplitude: float = 0.7,
    self_only: bool = False,
    world_only: bool = False,
    similarity_threshold: float = 0.45,
    speech_override: str = "vad",
    onset_latency_ms: float = 0.0,
    onset_method: str = "xcorr",
    causal_tolerance_ms: float = 25.0,
    self_pcm_override: np.ndarray | None = None,
    self_match_mode: str = "global_xcorr",
    self_match_search_ms: float = 250.0,
    room_filter_taps: int = 64,
    room_filter_ridge: float = 1e-4,
    # M4 self-voice detection
    m4_enable: bool = False,
    m4_min_buffer_ms: float = 60.0,
    m4_self_threshold: float = 0.50,
    cerebellum_model: str | None = None,
) -> dict[str, Any]:
    if both_policy == "family":
        decision = select_family_policy(self_stim, family_gain_overrides=family_gain_overrides)
        both_mode = decision.mode
        both_gain = decision.gain
        both_subtract_gain = decision.subtract_gain
        both_policy_reason = decision.reason
    else:
        both_policy_reason = "manual"

    if self_pcm_override is None:
        self_meta = generate_stimulus_with_metadata(self_stim, self_duration_ms, self_amplitude, seed=seed)
        self_pcm = self_meta["pcm"]
    else:
        self_pcm = np.asarray(self_pcm_override, dtype=np.float32)
        if self_pcm.ndim != 1:
            raise ValueError("self_pcm_override must be mono float PCM")
    
    # Build M4 detector if enabled
    m4_detector, m4_info = None, {}
    if m4_enable:  # Directly use parameter, not locals()
        from scripts.m2_benchmark import build_m4_detector_for_self_pcm
        args_ns = argparse.Namespace(
            m4_registration_threshold=getattr(locals(), 'm4_registration_threshold', 0.05),
            m4_enable=True,
            m4_mock=getattr(locals(), 'm4_mock', False),
            m4_min_buffer_ms=getattr(locals(), 'm4_min_buffer_ms', 60.0),
            m4_self_threshold=getattr(locals(), 'm4_self_threshold', 0.50),
            cerebellum_model=cerebellum_model,
        )
        m4_detector, m4_info = build_m4_detector_for_self_pcm(self_pcm, args_ns)
    
    if world_speech_file.endswith(".npy"):
        world_pcm = np.load(world_speech_file)
    else:
        world_pcm = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32) * 0.5

    if self_only and world_only:
        raise ValueError("--self-only and --world-only are mutually exclusive")
    if self_only:
        world_pcm = np.zeros_like(world_pcm)
    if world_only:
        self_pcm = np.zeros_like(self_pcm)

    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR

    world_start_s = lead_in_s
    self_start_s = world_start_s + (world_duration / 2.0) - (self_duration / 2.0)
    total_duration = max(world_start_s + world_duration, self_start_s + self_duration) + tail_s

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    def get_padded_pcm(pcm: np.ndarray, start_s: float, total_s: float, target_sr: float) -> np.ndarray:
        # Clamp start_s to non-negative to avoid negative padding
        start_s = max(0.0, start_s)
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

    print(f"\n--- M2 Speech Preservation Trial ---")
    print(f"Self: {self_stim} at {self_start_s:.2f}s")
    print(f"World Speech: {world_duration:.2f}s at {world_start_s:.2f}s")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()

    self_thread = threading.Thread(target=play_array_blocking, args=(self_padded, output_device, output_sr))
    world_thread = threading.Thread(target=play_array_blocking, args=(world_padded, noise_device, noise_sr))

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

    # Build ThalamicAttributor with optional M4 support
    # m4_detector is set in lines 316-323 if m4_enable=True
    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=similarity_threshold,
        causal_tolerance_ms=causal_tolerance_ms,
        m4_detector=m4_detector,  # Use the detector from lines 316-327 if m4_enable=True
        m4_self_threshold=m4_self_threshold,
        m4_min_buffer_ms=m4_min_buffer_ms,
    )
    vad = SmoothedVAD()

    self_expected_shift = int((self_start_s + onset_latency_ms / 1000.0) * SR)
    dry_global_score, xcorr_shift = normalized_xcorr(observed, self_pcm)
    dry_local_score, dry_local_shift = local_xcorr(
        observed,
        self_pcm,
        self_expected_shift,
        search_ms=self_match_search_ms,
    )
    room_local_score = None
    room_local_shift = None
    if self_match_mode == "global_xcorr":
        self_score = dry_global_score
        score_shift = xcorr_shift
    elif self_match_mode == "local_xcorr":
        self_score = dry_local_score
        score_shift = dry_local_shift
    elif self_match_mode == "room_local_oracle":
        # Experimental/oracle mode: fit room color on this same capture, then
        # score locally. Use only for analysis until held-out room filters are wired.
        h = fit_room_filter(
            observed,
            self_pcm,
            dry_local_shift,
            taps=room_filter_taps,
            ridge=room_filter_ridge,
        )
        room_ref = apply_room_filter(self_pcm, h)
        room_local_score, room_local_shift = local_xcorr(
            observed,
            room_ref,
            self_expected_shift,
            search_ms=self_match_search_ms,
        )
        self_score = room_local_score
        score_shift = room_local_shift
    else:
        raise ValueError(f"Unknown self_match_mode: {self_match_mode}")

    if onset_method == "energy":
        self_shift = energy_onset(observed, self_expected_shift)
    elif onset_method == "score":
        self_shift = score_shift
    else:
        self_shift = xcorr_shift

    num_frames = len(observed) // VAD_FRAME_SAMPLES
    verdicts = []
    frame_records: list[dict[str, Any]] = []
    speech_frames = 0

    for i in range(num_frames):
        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES
        frame = observed[start:end]

        vad_speech = vad.process_frame(frame)
        if speech_override == "true":
            is_speech = True
        elif speech_override == "false":
            is_speech = False
        else:
            is_speech = vad_speech
        if is_speech:
            speech_frames += 1

        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = db(rms)
        is_playing = (start < self_expected_shift + len(self_pcm)) and (end > self_expected_shift)

        res = attributor.attribute(
            match_score=self_score if is_playing else 0.0,
            observed_onset_samples=self_shift,
            expected_onset_samples=self_expected_shift if is_playing else None,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
            frame_audio=frame,
            self_pcm=self_pcm if is_playing else None,
            agent_start_sample=0,
        )
        verdicts.append(res.verdict)
        frame_records.append({
            "index": i,
            "start_sample": start,
            "end_sample": end,
            "start_s": start / SR,
            "end_s": end / SR,
            "verdict": res.verdict.name,
            "is_speech": bool(is_speech),
            "rms_db": rms_db,
            "match_score": float(self_score if is_playing else 0.0),
            "is_playing": bool(is_playing),
            "causal_score": float(res.causal_score),
            "onset_error_ms": res.onset_error_ms,
            # M4 metadata
            "m4_outcome": res.metadata.get("m4_outcome") if res.metadata else None,
            "m4_confidence": res.metadata.get("m4_confidence") if res.metadata else None,
            "m4_attempted": res.metadata.get("m4_attempted") if res.metadata else None,
        })

    gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), both_gain)
    mask = gain > 0.0
    if both_mode == "subtract":
        processed = subtract_self_template(
            observed,
            self_pcm,
            verdicts,
            VAD_FRAME_SAMPLES,
            self_shift,
            both_subtract_gain,
        )
        asr_mask = mask.astype(np.float32)
    else:
        processed = observed
        asr_mask = gain

    print("\nRunning ASR on Raw Audio...")
    raw_res = transcribe(observed, model_size="base")
    print(f"Raw ASR: \"{raw_res.text}\"")

    print("\nRunning ASR on Gated Audio...")
    gated_res = transcribe(processed, mask=asr_mask, model_size="base")
    print(f"Gated ASR: \"{gated_res.text}\"")

    metrics: dict[str, Any] = {}
    if expected_text:
        raw_wer = word_error_rate(expected_text, raw_res.text)
        gated_wer = word_error_rate(expected_text, gated_res.text)
        metrics = {
            "expected_text": expected_text,
            "raw_wer": raw_wer,
            "gated_wer": gated_wer,
            "wer_delta": gated_wer - raw_wer,
        }
        print("\nReference Transcript Metrics:")
        print(f"Expected: \"{expected_text}\"")
        print(f"Raw WER:   {raw_wer:.3f}")
        print(f"Gated WER: {gated_wer:.3f}")

    return {
        "raw_text": raw_res.text,
        "gated_text": gated_res.text,
        "self_score": self_score,
        "speech_ratio": speech_frames / num_frames,
        "self_frames": sum(1 for v in verdicts if v == SourceVerdict.SELF),
        "both_frames": sum(1 for v in verdicts if v == SourceVerdict.BOTH),
        "world_frames": sum(1 for v in verdicts if v == SourceVerdict.WORLD),
        "metrics": metrics,
        "frame_records": frame_records,
        "observed": observed,
        "mask": mask,
        "gain": gain,
        "processed": processed,
        "both_gain": both_gain,
        "both_mode": both_mode,
        "both_subtract_gain": both_subtract_gain,
        "both_policy": both_policy,
        "both_policy_reason": both_policy_reason,
        "family_gain_overrides": family_gain_overrides,
        "seed": seed,
        "self_duration_ms": self_duration_ms,
        "self_amplitude": self_amplitude,
        "self_pcm_override": self_pcm_override is not None,
        "self_only": self_only,
        "world_only": world_only,
        "similarity_threshold": similarity_threshold,
        "speech_override": speech_override,
        "onset_latency_ms": onset_latency_ms,
        "onset_method": onset_method,
        "self_shift_samples": int(self_shift),
        "self_xcorr_shift_samples": int(xcorr_shift),
        "self_expected_shift_samples": int(self_expected_shift),
        "self_match_mode": self_match_mode,
        "self_match_search_ms": self_match_search_ms,
        "dry_global_self_score": dry_global_score,
        "dry_local_self_score": dry_local_score,
        "dry_local_shift_samples": int(dry_local_shift),
        "room_local_self_score": room_local_score,
        "room_local_shift_samples": None if room_local_shift is None else int(room_local_shift),
        "room_filter_taps": room_filter_taps,
        "room_filter_ridge": room_filter_ridge,
        "self_onset_error_ms": float((self_shift - self_expected_shift) * 1000.0 / SR),
        "causal_tolerance_ms": causal_tolerance_ms,
    }


def save_trial_artifacts(save_dir: str, result: dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "observed.npy"), result["observed"])
    np.save(os.path.join(save_dir, "gating_mask.npy"), result["mask"].astype(np.uint8))
    np.save(os.path.join(save_dir, "gating_gain.npy"), result["gain"].astype(np.float32))
    np.save(os.path.join(save_dir, "processed.npy"), result["processed"].astype(np.float32))
    with open(os.path.join(save_dir, "frame_verdicts.jsonl"), "w", encoding="utf-8") as f:
        for row in result["frame_records"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "raw_text": result["raw_text"],
        "gated_text": result["gated_text"],
        "self_score": result["self_score"],
        "speech_ratio": result["speech_ratio"],
        "self_frames": result["self_frames"],
        "both_frames": result["both_frames"],
        "world_frames": result["world_frames"],
        "both_gain": result["both_gain"],
        "both_mode": result["both_mode"],
        "both_subtract_gain": result["both_subtract_gain"],
        "both_policy": result["both_policy"],
        "both_policy_reason": result["both_policy_reason"],
        "seed": result["seed"],
        "self_duration_ms": result["self_duration_ms"],
        "self_amplitude": result["self_amplitude"],
        "self_only": result["self_only"],
        "world_only": result["world_only"],
        "similarity_threshold": result["similarity_threshold"],
        "speech_override": result["speech_override"],
        "onset_latency_ms": result["onset_latency_ms"],
        "onset_method": result["onset_method"],
        "self_shift_samples": result["self_shift_samples"],
        "self_xcorr_shift_samples": result["self_xcorr_shift_samples"],
        "self_expected_shift_samples": result["self_expected_shift_samples"],
        "self_onset_error_ms": result["self_onset_error_ms"],
        "causal_tolerance_ms": result["causal_tolerance_ms"],
        "self_match_mode": result["self_match_mode"],
        "self_match_search_ms": result["self_match_search_ms"],
        "dry_global_self_score": result["dry_global_self_score"],
        "dry_local_self_score": result["dry_local_self_score"],
        "dry_local_shift_samples": result["dry_local_shift_samples"],
        "room_local_self_score": result["room_local_self_score"],
        "room_local_shift_samples": result["room_local_shift_samples"],
        "room_filter_taps": result["room_filter_taps"],
        "room_filter_ridge": result["room_filter_ridge"],
        "frame_verdicts": "frame_verdicts.jsonl",
        "metrics": result["metrics"],
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="M2 Speech Preservation Harness")
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--noise-device", type=int, default=3)
    parser.add_argument("--expected-text", default=None, help="Reference transcript for WER measurement")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save observed audio, mask, and summary")
    parser.add_argument("--both-gain", type=float, default=0.5, help="Soft gain applied to BOTH frames before ASR")
    parser.add_argument("--both-mode", choices=["gain", "subtract"], default="gain", help="How to handle BOTH frames: whole-frame gain or aligned self-template subtraction")
    parser.add_argument("--both-subtract-gain", type=float, default=0.5, help="Template subtraction strength for BOTH frames")
    parser.add_argument("--both-policy", choices=["manual", "family"], default="manual", help="Override BOTH processing from deterministic M2 policy")
    parser.add_argument("--seed", type=int, default=0, help="Stimulus seed")
    parser.add_argument("--self-duration-ms", type=int, default=350, help="Nominal self-stimulus duration in milliseconds")
    parser.add_argument("--self-amplitude", type=float, default=0.7, help="Self-stimulus playback amplitude")
    parser.add_argument("--self-only", action="store_true", help="Play only self stimulus; replace world audio with silence")
    parser.add_argument("--world-only", action="store_true", help="Play only world speech; replace self stimulus with silence")
    parser.add_argument("--similarity-threshold", type=float, default=0.45, help="Attributor acoustic-similarity threshold")
    parser.add_argument("--speech-override", choices=["vad", "true", "false"], default="vad", help="Override VAD speech flag for calibration")
    parser.add_argument("--onset-latency-ms", type=float, default=0.0, help="Playback/capture onset latency compensation in milliseconds")
    parser.add_argument("--onset-method", choices=["xcorr", "energy", "score"], default="xcorr", help="Observed onset estimator for causal attribution")
    parser.add_argument("--causal-tolerance-ms", type=float, default=25.0, help="Attributor causal timing tolerance in milliseconds")
    parser.add_argument("--self-match-mode", choices=["global_xcorr", "local_xcorr", "room_local_oracle"], default="global_xcorr", help="Self-match scorer; room_local_oracle is experimental same-capture fitting")
    parser.add_argument("--self-match-search-ms", type=float, default=250.0, help="Local search radius for self-match scorer")
    parser.add_argument("--room-filter-taps", type=int, default=64, help="FIR taps for room_local_oracle self-match mode")
    parser.add_argument("--room-filter-ridge", type=float, default=1e-4, help="Ridge regularization for room_local_oracle self-match mode")
    args = parser.parse_args()

    res = run_m2_trial(
        self_stim=args.self_stim,
        world_speech_file=args.world_speech,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device,
        seed=args.seed,
        expected_text=args.expected_text,
        both_gain=args.both_gain,
        both_mode=args.both_mode,
        both_subtract_gain=args.both_subtract_gain,
        both_policy=args.both_policy,
        self_duration_ms=args.self_duration_ms,
        self_amplitude=args.self_amplitude,
        self_only=args.self_only,
        world_only=args.world_only,
        similarity_threshold=args.similarity_threshold,
        speech_override=args.speech_override,
        onset_latency_ms=args.onset_latency_ms,
        onset_method=args.onset_method,
        causal_tolerance_ms=args.causal_tolerance_ms,
        self_match_mode=args.self_match_mode,
        self_match_search_ms=args.self_match_search_ms,
        room_filter_taps=args.room_filter_taps,
        room_filter_ridge=args.room_filter_ridge,
    )

    if args.save_dir:
        save_trial_artifacts(args.save_dir, res)
        print(f"Saved artifacts to: {args.save_dir}")

    print("\n--- M2 Final Verdict ---")
    print(f"Self Match Score: {res['self_score']:.4f}")
    print(f"Speech Ratio: {res['speech_ratio']:.1%}")
    print(f"Self/Both/World Frames: {res['self_frames']}/{res['both_frames']}/{res['world_frames']}")

    if res["metrics"]:
        if res["metrics"]["gated_wer"] <= res["metrics"]["raw_wer"]:
            print("\n✅ SUCCESS: Gated transcript preserved or improved reference accuracy.")
        else:
            print("\n⚠️ MIXED: Gated transcript was worse than raw against the reference text.")
    elif len(res["gated_text"]) > 5:
        print("\n✅ SUCCESS: World speech preserved after self-voice gating!")
    else:
        print("\n❌ FAILURE: World speech was lost or masked.")


if __name__ == "__main__":
    main()
