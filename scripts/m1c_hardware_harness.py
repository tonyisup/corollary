#!/usr/bin/env python3
"""Hardware-facing M1C mixed-source attribution harness.

Executes hardware-in-the-loop experiments for:
1. Sequential world-before-self
2. Sequential self-before-world
3. Partial overlap
4. Full overlap

Supports dual-output: self on one device, world on another.
"""

from __future__ import annotations

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
    print("Install with: pip install sounddevice")
    sys.exit(1)

# Add corollary packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from thalamus.attributor import ThalamicAttributor, SourceVerdict

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata, WORLD_VARIANTS

PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB = 2.62


def append_jsonl_record(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def infer_expected_label(mode: str) -> str:
    if mode == "sequential_world_first":
        return "world_before_self"
    if mode == "sequential_self_first":
        return "self_before_world"
    if mode in {"partial_overlap_self_first", "partial_overlap_world_first", "full_overlap"}:
        return "overlap"
    return "unknown"


def provisional_order_prediction(self_evidence_db: float) -> str:
    return (
        "world_before_self"
        if self_evidence_db <= PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB
        else "self_before_world"
    )


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        factor = max(int(orig_sr / target_sr), 1)
        return audio[::factor].astype(np.float32)

def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))

def play_array_blocking(data: np.ndarray, device: int, samplerate: float) -> None:
    """Play mono float32 in [-1, 1] on *device* (safe concurrent with other OutputStreams)."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(
        device=device,
        channels=data.shape[1],
        samplerate=samplerate,
        dtype="float32",
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

def local_rms_db(audio: np.ndarray, start: int, length: int) -> float:
    if len(audio) == 0 or length <= 0:
        return -120.0
    start = max(0, min(start, len(audio)))
    end = max(start, min(start + length, len(audio)))
    window = audio[start:end]
    if len(window) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(window ** 2)))
    return db(rms)


def run_mixed_trial(
    mode: str,
    self_stim: str,
    world_stim: str,
    duration_ms: int,
    amplitude: float,
    input_device: int | None,
    output_device: int | None,
    noise_device: int | None,
    overlap_ratio: float = 0.5,
    gap_s: float = 0.5,
    lead_in_s: float = 0.5,
    tail_s: float = 1.0,
    seed: int = 0,
    setup_label: str | None = None,
    expected_label: str | None = None,
    calibration_split: str = "train",
    session_label: str | None = None,
    countdown_s: float = 3.0,
    self_expected_offset_samples: int = 0,
    similarity_threshold: float = 0.45,
    causal_tolerance_ms: float = 25.0,
) -> dict[str, Any]:
    self_meta = generate_stimulus_with_metadata(self_stim, duration_ms, amplitude, seed=seed)
    self_pcm = self_meta["pcm"]
    world_meta = generate_stimulus_with_metadata(world_stim, duration_ms, amplitude, seed=seed + 100)
    world_pcm = world_meta["pcm"]

    # Calculate schedule
    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR

    if mode == "sequential_world_first":
        world_start_s = lead_in_s
        self_start_s = world_start_s + world_duration + gap_s
        total_duration = self_start_s + self_duration + tail_s
    elif mode == "sequential_self_first":
        self_start_s = lead_in_s
        world_start_s = self_start_s + self_duration + gap_s
        total_duration = world_start_s + world_duration + tail_s
    elif mode == "partial_overlap_self_first":
        self_start_s = lead_in_s
        world_start_s = self_start_s + self_duration * (1.0 - overlap_ratio)
        total_duration = max(self_start_s + self_duration, world_start_s + world_duration) + tail_s
    elif mode == "partial_overlap_world_first":
        world_start_s = lead_in_s
        self_start_s = world_start_s + world_duration * (1.0 - overlap_ratio)
        total_duration = max(self_start_s + self_duration, world_start_s + world_duration) + tail_s
    elif mode == "full_overlap":
        self_start_s = lead_in_s
        world_start_s = lead_in_s
        total_duration = max(self_duration, world_duration) + lead_in_s + tail_s
    else:
        raise ValueError(f"Unknown mode: {mode}")

    input_sr = SR
    if input_device is not None:
        input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    
    output_sr = SR
    if output_device is not None:
        output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    
    noise_sr = SR
    if noise_device is not None:
        noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    print(f"\nMode: {mode}")
    print(f"Self: {self_stim} (dev {output_device}) at {self_start_s:.3f}s")
    print(f"World: {world_stim} (dev {noise_device}) at {world_start_s:.3f}s")
    print(f"Total Duration: {total_duration:.3f}s")

    total_samples = int(total_duration * input_sr)
    captured_audio = np.zeros(total_samples, dtype=np.float32)

    # We need to play at specific times. 
    # Since play_array_blocking is blocking, we use timers or offsets.
    # For now, let's pad the PCM arrays with silence to match the full window and play them at once.
    
    def get_padded_pcm(pcm: np.ndarray, start_s: float, total_s: float, target_sr: float) -> np.ndarray:
        padded = np.zeros(int(total_s * target_sr), dtype=np.float32)
        start_idx = int(start_s * target_sr)
        # Resample pcm to target_sr if needed
        resampled_pcm = resample_audio(pcm, SR, int(target_sr))
        end_idx = min(len(padded), start_idx + len(resampled_pcm))
        padded[start_idx:end_idx] = resampled_pcm[:end_idx-start_idx]
        return padded

    self_padded = get_padded_pcm(self_pcm, self_start_s, total_duration, output_sr)
    world_padded = get_padded_pcm(world_pcm, world_start_s, total_duration, noise_sr)

    if countdown_s > 0:
        print(f"Starting in {countdown_s:.1f}s...")
        time.sleep(countdown_s)

    # Start capture
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()
    
    t_start = time.perf_counter()
    
    # Play threads
    self_thread = threading.Thread(target=play_array_blocking, args=(self_padded, output_device, output_sr))
    world_thread = threading.Thread(target=play_array_blocking, args=(world_padded, noise_device, noise_sr))
    
    self_thread.start()
    world_thread.start()
    
    pos = 0
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured_audio[pos:pos+n] = data[:n, 0]
        pos += n
        
    self_thread.join()
    world_thread.join()
    stream.stop()
    stream.close()
    
    if input_sr != SR:
        observed = resample_audio(captured_audio, input_sr, SR)
    else:
        observed = captured_audio

    # Analysis
    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=similarity_threshold,
        causal_tolerance_ms=causal_tolerance_ms,
    )

    self_score, self_shift = normalized_xcorr(observed, self_pcm)
    self_expected_shift = int(self_start_s * SR) + int(self_expected_offset_samples)

    world_score, world_shift = normalized_xcorr(observed, world_pcm)
    world_expected_shift = int(world_start_s * SR)

    whole_rms = float(np.sqrt(np.mean(observed**2))) if len(observed) else 0.0
    whole_rms_db = db(whole_rms)

    baseline_len = int(min(0.25 * SR, max(world_expected_shift, 1)))
    baseline_rms_db = local_rms_db(observed, 0, baseline_len)
    self_rms_db = local_rms_db(observed, self_shift, len(self_pcm))
    world_rms_db = local_rms_db(observed, world_shift, len(world_pcm))
    expected_self_rms_db = local_rms_db(observed, self_expected_shift, len(self_pcm))
    expected_world_rms_db = local_rms_db(observed, world_expected_shift, len(world_pcm))

    if mode == "sequential_world_first":
        world_evidence_db = expected_world_rms_db - baseline_rms_db
        world_first_detected = (world_score >= 0.15) and (world_evidence_db >= 6.0)
    else:
        world_evidence_db = world_rms_db - baseline_rms_db
        world_first_detected = False

    verdict_res = attributor.attribute(
        match_score=self_score,
        observed_onset_samples=self_shift,
        expected_onset_samples=self_expected_shift,
        rms_db=expected_self_rms_db,
        is_playing=expected_self_rms_db >= attributor.energy_threshold_db,
    )

    self_evidence_db = expected_self_rms_db - baseline_rms_db
    overlap_mode = mode in {"partial_overlap_self_first", "partial_overlap_world_first", "full_overlap"}
    mixed_source_detected = overlap_mode and self_evidence_db >= 6.0 and world_evidence_db >= 3.0
    final_verdict = SourceVerdict.BOTH.name if mixed_source_detected else verdict_res.verdict.name
    final_confidence = 0.75 if mixed_source_detected else verdict_res.confidence
    order_prediction = provisional_order_prediction(self_evidence_db)
    result = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_label": session_label,
        "calibration_split": calibration_split,
        "expected_label": expected_label or infer_expected_label(mode),
        "mode": mode,
        "self_stim": self_stim,
        "world_stim": world_stim,
        "duration_ms": int(duration_ms),
        "amplitude": float(amplitude),
        "seed": int(seed),
        "sample_rate": int(SR),
        "input_device": input_device,
        "output_device": output_device,
        "noise_device": noise_device,
        "setup_label": setup_label,
        "gap_s": float(gap_s),
        "lead_in_s": float(lead_in_s),
        "tail_s": float(tail_s),
        "overlap_ratio": float(overlap_ratio),
        "self_start_s": round(self_start_s, 6),
        "world_start_s": round(world_start_s, 6),
        "self_score": round(self_score, 6),
        "world_score": round(world_score, 6),
        "self_error_ms": round(verdict_res.onset_error_ms if verdict_res.onset_error_ms is not None else 0.0, 4),
        "verdict": final_verdict,
        "raw_attributor_verdict": verdict_res.verdict.name,
        "confidence": round(final_confidence, 4),
        "rms_db": round(whole_rms_db, 2),
        "baseline_rms_db": round(baseline_rms_db, 2),
        "self_rms_db": round(self_rms_db, 2),
        "world_rms_db": round(world_rms_db, 2),
        "expected_self_rms_db": round(expected_self_rms_db, 2),
        "expected_world_rms_db": round(expected_world_rms_db, 2),
        "self_evidence_db": round(self_evidence_db, 2),
        "world_evidence_db": round(world_evidence_db, 2),
        "provisional_order_threshold_db": PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB,
        "provisional_order_prediction": order_prediction,
        "world_first_detected": bool(world_first_detected),
        "mixed_source_detected": bool(mixed_source_detected),
        "self_shift_samples": int(self_shift),
        "world_shift_samples": int(world_shift),
        "self_expected_shift_samples": int(self_expected_shift),
        "world_expected_shift_samples": int(world_expected_shift),
        "self_expected_offset_samples": int(self_expected_offset_samples),
        "similarity_threshold": round(float(similarity_threshold), 6),
        "causal_tolerance_ms": round(float(causal_tolerance_ms), 4),
        "self_shift_error_samples": int(self_shift - self_expected_shift),
        "world_shift_error_samples": int(world_shift - world_expected_shift),
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="M1C Hardware Mixed-Source Harness")
    parser.add_argument("--mode", choices=["sequential_world_first", "sequential_self_first", "partial_overlap_self_first", "partial_overlap_world_first", "full_overlap", "all"], default="all")
    parser.add_argument("--self-stim", default="babble_ba_var")
    parser.add_argument("--world-stim", default="chirp")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--countdown-s", type=float, default=3.0)
    parser.add_argument("--setup-label", default="m1c_initial")
    parser.add_argument("--expected-label", default=None, help="Optional explicit ground-truth label for calibration rows")
    parser.add_argument("--calibration-split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--session-label", default=None, help="Optional label to group related runs into one calibration session")
    parser.add_argument("--jsonl-out", default=None, help="Optional JSONL path for calibration-ready per-run records")
    parser.add_argument("--self-expected-offset-samples", type=int, default=0, help="Calibrated self onset offset added to scheduled self start")
    parser.add_argument("--similarity-threshold", type=float, default=0.45, help="ThalamicAttributor similarity threshold")
    parser.add_argument("--causal-tolerance-ms", type=float, default=25.0, help="ThalamicAttributor causal timing tolerance")
    args = parser.parse_args()

    modes = [args.mode] if args.mode != "all" else [
        "sequential_world_first",
        "sequential_self_first",
        "partial_overlap_self_first",
        "partial_overlap_world_first",
        "full_overlap"
    ]

    results = []
    for mode in modes:
        res = run_mixed_trial(
            mode=mode,
            self_stim=args.self_stim,
            world_stim=args.world_stim,
            duration_ms=args.duration_ms,
            amplitude=args.amplitude,
            input_device=args.input_device,
            output_device=args.output_device,
            noise_device=args.noise_device,
            overlap_ratio=args.overlap_ratio,
            gap_s=args.gap_s,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
            seed=args.seed,
            setup_label=args.setup_label,
            expected_label=args.expected_label,
            calibration_split=args.calibration_split,
            session_label=args.session_label,
            countdown_s=args.countdown_s,
            self_expected_offset_samples=args.self_expected_offset_samples,
            similarity_threshold=args.similarity_threshold,
            causal_tolerance_ms=args.causal_tolerance_ms,
        )
        results.append(res)
        print(res)
        if args.jsonl_out:
            append_jsonl_record(args.jsonl_out, res)
        time.sleep(1)

    print("\nSummary Results:")
    print(f"{'Mode':<30} {'SelfEv':<8} {'WorldEv':<8} {'PredictedOrder':<20} {'Verdict':<10}")
    for r in results:
        print(f"{r['mode']:<30} {r['self_evidence_db']:<8.2f} {r['world_evidence_db']:<8.2f} {r['provisional_order_prediction']:<20} {r['verdict']:<10}")

if __name__ == "__main__":
    main()
