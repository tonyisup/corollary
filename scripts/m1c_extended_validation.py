#!/usr/bin/env python3
"""Extended hardware-facing M1C validation run.

Runs a compact validation matrix for a fixed calibrated geometry:
- self_only: self output plays, motor command is active, expect SELF
- world_only: world output plays self-like/non-self audio externally, expect WORLD
- sequential / overlap mixed-source trials via m1c_hardware_harness

The goal is not physical geometry tuning; it is to produce calibration-ready
records and a concise pass/fail summary for SELF / WORLD / BOTH readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
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
from thalamus.attributor import SourceVerdict, ThalamicAttributor

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata
from m1c_hardware_harness import append_jsonl_record, db, normalized_xcorr, play_array_blocking, resample_audio, run_mixed_trial


def verdict_ok(case: str, verdict: str) -> bool:
    if case == "self_only":
        return verdict == SourceVerdict.SELF.name
    if case in {"world_only", "replay_control"}:
        return verdict == SourceVerdict.WORLD.name
    if case in {"partial_overlap_self_first", "partial_overlap_world_first", "full_overlap"}:
        return verdict == SourceVerdict.BOTH.name
    if case == "sequential_self_first":
        return verdict == SourceVerdict.SELF.name
    if case == "sequential_world_first":
        return verdict in {SourceVerdict.SELF.name, SourceVerdict.BOTH.name}
    return True


def capture_single_source(
    *,
    case: str,
    stim: str,
    template_stim: str,
    duration_ms: int,
    amplitude: float,
    input_device: int,
    playback_device: int,
    is_playing: bool,
    seed: int,
    lead_in_s: float,
    tail_s: float,
    countdown_s: float,
    setup_label: str,
    session_label: str,
    expected_offset_samples: int = 0,
    similarity_threshold: float = 0.45,
    causal_tolerance_ms: float = 25.0,
) -> dict[str, Any]:
    stim_meta = generate_stimulus_with_metadata(stim, duration_ms, amplitude, seed=seed)
    playback_pcm = stim_meta["pcm"].astype(np.float32)
    template_meta = generate_stimulus_with_metadata(template_stim, duration_ms, amplitude, seed=seed)
    template_pcm = template_meta["pcm"].astype(np.float32)

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(playback_device)["default_samplerate"] or SR)
    total_duration_s = lead_in_s + len(playback_pcm) / SR + tail_s
    total_samples = int(total_duration_s * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)

    padded = np.zeros(int(total_duration_s * output_sr), dtype=np.float32)
    start_idx = int(lead_in_s * output_sr)
    playback_rs = resample_audio(playback_pcm, SR, output_sr)
    end_idx = min(len(padded), start_idx + len(playback_rs))
    padded[start_idx:end_idx] = playback_rs[: end_idx - start_idx]

    print(f"\nCase: {case}")
    print(f"Stimulus: {stim} on dev {playback_device}; input dev {input_device}; motor_active={is_playing}")
    if countdown_s > 0:
        print(f"Starting in {countdown_s:.1f}s...")
        time.sleep(countdown_s)

    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()
    thread = threading.Thread(target=play_array_blocking, args=(padded, playback_device, output_sr))
    thread.start()

    pos = 0
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured[pos : pos + n] = data[:n, 0]
        pos += n

    thread.join()
    stream.stop()
    stream.close()

    observed = resample_audio(captured, input_sr, SR) if input_sr != SR else captured
    match_score, observed_shift = normalized_xcorr(observed, template_pcm)
    expected_shift = int(lead_in_s * SR) + int(expected_offset_samples) if is_playing else None
    expected_start = int(lead_in_s * SR)
    expected_end = min(len(observed), expected_start + len(template_pcm))
    expected_window = observed[expected_start:expected_end]
    rms = float(np.sqrt(np.mean(expected_window ** 2))) if len(expected_window) else 0.0
    rms_db = db(rms)

    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=similarity_threshold,
        causal_tolerance_ms=causal_tolerance_ms,
    )
    res = attributor.attribute(
        match_score=match_score,
        observed_onset_samples=observed_shift,
        expected_onset_samples=expected_shift,
        rms_db=rms_db,
        is_playing=is_playing,
    )

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_label": session_label,
        "setup_label": setup_label,
        "case": case,
        "mode": case,
        "stim": stim,
        "template_stim": template_stim,
        "duration_ms": int(duration_ms),
        "amplitude": float(amplitude),
        "seed": int(seed),
        "input_device": int(input_device),
        "playback_device": int(playback_device),
        "is_playing": bool(is_playing),
        "match_score": round(match_score, 6),
        "observed_shift_samples": int(observed_shift),
        "expected_shift_samples": int(expected_shift) if expected_shift is not None else None,
        "expected_offset_samples": int(expected_offset_samples),
        "onset_error_ms": round(res.onset_error_ms if res.onset_error_ms is not None else 0.0, 4),
        "similarity_threshold": round(float(similarity_threshold), 6),
        "causal_tolerance_ms": round(float(causal_tolerance_ms), 4),
        "rms_db": round(rms_db, 2),
        "verdict": res.verdict.name,
        "confidence": round(res.confidence, 4),
        "pass": verdict_ok(case, res.verdict.name),
    }
    print(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended M1C geometry-locked hardware validation")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--self-output-device", type=int, default=5)
    parser.add_argument("--world-output-device", type=int, default=4)
    parser.add_argument("--self-stim", default="babble_ba_var")
    parser.add_argument("--world-stim", default="babble_ma_var")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--countdown-s", type=float, default=1.0)
    parser.add_argument("--setup-label", default="m1c_geometry_locked")
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--jsonl-out", default=None)
    parser.add_argument("--calibration-trials", type=int, default=3, help="Self-only trials used to estimate onset offset and threshold")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable self-onset/similarity calibration")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    session_label = args.session_label or f"m1c_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jsonl_out = args.jsonl_out or os.path.join("data", "m1c_extended", f"{session_label}.jsonl")

    modes = [
        "sequential_world_first",
        "sequential_self_first",
        "partial_overlap_self_first",
        "partial_overlap_world_first",
        "full_overlap",
    ]

    records: list[dict[str, Any]] = []
    expected_base_shift = int(args.lead_in_s * SR)
    expected_offset_samples = 0
    similarity_threshold = 0.45
    causal_tolerance_ms = 25.0

    if not args.no_calibrate and args.calibration_trials > 0:
        print("\nCalibrating self onset/similarity baseline...")
        calibration_records = []
        for i in range(args.calibration_trials):
            record = capture_single_source(
                case="self_calibration",
                stim=args.self_stim,
                template_stim=args.self_stim,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                playback_device=args.self_output_device,
                is_playing=True,
                seed=seeds[i % len(seeds)] if seeds else i,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                countdown_s=args.countdown_s,
                setup_label=args.setup_label,
                session_label=session_label,
            )
            calibration_records.append(record)
            records.append(record)
            append_jsonl_record(jsonl_out, record)
            time.sleep(0.5)

        observed_offsets = np.array(
            [int(r["observed_shift_samples"]) - expected_base_shift for r in calibration_records],
            dtype=np.int64,
        )
        self_scores = np.array([float(r["match_score"]) for r in calibration_records], dtype=np.float32)
        expected_offset_samples = int(np.median(observed_offsets))
        calibrated_errors_ms = np.abs(observed_offsets - expected_offset_samples) * 1000.0 / SR
        causal_tolerance_ms = float(max(25.0, np.percentile(calibrated_errors_ms, 95) + 15.0))
        similarity_threshold = float(np.clip(np.percentile(self_scores, 20) * 0.75, 0.08, 0.45))
        calibration_summary = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "session_label": session_label,
            "setup_label": args.setup_label,
            "case": "calibration_summary",
            "expected_offset_samples": expected_offset_samples,
            "expected_offset_ms": round(expected_offset_samples * 1000.0 / SR, 4),
            "similarity_threshold": round(similarity_threshold, 6),
            "causal_tolerance_ms": round(causal_tolerance_ms, 4),
            "self_match_scores": [round(float(x), 6) for x in self_scores],
            "observed_offsets_samples": [int(x) for x in observed_offsets],
        }
        print(calibration_summary)
        records.append(calibration_summary)
        append_jsonl_record(jsonl_out, calibration_summary)

    for seed in seeds:
        for record in [
            capture_single_source(
                case="self_only",
                stim=args.self_stim,
                template_stim=args.self_stim,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                playback_device=args.self_output_device,
                is_playing=True,
                seed=seed,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                countdown_s=args.countdown_s,
                setup_label=args.setup_label,
                session_label=session_label,
                expected_offset_samples=expected_offset_samples,
                similarity_threshold=similarity_threshold,
                causal_tolerance_ms=causal_tolerance_ms,
            ),
            capture_single_source(
                case="world_only",
                stim=args.world_stim,
                template_stim=args.self_stim,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                playback_device=args.world_output_device,
                is_playing=False,
                seed=seed + 100,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                countdown_s=args.countdown_s,
                setup_label=args.setup_label,
                session_label=session_label,
                expected_offset_samples=expected_offset_samples,
                similarity_threshold=similarity_threshold,
                causal_tolerance_ms=causal_tolerance_ms,
            ),
            capture_single_source(
                case="replay_control",
                stim=args.self_stim,
                template_stim=args.self_stim,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                playback_device=args.world_output_device,
                is_playing=False,
                seed=seed,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                countdown_s=args.countdown_s,
                setup_label=args.setup_label,
                session_label=session_label,
                expected_offset_samples=expected_offset_samples,
                similarity_threshold=similarity_threshold,
                causal_tolerance_ms=causal_tolerance_ms,
            ),
        ]:
            records.append(record)
            append_jsonl_record(jsonl_out, record)
            time.sleep(0.5)

        for mode in modes:
            record = run_mixed_trial(
                mode=mode,
                self_stim=args.self_stim,
                world_stim=args.world_stim,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                output_device=args.self_output_device,
                noise_device=args.world_output_device,
                overlap_ratio=args.overlap_ratio,
                gap_s=args.gap_s,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                seed=seed,
                setup_label=args.setup_label,
                expected_label=None,
                calibration_split="validation",
                session_label=session_label,
                countdown_s=args.countdown_s,
                self_expected_offset_samples=expected_offset_samples,
                similarity_threshold=similarity_threshold,
                causal_tolerance_ms=causal_tolerance_ms,
            )
            record["case"] = mode
            record["pass"] = verdict_ok(mode, record["verdict"])
            if mode == "sequential_world_first":
                record["pass"] = bool(record.get("world_first_detected")) and record["pass"]
            records.append(record)
            append_jsonl_record(jsonl_out, record)
            print(record)
            time.sleep(0.5)

    print("\nExtended M1C Summary")
    print(f"session_label={session_label}")
    print(f"jsonl_out={jsonl_out}")
    for case in sorted({r.get("case", r.get("mode", "unknown")) for r in records}):
        case_records = [r for r in records if r.get("case", r.get("mode")) == case]
        verdicts: dict[str, int] = {}
        pass_count = 0
        for r in case_records:
            verdict = str(r.get("verdict", "n/a"))
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if r.get("pass") is True:
                pass_count += 1
        print(f"{case:<30} n={len(case_records):<3} pass={pass_count:<3} verdicts={verdicts}")


if __name__ == "__main__":
    main()
