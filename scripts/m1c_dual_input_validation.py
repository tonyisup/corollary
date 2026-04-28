#!/usr/bin/env python3
"""Dual-input M1C validation harness.

This is the next step after geometry calibration: capture both microphones at
once and fuse evidence instead of trying to classify mixed-source cases from a
single input channel.

Default calibrated geometry:
- input 0 / self-ish mic: Fifine, device 3
- input 1 / world-ish mic: BL-2021USB, device 1
- self output: Mac mini Speakers, device 5
- world output: External speakers/headphones, device 4
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
from thalamus.attributor import SourceVerdict
from thalamus.dual_input_fusion import ChannelEvidence, DualInputFusionConfig, fuse_dual_input_evidence

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata
from m1c_hardware_harness import append_jsonl_record, db, normalized_xcorr, play_array_blocking, resample_audio


def parse_pair(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated IDs")
    return int(parts[0]), int(parts[1])


def build_schedule(mode: str, self_pcm: np.ndarray, world_pcm: np.ndarray, lead_in_s: float, gap_s: float, overlap_ratio: float, tail_s: float) -> tuple[float | None, float | None, float]:
    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR
    if mode == "self_only":
        self_start_s = lead_in_s
        world_start_s = None
        total_s = self_start_s + self_duration + tail_s
    elif mode == "world_only":
        self_start_s = None
        world_start_s = lead_in_s
        total_s = world_start_s + world_duration + tail_s
    elif mode == "replay_control":
        self_start_s = None
        world_start_s = lead_in_s
        total_s = world_start_s + self_duration + tail_s
    elif mode == "sequential_world_first":
        world_start_s = lead_in_s
        self_start_s = world_start_s + world_duration + gap_s
        total_s = self_start_s + self_duration + tail_s
    elif mode == "sequential_self_first":
        self_start_s = lead_in_s
        world_start_s = self_start_s + self_duration + gap_s
        total_s = world_start_s + world_duration + tail_s
    elif mode == "partial_overlap_self_first":
        self_start_s = lead_in_s
        world_start_s = self_start_s + self_duration * (1.0 - overlap_ratio)
        total_s = max(self_start_s + self_duration, world_start_s + world_duration) + tail_s
    elif mode == "partial_overlap_world_first":
        world_start_s = lead_in_s
        self_start_s = world_start_s + world_duration * (1.0 - overlap_ratio)
        total_s = max(self_start_s + self_duration, world_start_s + world_duration) + tail_s
    elif mode == "full_overlap":
        self_start_s = lead_in_s
        world_start_s = lead_in_s
        total_s = max(self_duration, world_duration) + lead_in_s + tail_s
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return self_start_s, world_start_s, total_s


def pad_for_output(pcm: np.ndarray, start_s: float | None, total_s: float, output_sr: int) -> np.ndarray:
    padded = np.zeros(int(total_s * output_sr), dtype=np.float32)
    if start_s is None:
        return padded
    pcm_rs = resample_audio(pcm, SR, output_sr)
    start = int(start_s * output_sr)
    end = min(len(padded), start + len(pcm_rs))
    padded[start:end] = pcm_rs[: end - start]
    return padded


def capture_dual(input_devices: tuple[int, int], total_s: float, countdown_s: float) -> tuple[list[np.ndarray], list[int], list[sd.InputStream]]:
    input_srs = [int(sd.query_devices(dev)["default_samplerate"] or SR) for dev in input_devices]
    total_samples = [int(total_s * sr) for sr in input_srs]
    buffers = [np.zeros(n, dtype=np.float32) for n in total_samples]
    streams = [sd.InputStream(device=dev, channels=1, samplerate=sr, dtype="float32", blocksize=1024) for dev, sr in zip(input_devices, input_srs)]
    if countdown_s > 0:
        print(f"Starting in {countdown_s:.1f}s...")
        time.sleep(countdown_s)
    for stream in streams:
        stream.start()
    return buffers, input_srs, streams


def read_dual(buffers: list[np.ndarray], streams: list[sd.InputStream]) -> None:
    positions = [0, 0]
    while any(pos < len(buf) for pos, buf in zip(positions, buffers)):
        for idx, stream in enumerate(streams):
            remaining = len(buffers[idx]) - positions[idx]
            if remaining <= 0:
                continue
            data, _ = stream.read(min(1024, remaining))
            n = min(len(data), remaining)
            buffers[idx][positions[idx] : positions[idx] + n] = data[:n, 0]
            positions[idx] += n


def analyze_channel(observed: np.ndarray, self_pcm: np.ndarray, world_pcm: np.ndarray, self_start_s: float | None, world_start_s: float | None) -> dict[str, Any]:
    self_score, self_shift = normalized_xcorr(observed, self_pcm)
    world_score, world_shift = normalized_xcorr(observed, world_pcm)
    baseline_len = int(0.25 * SR)
    baseline = observed[:baseline_len]
    baseline_db = db(float(np.sqrt(np.mean(baseline ** 2))) if len(baseline) else 0.0)

    def window_db(start_s: float | None, length: int) -> float:
        if start_s is None:
            return -120.0
        start = max(0, int(start_s * SR))
        end = min(len(observed), start + length)
        window = observed[start:end]
        return db(float(np.sqrt(np.mean(window ** 2))) if len(window) else 0.0)

    expected_self_db = window_db(self_start_s, len(self_pcm))
    expected_world_db = window_db(world_start_s, len(world_pcm))
    return {
        "baseline_db": round(baseline_db, 2),
        "self_score": round(self_score, 6),
        "world_score": round(world_score, 6),
        "self_shift_samples": int(self_shift),
        "world_shift_samples": int(world_shift),
        "expected_self_rms_db": round(expected_self_db, 2),
        "expected_world_rms_db": round(expected_world_db, 2),
        "self_evidence_db": round(expected_self_db - baseline_db, 2),
        "world_evidence_db": round(expected_world_db - baseline_db, 2),
    }


def fuse_verdict(mode: str, channels: list[dict[str, Any]], self_score_threshold: float, world_evidence_threshold_db: float) -> tuple[str, dict[str, Any]]:
    result = fuse_dual_input_evidence(
        [
            ChannelEvidence(
                self_score=float(ch["self_score"]),
                world_score=float(ch["world_score"]),
                self_evidence_db=float(ch["self_evidence_db"]),
                world_evidence_db=float(ch["world_evidence_db"]),
            )
            for ch in channels
        ],
        mode=mode,
        config=DualInputFusionConfig(
            self_score_threshold=self_score_threshold,
            world_evidence_threshold_db=world_evidence_threshold_db,
        ),
    )
    return result.verdict.name, {
        "self_present": result.self_present,
        "world_present": result.world_present,
        "world_evidence_max_db": round(result.world_evidence_max_db, 2),
        "world_score_max": round(result.world_score_max, 6),
        "world_favored_score": round(result.world_favored_score, 6),
    }


def expected_verdict_ok(mode: str, verdict: str, fusion: dict[str, Any]) -> bool:
    if mode == "self_only":
        return verdict == SourceVerdict.SELF.name
    if mode in {"world_only", "replay_control"}:
        return verdict == SourceVerdict.WORLD.name
    if mode in {"partial_overlap_self_first", "partial_overlap_world_first", "full_overlap"}:
        return verdict == SourceVerdict.BOTH.name
    if mode == "sequential_self_first":
        # Sequential conditions are not single-frame verdicts; require at least
        # the expected first/primary source evidence to be present.
        return bool(fusion["self_present"])
    if mode == "sequential_world_first":
        return bool(fusion["world_present"])
    return False


def run_trial(args: argparse.Namespace, mode: str, seed: int, session_label: str) -> dict[str, Any]:
    self_meta = generate_stimulus_with_metadata(args.self_stim, args.duration_ms, args.amplitude, seed=seed)
    world_seed = seed if mode == "replay_control" else seed + 100
    world_stim = args.self_stim if mode == "replay_control" else args.world_stim
    world_meta = generate_stimulus_with_metadata(world_stim, args.duration_ms, args.amplitude, seed=world_seed)
    self_pcm = self_meta["pcm"].astype(np.float32)
    world_pcm = world_meta["pcm"].astype(np.float32)
    self_start_s, world_start_s, total_s = build_schedule(mode, self_pcm, world_pcm, args.lead_in_s, args.gap_s, args.overlap_ratio, args.tail_s)

    output_srs = [int(sd.query_devices(args.self_output_device)["default_samplerate"] or SR), int(sd.query_devices(args.world_output_device)["default_samplerate"] or SR)]
    self_out = pad_for_output(self_pcm, self_start_s, total_s, output_srs[0])
    world_out = pad_for_output(world_pcm, world_start_s, total_s, output_srs[1])

    print(f"\nMode: {mode} seed={seed} self_start={self_start_s} world_start={world_start_s}")
    buffers, input_srs, streams = capture_dual(args.input_devices, total_s, args.countdown_s)
    try:
        self_thread = threading.Thread(target=play_array_blocking, args=(self_out, args.self_output_device, output_srs[0]))
        world_thread = threading.Thread(target=play_array_blocking, args=(world_out, args.world_output_device, output_srs[1]))
        self_thread.start()
        world_thread.start()
        read_dual(buffers, streams)
        self_thread.join()
        world_thread.join()
    finally:
        for stream in streams:
            stream.stop()
            stream.close()

    observed = [resample_audio(buf, sr, SR) if sr != SR else buf for buf, sr in zip(buffers, input_srs)]
    channels = [analyze_channel(obs, self_pcm, world_pcm, self_start_s, world_start_s) for obs in observed]
    verdict, fusion = fuse_verdict(mode, channels, args.self_score_threshold, args.world_evidence_threshold_db)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_label": session_label,
        "setup_label": args.setup_label,
        "mode": mode,
        "case": mode,
        "seed": int(seed),
        "self_stim": args.self_stim,
        "world_stim": world_stim,
        "input_devices": list(args.input_devices),
        "self_output_device": args.self_output_device,
        "world_output_device": args.world_output_device,
        "self_start_s": self_start_s,
        "world_start_s": world_start_s,
        "channels": channels,
        "verdict": verdict,
        "pass": expected_verdict_ok(mode, verdict, fusion),
        **fusion,
    }
    print(json.dumps(record, sort_keys=True))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-input M1C validation / evidence fusion")
    parser.add_argument("--input-devices", type=parse_pair, default=(3, 1))
    parser.add_argument("--self-output-device", type=int, default=5)
    parser.add_argument("--world-output-device", type=int, default=4)
    parser.add_argument("--self-stim", default="babble_ba_var")
    parser.add_argument("--world-stim", default="babble_ma_var")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--countdown-s", type=float, default=0.15)
    parser.add_argument("--self-score-threshold", type=float, default=0.20)
    parser.add_argument("--world-evidence-threshold-db", type=float, default=3.0)
    parser.add_argument("--setup-label", default="m1c_dual_input")
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--jsonl-out", default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    session_label = args.session_label or f"m1c_dual_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jsonl_out = args.jsonl_out or os.path.join("data", "m1c_dual_input", f"{session_label}.jsonl")
    modes = [
        "self_only",
        "world_only",
        "replay_control",
        "sequential_world_first",
        "sequential_self_first",
        "partial_overlap_self_first",
        "partial_overlap_world_first",
        "full_overlap",
    ]

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            record = run_trial(args, mode, seed, session_label)
            records.append(record)
            append_jsonl_record(jsonl_out, record)
            time.sleep(0.5)

    print("\nDual-input M1C Summary")
    print(f"session_label={session_label}")
    print(f"jsonl_out={jsonl_out}")
    for mode in modes:
        mode_records = [r for r in records if r["mode"] == mode]
        pass_count = sum(1 for r in mode_records if r["pass"])
        verdicts: dict[str, int] = {}
        for r in mode_records:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        print(f"{mode:<30} n={len(mode_records):<3} pass={pass_count:<3} verdicts={verdicts}")


if __name__ == "__main__":
    main()
