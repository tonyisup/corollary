#!/usr/bin/env python3
"""Offline M2 strategy evaluator.

Given saved M2 trial artifacts, re-run attribution and multiple BOTH-frame
processing strategies against the exact same captured `observed.npy`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from m1_tone_attribution import generate_stimulus_with_metadata
from m2_speech_preservation_harness import (
    build_gating_gain,
    db,
    energy_onset,
    normalized_xcorr,
    subtract_self_template,
    word_error_rate,
)
from stg.asr import transcribe
from thalamus.attributor import SourceVerdict, ThalamicAttributor


EXPECTED_TEXT = "this is a test of world speech conservation"


def safe_float_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_verdicts(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    self_start_s: float,
    similarity_threshold: float,
    onset_latency_ms: float,
    onset_method: str,
    causal_tolerance_ms: float,
    speech_override: str,
) -> dict[str, Any]:
    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=similarity_threshold,
        causal_tolerance_ms=causal_tolerance_ms,
    )
    vad = SmoothedVAD()

    self_score, xcorr_shift = normalized_xcorr(observed, self_pcm)
    self_expected_shift = int((self_start_s + onset_latency_ms / 1000.0) * SR)
    if onset_method == "energy":
        self_shift = energy_onset(observed, self_expected_shift)
    else:
        self_shift = xcorr_shift

    verdicts: list[SourceVerdict] = []
    speech_frames = 0
    num_frames = len(observed) // VAD_FRAME_SAMPLES

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
        is_playing = start >= self_expected_shift and end <= self_expected_shift + len(self_pcm)
        result = attributor.attribute(
            match_score=self_score if is_playing else 0.0,
            observed_onset_samples=self_shift,
            expected_onset_samples=self_expected_shift if is_playing else None,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
        )
        verdicts.append(result.verdict)

    return {
        "verdicts": verdicts,
        "speech_ratio": speech_frames / num_frames if num_frames else 0.0,
        "self_score": self_score,
        "self_shift_samples": int(self_shift),
        "self_xcorr_shift_samples": int(xcorr_shift),
        "self_expected_shift_samples": int(self_expected_shift),
        "self_onset_error_ms": float((self_shift - self_expected_shift) * 1000.0 / SR),
        "self_frames": sum(1 for v in verdicts if v == SourceVerdict.SELF),
        "both_frames": sum(1 for v in verdicts if v == SourceVerdict.BOTH),
        "world_frames": sum(1 for v in verdicts if v == SourceVerdict.WORLD),
    }


def both_mask_from_verdicts(verdicts: list[SourceVerdict], total_samples: int) -> np.ndarray:
    mask = np.zeros(total_samples, dtype=bool)
    for i, verdict in enumerate(verdicts):
        if verdict == SourceVerdict.BOTH:
            start = i * VAD_FRAME_SAMPLES
            end = min(start + VAD_FRAME_SAMPLES, total_samples)
            mask[start:end] = True
    return mask


def dry_template_track(self_pcm: np.ndarray, total_samples: int, self_shift: int) -> np.ndarray:
    track = np.zeros(total_samples, dtype=np.float32)
    start = max(0, int(self_shift))
    src_start = 0 if self_shift >= 0 else -int(self_shift)
    n = min(len(self_pcm) - src_start, total_samples - start)
    if n > 0:
        track[start:start + n] = self_pcm[src_start:src_start + n]
    return track


def estimate_room_track(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    verdicts: list[SourceVerdict],
    self_shift: int,
    taps: int = 64,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Estimate a small FIR room transform for the self template in BOTH windows."""
    dry = dry_template_track(self_pcm, len(observed), self_shift)
    both_mask = both_mask_from_verdicts(verdicts, len(observed))
    idx = np.flatnonzero(both_mask)
    idx = idx[idx >= taps - 1]
    if len(idx) < taps * 2:
        return dry

    # Keep solve bounded and deterministic.
    if len(idx) > 4000:
        idx = idx[np.linspace(0, len(idx) - 1, 4000).astype(int)]

    x_cols = [dry[idx - k] for k in range(taps)]
    x = np.stack(x_cols, axis=1).astype(np.float64)
    y = observed[idx].astype(np.float64)
    xtx = x.T @ x + ridge * np.eye(taps)
    xty = x.T @ y
    try:
        h = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return dry
    h = np.clip(h, -2.0, 2.0)
    return np.convolve(dry, h, mode="full")[: len(observed)].astype(np.float32)


def subtract_track(
    observed: np.ndarray,
    self_track: np.ndarray,
    verdicts: list[SourceVerdict],
    subtract_gain: float,
) -> np.ndarray:
    out = observed.astype(np.float32).copy()
    mask = both_mask_from_verdicts(verdicts, len(observed))
    if not np.any(mask):
        return out
    t = self_track[mask]
    y = out[mask]
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return out
    scale = float(np.dot(y, t) / denom)
    scale = float(np.clip(scale, 0.0, 2.0)) * float(np.clip(subtract_gain, 0.0, 1.0))
    out[mask] = y - scale * t
    return out


def process_strategy(
    observed: np.ndarray,
    self_pcm: np.ndarray,
    verdicts: list[SourceVerdict],
    self_shift: int,
    strategy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if strategy["mode"] == "gain":
        gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), strategy["both_gain"])
        return observed, gain
    if strategy["mode"] == "subtract":
        gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), 1.0)
        processed = subtract_self_template(
            observed,
            self_pcm,
            verdicts,
            VAD_FRAME_SAMPLES,
            self_shift,
            strategy["both_subtract_gain"],
        )
        return processed, gain
    if strategy["mode"] == "room_subtract":
        gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), 1.0)
        room_track = estimate_room_track(
            observed,
            self_pcm,
            verdicts,
            self_shift,
            taps=strategy.get("taps", 64),
        )
        processed = subtract_track(observed, room_track, verdicts, strategy["both_subtract_gain"])
        return processed, gain
    raise ValueError(strategy)


def strategy_name(strategy: dict[str, Any]) -> str:
    if strategy["mode"] == "gain":
        return f"gain_{safe_float_label(strategy['both_gain'])}"
    if strategy["mode"] == "room_subtract":
        return f"room_subtract_{safe_float_label(strategy['both_subtract_gain'])}_t{strategy.get('taps', 64)}"
    return f"subtract_{safe_float_label(strategy['both_subtract_gain'])}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline M2 strategy comparison")
    parser.add_argument("trial_dirs", nargs="+", help="Saved M2 trial directories containing observed.npy")
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--self-duration-ms", type=int, default=900)
    parser.add_argument("--self-amplitude", type=float, default=1.0)
    parser.add_argument("--world-duration-s", type=float, default=3.31)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.20)
    parser.add_argument("--onset-latency-ms", type=float, default=400.0)
    parser.add_argument("--onset-method", choices=["xcorr", "energy"], default="energy")
    parser.add_argument("--causal-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--speech-override", choices=["vad", "true", "false"], default="vad")
    parser.add_argument("--expected-text", default=EXPECTED_TEXT)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    strategies = [
        {"mode": "gain", "both_gain": 1.0},
        {"mode": "gain", "both_gain": 0.5},
        {"mode": "subtract", "both_subtract_gain": 0.25},
        {"mode": "room_subtract", "both_subtract_gain": 0.10, "taps": 64},
        {"mode": "room_subtract", "both_subtract_gain": 0.25, "taps": 64},
    ]

    output_dir = Path(args.output_dir or "data/m2_baseline/m2_offline_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for trial_dir_raw in args.trial_dirs:
        trial_dir = Path(trial_dir_raw)
        observed = np.load(trial_dir / "observed.npy").astype(np.float32)
        summary_path = trial_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        seed = int(summary.get("seed", 0))
        self_pcm = generate_stimulus_with_metadata(
            args.self_stim,
            args.self_duration_ms,
            args.self_amplitude,
            seed=seed,
        )["pcm"]
        self_start_s = args.lead_in_s + (args.world_duration_s / 2.0) - ((len(self_pcm) / SR) / 2.0)
        attribution = build_verdicts(
            observed=observed,
            self_pcm=self_pcm,
            self_start_s=self_start_s,
            similarity_threshold=args.similarity_threshold,
            onset_latency_ms=args.onset_latency_ms,
            onset_method=args.onset_method,
            causal_tolerance_ms=args.causal_tolerance_ms,
            speech_override=args.speech_override,
        )

        raw_res = transcribe(observed, model_size="base")
        raw_wer = word_error_rate(args.expected_text, raw_res.text)

        for strategy in strategies:
            name = strategy_name(strategy)
            processed, mask = process_strategy(
                observed,
                self_pcm,
                attribution["verdicts"],
                attribution["self_shift_samples"],
                strategy,
            )
            gated_res = transcribe(processed, mask=mask, model_size="base")
            gated_wer = word_error_rate(args.expected_text, gated_res.text)
            row = {
                "trial_dir": str(trial_dir),
                "seed": seed,
                "strategy": name,
                "raw_text": raw_res.text,
                "gated_text": gated_res.text,
                "raw_wer": raw_wer,
                "gated_wer": gated_wer,
                "wer_delta": gated_wer - raw_wer,
                "self_score": attribution["self_score"],
                "speech_ratio": attribution["speech_ratio"],
                "self_frames": attribution["self_frames"],
                "both_frames": attribution["both_frames"],
                "world_frames": attribution["world_frames"],
                "self_onset_error_ms": attribution["self_onset_error_ms"],
            }
            rows.append(row)
            print(f"{trial_dir.name} {name}: {raw_wer:.3f} -> {gated_wer:.3f} ({row['wer_delta']:+.3f})")

    aggregate: dict[str, Any] = {"rows": rows, "strategies": {}}
    for name in sorted({r["strategy"] for r in rows}):
        strategy_rows = [r for r in rows if r["strategy"] == name]
        deltas = [r["wer_delta"] for r in strategy_rows]
        aggregate["strategies"][name] = {
            "trials": len(strategy_rows),
            "mean_wer_delta": sum(deltas) / len(deltas),
            "improved": sum(1 for d in deltas if d < 0),
            "neutral": sum(1 for d in deltas if d == 0),
            "regressed": sum(1 for d in deltas if d > 0),
        }

    out = output_dir / "offline_summary.json"
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate["strategies"], indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
