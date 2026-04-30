#!/usr/bin/env python3
"""Analyze room-colored Broca self references from saved calibration artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from cochlea.capture import SR
from m2_speech_preservation_harness import normalized_xcorr

DEFAULT_TEXT = "I was speaking, then paused."


def dry_template_track(template: np.ndarray, total_samples: int, shift: int) -> np.ndarray:
    track = np.zeros(total_samples, dtype=np.float32)
    start = max(0, int(shift))
    src_start = 0 if shift >= 0 else -int(shift)
    n = min(len(template) - src_start, total_samples - start)
    if n > 0:
        track[start:start + n] = template[src_start:src_start + n]
    return track


def fit_room_filter(observed: np.ndarray, template: np.ndarray, shift: int, taps: int = 128, ridge: float = 1e-3) -> np.ndarray:
    """Fit a small FIR that maps dry Broca PCM to observed mic capture."""
    dry = dry_template_track(template, len(observed), shift)
    start = max(taps - 1, int(shift) + taps - 1)
    stop = min(len(observed), int(shift) + len(template))
    if stop <= start + taps * 2:
        return np.r_[1.0, np.zeros(taps - 1, dtype=np.float64)]

    idx = np.arange(start, stop)
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


def local_score(observed: np.ndarray, reference: np.ndarray, expected_shift: int, search_ms: float = 250.0) -> tuple[float, int]:
    """Score reference around expected shift instead of whole-record global xcorr."""
    if len(reference) <= 0 or len(observed) < len(reference):
        return 0.0, 0
    radius = int(search_ms * SR / 1000.0)
    start = max(0, expected_shift - radius)
    stop = min(len(observed) - len(reference), expected_shift + radius)
    if stop <= start:
        return 0.0, expected_shift
    best_score = -1.0
    best_shift = start
    step = max(len(reference) // 160, 1)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm < 1e-9:
        return 0.0, expected_shift
    for shift in range(start, stop + 1, step):
        window = observed[shift:shift + len(reference)]
        denom = float(np.linalg.norm(window) * ref_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, reference) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def analyze_artifact(path: Path, template: np.ndarray, taps: int) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    observed = np.load(path / "observed.npy")
    expected_shift = int(summary["self_expected_shift_samples"])

    dry_global_score, dry_global_shift = normalized_xcorr(observed, template)
    dry_local_score, dry_local_shift = local_score(observed, template, expected_shift)

    h = fit_room_filter(observed, template, dry_local_shift, taps=taps)
    room_ref = apply_room_filter(template, h)
    room_global_score, room_global_shift = normalized_xcorr(observed, room_ref)
    room_local_score, room_local_shift = local_score(observed, room_ref, expected_shift)

    return {
        "artifact": str(path),
        "label": path.name,
        "self_frames": summary.get("self_frames"),
        "both_frames": summary.get("both_frames"),
        "world_frames": summary.get("world_frames"),
        "summary_self_score": summary.get("self_score"),
        "expected_shift": expected_shift,
        "dry_global_score": dry_global_score,
        "dry_global_shift": dry_global_shift,
        "dry_local_score": dry_local_score,
        "dry_local_shift": dry_local_shift,
        "room_global_score": room_global_score,
        "room_global_shift": room_global_shift,
        "room_local_score": room_local_score,
        "room_local_shift": room_local_shift,
        "room_gain_l2": float(np.linalg.norm(h)),
        "room_first_tap": float(h[0]),
        "room_improvement": room_local_score - dry_local_score,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Analyze dry vs room-colored Broca self reference scores")
    parser.add_argument("artifact_dirs", nargs="+", help="Saved Broca self calibration artifact directories")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--taps", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Synthesizing reference text: {args.text!r}")
    template = await synthesize(args.text)
    rows = [analyze_artifact(Path(p), template, args.taps) for p in args.artifact_dirs]
    rows = sorted(rows, key=lambda r: r["room_improvement"], reverse=True)

    for r in rows:
        print(
            f"{r['label']}: dry_local={r['dry_local_score']:.3f} "
            f"room_local={r['room_local_score']:.3f} improvement={r['room_improvement']:.3f} "
            f"dry_shift={r['dry_local_shift']} room_shift={r['room_local_shift']}"
        )

    out = {"text": args.text, "taps": args.taps, "rows": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
