#!/usr/bin/env python3
"""Held-out validation for room-colored Broca self references."""

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
from m3_broca_room_reference_analysis import apply_room_filter, fit_room_filter, local_score

DEFAULT_TEXT = "I was speaking, then paused."


def load_artifact(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    p = Path(path)
    return json.loads((p / "summary.json").read_text(encoding="utf-8")), np.load(p / "observed.npy")


def validate_pair(train_path: Path, test_path: Path, template: np.ndarray, taps: int) -> dict[str, Any]:
    train_summary, train_observed = load_artifact(train_path)
    test_summary, test_observed = load_artifact(test_path)

    train_expected = int(train_summary["self_expected_shift_samples"])
    test_expected = int(test_summary["self_expected_shift_samples"])
    train_dry_score, train_dry_shift = local_score(train_observed, template, train_expected)
    h = fit_room_filter(train_observed, template, train_dry_shift, taps=taps)
    room_ref = apply_room_filter(template, h)

    test_dry_score, test_dry_shift = local_score(test_observed, template, test_expected)
    test_room_score, test_room_shift = local_score(test_observed, room_ref, test_expected)

    return {
        "train": str(train_path),
        "test": str(test_path),
        "train_label": train_path.name,
        "test_label": test_path.name,
        "train_dry_score": train_dry_score,
        "train_dry_shift": train_dry_shift,
        "test_dry_score": test_dry_score,
        "test_dry_shift": test_dry_shift,
        "test_room_score": test_room_score,
        "test_room_shift": test_room_shift,
        "holdout_improvement": test_room_score - test_dry_score,
        "room_gain_l2": float(np.linalg.norm(h)),
        "test_self_frames": test_summary.get("self_frames"),
        "test_both_frames": test_summary.get("both_frames"),
        "test_world_frames": test_summary.get("world_frames"),
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Validate room-colored Broca reference on held-out artifacts")
    parser.add_argument("artifact_dirs", nargs="+", help="Saved Broca self calibration artifact directories")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--taps", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Synthesizing reference text: {args.text!r}")
    template = await synthesize(args.text)
    paths = [Path(p) for p in args.artifact_dirs]

    rows: list[dict[str, Any]] = []
    for train in paths:
        for test in paths:
            if train == test:
                continue
            rows.append(validate_pair(train, test, template, args.taps))

    rows = sorted(rows, key=lambda r: r["holdout_improvement"], reverse=True)
    improved = sum(1 for r in rows if r["holdout_improvement"] > 0)
    mean_improvement = sum(r["holdout_improvement"] for r in rows) / len(rows) if rows else 0.0
    out = {
        "text": args.text,
        "taps": args.taps,
        "pairs": len(rows),
        "improved_pairs": improved,
        "mean_holdout_improvement": mean_improvement,
        "rows": rows,
    }

    print(f"pairs={len(rows)} improved={improved} mean_improvement={mean_improvement:.3f}")
    for r in rows[:10]:
        print(
            f"train={r['train_label']} test={r['test_label']} "
            f"dry={r['test_dry_score']:.3f} room={r['test_room_score']:.3f} "
            f"improvement={r['holdout_improvement']:.3f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
