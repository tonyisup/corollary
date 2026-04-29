#!/usr/bin/env python3
"""M2 paired evaluation runner.

Runs the speech-preservation harness across the same calibrated seed set for
multiple BOTH-frame processing strategies, then writes an aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_TEXT = "this is a test of world speech conservation"


def safe_float_label(value: float) -> str:
    return str(value).replace(".", "p")


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_trial(args: argparse.Namespace, seed: int, strategy: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/m2_speech_preservation_harness.py",
        "--input-device",
        str(args.input_device),
        "--output-device",
        str(args.output_device),
        "--noise-device",
        str(args.noise_device),
        "--self-stim",
        args.self_stim,
        "--self-duration-ms",
        str(args.self_duration_ms),
        "--self-amplitude",
        str(args.self_amplitude),
        "--seed",
        str(seed),
        "--speech-override",
        "vad",
        "--similarity-threshold",
        str(args.similarity_threshold),
        "--onset-latency-ms",
        str(args.onset_latency_ms),
        "--onset-method",
        args.onset_method,
        "--causal-tolerance-ms",
        str(args.causal_tolerance_ms),
        "--expected-text",
        args.expected_text,
        "--save-dir",
        str(save_dir),
    ]

    if strategy["mode"] == "gain":
        cmd.extend(["--both-mode", "gain", "--both-gain", str(strategy["both_gain"])])
    elif strategy["mode"] == "subtract":
        cmd.extend([
            "--both-mode",
            "subtract",
            "--both-gain",
            "1.0",
            "--both-subtract-gain",
            str(strategy["both_subtract_gain"]),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (save_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (save_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Trial failed ({save_dir}) with code {proc.returncode}\n{proc.stderr}")

    summary = load_summary(save_dir / "summary.json")
    summary["strategy"] = strategy
    summary["seed"] = seed
    summary["save_dir"] = str(save_dir)
    return summary


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        strategy = result["strategy"]
        if strategy["mode"] == "gain":
            key = f"gain_{safe_float_label(strategy['both_gain'])}"
        else:
            key = f"subtract_{safe_float_label(strategy['both_subtract_gain'])}"
        by_strategy.setdefault(key, []).append(result)

    aggregate: dict[str, Any] = {"strategies": {}}
    for key, rows in by_strategy.items():
        deltas = [r["metrics"].get("wer_delta") for r in rows if r.get("metrics")]
        raw_wers = [r["metrics"].get("raw_wer") for r in rows if r.get("metrics")]
        gated_wers = [r["metrics"].get("gated_wer") for r in rows if r.get("metrics")]
        aggregate["strategies"][key] = {
            "trials": len(rows),
            "mean_raw_wer": sum(raw_wers) / len(raw_wers) if raw_wers else None,
            "mean_gated_wer": sum(gated_wers) / len(gated_wers) if gated_wers else None,
            "mean_wer_delta": sum(deltas) / len(deltas) if deltas else None,
            "improved": sum(1 for d in deltas if d is not None and d < 0),
            "neutral": sum(1 for d in deltas if d == 0),
            "regressed": sum(1 for d in deltas if d is not None and d > 0),
            "rows": [
                {
                    "seed": r["seed"],
                    "raw_wer": r["metrics"].get("raw_wer"),
                    "gated_wer": r["metrics"].get("gated_wer"),
                    "wer_delta": r["metrics"].get("wer_delta"),
                    "self_frames": r["self_frames"],
                    "both_frames": r["both_frames"],
                    "world_frames": r["world_frames"],
                    "speech_ratio": r["speech_ratio"],
                    "self_score": r["self_score"],
                    "save_dir": r["save_dir"],
                }
                for r in rows
            ],
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired M2 strategy evaluation")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--self-duration-ms", type=int, default=900)
    parser.add_argument("--self-amplitude", type=float, default=1.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.20)
    parser.add_argument("--onset-latency-ms", type=float, default=400.0)
    parser.add_argument("--onset-method", choices=["xcorr", "energy"], default="energy")
    parser.add_argument("--causal-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--expected-text", default=EXPECTED_TEXT)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    strategies = [
        {"mode": "gain", "both_gain": 1.0},
        {"mode": "gain", "both_gain": 0.5},
        {"mode": "subtract", "both_subtract_gain": 0.25},
    ]

    root = Path(args.output_root or f"data/m2_baseline/m2_paired_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for strategy in strategies:
        if strategy["mode"] == "gain":
            strategy_name = f"gain_{safe_float_label(strategy['both_gain'])}"
        else:
            strategy_name = f"subtract_{safe_float_label(strategy['both_subtract_gain'])}"
        for seed in seeds:
            save_dir = root / strategy_name / f"seed{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running {strategy_name} seed={seed} -> {save_dir}", flush=True)
            results.append(run_trial(args, seed, strategy, save_dir))

    aggregate = summarize(results)
    aggregate["output_root"] = str(root)
    aggregate["seeds"] = seeds
    aggregate["expected_text"] = args.expected_text
    aggregate["calibration"] = {
        "similarity_threshold": args.similarity_threshold,
        "onset_latency_ms": args.onset_latency_ms,
        "onset_method": args.onset_method,
        "causal_tolerance_ms": args.causal_tolerance_ms,
    }
    (root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
