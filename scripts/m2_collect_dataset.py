#!/usr/bin/env python3
"""Collect M2 captures and build a policy dataset.

Capture-once/process-offline workflow:
1. capture raw mixed trials with preserve-BOTH (`gain_1p0`)
2. run offline strategy evaluation against identical captures
3. export canonical JSONL rows for policy learning
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_TEXT = "this is a test of world speech conservation"


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def export_jsonl(offline_summary: Path, output_jsonl: Path) -> None:
    data = json.loads(offline_summary.read_text(encoding="utf-8"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in data.get("rows", []):
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect M2 policy dataset")
    parser.add_argument("--seeds", default="6-9", help="Comma list/ranges, e.g. 0,1,2 or 6-9")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--self-duration-ms", type=int, default=900)
    parser.add_argument("--self-amplitude", type=float, default=1.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.20)
    parser.add_argument("--onset-latency-ms", type=float, default=400.0)
    parser.add_argument("--onset-method", choices=["xcorr", "energy"], default="energy")
    parser.add_argument("--causal-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--expected-text", default=EXPECTED_TEXT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-capture", action="store_true", help="Only run offline export for existing capture dirs under output root")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    root = Path(args.output_root or f"data/m2_policy/collection_{time.strftime('%Y%m%d_%H%M%S')}")
    capture_root = root / "captures" / "gain_1p0"
    offline_root = root / "offline_eval"
    dataset_jsonl = root / "m2_policy_dataset.jsonl"
    root.mkdir(parents=True, exist_ok=True)

    capture_dirs: list[Path] = []
    for seed in seeds:
        save_dir = capture_root / f"seed{seed}"
        capture_dirs.append(save_dir)
        if args.skip_capture:
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable,
            "scripts/m2_speech_preservation_harness.py",
            "--input-device", str(args.input_device),
            "--output-device", str(args.output_device),
            "--noise-device", str(args.noise_device),
            "--self-stim", args.self_stim,
            "--self-duration-ms", str(args.self_duration_ms),
            "--self-amplitude", str(args.self_amplitude),
            "--both-mode", "gain",
            "--both-gain", "1.0",
            "--seed", str(seed),
            "--speech-override", "vad",
            "--similarity-threshold", str(args.similarity_threshold),
            "--onset-latency-ms", str(args.onset_latency_ms),
            "--onset-method", args.onset_method,
            "--causal-tolerance-ms", str(args.causal_tolerance_ms),
            "--expected-text", args.expected_text,
            "--save-dir", str(save_dir),
        ])

    run([
        sys.executable,
        "scripts/m2_offline_eval.py",
        *[str(path) for path in capture_dirs],
        "--self-stim", args.self_stim,
        "--self-duration-ms", str(args.self_duration_ms),
        "--self-amplitude", str(args.self_amplitude),
        "--similarity-threshold", str(args.similarity_threshold),
        "--onset-latency-ms", str(args.onset_latency_ms),
        "--onset-method", args.onset_method,
        "--causal-tolerance-ms", str(args.causal_tolerance_ms),
        "--expected-text", args.expected_text,
        "--output-dir", str(offline_root),
    ])

    export_jsonl(offline_root / "offline_summary.json", dataset_jsonl)
    print(json.dumps({
        "root": str(root),
        "captures": [str(path) for path in capture_dirs],
        "offline_summary": str(offline_root / "offline_summary.json"),
        "dataset_jsonl": str(dataset_jsonl),
    }, indent=2))


if __name__ == "__main__":
    main()
