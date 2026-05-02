#!/usr/bin/env python3
"""
M4 autoresearch loop: find minimal buffer/threshold/duration that yields self_frames.

Runs repeated M2 benchmarks while sweeping:
  - m4_min_buffer_ms
  - m4_registration_threshold
  - m4_self_threshold
  - self_duration_ms

Reports best config (highest self_frame count) and saves it to best_config.json.
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

# Config search space
BUFFER_MS   = [300, 400, 500, 700]
THRESHOLDS  = [0.15, 0.2, 0.25, 0.3]
DURATIONS   = [700, 1000]

BASE_CMD = [
    sys.executable, "scripts/m2_benchmark.py",
    "--mode", "benchmark",
    "--self-stim", "vowel_ah_var_nearfield",
    "--input-device", "3",
    "--output-device", "5",
    "--noise-device", "4",
    "--m4-enable",
    "--self-amplitude", "0.2",
    "--similarity-threshold", "0.2",
    "--onset-latency-ms", "400.0",
    "--onset-method", "energy",
    "--causal-tolerance-ms", "75.0",
    "--self-match-mode", "embedding",
    "--self-match-search-ms", "50",
    "--cerebellum-model", "data/m4_forward_model_focused/forward_model_1777727778.json",
]

def run_one(buffer_ms, th, dur_ms, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = BASE_CMD + [
        "--m4-min-buffer-ms", str(buffer_ms),
        "--m4-registration-threshold", str(th),
        "--m4-self-threshold", str(th),
        "--self-duration-ms", str(dur_ms),
        "--save-root", str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    # parse summary file for self_frames
    summaries = list(run_dir.glob("*/summary.json"))
    total_self = 0
    for sf in summaries:
        with open(sf) as f:
            d = json.load(f)
            total_self += d.get("self_frames", 0)
    return total_self, result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--output-root", default="data/autoresearch_m4")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    all_results = []
    trial = 0

    for buf, th, dur in product(BUFFER_MS, THRESHOLDS, DURATIONS):
        if args.max_trials and trial >= args.max_trials:
            break
        trial_dir = out_root / f"trial_{trial}"
        print(f"\n=== Trial {trial}: buf={buf}ms, th={th}, dur={dur}ms ===")
        total_self, rc = run_one(buf, th, dur, trial_dir)
        all_results.append({
            "buffer_ms": buf,
            "threshold": th,
            "duration_ms": dur,
            "self_frames": total_self,
            "returncode": rc,
        })
        print(f"  -> self_frames = {total_self}")
        trial += 1

    best = max(all_results, key=lambda r: r["self_frames"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (out_root / "best_config.json").write_text(json.dumps(best, indent=2))

    print("\n=== Best config ===")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
