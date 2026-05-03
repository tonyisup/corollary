#!/usr/bin/env python3
"""Corrected M4 Sweep with fixed registration threshold."""
import subprocess
import json
import os
import sys
from datetime import datetime

# Sweep configuration
BUFFERS = [60, 90, 120, 200]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
SEEDS = 3
BASE_DIR = "data/m4_hyperparameter_sweep_corrected"

# Fixed parameters (with registration threshold fix)
FIXED_ARGS = [
    "--mode", "benchmark",
    "--m4-enable",
    "--self-duration-ms", "700",
    "--onset-latency-ms", "24",
    "--causal-tolerance-ms", "100",
    # Note: registration threshold now auto-capped at 0.05 by the code fix
    "--m4-registration-threshold", "0.05",  # Just a hint; code caps at 0.05
    "--speech-override", "true",
]

def run_benchmark(buffer_ms, threshold, run_idx):
    """Run a single benchmark with given parameters."""
    save_root = f"{BASE_DIR}/sweep_b{buffer_ms}_t{threshold:.2f}_r{run_idx}"
    cmd = [
        sys.executable, "scripts/m2_benchmark.py",
        *FIXED_ARGS,
        "--m4-min-buffer-ms", str(buffer_ms),
        "--m4-self-threshold", str(threshold),
        "--save-root", save_root,
    ]
    
    print(f"[Run {run_idx}] buffer={buffer_ms}ms, threshold={threshold:.2f}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    return result.returncode, save_root

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{BASE_DIR}/sweep_results_{timestamp}.jsonl"
    
    total_runs = len(BUFFERS) * len(THRESHOLDS) * SEEDS
    run_count = 0
    
    print(f"=== Corrected M4 Sweep (registration threshold fixed) ===")
    print(f"Buffers: {BUFFERS}ms")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Seeds per config: {SEEDS}")
    print(f"Total runs: {total_runs}\n")
    
    with open(results_file, "w") as fout:
        for buffer_ms in BUFFERS:
            for threshold in THRESHOLDS:
                for seed_idx in range(SEEDS):
                    run_count += 1
                    print(f"[{run_count}/{total_runs}] ", end="")
                    
                    rc, save_root = run_benchmark(buffer_ms, threshold, seed_idx)
                    
                    # Extract summary stats
                    summary_path = None
                    for root, dirs, files in os.walk(save_root):
                        if "summary.json" in files:
                            summary_path = os.path.join(root, "summary.json")
                            break
                    
                    result = {
                        "buffer_ms": buffer_ms,
                        "threshold": threshold,
                        "seed_idx": seed_idx,
                        "returncode": rc,
                        "save_root": save_root,
                    }
                    
                    if summary_path and os.path.exists(summary_path):
                        with open(summary_path) as sf:
                            summary = json.load(sf)
                            result["m4_triggered_frames"] = summary.get("m4_triggered_frames", 0)
                            result["self_frames"] = summary.get("self_frames", 0)
                            result["both_frames"] = summary.get("both_frames", 0)
                            result["world_frames"] = summary.get("world_frames", 0)
                            result["raw_wer"] = summary.get("raw_wer", None)
                            result["gated_wer"] = summary.get("gated_wer", None)
                    
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    
                    if rc != 0:
                        print(f"  WARNING: Exit code {rc}")
    
    print(f"\n=== Sweep Complete ===")
    print(f"Results saved to: {results_file}")
    
    # Print summary table
    print("\n--- Summary by Config (averaged over seeds) ---")
    configs = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            key = (r["buffer_ms"], r["threshold"])
            if key not in configs:
                configs[key] = []
            configs[key].append(r)
    
    print(f"{'Buffer':>6} {'Thresh':>6} {'Avg Trig':>8} {'Avg Self':>9} {'Self%':>7} {'Avg WERΔ':>9}")
    print("-" * 55)
    for (buf, thr), runs in sorted(configs.items()):
        avg_trig = sum(r.get("m4_triggered_frames",0) for r in runs) / len(runs)
        avg_self = sum(r.get("self_frames",0) for r in runs) / len(runs)
        self_pct = (avg_self / avg_trig * 100) if avg_trig > 0 else 0
        avg_wer_delta = sum((r.get("gated_wer",0) or 0) - (r.get("raw_wer",0) or 0) for r in runs) / len(runs)
        print(f"{buf:6d} {thr:6.2f} {avg_trig:8.1f} {avg_self:9.1f} {self_pct:6.1f}% {avg_wer_delta:9.3f}")

if __name__ == "__main__":
    main()
