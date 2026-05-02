#!/usr/bin/env python3
"""M4 Buffer and Threshold Sweep for Corollary

Sweeps m4_min_buffer_ms and m4_registration_threshold / m4_self_threshold
to find minimal working conditions that trigger M4 self-voice classification.

Run:
    ./.venv/bin/python scripts/m4_buffer_threshold_sweep.py \\
        --output-root data/m4_sweep \\
        --self-duration-ms 350 \\
        --self-match-mode embedding
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent


def build_benchmark_cmd(args, buf_ms, th, extra_args):
    """Build command line for m2_benchmark.py."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "m2_benchmark.py"),
        "--mode", "benchmark",
        "--self-stim", "vowel_ah_var_nearfield",
        "--self-amplitude", "0.15",
        "--self-duration-ms", str(args.self_duration_ms),
        "--world-speech", str(REPO_ROOT / "data" / "world_speech_sample.npy"),
        "--expected-text",
        "the quick brown fox jumps over the lazy dog and then runs through the forest to find a quiet place near the river",
        "--input-device", str(args.input_device),
        "--output-device", str(args.output_device),
        "--noise-device", str(args.noise_device),
        "--save-root", args.output_root,
        "--num-seeds", str(args.num_seeds),
        "--similarity-threshold", "0.20",
        "--onset-latency-ms", "400.0",
        "--onset-method", "energy",
        "--causal-tolerance-ms", "75.0",
        "--self-match-mode", args.self_match_mode,
        "--both-gain", "0.5",
        "--both-mode", "gain",
        # M4 params
        "--m4-enable",
        "--m4-self-threshold", str(th),
        "--m4-registration-threshold", str(th),
        "--m4-min-buffer-ms", str(buf_ms),
    ]
    if args.m4_mock:
        cmd.append("--m4-mock")
    cmd.extend(extra_args)
    return cmd


def build_control_cmd(args, extra_args):
    """Build control run command."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "m2_benchmark.py"),
        "--mode", "benchmark",
        "--self-stim", "vowel_ah_var_nearfield",
        "--self-amplitude", "0.15",
        "--self-duration-ms", "700",
        "--world-speech", str(REPO_ROOT / "data" / "world_speech_sample.npy"),
        "--expected-text",
        "the quick brown fox jumps over the lazy dog and then runs through the forest to find a quiet place near the river",
        "--input-device", str(args.input_device),
        "--output-device", str(args.output_device),
        "--noise-device", str(args.noise_device),
        "--save-root", args.output_root,
        "--num-seeds", str(args.num_seeds),
        "--similarity-threshold", "0.20",
        "--onset-latency-ms", "400.0",
        "--onset-method", "energy",
        "--causal-tolerance-ms", "75.0",
        "--self-match-mode", args.self_match_mode,
        "--both-gain", "0.5",
        "--both-mode", "gain",
        "--m4-enable",
        "--m4-self-threshold", "0.4",
        "--m4-registration-threshold", "0.4",
        "--m4-min-buffer-ms", "500",
    ]
    if args.m4_mock:
        cmd.append("--m4-mock")
    cmd.extend(extra_args)
    return cmd


def run_cmd(cmd, output_root):
    """Run a command and return parsed summary JSON."""
    print(f"Running: {' '.join(cmd[:15])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    # Find the most recent benchmark summary
    out_dir = Path(output_root)
    if not out_dir.exists():
        print(f"  ERROR: output dir not created")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return None
    json_files = sorted(out_dir.glob("benchmark_*/benchmark_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        try:
            return json.loads(json_files[0].read_text())
        except Exception as e:
            print(f"  ERROR reading {json_files[0]}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="M4 buffer and threshold sweep")
    parser.add_argument("--output-root", default="data/m4_sweep")
    parser.add_argument("--self-duration-ms", type=int, default=350)
    parser.add_argument("--self-match-mode", choices=["global_xcorr", "local_xcorr", "embedding"], default="embedding")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--m4-mock", action="store_true", help="Use mock embeddings for deterministic results")
    parser.add_argument("--skip-buffer-sweep", action="store_true", help="Skip buffer sweep (threshold sweep only)")
    parser.add_argument("--skip-control", action="store_true", help="Skip control run")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra args passed to benchmark")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    buffers = [250, 300, 350, 500]
    thresholds = [0.2, 0.25, 0.3, 0.4]

    results = []

    print("=" * 70)
    print("M4 BUFFER AND THRESHOLD SWEEP")
    print(f"  Self duration: {args.self_duration_ms}ms")
    print(f"  Self match mode: {args.self_match_mode}")
    print(f"  Num seeds: {args.num_seeds}")
    print(f"  Buffers: {buffers}")
    print(f"  Thresholds: {thresholds}")
    print("=" * 70)

    # Buffer/threshold sweep
    if not args.skip_buffer_sweep:
        for buf_ms in buffers:
            for th in thresholds:
                print(f"\n--- Buffer={buf_ms}ms, Threshold={th} ---")
                cmd = build_benchmark_cmd(args, buf_ms, th, args.extra_args)
                summary = run_cmd(cmd, args.output_root)
                if summary:
                    row = {
                        "buffer_ms": buf_ms,
                        "threshold": th,
                        "run_type": "sweep",
                        "config": summary.get("config", {}),
                        "aggregate": summary.get("aggregate", {}),
                        "num_valid_seeds": len(summary.get("results", [])),
                    }
                    # Extract self_frames from results
                    if summary.get("results"):
                        total_self_frames = sum(
                            r.get("self_frames", 0) for r in summary["results"]
                            if "error" not in r
                        )
                        row["total_self_frames"] = total_self_frames
                        row["mean_self_frames"] = total_self_frames / len([r for r in summary["results"] if "error" not in r]) if [r for r in summary["results"] if "error" not in r] else 0
                        # Also get whether M4 triggered
                        total_m4_triggered = sum(
                            r.get("m4_triggered_frames", 0) for r in summary["results"]
                            if "error" not in r
                        )
                        row["total_m4_triggered_frames"] = total_m4_triggered
                    results.append(row)
                    print(f"  Mean self_frames: {row.get('mean_self_frames', 0):.1f}, M4 triggered frames: {row.get('total_m4_triggered_frames', 0)}")
                else:
                    results.append({"buffer_ms": buf_ms, "threshold": th, "run_type": "sweep", "error": "run_failed"})
                    print("  FAILED")

    # Control run (longer stimulus)
    if not args.skip_control:
        print(f"\n--- CONTROL: 700ms stimulus, buffer=500ms, th=0.4 ---")
        cmd = build_control_cmd(args, args.extra_args)
        summary = run_cmd(cmd, args.output_root)
        if summary:
            row = {
                "buffer_ms": 500,
                "threshold": 0.4,
                "self_duration_ms": 700,
                "run_type": "control",
                "config": summary.get("config", {}),
                "aggregate": summary.get("aggregate", {}),
                "num_valid_seeds": len(summary.get("results", [])),
            }
            if summary.get("results"):
                total_self_frames = sum(
                    r.get("self_frames", 0) for r in summary["results"]
                    if "error" not in r
                )
                row["total_self_frames"] = total_self_frames
                row["mean_self_frames"] = total_self_frames / len([r for r in summary["results"] if "error" not in r]) if [r for r in summary["results"] if "error" not in r] else 0
                total_m4_triggered = sum(
                    r.get("m4_triggered_frames", 0) for r in summary["results"]
                    if "error" not in r
                )
                row["total_m4_triggered_frames"] = total_m4_triggered
            results.append(row)
            print(f"  Mean self_frames: {row.get('mean_self_frames', 0):.1f}, M4 triggered frames: {row.get('total_m4_triggered_frames', 0)}")
        else:
            results.append({"run_type": "control", "error": "run_failed"})
            print("  FAILED")

    # Save aggregate
    output = {
        "timestamp": datetime.now().isoformat(),
        "sweep_config": {
            "buffers_ms": buffers if not args.skip_buffer_sweep else [],
            "thresholds": thresholds if not args.skip_buffer_sweep else [],
            "self_duration_ms": args.self_duration_ms,
            "self_match_mode": args.self_match_mode,
            "num_seeds": args.num_seeds,
        },
        "results": results,
    }
    out_path = Path(args.output_root) / "sweep_summary.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Buffer(ms)':>12} {'Thresh':>8} {'Mean SelfF':>12} {'M4 TrigF':>12} {'Mean RawWER':>12} {'Mean D(WER)':>12}")
    print('-' * 72)
    for r in results:
        if "error" in r:
            print(f"{r['buffer_ms']:>12} {r['threshold']:>8.3f} {'ERROR':>12} {'':>12} {'':>12} {'':>12}")
        else:
            agg = r.get("aggregate", {})
            print(f"{r['buffer_ms']:>12} {r['threshold']:>8.3f} {r.get('mean_self_frames', 0):>12.1f} {r.get('total_m4_triggered_frames', 0):>12} {agg.get('mean_raw_wer', 0):>12.3f} {agg.get('mean_wer_delta', 0):>+12.3f}")
    print(f"\nFull summary: {out_path}")


if __name__ == "__main__":
    main()
