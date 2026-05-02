
#!/usr/bin/env python3
"""M2 Benchmark Quick Reference Card

Minimal CLI wrapper for the m2-benchmark Hermes skill.

Usage:
    python m2_benchmark_quick.py <command> [options]

Commands:
    calibrate    - Sweep self-amplitude to find operating point
    benchmark    - Run multi-seed benchmark at fixed amplitude
    offline      - Replay strategies against saved captures

Examples:
    # Calibration
    python m2_benchmark_quick.py calibrate

    # Benchmark (after choosing amplitude from calibration)
    python m2_benchmark_quick.py benchmark --amplitude 0.15 --seeds 5

    # Offline replay
    python m2_benchmark_quick.py offline --capture-root data/m2_benchmark/calibration_*/captures

"""

import subprocess
import sys
import os

SCRIPT_PATH = "scripts/m2_benchmark.py"
PYTHON_CMD = "./.venv/bin/python"
DEFAULT_ROOT = "data/m2_benchmark"

def build_args(mode, **kwargs):
    """Build CLI arguments for m2_benchmark.py"""
    args = []
    
    # Mode
    args.extend(["--mode", mode])
    
    # Common flags
    args.extend(["--self-stim", kwargs.get("stim", "vowel_ah_var_nearfield")])
    args.extend(["--input-device", str(kwargs.get("input", 3))])
    args.extend(["--output-device", str(kwargs.get("output", 5))])
    args.extend(["--noise-device", str(kwargs.get("noise", 4))])
    args.extend(["--save-root", kwargs.get("root", DEFAULT_ROOT)])
    args.extend(["--similarity-threshold", str(kwargs.get("threshold", 0.20))])
    args.extend(["--onset-method", kwargs.get("onset_method", "energy")])
    args.extend(["--causal-tolerance-ms", str(kwargs.get("causal_tol", 75.0))])
    args.extend(["--speech-override", kwargs.get("speech_override", "vad")])
    
    if mode == "calibrate":
        args.extend(["--self-duration-ms", str(kwargs.get("duration", 350))])
    
    elif mode == "benchmark":
        args.extend(["--self-amplitude", str(kwargs.get("amplitude", 0.15))])
        args.extend(["--self-duration-ms", str(kwargs.get("duration", 350))])
        args.extend(["--num-seeds", str(kwargs.get("seeds", 5))])
        args.extend(["--both-gain", str(kwargs.get("both_gain", 0.5))])
        args.extend(["--both-mode", kwargs.get("both_mode", "gain")])
    
    elif mode == "offline":
        if "capture_root" in kwargs:
            args.extend(["--capture-root", kwargs["capture_root"]])
        args.extend(["--num-seeds", str(kwargs.get("seeds", 1))])
    
    # M4 override
    if kwargs.get("m4_enable", False):
        args.append("--m4-enable")
        args.extend(["--m4-self-threshold", str(kwargs.get("m4_threshold", 0.4))])
        args.extend(["--m4-min-buffer-ms", str(kwargs.get("m4_buffer", 500))])
    
    # Subtract mode
    if kwargs.get("both_mode", "gain") == "subtract":
        args.extend(["--both-subtract-gain", str(kwargs.get("subtract_gain", 0.25))])
    
    return args

def run(mode, **kwargs):
    """Execute m2_benchmark.py"""
    args = build_args(mode, **kwargs)
    cmd = [PYTHON_CMD, SCRIPT_PATH] + args
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "calibrate":
        run("calibrate",
            stim=sys.argv[2] if len(sys.argv) > 2 else "vowel_ah_var_nearfield",
            input=int(sys.argv[3]) if len(sys.argv) > 3 else 3,
            output=int(sys.argv[4]) if len(sys.argv) > 4 else 5,
            noise=int(sys.argv[5]) if len(sys.argv) > 5 else 4)
    
    elif mode == "benchmark":
        amplitude = 0.15
        seeds = 5
        # Parse simple flags
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--amplitude" and i+1 < len(sys.argv):
                amplitude = float(sys.argv[i+1])
                i += 2
            elif sys.argv[i] == "--seeds" and i+1 < len(sys.argv):
                seeds = int(sys.argv[i+1])
                i += 2
            else:
                i += 1
        run("benchmark",
            amplitude=amplitude,
            seeds=seeds)
    
    elif mode == "offline":
        capture_root = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--capture-root" and i+1 < len(sys.argv):
                capture_root = sys.argv[i+1]
                i += 2
            else:
                i += 1
        run("offline", capture_root=capture_root)
    
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
