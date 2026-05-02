#!/usr/bin/env python3
"""
Quick smoke test for Corollary audio devices and sample rates.
Exits with 0 if devices exist and support expected sample rates,
otherwise prints a clear error and exits with 1.
"""

import sys
import json

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed.")
    sys.exit(1)

EXPECTED_SR = 16000
INPUT_DEV = 3
OUTPUT_DEV = 5
NOISE_DEV = 4

def check_device(idx, kind):
    try:
        info = sd.query_devices(idx)
    except Exception as e:
        print(f"ERROR: Cannot query {kind} device index {idx}: {e}")
        return False
    name = info.get("name", "<unknown>")
    max_in = info.get("max_input_channels", 0)
    max_out = info.get("max_output_channels", 0)
    if kind == "input" and max_in < 1:
        print(f"ERROR: {kind} device {idx} ({name}) has no input channels.")
        return False
    if kind == "output" and max_out < 1:
        print(f"ERROR: {kind} device {idx} ({name}) has no output channels.")
        return False
    srs = sd.query_devices(idx).get("default_samplerate", 0)
    # Try to open a stream at expected SR (lightweight check)
    ok = True
    try:
        if kind in ("input", "output"):
            kwargs = {"device": idx, "channels": 1, "samplerate": EXPECTED_SR, "dtype": "float32"}
            if kind == "input":
                stream = sd.InputStream(**kwargs)
            else:
                stream = sd.OutputStream(**kwargs)
            stream.close()
    except Exception as e:
        print(f"ERROR: {kind} device {idx} ({name}) cannot open at {EXPECTED_SR} Hz: {e}")
        ok = False
    if ok:
        print(f"OK: {kind} device {idx} ({name}) supports {EXPECTED_SR} Hz.")
    return ok

def main():
    print("Checking audio devices...")
    ok = True
    ok = check_device(INPUT_DEV, "input") and ok
    ok = check_device(OUTPUT_DEV, "output") and ok
    ok = check_device(NOISE_DEV, "output") and ok
    if not ok:
        print("\nSmoke test FAILED. Fix device configuration or indices.")
        sys.exit(1)
    print("\nSmoke test PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
