
# M2 Benchmark Timing Calibration

## Problem
The M2 benchmark uses theoretical timing to determine when Corollary's self-audio is playing (`is_playing` window). Hardware latencies (DAC, speaker, mic, ADC) mean the actual self-audio onset in the captured signal differs from the theoretical value. This causes:
- `is_playing` window misalignment with actual self-audio
- Lower acoustic similarity scores (`is_self_like` fails)
- M4 self-voice detector never triggers
- `m4_triggered_frames` stays at 0 → benchmark sweeps waste time

## Quick Start

### 1. Calibrate
```bash
./.venv/bin/python scripts/calibrate_self_start.py \
    --self-stim vowel_ah_var_nearfield \
    --self-duration-ms 350 \
    --self-amplitude 0.15 \
    --input-device 3 --output-device 5 --noise-device 4 \
    --num-trials 5
```

### 2. Read suggested onset_latency_ms
```bash
cat data/calibration/calibrate_self_start_<timestamp>/calibration_summary.json
# aggregate.mean_suggested_onset_latency_ms
```

### 3. Run benchmark with calibrated value
```bash
./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \
    --self-stim vowel_ah_var_nearfield \
    --self-amplitude 0.15 \
    --self-duration-ms 350 \
    --world-speech data/world_speech_sample.npy \
    --input-device 3 --output-device 5 --noise-device 4 \
    --onset-latency-ms <CALIBRATED> \
    --num-seeds 5 \
    --save-root data/m2_benchmark
```

Or use inline offset:
```bash
./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \
    --onset-latency-ms 400.0 \
    --calibrated-offset-ms -15.3  # adds to latency
```

## Files Changed
- `scripts/calibrate_self_start.py` — New calibration script
- `scripts/m2_benchmark.py` — Added `--calibrated-offset-ms` argument

## What Calibration Does
1. Records N trials of self-stim mixed with world speech
2. Cross-correlates captured audio with known self PCM (global + local)
3. Measures actual offset and suggests corrected `onset_latency_ms`
4. Reports statistics across trials

## Verification
After calibration, verify:
- `m4_triggered_frames` > 0 (M4 actually runs)
- Self frames properly attributed during playback window
- Meaningful M4 buffer/threshold sweep results
