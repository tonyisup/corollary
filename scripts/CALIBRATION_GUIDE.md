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
    --num-trials 3
```

**Latest calibration result (2026-05-02):**
```json
{
  "aggregate": {
    "num_trials": 1,
    "mean_suggested_onset_latency_ms": 23.56,
    "std_suggested_onset_latency_ms": 0.00,
    "mean_offset_ms": 23.56,
    "mean_lxcorr_score": 0.5537
  }
}
```
**Suggested onset_latency_ms: 24ms** (was 400ms in initial config)

### 2. Read suggested onset_latency_ms
```bash
cat data/m2_calibration_test/calibrate_self_start_<timestamp>/calibration_summary.json
# aggregate.mean_suggested_onset_latency_ms
```

### 3. Run benchmark with calibrated value
```bash
./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \
    --self-stim vowel_ah_var_nearfield \
    --self-amplitude 0.15 \
    --self-duration-ms 350 \
    --world-speech data/world_speech_sample.npy \
    --onset-latency-ms 24 \
    --num-seeds 3 \
    --save-root data/m2_benchmark_calibrated_24ms
```

**Result with 24ms calibration:**
- Mean WER delta: -0.014 (improvement)
- Self-scores: 0.4-0.63 (vs 0.0 with 400ms)
- Mean S/W ratio: 16.4 dB
- Artifacts: `data/m2_benchmark_calibrated_24ms/`

Or use inline offset:
```bash
./.venv/bin/python scripts/m2_benchmark.py --mode benchmark \
    --onset-latency-ms 400.0 \
    --calibrated-offset-ms -376  # 400 - 376 = 24ms
```

## Files Changed
- `scripts/calibrate_self_start.py` — New calibration script
- `scripts/m2_benchmark.py` — Added `--calibrated-offset-ms` argument
- `packages/thalamus/attributor.py` — Added M4 telemetry

## What Calibration Does
1. Records N trials of self-stim mixed with world speech
2. Cross-correlates captured audio with known self PCM (global + local)
3. Measures actual offset and suggests corrected `onset_latency_ms`
4. Reports statistics across trials

## M4 Telemetry (New)
The `ThalamicAttributor` now tracks M4 diagnostic info in `AttributionResult.metadata`:
- `m4_attempted` (bool) — Was M4 invoked?
- `m4_outcome` (str) — One of: `accepted`, `below_threshold`, `exception`, `buffer_too_short`, `not_attempted`
- `m4_confidence` (float|None) — M4 confidence score if attempted
- `m4_buffer_length` (int) — Samples in M4 buffer when checked

## Verification
After calibration, verify:
- `m4_triggered_frames` > 0 (M4 actually runs)
- Self frames properly attributed during playback window
- Meaningful M4 buffer/threshold sweep results
- Check `metadata.m4_outcome` in frame_verdicts.jsonl for telemetry
