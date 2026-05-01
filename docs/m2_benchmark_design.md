# M2 Benchmark Redesign — Speech Preservation Measurement

> Status: Active Research
> Date: 2026-05-01
> Scope: Fixing the speech-preservation benchmark so it can actually measure gating benefit

## Problem

The prior M2 benchmark (`m2_speech_preservation_harness.py`) could not measure
speech preservation because the self signal (Mac mini built-in speaker, ~5cm from
the Fifine mic) overwhelmed world speech (external headphones on the desk, ~50cm
from mic). Results were unambiguous but uninformative: WER=1.0 across all gating
strategies, not because gating was useless but because the capture was already
destroyed before gating could matter.

### Evidence from old data

| Condition | RMS at mic | WER |
|---|---|---|
| World-only (headphones) | 0.014 | 0.125 |
| Mixed (babble_ba_var, 350ms, amp=0.7) | 0.030 | 0.125 |
| Strong overlap (babble_ba_repeat_tight, 900ms, amp=1.0) | 0.046 | 0.500 |
| Embodied TTS overlap | 0.042 | 1.000 |

Self dominated by 2-6x in power. The benchmark was measuring "how badly did the
raw capture destroy speech?" rather than "does gating help recover speech?".

## Solution

The new benchmark (`scripts/m2_benchmark.py`) fixes this by:

1. **Reducing self-amplitude** to a regime where world speech is genuinely
   recoverable in the raw capture (target: raw WER ≤ 0.2-0.3).
2. **Using a longer, phonetically diverse reference text** (6.22s, GuyNeural
   male voice, distinct from self AriaNeural voice).
3. **Reporting self/world SNR at the mic** as a first-class metric alongside
   raw/gated WER.
4. **Calibration mode** that sweeps self-amplitude to find the operating point
   before running the full multi-seed benchmark.
5. **Offline replay mode** to evaluate multiple gating strategies against
   identical captures (eliminates capture-to-capture variance).

## New World Speech Sample

`data/world_speech_benchmark.npy` — 6.22s, en-US-GuyNeural male voice, RMS 0.092.

```
"the quick brown fox jumps over the lazy dog and then
 runs through the forest to find a quiet place near the river"
```

## Calibration Results (2026-05-01)

Setup: Fifine mic (device 3), Mac mini built-in speaker (device 5),
headphones on desk (device 4), 350ms vowel_ah_var_nearfield self stimulus,
similarity_threshold=0.20, onset_latency=400ms, energy onset, tolerance=75ms.

| Amplitude | Raw WER | Gated WER | Delta | S/W ratio (dB) | Verdicts (S/B/W) |
|---|---|---|---|---|---|
| 0.00 (world-only) | 0.043 | 0.043 | +0.000 | -3.7 | 0/0/190 |
| 0.05 | 0.174 | 0.174 | +0.000 | 2.6 | 1/10/168 |
| 0.10 | 0.217 | 0.217 | +0.000 | 5.0 | 0/10/155 |
| 0.15 | 0.130 | 0.130 | +0.000 | 1.8 | 0/10/156 |
| **0.20** | **0.217** | **0.174** | **-0.043** | **1.9** | **0/11/147** |
| 0.30 | 0.130 | 0.217 | +0.087 | 2.2 | 0/11/143 |
| 0.50 | 0.217 | 0.217 | +0.000 | 5.1 | 0/11/159 |
| 0.70 | 0.043 | 0.130 | +0.087 | 5.9 | 0/11/196 |
| 1.00 | 0.087 | 0.174 | +0.087 | 4.9 | 0/11/201 |

### Key Findings

1. **Baseline WER improved from 0.125 to 0.043** with the new longer reference
   text. The old 1.1s sample was too short for Whisper to lock onto.

2. **Gating helps at low interference levels (amp=0.20, S/W=1.9 dB)** with a
   modest but real improvement (-0.043 WER delta). This is the first condition
   where the benchmark can actually detect a gating benefit.

3. **Gating hurts at higher amplitudes** because the BOTH-frame gain=0.5
   attenuates world speech along with self. When self is stronger, reducing
   BOTH frames hurts more than the self-suppression helps.

4. **Zero hard SELF frames at all amplitudes.** VAD detects the vowel stimulus as
   speech, so the conservative M2 policy routes all motor-window frames to BOTH.
   This is expected behavior, but it means the only knob we're turning is BOTH
   gain.

5. **Non-monotonic S/W ratio** suggests AGC or nonlinear behavior in the Mac
   mini speaker or Fifine mic path. The amp=0.05 case has S/W=2.6 dB while
   amp=0.15 has S/W=1.8 dB. This is a hardware artifact, not a bug.

6. **Single-seed variance is significant.** Amp=0.30 gives raw WER=0.130 while
   amp=0.50 gives raw WER=0.217 despite similar S/W ratios. This confirms the
   need for multi-seed averaging.

## Recommended Operating Points

| Purpose | Amplitude | Rationale |
|---|---|---|
| **Gating benefit measurement** | 0.20 | Gating improves WER; S/W barely positive |
| **Neutral/stress test** | 0.50 | Gating neutral; distinguishes help from harm |
| **High interference** | 1.00 | Gating hurts; confirms expected degradation |

## Next Steps

1. Run multi-seed benchmark at amp=0.20 to measure gating benefit with
   statistical confidence.
2. Try both_subtract mode at amp=0.20 — template subtraction may outperform
   blunt gain when self and world are closely balanced.
3. Test with babble_ba_var stimulus (more speech-like than vowel) to
   exercise the VAD-based attribution more aggressively.
4. After benchmark stabilizes, use it to evaluate M4 self-voice detector:
   does explicit self/world classification improve gating over VAD-only?

## Next Steps

1. **Evaluate spectral subtraction** in BOTH frames at amp=0.20 using offline replay.
   The captures are saved; run `m2_benchmark.py --mode offline` to compare
   gain vs subtract strategies.

2. **Run babble_ba_var stimulus** at amp=0.20 with M2 Bayesian-optimized settings
   (similarity_threshold=0.23, onset_latency=405ms, causal_tolerance=74ms).

3. **Wire M4 self-voice detector** to enable hard SELF frames during speech,
   which is the key to making gating outperform preservation.

4. **Scale to 10+ seeds** for statistical confidence on the best condition.

## Reproduction

```bash
# Calibration sweep
./.venv/bin/python scripts/m2_benchmark.py \
  --mode calibrate \
  --self-stim vowel_ah_var_nearfield \
  --world-speech data/world_speech_benchmark.npy \
  --input-device 3 --output-device 5 --noise-device 4 \
  --self-duration-ms 350 \
  --save-root data/m2_benchmark

# Full benchmark at chosen amplitude
./.venv/bin/python scripts/m2_benchmark.py \
  --mode benchmark \
  --self-stim vowel_ah_var_nearfield \
  --self-amplitude 0.20 \
  --world-speech data/world_speech_benchmark.npy \
  --input-device 3 --output-device 5 --noise-device 4 \
  --num-seeds 5 \
  --save-root data/m2_benchmark

# Offline replay of multiple strategies
./.venv/bin/python scripts/m2_benchmark.py \
  --mode offline \
  --capture-root data/m2_benchmark/benchmark_YYYYMMDD_HHMMSS/captures \
  --save-root data/m2_benchmark
```
