# M2 Benchmark — Full Results

> Date: 2026-05-01
> Author: OWL (automated run)
> Hardware: Fifine mic (device 3), Mac mini built-in speaker (device 5),
>           external headphones (device 4)

## Summary

The redesigned benchmark (`scripts/m2_benchmark.py`) successfully measures
speech preservation in a regime where the old benchmark could not. Three
conditions were tested across 5 seeds each.

## Key Finding

**M2 gating is neutral-to-harmful across all tested conditions.** The
conservative BOTH-frame policy with gain=0.5 does not improve world-speech
recoverability. The root cause is that VAD detects vowel stimuli as speech,
so the attributor never produces hard SELF frames — it only reduces BOTH
frames by half, which attenuates both self and world equally.

## Results Table

| Condition | Stimulus | Amp | Dur | Raw WER | Gated WER | Delta | Improved/Neutral/Regressed |
|---|---|---|---|---|---|---|---|
| A | vowel_ah_var_nearfield | 0.20 | 350ms | 0.252 | 0.261 | +0.009 | 0/4/1 |
| B | vowel_ah_var_nearfield | 0.50 | 350ms | 0.452 | 0.548 | +0.096 | 2/1/2 |
| C | chirp | 0.30 | 200ms | 0.139 | 0.104 | -0.035 | 1/4/0 |

## Analysis

### Condition A (vowel amp=0.20)
- Gating is neutral. Self is present but not dominant at the mic.
- Zero hard SELF frames (all motor-window frames route to BOTH).
- Seed 4 is an outlier (WER 0.435) — ASR had trouble with that capture.

### Condition B (vowel amp=0.50)
- Gating hurts more than it helps. Higher self amplitude means BOTH-frame
  attenuation removes significant world speech energy.
- Seed 0 and 1 show large regressions (+0.304 each).
- Seeds 2 and 3 show modest improvements (-0.087, -0.043) — gating helped
  when the raw capture was already poor.

### Condition C (chirp amp=0.30)
- Chirp is at -16 dB below world speech at the mic — barely detectable.
- WER improvement on seed 0 (0.174 → 0.000) is likely ASR variance, not
  real gating benefit. The 6 BOTH frames are at -50 dB (noise floor).
- Even with a non-speech stimulus, zero hard SELF frames are produced
  because VAD flags world speech in the overlap zone.

### Seed Variance
- Raw WER varies significantly across seeds at the same amplitude:
  amp=0.50 ranges from 0.217 to 0.696 (3.2x spread).
- This confirms the need for multi-seed averaging.

### S/W Ratio vs WER
- S/W ratio alone does not predict WER. The spatial/temporal structure of
  the overlap matters more than raw power ratio.

## Implications

The M2 conservative policy needs one of these to outperform preservation:

1. **Non-speech self stimulus during normal operation** — use chirp/tone for
   self-monitoring, so VAD doesn't flag it. Enables hard SELF suppression.

2. **Spectral subtraction in BOTH frames** — estimate and subtract self
   component using the efference copy instead of blunt gain reduction.
   Requires a good forward model (M1b cerebellum).

3. **M4 self-voice detector** — use embeddings to distinguish self from
   other during speech, enabling confident SELF classification.

## Artifacts

- `data/m2_benchmark/calibration_20260501_100348/` — Amplitude sweep (9 levels, 1 seed each)
- `data/m2_benchmark/benchmark_20260501_101815/` — Condition A (vowel amp=0.20, 5 seeds)
- `data/m2_benchmark/benchmark_20260501_102103/` — Condition B (vowel amp=0.50, 5 seeds)
- `data/m2_benchmark/benchmark_20260501_102329/` — Condition C (chirp amp=0.30, 5 seeds)
- `data/m2_benchmark/offline_20260501_102950/` — Offline replay (preserve vs gain_0p5)
