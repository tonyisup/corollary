# M2 Consolidated Benchmark Results

> Date: 2026-05-01
> Stimulus: vowel_ah_var_nearfield, amp=0.2, 350ms
> World: 6.22s GuyNeural male voice
> Seeds: 10 (5 per benchmark run × 2 runs)
> Attribution: similarity_threshold=0.20, onset_latency=400ms, energy onset, tolerance=75ms

## Summary

| Strategy | N | Mean ΔWER | Std | I/N/R | Retention | Energy | D/S/I |
|---|---|---|---|---|---|---|---|
| **preserve** | 10 | +0.000 | 0.000 | 0/10/0 | 0.83 | 1.00 | 0.0/2.6/0.4 |
| **gain_0p5** | 10 | +0.017 | 0.029 | 0/7/3 | 0.80 | 1.00 | 0.8/2.6/0.0 |
| subtract_0p1 | 5 | +0.009 | 0.017 | 0/4/1 | 0.89 | 1.00 | 0.0/2.2/0.8 |
| **subtract_0p25** | 5 | +0.000 | 0.000 | 0/5/0 | 0.90 | 0.99 | 0.0/2.0/0.8 |
| subtract_0p5 | 5 | +0.009 | 0.017 | 0/4/1 | 0.89 | 0.99 | 0.0/2.2/0.8 |
| subtract_0p75 | 5 | +0.009 | 0.017 | 0/4/1 | 0.88 | 0.98 | 0.0/2.6/0.4 |
| subtract_1p0 | 5 | +0.009 | 0.017 | 0/4/1 | 0.88 | 0.98 | 0.0/2.6/0.4 |

## Key Findings

### 1. No strategy improves WER
Zero strategies show mean improvement across seeds. The best (preserve and subtract_0p25) are neutral. This confirms the M2 fundamental limitation: without a room-colored self template, suppression always hurts or is neutral.

### 2. subtract_0p25 is the best subtraction strength
- Neutral mean delta (0.000) vs +0.009 for other strengths
- Highest retention (0.90) and lowest substitution rate (2.0)
- Energy retained is 0.99 (vs 1.00 for preserve — minimal additional attenuation)
- Higher strengths (0.5+) show slightly worse retention and more substitutions, suggesting the dry template doesn't match the room-colored self

### 3. gain_0p5 is the worst strategy
- Highest mean delta (+0.017) and most regressions (3/10)
- Lowest retention (0.80)
- Blunt attenuation of BOTH frames removes world speech energy along with self

### 4. Error breakdown reveals the failure mode
- **Deletions are near-zero** for all strategies — world speech words are not being dropped
- **Substitutions dominate** (2.0-2.6 per transcript) — the ASR is misrecognizing words, likely because the self signal is masking phonemes
- **Insertions are low** (0.0-0.8) — self signal is not leaking through as spurious words
- This suggests the problem is **masking** (self energy obscuring world phonemes), not **leakage** (self being mistaken for world)

### 5. Energy retained is near 1.0 for all strategies
The BOTH-frame gating barely affects the overall energy. This is because:
- BOTH frames are a small fraction of the total capture (~10-15 frames out of ~200)
- The self window (350ms) is short relative to the world speech (6.22s)
- Most of the world speech energy is in WORLD frames that pass through unchanged

## Conclusion

**The M2 gating problem is a masking problem, not an energy problem.** The self signal masks world speech phonemes during the overlap window, causing ASR substitutions. But the total energy affected is small, so energy-based metrics don't capture the damage.

**Without the forward model**, no suppression strategy can help because:
1. We can't distinguish self from world in BOTH frames (VAD says "speech" for both)
2. The dry template doesn't match the room-colored self at the mic
3. Any suppression in BOTH frames removes world speech along with self

**The path forward is M4 + forward model training**, which would enable hard SELF frames during speech, eliminating the BOTH-frame masking problem entirely.

## Artifacts

- 10 captures: `data/m2_benchmark/benchmark_20260501_101815/` (seeds 0-4) + `data/m2_benchmark/benchmark_20260501_133449/` (seeds 0-4)
- Offline results: `data/m2_benchmark/offline_20260501_134611/` (old 5) + `data/m2_benchmark/offline_20260501_134651/` (new 5, all strategies)
