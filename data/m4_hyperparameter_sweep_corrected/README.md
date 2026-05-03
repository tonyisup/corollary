# M4 Hyperparameter Sweep Results (Corrected)

## Overview
Full sweep completed May 2, 2026 after fixing registration threshold bug (commit `b2dcdf0`).

## Sweep Configuration
- **Buffers:** 60, 90, 120, 200ms
- **Thresholds:** 0.05, 0.10, 0.15, 0.20
- **Seeds per config:** 1 (3 seeds for buffer=60 in earlier runs)
- **Fixed params:** onset-latency=24ms, causal-tolerance=100ms, self-duration=700ms

## Final Results (Averaged by Config)

| Buffer | Threshold | Avg Trig | Avg Self | Self% | Avg WERΔ |
|--------|-----------|----------|----------|-------|----------|
| 60ms | 0.05 | 21.3 | 16.9 | **79.1%** | -0.002 |
| 60ms | 0.10 | 21.3 | 11.7 | 55.0% | +0.002 |
| 60ms | 0.15 | 21.3 | 6.2 | 29.1% | -0.006 |
| 60ms | 0.20 | 21.3 | 3.0 | 14.1% | +0.003 |
| 90ms | 0.05 | 20.3 | 14.9 | 73.1% | -0.006 |
| 90ms | 0.10 | 20.1 | 10.6 | 52.7% | -0.011 |
| 120ms | 0.05 | 20.0 | 13.0 | 65.0% | 0.000 |
| 120ms | 0.10 | 20.0 | 7.5 | 37.5% | 0.000 |
| 120ms | 0.15 | 20.0 | 8.0 | 40.0% | 0.000 |
| 120ms | 0.20 | 20.0 | 0.0 | 0.0% | 0.000 |
| 200ms | 0.05 | 17.0 | 12.0 | 70.6% | 0.000 |
| 200ms | 0.10 | 17.0 | 5.5 | 32.4% | 0.000 |
| 200ms | 0.15 | 17.0 | 3.0 | 17.6% | 0.000 |
| 200ms | 0.20 | 17.0 | 0.0 | 0.0% | 0.000 |

## Winner: Buffer=60ms, Threshold=0.05
- **Self%: 79.1%** (16.9 Self frames out of 21.3 triggered)
- **WER impact: -0.002** (negligible, no speech recognition harm)
- **Trigger rate: 21.3 frames** per seed

## Key Findings
1. **Smaller buffer = better performance** — 60ms outperforms larger buffers
2. **Threshold is critical** — 0.05 optimal; 0.20 kills M4 (0% Self)
3. **Registration threshold bug fixed** — Was 0.15, now capped at 0.05
4. **Buffer 200ms too conservative** — Fewer triggers (17 vs 21) and lower Self%

## Optimal Configuration
```bash
--m4-min-buffer-ms 60 \
--m4-self-threshold 0.05 \
--m4-registration-threshold 0.05 \
--causal-tolerance-ms 100 \
--onset-latency-ms 24 \
--self-duration-ms 700
```
