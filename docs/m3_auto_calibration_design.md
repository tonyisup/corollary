# Auto-Calibration Bootstrap Design

> Status: Design document — not yet implemented

## Motivation

Currently, the room filter must be manually calibrated:
1. Record a self-playback capture
2. Run `m3_broca_causal_filter_calibration.py`
3. Copy filter to production location

Auto-calibration would make the system "just work" at startup by bootstrapping the room filter automatically.

## Design

### Bootstrap Phase (at startup)

```
1. Synthesize calibration phrase (known text, e.g. "I was speaking, then paused.")
2. Play through TTS output device
3. Capture through mic input device
4. Fit room filter on capture (self-only, no human speech)
5. Validate filter quality (self-score should be high)
6. Export filter + threshold to production location
7. Proceed to main loop with calibrated filter
```

### Integration Points

**New script:** `scripts/m3_auto_calibrate.py`
- Standalone bootstrap that can be run at startup
- Outputs filter to `data/m3_broca_room_filter_production.npy`
- Outputs calibration metadata to `data/m3_broca_room_filter_production.json`

**Harness integration:** `--auto-calibrate` flag on `m3_live_e2e_harness.py`
- Runs bootstrap before scenarios
- Uses the fitted filter for all subsequent scenarios

**Production runtime:** Call `auto_calibrate()` at startup
- Block until calibration completes
- Fall back to default filter if calibration fails

### Calibration Quality Checks

After fitting, validate:
1. **Self-score threshold:** room_local_score of calibration capture must be > 0.02
2. **Filter stability:** filter taps should be reasonable (no NaN, no extreme values)
3. **Fallback:** if validation fails, use identity filter (no room coloring) + conservative threshold

### Threshold Selection

The bootstrap produces a single self-playback capture. Without separate positive/negative controls, we can't compute a data-driven threshold. Options:

1. **Fixed threshold:** Use 0.020 (validated across multiple calibrations)
2. **Self-relative threshold:** Set threshold at 50% of the calibration capture's room_local_score
3. **Conservative default:** Use 0.010 (lowest observed threshold that maintains separation)

**Recommendation:** Option 2 (self-relative). The calibration capture's room_local_score indicates the current room's coloring strength. Setting threshold at 50% of this value adapts to different room acoustics.

### Error Handling

| Failure | Fallback |
|---------|----------|
| TTS synthesis fails | Use pre-bundled filter from last successful calibration |
| Capture is silent (mic not connected) | Use identity filter + fixed threshold 0.020 |
| Filter fitting fails (singular matrix) | Use identity filter + fixed threshold 0.020 |
| Self-score too low (< 0.01) | Warn user, use fixed threshold 0.010 |

### Implementation Plan

1. `scripts/m3_auto_calibrate.py` — standalone bootstrap script
2. `--auto-calibrate` flag on live harness
3. `auto_calibrate()` function in production runtime
4. Quality checks + fallback logic
5. Unit tests for calibration pipeline

### Open Questions

- Should the bootstrap phrase be the same as the production TTS text, or a dedicated calibration phrase?
- How long should the calibration capture be? (Current: ~5s of TTS)
- Should we re-calibrate periodically, or only at startup?
- How do we handle multi-room setups (different filters for different rooms)?
