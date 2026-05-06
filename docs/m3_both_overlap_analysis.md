# M3 BOTH-Overlap Discrimination Analysis

> **Date:** 2026-05-06
> **Scope:** Can the controller distinguish self-dominant BOTH from human-overlap BOTH during active playback?

## Problem

When the agent is speaking (`is_in_motor_window=True`) and the attributor classifies a frame as `BOTH`, the controller currently treats it as self-activity (sustains SPEAK). This is correct for self-dominant BOTH (room reflections, TTS bleed), but incorrect for human-overlap BOTH (human speaking over the agent).

The controller needs to yield to human speech during self-playback, but cannot reliably distinguish the two cases using current signals.

## Signals Evaluated

### 1. Match Score (xcorr)

**Result: No discrimination**

The match_score is computed by normalized cross-correlation of the entire captured signal against the full agent template. It produces a single value per capture that is constant across all frames.

```
human_early: self-only match=0.213, overlap match=0.213 (identical)
```

Human speech adds energy but doesn't significantly change the correlation peak because the self signal dominates.

**Verdict:** Cannot discriminate. Would require per-frame local xcorr against expected self audio at each time point.

### 2. Excess RMS Energy

**Result: Partial discrimination**

Computed as `observed_RMS - self_only_mean_RMS` for BOTH frames during agent playback.

| Scenario | Self RMS | Overlap RMS | Excess dB | Discriminates? |
|----------|----------|-------------|-----------|----------------|
| brief_noise | -27.6 dB | -11.9 dB | +15.7 dB | ✓ Yes |
| overlap_human_first | -31.4 dB | -18.3 dB | +13.1 dB | ✓ Yes |
| overlap_self_first | -30.0 dB | -20.4 dB | +9.6 dB | ✓ Yes |
| human_late | -26.3 dB | -22.8 dB | +3.5 dB | Marginal |
| human_early | -25.7 dB | -26.5 dB | -0.8 dB | ✗ No |
| human_before | -27.1 dB | -27.3 dB | -0.2 dB | ✗ No |
| human_after | -27.9 dB | -37.7 dB | -9.8 dB | ✗ No (negative) |
| human_double | -24.1 dB | -36.1 dB | -12.0 dB | ✗ No (negative) |

**Verdict:** Works when human is louder than TTS. Fails when human is quieter or similar volume.

### 3. Absolute RMS Threshold

**Result: Fails due to TTS variability**

TTS audio has highly variable RMS (std ~10 dB). Self-only BOTH frames have 59% of frames above -25 dB, with peaks at -12.9 dB. Any threshold that catches quiet human speech also catches TTS peaks.

**Verdict:** Cannot use absolute threshold. Would need per-scenario adaptive baseline.

### 4. M4 Below-Threshold

**Result: Fires too often during self-playback**

M4 confidence during self-playback is often below 0.50 (the SELF threshold). With a 5-frame sustained requirement, the combined policy produced 435 false interrupts.

**Verdict:** M4 confidence distribution overlaps too much between self and non-self for the current threshold.

### 5. Combined Policy (Sustained BOTH + Excess Energy)

**Result: Best but still inadequate**

From the sweep (threshold=15 frames, excess=12 dB):
- 24 false interrupts
- 5 true interrupts  
- 4 missed interrupts

Best aggregate F1: 0.31

**Verdict:** Too many false interrupts for production use.

## Root Cause

The fundamental issue is that the current attribution architecture produces a **single self-match score per capture**, not per-frame scores. The xcorr is computed against the full template, and the result is constant regardless of human speech.

To reliably discriminate, we need one of:

1. **Per-frame self-match scoring** — Compute local xcorr for each frame against the expected self audio at that time point. When human speech overlaps, the local correlation drops.

2. **Dedicated overlap detector** — A separate module that estimates expected self energy and detects excess.

3. **M4 with separate thresholds** — Lower threshold for overlap detection, higher for SELF classification. Requires M4 buffer reset logic.

## Current Status

The controller treats all `BOTH + is_in_motor_window=True` as self-activity. This is the **safest** option:

- 0 false interrupts
- 0 false defers
- Playback-onset ambient-noise issue fixed (motor holdoff + LISTEN recovery)

The tradeoff: the controller does not yield to human speech during self-playback. This is a known limitation documented in the controller code.

## Recommended Next Steps

### Short-term (safe)
Keep current behavior. The controller is safe (no false interrupts/defers) but doesn't yield during overlap. This is acceptable for v1.

### Medium-term (requires attributor changes)
Add per-frame self-match scoring to the attributor. This requires:
- Computing local xcorr for each frame against the corresponding slice of the agent PCM
- Storing per-frame match_score in the AttributionResult
- Using the per-frame score in the controller's overlap detection

### Long-term (requires new module)
Build a dedicated overlap detector that:
- Estimates expected self energy from the room-colored forward model
- Computes residual energy (observed - expected)
- Flags overlap when residual exceeds threshold for N consecutive frames
