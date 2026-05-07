# M3 Next Steps — Current Status

> **Date:** 2026-05-07
> **Status:** Controller validated on real data, all 4 targets met

## Where We Are

| Component | Status | Evidence |
|-----------|--------|----------|
| M3 controller controller | ✓ Validated | 8/8 synthetic scenarios, 33/33 decision checks pass |
| M3 controller real data | ✓ Validated | 0% FI, 0% FD, 1.1% MI, 100% resume on 10 scenarios |
| Per-frame self-match | ✓ Implemented | Energy ratio computed per-frame in attributor |
| Overlap detection | ✓ Working | Hysteresis + sustained evidence + post-interrupt resume |
| M4 self-voice | ~ Calibrated | Threshold 0.50, combined M2+M4 gating |
| M2 attribution | ~ Functional | Geometry-sensitive, ambient noise causes false WORLD |
| Live end-to-end test | ✗ Not done | Everything validated offline only |

## Key Metrics (data/m3_interrupt_v2/)

| Metric | Target | Result |
|--------|--------|--------|
| False interrupt rate | <5% | 0.0% ✓ |
| False defer rate | <10% | 0.0% ✓ |
| Missed interrupt rate | <10% | 1.1% ✓ |
| Resume rate | >80% | 100% ✓ |

## What Was Built

### Per-frame self-match scoring
- `AttributionResult` carries `self_match_score_per_frame` (observed / expected self RMS)
- `ThalamicAttributor` computes this by comparing each frame against the corresponding slice of the agent PCM template
- Callers pass `self_pcm` and `agent_start_sample`

### Overlap detection policy
- Energy ratio threshold: 2.0 (configurable)
- Sustained evidence: 7 consecutive frames (configurable)
- Hysteresis: 300ms minimum LISTEN duration after overlap interrupt
- Post-interrupt resume: Return to IDLE after 2x min_listen_duration if agent not playing

### Known limitations
- **Pre-agent human speech**: Overlap detector only fires during agent playback. Human speech before agent starts is handled by the post-agent-finished LISTEN path (17 MI in `overlap_human_first`).
- **Post-agent human speech**: Same limitation (10 MI in `human_after`).
- **Energy ratio ceiling**: TTS peaks produce ratios up to 4.3-4.4, overlapping with human speech. The sustained-frame requirement helps but can't fully separate.

## Recommended Next Steps

### 1. Live end-to-end test (highest priority, ~2-4 hours)

Everything validated offline. The real test is running the full pipeline live:
- Play TTS through the speaker
- Speak at various points (before, during, after TTS)
- Measure: does Corollary correctly yield? Does it resume after you stop?
- Record and run through offline analysis

### 2. Tune overlap detector for live use (medium priority, ~1 hour)

Current settings are conservative (0 FI/FD). For live use, may want to:
- Lower `both_external_min_frames` to 5 for faster response
- Lower `both_external_energy_ratio_threshold` to 1.5 for quieter human speech
- Accept some false positives for better responsiveness

### 3. Handle pre-agent human speech (medium priority, ~1-2 hours)

Add a "pre-agent deferral" path: if the controller deferred output before the agent started (due to WORLD speech), and the human speech ends, resume after a timeout.

### 4. M2 attribution quality (longer term)

The ambient noise sensitivity (VAD detecting speech in room noise) is the root cause of many edge cases. Improving M2 attribution would have cascading benefits.
