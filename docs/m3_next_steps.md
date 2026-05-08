# M3 Next Steps — Current Status

> **Date:** 2026-05-07
> **Status:** Offline validation complete, live E2E harness ready

## Where We Are

| Component | Status | Evidence |
|-----------|--------|----------|
| M3 controller (synthetic) | ✓ Validated | 8/8 scenarios, 33/33 decision checks pass |
| M3 controller (offline real) | ✓ Validated | 0% FI, 0% FD, 1.1% MI, 100% resume on 10 scenarios |
| Per-frame self-match | ✓ Implemented | Energy ratio computed per-frame in attributor |
| Overlap detection | ✓ Working | Hysteresis + sustained evidence + post-interrupt resume |
| Live E2E harness | ✓ Built | `scripts/m3_live_e2e_harness.py` — ready to run |
| Live E2E test | ✗ Not run | Needs real-time audio I/O |

## Key Metrics (offline, data/m3_interrupt_v2/)

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

### Live E2E harness (`scripts/m3_live_e2e_harness.py`)
- Replays pre-recorded captured audio through speaker
- Re-captures through mic (tests full room acoustics chain)
- Runs attribution + controller pipeline in real-time
- Compares live results against offline analysis
- Scenarios: human_early, human_late, human_before, human_after, overlap_self_first

## Recommended Next Steps

### 1. Run live E2E validation (highest priority, ~2-4 hours)

```bash
# Run all scenarios
python scripts/m3_live_e2e_harness.py --all --save-dir data/m3_live_e2e/

# Run single scenario
python scripts/m3_live_e2e_harness.py --scenario human_early --save-dir data/m3_live_e2e/
```

Expected outcomes:
- Validate that controller behaves correctly with real room acoustics
- Measure live latency (capture + attribution + controller)
- Identify any timing drift or jitter issues
- Compare live traces against offline assumptions

### 2. Analyze live vs offline comparison (~1-2 hours)

After running the harness, compare:
- Verdict distributions (live vs offline)
- State transition timing
- Energy ratio distributions
- Latency measurements

### 3. Mic position sweep (later milestone)

Once the live harness is validated, test with mic at different positions:
- Close to speaker (current)
- 1 meter away
- 2 meters away
- Different angles

This measures how geometry changes the attribution landscape and whether the controller remains robust.

### 4. M2 attribution quality (longer term)

The ambient noise sensitivity (VAD detecting speech in room noise) is the root cause of many edge cases. Improving M2 attribution would have cascading benefits.

### 5. Collision repair / floor negotiation (future milestone)

Do not add repair utterances to current M3 reflex validation yet. Keep M3 focused on detecting overlap, yielding, and resuming. Future phone-like conversation behavior should add a higher-level repair policy for collisions such as "Sorry, go ahead" / "No, you first" once overlap/resume metrics are stable.

See `docs/m3_collision_repair_future.md`.
