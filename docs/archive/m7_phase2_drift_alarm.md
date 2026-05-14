# M7 Phase 2: SelfModel Drift Alarm

> **Date:** 2026-05-14
> **Status:** Complete

## Concept

When the room changes, the cerebellum's forward model degrades. The drift alarm detects this degradation (via rising prediction error trends) and either:
- Temporarily boosts the NLMS adaptation rate (learn faster)
- Triggers a full rebabble cycle if degradation is sustained

## Implementation

New module: `packages/cerebellum/drift_alarm.py`

### DriftAlarm

- Tracks a rolling window of prediction errors (in dB)
- Compares recent average against a baseline (mean during stable period)
- When degradation exceeds `spike_threshold_db`: enters DEGRADED status
- When degraded for `rebabble_after_spikes` consecutive detections: enters STALE status
- During stable periods: slowly updates baseline

### Status Levels

- **NOMINAL:** Error within threshold of baseline. No action needed.
- **DEGRADED:** Error spiked. NLMS learning rate is boosted via `adaptation_rate_multiplier`.
- **STALE:** Sustained degradation. Full rebabble recommended.

### Integration into BabblingCerebellum

- `set_drift_alarm(alarm)` attaches a `DriftAlarm` to the cerebellum
- After each `update()`, the prediction error (dB) is recorded in the alarm
- In `_nlms_update()`, the effective `mu` is multiplied by `alarm.adaptation_rate_multiplier`
- The base `nlms_mu` is unchanged; only the effective rate during adaptation is affected

### Key Design Decisions

- The drift alarm is external to the existing `TrustTracker` — it's a separate concern
- The alarm modulates learning rate, not the trust gate
- Ordinary `update()` keeps the old high-trust interference guard unchanged
- `update_trust_gated()` bypasses the high-trust guard (phase 1 fix)

## Tests

8 new unit tests in `tests/unit/test_cerebellum_drift_alarm.py`:
- Stable error keeps nominal status
- Error spike triggers adaptation boost
- Sustained degradation triggers rebabble
- Recovery clears drift
- Adaptation rate multiplier scales with drift
- Serialization roundtrip
- Cerebellum integration: drift alarm boosts effective learning rate
- Cerebellum integration: sustained drift triggers rebabble flag
