# M7 Trust-Gated Room Adaptation — Phase 1

> **Date:** 2026-05-14
> **Status:** Replay-validated

## Implementation

Thin attribution gate around existing `BabblingCerebellum.update()`:

- `BabblingCerebellum.update_trust_gated(...)` adapts NLMS only when `verdict == SELF`, `m4_confidence >= threshold`, and room-local/self-match score passes threshold.
- `StreamingConfig` fields: `enable_room_filter_adaptation` (default False), `room_adapter`, `room_adapt_confidence_threshold` (default 0.80), `room_adapt_match_threshold` (default 0.20).
- `StreamingRuntime` uses the actual emitted TTS chunk as efference copy and the mic frame as capture. If output is gated/silent, result is `no_efference` and no adapter call.
- On successful adaptation, runtime publishes `room_adapter.get_learned_ir()` back into `ThalamicAttributor.room_filter`.
- Frame logs include `room_adapted`, `room_adaptation_reason`, `room_adaptation_error`, `room_adaptation_updates`.

## Safety Invariant

BOTH/WORLD/SILENCE never update the room filter, even with high M4 confidence. This avoids learning overlap or another speaker into the self room model.

## Deterministic Replay Validator

`scripts/m7_trust_gated_replay.py` simulates room A → room B drift with interleaved trusted SELF, WORLD, and BOTH frames.

**Result:** room-B prediction MSE improved from 0.03226545 to 0.01715044 (46.8% improvement); trusted SELF produced 8 updates; WORLD and BOTH produced 0 updates.

## Important Bug Fix

The older high-trust interference guard in `BabblingCerebellum.update()` can mistake room drift for interference and freeze NLMS. `update_trust_gated()` bypasses that internal guard after external attribution has proven trusted SELF. Ordinary `update()` keeps the old guard unchanged.

## Tests

13 new M7 tests across cerebellum unit coverage, streaming integration, and deterministic replay. Full suite: 402 passed, 1 skipped.
