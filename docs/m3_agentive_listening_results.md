# M3 Agentive Listening — Calibrated Assessment

> **Status:** Active Research — calibration phase
> **Date:** 2026-05-03
> **Scope:** M3 turn-taking with M2+M4 attribution

## What We Know

### M3 controller works correctly for clear cases

The `AgentiveController` correctly handles:
- **Self-only**: Agent speaks without world speech → completes (IDLE→SPEAK→IDLE)
- **World-only**: World speech present → defers (IDLE→LISTEN→IDLE)
- **Clear overlap**: Agent speaking + world speech → interrupts, defers, resumes

### The hard problem is attribution, not control

The controller's behavior is only as good as the M2/M4 attribution it receives:

| Scenario | Problem | Root cause |
|----------|---------|------------|
| BOTH during playback | Controller enters LISTEN, defers | BOTH frames were treated as external speech even during self-playback |
| Ambient noise before playback | WORLD classification triggers LISTEN | M2 geometry sensitivity — room reflections classified as speech |
| Low-confidence M4 | M4 assigned SELF on weak evidence (conf 0.14-0.30) | Threshold too low (0.10), no M2 requirement |

### Fixes applied (commits d54b44a, 9ec936f)

1. **M4 threshold raised 0.10→0.50**: ECAPA cosine similarity for TTS-vs-human overlaps significantly above 0.10. The 0.50 threshold keeps TTS self-probes accepted (min 0.51) while rejecting most human speech.

2. **M4 requires M2 evidence**: M4 can no longer assign SELF independently. Both similarity (is_self_like) AND speaker embedding (confidence ≥ threshold) must agree. This reclassified 93 false SELF frames as BOTH.

3. **BOTH during motor window = self-activity**: BOTH frames where `is_in_motor_window=True` are treated as self speech, not external agent speech. This prevents the controller from entering LISTEN during self-playback.

### Remaining issues

1. **Ambient WORLD before playback**: 1-3 WORLD frames immediately before playback starts (rms -18 to -20 dB) trigger LISTEN. These are likely TTS reflections or geometry artifacts. This is an M2 attribution problem, not an M3 controller problem.

2. **Recovery rate still 0%**: No test scenario with validated interruption→recovery exists yet. The synthetic scenario passes, but real data doesn't have genuine interruption events.

## Turn-Taking Evaluation Results

`scripts/m3_turn_taking_eval.py` — 10 scenarios:

| Scenario | Result | Notes |
|----------|--------|-------|
| self_only (synthetic) | PASS | IDLE→SPEAK→IDLE |
| world_only (synthetic) | PASS | IDLE→LISTEN→IDLE |
| overlap_interrupt (synthetic) | PARTIAL | State machine correct, but resume validation needs work |
| world_after_self (synthetic) | PASS | Correct non-interrupt |
| recovery (synthetic) | PARTIAL | State machine correct, resume validation needs work |
| turn_01-05 (real, self only) | 3/5 pass | 2 fail from ambient WORLD before playback |

**No false interrupts in any scenario.** The controller never incorrectly interrupts.

## What "Proven" Means Here

M3 is **not yet proven** in the sense of demonstrating clear turn-taking wins. What we have:

- A controller that behaves correctly given correct attribution
- An attribution pipeline (M2+M4) that is calibrated but not yet reliable
- Clear metrics for what "proven" would mean: false interrupts, false defers, successful resumes

## Next Steps

1. **Build interruption test scenarios**: Record or synthesize genuine overlap events where Corollary is speaking and world speech interrupts
2. **Fix ambient WORLD classification**: Either improve M2's silence detection or add a pre-playback mask
3. **Measure recovery**: With real interruption data, verify the resume path works end-to-end
4. **Update README/docs**: Remove "massive impact" language; replace with calibrated claims
