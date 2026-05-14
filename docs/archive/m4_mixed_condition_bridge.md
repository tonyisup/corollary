# M4 Mixed-Condition Bridge

> **Date:** 2026-05-14
> **Status:** Partial bridge complete

## Key Reframing

`BOTH` is often the correct attribution verdict during true self+human overlap. The PoC does not need to force those frames into SELF or WORLD; it needs speaker-ID continuity while overlap is happening.

## Runtime Rule

- Cold-start conversational registration accumulates clean `WORLD` frames only, avoiding registration from a self+human mixture.
- Once any non-self speaker is registered/pre-registered, `BOTH` frames become eligible for speaker-ID buffering so identification can persist during overlap.
- Adaptive-threshold profile learning remains WORLD-only; overlap-contaminated `BOTH` frames do not ratchet non-self thresholds upward.

## Evidence

- Regression tests: `tests/integration/test_streaming_speaker_id.py` covers both sides of the rule.
- Fixed `_save_logs()` to use `_log_to_dict()` so speaker fields persist into JSONL artifacts.
- Live mixed split-output validation (speaker2, world_gain=0.8): 257/282 frames were BOTH; 241/257 BOTH frames had best-match diagnostics (93.8%); 200/257 BOTH frames passed the speaker registry threshold (77.8%).
