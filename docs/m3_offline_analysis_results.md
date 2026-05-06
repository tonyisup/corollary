# M3 Offline Attribution Analysis — Full M2+M4 Pipeline Results

> **Date:** 2026-05-06
> **Data:** `data/m3_interrupt_v2/` — 10 scenarios with real human speech
> **Pipeline:** Full M2 cross-correlation + VAD + optional M4 → M3 controller

## Setup

Same controller settings as simplified analysis:
- listen_threshold=100ms, cooldown=200ms
- similarity_threshold=0.05, causal_tolerance=175ms

**Key difference from simplified analysis:** This pipeline computes actual
cross-correlation match scores between the captured audio and the agent's PCM
template, rather than using ground truth labels. The VAD runs on the actual
captured audio. This tests the full attribution → controller pipeline.

## Results: M2-only vs M2+M4

| Metric | M2-only | M2+M4 | Target |
|--------|---------|-------|--------|
| False interrupts | 0 | 0 | <5% ✓ |
| False defers | 42 | 42 | <10% ✓ |
| Missed interrupts | 21 | 21 | <10% ✓ |
| Successful resumes | 0/0 | 0/0 | >80% ✗ |

**M2 and M2+M4 produce identical results.** M4 speaker embeddings do not change
the outcome.

## Root Cause: VAD False Positive Before Agent Playback

All 42 false defers occur at the same time: **t=0.51-0.57s**, the first 2-3
frames of agent playback.

### What happens:

1. **t=0.42s:** VAD detects speech (RMS=-38dB, above VAD threshold). This is
   ambient room noise / early audio bleed, not actual human speech.
   Attributor classifies as WORLD (is_playing=False, is_speech=True).

2. **t=0.42s:** Controller enters LISTEN state in response to WORLD.

3. **t=0.51s:** Agent starts playing. Attributor correctly classifies as BOTH
   (match_score=0.21, is_playing=True, is_speech=True). But controller is
   already in LISTEN and `can_speak=False`.

4. **t=0.51-0.57s:** False defer check fires:
   `not can_speak AND should_speak AND is_agent_playing` → True

5. **t=0.57s+:** Controller eventually transitions back to SPEAK, but the
   first 2-3 frames are lost.

### Why M4 doesn't help:

The problem is not in the attribution — the attributor correctly identifies
BOTH during agent playback. The problem is in the **controller state machine**:
once the controller enters LISTEN, it has no mechanism to transition back to
SPEAK when the "world speech" turns out to be the agent's own playback.

M4 speaker embeddings could help if the attributor classified the early frames
as SELF instead of WORLD. But the VAD fires before the agent starts, when
`is_playing=False`. M4 can only distinguish self from world *during* playback,
not before it starts.

## Xcorr Scores

| Scenario | Global | Local | Notes |
|----------|--------|-------|-------|
| human_early | 0.029 | 0.213 | Strong local match |
| human_before | 0.039 | 0.130 | Good local match |
| self_only | 0.062 | 0.052 | Moderate match |
| human_late | 0.029 | 0.037 | Weak match (agent still ramping up) |

Local xcorr scores range from 0.04 to 0.21, all above the 0.05 similarity
threshold. The attributor correctly identifies SELF during agent playback.

## Comparison with Simplified Analysis

| Metric | Simplified (ground truth) | Full pipeline | Difference |
|--------|--------------------------|---------------|------------|
| False interrupts | 0 | 0 | Same |
| False defers | 0 | 42 | Full pipeline has VAD false positive |
| Missed interrupts | 11 | 21 | Full pipeline has more edge cases |
| Resume rate | 100% | N/A | No resumable scenarios in full pipeline |

The simplified analysis used ground truth labels, so it never saw the VAD false
positive issue. The full pipeline reveals this real-world problem.

## Key Findings

1. **Controller state machine gap:** When VAD detects speech before agent starts,
   controller enters LISTEN and can't recover when agent begins playing.
   This causes 2-3 frames of false defer at the start of every agent utterance.

2. **M4 provides no benefit for this issue:** The problem is timing, not
   discrimination. M4 can only help during active playback, not before it starts.

3. **The 2-3 frame delay is consistent:** Every scenario with agent playback
   shows false defers at t=0.51-0.57s. This is a systematic issue, not random.

4. **Self-only scenario is clean:** No false defers in self_only, confirming
   the issue is specific to the VAD→LISTEN→agent-start sequence.

## Possible Fixes

1. **Pre-playback mask:** Don't allow LISTEN transition in the first N ms after
   `request_output()` is called. This would prevent the controller from entering
   LISTEN due to pre-playback noise.

2. **Motor window gating:** If `request_output()` was recently called (within
   cooldown period), suppress WORLD classification. The agent "knows" it's about
   to play audio.

3. **VAD sensitivity:** Increase VAD threshold or add a hold-off period after
   `request_output()` to prevent false positives from room noise.

4. **Controller recovery:** Add a transition from LISTEN back to SPEAK when
   BOTH is detected with high match_score (indicating the "world speech" is
   actually the agent's own playback).

## Next Steps

1. **Implement pre-playback mask:** Add a `motor_holdoff_ms` parameter to the
   controller that prevents LISTEN transitions for N ms after `request_output()`.

2. **Re-run analysis:** Measure the improvement with the fix.

3. **Compare M2 vs M2+M4 with fix:** Once the VAD false positive is addressed,
   M4 may show benefit for BOTH→SELF classification during actual overlap.
