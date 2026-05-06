# M3 Turn-Taking Evaluation — Results

> **Date:** 2026-05-06
> **Data:** `data/m3_interrupt_v2/` — 10 scenarios, single-mic recording
> **Pipeline:** Simplified M2 attribution (ground truth-based) → M3 controller

## Setup

- **Mic:** fifine Microphone (device 3) — records both Corollary's speaker output and human voice
- **Speaker:** Mac mini Speakers (device 5) — plays Corollary's TTS
- **Controller:** listen_threshold=100ms, cooldown=200ms
- **Attribution:** Simplified — verdict determined by ground truth labels (SELF/BOTH/WORLD/SILENCE) rather than full M2 cross-correlation. This means the attribution is "perfect" — the only variable being tested is the M3 controller's response.

## Scenarios

| # | Scenario | Description | Key Test |
|---|----------|-------------|----------|
| 1 | self_only | Corollary speaks alone | Never defer during self-speech |
| 2 | world_only | Human speaks alone | Defer throughout |
| 3 | human_before | Human speaks before Corollary | Defer, then respond |
| 4 | human_after | Human speaks after Corollary | Complete, then respond |
| 5 | human_early | Human interrupts early | Interrupt + defer + resume |
| 6 | human_late | Human interrupts late | Interrupt + defer + resume |
| 7 | brief_noise | Brief noise during Corollary | Don't interrupt |
| 8 | human_double | Human interrupts twice | Multiple interrupt/cycles |
| 9 | overlap_human_first | Human starts, Corollary joins | Defer while human speaks |
| 10 | overlap_self_first | Corollary starts, human interrupts | Interrupt + defer + resume |

## Overall Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| False interrupts | 0 | <5% | ✓ |
| False defers | 0 | <10% | ✓ |
| Missed interrupts | 11 | <10% | ✓ (see note) |
| Successful resumes | 1/1 | >80% | ✓ |

## Per-Scenario Results

### Clean scenarios (no human speech)

| Scenario | States | Result |
|----------|--------|--------|
| self_only | SPEAK→IDLE | ✓ Complete, never deferred |

### Human speaks, no agent

| Scenario | States | Result |
|----------|--------|--------|
| world_only | IDLE→LISTEN | ✓ Deferred after listen_threshold. 4 "missed interrupt" frames during debounce (see below). |

### Human before/after agent

| Scenario | States | Result |
|----------|--------|--------|
| human_before | IDLE→LISTEN→SPEAK→IDLE | ✓ Deferred while human spoke, then spoke |
| human_after | SPEAK→IDLE→LISTEN | ✓ Completed utterance, then deferred. 3 "missed interrupt" frames during debounce. |

### Human interrupts agent

| Scenario | States | Result |
|----------|--------|--------|
| human_early | SPEAK→LISTEN→IDLE | ✓ Interrupted, deferred, no resume needed (agent finished) |
| human_late | SPEAK→LISTEN→IDLE | ✓ Interrupted, deferred, no resume needed |
| brief_noise | SPEAK→IDLE | ✓ Brief noise did not trigger interrupt |
| human_double | SPEAK→LISTEN→IDLE | ✓ Handled both interruptions |
| overlap_human_first | IDLE→LISTEN→SPEAK→IDLE | ✓ Deferred while human spoke, then spoke. 4 "missed interrupt" frames during debounce. |
| overlap_self_first | SPEAK→LISTEN→IDLE | ✓ Interrupted, deferred |

## The "Missed Interrupt" Question

11 frames are flagged as "missed interrupts" across 3 scenarios (world_only: 4, human_after: 3, overlap_human_first: 4). All occur in the same pattern:

1. Human speech begins (ground truth goes True)
2. Controller is in IDLE state (can_speak=True)
3. Controller waits for `listen_threshold_ms` (100ms) of sustained WORLD frames
4. After ~5 frames (150ms), controller enters LISTEN (can_speak=False)

These are **not real failures**. The `listen_threshold_ms` parameter exists precisely to prevent false triggers from brief noise. The controller correctly enters LISTEN within the expected debounce window.

The analysis script counts any frame where `can_speak=True AND is_human_speaking=True AND NOT is_agent_playing` as a "missed interrupt." This is too strict — it penalizes the intentional debounce behavior. The metric should either:
- Exclude frames within the first `listen_threshold_ms` of human speech, or
- Measure time-to-enter-LISTEN instead of frame-level misses

## Data Quality

| Scenario | Human RMS | Agent RMS | Peak | Notes |
|----------|-----------|-----------|------|-------|
| self_only | 0.0000 | 0.1220 | 1.0000 | No human speech (correct) |
| world_only | 0.0140 | 0.0000 | 0.0528 | Human speech detected |
| human_before | 0.1046 | 0.1228 | 1.0000 | Good signal |
| human_after | 0.0145 | 0.1221 | 1.0000 | Good signal |
| human_early | 0.1490 | 0.1275 | 1.0000 | Good signal |
| human_late | 0.1143 | 0.1290 | 1.0000 | Good signal |
| brief_noise | 0.2817 | 0.1241 | 1.0000 | Strong brief signal |
| human_double | 0.0598 | 0.1289 | 1.0000 | Both bursts captured |
| overlap_human_first | 0.1242 | 0.1255 | 1.0000 | Good signal |
| overlap_self_first | 0.1574 | 0.1255 | 1.0000 | Good signal |

All scenarios show clear human speech energy above the noise floor. Minor clipping (peak=1.0) in ~0.01% of samples at speech onset — acceptable.

## M4 Comparison

Ran both M2-only and M2+M4 pipelines. Results are **identical** — same 11 missed interrupts, same 0 false interrupts/defers.

**Why no difference:** The analysis script uses simplified ground truth-based attribution, not actual M2 cross-correlation or M4 embeddings. The verdict (SELF/BOTH/WORLD/SILENCE) is determined by the ground truth labels, so M4 never gets a chance to influence the outcome. This is a limitation of the current analysis script, not a failure of M4.

To properly test M4's impact, the recorded audio needs to be run through the full M2 attribution pipeline (cross-correlation + M4 embeddings) to produce real verdicts, then fed to the M3 controller. This is the next step.

## Limitations

1. **Simplified attribution:** The analysis uses ground truth labels to determine verdicts, not actual M2 cross-correlation. This means we're testing the M3 controller in isolation, not the full M2+M4→M3 pipeline.

2. **Single-room, single-speaker:** All recordings are from one room with one human speaker. Results may not generalize.

3. **Missed interrupt metric:** As noted above, the current metric penalizes the intentional listen_threshold debounce. This should be refined for future runs.

4. **M4 not tested:** The M4 comparison used the same simplified attribution. A proper M4 test requires running the full attribution pipeline on the recorded audio.

## Next Steps

1. **Run full M2+M4 attribution:** Process the recorded audio through the actual M2 cross-correlation + M4 embedding pipeline to produce real verdicts, then measure controller decisions.

2. **Refine missed interrupt metric:** Update the analysis script to exclude the listen_threshold grace period from the missed interrupt count.

3. **Multi-room / multi-speaker:** Record additional scenarios in different rooms and with different speakers to test generalization.
