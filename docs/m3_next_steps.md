# M3 Next Steps — From Controller Validation to Conversational Proof

> **Date:** 2026-05-03
> **Premise:** The M3 controller is validated (8/8 synthetic scenarios). The next step is proving it works with real humans.

## Where We Are

| Component | Status | Evidence |
|-----------|--------|----------|
| M3 controller | ✓ Validated | 8/8 scenarios, 33/33 decision checks pass |
| M4 self-voice | ~ Calibrated | Threshold 0.50, combined M2+M4 gating, 93 false SELF frames reclassified |
| M2 attribution | ~ Functional | Geometry-sensitive, ambient noise causes false WORLD |
| M1A hardware | ✓ Working | Vowel/babble probes, room transfer with trust-gated re-adaptation |
| End-to-end M3 | ✗ Not proven | No real human interruption data |

## The Gap

The controller makes correct decisions given correct attribution. But we have never measured whether the *full pipeline* (M2+M4 attribution → M3 controller → Broca response) produces correct turn-taking behavior with a real human.

## Proposed Plan

### Phase 1: Record Interruption Scenarios (1-2 sessions)

Record 5-10 controlled scenarios where a human speaks while Corollary is playing audio. These are the minimal dataset needed to measure false interrupt / false defer rates.

**Setup:**
- Corollary plays a TTS utterance through the speaker
- Human speaks at various points: before, during, after
- Record the raw mic audio + playback timing

**Scenarios to record:**

| # | Scenario | Expected M3 behavior |
|---|----------|---------------------|
| 1 | Human speaks before Corollary starts | Defer (don't start speaking) |
| 2 | Human speaks after Corollary finishes | Respond (start next turn) |
| 3 | Human speaks during Corollary (early) | Interrupt + defer + resume |
| 4 | Human speaks during Corollary (late) | Interrupt + defer + resume |
| 5 | Brief human noise during Corollary | Don't interrupt (single frame) |
| 6 | Human speaks, stops, speaks again | Interrupt, defer, resume, interrupt again |
| 7 | Corollary speaks alone (control) | Complete without interruption |
| 8 | Human speaks alone (control) | Defer throughout |
| 9 | Overlap with human finishing first | Interrupt, defer, resume |
| 10 | Overlap with Corollary finishing first | Complete, then respond |

**What to save:**
- Raw mic audio (`.npy`)
- Playback timing (when Corollary started/stopped emitting)
- Ground truth labels (human speaking yes/no per frame, Corollary should speak yes/no per frame)

### Phase 2: Label and Measure (1 session)

Run the recorded audio through the M2+M4 attribution pipeline. For each frame, compare the controller's decision against ground truth.

**Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| False interrupt rate | Controller interrupts when it should be speaking | <5% |
| False defer rate | Controller defers when it should be speaking | <10% |
| Missed interrupt rate | Controller keeps speaking when human is talking | <10% |
| Successful resume rate | After interruption, controller resumes deferred utterance | >80% |
| Response latency | Time from human speech end to Corollary response | <300ms |

### Phase 3: Compare M2-only vs M2+M4 (1 session)

Run the same recorded scenarios through two pipelines:
1. **M2-only**: No M4 speaker embeddings (current `m4_enable=False`)
2. **M2+M4**: With M4 speaker embeddings (current `m4_enable=True`)

Compare the metrics above. The hypothesis: M4 reduces false defers during self-speech (BOTH frames correctly classified as self) without increasing false interrupts.

### Phase 4: Iterate on Attribution (as needed)

If M4 doesn't improve metrics:
- Try lower threshold (0.40) with stronger M2 timing requirements
- Try room-colored registration with the current room's forward model
- Try multi-frame M4 voting (require 3/5 frames to agree)

If M4 does improve metrics:
- Measure the magnitude of improvement
- Document the optimal threshold for this room/setup
- Update the default configuration

## What "Proven" Looks Like

After these phases, we can say M3 is "proven" when:

1. **False interrupt rate <5%**: Corollary rarely talks over a human who is speaking
2. **False defer rate <10%**: Corollary rarely stays silent when it should be speaking
3. **Successful resume rate >80%**: After being interrupted, Corollary resumes its thought
4. **M4 measurably improves metrics**: Compared to M2-only, M4 reduces false defers without increasing false interrupts

## Estimated Effort

| Phase | Time | Dependencies |
|-------|------|--------------|
| 1: Record scenarios | 1-2 hours | Mic + speaker setup, human willing to talk at a computer |
| 2: Label and measure | 2-3 hours | Phase 1 complete, attribution pipeline running |
| 3: Compare M2 vs M2+M4 | 1-2 hours | Phase 2 complete |
| 4: Iterate | 2-4 hours | Phase 3 shows room for improvement |

**Total: 6-11 hours of focused work**

## Risks

1. **Ambient noise**: The current setup has ambient WORLD classifications before playback. This may inflate false defer rates. Mitigation: add a pre-playback mask or improve silence detection.

2. **Room dependency**: Results may not transfer to other rooms. Mitigation: record in 2-3 rooms if possible.

3. **Human variability**: Different speakers may produce different results. Mitigation: use 2-3 different human speakers.

4. **M4 may not help**: If the ECAPA overlap zone is too large, M4 may not improve metrics over M2-only. This is still a valid result — it tells us the ceiling of embedding-based attribution for TTS voices.
