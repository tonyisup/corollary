# Evaluation Protocol — Corollary M0 (Simulated Loop)

> **Status:** Design (M0)  
> **Created:** 2026-04-03  
> **Scope:** M0 milestone acceptance criteria and reproducible tests

## Overview

Each milestone is evaluated independently. You do not advance from M0 into later hardware milestones until M0's evals pass. For the developmental hardware path, progression should be staged: first self-sound attribution, then babbling-based room learning, then mixed-source attribution, then speech-aware self/world separation, then agentive listening and proto-turn-taking, and only after that explicit attribution classifiers, memory, and richer conversation. The same eval scripts should run in CI where practical.

## Eval Structure

Each milestone has:
1. **Unit tests** — `tests/unit/` (pure Python, no audio files needed)
2. **Integration tests** — `tests/integration/` (may require audio fixtures)
3. **Benchmark evals** — `tests/eval/` (measurable metrics against ground truth)

```
scripts/
  run_eval.sh          # Run all evals for current milestone
  gen_dataset.py       # Generate synthetic test data
tests/
  unit/                # Fast, no external deps
  integration/         # May need audio fixtures
  eval/                # Benchmarks with acceptance thresholds
```

## M0 Eval Targets

### 1. TTS → Room → ASR Round-Trip (WER)

**What:** End-to-end speech pipeline quality.

**Method:**
- Feed M known sentences through the simulated loop
- ASR transcribes the result
- Compute Word Error Rate (WER) against ground truth

**Acceptance:** WER ≤ 10% for clean simulated audio

**How to run:**
```bash
python -m pytest tests/eval/test_roundtrip.py --wer-threshold=0.10
```

### 2. Efference Copy Accuracy

**What:** Can the cerebellum detect self-voice?

**Method:**
- Generate self-voice audio (no world interference in M0)
- Compute cerebellum residual (predicted vs actual)
- Check that residual indicates self-attribution

**Acceptance:** Residual < 0.2 for pure self-voice

**How to run:**
```bash
python -m pytest tests/eval/test_efference.py --residual-threshold=0.2
```

### 3. Semantic Grounding Accuracy

**What:** Does SAG resolve ASR tokens to correct concepts?

**Method:**
- Initialize SAG with known concept graph
- Feed ASR transcription through SAG resolve()
- Check that key entities map to correct nodes

**Acceptance:** ≥80% of unique words resolved to SAG nodes

**How to run:**
```bash
python -m pytest tests/eval/test_sag_grounding.py --resolve-threshold=0.80
```

### 4. Disambiguation

**What:** Can SAG context resolve homonyms?

**Method:**
- Set up ambiguous tokens (e.g., "bank" → river vs financial)
- Provide context in SAG.nodes[...].context
- Check that resolve() returns the appropriate node

**Acceptance:** 100% disambiguation for seeded examples

**How to run:**
```bash
python -m pytest tests/eval/test_disambiguation.py
```

### 5. Integration: Full Simulated Loop

**What:** All components work together.

**Method:**
- TTS generates audio from known prompt
- Audio passes through simulated room
- ASR transcribes
- SAG resolves entities
- Cerebellum detects self-voice
- PFC orchestrates the loop

**Acceptance:** 
- No crashes
- Final transcript preserves ≥90% of meaning
- SAG updates correctly (nodes created, edges asserted)

**How to run:**
```bash
python -m pytest tests/eval/test_full_loop.py
```

## Running All M0 Evals

```bash
cd ~/Projects/corollary
./scripts/run_eval.sh m0
```

Or directly:
```bash
python -m pytest tests/unit/ tests/eval/ -v --tb=short
```

## CI Integration

GitHub Actions runs unit tests on every push:

```yaml
# .github/workflows/ci.yml
name: Corollary CI
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.9"
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest tests/unit/ -v
```

## M1 Eval Targets (Revised)

### Synthetic vs hardware eval boundary

The current M1 scaffolds mix two different eval modes that should be reported separately:

- **Synthetic evals:** waveform generation, simulated room effects, and structured negatives run entirely in code. These are regression-friendly and silent in the physical room.
- **Hardware-facing evals:** real speaker playback plus real microphone capture. These are the point where audible chirps, vowels, or babble should actually be expected.

Synthetic M1A/M1C progress should not be overstated as physical-world validation. Hardware-facing runs should be called out explicitly when reporting results.

### 1. Self-Sound Attribution

**What:** Can the system determine whether an acoustic event was self-caused, independent of words?

**Method:**
- Emit simple sounds: tones, clicks, vowels, babble, nonsense syllables
- Record mic input with efference-copy timing
- Evaluate self vs not-self attribution and onset/offset alignment

**Acceptance:**
- High self/not-self attribution accuracy
- Stable temporal alignment between emitted and observed sound
- Reproducible performance across sessions in the same room

### 2. Babbling Convergence

**What:** Does repeated babbling improve the forward model?

**Method:**
- Run repeated babbling iterations in a fixed environment
- Measure prediction error and attribution performance over time

**Acceptance:**
- Prediction error decreases over repeated trials
- Attribution improves after adaptation
- Convergence is reproducible

### 3. Mixed-Source Attribution

**What:** Can the system distinguish self-only, world-only, and overlapping audio?

**Method:**
- Introduce external/world audio during or near self-generated sound
- Score frame-level or chunk-level attribution labels

**Acceptance:**
- Strong self/world/overlap attribution performance
- Low false self-attribution rate

## M2 Eval Targets

### 1. Speech-Aware Separation

**What:** Can the system distinguish speech from non-speech while preserving external speech under self overlap?

**Method:**
- Introduce speech and non-speech external sources during self-generated sound
- Score speech-preservation and speech/non-speech discrimination under overlap

**Acceptance:**
- Stable speech/non-speech discrimination
- External speech preserved with low degradation during self activity

## M3 Eval Targets

### 1. Agentive Listening

**What:** Can the system distinguish vocal agents from non-agentive environment noise?

**Method:**
- Compare external vocal sources against environmental non-speech and non-agentive sound sources
- Score agentive-vs-ambient discrimination and listen-state transitions

**Acceptance:**
- Reliable agentive-vs-ambient discrimination
- Self-output is gated or deferred appropriately during external vocal activity

### 2. Proto-Turn-Taking

**What:** Can the system detect contingent timing windows for early vocal exchange?

**Method:**
- Present simple alternating and interrupting vocal patterns
- Measure interruption sensitivity, listen-state stability, and contingent response timing

**Acceptance:**
- Measurable turn-taking sensitivity
- Stable inhibition of self-output during active external vocal windows

## Metrics Dashboard (Future)

| Milestone | Core Metric | Status |
|-----------|-------------|--------|
| M0        | WER, residual, SAG resolve | TBD |
| M1a       | Self/not-self attribution | — |
| M1b       | Convergence / prediction error | — |
| M1c       | Overlap attribution F1 | — |
| M2        | Speech/non-speech stability, duplex WER | — |
| M3        | Agentive listening, turn-taking sensitivity | — |
| M4        | Explicit attribution F1 | — |
| M5        | Memory + dialog retrieval | — |
